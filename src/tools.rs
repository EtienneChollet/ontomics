use ontomics::diff;
use ontomics::graph::ConceptGraph;
use ontomics::parser::LanguageParser;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParam, CallToolResult, Content, Implementation, InitializeResult,
    ListToolsResult, PaginatedRequestParam, RawResource, ReadResourceRequestParam,
    ReadResourceResult, Resource, ResourceContents, ServerCapabilities, Tool,
    ToolsCapability,
};
use rmcp::service::{RequestContext, RoleServer};
use ontomics::types::ConceptQueryResult;
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, RwLock};

/// Budget in bytes for MCP tool output (~10K tokens).
const OUTPUT_BUDGET_BYTES: usize = 40_000;

/// Strip internal-only data and progressively compact a query result
/// so that the serialized output stays within `OUTPUT_BUDGET_BYTES`.
fn compact_query_result(r: &mut ConceptQueryResult) {
    // Always strip data that is useless to MCP consumers:
    // - concept.occurrences is redundant with top_occurrences
    // - embedding vectors are internal implementation details
    r.concept.occurrences.clear();
    r.concept.embedding = None;
    for sc in &mut r.concept.subconcepts {
        sc.embedding = None;
    }

    if estimated_json_len(r) <= OUTPUT_BUDGET_BYTES {
        return;
    }

    // Level 1: trim unbounded / low-priority fields
    r.call_graph.truncate(10);
    r.classes.truncate(3);
    r.variants.truncate(10);
    for sc in &mut r.concept.subconcepts {
        sc.occurrences.truncate(3);
    }

    if estimated_json_len(r) <= OUTPUT_BUDGET_BYTES {
        return;
    }

    // Level 2: aggressive — keep only the essentials
    r.call_graph.clear();
    r.classes.truncate(1);
    r.signatures.truncate(2);
    r.variants.truncate(5);
    r.concept.subconcepts.clear();
}

/// Cheap size estimate: serialize to JSON and measure byte length.
fn estimated_json_len(r: &ConceptQueryResult) -> usize {
    serde_json::to_string(r).map(|s| s.len()).unwrap_or(0)
}

#[derive(Clone)]
pub struct OntomicsServer {
    graph: Arc<RwLock<ConceptGraph>>,
    repo_root: PathBuf,
    parser: Arc<dyn LanguageParser>,
    warnings: Arc<Mutex<Vec<String>>>,
    indexing_ready: Arc<AtomicBool>,
}

impl OntomicsServer {
    pub fn new(
        graph: ConceptGraph,
        repo_root: PathBuf,
        parser: Arc<dyn LanguageParser>,
        warnings: Vec<String>,
    ) -> Self {
        Self {
            graph: Arc::new(RwLock::new(graph)),
            repo_root,
            parser,
            warnings: Arc::new(Mutex::new(warnings)),
            indexing_ready: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Create a server with shared handles for deferred startup.
    /// The graph starts empty and is populated by a background thread.
    pub fn new_deferred(
        graph: Arc<RwLock<ConceptGraph>>,
        repo_root: PathBuf,
        parser: Arc<dyn LanguageParser>,
        warnings: Arc<Mutex<Vec<String>>>,
        indexing_ready: Arc<AtomicBool>,
    ) -> Self {
        Self {
            graph,
            repo_root,
            parser,
            warnings,
            indexing_ready,
        }
    }

    pub fn graph_handle(&self) -> Arc<RwLock<ConceptGraph>> {
        Arc::clone(&self.graph)
    }


    fn handle_query_concept(&self, args: &Value) -> Result<Value, String> {
        use ontomics::types::QueryConceptParams;

        let term = args
            .get("term")
            .and_then(|v| v.as_str())
            .ok_or("missing required argument 'term'")?;

        let defaults = QueryConceptParams::default();
        let params = QueryConceptParams {
            max_related: args
                .get("max_related")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(defaults.max_related),
            max_occurrences: args
                .get("max_occurrences")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(defaults.max_occurrences),
            max_variants: args
                .get("max_variants")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(defaults.max_variants),
            max_signatures: args
                .get("max_signatures")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(defaults.max_signatures),
            max_entities: args
                .get("max_entities")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(defaults.max_entities),
        };

        let graph = self.graph.read().map_err(|e| format!("lock error: {e}"))?;
        match graph.query_concept(term, &params) {
            Some(mut result) => {
                compact_query_result(&mut result);
                serde_json::to_value(result)
                    .map_err(|e| format!("serialization error: {e}"))
            }
            None => Ok(json!({
                "error": "not_found",
                "message": format!("no concept matching '{term}'"),
            })),
        }
    }

    fn handle_check_naming(&self, args: &Value) -> Result<Value, String> {
        let identifier = args
            .get("identifier")
            .and_then(|v| v.as_str())
            .ok_or("missing required argument 'identifier'")?;

        let graph = self.graph.read().map_err(|e| format!("lock error: {e}"))?;
        let result = graph.check_naming(identifier);
        serde_json::to_value(result)
            .map_err(|e| format!("serialization error: {e}"))
    }

    fn handle_suggest_name(&self, args: &Value) -> Result<Value, String> {
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or("missing required argument 'description'")?;

        let graph = self.graph.read().map_err(|e| format!("lock error: {e}"))?;
        let result = graph.suggest_name(description);
        serde_json::to_value(result)
            .map_err(|e| format!("serialization error: {e}"))
    }

    fn handle_ontology_diff(&self, args: &Value) -> Result<Value, String> {
        let since = args
            .get("since")
            .and_then(|v| v.as_str())
            .unwrap_or("HEAD~5");

        let graph = self.graph.read().map_err(|e| format!("lock error: {e}"))?;
        let result = diff::ontology_diff(
            &self.repo_root,
            since,
            &graph.concepts,
            &*self.parser,
        )
        .map_err(|e| format!("ontology_diff failed: {e}"))?;

        serde_json::to_value(result)
            .map_err(|e| format!("serialization error: {e}"))
    }

    fn handle_list_concepts(&self, args: &Value) -> Result<Value, String> {
        let top_k = args
            .get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        let graph = self.graph.read().map_err(|e| format!("lock error: {e}"))?;
        let mut concepts = graph.list_concepts();
        if let Some(k) = top_k {
            concepts.truncate(k);
        }

        let summary: Vec<Value> = concepts
            .iter()
            .map(|c| {
                let entity_count = graph
                    .entities
                    .values()
                    .filter(|e| e.concept_tags.contains(&c.id))
                    .count();
                json!({
                    "canonical": c.canonical,
                    "occurrences": c.occurrences.len(),
                    "entity_types": c.entity_types,
                    "entity_count": entity_count,
                })
            })
            .collect();

        let warnings = self.warnings.lock()
            .map(|w| w.clone())
            .unwrap_or_default();
        if warnings.is_empty() {
            Ok(json!(summary))
        } else {
            Ok(json!({
                "warnings": warnings,
                "concepts": summary,
            }))
        }
    }

    fn handle_describe_symbol(&self, args: &Value) -> Result<Value, String> {
        let name = args
            .get("name")
            .and_then(|v| v.as_str())
            .ok_or("missing required argument 'name'")?;

        let graph = self.graph.read().map_err(|e| format!("lock error: {e}"))?;
        match graph.describe_symbol(name) {
            Some(result) => serde_json::to_value(result)
                .map_err(|e| format!("serialization error: {e}")),
            None => Ok(json!({
                "error": "not_found",
                "message": format!("no symbol matching '{name}'"),
            })),
        }
    }

    fn handle_locate_concept(&self, args: &Value) -> Result<Value, String> {
        let term = args
            .get("term")
            .and_then(|v| v.as_str())
            .ok_or("missing required argument 'term'")?;

        let graph = self.graph.read().map_err(|e| format!("lock error: {e}"))?;
        match graph.locate_concept(term) {
            Some(result) => serde_json::to_value(result)
                .map_err(|e| format!("serialization error: {e}")),
            None => Ok(json!({
                "error": "not_found",
                "message": format!("no concept matching '{term}'"),
            })),
        }
    }

    fn handle_list_conventions(&self, _args: &Value) -> Result<Value, String> {
        let graph = self.graph.read().map_err(|e| format!("lock error: {e}"))?;
        let conventions = graph.list_conventions();
        serde_json::to_value(conventions)
            .map_err(|e| format!("serialization error: {e}"))
    }

    fn handle_export_domain_pack(&self, _args: &Value) -> Result<Value, String> {
        let graph = self.graph.read().map_err(|e| format!("lock error: {e}"))?;
        let pack = ontomics::domain_pack::export_domain_pack(&graph);
        let yaml = serde_yaml::to_string(&pack)
            .map_err(|e| format!("YAML serialization error: {e}"))?;
        let stats = json!({
            "abbreviations": pack.abbreviations.len(),
            "conventions": pack.conventions.len(),
            "domain_terms": pack.domain_terms.len(),
            "concept_associations": pack.concept_associations.len(),
        });
        Ok(json!({
            "yaml": yaml,
            "stats": stats,
        }))
    }

    fn handle_vocabulary_health(
        &self,
        _args: &Value,
    ) -> Result<Value, String> {
        use ontomics::config::HealthConfig;

        let graph = self
            .graph
            .read()
            .map_err(|e| format!("lock error: {e}"))?;
        let config = HealthConfig::default();
        let result = graph.vocabulary_health(&config);
        serde_json::to_value(result)
            .map_err(|e| format!("serialization error: {e}"))
    }

    fn handle_list_entities(&self, args: &Value) -> Result<Value, String> {
        use ontomics::types::EntityKind;

        let concept = args.get("concept").and_then(|v| v.as_str());
        let role = args.get("role").and_then(|v| v.as_str());
        let kind = args.get("kind").and_then(|v| v.as_str()).and_then(|k| {
            match k.to_lowercase().as_str() {
                "class" => Some(EntityKind::Class),
                "function" => Some(EntityKind::Function),
                _ => None,
            }
        });
        let top_k = args
            .get("top_k")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(20);

        let graph = self.graph.read().map_err(|e| format!("lock error: {e}"))?;
        let results = graph.list_entities(concept, role, kind.as_ref(), top_k);
        serde_json::to_value(results)
            .map_err(|e| format!("serialization error: {e}"))
    }

    fn handle_trace_concept(
        &self,
        args: &Value,
    ) -> Result<Value, String> {
        let concept = args
            .get("concept")
            .and_then(|v| v.as_str())
            .ok_or("missing required argument 'concept'")?;
        let max_depth = args
            .get("max_depth")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(5);

        let graph = self
            .graph
            .read()
            .map_err(|e| format!("lock error: {e}"))?;
        match graph.trace_concept(concept, max_depth) {
            Some(result) => serde_json::to_value(result)
                .map_err(|e| format!("serialization error: {e}")),
            None => Ok(json!({
                "error": "not_found",
                "message": format!(
                    "no concept matching '{concept}'"
                ),
            })),
        }
    }
}

fn tool_schema(
    properties: Value,
    required: &[&str],
) -> Arc<serde_json::Map<String, Value>> {
    let schema = json!({
        "type": "object",
        "properties": properties,
        "required": required,
    });
    Arc::new(match schema {
        Value::Object(map) => map,
        _ => unreachable!(),
    })
}

fn tool_definitions() -> Vec<Tool> {
    vec![
        Tool::new(
            "query_concept",
            "Semantic concept lookup — returns all variants (including abbreviations \
             like trf for transform), related concepts, naming conventions, function \
             signatures, and file locations in one call. Richer than grep: querying \
             'transform' also finds 'trf', 'spatial_transform', 'apply_transform', \
             and related concepts like 'displacement'. Use when asked 'what is X', \
             'what does X mean', or 'where is X used'.",
            tool_schema(
                json!({
                    "term": {
                        "type": "string",
                        "description": "The concept term to look up (e.g. 'transform')",
                    },
                    "max_related": {
                        "type": "integer",
                        "description": "Max related concepts to return (default: 10)",
                    },
                    "max_occurrences": {
                        "type": "integer",
                        "description": "Max occurrence locations to return (default: 5)",
                    },
                    "max_variants": {
                        "type": "integer",
                        "description": "Max variant identifiers to return (default: 20)",
                    },
                    "max_signatures": {
                        "type": "integer",
                        "description": "Max function signatures to return (default: 5)",
                    }
                }),
                &["term"],
            ),
        ),
        Tool::new(
            "check_naming",
            "Check if an identifier follows project naming conventions — detects \
             inconsistencies like 'n_dims' vs the project's 'ndim' convention, or \
             'numFeatures' vs 'nb_features'. Returns a consistent/inconsistent \
             verdict with the canonical form and suggestions. Use before committing \
             new code or when reviewing identifier names.",
            tool_schema(
                json!({
                    "identifier": {
                        "type": "string",
                        "description": "The identifier to check (e.g. 'n_dims')",
                    }
                }),
                &["identifier"],
            ),
        ),
        Tool::new(
            "suggest_name",
            "Generate project-consistent identifier names from a natural language \
             description. Uses the project's actual conventions (prefixes like nb_, \
             patterns like is_/has_, conversion patterns like x_to_y) to suggest \
             names that fit the existing codebase style. Use when naming new \
             functions, variables, or parameters.",
            tool_schema(
                json!({
                    "description": {
                        "type": "string",
                        "description": "Natural language description (e.g. 'count of features')",
                    }
                }),
                &["description"],
            ),
        ),
        Tool::new(
            "ontology_diff",
            "Compare the domain ontology between two git revisions — shows \
             concepts that were added, removed, or changed. Useful for \
             understanding how the project's vocabulary evolved across commits \
             or for reviewing whether a PR introduced naming inconsistencies.",
            tool_schema(
                json!({
                    "since": {
                        "type": "string",
                        "description": "Git ref to diff from (default: HEAD~5)",
                    }
                }),
                &[],
            ),
        ),
        Tool::new(
            "list_concepts",
            "List the project's domain vocabulary ranked by importance — a \
             semantic overview of what this codebase is about that reading \
             individual files cannot provide. Returns concept names with \
             occurrence counts and entity counts. Use when asked 'what is \
             this project about', 'what are the main concepts', or when \
             orienting in an unfamiliar codebase.",
            tool_schema(
                json!({
                    "top_k": {
                        "type": "integer",
                        "description": "Maximum number of concepts to return",
                    }
                }),
                &[],
            ),
        ),
        Tool::new(
            "list_conventions",
            "List the project's actual naming conventions detected from code — \
             prefix patterns (nb_, is_, has_), suffix patterns, conversion \
             patterns (x_to_y), and casing rules, each with real examples from \
             the codebase. More accurate than guessing from a few files. Use \
             when asked about coding style, before writing new code, or when \
             onboarding to a project.",
            tool_schema(json!({}), &[]),
        ),
        Tool::new(
            "describe_symbol",
            "Describe a function or class WITHOUT reading its source file — \
             returns signature, parameters, callers, callees, semantic role, \
             and related domain concepts. Faster and more informative than \
             Read for understanding what a symbol does and how it fits into \
             the codebase. Use when asked 'what does function X do' or \
             'what is class X'.",
            tool_schema(
                json!({
                    "name": {
                        "type": "string",
                        "description": "Function or class name (e.g. 'spatial_transform')",
                    }
                }),
                &["name"],
            ),
        ),
        Tool::new(
            "locate_concept",
            "Find the best entry points for understanding a concept — returns \
             a ranked shortlist of key functions, classes, and files to read, \
             plus contrastive concepts that clarify boundaries. Saves reading \
             dozens of grep matches by surfacing the most important locations \
             first. Use when asked 'how does X work', 'where should I look \
             for X', or 'where are X defined'.",
            tool_schema(
                json!({
                    "term": {
                        "type": "string",
                        "description": "Concept to locate (e.g. 'transform')",
                    }
                }),
                &["term"],
            ),
        ),
        Tool::new(
            "export_domain_pack",
            "Export the project's full domain knowledge as a portable YAML \
             pack — abbreviations, conventions, domain terms, and concept \
             associations. Use when asked to 'export conventions', 'create \
             a domain pack', or 'share naming rules'. The output can be \
             reviewed, curated, and saved for use in other projects or tools.",
            tool_schema(json!({}), &[]),
        ),
        Tool::new(
            "list_entities",
            "Find all classes and functions matching a semantic role or concept — \
             e.g. all loss functions, all network architectures, all transform \
             utilities. Returns entities with semantic roles and concept tags. \
             Impossible with grep alone because it understands which functions \
             *implement* a concept, not just mention it. Use when asked 'what \
             loss functions exist', 'show me the network classes', 'what uses \
             concept X', or 'list all X'.",
            tool_schema(
                json!({
                    "concept": {
                        "type": "string",
                        "description": "Filter by concept (e.g. 'loss')",
                    },
                    "role": {
                        "type": "string",
                        "description": "Filter by semantic role substring (e.g. 'module')",
                    },
                    "kind": {
                        "type": "string",
                        "enum": ["class", "function"],
                        "description": "Filter by entity kind",
                    },
                    "top_k": {
                        "type": "integer",
                        "description": "Max entities to return (default: 20)",
                    }
                }),
                &[],
            ),
        ),
        Tool::new(
            "vocabulary_health",
            "Measure the vocabulary health of this codebase \
             — returns convention coverage (how many identifiers \
             follow conventions), consistency ratio (how uniformly \
             concepts are spelled), and cluster cohesion (how well \
             semantic clusters hold together). Use when asked about \
             code quality, naming consistency, or vocabulary health. \
             Returns an overall score plus top inconsistencies and \
             uncovered identifiers.",
            tool_schema(json!({}), &[]),
        ),
        Tool::new(
            "trace_concept",
            "Trace how a domain concept flows through the codebase \
             via function call chains — shows which functions produce \
             the concept, which consume it, and the call path between \
             them including bridge functions on the path. Returns an \
             ordered call chain with file locations and producer/ \
             consumer roles. Use when asked 'how does concept X \
             propagate', 'what calls what for X', or 'trace X through \
             the code'.",
            tool_schema(
                json!({
                    "concept": {
                        "type": "string",
                        "description": "The concept to trace \
                            (e.g. 'transform')",
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum call chain depth \
                            (default: 5)",
                    }
                }),
                &["concept"],
            ),
        ),
    ]
}

impl ServerHandler for OntomicsServer {
    fn get_info(&self) -> InitializeResult {
        InitializeResult {
            protocol_version: Default::default(),
            capabilities: ServerCapabilities::builder()
                .enable_tools_with(ToolsCapability {
                    list_changed: Some(false),
                })
                .enable_resources()
                .build(),
            server_info: Implementation {
                name: "ontomics".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            instructions: Some(
                "ontomics is a SEMANTIC INDEX of this codebase. It knows every domain \
                 concept, naming convention, abbreviation, and function relationship. \
                 It resolves abbreviations (trf → transform, nb_ → count prefix), finds \
                 related concepts, and gives semantic context that text search cannot.\n\
                 \n\
                 ALWAYS try ontomics BEFORE grep/glob/read for these questions:\n\
                 - \"what is this codebase about\" / \"what are the main concepts\" → list_concepts\n\
                 - \"what is X\" / \"what does X mean\" → query_concept\n\
                 - \"where is X\" / \"how does X work\" / \"where are X defined\" → locate_concept\n\
                 - \"what does function/class X do\" → describe_symbol\n\
                 - \"what X functions/classes exist\" / \"list all X\" → list_entities\n\
                 - \"what naming conventions\" / \"what style\" → list_conventions\n\
                 - \"is this name right\" → check_naming\n\
                 - \"what should I call this\" → suggest_name\n\
                 - \"how does X flow\" / \"trace X\" / \"call chain for X\" → trace_concept\n\
                 \n\
                 Fall back to file reading ONLY when ontomics returns insufficient detail."
                    .to_string(),
            ),
        }
    }

    fn list_tools(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ListToolsResult, rmcp::Error>>
           + Send
           + '_ {
        std::future::ready(Ok(ListToolsResult {
            tools: tool_definitions(),
            next_cursor: None,
        }))
    }

    fn call_tool(
        &self,
        request: CallToolRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<CallToolResult, rmcp::Error>>
           + Send
           + '_ {
        let name: &str = &request.name;

        if !self.indexing_ready.load(Ordering::Acquire) {
            let msg = json!({
                "status": "indexing_in_progress",
                "message": "ontomics is still indexing this codebase. \
                    Results will be available shortly. Please try again in a moment.",
                "tool": name,
            });
            return std::future::ready(Ok(CallToolResult::success(vec![
                Content::text(serde_json::to_string_pretty(&msg).unwrap_or_default()),
            ])));
        }

        let args = request
            .arguments
            .as_ref()
            .map(|m| Value::Object(m.clone()))
            .unwrap_or(Value::Object(Default::default()));

        let result = match name {
            "query_concept" => self.handle_query_concept(&args),
            "check_naming" => self.handle_check_naming(&args),
            "suggest_name" => self.handle_suggest_name(&args),
            "ontology_diff" => self.handle_ontology_diff(&args),
            "list_concepts" => self.handle_list_concepts(&args),
            "list_conventions" => self.handle_list_conventions(&args),
            "describe_symbol" => self.handle_describe_symbol(&args),
            "locate_concept" => self.handle_locate_concept(&args),
            "list_entities" => self.handle_list_entities(&args),
            "export_domain_pack" => self.handle_export_domain_pack(&args),
            "vocabulary_health" => self.handle_vocabulary_health(&args),
            "trace_concept" => self.handle_trace_concept(&args),
            other => Err(format!("unknown tool: {other}")),
        };

        std::future::ready(Ok(match result {
            Ok(value) => {
                let pretty = serde_json::to_string_pretty(&value)
                    .unwrap_or_default();
                let text = if pretty.len() > OUTPUT_BUDGET_BYTES {
                    // Fall back to compact JSON to save ~30% on large output
                    serde_json::to_string(&value).unwrap_or(pretty)
                } else {
                    pretty
                };
                CallToolResult::success(vec![Content::text(text)])
            }
            Err(msg) => CallToolResult::error(vec![Content::text(msg)]),
        }))
    }

    fn list_resources(
        &self,
        _request: PaginatedRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<rmcp::model::ListResourcesResult, rmcp::Error>>
           + Send
           + '_ {
        let resource = Resource::new(
            RawResource {
                uri: "ontomics://briefing".to_string(),
                name: "Session Briefing".to_string(),
                description: Some(
                    "Project conventions, abbreviations, top concepts, \
                     contrastive pairs, and vocabulary warnings."
                        .to_string(),
                ),
                mime_type: Some("application/json".to_string()),
                size: None,
            },
            None,
        );
        std::future::ready(Ok(rmcp::model::ListResourcesResult {
            resources: vec![resource],
            next_cursor: None,
        }))
    }

    fn read_resource(
        &self,
        request: ReadResourceRequestParam,
        _context: RequestContext<RoleServer>,
    ) -> impl std::future::Future<Output = Result<ReadResourceResult, rmcp::Error>>
           + Send
           + '_ {
        let result = if !self.indexing_ready.load(Ordering::Acquire) {
            let msg = json!({
                "status": "indexing_in_progress",
                "message": "ontomics is still indexing this codebase. \
                    The briefing will be available shortly.",
            });
            Ok(ReadResourceResult {
                contents: vec![ResourceContents::text(
                    serde_json::to_string_pretty(&msg).unwrap_or_default(),
                    "ontomics://briefing",
                )],
            })
        } else if request.uri == "ontomics://briefing" {
            match self.graph.read() {
                Ok(graph) => {
                    let mut briefing = graph.session_briefing();
                    let warnings = self.warnings.lock()
                        .map(|w| w.clone())
                        .unwrap_or_default();
                    for w in warnings {
                        briefing.vocabulary_warnings.push(w);
                    }
                    let json = serde_json::to_string_pretty(&briefing)
                        .unwrap_or_default();
                    Ok(ReadResourceResult {
                        contents: vec![ResourceContents::text(
                            json,
                            "ontomics://briefing",
                        )],
                    })
                }
                Err(e) => Err(rmcp::Error::internal_error(
                    format!("lock error: {e}"),
                    None,
                )),
            }
        } else {
            Err(rmcp::Error::resource_not_found(
                "resource not found",
                Some(json!({"uri": request.uri})),
            ))
        };
        std::future::ready(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ontomics::embeddings::EmbeddingIndex;
    use ontomics::graph::ConceptGraph;
    use ontomics::types::*;
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn make_concept(
        id: u64,
        canonical: &str,
        identifiers: &[&str],
    ) -> Concept {
        Concept {
            id,
            canonical: canonical.to_string(),
            subtokens: vec![canonical.to_string()],
            occurrences: identifiers
                .iter()
                .map(|name| Occurrence {
                    file: PathBuf::from("test.py"),
                    line: 1,
                    identifier: name.to_string(),
                    entity_type: EntityType::Function,
                })
                .collect(),
            entity_types: HashSet::from([EntityType::Function]),
            embedding: None,
            cluster_id: None,
            subconcepts: Vec::new(),
        }
    }

    fn make_test_server() -> OntomicsServer {
        let analysis = AnalysisResult {
            concepts: vec![
                make_concept(
                    1,
                    "transform",
                    &["spatial_transform", "apply_transform"],
                ),
                make_concept(2, "spatial", &["spatial_transform"]),
            ],
            conventions: vec![Convention {
                pattern: PatternKind::Prefix("nb_".to_string()),
                entity_type: EntityType::Parameter,
                semantic_role: "count".to_string(),
                examples: vec![
                    "nb_features".into(),
                    "nb_bins".into(),
                    "nb_steps".into(),
                ],
                frequency: 3,
            }],
            co_occurrence_matrix: vec![((1, 2), 1.0)],
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
        };
        let graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();
        OntomicsServer::new(
            graph,
            PathBuf::from("/tmp"),
            Arc::new(ontomics::parser::python_parser()),
            Vec::new(),
        )
    }

    #[test]
    fn test_query_concept_found() {
        let server = make_test_server();
        let args = json!({"term": "transform"});
        let result = server.handle_query_concept(&args);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert!(val.get("concept").is_some());
    }

    #[test]
    fn test_query_concept_not_found() {
        let server = make_test_server();
        let args = json!({"term": "nonexistent"});
        let result = server.handle_query_concept(&args);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert_eq!(val.get("error").unwrap(), "not_found");
    }

    #[test]
    fn test_check_naming() {
        let server = make_test_server();
        let args = json!({"identifier": "nb_features"});
        let result = server.handle_check_naming(&args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_suggest_name() {
        let server = make_test_server();
        let args = json!({"description": "count of features"});
        let result = server.handle_suggest_name(&args);
        assert!(result.is_ok());
    }

    #[test]
    fn test_list_concepts_with_top_k() {
        let server = make_test_server();
        let args = json!({"top_k": 1});
        let result = server.handle_list_concepts(&args).unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn test_list_conventions() {
        let server = make_test_server();
        let args = json!({});
        let result = server.handle_list_conventions(&args).unwrap();
        let arr = result.as_array().unwrap();
        assert_eq!(arr.len(), 1);
    }

    #[test]
    fn test_ontology_diff_returns_error_for_non_repo() {
        let server = make_test_server();
        let args = json!({"since": "HEAD~3"});
        let result = server.handle_ontology_diff(&args);
        // /tmp is not a git repo, so this should error
        assert!(result.is_err());
    }

    #[test]
    fn test_missing_required_arg() {
        let server = make_test_server();
        let args = json!({});
        let result = server.handle_query_concept(&args);
        assert!(result.is_err());
    }

    #[test]
    fn test_tool_definitions_count() {
        let tools = tool_definitions();
        assert_eq!(tools.len(), 12);
    }

    #[test]
    fn test_get_info() {
        let server = make_test_server();
        let info = server.get_info();
        assert_eq!(info.server_info.name, "ontomics");
        assert!(info.capabilities.tools.is_some());
    }

    #[test]
    fn test_describe_symbol_not_found() {
        let server = make_test_server();
        let args = json!({"name": "nonexistent"});
        let result = server.handle_describe_symbol(&args);
        assert!(result.is_ok());
        let val = result.unwrap();
        assert_eq!(val.get("error").unwrap(), "not_found");
    }
}

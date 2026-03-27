use crate::diff;
use crate::graph::ConceptGraph;
use rmcp::handler::server::ServerHandler;
use rmcp::model::{
    CallToolRequestParam, CallToolResult, Content, Implementation, InitializeResult,
    ListToolsResult, PaginatedRequestParam, ServerCapabilities, Tool, ToolsCapability,
};
use rmcp::service::{RequestContext, RoleServer};
use serde_json::{json, Value};
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Clone)]
pub struct SemexServer {
    graph: Arc<ConceptGraph>,
    repo_root: PathBuf,
}

impl SemexServer {
    pub fn new(graph: ConceptGraph, repo_root: PathBuf) -> Self {
        Self {
            graph: Arc::new(graph),
            repo_root,
        }
    }

    fn handle_query_concept(&self, args: &Value) -> Result<Value, String> {
        let term = args
            .get("term")
            .and_then(|v| v.as_str())
            .ok_or("missing required argument 'term'")?;

        match self.graph.query_concept(term) {
            Some(result) => serde_json::to_value(result)
                .map_err(|e| format!("serialization error: {e}")),
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

        let result = self.graph.check_naming(identifier);
        serde_json::to_value(result)
            .map_err(|e| format!("serialization error: {e}"))
    }

    fn handle_suggest_name(&self, args: &Value) -> Result<Value, String> {
        let description = args
            .get("description")
            .and_then(|v| v.as_str())
            .ok_or("missing required argument 'description'")?;

        let result = self.graph.suggest_name(description);
        serde_json::to_value(result)
            .map_err(|e| format!("serialization error: {e}"))
    }

    fn handle_ontology_diff(&self, args: &Value) -> Result<Value, String> {
        let since = args
            .get("since")
            .and_then(|v| v.as_str())
            .unwrap_or("HEAD~5");

        let result = diff::ontology_diff(
            &self.repo_root,
            since,
            &self.graph.concepts,
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

        let mut concepts = self.graph.list_concepts();
        if let Some(k) = top_k {
            concepts.truncate(k);
        }

        let summary: Vec<Value> = concepts
            .iter()
            .map(|c| {
                json!({
                    "canonical": c.canonical,
                    "occurrences": c.occurrences.len(),
                    "entity_types": c.entity_types,
                })
            })
            .collect();

        Ok(json!(summary))
    }

    fn handle_list_conventions(&self, _args: &Value) -> Result<Value, String> {
        let conventions = self.graph.list_conventions();
        serde_json::to_value(conventions)
            .map_err(|e| format!("serialization error: {e}"))
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
            "Look up a domain concept by name. Returns the concept, its variants, \
             related concepts, matching conventions, and top occurrences.",
            tool_schema(
                json!({
                    "term": {
                        "type": "string",
                        "description": "The concept term to look up (e.g. 'transform')",
                    }
                }),
                &["term"],
            ),
        ),
        Tool::new(
            "check_naming",
            "Check an identifier against project naming conventions. Returns \
             whether the name is consistent, inconsistent, or unknown, with \
             suggestions if applicable.",
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
            "Suggest an identifier name given a natural language description, \
             based on project conventions and existing vocabulary.",
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
            "Compare the domain ontology between two git revisions. Shows \
             added, removed, and changed concepts.",
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
            "List all detected domain concepts, ordered by frequency. \
             Optionally truncate to top_k.",
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
            "List all detected naming conventions (prefix, suffix, conversion \
             patterns) with examples and frequency counts.",
            tool_schema(json!({}), &[]),
        ),
    ]
}

impl ServerHandler for SemexServer {
    fn get_info(&self) -> InitializeResult {
        InitializeResult {
            protocol_version: Default::default(),
            capabilities: ServerCapabilities::builder()
                .enable_tools_with(ToolsCapability {
                    list_changed: Some(false),
                })
                .build(),
            server_info: Implementation {
                name: "semex".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
            instructions: Some(
                "semex extracts domain ontologies from Python codebases. \
                 Use query_concept to explore vocabulary, check_naming to \
                 validate identifiers, and list_conventions to see patterns."
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
            other => Err(format!("unknown tool: {other}")),
        };

        std::future::ready(Ok(match result {
            Ok(value) => CallToolResult::success(vec![
                Content::text(serde_json::to_string_pretty(&value).unwrap_or_default()),
            ]),
            Err(msg) => CallToolResult::error(vec![Content::text(msg)]),
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingIndex;
    use crate::graph::ConceptGraph;
    use crate::types::*;
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
        }
    }

    fn make_test_server() -> SemexServer {
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
        };
        let graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();
        SemexServer::new(graph, PathBuf::from("/tmp"))
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
        assert_eq!(tools.len(), 6);
    }

    #[test]
    fn test_get_info() {
        let server = make_test_server();
        let info = server.get_info();
        assert_eq!(info.server_info.name, "semex");
        assert!(info.capabilities.tools.is_some());
    }
}

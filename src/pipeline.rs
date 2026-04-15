use std::collections::HashMap;
use std::path::{Path, PathBuf};

use anyhow::Result;

use crate::analyzer;
use crate::cache;
use crate::config;
use crate::domain_pack;
use crate::embeddings;
use crate::entity;
use crate::graph;
#[cfg(feature = "lsp")]
use crate::lsp;
use crate::parser;
use crate::resolve;
use crate::types;

/// Result of building the initial graph. If embeddings were deferred,
/// `needs_embedding` contains what the background thread needs.
pub struct BuildResult {
    pub graph: graph::ConceptGraph,
    pub needs_embedding: Option<EmbeddingWork>,
}

/// Work deferred to a background thread for embedding enrichment.
/// Holds only the concept list (not full analysis) so it can be
/// reconstructed from a cached graph with partial embeddings.
pub struct EmbeddingWork {
    pub concepts: Vec<types::Concept>,
    pub model_cache_dir: Option<PathBuf>,
    pub similarity_threshold: f32,
    pub batch_size: usize,
    pub domain_packs: Vec<types::DomainPack>,
    pub logic_enabled: bool,
    pub logic_similarity_threshold: f32,
}

/// A language-parser pair: `(parser, language_name)`.
pub type LangParser = (Box<dyn parser::LanguageParser>, String);

/// Builder that runs the ontomics pipeline step by step.
///
/// Each method corresponds to a pipeline stage and returns
/// `Result<&mut Self>` for chaining. The `build()` method
/// finalizes and returns the `BuildResult`.
pub struct GraphBuilder<'a> {
    repo: &'a Path,
    config: &'a config::Config,
    defer_embeddings: bool,
    lang_parsers: &'a [LangParser],
    warnings: &'a mut Vec<String>,

    // Accumulated state
    cache: Option<cache::IndexCache>,
    model_cache_dir: Option<PathBuf>,
    loaded_packs: Vec<types::DomainPack>,
    lang_key: String,
    lang_names: Vec<String>,

    // Pipeline products
    all_parse_results: Vec<types::ParseResult>,
    per_lang_results: Vec<(String, Vec<types::ParseResult>)>,
    all_imports: Vec<types::ImportStatement>,
    analysis: Option<types::AnalysisResult>,
    built_entities: Vec<types::Entity>,
    entity_rels: Vec<types::Relationship>,
    graph: Option<graph::ConceptGraph>,

    // Early return when cache hit
    early_result: Option<BuildResult>,
}

impl<'a> GraphBuilder<'a> {
    pub fn new(
        repo: &'a Path,
        config: &'a config::Config,
        defer_embeddings: bool,
        lang_parsers: &'a [LangParser],
        warnings: &'a mut Vec<String>,
    ) -> Self {
        assert!(!lang_parsers.is_empty(), "at least one language required");

        let lang_names: Vec<String> =
            lang_parsers.iter().map(|(_, n)| n.clone()).collect();
        let lang_key = lang_names.join(",");

        Self {
            repo,
            config,
            defer_embeddings,
            lang_parsers,
            warnings,
            cache: None,
            model_cache_dir: None,
            loaded_packs: Vec::new(),
            lang_key,
            lang_names,
            all_parse_results: Vec::new(),
            per_lang_results: Vec::new(),
            all_imports: Vec::new(),
            analysis: None,
            built_entities: Vec::new(),
            entity_rels: Vec::new(),
            graph: None,
            early_result: None,
        }
    }

    /// Open the cache and load domain packs. Checks for a cached
    /// graph and returns early if found (with optional deferred
    /// embedding work).
    pub fn load_cache(&mut self) -> Result<&mut Self> {
        self.cache = match cache::IndexCache::open(self.repo) {
            Ok(c) => Some(c),
            Err(e) => {
                let msg = format!(
                    "Cache unavailable: {e}. \
                     ontomics will work but won't persist the index \
                     across restarts."
                );
                eprintln!("{msg}");
                self.warnings.push(msg);
                None
            }
        };

        self.model_cache_dir =
            resolve_cache_dir(&self.config.embeddings.model_cache_dir);
        self.loaded_packs = load_domain_packs(self.repo, self.config);

        let lang_name_refs: Vec<&str> =
            self.lang_names.iter().map(|s| s.as_str()).collect();
        let cached_graph = self.cache.as_ref()
            .and_then(|c| c.load_multi(&lang_name_refs).transpose())
            .transpose()?;

        if let Some(mut cached) = cached_graph {
            let nb_concepts = cached.concepts.len();
            let nb_embedded = cached.embeddings.nb_vectors();
            eprintln!(
                "Loaded graph from cache \
                 ({nb_embedded}/{nb_concepts} embedded)"
            );

            if self.config.embeddings.enabled {
                if let Some(ref dir) = self.model_cache_dir {
                    cached.embeddings.set_cache_dir(dir.clone());
                }

                if self.defer_embeddings && nb_embedded < nb_concepts {
                    let concepts: Vec<types::Concept> =
                        cached.concepts.values().cloned().collect();
                    self.early_result = Some(BuildResult {
                        graph: cached,
                        needs_embedding: Some(EmbeddingWork {
                            concepts,
                            model_cache_dir:
                                self.model_cache_dir.clone(),
                            similarity_threshold: self
                                .config
                                .embeddings
                                .similarity_threshold,
                            batch_size: self
                                .config
                                .resources
                                .embedding_batch_size,
                            domain_packs: self.loaded_packs.clone(),
                            logic_enabled:
                                self.config.logic.enabled,
                            logic_similarity_threshold:
                                self.config.logic.similarity_threshold,
                        }),
                    });
                    return Ok(self);
                }

                if let Err(e) = cached.embeddings.load_model() {
                    let msg = format!(
                        "Embedding model unavailable: {e}. \
                         Concept clustering and similarity search \
                         are disabled. Restart ontomics to retry."
                    );
                    eprintln!("{msg}");
                    self.warnings.push(msg);
                }
            }
            cached.recompute_centroids();
            self.early_result = Some(BuildResult {
                graph: cached,
                needs_embedding: None,
            });
        }
        Ok(self)
    }

    /// Parse source files for each language, using cached parse
    /// results for fresh ones.
    pub fn parse(&mut self) -> Result<&mut Self> {
        if self.early_result.is_some() {
            return Ok(self);
        }

        let lang_name_refs: Vec<&str> =
            self.lang_names.iter().map(|s| s.as_str()).collect();
        let stale = self.cache
            .as_ref()
            .and_then(|c| c.stale_languages(&lang_name_refs).ok())
            .unwrap_or_else(|| {
                lang_name_refs.iter().map(|s| s.to_string()).collect()
            });

        eprintln!("Indexing {}...", self.repo.display());

        for (lp, lang_name) in self.lang_parsers {
            if stale.contains(lang_name) {
                let lang_enum = lang_name.parse::<config::Language>()
                    .unwrap_or(config::Language::Python);
                let parse_opts = parser::ParseOptions {
                    include: lang_enum.default_include(),
                    exclude: lang_enum.default_exclude(),
                    respect_gitignore:
                        self.config.index.respect_gitignore,
                };
                let results = parser::parse_directory_with(
                    self.repo, &parse_opts, &**lp,
                )?;
                eprintln!(
                    "Parsed {} {} files",
                    results.len(),
                    lang_name,
                );
                self.all_parse_results.extend(results.clone());
                self.per_lang_results
                    .push((lang_name.clone(), results));
            } else if let Some(ref c) = self.cache {
                if let Some(cached_results) =
                    c.load_parse_results(lang_name)?
                {
                    eprintln!(
                        "Loaded {} cached {} parse results",
                        cached_results.len(),
                        lang_name,
                    );
                    self.all_parse_results
                        .extend(cached_results.clone());
                    self.per_lang_results
                        .push((lang_name.clone(), cached_results));
                }
            }
        }

        eprintln!(
            "Total: {} parsed files",
            self.all_parse_results.len()
        );
        Ok(self)
    }

    /// Run TF-IDF scoring and extract concepts and conventions.
    pub fn analyze(&mut self) -> Result<&mut Self> {
        if self.early_result.is_some() {
            return Ok(self);
        }

        self.all_imports = self.all_parse_results
            .iter()
            .flat_map(|r| r.imports.iter().cloned())
            .collect();
        resolve::resolve_imports(&mut self.all_imports, self.repo);

        let analysis_params = analyzer::AnalysisParams {
            min_frequency: self.config.index.min_frequency,
            tfidf_threshold:
                self.config.analysis.domain_specificity_threshold,
            convention_threshold:
                self.config.analysis.convention_threshold,
            language: self.lang_key.clone(),
        };
        self.analysis = Some(
            analyzer::analyze(
                &self.all_parse_results,
                &analysis_params,
            )?
        );
        Ok(self)
    }

    /// Merge bootstrap conventions from the config file.
    pub fn merge_conventions(&mut self) -> Result<&mut Self> {
        if self.early_result.is_some() {
            return Ok(self);
        }

        let analysis = self.analysis.as_mut()
            .expect("analyze() must run before merge_conventions()");

        for conv_cfg in &self.config.conventions {
            let pattern = match conv_cfg.pattern.as_str() {
                "prefix" => {
                    types::PatternKind::Prefix(
                        conv_cfg.value.clone(),
                    )
                }
                "suffix" => {
                    types::PatternKind::Suffix(
                        conv_cfg.value.clone(),
                    )
                }
                "compound" => {
                    types::PatternKind::Compound(
                        conv_cfg.value.clone(),
                    )
                }
                "conversion" => {
                    types::PatternKind::Conversion(
                        conv_cfg.value.clone(),
                    )
                }
                other => {
                    eprintln!(
                        "Warning: unknown convention pattern \
                         '{other}', skipping"
                    );
                    continue;
                }
            };
            let entity_type = conv_cfg
                .entity_types
                .first()
                .and_then(|s| match s.as_str() {
                    "Function" => Some(types::EntityType::Function),
                    "Parameter" => {
                        Some(types::EntityType::Parameter)
                    }
                    "Variable" => Some(types::EntityType::Variable),
                    "Class" => Some(types::EntityType::Class),
                    _ => None,
                })
                .unwrap_or(types::EntityType::Variable);

            let already_exists = analysis.conventions.iter().any(
                |c| c.pattern == pattern,
            );
            if !already_exists {
                analysis.conventions.push(types::Convention {
                    pattern,
                    entity_type,
                    semantic_role: conv_cfg.role.clone(),
                    examples: Vec::new(),
                    frequency: 0,
                });
            }
        }
        Ok(self)
    }

    /// Merge domain pack conventions and terms into analysis.
    pub fn merge_domain_packs(&mut self) -> Result<&mut Self> {
        if self.early_result.is_some() {
            return Ok(self);
        }

        let analysis = self.analysis.as_mut()
            .expect(
                "analyze() must run before merge_domain_packs()"
            );

        for pack in &self.loaded_packs {
            domain_pack::merge_pack_into_analysis(pack, analysis);
        }

        eprintln!(
            "Found {} concepts, {} conventions",
            analysis.concepts.len(),
            analysis.conventions.len()
        );
        Ok(self)
    }

    /// Promote functions/classes to entities with concept tags.
    pub fn build_entities(&mut self) -> Result<&mut Self> {
        if self.early_result.is_some() {
            return Ok(self);
        }

        let analysis = self.analysis.as_ref()
            .expect(
                "analyze() must run before build_entities()"
            );

        let concept_map: HashMap<u64, types::Concept> = analysis
            .concepts
            .iter()
            .map(|c| (c.id, c.clone()))
            .collect();
        let (entities, rels) = entity::build_entities(
            &analysis.signatures,
            &analysis.classes,
            &analysis.call_sites,
            &concept_map,
            &self.all_imports,
        );
        eprintln!("Built {} entities", entities.len());
        self.built_entities = entities;
        self.entity_rels = rels;
        Ok(self)
    }

    /// Handle embedding: deferred path returns EmbeddingWork,
    /// synchronous path embeds concepts inline.
    pub fn embed(&mut self) -> Result<&mut Self> {
        if self.early_result.is_some() {
            return Ok(self);
        }

        let analysis = self.analysis.take()
            .expect("analyze() must run before embed()");

        if self.defer_embeddings && self.config.embeddings.enabled {
            return self.embed_deferred(analysis);
        }
        self.embed_synchronous(analysis)
    }

    fn embed_deferred(
        &mut self,
        analysis: types::AnalysisResult,
    ) -> Result<&mut Self> {
        let concepts = analysis.concepts.clone();
        let classes_for_roles = analysis.classes.clone();
        let entities = std::mem::take(&mut self.built_entities);
        let entity_rels = std::mem::take(&mut self.entity_rels);

        let mut g = graph::ConceptGraph::build_with_entities(
            analysis,
            embeddings::EmbeddingIndex::empty(),
            entities,
            entity_rels,
            self.all_imports.clone(),
        )?;
        g.add_abbreviation_edges();
        g.add_contrastive_edges();
        for pack in &self.loaded_packs {
            domain_pack::merge_pack_associations(pack, &mut g);
        }

        // Infer semantic roles
        let mut ents: Vec<types::Entity> =
            g.entities.values().cloned().collect();
        entity::infer_semantic_roles(
            &mut ents,
            &classes_for_roles,
            &g.conventions,
            &g.concepts,
        );
        for ent in ents {
            g.entities.insert(ent.id, ent);
        }

        // Centrality (no model needed)
        if self.config.centrality.enabled {
            let centrality =
                crate::centrality::compute_centrality(
                    &g.entities,
                    &g.relationships,
                    self.config.centrality.damping,
                    self.config.centrality.iterations,
                );
            eprintln!(
                "Computed centrality for {} entities",
                centrality.len()
            );
            g.centrality = centrality;
        }

        // Cache the lightweight graph
        if let Some(ref c) = self.cache {
            let lang_name_refs: Vec<&str> =
                self.lang_names.iter().map(|s| s.as_str()).collect();
            let per_lang_refs: Vec<(&str, &[types::ParseResult])> =
                self.per_lang_results
                    .iter()
                    .map(|(n, r)| (n.as_str(), r.as_slice()))
                    .collect();
            if let Err(e) =
                c.save_multi(&g, &lang_name_refs, &per_lang_refs)
            {
                eprintln!(
                    "Warning: failed to cache initial index: {e}"
                );
            }
        }

        eprintln!(
            "Graph ready (embeddings deferred to background)"
        );
        self.early_result = Some(BuildResult {
            graph: g,
            needs_embedding: Some(EmbeddingWork {
                concepts,
                model_cache_dir: self.model_cache_dir.clone(),
                similarity_threshold:
                    self.config.embeddings.similarity_threshold,
                batch_size:
                    self.config.resources.embedding_batch_size,
                domain_packs: self.loaded_packs.clone(),
                logic_enabled: self.config.logic.enabled,
                logic_similarity_threshold:
                    self.config.logic.similarity_threshold,
            }),
        });
        Ok(self)
    }

    fn embed_synchronous(
        &mut self,
        analysis: types::AnalysisResult,
    ) -> Result<&mut Self> {
        let logic_cache_dir = self.model_cache_dir.clone();
        let mut embedding_index = if self.config.embeddings.enabled {
            match embeddings::EmbeddingIndex::new(
                self.model_cache_dir.take(),
            ) {
                Ok(idx) => {
                    eprintln!("Loaded embedding model");
                    idx
                }
                Err(e) => {
                    let msg = format!(
                        "Embedding model unavailable: {e}. \
                         Concept clustering and similarity search \
                         are disabled. Restart ontomics to retry."
                    );
                    eprintln!("{msg}");
                    self.warnings.push(msg);
                    embeddings::EmbeddingIndex::empty()
                }
            }
        } else {
            embeddings::EmbeddingIndex::empty()
        };

        if self.config.embeddings.enabled {
            if let Err(e) = embedding_index
                .embed_concepts_batch(&analysis.concepts)
            {
                eprintln!(
                    "Warning: failed to embed concepts: {e}"
                );
            }
        }

        let entities = std::mem::take(&mut self.built_entities);
        let entity_rels = std::mem::take(&mut self.entity_rels);
        let imports = std::mem::take(&mut self.all_imports);

        let mut g = graph::ConceptGraph::build_with_entities(
            analysis,
            embedding_index,
            entities,
            entity_rels,
            imports,
        )?;

        if self.config.embeddings.enabled {
            g.cluster_and_add_similarity_edges(
                self.config.embeddings.similarity_threshold,
            );
        }

        g.add_abbreviation_edges();
        g.add_contrastive_edges();
        g.detect_subconcepts();
        for pack in &self.loaded_packs {
            domain_pack::merge_pack_associations(pack, &mut g);
        }

        // Infer semantic roles
        let mut ents: Vec<types::Entity> =
            g.entities.values().cloned().collect();
        entity::infer_semantic_roles(
            &mut ents,
            &g.classes,
            &g.conventions,
            &g.concepts,
        );
        for ent in ents {
            g.entities.insert(ent.id, ent);
        }

        // Optional LSP enrichment
        #[cfg(feature = "lsp")]
        if self.config.lsp.enabled {
            eprintln!("Running pyright LSP enrichment...");
            match lsp::enrich_with_pyright(
                self.repo,
                &self.config.lsp.pyright_path,
                self.config.lsp.timeout_secs,
            ) {
                Ok(enrichment) => {
                    let ents: Vec<types::Entity> =
                        g.entities.values().cloned().collect();
                    lsp::apply_enrichment(
                        &ents,
                        &enrichment,
                        &mut g.relationships,
                    );
                    eprintln!("LSP enrichment applied");
                }
                Err(e) => {
                    eprintln!(
                        "Warning: LSP enrichment failed: {e}"
                    );
                }
            }
        }

        // Logic embeddings
        if self.config.logic.enabled
            && self.config.embeddings.enabled
        {
            match crate::logic::LogicIndex::new(logic_cache_dir) {
                Ok(mut logic_idx) => {
                    let items: Vec<(u64, String)> = g.signatures
                        .iter()
                        .filter_map(|sig| {
                            let body = sig.body.as_ref()?;
                            let entity =
                                g.entities.values().find(|e| {
                                    e.name == sig.name
                                        && e.file == sig.file
                                })?;
                            Some((
                                entity.id,
                                body.body_text.clone(),
                            ))
                        })
                        .collect();
                    if let Err(e) = logic_idx.embed_batch(items) {
                        eprintln!(
                            "Warning: logic embedding failed: {e}"
                        );
                    }
                    let entity_ids: Vec<u64> =
                        g.entities.keys().copied().collect();
                    g.logic_clusters =
                        crate::logic::cluster_logic(
                            &logic_idx,
                            &entity_ids,
                            1.0 - self
                                .config
                                .logic
                                .similarity_threshold,
                        );
                    let mut clusters =
                        std::mem::take(&mut g.logic_clusters);
                    let entity_cs =
                        build_entity_call_sites(&g);
                    crate::logic::label_clusters(
                        &mut clusters,
                        &entity_cs,
                    );
                    g.logic_clusters = clusters;
                    eprintln!(
                        "Logic: {} embeddings, {} clusters",
                        logic_idx.nb_vectors(),
                        g.logic_clusters.len()
                    );
                    g.logic_index = logic_idx;
                }
                Err(e) => {
                    eprintln!(
                        "Warning: logic model unavailable: {e}"
                    );
                }
            }
        }

        // Centrality
        if self.config.centrality.enabled {
            let centrality =
                crate::centrality::compute_centrality(
                    &g.entities,
                    &g.relationships,
                    self.config.centrality.damping,
                    self.config.centrality.iterations,
                );
            eprintln!(
                "Computed centrality for {} entities",
                centrality.len()
            );
            g.centrality = centrality;
        }

        // Cross-reference logic and concept clusters
        if !g.logic_clusters.is_empty() {
            let concept_assignments: HashMap<u64, usize> = g
                .entities
                .values()
                .filter_map(|ent| {
                    ent.concept_tags
                        .iter()
                        .find_map(|&cid| {
                            g.concepts.get(&cid)?.cluster_id
                        })
                        .map(|cid| (ent.id, cid))
                })
                .collect();
            g.logic_concept_overlaps =
                crate::logic::cross_reference(
                    &g.logic_clusters,
                    &concept_assignments,
                    0.1,
                );
            if !g.logic_concept_overlaps.is_empty() {
                eprintln!(
                    "Cross-references: {} logic-concept cluster \
                     overlaps",
                    g.logic_concept_overlaps.len()
                );
            }
        }

        self.graph = Some(g);
        Ok(self)
    }

    /// Save the graph and parse results to the SQLite cache.
    pub fn cache(&mut self) -> Result<&mut Self> {
        if self.early_result.is_some() {
            return Ok(self);
        }

        if let (Some(ref c), Some(ref g)) =
            (&self.cache, &self.graph)
        {
            let lang_name_refs: Vec<&str> =
                self.lang_names.iter().map(|s| s.as_str()).collect();
            let per_lang_refs: Vec<(&str, &[types::ParseResult])> =
                self.per_lang_results
                    .iter()
                    .map(|(n, r)| (n.as_str(), r.as_slice()))
                    .collect();
            if let Err(e) =
                c.save_multi(g, &lang_name_refs, &per_lang_refs)
            {
                eprintln!("Warning: failed to cache index: {e}");
            } else {
                eprintln!("Cached index to .ontomics/index.db");
            }
        }
        Ok(self)
    }

    /// Finalize the builder and return the `BuildResult`.
    pub fn build(&mut self) -> Result<BuildResult> {
        if let Some(result) = self.early_result.take() {
            return Ok(result);
        }

        Ok(BuildResult {
            graph: self.graph.take()
                .expect("embed() must run before build()"),
            needs_embedding: None,
        })
    }
}

/// Build a map of entity_id -> call_sites for cluster labeling.
pub fn build_entity_call_sites(
    graph: &graph::ConceptGraph,
) -> HashMap<u64, Vec<&types::CallSite>> {
    let mut map: HashMap<u64, Vec<&types::CallSite>> =
        HashMap::new();
    for cs in &graph.call_sites {
        if let Some(scope) = &cs.caller_scope {
            if let Some(entity) = graph.entities.values().find(|e| {
                e.name == *scope && e.file == cs.file
            }) {
                map.entry(entity.id).or_default().push(cs);
            }
        }
    }
    map
}

/// Load domain packs from paths in config, resolving relative
/// to repo root.
pub fn load_domain_packs(
    repo: &Path,
    config: &config::Config,
) -> Vec<types::DomainPack> {
    let mut packs = Vec::new();
    for pack_path_str in &config.domain_packs {
        let pack_path =
            if std::path::Path::new(pack_path_str).is_absolute() {
                PathBuf::from(pack_path_str)
            } else {
                repo.join(pack_path_str)
            };
        match domain_pack::load_domain_pack(&pack_path) {
            Ok(pack) => {
                eprintln!(
                    "Loaded domain pack: {}",
                    pack_path.display()
                );
                packs.push(pack);
            }
            Err(e) => {
                eprintln!(
                    "Warning: failed to load domain pack '{}': {e}",
                    pack_path.display()
                );
            }
        }
    }
    packs
}

/// Expand `~` prefix and return Some(path) if non-empty, None
/// otherwise.
pub fn resolve_cache_dir(raw: &str) -> Option<PathBuf> {
    if raw.is_empty() {
        return None;
    }
    if let Some(rest) = raw.strip_prefix("~/") {
        if let Some(home) = std::env::var_os("HOME") {
            return Some(PathBuf::from(home).join(rest));
        }
    }
    Some(PathBuf::from(raw))
}

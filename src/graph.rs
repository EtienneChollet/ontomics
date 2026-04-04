use crate::embeddings::EmbeddingIndex;
use crate::logic::LogicIndex;
use crate::pseudocode::format_pseudocode;
use crate::tokenizer::{find_abbreviation, split_identifier};
use crate::config::HealthConfig;
use crate::types::{
    AnalysisResult, CallSite, CentralityScore, ClassInfo, CompactContext,
    Concept, ConceptMap, ConceptQueryResult, ConceptTrace, Convention,
    DescribeFileResult, DescribeSymbolResult, Entity, EntitySummary,
    FileSymbol, FileNestingTree, InconsistencyPair, LocateConceptResult,
    LogicCluster, LogicClusterSummary, LogicDescription, MethodSummary,
    ModuleMapEntry, NameSuggestion, NamingCheckResult, PatternKind,
    Pseudocode, QueryConceptParams, RelatedConcept, Relationship,
    RelationshipKind, SessionBriefing, Signature, SimilarLogicResult,
    Subconcept, SymbolKind, TraceEdge, TraceNode, TraceRole, TypeFlow,
    TypeFlowResult, TypeFrequency, Verdict, VocabularyHealth,
};
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

/// Find the longest common directory prefix across a set of file paths.
fn longest_common_directory(paths: &[&std::path::Path]) -> PathBuf {
    if paths.is_empty() {
        return PathBuf::new();
    }
    let first: Vec<_> = paths[0].components().collect();
    let mut prefix_len = first.len();
    for path in &paths[1..] {
        let comps: Vec<_> = path.components().collect();
        prefix_len = prefix_len.min(comps.len());
        for i in 0..prefix_len {
            if first[i] != comps[i] {
                prefix_len = i;
                break;
            }
        }
    }
    // Walk back to the last directory component (strip filename from prefix)
    let prefix: PathBuf = first[..prefix_len].iter().collect();
    if prefix.extension().is_some() || paths.iter().all(|p| *p == prefix) {
        prefix.parent().unwrap_or(&prefix).to_path_buf()
    } else {
        prefix
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

/// Normalize a type annotation for type-flow tracking.
/// Strips `Optional[...]` to its inner type and `Union[..., None]`
/// to its first non-None variant.
fn normalize_type_annotation(ann: &str) -> String {
    let trimmed = ann.trim();

    // Optional[X] -> X
    if let Some(inner) = trimmed
        .strip_prefix("Optional[")
        .and_then(|s| s.strip_suffix(']'))
    {
        return normalize_type_annotation(inner);
    }

    // Union[X, None] or Union[X, Y, ...] -> first non-None type
    if let Some(inner) = trimmed
        .strip_prefix("Union[")
        .and_then(|s| s.strip_suffix(']'))
    {
        // Simple split on top-level commas (handles nested brackets)
        let parts = split_top_level_commas(inner);
        for part in &parts {
            let p = part.trim();
            if !p.eq_ignore_ascii_case("none") {
                return normalize_type_annotation(p);
            }
        }
    }

    trimmed.to_string()
}

/// Split a string by commas, but only at the top level
/// (not inside brackets).
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

pub struct ConceptGraph {
    pub concepts: HashMap<u64, Concept>,
    pub relationships: Vec<Relationship>,
    pub conventions: Vec<Convention>,
    pub embeddings: EmbeddingIndex,
    pub signatures: Vec<Signature>,
    pub classes: Vec<ClassInfo>,
    pub call_sites: Vec<CallSite>,
    pub entities: HashMap<u64, Entity>,
    /// Mean embedding per cluster label, computed after clustering.
    pub cluster_centroids: HashMap<usize, Vec<f32>>,
    /// L4: Pseudocode for entities, keyed by entity ID.
    pub pseudocode: HashMap<u64, Pseudocode>,
    /// L4: Logic embedding index for behavioral similarity search.
    pub logic_index: LogicIndex,
    /// L4: Clusters of entities with similar behavioral patterns.
    pub logic_clusters: Vec<LogicCluster>,
    /// L4: PageRank centrality scores, keyed by entity ID.
    pub centrality: HashMap<u64, CentralityScore>,
    /// L4: Jaccard overlaps between logic clusters and concept clusters.
    /// Each tuple is (logic_cluster_id, concept_cluster_id, jaccard_score).
    pub logic_concept_overlaps: Vec<(usize, usize, f32)>,
    pub nesting_trees: Vec<FileNestingTree>,
}

impl ConceptGraph {
    /// Compute domain density score: sum of concept occurrence counts / subtoken count.
    fn domain_density(&self, entity: &Entity) -> f64 {
        let subtokens = split_identifier(&entity.name).len().max(1);
        let occ_sum: usize = entity
            .concept_tags
            .iter()
            .filter_map(|id| self.concepts.get(id))
            .map(|c| c.occurrences.len())
            .sum();
        occ_sum as f64 / subtokens as f64
    }

    /// Create an empty graph (no concepts, relationships, or entities).
    /// Used as placeholder during deferred startup.
    pub fn empty() -> Self {
        Self {
            concepts: HashMap::new(),
            relationships: Vec::new(),
            conventions: Vec::new(),
            embeddings: EmbeddingIndex::empty(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            entities: HashMap::new(),
            cluster_centroids: HashMap::new(),
            pseudocode: HashMap::new(),
            logic_index: LogicIndex::empty(),
            logic_clusters: Vec::new(),
            centrality: HashMap::new(),
            logic_concept_overlaps: Vec::new(),
            nesting_trees: Vec::new(),
        }
    }

    /// Build graph from analysis results + embeddings + entities.
    pub fn build(
        analysis: AnalysisResult,
        embeddings: EmbeddingIndex,
    ) -> Result<Self> {
        Self::build_with_entities(analysis, embeddings, Vec::new(), Vec::new())
    }

    /// Build graph with pre-built entities and their relationships.
    pub fn build_with_entities(
        analysis: AnalysisResult,
        embeddings: EmbeddingIndex,
        entities: Vec<Entity>,
        entity_relationships: Vec<Relationship>,
    ) -> Result<Self> {
        let concepts: HashMap<u64, Concept> = analysis
            .concepts
            .into_iter()
            .map(|c| (c.id, c))
            .collect();

        let mut relationships: Vec<Relationship> = analysis
            .co_occurrence_matrix
            .into_iter()
            .map(|((src, tgt), weight)| Relationship {
                source: src,
                target: tgt,
                kind: RelationshipKind::CoOccurs,
                weight,
            })
            .collect();

        relationships.extend(entity_relationships);

        let entity_map: HashMap<u64, Entity> =
            entities.into_iter().map(|e| (e.id, e)).collect();

        Ok(Self {
            concepts,
            relationships,
            conventions: analysis.conventions,
            embeddings,
            signatures: analysis.signatures,
            classes: analysis.classes,
            call_sites: analysis.call_sites,
            entities: entity_map,
            cluster_centroids: HashMap::new(),
            pseudocode: HashMap::new(),
            logic_index: LogicIndex::empty(),
            logic_clusters: Vec::new(),
            centrality: HashMap::new(),
            logic_concept_overlaps: Vec::new(),
            nesting_trees: analysis.nesting_trees,
        })
    }

    /// Look up a concept by name (exact or fuzzy match via embeddings).
    pub fn query_concept(
        &self,
        term: &str,
        params: &QueryConceptParams,
    ) -> Option<ConceptQueryResult> {
        let term_lower = term.to_lowercase();
        let subtokens = split_identifier(term);

        // 1. Exact match on canonical name
        let matched = self
            .concepts
            .values()
            .find(|c| c.canonical == term_lower)
            .or_else(|| {
                // Match any concept whose canonical equals one of the
                // query's subtokens
                subtokens.iter().find_map(|st| {
                    self.concepts.values().find(|c| c.canonical == *st)
                })
            })
            .or_else(|| {
                // 2. Match by occurrence identifier
                self.concepts.values().find(|c| {
                    c.occurrences
                        .iter()
                        .any(|o| o.identifier.to_lowercase() == term_lower)
                })
            });

        // 3. Embedding-based fuzzy search (if no exact/occurrence match)
        let matched = matched.or_else(|| {
            // Try to find a concept whose embedding is close to the term.
            // We need an embedding for the query — look for a concept whose
            // canonical is a subtoken of the query as a proxy.
            let query_vec = subtokens.iter().find_map(|st| {
                self.concepts.values().find_map(|c| {
                    if c.canonical == *st {
                        self.embeddings.get_vector(c.id)
                    } else {
                        None
                    }
                })
            });
            if let Some(qv) = query_vec {
                let similar = self.embeddings.find_similar(qv, 1);
                similar
                    .first()
                    .and_then(|(id, score)| {
                        if *score > 0.5 {
                            self.concepts.get(id)
                        } else {
                            None
                        }
                    })
            } else {
                None
            }
        });

        let concept = matched?;

        // Collect variants: all unique identifier names from this concept's
        // occurrences, PLUS identifiers from other concepts that contain the
        // search term in their identifiers.
        let mut variants: Vec<String> = concept
            .occurrences
            .iter()
            .map(|o| o.identifier.clone())
            .collect();

        // Also gather identifiers from ALL concepts that contain the term
        for other in self.concepts.values() {
            if other.id == concept.id {
                continue;
            }
            for occ in &other.occurrences {
                let id_lower = occ.identifier.to_lowercase();
                if id_lower.contains(&term_lower) {
                    variants.push(occ.identifier.clone());
                }
            }
        }

        // Check for abbreviation relationships: find concepts whose
        // canonical is an abbreviation of the search term (or vice versa)
        let term_as_slice = [term_lower.clone()];
        for other in self.concepts.values() {
            if other.id == concept.id {
                continue;
            }
            let canon_as_slice = [other.canonical.clone()];
            // Is other's canonical an abbreviation of the search term?
            let is_abbrev = find_abbreviation(
                &other.canonical,
                &term_as_slice,
            )
            .is_some();
            // Is the search term an abbreviation of other's canonical?
            let is_expansion =
                find_abbreviation(&term_lower, &canon_as_slice)
                    .is_some();
            if is_abbrev || is_expansion {
                for occ in &other.occurrences {
                    variants.push(occ.identifier.clone());
                }
            }
        }

        variants.sort();
        variants.dedup();
        variants.truncate(params.max_variants);

        // Related concepts via relationships (lightweight summaries)
        let mut related: Vec<RelatedConcept> = self
            .relationships
            .iter()
            .filter_map(|rel| {
                let other_id = if rel.source == concept.id {
                    Some(rel.target)
                } else if rel.target == concept.id {
                    Some(rel.source)
                } else {
                    None
                }?;
                let other = self.concepts.get(&other_id)?;
                Some(RelatedConcept {
                    canonical: other.canonical.clone(),
                    kind: rel.kind.clone(),
                    weight: rel.weight,
                    occurrences: other.occurrences.len(),
                })
            })
            .collect();
        // Contrastive first, then by weight descending
        related.sort_by(|a, b| {
            let a_contrastive =
                a.kind == RelationshipKind::Contrastive;
            let b_contrastive =
                b.kind == RelationshipKind::Contrastive;
            b_contrastive
                .cmp(&a_contrastive)
                .then(b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal))
        });
        related.truncate(params.max_related);

        // Conventions that match any of this concept's identifiers
        let concept_identifiers: Vec<String> = concept
            .occurrences
            .iter()
            .map(|o| o.identifier.clone())
            .collect();
        let conventions: Vec<Convention> = self
            .conventions
            .iter()
            .filter(|conv| {
                conv.examples
                    .iter()
                    .any(|ex| concept_identifiers.contains(ex))
            })
            .cloned()
            .collect();

        let top_occurrences: Vec<_> = concept
            .occurrences
            .iter()
            .take(params.max_occurrences)
            .cloned()
            .collect();

        // L2: filter signatures matching this concept
        let mut matching_signatures: Vec<Signature> = self
            .signatures
            .iter()
            .filter(|sig| {
                concept_identifiers.contains(&sig.name)
                    || concept
                        .subtokens
                        .iter()
                        .any(|st| sig.name.to_lowercase().contains(st.as_str()))
            })
            .cloned()
            .collect();
        matching_signatures.truncate(params.max_signatures);

        // L2: filter classes matching this concept
        let matching_classes: Vec<ClassInfo> = self
            .classes
            .iter()
            .filter(|cls| {
                concept_identifiers.contains(&cls.name)
                    || cls.methods.iter().any(|m| {
                        concept_identifiers.iter().any(|id| id == m)
                    })
                    || concept.subtokens.iter().any(|st| {
                        cls.name.to_lowercase().contains(st.as_str())
                    })
            })
            .cloned()
            .collect();

        // L2: build call graph pairs involving matching signatures
        let sig_names: Vec<&str> =
            matching_signatures.iter().map(|s| s.name.as_str()).collect();
        let call_graph: Vec<(String, String)> = self
            .call_sites
            .iter()
            .filter(|cs| {
                sig_names.iter().any(|sn| {
                    cs.callee.contains(sn)
                        || cs.caller_scope.as_deref() == Some(sn)
                        || cs
                            .caller_scope
                            .as_deref()
                            .is_some_and(|s| s.ends_with(&format!(".{sn}")))
                })
            })
            .map(|cs| {
                (
                    cs.caller_scope.clone().unwrap_or_default(),
                    cs.callee.clone(),
                )
            })
            .collect();

        // Collect entities that instantiate this concept
        let entities: Vec<_> = self
            .entities
            .values()
            .filter(|e| e.concept_tags.contains(&concept.id))
            .take(params.max_entities)
            .map(|e| e.summary())
            .collect();

        let mut result = ConceptQueryResult {
            concept: concept.clone(),
            variants,
            related,
            conventions,
            top_occurrences,
            signatures: matching_signatures,
            classes: matching_classes,
            call_graph,
            entities,
        };
        Self::compact_query_result(&mut result);
        Some(result)
    }

    /// Budget in bytes for MCP tool output (~10K tokens).
    const OUTPUT_BUDGET_BYTES: usize = 40_000;

    /// Strip internal-only data and progressively compact a query result
    /// so that the serialized output stays within the token budget.
    fn compact_query_result(r: &mut ConceptQueryResult) {
        // Strip internal-only fields from concept
        r.concept.occurrences.clear();
        r.concept.embedding = None;
        r.concept.entity_types.clear();
        r.concept.subtokens.clear();
        for sc in &mut r.concept.subconcepts {
            sc.embedding = None;
            sc.identifiers.clear();
        }

        if Self::estimated_json_len(r) <= Self::OUTPUT_BUDGET_BYTES {
            return;
        }

        // Level 1: trim unbounded / low-priority fields
        r.call_graph.truncate(10);
        r.classes.truncate(3);
        r.variants.truncate(10);
        for sc in &mut r.concept.subconcepts {
            sc.occurrences.truncate(3);
        }

        if Self::estimated_json_len(r) <= Self::OUTPUT_BUDGET_BYTES {
            return;
        }

        // Level 2: aggressive — keep only the essentials
        r.call_graph.clear();
        r.classes.truncate(1);
        r.signatures.truncate(2);
        r.variants.truncate(5);
        r.concept.subconcepts.clear();
    }

    fn estimated_json_len(r: &ConceptQueryResult) -> usize {
        serde_json::to_string(r).map(|s| s.len()).unwrap_or(0)
    }

    /// Check an identifier against project conventions.
    pub fn check_naming(&self, identifier: &str) -> NamingCheckResult {
        let subtokens = split_identifier(identifier);
        let id_lower = identifier.to_lowercase();

        // Collect all identifiers and their frequencies across the corpus
        let mut id_freq: HashMap<String, usize> = HashMap::new();
        for concept in self.concepts.values() {
            for occ in &concept.occurrences {
                *id_freq
                    .entry(occ.identifier.to_lowercase())
                    .or_insert(0) += 1;
            }
        }

        // 1. Check if a higher-frequency variant exists in the corpus.
        // For each subtoken, look for concepts whose canonical matches,
        // then check if a different identifier form is more popular.
        let input_freq = id_freq.get(&id_lower).copied().unwrap_or(0);

        // Build a set of "related" identifiers: those sharing subtokens
        // with the input. Group by the stem concept.
        let mut related_ids: Vec<(String, usize)> = Vec::new();
        for concept in self.concepts.values() {
            // Does this concept share any subtoken with the input?
            let shares_subtoken = subtokens
                .contains(&concept.canonical)
                || concept.subtokens.iter().any(|cst| {
                    subtokens.iter().any(|st| {
                        st.contains(cst.as_str())
                            || cst.contains(st.as_str())
                    })
                });

            if !shares_subtoken {
                continue;
            }

            // Collect unique identifiers from this concept
            let mut seen = HashMap::new();
            for occ in &concept.occurrences {
                *seen
                    .entry(occ.identifier.to_lowercase())
                    .or_insert(0usize) += 1;
            }
            for (name, count) in seen {
                if name != id_lower {
                    related_ids.push((name, count));
                }
            }
        }

        // Also check canonical names directly as identifiers
        // (e.g., "ndim" is both a concept canonical and an identifier)
        for concept in self.concepts.values() {
            let canon = &concept.canonical;
            // Check if any subtoken of the input is a substring of the
            // canonical, or vice versa
            let is_related = subtokens.iter().any(|st| {
                canon.contains(st.as_str())
                    || st.contains(canon.as_str())
            });
            if !is_related {
                continue;
            }
            let canon_freq =
                id_freq.get(canon.as_str()).copied().unwrap_or(0);
            if canon_freq > 0 && *canon != id_lower {
                related_ids.push((canon.clone(), canon_freq));
            }
        }

        // 1b. Prefix expansion: if a subtoken is a prefix of a concept
        // canonical, include it. Handles short inputs ("trans" → "transform")
        // where embeddings match surface form instead of semantics.
        for concept in self.concepts.values() {
            let canon = &concept.canonical;
            let is_prefix_match = subtokens.iter().any(|st| {
                st.len() >= 2
                    && canon.len() > st.len()
                    && canon.starts_with(st.as_str())
            });
            if is_prefix_match && *canon != id_lower {
                related_ids.push((
                    canon.clone(),
                    concept.occurrences.len(),
                ));
            }
        }

        // 1c. Embedding-based similarity: find concepts semantically
        // close to the input, even if they share no subtokens.
        // Use concept-level occurrence count (not per-identifier) so
        // that a broad concept like "transform" (17 total) beats a
        // narrow substring match like "transformer" (11).
        if let Some(query_vec) = self.embeddings.embed_text(&id_lower) {
            for (cid, score) in self.embeddings.find_similar(&query_vec, 5)
            {
                if score < 0.5 {
                    continue;
                }
                if let Some(concept) = self.concepts.get(&cid) {
                    let canon = &concept.canonical;
                    if *canon != id_lower {
                        let concept_freq = concept.occurrences.len();
                        related_ids.push((
                            canon.clone(),
                            concept_freq,
                        ));
                    }
                }
            }
        }

        // Deduplicate
        related_ids.sort_by(|a, b| b.1.cmp(&a.1));
        related_ids.dedup_by(|a, b| a.0 == b.0);

        // Build similar_identifiers from related_ids (all signals merged)
        let similar_from_related = |skip: Option<&str>| -> Vec<(String, usize)> {
            related_ids
                .iter()
                .filter(|(name, _)| {
                    skip.is_none_or(|s| name != s)
                })
                .take(10)
                .cloned()
                .collect()
        };

        // If a related identifier has strictly higher frequency, suggest it.
        // Only flag Inconsistent when the input was observed in the corpus.
        // Zero-occurrence identifiers are Unknown (not seen), not Inconsistent
        // (seen but less common than an alternative).
        if let Some((best_name, best_freq)) = related_ids.first() {
            if input_freq > 0 && *best_freq > input_freq {
                let total =
                    (input_freq + *best_freq) as f32;
                let ratio = 1.0
                    - (input_freq as f32
                        / (*best_freq).max(1) as f32);
                let sample_scale =
                    (total.max(1.0).log2() / 4.0).min(1.0);
                let confidence =
                    (ratio * sample_scale).clamp(0.0, 1.0);
                return NamingCheckResult {
                    input: identifier.to_string(),
                    subtokens,
                    verdict: Verdict::Inconsistent,
                    reason: format!(
                        "'{}' appears {} times vs '{}' \
                         appears {} times",
                        best_name,
                        best_freq,
                        id_lower,
                        input_freq,
                    ),
                    suggestion: Some(best_name.clone()),
                    matching_convention: None,
                    similar_identifiers: similar_from_related(
                        Some(best_name.as_str()),
                    ),
                    confidence,
                };
            }
        }

        // 2. Check prefix conventions — if the identifier uses a
        // count prefix that conflicts with the project's count prefix.
        // Pick the highest-frequency count convention as canonical.
        let known_count_prefixes = ["n_", "nb_", "num_"];
        let best_count_conv = self
            .conventions
            .iter()
            .filter(|c| {
                matches!(&c.pattern, PatternKind::Prefix(p)
                    if known_count_prefixes.contains(&p.as_str()))
            })
            .max_by_key(|c| c.frequency);
        if let Some(conv) = best_count_conv {
            if let PatternKind::Prefix(conv_prefix) = &conv.pattern
            {
                for &alt_prefix in &known_count_prefixes {
                    if alt_prefix == conv_prefix.as_str() {
                        continue;
                    }
                    if !id_lower.starts_with(alt_prefix) {
                        continue;
                    }
                    let suffix = &id_lower[alt_prefix.len()..];
                    let suggested =
                        format!("{}{}", conv_prefix, suffix);

                    return NamingCheckResult {
                        input: identifier.to_string(),
                        subtokens,
                        verdict: Verdict::Inconsistent,
                        reason: format!(
                            "project uses '{}' prefix for {} \
                             (not '{}')",
                            conv_prefix,
                            conv.semantic_role,
                            alt_prefix
                        ),
                        suggestion: Some(suggested),
                        matching_convention: Some(
                            conv.clone(),
                        ),
                        similar_identifiers:
                            find_similar_identifiers(
                                identifier, &id_freq,
                            ),
                        confidence: 0.8,
                    };
                }
            }
        }

        // 3. Check if identifier follows a known convention (consistent)
        for conv in &self.conventions {
            let matches = match &conv.pattern {
                PatternKind::Prefix(p) => id_lower.starts_with(p.as_str()),
                PatternKind::Suffix(s) => id_lower.ends_with(s.as_str()),
                PatternKind::Conversion(c) => id_lower.contains(c.as_str()),
                PatternKind::Compound(c) => id_lower.contains(c.as_str()),
            };
            if matches {
                return NamingCheckResult {
                    input: identifier.to_string(),
                    subtokens,
                    verdict: Verdict::Consistent,
                    reason: format!(
                        "follows '{}' convention for {}",
                        match &conv.pattern {
                            PatternKind::Prefix(p) => p.as_str(),
                            PatternKind::Suffix(s) => s,
                            PatternKind::Conversion(c) => c,
                            PatternKind::Compound(c) => c,
                        },
                        conv.semantic_role,
                    ),
                    suggestion: None,
                    matching_convention: Some(conv.clone()),
                    similar_identifiers: similar_from_related(None),
                    confidence: 0.8,
                };
            }
        }

        // 4. If identifier exists in corpus at reasonable frequency, it's OK
        if input_freq > 0 {
            return NamingCheckResult {
                input: identifier.to_string(),
                subtokens,
                verdict: Verdict::Consistent,
                reason: format!(
                    "found {} occurrences in corpus",
                    input_freq
                ),
                suggestion: None,
                matching_convention: None,
                similar_identifiers: find_similar_identifiers(
                    identifier,
                    &id_freq,
                ),
                confidence: 0.6,
            };
        }

        // 5. Unknown — identifier not found, no convention match
        NamingCheckResult {
            input: identifier.to_string(),
            subtokens,
            verdict: Verdict::Unknown,
            reason: "identifier not found in corpus and \
                     matches no convention"
                .to_string(),
            suggestion: None,
            matching_convention: None,
            similar_identifiers: similar_from_related(None),
            confidence: 0.0,
        }
    }

    /// Suggest an identifier name given a natural language description.
    ///
    /// Primary path: embed the description, find semantically similar
    /// concepts, then suggest their canonical forms (the vocabulary this
    /// codebase uses).  Fallback (no embeddings yet): keyword matching
    /// against concept canonicals and AbbreviationOf edges.
    pub fn suggest_name(
        &self,
        description: &str,
    ) -> Vec<NameSuggestion> {
        let words: Vec<String> = description
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        // Phase 1: Find relevant concepts
        let mut concept_scores: Vec<(u64, f32, String)> =
            if let Some(query_vec) =
                self.embeddings.embed_text(description)
            {
                self.embeddings
                    .find_similar(&query_vec, 10)
                    .into_iter()
                    .filter(|(_, sim)| *sim >= 0.40)
                    .map(|(cid, sim)| {
                        (cid, sim, format!("similarity: {:.2}", sim))
                    })
                    .collect()
            } else {
                self.suggest_name_fallback(&words)
            };

        // Phase 1b: Inject concepts that match query words by exact
        // canonical or prefix (catches "convolution" → concept "conv").
        let mut seen: HashSet<u64> =
            concept_scores.iter().map(|(id, _, _)| *id).collect();
        for word in &words {
            // Skip stopwords and single-char words for prefix matching
            if word.len() < 3 {
                continue;
            }
            for concept in self.concepts.values() {
                if seen.contains(&concept.id) {
                    continue;
                }
                let canon = &concept.canonical;
                // Both sides must be >= 3 chars for prefix matching
                if canon.len() < 3 {
                    continue;
                }
                if *canon == *word {
                    seen.insert(concept.id);
                    concept_scores.push((
                        concept.id,
                        0.85,
                        format!("exact match '{}'", canon),
                    ));
                } else if word.starts_with(canon.as_str())
                    || canon.starts_with(word.as_str())
                {
                    seen.insert(concept.id);
                    concept_scores.push((
                        concept.id,
                        0.80,
                        format!("'{}' ↔ '{}'", canon, word),
                    ));
                }
            }
        }

        // Phase 2: Boost scores with structural signals
        let matched_ids: HashSet<u64> =
            concept_scores.iter().map(|(id, _, _)| *id).collect();
        let mut adjusted: Vec<(u64, f32, String)> = concept_scores
            .into_iter()
            .filter_map(|(cid, raw, reason)| {
                let concept = self.concepts.get(&cid)?;
                let mut score = raw;

                if words.contains(&concept.canonical) {
                    score += 0.10;
                }

                let has_abbrev_link =
                    self.relationships.iter().any(|r| {
                        r.kind == RelationshipKind::AbbreviationOf
                            && (r.source == cid || r.target == cid)
                            && matched_ids.contains(
                                if r.source == cid {
                                    &r.target
                                } else {
                                    &r.source
                                },
                            )
                    });
                if has_abbrev_link {
                    score += 0.05;
                }

                let occ_bonus = (concept.occurrences.len() as f32
                    + 1.0)
                    .ln()
                    * 0.02;
                score = (score + occ_bonus).min(0.99);

                Some((cid, score, reason))
            })
            .collect();
        adjusted.sort_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });

        // Phase 3: Convention matching
        let matched_convention =
            self.find_matching_convention(&words);

        // Phase 4: Generate suggestions
        let mut suggestions: Vec<NameSuggestion> = Vec::new();
        let top: Vec<&(u64, f32, String)> =
            adjusted.iter().take(5).collect();

        // Convention-based suggestions (e.g. nb_ + concept)
        if let Some(conv) = matched_convention {
            for &(cid, adj_score, _) in &top {
                let concept = match self.concepts.get(cid) {
                    Some(c) => c,
                    None => continue,
                };
                match &conv.pattern {
                    PatternKind::Prefix(prefix) => {
                        let name = format!(
                            "{}{}",
                            prefix, concept.canonical
                        );
                        let exists = concept
                            .occurrences
                            .iter()
                            .any(|o| o.identifier == name);
                        let conf = if exists {
                            (adj_score * 0.5 + 0.45).min(0.95)
                        } else {
                            (adj_score * 0.4 + 0.25).min(0.80)
                        };
                        suggestions.push(NameSuggestion {
                            name,
                            confidence: conf,
                            based_on: vec![
                                format!(
                                    "'{}' prefix for {}",
                                    prefix, conv.semantic_role
                                ),
                                format!(
                                    "concept '{}' ({}x)",
                                    concept.canonical,
                                    concept.occurrences.len()
                                ),
                            ],
                        });
                    }
                    PatternKind::Suffix(suffix) => {
                        let name = format!(
                            "{}{}",
                            concept.canonical, suffix
                        );
                        let exists = concept
                            .occurrences
                            .iter()
                            .any(|o| o.identifier == name);
                        let conf = if exists {
                            (adj_score * 0.5 + 0.45).min(0.95)
                        } else {
                            (adj_score * 0.4 + 0.25).min(0.80)
                        };
                        suggestions.push(NameSuggestion {
                            name,
                            confidence: conf,
                            based_on: vec![
                                format!(
                                    "'{}' suffix for {}",
                                    suffix, conv.semantic_role
                                ),
                                format!(
                                    "concept '{}' ({}x)",
                                    concept.canonical,
                                    concept.occurrences.len()
                                ),
                            ],
                        });
                    }
                    PatternKind::Conversion(sep) => {
                        if top.len() >= 2 {
                            let c0 =
                                self.concepts.get(&top[0].0);
                            let c1 =
                                self.concepts.get(&top[1].0);
                            if let (Some(c0), Some(c1)) = (c0, c1)
                            {
                                suggestions.push(NameSuggestion {
                                    name: format!(
                                        "{}{}{}",
                                        c0.canonical,
                                        sep,
                                        c1.canonical
                                    ),
                                    confidence: (adj_score * 0.4
                                        + 0.2)
                                        .min(0.75),
                                    based_on: vec![format!(
                                        "'{}' conversion pattern",
                                        sep
                                    )],
                                });
                            }
                        }
                        break;
                    }
                    PatternKind::Compound(template) => {
                        suggestions.push(NameSuggestion {
                            name: template.replace(
                                "[]",
                                &concept.canonical,
                            ),
                            confidence: (adj_score * 0.4 + 0.25)
                                .min(0.80),
                            based_on: vec![format!(
                                "compound pattern '{}'",
                                template
                            )],
                        });
                    }
                }
            }
        }

        // Concept canonical suggestions (the codebase vocabulary)
        for &(cid, adj_score, ref reason) in &top {
            let concept = match self.concepts.get(cid) {
                Some(c) => c,
                None => continue,
            };
            suggestions.push(NameSuggestion {
                name: concept.canonical.clone(),
                confidence: (adj_score * 0.85).min(0.92),
                based_on: vec![
                    format!(
                        "'{}' ({}x in codebase)",
                        concept.canonical,
                        concept.occurrences.len()
                    ),
                    reason.clone(),
                ],
            });
        }

        // Deduplicate by name, keeping highest confidence
        suggestions.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        suggestions.dedup_by(|a, b| {
            if a.name.to_lowercase() == b.name.to_lowercase() {
                b.based_on.append(&mut a.based_on);
                true
            } else {
                false
            }
        });
        suggestions.truncate(5);
        suggestions
    }

    /// Fallback concept matching when embeddings aren't loaded.
    /// Uses exact canonical matches and AbbreviationOf edges.
    fn suggest_name_fallback(
        &self,
        words: &[String],
    ) -> Vec<(u64, f32, String)> {
        let mut results: Vec<(u64, f32, String)> = Vec::new();
        let mut seen: HashSet<u64> = HashSet::new();

        // Exact canonical match
        for concept in self.concepts.values() {
            if words.contains(&concept.canonical)
                && seen.insert(concept.id)
            {
                results.push((
                    concept.id,
                    0.80,
                    format!("exact match '{}'", concept.canonical),
                ));
            }
        }

        // Follow AbbreviationOf edges (source=short, target=long)
        let initial_ids: HashSet<u64> =
            results.iter().map(|(id, _, _)| *id).collect();
        for rel in &self.relationships {
            if rel.kind != RelationshipKind::AbbreviationOf {
                continue;
            }
            let linked = if initial_ids.contains(&rel.source) {
                Some((rel.target, 0.75))
            } else if initial_ids.contains(&rel.target) {
                Some((rel.source, 0.70))
            } else {
                // Check if a word matches an endpoint not in initial
                let src_match = self
                    .concepts
                    .get(&rel.source)
                    .is_some_and(|c| {
                        words.contains(&c.canonical)
                    });
                let tgt_match = self
                    .concepts
                    .get(&rel.target)
                    .is_some_and(|c| {
                        words.contains(&c.canonical)
                    });
                if src_match {
                    Some((rel.target, 0.75))
                } else if tgt_match {
                    Some((rel.source, 0.70))
                } else {
                    None
                }
            };
            if let Some((id, score)) = linked {
                if seen.insert(id) {
                    if let Some(c) = self.concepts.get(&id) {
                        results.push((
                            id,
                            score,
                            format!(
                                "abbreviation of '{}'",
                                c.canonical
                            ),
                        ));
                    }
                }
            }
        }

        // Identifier substring containment (weakest signal)
        for concept in self.concepts.values() {
            if seen.contains(&concept.id) {
                continue;
            }
            if words.iter().any(|w| {
                concept.occurrences.iter().any(|o| {
                    o.identifier.to_lowercase().contains(w.as_str())
                })
            }) && seen.insert(concept.id)
            {
                results.push((
                    concept.id,
                    0.50,
                    format!(
                        "identifier contains '{}'",
                        concept.canonical
                    ),
                ));
            }
        }

        results
    }

    /// Find a convention whose semantic role matches a description word.
    fn find_matching_convention(
        &self,
        words: &[String],
    ) -> Option<&Convention> {
        const ROLE_SYNONYMS: &[(&str, &str)] = &[
            ("count", "count"),
            ("number", "count"),
            ("boolean", "boolean predicate"),
            ("flag", "boolean predicate"),
            ("convert", "conversion"),
            ("conversion", "conversion"),
        ];

        for word in words {
            if let Some(conv) = self
                .conventions
                .iter()
                .find(|c| c.semantic_role == word.as_str())
            {
                return Some(conv);
            }
            for &(synonym, role) in ROLE_SYNONYMS {
                if word == synonym {
                    if let Some(conv) = self
                        .conventions
                        .iter()
                        .find(|c| c.semantic_role == role)
                    {
                        return Some(conv);
                    }
                }
            }
        }
        None
    }

    /// List all detected conventions.
    pub fn list_conventions(&self) -> &[Convention] {
        &self.conventions
    }

    /// Cluster concepts by embedding similarity using agglomerative clustering
    /// with average linkage, then add SimilarTo edges between cluster members.
    /// Replaces the naive pairwise threshold approach that caused transitive chaining.
    pub fn cluster_and_add_similarity_edges(&mut self, threshold: f32) {
        // Clear existing SimilarTo edges, cluster labels, and centroids (idempotent)
        self.relationships
            .retain(|r| r.kind != RelationshipKind::SimilarTo);
        for concept in self.concepts.values_mut() {
            concept.cluster_id = None;
        }
        self.cluster_centroids.clear();

        let ids: Vec<u64> = self.concepts.keys().copied().collect();
        if ids.is_empty() {
            return;
        }

        let distance_threshold = 1.0 - threshold;
        let result = crate::cluster::agglomerative_cluster(
            &ids,
            &self.embeddings,
            distance_threshold,
        );

        // Assign cluster labels to concepts
        for (&concept_id, &cluster_label) in &result.assignments {
            if let Some(concept) = self.concepts.get_mut(&concept_id) {
                concept.cluster_id = Some(cluster_label);
            }
        }

        // Emit SimilarTo edges between all pairs within each cluster
        let mut clusters: HashMap<usize, Vec<u64>> = HashMap::new();
        for (&concept_id, &label) in &result.assignments {
            clusters.entry(label).or_default().push(concept_id);
        }

        for members in clusters.values() {
            for (i, &id_a) in members.iter().enumerate() {
                let vec_a = match self.embeddings.get_vector(id_a) {
                    Some(v) => v.clone(),
                    None => continue,
                };
                for &id_b in &members[i + 1..] {
                    let vec_b = match self.embeddings.get_vector(id_b) {
                        Some(v) => v,
                        None => continue,
                    };
                    let sim = cosine_similarity(&vec_a, vec_b);
                    self.relationships.push(Relationship {
                        source: id_a,
                        target: id_b,
                        kind: RelationshipKind::SimilarTo,
                        weight: sim,
                    });
                }
            }
        }

        // Compute cluster centroids (mean embedding per cluster)
        for (&label, members) in &clusters {
            let vecs: Vec<&Vec<f32>> = members
                .iter()
                .filter_map(|&id| self.embeddings.get_vector(id))
                .collect();
            if vecs.is_empty() {
                continue;
            }
            let dim = vecs[0].len();
            let mut centroid = vec![0.0_f32; dim];
            for v in &vecs {
                for (c, &val) in centroid.iter_mut().zip(v.iter()) {
                    *c += val;
                }
            }
            let n = vecs.len() as f32;
            for c in &mut centroid {
                *c /= n;
            }
            self.cluster_centroids.insert(label, centroid);
        }
    }

    /// Recompute cluster centroids from existing cluster_id assignments.
    /// Use after cache load when cluster_ids are present but centroids
    /// were not serialized.
    pub fn recompute_centroids(&mut self) {
        self.cluster_centroids.clear();
        let mut groups: HashMap<usize, Vec<u64>> = HashMap::new();
        for concept in self.concepts.values() {
            if let Some(label) = concept.cluster_id {
                groups.entry(label).or_default().push(concept.id);
            }
        }
        for (label, members) in &groups {
            let vecs: Vec<&Vec<f32>> = members
                .iter()
                .filter_map(|&id| self.embeddings.get_vector(id))
                .collect();
            if vecs.is_empty() {
                continue;
            }
            let dim = vecs[0].len();
            let mut centroid = vec![0.0_f32; dim];
            for v in &vecs {
                for (c, &val) in centroid.iter_mut().zip(v.iter()) {
                    *c += val;
                }
            }
            let n = vecs.len() as f32;
            for c in &mut centroid {
                *c /= n;
            }
            self.cluster_centroids.insert(*label, centroid);
        }
    }

    /// List all concepts, ordered by frequency (descending).
    pub fn list_concepts(&self) -> Vec<&Concept> {
        let mut concepts: Vec<&Concept> =
            self.concepts.values().collect();
        concepts.sort_by(|a, b| b.occurrences.len().cmp(&a.occurrences.len()));
        concepts.truncate(350);
        concepts
    }

    /// Assign an entity to a cluster via embedding centroid distance.
    /// Averages the entity's concept tag embeddings and finds the nearest
    /// cluster centroid. Returns None if no concept tags have embeddings
    /// or no centroids exist.
    fn entity_cluster(&self, entity: &Entity) -> Option<usize> {
        if self.cluster_centroids.is_empty() {
            return None;
        }
        let vecs: Vec<&Vec<f32>> = entity
            .concept_tags
            .iter()
            .filter_map(|&id| self.embeddings.get_vector(id))
            .collect();
        if vecs.is_empty() {
            return None;
        }
        let dim = vecs[0].len();
        let mut mean = vec![0.0_f32; dim];
        for v in &vecs {
            for (m, &val) in mean.iter_mut().zip(v.iter()) {
                *m += val;
            }
        }
        let n = vecs.len() as f32;
        for m in &mut mean {
            *m /= n;
        }
        self.cluster_centroids
            .iter()
            .map(|(&label, centroid)| {
                (label, cosine_similarity(&mean, centroid))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, _)| label)
    }

    /// Select up to `top_k` entities via cluster-aware round-robin.
    /// Largest cluster first, highest domain-density entity per cluster,
    /// cycling until top_k is filled. Falls back to domain density
    /// sorting when no clusters exist.
    fn select_by_cluster(
        &self,
        candidates: &[&Entity],
        top_k: usize,
    ) -> Vec<crate::types::EntitySummary> {
        let mut clustered: HashMap<usize, Vec<&Entity>> = HashMap::new();
        let mut unclustered: Vec<&Entity> = Vec::new();
        for &entity in candidates {
            match self.entity_cluster(entity) {
                Some(label) => clustered.entry(label).or_default().push(entity),
                None => unclustered.push(entity),
            }
        }

        for entities in clustered.values_mut() {
            entities.sort_by(|a, b| {
                self.domain_density(b)
                    .partial_cmp(&self.domain_density(a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        unclustered.sort_by(|a, b| {
            self.domain_density(b)
                .partial_cmp(&self.domain_density(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut cluster_labels: Vec<usize> = clustered.keys().copied().collect();
        cluster_labels.sort_by(|a, b| {
            clustered[b].len().cmp(&clustered[a].len())
        });

        let mut result: Vec<crate::types::EntitySummary> =
            Vec::with_capacity(top_k);
        let mut cursors: HashMap<usize, usize> = HashMap::new();
        let mut unclustered_cursor = 0;

        loop {
            if result.len() >= top_k {
                break;
            }
            let mut made_progress = false;
            for &label in &cluster_labels {
                if result.len() >= top_k {
                    break;
                }
                let cursor = cursors.entry(label).or_insert(0);
                if let Some(entity) = clustered[&label].get(*cursor) {
                    result.push(entity.summary());
                    *cursor += 1;
                    made_progress = true;
                }
            }
            if result.len() < top_k && unclustered_cursor < unclustered.len() {
                result.push(unclustered[unclustered_cursor].summary());
                unclustered_cursor += 1;
                made_progress = true;
            }
            if !made_progress {
                break;
            }
        }
        result
    }

    /// List entities with optional filtering, ranked by cluster-diverse
    /// round-robin.
    pub fn list_entities(
        &self,
        concept_filter: Option<&str>,
        role_filter: Option<&str>,
        kind_filter: Option<&crate::types::EntityKind>,
        top_k: usize,
    ) -> Vec<crate::types::EntitySummary> {
        let concept_id: Option<u64> = concept_filter.and_then(|term| {
            let term_lower = term.to_lowercase();
            self.concepts
                .values()
                .find(|c| c.canonical == term_lower)
                .map(|c| c.id)
        });

        let matched: Vec<&Entity> = self
            .entities
            .values()
            .filter(|e| {
                if let Some(cid) = concept_id {
                    if !e.concept_tags.contains(&cid) {
                        return false;
                    }
                }
                if let Some(role) = role_filter {
                    let role_lower = role.to_lowercase();
                    if !e.semantic_role.to_lowercase().contains(&role_lower) {
                        return false;
                    }
                }
                if let Some(kind) = kind_filter {
                    if e.kind != *kind {
                        return false;
                    }
                }
                true
            })
            .collect();

        self.select_by_cluster(&matched, top_k)
    }

    /// Describe a symbol (function or class) by name.
    pub fn describe_symbol(
        &self,
        name: &str,
    ) -> Option<DescribeSymbolResult> {
        let name_lower = name.to_lowercase();

        let signature = self
            .signatures
            .iter()
            .find(|s| s.name.to_lowercase() == name_lower)
            .cloned();

        let class_info = self
            .classes
            .iter()
            .find(|c| c.name.to_lowercase() == name_lower)
            .cloned();

        if signature.is_none() && class_info.is_none() {
            return None;
        }

        let kind = if class_info.is_some() {
            SymbolKind::Class
        } else if let Some(ref sig) = signature {
            if sig.scope.as_ref().is_some_and(|s| {
                self.classes.iter().any(|c| s.contains(&c.name))
            }) {
                SymbolKind::Method
            } else {
                SymbolKind::Function
            }
        } else {
            SymbolKind::Function
        };

        let callers: Vec<CallSite> = self
            .call_sites
            .iter()
            .filter(|cs| {
                cs.callee.to_lowercase() == name_lower
                    || cs
                        .callee
                        .to_lowercase()
                        .ends_with(&format!(".{name_lower}"))
            })
            .cloned()
            .collect();

        let callees: Vec<CallSite> = self
            .call_sites
            .iter()
            .filter(|cs| {
                cs.caller_scope.as_ref().is_some_and(|s| {
                    s.to_lowercase() == name_lower
                        || s.to_lowercase()
                            .ends_with(&format!(".{name_lower}"))
                })
            })
            .cloned()
            .collect();

        let subtokens = split_identifier(name);
        let concepts: Vec<String> = self
            .concepts
            .values()
            .filter(|c| subtokens.contains(&c.canonical))
            .map(|c| c.canonical.clone())
            .collect();

        // Entity enrichment — prefer exact case match, then case-insensitive,
        // and prefer matching SymbolKind to avoid class/function confusion.
        let entity = self
            .entities
            .values()
            .find(|e| e.name == name)
            .or_else(|| {
                let kind_match = |e: &&Entity| {
                    matches!(
                        (&kind, &e.kind),
                        (SymbolKind::Class, crate::types::EntityKind::Class)
                            | (SymbolKind::Function, crate::types::EntityKind::Function)
                            | (SymbolKind::Method, crate::types::EntityKind::Method)
                    )
                };
                self.entities
                    .values()
                    .filter(|e| e.name.to_lowercase() == name_lower)
                    .find(|e| kind_match(e))
                    .or_else(|| {
                        self.entities
                            .values()
                            .find(|e| e.name.to_lowercase() == name_lower)
                    })
            });

        let related_entities: Vec<_> = entity
            .map(|e| {
                self.relationships
                    .iter()
                    .filter(|r| {
                        (r.kind == RelationshipKind::InheritsFrom
                            || r.kind == RelationshipKind::Uses)
                            && (r.source == e.id || r.target == e.id)
                    })
                    .filter_map(|r| {
                        let other_id = if r.source == e.id {
                            r.target
                        } else {
                            r.source
                        };
                        self.entities.get(&other_id).map(|e| e.summary())
                    })
                    .collect()
            })
            .unwrap_or_default();

        Some(DescribeSymbolResult {
            name: name.to_string(),
            kind,
            signature,
            class_info,
            callers,
            callees,
            concepts,
            related_entities,
        })
    }

    /// Detect abbreviation relationships between concepts and persist them
    /// as AbbreviationOf edges. For each pair of concepts where one canonical
    /// is shorter, calls find_abbreviation(short, &[long]). Must run BEFORE
    /// add_contrastive_edges (which checks for AbbreviationOf to suppress
    /// false contrastives).
    pub fn add_abbreviation_edges(&mut self) {
        self.relationships
            .retain(|r| r.kind != RelationshipKind::AbbreviationOf);

        let canonicals: Vec<(u64, String)> = self
            .concepts
            .values()
            .map(|c| (c.id, c.canonical.clone()))
            .collect();

        let mut new_edges: Vec<Relationship> = Vec::new();

        for (i, (id_a, canon_a)) in canonicals.iter().enumerate() {
            for (id_b, canon_b) in &canonicals[i + 1..] {
                let (short_id, short, long_id, long) =
                    if canon_a.len() < canon_b.len() {
                        (id_a, canon_a.as_str(), id_b, canon_b.clone())
                    } else if canon_b.len() < canon_a.len() {
                        (id_b, canon_b.as_str(), id_a, canon_a.clone())
                    } else {
                        continue;
                    };

                // Skip very short abbreviations — too noisy
                if short.len() < 3 {
                    continue;
                }

                let candidates = [long];
                if find_abbreviation(short, &candidates).is_some() {
                    new_edges.push(Relationship {
                        source: *short_id,
                        target: *long_id,
                        kind: RelationshipKind::AbbreviationOf,
                        weight: 1.0,
                    });
                }
            }
        }

        self.relationships.extend(new_edges);
    }

    /// Detect contrastive concept pairs and add Contrastive edges.
    /// Must run AFTER cluster_and_add_similarity_edges. Suppresses
    /// conflicting SimilarTo edges.
    pub fn add_contrastive_edges(&mut self) {
        // Clear existing contrastive edges so this is idempotent.
        self.relationships
            .retain(|r| r.kind != RelationshipKind::Contrastive);
        const KNOWN_PAIRS: &[(&str, &str)] = &[
            ("source", "target"),
            ("src", "trg"),
            ("input", "output"),
            ("pred", "true"),
            ("before", "after"),
            ("left", "right"),
            ("fixed", "moving"),
            ("old", "new"),
            ("expected", "actual"),
            ("query", "key"),
        ];

        let concept_ids: HashMap<&str, u64> = self
            .concepts
            .values()
            .map(|c| (c.canonical.as_str(), c.id))
            .collect();

        let mut pair_scores: HashMap<(u64, u64), i32> = HashMap::new();

        // Signal 1: param co-occurrence — concepts appearing as separate
        // params in the same signature
        let mut param_co: HashMap<(u64, u64), usize> = HashMap::new();
        for sig in &self.signatures {
            let mut param_concepts: Vec<u64> = Vec::new();
            for param in &sig.params {
                let subtokens = split_identifier(&param.name);
                for st in &subtokens {
                    if let Some(&cid) = concept_ids.get(st.as_str()) {
                        if !param_concepts.contains(&cid) {
                            param_concepts.push(cid);
                        }
                    }
                }
            }
            for i in 0..param_concepts.len() {
                for j in (i + 1)..param_concepts.len() {
                    let pair = if param_concepts[i] < param_concepts[j] {
                        (param_concepts[i], param_concepts[j])
                    } else {
                        (param_concepts[j], param_concepts[i])
                    };
                    *param_co.entry(pair).or_insert(0) += 1;
                }
            }
        }
        for (pair, count) in &param_co {
            if *count >= 3 {
                *pair_scores.entry(*pair).or_insert(0) += 2;
            }
        }

        // Signal 3: known patterns
        for &(a, b) in KNOWN_PAIRS {
            if let (Some(&id_a), Some(&id_b)) =
                (concept_ids.get(a), concept_ids.get(b))
            {
                let pair = if id_a < id_b {
                    (id_a, id_b)
                } else {
                    (id_b, id_a)
                };
                *pair_scores.entry(pair).or_insert(0) += 3;
            }
        }

        // Add contrastive edges for pairs scoring >= 3
        let mut contrastive_pairs: Vec<(u64, u64)> = Vec::new();
        for (&pair, &score) in &pair_scores {
            if score >= 3 {
                // Check no AbbreviationOf edge between them
                let has_abbrev = self.relationships.iter().any(|r| {
                    r.kind == RelationshipKind::AbbreviationOf
                        && ((r.source == pair.0 && r.target == pair.1)
                            || (r.source == pair.1 && r.target == pair.0))
                });
                if !has_abbrev {
                    contrastive_pairs.push(pair);
                }
            }
        }

        // Suppress SimilarTo edges and add Contrastive edges
        for &(a, b) in &contrastive_pairs {
            self.relationships.retain(|r| {
                !(r.kind == RelationshipKind::SimilarTo
                    && ((r.source == a && r.target == b)
                        || (r.source == b && r.target == a)))
            });
            self.relationships.push(Relationship {
                source: a,
                target: b,
                kind: RelationshipKind::Contrastive,
                weight: 1.0,
            });
        }
    }

    /// Detect subconcepts for polysemous high-frequency concepts.
    /// Must run AFTER embeddings and co-occurrence are built.
    pub fn detect_subconcepts(&mut self) {
        let concept_ids: Vec<u64> =
            self.concepts.keys().copied().collect();

        for cid in concept_ids {
            let concept = match self.concepts.get(&cid) {
                Some(c) => c.clone(),
                None => continue,
            };
            if concept.occurrences.len() < 6 {
                continue;
            }

            let mut unique_ids: Vec<String> = concept
                .occurrences
                .iter()
                .map(|o| o.identifier.clone())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect();
            unique_ids.sort();

            if unique_ids.len() < 4 {
                continue;
            }

            // Build affinity matrix using scope co-occurrence +
            // embedding similarity
            let n = unique_ids.len();
            let mut affinity = vec![vec![0.0f32; n]; n];

            // Batch-embed all unique identifiers up front
            let id_embeddings: Vec<Option<Vec<f32>>> =
                if let Some(batch) = self
                    .embeddings
                    .embed_texts_batch(&unique_ids)
                {
                    batch.into_iter().map(Some).collect()
                } else {
                    vec![None; n]
                };

            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                for j in (i + 1)..n {
                    let mut aff = 0.0f32;

                    // Signal 1: scope co-occurrence (weight 0.3)
                    let scopes_i: HashSet<&str> = concept
                        .occurrences
                        .iter()
                        .filter(|o| o.identifier == unique_ids[i])
                        .filter_map(|o| {
                            o.file.to_str()
                        })
                        .collect();
                    let scopes_j: HashSet<&str> = concept
                        .occurrences
                        .iter()
                        .filter(|o| o.identifier == unique_ids[j])
                        .filter_map(|o| {
                            o.file.to_str()
                        })
                        .collect();
                    let shared = scopes_i.intersection(&scopes_j).count();
                    if shared > 0 {
                        aff += 0.3
                            * (shared.min(3) as f32 / 3.0);
                    }

                    // Signal 2: embedding similarity (weight 0.3)
                    if let (Some(ei), Some(ej)) =
                        (&id_embeddings[i], &id_embeddings[j])
                    {
                        let sim = cosine_similarity(ei, ej);
                        aff += 0.3 * sim.max(0.0);
                    }

                    // Signal 3: structural context (weight 0.25)
                    let sig_i = self.signatures.iter().find(|s| {
                        s.name == unique_ids[i]
                    });
                    let sig_j = self.signatures.iter().find(|s| {
                        s.name == unique_ids[j]
                    });
                    if let (Some(si), Some(sj)) = (sig_i, sig_j) {
                        let params_i: HashSet<String> = si
                            .params
                            .iter()
                            .flat_map(|p| split_identifier(&p.name))
                            .collect();
                        let params_j: HashSet<String> = sj
                            .params
                            .iter()
                            .flat_map(|p| split_identifier(&p.name))
                            .collect();
                        let union_len =
                            params_i.union(&params_j).count();
                        if union_len > 0 {
                            let intersection_len = params_i
                                .intersection(&params_j)
                                .count();
                            let jaccard = intersection_len as f32
                                / union_len as f32;
                            aff += 0.25 * jaccard;
                        }
                    }

                    affinity[i][j] = aff;
                    affinity[j][i] = aff;
                }
            }

            // Connected components with threshold
            let threshold = 0.25f32;
            let mut parent: Vec<usize> = (0..n).collect();

            fn find(parent: &mut [usize], x: usize) -> usize {
                if parent[x] != x {
                    parent[x] = find(parent, parent[x]);
                }
                parent[x]
            }

            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                for j in (i + 1)..n {
                    if affinity[i][j] >= threshold {
                        let pi = find(&mut parent, i);
                        let pj = find(&mut parent, j);
                        if pi != pj {
                            parent[pi] = pj;
                        }
                    }
                }
            }

            // Group into clusters
            let mut clusters: HashMap<usize, Vec<usize>> =
                HashMap::new();
            for i in 0..n {
                let root = find(&mut parent, i);
                clusters.entry(root).or_default().push(i);
            }

            // Filter clusters with >= 2 members
            let valid_clusters: Vec<Vec<usize>> = clusters
                .into_values()
                .filter(|c| c.len() >= 2)
                .collect();

            if valid_clusters.len() < 2 {
                continue;
            }

            // Determine qualifiers and build subconcepts
            let mut subconcepts = Vec::new();
            let all_cluster_subtokens: Vec<HashSet<String>> =
                valid_clusters
                    .iter()
                    .map(|cluster| {
                        cluster
                            .iter()
                            .flat_map(|&idx| {
                                split_identifier(&unique_ids[idx])
                            })
                            .collect()
                    })
                    .collect();

            for (ci, cluster) in valid_clusters.iter().enumerate() {
                let my_subtokens = &all_cluster_subtokens[ci];
                let other_subtokens: HashSet<String> =
                    all_cluster_subtokens
                        .iter()
                        .enumerate()
                        .filter(|&(i, _)| i != ci)
                        .flat_map(|(_, s)| s.iter().cloned())
                        .collect();

                let mut distinguishing: Vec<String> = my_subtokens
                    .iter()
                    .filter(|st| {
                        !other_subtokens.contains(*st)
                            && **st != concept.canonical
                    })
                    .cloned()
                    .collect();
                distinguishing.sort();

                let qualifier = distinguishing
                    .first()
                    .cloned()
                    .unwrap_or_else(|| {
                        my_subtokens
                            .iter()
                            .find(|st| **st != concept.canonical)
                            .cloned()
                            .unwrap_or_else(|| {
                                format!("group_{ci}")
                            })
                    });

                let cluster_ids: Vec<String> = cluster
                    .iter()
                    .map(|&idx| unique_ids[idx].clone())
                    .collect();
                let cluster_occurrences = concept
                    .occurrences
                    .iter()
                    .filter(|o| cluster_ids.contains(&o.identifier))
                    .cloned()
                    .collect();

                subconcepts.push(Subconcept {
                    qualifier: qualifier.clone(),
                    canonical: format!(
                        "{}.{}",
                        concept.canonical, qualifier
                    ),
                    occurrences: cluster_occurrences,
                    identifiers: cluster_ids,
                    embedding: None,
                });
            }

            if let Some(c) = self.concepts.get_mut(&cid) {
                c.subconcepts = subconcepts;
            }
        }
    }

    /// Locate the best entry points for working with a concept.
    pub fn locate_concept(
        &self,
        term: &str,
    ) -> Option<LocateConceptResult> {
        // Handle dotted subconcept query: "transform.spatial"
        let (base_term, subconcept_filter) =
            if let Some((base, sub)) = term.split_once('.') {
                (base, Some(sub))
            } else {
                (term, None)
            };

        let term_lower = base_term.to_lowercase();
        let concept = self
            .concepts
            .values()
            .find(|c| c.canonical == term_lower)?;

        // If subconcept filter, only use identifiers from that cluster
        let filter_ids: Option<HashSet<&str>> =
            subconcept_filter.and_then(|sub| {
                concept.subconcepts.iter().find_map(|sc| {
                    if sc.qualifier == sub
                        || sc.canonical == term
                    {
                        Some(
                            sc.identifiers
                                .iter()
                                .map(|s| s.as_str())
                                .collect(),
                        )
                    } else {
                        None
                    }
                })
            });

        // Rank signatures
        let mut scored_sigs: Vec<(&Signature, i32)> = self
            .signatures
            .iter()
            .filter_map(|sig| {
                let mut score = 0i32;
                let name_lower = sig.name.to_lowercase();

                // Check filter
                if let Some(ref ids) = filter_ids {
                    if !ids.contains(sig.name.as_str()) {
                        return None;
                    }
                }

                if concept
                    .occurrences
                    .iter()
                    .any(|o| o.identifier == sig.name)
                {
                    score += 3;
                } else if concept
                    .subtokens
                    .iter()
                    .any(|st| name_lower.contains(st.as_str()))
                {
                    score += 2;
                }

                // Param match
                for param in &sig.params {
                    let param_subtokens =
                        split_identifier(&param.name);
                    if param_subtokens
                        .iter()
                        .any(|st| st == &concept.canonical)
                    {
                        score += 1;
                    }
                }

                if score > 0 {
                    Some((sig, score))
                } else {
                    None
                }
            })
            .collect();
        scored_sigs.sort_by(|a, b| b.1.cmp(&a.1));
        let exemplar_signatures: Vec<Signature> = scored_sigs
            .into_iter()
            .take(5)
            .map(|(s, _)| s.clone())
            .collect();

        // Rank classes
        let mut scored_cls: Vec<(&ClassInfo, i32)> = self
            .classes
            .iter()
            .filter_map(|cls| {
                let mut score = 0i32;
                if concept
                    .occurrences
                    .iter()
                    .any(|o| o.identifier == cls.name)
                {
                    score += 3;
                }
                for method in &cls.methods {
                    if concept
                        .occurrences
                        .iter()
                        .any(|o| o.identifier == *method)
                    {
                        score += 1;
                    }
                }
                if score > 0 {
                    Some((cls, score))
                } else {
                    None
                }
            })
            .collect();
        scored_cls.sort_by(|a, b| b.1.cmp(&a.1));
        let exemplar_classes: Vec<ClassInfo> = scored_cls
            .into_iter()
            .take(3)
            .map(|(c, _)| c.clone())
            .collect();

        // Rank files by concept density
        let mut file_counts: HashMap<&std::path::Path, usize> =
            HashMap::new();
        for occ in &concept.occurrences {
            if let Some(ref ids) = filter_ids {
                if !ids.contains(occ.identifier.as_str()) {
                    continue;
                }
            }
            *file_counts.entry(&occ.file).or_insert(0) += 1;
        }
        let mut files: Vec<(std::path::PathBuf, usize)> = file_counts
            .into_iter()
            .map(|(p, c)| (p.to_path_buf(), c))
            .collect();
        files.sort_by(|a, b| b.1.cmp(&a.1));

        // Contrastive concepts
        let contrastive_concepts: Vec<String> = self
            .relationships
            .iter()
            .filter(|r| {
                r.kind == RelationshipKind::Contrastive
                    && (r.source == concept.id
                        || r.target == concept.id)
            })
            .filter_map(|r| {
                let other_id = if r.source == concept.id {
                    r.target
                } else {
                    r.source
                };
                self.concepts
                    .get(&other_id)
                    .map(|c| c.canonical.clone())
            })
            .collect();

        // Select key entities via cluster round-robin for diversity
        let key_candidates: Vec<&Entity> = self
            .entities
            .values()
            .filter(|e| e.concept_tags.contains(&concept.id))
            .collect();
        let key_entities = self.select_by_cluster(&key_candidates, 5);

        Some(LocateConceptResult {
            concept: concept.canonical.clone(),
            exemplar_signatures,
            exemplar_classes,
            files,
            contrastive_concepts,
            key_entities,
        })
    }

    /// Generate a session briefing from current graph state.
    pub fn session_briefing(&self) -> SessionBriefing {
        let conventions = self.conventions.clone();

        // Abbreviations from AbbreviationOf relationships
        let abbreviations: Vec<(String, String)> = self
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::AbbreviationOf)
            .filter_map(|r| {
                let short = self
                    .concepts
                    .get(&r.source)
                    .map(|c| c.canonical.clone())?;
                let long = self
                    .concepts
                    .get(&r.target)
                    .map(|c| c.canonical.clone())?;
                Some((short, long))
            })
            .collect();

        // Top 20 concepts
        let mut concepts_sorted: Vec<&Concept> =
            self.concepts.values().collect();
        concepts_sorted
            .sort_by(|a, b| b.occurrences.len().cmp(&a.occurrences.len()));
        let top_concepts: Vec<(String, usize)> = concepts_sorted
            .iter()
            .take(20)
            .map(|c| (c.canonical.clone(), c.occurrences.len()))
            .collect();

        // Contrastive pairs
        let contrastive_pairs: Vec<(String, String)> = self
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::Contrastive)
            .filter_map(|r| {
                let a = self
                    .concepts
                    .get(&r.source)
                    .map(|c| c.canonical.clone())?;
                let b = self
                    .concepts
                    .get(&r.target)
                    .map(|c| c.canonical.clone())?;
                Some((a, b))
            })
            .collect();

        // Vocabulary warnings
        let vocabulary_warnings: Vec<String> = self
            .concepts
            .values()
            .filter_map(|c| {
                let check = self.check_naming(&c.canonical);
                if check.verdict == Verdict::Inconsistent {
                    Some(format!(
                        "use '{}', not '{}'",
                        check
                            .suggestion
                            .as_deref()
                            .unwrap_or("?"),
                        c.canonical,
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Entity clusters grouped by semantic role (min 2 per cluster)
        let mut role_groups: HashMap<&str, Vec<&str>> = HashMap::new();
        for entity in self.entities.values() {
            if !entity.semantic_role.is_empty() {
                role_groups
                    .entry(&entity.semantic_role)
                    .or_default()
                    .push(&entity.name);
            }
        }
        let mut entity_clusters: Vec<_> = role_groups
            .into_iter()
            .filter(|(_, names)| names.len() >= 2)
            .map(|(role, names)| {
                use crate::types::EntityCluster;
                EntityCluster {
                    role: role.to_string(),
                    count: names.len(),
                    examples: names.into_iter().take(5).map(|s| s.to_string()).collect(),
                }
            })
            .collect();
        entity_clusters.sort_by(|a, b| b.count.cmp(&a.count));

        SessionBriefing {
            conventions,
            abbreviations,
            top_concepts,
            contrastive_pairs,
            vocabulary_warnings,
            entity_clusters,
        }
    }

    /// Semantic topology of the codebase: which directories concentrate
    /// which domain concepts, with entity counts and concept density.
    pub fn concept_map(&self) -> ConceptMap {
        use crate::types::EntityKind;

        if self.entities.is_empty() {
            return ConceptMap {
                modules: Vec::new(),
                total_entities: 0,
                total_concepts: self.concepts.len(),
            };
        }

        // Find the longest common prefix of all entity file paths
        let all_files: Vec<&std::path::Path> = self
            .entities
            .values()
            .map(|e| e.file.as_path())
            .collect();
        let common_prefix = longest_common_directory(&all_files);

        // Group entities by directory (relative to common prefix)
        let mut dir_groups: HashMap<String, Vec<&Entity>> = HashMap::new();
        for entity in self.entities.values() {
            let rel = entity
                .file
                .strip_prefix(&common_prefix)
                .unwrap_or(&entity.file);
            let dir = rel
                .parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            let dir = if dir.is_empty() {
                ".".to_string()
            } else {
                dir
            };
            dir_groups.entry(dir).or_default().push(entity);
        }

        // Build module entries
        let mut modules: Vec<ModuleMapEntry> = dir_groups
            .into_iter()
            .map(|(path, entities)| {
                let nb_classes = entities
                    .iter()
                    .filter(|e| e.kind == EntityKind::Class)
                    .count();
                let nb_functions = entities
                    .iter()
                    .filter(|e| {
                        e.kind == EntityKind::Function
                            || e.kind == EntityKind::Method
                    })
                    .count();
                let nb_entities = entities.len();

                // Collect concept tags, count frequency within directory
                let mut concept_freq: HashMap<u64, usize> = HashMap::new();
                let mut total_tags = 0usize;
                for entity in &entities {
                    for &cid in &entity.concept_tags {
                        *concept_freq.entry(cid).or_insert(0) += 1;
                        total_tags += 1;
                    }
                }

                // Rank concepts by frequency, take top 5
                let mut freq_vec: Vec<(u64, usize)> =
                    concept_freq.into_iter().collect();
                freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
                let dominant_concepts: Vec<String> = freq_vec
                    .iter()
                    .take(5)
                    .filter_map(|(id, _)| {
                        self.concepts
                            .get(id)
                            .map(|c| c.canonical.clone())
                    })
                    .collect();

                let concept_density = if nb_entities > 0 {
                    Some(total_tags as f64 / nb_entities as f64)
                } else {
                    None
                };

                ModuleMapEntry {
                    path,
                    dominant_concepts,
                    nb_classes,
                    nb_functions,
                    nb_entities,
                    concept_density,
                }
            })
            .collect();

        // Sort by entity count descending
        modules.sort_by(|a, b| b.nb_entities.cmp(&a.nb_entities));

        ConceptMap {
            total_entities: self.entities.len(),
            total_concepts: self.concepts.len(),
            modules,
        }
    }

    /// Look up the nesting tree for a file whose path ends with `path`.
    pub fn nesting_tree(&self, path: &str) -> Option<&FileNestingTree> {
        self.nesting_trees
            .iter()
            .find(|st| st.file.to_string_lossy().ends_with(path))
    }

    /// Trace how declared type annotations propagate along call
    /// edges. Builds TypeFlow edges from callee return types and
    /// parameter types using existing signatures and call sites.
    pub fn type_flows(&self) -> TypeFlowResult {
        // Index signatures by name for callee lookup.
        let mut sig_by_name: HashMap<&str, Vec<&Signature>> = HashMap::new();
        for sig in &self.signatures {
            sig_by_name
                .entry(sig.name.as_str())
                .or_default()
                .push(sig);
        }

        let mut flows: Vec<TypeFlow> = Vec::new();
        let mut typed_edges = 0usize;
        let mut untyped_edges = 0usize;

        for cs in &self.call_sites {
            // Find callee signature (match by name, pick first)
            let callee_sig = sig_by_name
                .get(cs.callee.as_str())
                .and_then(|sigs| sigs.first())
                .copied();

            // Determine caller display name from scope
            let caller_name = cs
                .caller_scope
                .as_deref()
                .unwrap_or("<module>");

            let callee_sig = match callee_sig {
                Some(s) => s,
                None => {
                    untyped_edges += 1;
                    continue;
                }
            };

            let mut found_type = false;

            // Return type flow: callee -> caller
            if let Some(ref ret) = callee_sig.return_type {
                let normalized = normalize_type_annotation(ret);
                if !normalized.is_empty() {
                    flows.push(TypeFlow {
                        from_entity: cs.callee.clone(),
                        to_entity: caller_name.to_string(),
                        type_name: normalized,
                        from_file: callee_sig.file.clone(),
                        to_file: cs.file.clone(),
                    });
                    found_type = true;
                }
            }

            // Parameter type flows: for each typed param, create
            // a flow from caller -> callee
            for param in &callee_sig.params {
                if let Some(ref ann) = param.type_annotation {
                    let normalized = normalize_type_annotation(ann);
                    if !normalized.is_empty() {
                        flows.push(TypeFlow {
                            from_entity: caller_name.to_string(),
                            to_entity: cs.callee.clone(),
                            type_name: normalized,
                            from_file: cs.file.clone(),
                            to_file: callee_sig.file.clone(),
                        });
                        found_type = true;
                    }
                }
            }

            if found_type {
                typed_edges += 1;
            } else {
                untyped_edges += 1;
            }
        }

        // Compute dominant types by frequency
        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for flow in &flows {
            *type_counts.entry(flow.type_name.clone()).or_insert(0) += 1;
        }
        let mut dominant_types: Vec<TypeFrequency> = type_counts
            .into_iter()
            .map(|(type_name, count)| TypeFrequency { type_name, count })
            .collect();
        dominant_types.sort_by(|a, b| b.count.cmp(&a.count));

        TypeFlowResult {
            flows,
            dominant_types,
            total_typed_edges: typed_edges,
            total_untyped_edges: untyped_edges,
        }
    }

    /// Filter type flows to those involving a specific type name.
    /// Matches by substring (case-insensitive).
    pub fn trace_type(&self, type_name: &str) -> Vec<TypeFlow> {
        let needle = type_name.to_lowercase();
        self.type_flows()
            .flows
            .into_iter()
            .filter(|f| f.type_name.to_lowercase().contains(&needle))
            .collect()
    }

    /// Measure vocabulary health across three dimensions:
    /// convention coverage, spelling consistency, and cluster
    /// cohesion. Returns an overall weighted score plus
    /// actionable lists of inconsistencies and uncovered names.
    pub fn vocabulary_health(
        &self,
        config: &HealthConfig,
    ) -> VocabularyHealth {
        let convention_coverage = self.compute_convention_coverage();
        let consistency_ratio = self.compute_consistency_ratio();
        let cluster_cohesion = self.compute_cluster_cohesion();

        let w1 = config.convention_coverage_weight;
        let w2 = config.consistency_ratio_weight;
        let w3 = config.cluster_cohesion_weight;
        let w_sum = w1 + w2 + w3;
        let overall = if w_sum > 0.0 {
            (convention_coverage * w1
                + consistency_ratio * w2
                + cluster_cohesion * w3)
                / w_sum
        } else {
            0.0
        };

        let top_inconsistencies =
            self.find_top_inconsistencies(10);
        let uncovered_identifiers =
            self.find_uncovered_identifiers(10);

        VocabularyHealth {
            convention_coverage,
            consistency_ratio,
            cluster_cohesion,
            overall,
            top_inconsistencies,
            uncovered_identifiers,
        }
    }

    /// Fraction of concepts whose occurrences match at least one
    /// convention pattern.
    fn compute_convention_coverage(&self) -> f32 {
        if self.concepts.is_empty() {
            return 0.0;
        }
        let matching = self
            .concepts
            .values()
            .filter(|c| {
                c.occurrences.iter().any(|occ| {
                    self.identifier_matches_any_convention(
                        &occ.identifier,
                    )
                })
            })
            .count();
        matching as f32 / self.concepts.len() as f32
    }

    /// Average per-concept spelling consistency. A concept where
    /// every occurrence uses the same identifier scores 1.0; one
    /// with many distinct spellings scores lower.
    fn compute_consistency_ratio(&self) -> f32 {
        let scores: Vec<f32> = self
            .concepts
            .values()
            .filter(|c| !c.occurrences.is_empty())
            .map(|c| {
                let mut unique = HashSet::new();
                for occ in &c.occurrences {
                    unique.insert(occ.identifier.to_lowercase());
                }
                let nb_unique = unique.len();
                let nb_occ = c.occurrences.len().max(1);
                1.0 - ((nb_unique - 1) as f32 / nb_occ as f32)
            })
            .collect();
        if scores.is_empty() {
            return 1.0;
        }
        scores.iter().sum::<f32>() / scores.len() as f32
    }

    /// Average cosine similarity of cluster members to their
    /// centroid, averaged across all clusters.
    fn compute_cluster_cohesion(&self) -> f32 {
        if self.cluster_centroids.is_empty() {
            return 1.0;
        }
        let mut cluster_sims: Vec<f32> = Vec::new();
        for (&cluster_id, centroid) in &self.cluster_centroids {
            let members: Vec<&Concept> = self
                .concepts
                .values()
                .filter(|c| c.cluster_id == Some(cluster_id))
                .collect();
            if members.is_empty() {
                continue;
            }
            let mut sims: Vec<f32> = Vec::new();
            for member in &members {
                if let Some(vec) =
                    self.embeddings.get_vector(member.id)
                {
                    sims.push(cosine_similarity(vec, centroid));
                }
            }
            if !sims.is_empty() {
                let avg =
                    sims.iter().sum::<f32>() / sims.len() as f32;
                cluster_sims.push(avg);
            }
        }
        if cluster_sims.is_empty() {
            return 1.0;
        }
        cluster_sims.iter().sum::<f32>()
            / cluster_sims.len() as f32
    }

    /// Check whether an identifier matches any convention pattern.
    fn identifier_matches_any_convention(
        &self,
        identifier: &str,
    ) -> bool {
        let id_lower = identifier.to_lowercase();
        self.conventions.iter().any(|conv| match &conv.pattern {
            PatternKind::Prefix(p) => {
                id_lower.starts_with(p.as_str())
            }
            PatternKind::Suffix(s) => {
                id_lower.ends_with(s.as_str())
            }
            PatternKind::Conversion(c) => {
                id_lower.contains(c.as_str())
            }
            PatternKind::Compound(c) => {
                id_lower.contains(c.as_str())
            }
        })
    }

    /// Find concepts with multiple identifier spellings. For each,
    /// pair the dominant spelling against each minority spelling.
    fn find_top_inconsistencies(
        &self,
        limit: usize,
    ) -> Vec<InconsistencyPair> {
        let mut pairs: Vec<InconsistencyPair> = Vec::new();
        for concept in self.concepts.values() {
            let mut counts: HashMap<String, usize> = HashMap::new();
            for occ in &concept.occurrences {
                *counts
                    .entry(occ.identifier.to_lowercase())
                    .or_insert(0) += 1;
            }
            if counts.len() < 2 {
                continue;
            }
            let mut sorted: Vec<(String, usize)> =
                counts.into_iter().collect();
            sorted.sort_by(|a, b| b.1.cmp(&a.1));

            let (dominant, dominant_count) = &sorted[0];
            for &(ref minority, minority_count) in &sorted[1..] {
                pairs.push(InconsistencyPair {
                    dominant: dominant.clone(),
                    minority: minority.clone(),
                    dominant_count: *dominant_count,
                    minority_count,
                });
            }
        }
        pairs.sort_by(|a, b| {
            let diff_a = a.dominant_count.saturating_sub(
                a.minority_count,
            );
            let diff_b = b.dominant_count.saturating_sub(
                b.minority_count,
            );
            diff_b.cmp(&diff_a)
        });
        pairs.truncate(limit);
        pairs
    }

    /// Collect identifiers that match no convention pattern,
    /// sorted by frequency descending.
    fn find_uncovered_identifiers(
        &self,
        limit: usize,
    ) -> Vec<String> {
        let mut id_freq: HashMap<String, usize> = HashMap::new();
        for concept in self.concepts.values() {
            for occ in &concept.occurrences {
                let id = occ.identifier.to_lowercase();
                *id_freq.entry(id).or_insert(0) += 1;
            }
        }
        let mut uncovered: Vec<(String, usize)> = id_freq
            .into_iter()
            .filter(|(id, _)| {
                !self.identifier_matches_any_convention(id)
            })
            .collect();
        uncovered.sort_by(|a, b| b.1.cmp(&a.1));
        uncovered.truncate(limit);
        uncovered.into_iter().map(|(id, _)| id).collect()
    }

    /// Trace a concept through the call graph, showing which
    /// entities produce it, consume it, and bridge between them.
    pub fn trace_concept(
        &self,
        concept_name: &str,
        max_depth: usize,
    ) -> Option<ConceptTrace> {
        use petgraph::graph::{DiGraph, NodeIndex};
        use std::collections::VecDeque;

        let term_lower = concept_name.to_lowercase();
        let subtokens = split_identifier(concept_name);

        // 1. Resolve: entity-first, then concept fallback.
        //    If the query matches an entity name exactly, seed from that
        //    entity rather than going through concept resolution (which
        //    would split "BasicUNet" into subtoken "basic" and lose
        //    specificity).
        let entity_match: Option<&Entity> = self
            .entities
            .values()
            .find(|e| e.name.to_lowercase() == term_lower);

        let is_entity_trace = entity_match.is_some();

        let (concept_label, seed_ids) =
            if let Some(matched) = entity_match {
                let label = self
                    .concepts
                    .get(
                        matched
                            .concept_tags
                            .first()
                            .unwrap_or(&0),
                    )
                    .map(|c| c.canonical.clone())
                    .unwrap_or_else(|| {
                        matched.name.to_lowercase()
                    });
                let ids: HashSet<u64> =
                    std::iter::once(matched.id).collect();
                (label, ids)
            } else {
                // Concept-based fallback
                let concept = self
                    .concepts
                    .values()
                    .find(|c| c.canonical == term_lower)
                    .or_else(|| {
                        subtokens.iter().find_map(|st| {
                            self.concepts
                                .values()
                                .find(|c| c.canonical == *st)
                        })
                    })
                    .or_else(|| {
                        self.concepts.values().find(|c| {
                            c.occurrences.iter().any(|o| {
                                o.identifier
                                    .to_lowercase()
                                    .contains(&term_lower)
                            })
                        })
                    })?;
                let ids: HashSet<u64> = self
                    .entities
                    .values()
                    .filter(|e| {
                        e.concept_tags.contains(&concept.id)
                    })
                    .map(|e| e.id)
                    .collect();
                (concept.canonical.clone(), ids)
            };

        if seed_ids.is_empty() {
            return Some(ConceptTrace {
                concept: concept_label,
                producers: Vec::new(),
                consumers: Vec::new(),
                call_chain: Vec::new(),
                edges: Vec::new(),
            });
        }

        // Collect concept subtokens for role classification
        let role_subtokens: Vec<String> = seed_ids
            .iter()
            .filter_map(|id| self.entities.get(id))
            .flat_map(|e| {
                e.concept_tags.iter().filter_map(|cid| {
                    self.concepts.get(cid)
                })
            })
            .flat_map(|c| c.subtokens.clone())
            .collect();

        // 2. Classify seeds as producer/consumer/both
        let classify =
            |entity: &Entity| -> TraceRole {
                let sig = entity
                    .signature_idx
                    .and_then(|i| self.signatures.get(i));
                let Some(sig) = sig else {
                    return TraceRole::Both;
                };
                let is_producer = sig
                    .return_type
                    .as_ref()
                    .map(|rt| {
                        let rt_lower = rt.to_lowercase();
                        role_subtokens.iter().any(|s| {
                            rt_lower
                                .contains(&s.to_lowercase())
                        })
                    })
                    .unwrap_or(false);
                let is_consumer =
                    sig.params.iter().any(|p| {
                        let name_lower =
                            p.name.to_lowercase();
                        let type_lower = p
                            .type_annotation
                            .as_deref()
                            .unwrap_or("")
                            .to_lowercase();
                        role_subtokens.iter().any(|s| {
                            let s_lower = s.to_lowercase();
                            name_lower.contains(&s_lower)
                                || type_lower
                                    .contains(&s_lower)
                        })
                    });
                match (is_producer, is_consumer) {
                    (true, true) => TraceRole::Both,
                    (true, false) => TraceRole::Producer,
                    (false, true) => TraceRole::Consumer,
                    (false, false) => TraceRole::Both,
                }
            };

        let mut seed_roles: HashMap<u64, TraceRole> =
            HashMap::new();
        for &sid in &seed_ids {
            if let Some(e) = self.entities.get(&sid) {
                seed_roles.insert(sid, classify(e));
            }
        }

        // 3. Build entity-level graph from call sites + entity
        //    relationships.
        //    Normalize scope names to entity level so that
        //    "BasicUNet.__init__" → "BasicUNet" matches entity
        //    names.
        let mut call_graph: DiGraph<String, ()> =
            DiGraph::new();
        let mut node_map: HashMap<String, NodeIndex> =
            HashMap::new();

        let ensure_node =
            |g: &mut DiGraph<String, ()>,
             m: &mut HashMap<String, NodeIndex>,
             name: &str|
             -> NodeIndex {
                if let Some(&idx) = m.get(name) {
                    idx
                } else {
                    let idx = g.add_node(name.to_string());
                    m.insert(name.to_string(), idx);
                    idx
                }
            };

        // Normalize caller_scope to entity level (same as
        // entity.rs): "Cls.method" → "Cls"
        // Normalize callee to last component: "nn.Conv3d" →
        // "Conv3d"
        for cs in &self.call_sites {
            let Some(ref caller) = cs.caller_scope else {
                continue;
            };
            let caller_name = caller
                .split('.')
                .next()
                .unwrap_or(caller);
            let callee_name = cs
                .callee
                .rsplit('.')
                .next()
                .unwrap_or(&cs.callee);
            if caller_name == callee_name {
                continue;
            }
            let caller_idx = ensure_node(
                &mut call_graph,
                &mut node_map,
                caller_name,
            );
            let callee_idx = ensure_node(
                &mut call_graph,
                &mut node_map,
                callee_name,
            );
            call_graph.add_edge(caller_idx, callee_idx, ());
        }

        // Inject entity-level relationships (Uses,
        // InheritsFrom) as additional edges
        let entity_id_to_name: HashMap<u64, &str> = self
            .entities
            .values()
            .map(|e| (e.id, e.name.as_str()))
            .collect();

        for rel in &self.relationships {
            let is_entity_rel = matches!(
                rel.kind,
                RelationshipKind::Uses
                    | RelationshipKind::InheritsFrom
            );
            if !is_entity_rel {
                continue;
            }
            let (Some(src), Some(tgt)) = (
                entity_id_to_name.get(&rel.source),
                entity_id_to_name.get(&rel.target),
            ) else {
                continue;
            };
            if src == tgt {
                continue;
            }
            let src_idx = ensure_node(
                &mut call_graph,
                &mut node_map,
                src,
            );
            let tgt_idx = ensure_node(
                &mut call_graph,
                &mut node_map,
                tgt,
            );
            call_graph.add_edge(src_idx, tgt_idx, ());
        }

        // Map seed entity names to node indices
        let seed_node_indices: HashMap<String, NodeIndex> =
            self.entities
                .values()
                .filter(|e| seed_ids.contains(&e.id))
                .filter_map(|e| {
                    node_map.get(&e.name).map(|&idx| {
                        (e.name.clone(), idx)
                    })
                })
                .collect();

        // 4. Find bridges via BFS forward/backward from seeds
        let bfs_reachable = |starts: &[NodeIndex],
                             forward: bool|
         -> HashSet<NodeIndex> {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            for &s in starts {
                visited.insert(s);
                queue.push_back((s, 0usize));
            }
            while let Some((node, depth)) =
                queue.pop_front()
            {
                if depth >= max_depth {
                    continue;
                }
                let neighbors: Vec<NodeIndex> = if forward
                {
                    call_graph
                        .neighbors_directed(
                            node,
                            petgraph::Direction::Outgoing,
                        )
                        .collect()
                } else {
                    call_graph
                        .neighbors_directed(
                            node,
                            petgraph::Direction::Incoming,
                        )
                        .collect()
                };
                for nb in neighbors {
                    if visited.insert(nb) {
                        queue.push_back((nb, depth + 1));
                    }
                }
            }
            visited
        };

        // Build seed name→entity lookup for role
        // classification
        let seed_entity_by_name: HashMap<&str, &Entity> =
            self.entities
                .values()
                .filter(|e| seed_ids.contains(&e.id))
                .map(|e| (e.name.as_str(), e))
                .collect();

        // Split seeds by role for directional BFS
        let producer_indices: Vec<NodeIndex> =
            seed_node_indices
                .iter()
                .filter(|(name, _)| {
                    seed_entity_by_name
                        .get(name.as_str())
                        .and_then(|e| {
                            seed_roles.get(&e.id)
                        })
                        .map(|r| {
                            matches!(
                                r,
                                TraceRole::Producer
                                    | TraceRole::Both
                            )
                        })
                        .unwrap_or(false)
                })
                .map(|(_, &idx)| idx)
                .collect();
        let consumer_indices: Vec<NodeIndex> =
            seed_node_indices
                .iter()
                .filter(|(name, _)| {
                    seed_entity_by_name
                        .get(name.as_str())
                        .and_then(|e| {
                            seed_roles.get(&e.id)
                        })
                        .map(|r| {
                            matches!(
                                r,
                                TraceRole::Consumer
                                    | TraceRole::Both
                            )
                        })
                        .unwrap_or(false)
                })
                .map(|(_, &idx)| idx)
                .collect();
        let forward_set =
            bfs_reachable(&producer_indices, true);
        let backward_set =
            bfs_reachable(&consumer_indices, false);

        let seed_idx_set: HashSet<NodeIndex> =
            seed_node_indices.values().copied().collect();

        // Entity name set for filtering reachable nodes to
        // known entities only
        let entity_name_set: HashSet<&str> =
            entity_id_to_name.values().copied().collect();

        // 5. Build subgraph nodes.
        //    Entity-first traces: include all reachable
        //    entity nodes (full depth BFS).
        //    Concept-based traces: use forward∩backward
        //    bridge intersection.
        let subgraph_nodes: HashSet<NodeIndex> =
            if is_entity_trace {
                let all_seeds: Vec<NodeIndex> =
                    seed_idx_set.iter().copied().collect();
                let reachable_fwd =
                    bfs_reachable(&all_seeds, true);
                let reachable_bwd =
                    bfs_reachable(&all_seeds, false);
                reachable_fwd
                    .union(&reachable_bwd)
                    .copied()
                    .filter(|n| {
                        seed_idx_set.contains(n)
                            || entity_name_set.contains(
                                call_graph[*n].as_str(),
                            )
                    })
                    .collect()
            } else {
                let bridge_indices: HashSet<NodeIndex> =
                    forward_set
                        .intersection(&backward_set)
                        .copied()
                        .filter(|n| {
                            !seed_idx_set.contains(n)
                        })
                        .collect();
                // Also include direct entity neighbors of
                // seeds
                let mut neighbor_indices: HashSet<NodeIndex> =
                    HashSet::new();
                for &seed_idx in seed_idx_set.iter() {
                    for nb in call_graph
                        .neighbors_directed(
                            seed_idx,
                            petgraph::Direction::Outgoing,
                        )
                        .chain(
                            call_graph.neighbors_directed(
                                seed_idx,
                                petgraph::Direction::Incoming,
                            ),
                        )
                    {
                        if entity_name_set.contains(
                            call_graph[nb].as_str(),
                        ) {
                            neighbor_indices.insert(nb);
                        }
                    }
                }
                seed_idx_set
                    .iter()
                    .chain(bridge_indices.iter())
                    .chain(neighbor_indices.iter())
                    .copied()
                    .collect()
            };

        let mut sub: DiGraph<String, ()> = DiGraph::new();
        let mut sub_map: HashMap<NodeIndex, NodeIndex> =
            HashMap::new();
        for &orig in &subgraph_nodes {
            let new =
                sub.add_node(call_graph[orig].clone());
            sub_map.insert(orig, new);
        }
        for &orig in &subgraph_nodes {
            for neighbor in call_graph.neighbors_directed(
                orig,
                petgraph::Direction::Outgoing,
            ) {
                if let (Some(&from), Some(&to)) = (
                    sub_map.get(&orig),
                    sub_map.get(&neighbor),
                ) {
                    sub.add_edge(from, to, ());
                }
            }
        }

        // Topological sort; fall back to BFS from roots
        let ordered: Vec<NodeIndex> =
            match petgraph::algo::toposort(&sub, None) {
                Ok(sorted) => sorted,
                Err(_) => {
                    let roots: Vec<NodeIndex> = sub
                        .node_indices()
                        .filter(|n| {
                            sub.neighbors_directed(
                                *n,
                                petgraph::Direction::Incoming,
                            )
                            .next()
                            .is_none()
                        })
                        .collect();
                    let mut visited = HashSet::new();
                    let mut queue = VecDeque::new();
                    let mut result = Vec::new();
                    for r in roots {
                        if visited.insert(r) {
                            queue.push_back(r);
                        }
                    }
                    if queue.is_empty() {
                        if let Some(n) =
                            sub.node_indices().next()
                        {
                            visited.insert(n);
                            queue.push_back(n);
                        }
                    }
                    while let Some(n) = queue.pop_front() {
                        result.push(n);
                        for nb in sub.neighbors_directed(
                            n,
                            petgraph::Direction::Outgoing,
                        ) {
                            if visited.insert(nb) {
                                queue.push_back(nb);
                            }
                        }
                        if queue.is_empty() {
                            for c in sub.node_indices() {
                                if visited.insert(c) {
                                    queue.push_back(c);
                                    break;
                                }
                            }
                        }
                    }
                    result
                }
            };

        // 6. Assemble result
        let entity_by_name: HashMap<&str, &Entity> = self
            .entities
            .values()
            .map(|e| (e.name.as_str(), e))
            .collect();

        let concept_name_for_id = |cid: &u64| -> String {
            self.concepts
                .get(cid)
                .map(|c| c.canonical.clone())
                .unwrap_or_default()
        };

        // Class info lookup for bases/methods
        let class_by_name: HashMap<&str, &ClassInfo> = self
            .classes
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();

        // Method signatures grouped by class name
        let method_sigs_by_class: HashMap<&str, Vec<&Signature>> = {
            let mut map: HashMap<&str, Vec<&Signature>> =
                HashMap::new();
            for sig in &self.signatures {
                if let Some(ref scope) = sig.scope {
                    let class_name = scope
                        .split('.')
                        .next()
                        .unwrap_or(scope);
                    if class_by_name.contains_key(class_name)
                    {
                        map.entry(class_name)
                            .or_default()
                            .push(sig);
                    }
                }
            }
            map
        };

        let make_trace_node =
            |name: &str, role: TraceRole| -> TraceNode {
                if let Some(entity) =
                    entity_by_name.get(name)
                {
                    let (bases, methods) =
                        if entity.kind
                            == crate::types::EntityKind::Class
                        {
                            let bases = class_by_name
                                .get(name)
                                .map(|c| c.bases.clone())
                                .unwrap_or_default();
                            let methods = method_sigs_by_class
                                .get(name)
                                .map(|sigs| {
                                    sigs.iter()
                                        .map(|s| MethodSummary {
                                            name: s.name.clone(),
                                            params: s
                                                .params
                                                .iter()
                                                .filter(|p| {
                                                    p.name != "self"
                                                })
                                                .map(|p| {
                                                    match &p.type_annotation {
                                                        Some(t) => format!("{}: {}", p.name, t),
                                                        None => p.name.clone(),
                                                    }
                                                })
                                                .collect(),
                                            return_type: s
                                                .return_type
                                                .clone(),
                                            line: s.line,
                                        })
                                        .collect()
                                })
                                .unwrap_or_default();
                            (bases, methods)
                        } else {
                            (Vec::new(), Vec::new())
                        };

                    TraceNode {
                        entity_name: entity.name.clone(),
                        kind: entity.kind.clone(),
                        file: entity.file.clone(),
                        line: entity.line,
                        role,
                        concept_tags: entity
                            .concept_tags
                            .iter()
                            .map(concept_name_for_id)
                            .collect(),
                        bases,
                        methods,
                    }
                } else {
                    TraceNode {
                        entity_name: name.to_string(),
                        kind: crate::types::EntityKind::Function,
                        file: PathBuf::new(),
                        line: 0,
                        role,
                        concept_tags: Vec::new(),
                        bases: Vec::new(),
                        methods: Vec::new(),
                    }
                }
            };

        let mut call_chain = Vec::new();
        for &sub_idx in &ordered {
            let name = &sub[sub_idx];
            let orig_idx = sub_map.iter().find_map(
                |(&orig, &mapped)| {
                    if mapped == sub_idx {
                        Some(orig)
                    } else {
                        None
                    }
                },
            );
            let role = if let Some(oi) = orig_idx {
                if !seed_idx_set.contains(&oi) {
                    TraceRole::Bridge
                } else if let Some(entity) =
                    entity_by_name.get(name.as_str())
                {
                    seed_roles
                        .get(&entity.id)
                        .cloned()
                        .unwrap_or(TraceRole::Both)
                } else {
                    TraceRole::Both
                }
            } else {
                TraceRole::Both
            };
            call_chain.push(make_trace_node(name, role));
        }

        // Include orphan seeds (entities not in any call
        // site)
        let chain_names: HashSet<&str> = call_chain
            .iter()
            .map(|n| n.entity_name.as_str())
            .collect();
        let orphans: Vec<TraceNode> = seed_roles
            .iter()
            .filter_map(|(&sid, role)| {
                let entity = self.entities.get(&sid)?;
                if chain_names
                    .contains(entity.name.as_str())
                {
                    return None;
                }
                Some(make_trace_node(
                    &entity.name,
                    role.clone(),
                ))
            })
            .collect();
        call_chain.extend(orphans);

        // Build edges from the subgraph
        let subgraph_names: HashSet<&str> = subgraph_nodes
            .iter()
            .map(|&n| call_graph[n].as_str())
            .collect();

        let mut edges: Vec<TraceEdge> = Vec::new();
        let mut seen_edges: HashSet<(String, String)> =
            HashSet::new();

        // Edges from call sites (normalized to entity level)
        for cs in &self.call_sites {
            let Some(ref caller) = cs.caller_scope else {
                continue;
            };
            let caller_name = caller
                .split('.')
                .next()
                .unwrap_or(caller);
            let callee_name = cs
                .callee
                .rsplit('.')
                .next()
                .unwrap_or(&cs.callee);
            if subgraph_names.contains(caller_name)
                && subgraph_names.contains(callee_name)
                && caller_name != callee_name
            {
                let key = (
                    caller_name.to_string(),
                    callee_name.to_string(),
                );
                if seen_edges.insert(key) {
                    edges.push(TraceEdge {
                        caller: caller_name.to_string(),
                        callee: callee_name.to_string(),
                        file: cs.file.clone(),
                        line: cs.line,
                    });
                }
            }
        }

        // Edges from entity relationships
        for rel in &self.relationships {
            let is_entity_rel = matches!(
                rel.kind,
                RelationshipKind::Uses
                    | RelationshipKind::InheritsFrom
            );
            if !is_entity_rel {
                continue;
            }
            let (Some(src), Some(tgt)) = (
                entity_id_to_name.get(&rel.source),
                entity_id_to_name.get(&rel.target),
            ) else {
                continue;
            };
            if subgraph_names.contains(src)
                && subgraph_names.contains(tgt)
            {
                let key =
                    (src.to_string(), tgt.to_string());
                if seen_edges.insert(key) {
                    let file = self
                        .entities
                        .get(&rel.source)
                        .map(|e| e.file.clone())
                        .unwrap_or_default();
                    let line = self
                        .entities
                        .get(&rel.source)
                        .map(|e| e.line)
                        .unwrap_or(0);
                    edges.push(TraceEdge {
                        caller: src.to_string(),
                        callee: tgt.to_string(),
                        file,
                        line,
                    });
                }
            }
        }

        let producers: Vec<TraceNode> = call_chain
            .iter()
            .filter(|n| {
                matches!(
                    n.role,
                    TraceRole::Producer | TraceRole::Both
                ) && n.role != TraceRole::Bridge
            })
            .cloned()
            .collect();

        let consumers: Vec<TraceNode> = call_chain
            .iter()
            .filter(|n| {
                matches!(
                    n.role,
                    TraceRole::Consumer | TraceRole::Both
                ) && n.role != TraceRole::Bridge
            })
            .cloned()
            .collect();

        Some(ConceptTrace {
            concept: concept_label,
            producers,
            consumers,
            call_chain,
            edges,
        })
    }

    // -- L4: Logic query methods --

    /// Helper: build an EntitySummary from an Entity.
    fn entity_summary(entity: &Entity) -> EntitySummary {
        EntitySummary {
            name: entity.name.clone(),
            kind: entity.kind.clone(),
            file: entity.file.clone(),
            line: entity.line,
        }
    }

    /// L4: Describe the behavioral logic of an entity.
    pub fn describe_logic(&self, name: &str) -> Option<LogicDescription> {
        let name_lower = name.to_lowercase();
        let entity = self.entities.values().find(|e| {
            e.name.to_lowercase() == name_lower
        })?;

        let pc_text = self.pseudocode.get(&entity.id)
            .map(format_pseudocode)
            .unwrap_or_default();

        let cluster_summary = self.logic_clusters.iter()
            .find(|lc| lc.entity_ids.contains(&entity.id))
            .map(|lc| {
                let members: Vec<EntitySummary> = lc.entity_ids.iter()
                    .filter_map(|id| self.entities.get(id))
                    .map(Self::entity_summary)
                    .collect();
                LogicClusterSummary {
                    id: lc.id,
                    size: lc.entity_ids.len(),
                    behavioral_label: lc.behavioral_label.clone(),
                    members,
                }
            });

        let centrality = self.centrality.get(&entity.id)
            .cloned()
            .unwrap_or(CentralityScore {
                entity_id: entity.id,
                in_degree: 0,
                out_degree: 0,
                pagerank: 0.0,
            });

        Some(LogicDescription {
            entity: Self::entity_summary(entity),
            pseudocode_text: pc_text,
            logic_cluster: cluster_summary,
            centrality,
        })
    }

    /// L4: Find entities with similar behavioral patterns.
    pub fn find_similar_logic(
        &self,
        name: &str,
        top_k: usize,
    ) -> Option<SimilarLogicResult> {
        let name_lower = name.to_lowercase();
        let entity = self.entities.values().find(|e| {
            e.name.to_lowercase() == name_lower
        })?;

        let similar_ids = self.logic_index
            .find_similar_to_entity(entity.id, top_k);

        let similar: Vec<(EntitySummary, f32)> = similar_ids.iter()
            .filter_map(|(id, score)| {
                self.entities.get(id)
                    .map(|e| (Self::entity_summary(e), *score))
            })
            .collect();

        Some(SimilarLogicResult {
            query_entity: Self::entity_summary(entity),
            similar,
        })
    }

    /// L4: Assemble minimal context for a concept or entity.
    ///
    /// Scope resolution order: entity name > file path > concept name.
    /// Uses tiered truncation to stay within token budget.
    pub fn compact_context(
        &self,
        scope: &str,
        max_tokens: usize,
    ) -> Option<CompactContext> {
        // Priority levels for tiered truncation.
        // 1=highest (keep longest), 5=lowest (drop first).
        struct Section {
            priority: u8,
            text: String,
        }

        let scope_lower = scope.to_lowercase();

        // --- Scope resolution ---

        // 1. Try entity name first
        let mut entity = self.entities.values().find(|e| {
            e.name.to_lowercase() == scope_lower
        });

        // 2. If no entity match, try file path resolution.
        // Require a known extension to avoid false positives on concept
        // names containing "/" (e.g. "encoder/decoder").
        let has_ext = scope.ends_with(".py")
            || scope.ends_with(".rs")
            || scope.ends_with(".ts")
            || scope.ends_with(".js")
            || scope.ends_with(".tsx")
            || scope.ends_with(".jsx");
        let is_file_path = has_ext
            || (scope.contains('/') && scope.starts_with('/'))
            || (scope.contains('/') && scope.starts_with('.'));

        if entity.is_none() && is_file_path {
            let mut file_matches: Vec<&Entity> = self.entities.values()
                .filter(|e| {
                    let path_str = e.file.to_string_lossy();
                    path_str.ends_with(scope) || path_str.contains(scope)
                })
                .collect();
            // Rank by centrality (highest pagerank first)
            file_matches.sort_by(|a, b| {
                let pr_a = self.centrality.get(&a.id)
                    .map(|c| c.pagerank)
                    .unwrap_or(0.0);
                let pr_b = self.centrality.get(&b.id)
                    .map(|c| c.pagerank)
                    .unwrap_or(0.0);
                pr_b.partial_cmp(&pr_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            entity = file_matches.into_iter().next();
            // File path scope: if no entity matches, return None
            // (don't fall through to concept matching)
            entity?;
        }

        // 3. If no entity, try matching a concept
        let concept = if entity.is_none() {
            self.concepts.values().find(|c| {
                c.canonical == scope_lower
            })
        } else {
            None
        };

        if entity.is_none() && concept.is_none() {
            return None;
        }

        // --- Build sections with priority levels ---

        // Priority 1: Header
        let header = Section {
            priority: 1,
            text: format!("# {} — compact context", scope),
        };

        let mut tiered: Vec<Section> = vec![header];

        if let Some(ent) = entity {
            // Priority 2: Structure
            let mut structure = format!(
                "## Structure\n{} ({:?})",
                ent.name, ent.kind
            );
            let class_info = self.classes.iter().find(|c| {
                c.name == ent.name && c.file == ent.file
            });
            if let Some(cls) = class_info {
                if !cls.bases.is_empty() {
                    structure.push_str(&format!(
                        "\n  inherits: {}",
                        cls.bases.join(", ")
                    ));
                }
            }

            // Dependency names (outgoing: what this entity uses/inherits)
            let mut depends_on: Vec<&str> = self.relationships.iter()
                .filter(|r| {
                    (r.kind == RelationshipKind::InheritsFrom
                        || r.kind == RelationshipKind::Uses)
                        && r.source == ent.id
                })
                .filter_map(|r| {
                    self.entities.get(&r.target)
                        .map(|e| e.name.as_str())
                })
                .collect();
            depends_on.sort_unstable();
            depends_on.dedup();
            if !depends_on.is_empty() {
                structure.push_str(&format!(
                    "\n  depends on: {}",
                    depends_on.join(", ")
                ));
            }

            // Dependency names (incoming: what depends on this entity)
            let mut depended_on_by: Vec<&str> = self.relationships.iter()
                .filter(|r| {
                    (r.kind == RelationshipKind::InheritsFrom
                        || r.kind == RelationshipKind::Uses)
                        && r.target == ent.id
                })
                .filter_map(|r| {
                    self.entities.get(&r.source)
                        .map(|e| e.name.as_str())
                })
                .collect();
            depended_on_by.sort_unstable();
            depended_on_by.dedup();
            if !depended_on_by.is_empty() {
                structure.push_str(&format!(
                    "\n  depended on by: {}",
                    depended_on_by.join(", "),
                ));
            }

            if let Some(cs) = self.centrality.get(&ent.id) {
                structure.push_str(&format!(
                    "\n  centrality: pagerank={:.3}, in={}, out={}",
                    cs.pagerank, cs.in_degree, cs.out_degree
                ));
            }
            tiered.push(Section { priority: 2, text: structure });

            // Priority 3: Behavior (pseudocode)
            if let Some(pc) = self.pseudocode.get(&ent.id) {
                let pc_text = format_pseudocode(pc);
                if !pc_text.is_empty() {
                    tiered.push(Section {
                        priority: 3,
                        text: format!("## Behavior\n{}", pc_text),
                    });
                }
            }

            // Priority 4: Domain
            let mut domain_parts: Vec<String> = Vec::new();
            let concept_names: Vec<&str> = ent.concept_tags.iter()
                .filter_map(|id| {
                    self.concepts.get(id).map(|c| c.canonical.as_str())
                })
                .collect();
            if !concept_names.is_empty() {
                domain_parts.push(format!(
                    "Concepts: {}",
                    concept_names.join(", ")
                ));
            }

            let matching_convs: Vec<&str> = self.conventions.iter()
                .filter(|conv| {
                    conv.examples.iter().any(|ex| {
                        ex.to_lowercase()
                            .contains(&ent.name.to_lowercase())
                    })
                })
                .map(|conv| conv.semantic_role.as_str())
                .collect();
            if !matching_convs.is_empty() {
                domain_parts.push(format!(
                    "Conventions: {}",
                    matching_convs.join(", ")
                ));
            }

            if let Some(lc) = self.logic_clusters.iter().find(|lc| {
                lc.entity_ids.contains(&ent.id)
            }) {
                let label = lc.behavioral_label.as_deref()
                    .unwrap_or("unlabeled");
                let members: Vec<String> = lc.entity_ids.iter()
                    .filter(|id| **id != ent.id)
                    .filter_map(|id| {
                        self.entities.get(id)
                            .map(|e| e.name.clone())
                    })
                    .collect();
                let member_list = if members.is_empty() {
                    String::new()
                } else {
                    format!(": {}", members.join(", "))
                };
                domain_parts.push(format!(
                    "Logic cluster: \"{}\" ({} members){}",
                    label, lc.entity_ids.len(), member_list,
                ));
            }

            if !domain_parts.is_empty() {
                tiered.push(Section {
                    priority: 4,
                    text: format!(
                        "## Domain\n{}",
                        domain_parts.join("\n"),
                    ),
                });
            }

            // Priority 5: Related (top-3 similar by logic)
            let similar = self.logic_index
                .find_similar_to_entity(ent.id, 3);
            if !similar.is_empty() {
                let related_lines: Vec<String> = similar.iter()
                    .filter_map(|(id, score)| {
                        self.entities.get(id).map(|e| {
                            format!(
                                "- {} (similarity: {:.2})",
                                e.name, score,
                            )
                        })
                    })
                    .collect();
                if !related_lines.is_empty() {
                    tiered.push(Section {
                        priority: 5,
                        text: format!(
                            "## Related\n{}",
                            related_lines.join("\n"),
                        ),
                    });
                }
            }
        } else if let Some(concept) = concept {
            // Concept-based context (no tiered truncation needed,
            // these are already compact)
            let occ_count = concept.occurrences.len();
            let files: HashSet<&str> = concept.occurrences.iter()
                .filter_map(|o| o.file.to_str())
                .collect();
            tiered.push(Section {
                priority: 2,
                text: format!(
                    "## Concept: {}\nOccurrences: {}, Files: {}",
                    concept.canonical, occ_count, files.len(),
                ),
            });

            let tagged_entities: Vec<&Entity> = self.entities.values()
                .filter(|e| e.concept_tags.contains(&concept.id))
                .collect();
            if !tagged_entities.is_empty() {
                let entity_lines: Vec<String> = tagged_entities.iter()
                    .take(5)
                    .map(|e| format!("- {} ({:?})", e.name, e.kind))
                    .collect();
                tiered.push(Section {
                    priority: 5,
                    text: format!(
                        "## Entities\n{}",
                        entity_lines.join("\n"),
                    ),
                });
            }
        }

        // --- Tiered truncation ---
        // Apply stages in order until under budget.

        let estimate = |sections: &[Section]| -> usize {
            let total_chars: usize = sections.iter()
                .map(|s| s.text.len() + 2) // +2 for "\n\n" join
                .sum();
            total_chars / 4
        };

        // Stage 1: Trim pseudocode (Behavior, priority 3) to first 2
        // + last 2 lines if >4 lines.
        // Strip any existing omission banner from format_pseudocode to
        // avoid double banners.
        if estimate(&tiered) > max_tokens {
            for s in &mut tiered {
                if s.priority != 3 {
                    continue;
                }
                let lines: Vec<&str> = s.text.lines()
                    .filter(|l| !l.starts_with("... (") || !l.ends_with("omitted) ..."))
                    .collect();
                if lines.len() > 5 {
                    let header_line = lines[0];
                    let pc_lines = &lines[1..];
                    let omitted = pc_lines.len() - 4;
                    let trimmed = format!(
                        "{}\n{}\n{}\n... ({} lines omitted) ...\n{}\n{}",
                        header_line,
                        pc_lines[0],
                        pc_lines[1],
                        omitted,
                        pc_lines[pc_lines.len() - 2],
                        pc_lines[pc_lines.len() - 1],
                    );
                    s.text = trimmed;
                }
            }
        }

        // Stage 2: Reduce Related (priority 5) to 1 entry, then remove
        if estimate(&tiered) > max_tokens {
            for s in &mut tiered {
                if s.priority != 5
                    || !s.text.starts_with("## Related")
                {
                    continue;
                }
                let lines: Vec<&str> = s.text.lines().collect();
                // Keep header + first entry only
                if lines.len() > 2 {
                    s.text = format!("{}\n{}", lines[0], lines[1]);
                }
            }
        }
        if estimate(&tiered) > max_tokens {
            tiered.retain(|s| {
                s.priority != 5 || !s.text.starts_with("## Related")
            });
        }

        // Stage 3: Remove logic cluster member list from Domain
        // (keep label + count only)
        if estimate(&tiered) > max_tokens {
            for s in &mut tiered {
                if s.priority != 4 {
                    continue;
                }
                let new_lines: Vec<String> = s.text.lines()
                    .map(|line| {
                        if line.starts_with("Logic cluster: ") {
                            // Strip member list after the closing paren
                            if let Some(paren_end) =
                                line.find(" members)")
                            {
                                let end = paren_end + " members)".len();
                                line[..end].to_string()
                            } else {
                                line.to_string()
                            }
                        } else {
                            line.to_string()
                        }
                    })
                    .collect();
                s.text = new_lines.join("\n");
            }
        }

        // Stage 4: Drop entire Domain section (priority 4)
        if estimate(&tiered) > max_tokens {
            tiered.retain(|s| s.priority != 4);
        }

        // Stage 5: Reduce Structure (priority 2) to just name (kind)
        if estimate(&tiered) > max_tokens {
            if let Some(ent) = entity {
                for s in &mut tiered {
                    if s.priority != 2
                        || !s.text.starts_with("## Structure")
                    {
                        continue;
                    }
                    s.text = format!(
                        "## Structure\n{} ({:?})",
                        ent.name, ent.kind,
                    );
                }
            }
        }

        let text = tiered.iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join("\n\n");
        let token_estimate = text.len() / 4;

        Some(CompactContext {
            scope: scope.to_string(),
            text,
            token_estimate,
        })
    }

    /// Return a concept-annotated structural overview of every file whose
    /// path ends with `path`. Classes include nested method symbols.
    pub fn describe_file(&self, path: &str) -> Vec<DescribeFileResult> {
        // Collect all unique file paths from signatures and classes.
        let mut all_files: HashSet<PathBuf> = HashSet::new();
        for sig in &self.signatures {
            all_files.insert(sig.file.clone());
        }
        for cls in &self.classes {
            all_files.insert(cls.file.clone());
        }

        // Filter to files whose string path ends with the query.
        let matching_files: Vec<PathBuf> = all_files
            .into_iter()
            .filter(|f| f.display().to_string().ends_with(path))
            .collect();

        let mut results = Vec::new();

        for file in &matching_files {
            let file_classes: Vec<&ClassInfo> = self
                .classes
                .iter()
                .filter(|c| &c.file == file)
                .collect();

            let class_names: HashSet<&str> =
                file_classes.iter().map(|c| c.name.as_str()).collect();

            let mut symbols: Vec<FileSymbol> = Vec::new();

            // Build class symbols with nested methods.
            for cls in &file_classes {
                let entity = self.entities.values().find(|e| {
                    e.name == cls.name && e.file == cls.file
                });

                let concepts = self.resolve_concept_tags(entity);
                let role = entity.map(|e| e.semantic_role.clone());

                let methods: Vec<FileSymbol> = cls
                    .methods
                    .iter()
                    .map(|method_name| {
                        let method_sig =
                            self.signatures.iter().find(|s| {
                                s.file == cls.file
                                    && s.name == *method_name
                                    && s.scope.as_ref().is_some_and(
                                        |sc| sc.starts_with(&cls.name),
                                    )
                            });

                        let method_entity =
                            self.entities.values().find(|e| {
                                e.name == *method_name
                                    && e.file == cls.file
                                    && e.kind
                                        == crate::types::EntityKind::Method
                            });

                        let m_concepts =
                            self.resolve_concept_tags(method_entity);
                        let m_role =
                            method_entity.map(|e| e.semantic_role.clone());

                        let (params, return_type, line) =
                            match method_sig {
                                Some(sig) => (
                                    Self::format_params(&sig.params),
                                    sig.return_type.clone(),
                                    sig.line,
                                ),
                                None => (Vec::new(), None, cls.line),
                            };

                        FileSymbol {
                            name: method_name.clone(),
                            kind: SymbolKind::Method,
                            line,
                            concepts: m_concepts,
                            role: m_role,
                            params,
                            return_type,
                            bases: Vec::new(),
                            methods: Vec::new(),
                        }
                    })
                    .collect();

                symbols.push(FileSymbol {
                    name: cls.name.clone(),
                    kind: SymbolKind::Class,
                    line: cls.line,
                    concepts,
                    role,
                    params: Vec::new(),
                    return_type: None,
                    bases: cls.bases.clone(),
                    methods,
                });
            }

            // Standalone functions: not scoped to any class in this file.
            for sig in &self.signatures {
                if sig.file != *file {
                    continue;
                }
                let is_standalone =
                    sig.scope.as_ref().is_none_or(|s| {
                        !class_names.iter().any(|cn| s.starts_with(cn))
                    });
                if !is_standalone {
                    continue;
                }

                let entity = self.entities.values().find(|e| {
                    e.name == sig.name && e.file == sig.file
                });

                let concepts = self.resolve_concept_tags(entity);
                let role = entity.map(|e| e.semantic_role.clone());

                symbols.push(FileSymbol {
                    name: sig.name.clone(),
                    kind: SymbolKind::Function,
                    line: sig.line,
                    concepts,
                    role,
                    params: Self::format_params(&sig.params),
                    return_type: sig.return_type.clone(),
                    bases: Vec::new(),
                    methods: Vec::new(),
                });
            }

            symbols.sort_by_key(|s| s.line);

            results.push(DescribeFileResult {
                file: file.clone(),
                symbols,
            });
        }

        results.sort_by(|a, b| a.file.cmp(&b.file));
        results
    }

    /// Resolve entity concept tags to canonical concept names.
    fn resolve_concept_tags(&self, entity: Option<&Entity>) -> Vec<String> {
        entity
            .map(|e| {
                e.concept_tags
                    .iter()
                    .filter_map(|id| self.concepts.get(id))
                    .map(|c| c.canonical.clone())
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Format params as compact "name: Type" or just "name" strings.
    fn format_params(params: &[crate::types::Param]) -> Vec<String> {
        params
            .iter()
            .map(|p| match &p.type_annotation {
                Some(ty) => format!("{}: {}", p.name, ty),
                None => p.name.clone(),
            })
            .collect()
    }
}

/// Find identifiers in the corpus similar to the given one.
fn find_similar_identifiers(
    identifier: &str,
    id_freq: &HashMap<String, usize>,
) -> Vec<(String, usize)> {
    let id_lower = identifier.to_lowercase();
    let subtokens = split_identifier(identifier);

    let mut similar: Vec<(String, usize)> = id_freq
        .iter()
        .filter(|(name, _)| {
            if *name == &id_lower {
                return false;
            }
            // Share at least one subtoken
            let other_subtokens = split_identifier(name);
            subtokens
                .iter()
                .any(|st| other_subtokens.contains(st))
        })
        .map(|(name, &count)| (name.clone(), count))
        .collect();

    similar.sort_by(|a, b| b.1.cmp(&a.1));
    similar.truncate(10);
    similar
}

#[cfg(test)]
mod tests {
    use super::*;
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
            cluster_id: None,
            subconcepts: Vec::new(),
        }
    }

    fn make_test_graph() -> ConceptGraph {
        let analysis = AnalysisResult {
            concepts: vec![
                make_concept(
                    1,
                    "transform",
                    &[
                        "spatial_transform",
                        "apply_transform",
                        "transform",
                    ],
                ),
                make_concept(2, "spatial", &["spatial_transform"]),
                make_concept(3, "ndim", &["ndim", "ndim", "ndim"]),
                make_concept(
                    4,
                    "nb",
                    &[
                        "nb_features",
                        "nb_bins",
                        "nb_steps",
                        "nb_dims",
                    ],
                ),
                make_concept(
                    5,
                    "features",
                    &["nb_features", "features"],
                ),
            ],
            conventions: vec![Convention {
                pattern: PatternKind::Prefix("nb_".to_string()),
                entity_type: EntityType::Parameter,
                semantic_role: "count".to_string(),
                examples: vec![
                    "nb_features".into(),
                    "nb_bins".into(),
                    "nb_steps".into(),
                    "nb_dims".into(),
                ],
                frequency: 4,
            }],
            co_occurrence_matrix: vec![((1, 2), 1.0)],
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap()
    }

    #[test]
    fn test_query_concept_exact() {
        let graph = make_test_graph();
        let result =
            graph.query_concept("transform", &QueryConceptParams::default());
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result
            .variants
            .contains(&"spatial_transform".to_string()));
        assert!(
            result.variants.contains(&"apply_transform".to_string())
        );
    }

    #[test]
    fn test_check_naming_consistent() {
        let graph = make_test_graph();
        let result = graph.check_naming("nb_features");
        // nb_features follows the nb_ convention -> Consistent
        assert_eq!(result.verdict, Verdict::Consistent);
    }

    #[test]
    fn test_check_naming_inconsistent() {
        let graph = make_test_graph();
        let result = graph.check_naming("n_dims");
        assert_eq!(result.verdict, Verdict::Inconsistent);
        // Should suggest ndim or nb_dims
        assert!(result.suggestion.is_some());
    }

    #[test]
    fn test_list_concepts_sorted() {
        let graph = make_test_graph();
        let concepts = graph.list_concepts();
        assert!(!concepts.is_empty());
        // Should be sorted by occurrence count descending
        for i in 1..concepts.len() {
            assert!(
                concepts[i - 1].occurrences.len()
                    >= concepts[i].occurrences.len()
            );
        }
    }

    #[test]
    fn test_suggest_name_count() {
        let graph = make_test_graph();
        let suggestions = graph.suggest_name("count of features");
        assert!(!suggestions.is_empty());
        // Should suggest nb_features
        assert!(suggestions
            .iter()
            .any(|s| s.name.contains("nb_features")));
    }

    #[test]
    fn test_describe_symbol_not_found() {
        let graph = make_test_graph();
        assert!(graph.describe_symbol("nonexistent").is_none());
    }

    #[test]
    fn test_describe_symbol_function() {
        use crate::types::{Param, Signature};
        let mut analysis = AnalysisResult {
            concepts: vec![make_concept(
                1,
                "transform",
                &["spatial_transform"],
            )],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: vec![Signature {
                name: "spatial_transform".to_string(),
                params: vec![Param {
                    name: "vol".to_string(),
                    type_annotation: Some("Tensor".to_string()),
                    default: None,
                }],
                return_type: Some("Tensor".to_string()),
                decorators: Vec::new(),
                docstring_first_line: Some(
                    "Apply spatial transform.".to_string(),
                ),
                file: PathBuf::from("utils.py"),
                line: 10,
                scope: None,
                body: None,
            }],
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        analysis.call_sites.push(crate::types::CallSite {
            caller_scope: Some("register".to_string()),
            callee: "spatial_transform".to_string(),
            file: PathBuf::from("reg.py"),
            line: 20,
        });
        let graph = ConceptGraph::build(analysis, EmbeddingIndex::empty())
            .unwrap();
        let result = graph
            .describe_symbol("spatial_transform")
            .expect("should find symbol");
        assert!(result.signature.is_some());
        assert_eq!(result.callers.len(), 1);
        assert!(result.concepts.contains(&"transform".to_string()));
    }

    #[test]
    fn test_add_abbreviation_edges() {
        let analysis = AnalysisResult {
            concepts: vec![
                make_concept(1, "trf", &["trf"]),
                make_concept(2, "transform", &["transform"]),
                make_concept(3, "seg", &["seg"]),
                make_concept(4, "segmentation", &["segmentation"]),
            ],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        let mut graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty())
                .unwrap();

        assert!(graph
            .relationships
            .iter()
            .all(|r| r.kind != RelationshipKind::AbbreviationOf));

        graph.add_abbreviation_edges();

        let abbrevs: Vec<_> = graph
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::AbbreviationOf)
            .collect();
        assert_eq!(abbrevs.len(), 2);

        // trf → transform
        assert!(abbrevs.iter().any(|r| r.source == 1 && r.target == 2));
        // seg → segmentation
        assert!(abbrevs.iter().any(|r| r.source == 3 && r.target == 4));
    }

    #[test]
    fn test_list_entities_cluster_round_robin() {
        // Two clusters: transform-domain near [1,0,0], loss-domain near [0,1,0]
        // Embeddings go into EmbeddingIndex (matching production path),
        // NOT Concept.embedding.
        let concepts = vec![
            make_concept(
                1,
                "transform",
                &["transform", "apply_transform", "transform"],
            ),
            make_concept(2, "spatial", &["spatial_transform"]),
            make_concept(3, "loss", &["dice_loss"]),
        ];
        let entities = vec![
            // density = (3 + 1) / 2 = 2.0
            Entity {
                id: 10,
                name: "spatial_transform".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1, 2],
                semantic_role: "transform".to_string(),
                file: PathBuf::from("utils.py"),
                line: 10,
                signature_idx: None,
                class_info_idx: None,
            },
            // density = (3 + 1) / 3 = 1.33
            Entity {
                id: 11,
                name: "compute_spatial_transform".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1, 2],
                semantic_role: "transform".to_string(),
                file: PathBuf::from("utils.py"),
                line: 20,
                signature_idx: None,
                class_info_idx: None,
            },
            // density = 1 / 2 = 0.5
            Entity {
                id: 12,
                name: "dice_loss".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![3],
                semantic_role: "loss".to_string(),
                file: PathBuf::from("losses.py"),
                line: 1,
                signature_idx: None,
                class_info_idx: None,
            },
        ];

        let analysis = AnalysisResult {
            concepts,
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        let mut embeddings = EmbeddingIndex::empty();
        embeddings.insert_vector(1, vec![1.0, 0.0, 0.0]);
        embeddings.insert_vector(2, vec![0.95, 0.05, 0.0]);
        embeddings.insert_vector(3, vec![0.0, 1.0, 0.0]);
        let mut graph = ConceptGraph::build_with_entities(
            analysis,
            embeddings,
            entities,
            Vec::new(),
        )
        .unwrap();
        graph.cluster_and_add_similarity_edges(0.75);

        let results = graph.list_entities(None, None, None, 10);
        assert_eq!(results.len(), 3);

        // Transform cluster has 2 entities (larger), loss cluster has 1.
        // Round-robin: transform first (spatial_transform, highest density),
        // then loss (dice_loss), then back to transform
        // (compute_spatial_transform).
        assert_eq!(results[0].name, "spatial_transform");
        assert_eq!(results[1].name, "dice_loss");
        assert_eq!(results[2].name, "compute_spatial_transform");
    }

    #[test]
    fn test_list_entities_no_clusters_fallback() {
        // No embeddings → no clusters → sorted by domain density
        let analysis = AnalysisResult {
            concepts: vec![
                make_concept(
                    1,
                    "transform",
                    &["transform", "apply_transform", "transform"],
                ),
                make_concept(2, "spatial", &["spatial_transform"]),
                make_concept(3, "loss", &["dice_loss"]),
            ],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        let entities = vec![
            Entity {
                id: 10,
                name: "spatial_transform".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1, 2],
                semantic_role: "transform".to_string(),
                file: PathBuf::from("utils.py"),
                line: 10,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: 12,
                name: "dice_loss".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![3],
                semantic_role: "loss".to_string(),
                file: PathBuf::from("losses.py"),
                line: 1,
                signature_idx: None,
                class_info_idx: None,
            },
        ];
        let graph = ConceptGraph::build_with_entities(
            analysis,
            EmbeddingIndex::empty(),
            entities,
            Vec::new(),
        )
        .unwrap();

        let results = graph.list_entities(None, None, None, 10);
        assert_eq!(results.len(), 2);
        // Pure density sort: spatial_transform (2.0) > dice_loss (0.5)
        assert_eq!(results[0].name, "spatial_transform");
        assert_eq!(results[1].name, "dice_loss");
    }

    fn make_graph_with_embeddings(
        concepts: Vec<Concept>,
        embedding_pairs: Vec<(u64, Vec<f32>)>,
    ) -> ConceptGraph {
        let analysis = AnalysisResult {
            concepts,
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        let mut embeddings = EmbeddingIndex::empty();
        for (id, vec) in embedding_pairs {
            embeddings.insert_vector(id, vec);
        }
        ConceptGraph::build(analysis, embeddings).unwrap()
    }

    #[test]
    fn test_cluster_and_add_similarity_edges_basic() {
        // Group 1 (IDs 1,2,3): near [1,0,0], cosine sim ~0.99 within group.
        // Group 2 (IDs 4,5,6): near [0,1,0], cosine sim ~0.99 within group.
        // Cross-group sim ~0.0, so distance ~1.0 >> threshold distance 0.25.
        let concepts = vec![
            make_concept(1, "alpha", &["alpha"]),
            make_concept(2, "beta", &["beta"]),
            make_concept(3, "gamma", &["gamma"]),
            make_concept(4, "delta", &["delta"]),
            make_concept(5, "epsilon", &["epsilon"]),
            make_concept(6, "zeta", &["zeta"]),
        ];
        let embeddings = vec![
            (1, vec![1.0_f32, 0.0, 0.0]),
            (2, vec![0.98_f32, 0.1, 0.0]),
            (3, vec![0.95_f32, 0.15, 0.0]),
            (4, vec![0.0_f32, 1.0, 0.0]),
            (5, vec![0.1_f32, 0.98, 0.0]),
            (6, vec![0.15_f32, 0.95, 0.0]),
        ];
        let mut graph = make_graph_with_embeddings(concepts, embeddings);
        graph.cluster_and_add_similarity_edges(0.75);

        // All concepts must have a cluster_id assigned
        for id in 1u64..=6 {
            assert!(
                graph.concepts[&id].cluster_id.is_some(),
                "concept {id} missing cluster_id"
            );
        }

        let cid = |id: u64| graph.concepts[&id].cluster_id.unwrap();

        // Group 1 shares a cluster_id
        assert_eq!(cid(1), cid(2), "concepts 1 and 2 must share cluster");
        assert_eq!(cid(1), cid(3), "concepts 1 and 3 must share cluster");

        // Group 2 shares a cluster_id
        assert_eq!(cid(4), cid(5), "concepts 4 and 5 must share cluster");
        assert_eq!(cid(4), cid(6), "concepts 4 and 6 must share cluster");

        // The two groups have different cluster_ids
        assert_ne!(cid(1), cid(4), "groups 1 and 2 must have different cluster_ids");

        // SimilarTo edges must only connect within-group pairs
        let similar_edges: Vec<_> = graph
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::SimilarTo)
            .collect();
        assert!(!similar_edges.is_empty(), "expected some SimilarTo edges");

        let group1: HashSet<u64> = [1, 2, 3].into();
        let group2: HashSet<u64> = [4, 5, 6].into();
        for edge in &similar_edges {
            let src_in_g1 = group1.contains(&edge.source);
            let tgt_in_g1 = group1.contains(&edge.target);
            let src_in_g2 = group2.contains(&edge.source);
            let tgt_in_g2 = group2.contains(&edge.target);
            let within_group = (src_in_g1 && tgt_in_g1) || (src_in_g2 && tgt_in_g2);
            assert!(
                within_group,
                "cross-group SimilarTo edge: {} → {}",
                edge.source, edge.target
            );
        }
    }

    #[test]
    fn test_cluster_idempotent() {
        let concepts = vec![
            make_concept(1, "transform", &["transform"]),
            make_concept(2, "spatial", &["spatial"]),
            make_concept(3, "volume", &["volume"]),
        ];
        let embeddings = vec![
            (1, vec![1.0_f32, 0.0, 0.0]),
            (2, vec![0.98_f32, 0.1, 0.0]),
            (3, vec![0.0_f32, 1.0, 0.0]),
        ];
        let mut graph = make_graph_with_embeddings(concepts, embeddings);

        graph.cluster_and_add_similarity_edges(0.75);
        let edge_count_first = graph
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::SimilarTo)
            .count();
        let cluster_ids_first: Vec<Option<usize>> = (1u64..=3)
            .map(|id| graph.concepts[&id].cluster_id)
            .collect();

        graph.cluster_and_add_similarity_edges(0.75);
        let edge_count_second = graph
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::SimilarTo)
            .count();
        let cluster_ids_second: Vec<Option<usize>> = (1u64..=3)
            .map(|id| graph.concepts[&id].cluster_id)
            .collect();

        assert_eq!(
            edge_count_first, edge_count_second,
            "SimilarTo edge count must be identical on second call"
        );
        assert_eq!(
            cluster_ids_first, cluster_ids_second,
            "cluster_id assignments must be stable across calls"
        );
    }

    #[test]
    fn test_cluster_contrastive_interaction() {
        // "source" and "target" are a KNOWN_PAIR in add_contrastive_edges,
        // scoring 3 points → Contrastive edge emitted.
        // Give them similar embeddings so clustering puts them in the same
        // cluster and emits a SimilarTo edge. After add_contrastive_edges,
        // that SimilarTo edge must be suppressed.
        //
        // "alpha" and "beta" are also similar but NOT in KNOWN_PAIRS →
        // they keep their SimilarTo edge after add_contrastive_edges.
        let concepts = vec![
            make_concept(1, "source", &["source"]),
            make_concept(2, "target", &["target"]),
            make_concept(3, "alpha", &["alpha"]),
            make_concept(4, "beta", &["beta"]),
        ];
        let embeddings = vec![
            (1, vec![1.0_f32, 0.0, 0.0]),
            (2, vec![0.98_f32, 0.1, 0.0]),
            (3, vec![0.0_f32, 1.0, 0.0]),
            (4, vec![0.1_f32, 0.98, 0.0]),
        ];
        let mut graph = make_graph_with_embeddings(concepts, embeddings);

        graph.cluster_and_add_similarity_edges(0.75);

        // After clustering: source+target share a cluster, alpha+beta share a cluster.
        let cid = |id: u64| graph.concepts[&id].cluster_id.unwrap();
        assert_eq!(cid(1), cid(2), "source and target must share cluster");
        assert_eq!(cid(3), cid(4), "alpha and beta must share cluster");

        // SimilarTo edges must exist within each cluster before contrastive.
        let similar_before: HashSet<(u64, u64)> = graph
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::SimilarTo)
            .map(|r| (r.source.min(r.target), r.source.max(r.target)))
            .collect();
        assert!(similar_before.contains(&(1, 2)), "SimilarTo(source, target) expected before contrastive");
        assert!(similar_before.contains(&(3, 4)), "SimilarTo(alpha, beta) expected before contrastive");

        graph.add_contrastive_edges();

        let similar_after: HashSet<(u64, u64)> = graph
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::SimilarTo)
            .map(|r| (r.source.min(r.target), r.source.max(r.target)))
            .collect();

        // SimilarTo(source, target) must be suppressed — replaced by Contrastive.
        assert!(
            !similar_after.contains(&(1, 2)),
            "SimilarTo(source, target) must be removed after contrastive"
        );
        let has_contrastive = graph.relationships.iter().any(|r| {
            r.kind == RelationshipKind::Contrastive
                && ((r.source == 1 && r.target == 2)
                    || (r.source == 2 && r.target == 1))
        });
        assert!(has_contrastive, "Contrastive(source, target) expected");

        // SimilarTo(alpha, beta) must survive — no contrastive suppression.
        assert!(similar_after.contains(&(3, 4)), "SimilarTo(alpha, beta) must survive after contrastive");
    }

    #[test]
    fn test_confidence_bounded_zero_to_one() {
        let graph = make_test_graph();
        // Test all verdict paths produce bounded confidence
        let cases = [
            "nb_features", // Consistent (convention match)
            "ndim",        // Consistent (corpus presence)
            "n_dims",      // Inconsistent (prefix mismatch)
            "xyzzy_foo",   // Unknown
        ];
        for name in &cases {
            let result = graph.check_naming(name);
            assert!(
                (0.0..=1.0).contains(&result.confidence),
                "{}: confidence {} out of [0, 1]",
                name,
                result.confidence,
            );
        }
    }

    #[test]
    fn test_confidence_convention_match() {
        let graph = make_test_graph();
        let result = graph.check_naming("nb_features");
        assert_eq!(result.verdict, Verdict::Consistent);
        assert!(
            result.matching_convention.is_some(),
            "convention match path expected"
        );
        assert_eq!(result.confidence, 0.8);
    }

    #[test]
    fn test_confidence_corpus_presence() {
        // "ndim" is in corpus (3 occurrences) but matches no
        // convention, so it hits the corpus-presence path.
        let analysis = AnalysisResult {
            concepts: vec![make_concept(
                1,
                "ndim",
                &["ndim", "ndim", "ndim"],
            )],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        let graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty())
                .unwrap();
        let result = graph.check_naming("ndim");
        assert_eq!(result.verdict, Verdict::Consistent);
        assert!(
            result.matching_convention.is_none(),
            "should NOT match a convention"
        );
        assert_eq!(result.confidence, 0.6);
    }

    #[test]
    fn test_confidence_prefix_mismatch() {
        let graph = make_test_graph();
        let result = graph.check_naming("n_dims");
        assert_eq!(result.verdict, Verdict::Inconsistent);
        assert_eq!(result.confidence, 0.8);
    }

    #[test]
    fn test_confidence_unknown() {
        let graph = make_test_graph();
        let result = graph.check_naming("xyzzy_foo");
        assert_eq!(result.verdict, Verdict::Unknown);
        assert_eq!(result.confidence, 0.0);
    }

    #[test]
    fn test_confidence_frequency_ratio() {
        // "spatial_trf" appears once while "spatial_transform"
        // appears 8 times. Both share the "spatial" concept
        // subtoken, so the frequency-comparison path fires and
        // produces a dynamic confidence in (0, 1).
        let analysis = AnalysisResult {
            concepts: vec![make_concept(
                1,
                "spatial",
                &[
                    "spatial_transform",
                    "spatial_transform",
                    "spatial_transform",
                    "spatial_transform",
                    "spatial_transform",
                    "spatial_transform",
                    "spatial_transform",
                    "spatial_transform",
                    "spatial_trf",
                ],
            )],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        let graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty())
                .unwrap();
        let result = graph.check_naming("spatial_trf");
        assert_eq!(result.verdict, Verdict::Inconsistent);
        assert!(
            result.confidence > 0.0 && result.confidence < 1.0,
            "frequency confidence should be in (0, 1), \
             got {}",
            result.confidence,
        );
    }

    // --- vocabulary_health tests ---

    #[test]
    fn test_vocabulary_health_empty_graph() {
        use crate::config::HealthConfig;

        let graph = ConceptGraph::empty();
        let health =
            graph.vocabulary_health(&HealthConfig::default());

        // No concepts → coverage 0, consistency 1, cohesion 1
        assert_eq!(health.convention_coverage, 0.0);
        assert_eq!(health.consistency_ratio, 1.0);
        assert_eq!(health.cluster_cohesion, 1.0);
        assert!(health.overall > 0.0);
        assert!(health.top_inconsistencies.is_empty());
        assert!(health.uncovered_identifiers.is_empty());
    }

    #[test]
    fn test_vocabulary_health_with_conventions() {
        use crate::config::HealthConfig;

        let graph = make_test_graph();
        let health =
            graph.vocabulary_health(&HealthConfig::default());

        // The test graph has 5 concepts and one convention
        // (nb_ prefix). Concept "nb" has all identifiers
        // starting with "nb_", so it matches. That should
        // give non-zero convention_coverage.
        assert!(
            health.convention_coverage > 0.0,
            "at least one concept should be covered, got {}",
            health.convention_coverage,
        );
        assert!(
            health.convention_coverage <= 1.0,
            "coverage must be <= 1.0, got {}",
            health.convention_coverage,
        );

        // Consistency: concepts with uniform spellings should
        // score high, but some concepts in the test graph
        // (e.g. "features" with "nb_features" and "features")
        // have multiple spellings, pulling the ratio down.
        assert!(
            health.consistency_ratio > 0.0
                && health.consistency_ratio <= 1.0,
            "consistency_ratio should be in (0, 1], got {}",
            health.consistency_ratio,
        );

        // No embeddings → cohesion defaults to 1.0
        assert_eq!(health.cluster_cohesion, 1.0);

        // Overall should be between 0 and 1
        assert!(
            health.overall > 0.0 && health.overall <= 1.0,
            "overall should be in (0, 1], got {}",
            health.overall,
        );
    }

    #[test]
    fn test_vocabulary_health_inconsistencies() {
        use crate::config::HealthConfig;

        // Create a concept with mixed spellings
        let analysis = AnalysisResult {
            concepts: vec![make_concept(
                1,
                "transform",
                &[
                    "transform",
                    "transform",
                    "transform",
                    "trf",
                ],
            )],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        let graph = ConceptGraph::build(
            analysis,
            EmbeddingIndex::empty(),
        )
        .unwrap();
        let health =
            graph.vocabulary_health(&HealthConfig::default());

        assert_eq!(health.top_inconsistencies.len(), 1);
        assert_eq!(
            health.top_inconsistencies[0].dominant,
            "transform"
        );
        assert_eq!(
            health.top_inconsistencies[0].minority, "trf"
        );
        assert_eq!(
            health.top_inconsistencies[0].dominant_count, 3
        );
        assert_eq!(
            health.top_inconsistencies[0].minority_count, 1
        );
    }

    #[test]
    fn test_trace_concept_basic() {
        let concept = make_concept(
            1,
            "transform",
            &["apply_transform", "transform"],
        );
        let sigs = vec![
            Signature {
                name: "load_data".to_string(),
                params: Vec::new(),
                return_type: Some("Data".to_string()),
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("test.py"),
                line: 1,
                scope: None,
                body: None,
            },
            Signature {
                name: "apply_transform".to_string(),
                params: vec![Param {
                    name: "data".to_string(),
                    type_annotation: Some(
                        "Data".to_string(),
                    ),
                    default: None,
                }],
                return_type: Some(
                    "Transform".to_string(),
                ),
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("test.py"),
                line: 10,
                scope: None,
                body: None,
            },
            Signature {
                name: "save_result".to_string(),
                params: vec![Param {
                    name: "trf".to_string(),
                    type_annotation: Some(
                        "Transform".to_string(),
                    ),
                    default: None,
                }],
                return_type: None,
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("test.py"),
                line: 20,
                scope: None,
                body: None,
            },
        ];
        let entities = vec![
            Entity {
                id: 10,
                name: "apply_transform".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "transform".to_string(),
                file: PathBuf::from("test.py"),
                line: 10,
                signature_idx: Some(1),
                class_info_idx: None,
            },
            Entity {
                id: 11,
                name: "save_result".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "io".to_string(),
                file: PathBuf::from("test.py"),
                line: 20,
                signature_idx: Some(2),
                class_info_idx: None,
            },
        ];
        let call_sites = vec![
            CallSite {
                caller_scope: Some(
                    "load_data".to_string(),
                ),
                callee: "apply_transform".to_string(),
                file: PathBuf::from("test.py"),
                line: 5,
            },
            CallSite {
                caller_scope: Some(
                    "apply_transform".to_string(),
                ),
                callee: "save_result".to_string(),
                file: PathBuf::from("test.py"),
                line: 15,
            },
        ];
        let analysis = AnalysisResult {
            concepts: vec![concept],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: sigs,
            classes: Vec::new(),
            call_sites,
            nesting_trees: Vec::new(),
        };
        let graph = ConceptGraph::build_with_entities(
            analysis,
            EmbeddingIndex::empty(),
            entities,
            Vec::new(),
        )
        .unwrap();

        let result = graph.trace_concept("transform", 5);
        assert!(result.is_some(), "should find trace");
        let trace = result.unwrap();
        assert_eq!(trace.concept, "transform");
        assert!(
            !trace.call_chain.is_empty(),
            "call_chain must not be empty"
        );
        let chain_names: Vec<&str> = trace
            .call_chain
            .iter()
            .map(|n| n.entity_name.as_str())
            .collect();
        assert!(
            chain_names.contains(&"apply_transform"),
            "apply_transform must be in call_chain"
        );
        assert!(
            chain_names.contains(&"save_result"),
            "save_result must be in call_chain"
        );
        assert!(
            !trace.producers.is_empty(),
            "producers must not be empty"
        );
        assert!(
            !trace.consumers.is_empty(),
            "consumers must not be empty"
        );
    }

    #[test]
    fn test_vocabulary_health_uncovered() {
        use crate::config::HealthConfig;

        // No conventions → all identifiers are uncovered
        let analysis = AnalysisResult {
            concepts: vec![make_concept(
                1,
                "foo",
                &["foo_bar", "foo_baz"],
            )],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        let graph = ConceptGraph::build(
            analysis,
            EmbeddingIndex::empty(),
        )
        .unwrap();
        let health =
            graph.vocabulary_health(&HealthConfig::default());

        assert_eq!(health.uncovered_identifiers.len(), 2);
        assert_eq!(health.convention_coverage, 0.0);
    }

    #[test]
    fn test_trace_concept_bridge_entities() {
        let concept = make_concept(
            1,
            "transform",
            &["produce_transform", "consume_transform"],
        );
        let entities = vec![
            Entity {
                id: 10,
                name: "produce_transform".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "transform".to_string(),
                file: PathBuf::from("test.py"),
                line: 1,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: 11,
                name: "helper".to_string(),
                kind: EntityKind::Function,
                concept_tags: Vec::new(),
                semantic_role: "util".to_string(),
                file: PathBuf::from("test.py"),
                line: 10,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: 12,
                name: "consume_transform".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "transform".to_string(),
                file: PathBuf::from("test.py"),
                line: 20,
                signature_idx: None,
                class_info_idx: None,
            },
        ];
        let call_sites = vec![
            CallSite {
                caller_scope: Some(
                    "produce_transform".to_string(),
                ),
                callee: "helper".to_string(),
                file: PathBuf::from("test.py"),
                line: 5,
            },
            CallSite {
                caller_scope: Some(
                    "helper".to_string(),
                ),
                callee: "consume_transform".to_string(),
                file: PathBuf::from("test.py"),
                line: 15,
            },
        ];
        let analysis = AnalysisResult {
            concepts: vec![concept],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites,
            nesting_trees: Vec::new(),
        };
        let graph = ConceptGraph::build_with_entities(
            analysis,
            EmbeddingIndex::empty(),
            entities,
            Vec::new(),
        )
        .unwrap();

        let result = graph.trace_concept("transform", 5);
        assert!(result.is_some(), "should find trace");
        let trace = result.unwrap();
        let bridge = trace.call_chain.iter().find(|n| {
            n.entity_name == "helper"
        });
        assert!(
            bridge.is_some(),
            "helper must appear in call_chain"
        );
        assert_eq!(
            bridge.unwrap().role,
            TraceRole::Bridge,
            "helper must have Bridge role"
        );
    }

    #[test]
    fn test_trace_concept_not_found() {
        let graph = make_test_graph();
        let result = graph.trace_concept("nonexistent", 5);
        assert!(
            result.is_none(),
            "nonexistent concept should return None"
        );
    }

    #[test]
    fn test_trace_concept_no_call_sites__orphan_seeds__appear_in_chain_with_empty_edges() {
        /// Entities tagged with the concept but present in zero call_sites must
        /// surface as orphan seeds in `call_chain` while `edges` stays empty.
        ///
        /// Parameters
        /// ----------
        /// None — self-contained fixture.
        ///
        /// Expected
        /// --------
        /// `call_chain` contains both entities; `edges` is empty; `producers`
        /// and `consumers` are non-empty (each entity falls into `Both` because
        /// neither has a signature that matches the concept subtokens exactly).
        let concept = make_concept(1, "loss", &["dice_loss", "ncc_loss"]);
        let entities = vec![
            Entity {
                id: 20,
                name: "dice_loss".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "loss".to_string(),
                file: PathBuf::from("losses.py"),
                line: 1,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: 21,
                name: "ncc_loss".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "loss".to_string(),
                file: PathBuf::from("losses.py"),
                line: 10,
                signature_idx: None,
                class_info_idx: None,
            },
        ];
        let analysis = AnalysisResult {
            concepts: vec![concept],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(), // zero call sites
            nesting_trees: Vec::new(),
        };
        let graph = ConceptGraph::build_with_entities(
            analysis,
            EmbeddingIndex::empty(),
            entities,
            Vec::new(),
        )
        .unwrap();

        let trace = graph
            .trace_concept("loss", 5)
            .expect("concept exists — must return Some");

        let chain_names: Vec<&str> = trace
            .call_chain
            .iter()
            .map(|n| n.entity_name.as_str())
            .collect();
        assert!(
            chain_names.contains(&"dice_loss"),
            "dice_loss must appear as orphan seed in call_chain"
        );
        assert!(
            chain_names.contains(&"ncc_loss"),
            "ncc_loss must appear as orphan seed in call_chain"
        );
        assert!(
            trace.edges.is_empty(),
            "no call_sites means edges must be empty"
        );
        // Both entities have no signature, so they fall to TraceRole::Both —
        // they count as both producers AND consumers.
        assert!(
            !trace.producers.is_empty(),
            "producers must be non-empty even without call_sites"
        );
        assert!(
            !trace.consumers.is_empty(),
            "consumers must be non-empty even without call_sites"
        );
    }

    #[test]
    fn test_trace_concept_cyclic_calls__toposort_fallback__all_nodes_in_chain() {
        /// A→B→C→A forms a cycle that breaks toposort. The BFS fallback must
        /// include all three concept-tagged entities in `call_chain`.
        ///
        /// Parameters
        /// ----------
        /// None — self-contained fixture.
        ///
        /// Expected
        /// --------
        /// `call_chain` contains all three entity names; no panic from the
        /// toposort failure path.
        let concept = make_concept(
            1,
            "segment",
            &["seg_a", "seg_b", "seg_c"],
        );
        let entities = vec![
            Entity {
                id: 30,
                name: "seg_a".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "segment".to_string(),
                file: PathBuf::from("seg.py"),
                line: 1,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: 31,
                name: "seg_b".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "segment".to_string(),
                file: PathBuf::from("seg.py"),
                line: 10,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: 32,
                name: "seg_c".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "segment".to_string(),
                file: PathBuf::from("seg.py"),
                line: 20,
                signature_idx: None,
                class_info_idx: None,
            },
        ];
        // A→B→C→A cycle
        let call_sites = vec![
            CallSite {
                caller_scope: Some("seg_a".to_string()),
                callee: "seg_b".to_string(),
                file: PathBuf::from("seg.py"),
                line: 5,
            },
            CallSite {
                caller_scope: Some("seg_b".to_string()),
                callee: "seg_c".to_string(),
                file: PathBuf::from("seg.py"),
                line: 15,
            },
            CallSite {
                caller_scope: Some("seg_c".to_string()),
                callee: "seg_a".to_string(),
                file: PathBuf::from("seg.py"),
                line: 25,
            },
        ];
        let analysis = AnalysisResult {
            concepts: vec![concept],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites,
            nesting_trees: Vec::new(),
        };
        let graph = ConceptGraph::build_with_entities(
            analysis,
            EmbeddingIndex::empty(),
            entities,
            Vec::new(),
        )
        .unwrap();

        let trace = graph
            .trace_concept("segment", 5)
            .expect("concept exists — must return Some");

        let chain_names: Vec<&str> = trace
            .call_chain
            .iter()
            .map(|n| n.entity_name.as_str())
            .collect();
        assert!(
            chain_names.contains(&"seg_a"),
            "seg_a must be in call_chain after cycle fallback"
        );
        assert!(
            chain_names.contains(&"seg_b"),
            "seg_b must be in call_chain after cycle fallback"
        );
        assert!(
            chain_names.contains(&"seg_c"),
            "seg_c must be in call_chain after cycle fallback"
        );
    }

    #[test]
    fn test_trace_concept_producer_consumer_classification__roles_match_signature() {
        /// Verifies that TraceRole assignment follows signature shape:
        /// - return_type contains concept subtoken → Producer
        /// - param name/type contains concept subtoken → Consumer
        /// - both match → Both
        ///
        /// Parameters
        /// ----------
        /// None — self-contained fixture.
        ///
        /// Expected
        /// --------
        /// `make_vol` has role Producer; `apply_vol` has role Consumer;
        /// `process_vol` has role Both.
        let concept = make_concept(1, "vol", &["make_vol", "apply_vol", "process_vol"]);
        // Signatures placed at indices 0, 1, 2 in the analysis vector.
        let sigs = vec![
            Signature {
                // sig index 0: returns "Vol" → Producer
                name: "make_vol".to_string(),
                params: Vec::new(),
                return_type: Some("Vol".to_string()),
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("vol.py"),
                line: 1,
                scope: None,
                body: None,
            },
            Signature {
                // sig index 1: param type "Vol" → Consumer
                name: "apply_vol".to_string(),
                params: vec![Param {
                    name: "src".to_string(),
                    type_annotation: Some("Vol".to_string()),
                    default: None,
                }],
                return_type: None,
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("vol.py"),
                line: 10,
                scope: None,
                body: None,
            },
            Signature {
                // sig index 2: returns "Vol" AND param named "vol" → Both
                name: "process_vol".to_string(),
                params: vec![Param {
                    name: "vol".to_string(),
                    type_annotation: None,
                    default: None,
                }],
                return_type: Some("Vol".to_string()),
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("vol.py"),
                line: 20,
                scope: None,
                body: None,
            },
        ];
        let entities = vec![
            Entity {
                id: 40,
                name: "make_vol".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "vol".to_string(),
                file: PathBuf::from("vol.py"),
                line: 1,
                signature_idx: Some(0),
                class_info_idx: None,
            },
            Entity {
                id: 41,
                name: "apply_vol".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "vol".to_string(),
                file: PathBuf::from("vol.py"),
                line: 10,
                signature_idx: Some(1),
                class_info_idx: None,
            },
            Entity {
                id: 42,
                name: "process_vol".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "vol".to_string(),
                file: PathBuf::from("vol.py"),
                line: 20,
                signature_idx: Some(2),
                class_info_idx: None,
            },
        ];
        let analysis = AnalysisResult {
            concepts: vec![concept],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: sigs,
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        let graph = ConceptGraph::build_with_entities(
            analysis,
            EmbeddingIndex::empty(),
            entities,
            Vec::new(),
        )
        .unwrap();

        let trace = graph
            .trace_concept("vol", 5)
            .expect("concept exists — must return Some");

        let role_of = |name: &str| -> TraceRole {
            trace
                .call_chain
                .iter()
                .find(|n| n.entity_name == name)
                .unwrap_or_else(|| panic!("{name} must be in call_chain"))
                .role
                .clone()
        };

        assert_eq!(
            role_of("make_vol"),
            TraceRole::Producer,
            "make_vol returns Vol → Producer"
        );
        assert_eq!(
            role_of("apply_vol"),
            TraceRole::Consumer,
            "apply_vol takes Vol param → Consumer"
        );
        assert_eq!(
            role_of("process_vol"),
            TraceRole::Both,
            "process_vol returns Vol AND takes vol param → Both"
        );
    }

    fn make_entity(id: u64, name: &str) -> Entity {
        Entity {
            id,
            name: name.to_string(),
            kind: EntityKind::Function,
            concept_tags: Vec::new(),
            semantic_role: "utility".to_string(),
            file: PathBuf::from("test.py"),
            line: 1,
            signature_idx: None,
            class_info_idx: None,
        }
    }

    fn make_graph_with_l4(
        entities: Vec<Entity>,
        pseudocode: HashMap<u64, Pseudocode>,
        centrality: HashMap<u64, CentralityScore>,
    ) -> ConceptGraph {
        let mut graph = ConceptGraph::empty();
        for entity in entities {
            graph.entities.insert(entity.id, entity);
        }
        graph.pseudocode = pseudocode;
        graph.centrality = centrality;
        graph
    }

    // --- describe_logic tests ---

    #[test]
    fn test_describe_logic_basic_flow() {
        let entity = make_entity(42, "process_data");
        let pc = Pseudocode {
            entity_id: 42,
            steps: vec![PseudocodeStep::Call {
                callee: "validate".into(),
                args: vec![],
            }],
            body_hash: 0,
            omitted_count: 0,
        };
        let centrality_score = CentralityScore {
            entity_id: 42,
            in_degree: 3,
            out_degree: 1,
            pagerank: 0.42,
        };
        let mut pseudocode = HashMap::new();
        pseudocode.insert(42, pc);
        let mut centrality = HashMap::new();
        centrality.insert(42, centrality_score);

        let graph = make_graph_with_l4(vec![entity], pseudocode, centrality);
        let result = graph.describe_logic("process_data");

        assert!(result.is_some());
        let desc = result.unwrap();
        assert_eq!(desc.entity.name, "process_data");
        assert!(!desc.pseudocode_text.is_empty());
        assert!((desc.centrality.pagerank - 0.42).abs() < 1e-6);
        assert_eq!(desc.centrality.in_degree, 3);
        assert_eq!(desc.centrality.out_degree, 1);
    }

    #[test]
    fn test_describe_logic_missing_pseudocode_returns_empty_text() {
        let entity = make_entity(10, "my_func");
        let graph = make_graph_with_l4(vec![entity], HashMap::new(), HashMap::new());

        let result = graph.describe_logic("my_func");
        assert!(result.is_some());
        let desc = result.unwrap();
        assert_eq!(desc.pseudocode_text, "");
        // Centrality defaults to zero when absent
        assert_eq!(desc.centrality.in_degree, 0);
        assert_eq!(desc.centrality.pagerank, 0.0);
    }

    #[test]
    fn test_describe_logic_not_found_returns_none() {
        let graph = ConceptGraph::empty();
        assert!(graph.describe_logic("nonexistent_entity").is_none());
    }

    // --- find_similar_logic tests ---

    #[test]
    fn test_find_similar_logic_basic() {
        let entities = vec![
            make_entity(1, "entity_a"),
            make_entity(2, "entity_b"),
            make_entity(3, "entity_c"),
        ];
        let mut graph = make_graph_with_l4(entities, HashMap::new(), HashMap::new());
        graph.logic_index.insert_vector(1, vec![1.0, 0.0, 0.0]);
        graph.logic_index.insert_vector(2, vec![0.9, 0.1, 0.0]);
        graph.logic_index.insert_vector(3, vec![0.0, 1.0, 0.0]);

        let result = graph.find_similar_logic("entity_a", 2);
        assert!(result.is_some());
        let sim = result.unwrap();
        assert_eq!(sim.query_entity.name, "entity_a");
        assert_eq!(sim.similar.len(), 2);
        // entity_b is more similar to entity_a than entity_c
        assert_eq!(sim.similar[0].0.name, "entity_b");
        assert!(sim.similar[0].1 > sim.similar[1].1);
    }

    #[test]
    fn test_find_similar_logic_no_vector_returns_empty_similar() {
        let entity = make_entity(5, "bare_entity");
        let graph = make_graph_with_l4(vec![entity], HashMap::new(), HashMap::new());
        // No vectors inserted into logic_index

        let result = graph.find_similar_logic("bare_entity", 5);
        assert!(result.is_some());
        assert!(result.unwrap().similar.is_empty());
    }

    #[test]
    fn test_find_similar_logic_not_found_returns_none() {
        let graph = ConceptGraph::empty();
        assert!(graph.find_similar_logic("ghost", 3).is_none());
    }

    // --- compact_context tests ---

    #[test]
    fn test_compact_context_entity_scope_has_structure_and_behavior() {
        let entity = make_entity(7, "transform_vol");
        let pc = Pseudocode {
            entity_id: 7,
            steps: vec![PseudocodeStep::Return {
                value: Some("result".into()),
            }],
            body_hash: 0,
            omitted_count: 0,
        };
        let cs = CentralityScore {
            entity_id: 7,
            in_degree: 1,
            out_degree: 2,
            pagerank: 0.1,
        };
        let mut pseudocode = HashMap::new();
        pseudocode.insert(7, pc);
        let mut centrality = HashMap::new();
        centrality.insert(7, cs);

        let graph = make_graph_with_l4(vec![entity], pseudocode, centrality);
        let result = graph.compact_context("transform_vol", 500);

        assert!(result.is_some());
        let ctx = result.unwrap();
        assert!(ctx.text.contains("## Structure"));
        assert!(ctx.text.contains("## Behavior"));
        assert_eq!(ctx.scope, "transform_vol");
    }

    #[test]
    fn test_compact_context_concept_scope_lists_entities() {
        let mut graph = ConceptGraph::empty();

        // Add a concept
        let concept = make_concept(99, "segment", &["segment"]);
        graph.concepts.insert(99, concept);

        // Add entities tagged with that concept
        let mut e1 = make_entity(100, "apply_segment");
        e1.concept_tags = vec![99];
        let mut e2 = make_entity(101, "segment_volume");
        e2.concept_tags = vec![99];
        graph.entities.insert(100, e1);
        graph.entities.insert(101, e2);

        let result = graph.compact_context("segment", 500);
        assert!(result.is_some());
        let ctx = result.unwrap();
        // Concept scope renders entity list
        assert!(ctx.text.contains("## Entities"));
        assert!(ctx.scope == "segment");
    }

    #[test]
    fn test_compact_context_not_found_returns_none() {
        let graph = ConceptGraph::empty();
        assert!(graph.compact_context("no_such_scope", 500).is_none());
    }

    #[test]
    fn test_l4_tools_return_none_on_empty_graph() {
        let graph = ConceptGraph::empty();
        assert!(graph.describe_logic("anything").is_none());
        assert!(graph.find_similar_logic("anything", 5).is_none());
        assert!(graph.compact_context("anything", 500).is_none());
    }

    #[test]
    fn test_compact_context_tiered_truncation_drops_low_priority_sections() {
        // Entity with pseudocode (many steps), concept tags, logic cluster,
        // centrality, and logic embeddings for "similar" entities — build a
        // graph whose full context exceeds a tiny token budget so every
        // truncation stage fires.
        let steps: Vec<PseudocodeStep> = (0..10)
            .map(|i| PseudocodeStep::Call {
                callee: format!("step_{}", i),
                args: vec![],
            })
            .collect();
        let pc = Pseudocode {
            entity_id: 1,
            steps,
            body_hash: 0,
            omitted_count: 0,
        };
        let cs = CentralityScore {
            entity_id: 1,
            in_degree: 5,
            out_degree: 3,
            pagerank: 0.5,
        };

        let mut entity = make_entity(1, "fn_a");
        let concept = make_concept(10, "volume", &["volume"]);
        entity.concept_tags = vec![10];

        // Logic cluster containing the entity and a peer
        let cluster = LogicCluster {
            id: 0,
            entity_ids: vec![1, 2],
            centroid: vec![],
            behavioral_label: Some("transform".into()),
        };

        let mut pseudocode = HashMap::new();
        pseudocode.insert(1, pc);
        let mut centrality = HashMap::new();
        centrality.insert(1, cs);

        let peer = make_entity(2, "fn_b");
        let mut graph = make_graph_with_l4(
            vec![entity, peer],
            pseudocode,
            centrality,
        );
        graph.concepts.insert(10, concept);
        graph.logic_clusters.push(cluster);

        // Insert logic vectors so ## Related fires before truncation
        graph.logic_index.insert_vector(1, vec![1.0, 0.0]);
        graph.logic_index.insert_vector(2, vec![0.9, 0.1]);

        // Budget of 10 tokens forces maximum truncation.
        // The minimum achievable output is header + minimal structure
        // (~12-15 tokens), so we assert that low-priority sections are gone.
        let result = graph.compact_context("fn_a", 10);
        assert!(result.is_some());
        let ctx = result.unwrap();

        // Entity name must survive (Structure section kept at minimum)
        assert!(ctx.text.contains("fn_a"), "entity name must survive truncation");

        // Low-priority sections must be dropped under tight budget
        assert!(!ctx.text.contains("## Related"), "## Related must be dropped");
        assert!(!ctx.text.contains("## Domain"), "## Domain must be dropped");

        // Full context (no truncation) would carry header + structure +
        // 10-step behavior + domain + related — well over 100 tokens.
        // With budget=10 every stage fires; the survivor is header +
        // minimal structure + trimmed behavior (first 2 + last 2 steps).
        // That is ≤ 50 tokens, which proves the heavy sections were discarded.
        let full_ctx = graph.compact_context("fn_a", 10_000).unwrap();
        assert!(
            ctx.token_estimate < full_ctx.token_estimate,
            "truncated output ({} tokens) must be smaller than full output ({} tokens)",
            ctx.token_estimate,
            full_ctx.token_estimate,
        );
        assert!(
            ctx.token_estimate <= 50,
            "expected truncated output ≤50 tokens, got {}",
            ctx.token_estimate
        );
    }

    #[test]
    fn test_compact_context_resolves_file_path_scope() {
        // make_entity uses `file: PathBuf::from("test.py")` by default.
        // compact_context should resolve "test.py" to that entity.
        let entity = make_entity(20, "my_func");
        let graph = make_graph_with_l4(
            vec![entity],
            HashMap::new(),
            HashMap::new(),
        );

        let result = graph.compact_context("test.py", 500);
        assert!(result.is_some(), "file path scope must resolve to an entity");
        let ctx = result.unwrap();
        assert_eq!(ctx.scope, "test.py");
        assert!(ctx.text.contains("my_func"), "resolved entity name must appear");
    }

    #[test]
    fn test_compact_context_shows_dependency_names() {
        // Entity A (id=30) uses entity B (id=31).
        // compact_context("entity_a", 500) must contain "depends on: entity_b".
        let entity_a = make_entity(30, "entity_a");
        let entity_b = make_entity(31, "entity_b");
        let rel = Relationship {
            source: 30,
            target: 31,
            kind: RelationshipKind::Uses,
            weight: 1.0,
        };

        let mut graph = make_graph_with_l4(
            vec![entity_a, entity_b],
            HashMap::new(),
            HashMap::new(),
        );
        graph.relationships.push(rel);

        let result = graph.compact_context("entity_a", 500);
        assert!(result.is_some());
        let ctx = result.unwrap();
        assert!(
            ctx.text.contains("depends on: entity_b"),
            "outgoing Uses edge must appear as 'depends on: entity_b', got:\n{}",
            ctx.text
        );
    }

    /// Build a graph with classes, methods, and standalone functions
    /// spread across two files for describe_file tests.
    fn make_describe_file_graph() -> ConceptGraph {
        let concepts = vec![
            make_concept(1, "transform", &["spatial_transform"]),
            make_concept(2, "spatial", &["spatial_transform"]),
            make_concept(3, "loss", &["dice_loss"]),
            make_concept(4, "network", &["VoxNet"]),
        ];

        let signatures = vec![
            Signature {
                name: "forward".to_string(),
                params: vec![Param {
                    name: "x".to_string(),
                    type_annotation: Some("Tensor".to_string()),
                    default: None,
                }],
                return_type: Some("Tensor".to_string()),
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("src/networks.py"),
                line: 15,
                scope: Some("VoxNet".to_string()),
                body: None,
            },
            Signature {
                name: "init_weights".to_string(),
                params: vec![],
                return_type: None,
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("src/networks.py"),
                line: 25,
                scope: Some("VoxNet".to_string()),
                body: None,
            },
            Signature {
                name: "spatial_transform".to_string(),
                params: vec![
                    Param {
                        name: "vol".to_string(),
                        type_annotation: Some("Tensor".to_string()),
                        default: None,
                    },
                    Param {
                        name: "trf".to_string(),
                        type_annotation: None,
                        default: None,
                    },
                ],
                return_type: Some("Tensor".to_string()),
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("src/networks.py"),
                line: 40,
                scope: None,
                body: None,
            },
            Signature {
                name: "dice_loss".to_string(),
                params: vec![Param {
                    name: "pred".to_string(),
                    type_annotation: Some("Tensor".to_string()),
                    default: None,
                }],
                return_type: Some("float".to_string()),
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("src/losses.py"),
                line: 5,
                scope: None,
                body: None,
            },
        ];

        let classes = vec![ClassInfo {
            name: "VoxNet".to_string(),
            bases: vec!["nn.Module".to_string()],
            methods: vec![
                "forward".to_string(),
                "init_weights".to_string(),
            ],
            attributes: Vec::new(),
            docstring_first_line: Some("Voxel network.".to_string()),
            file: PathBuf::from("src/networks.py"),
            line: 10,
        }];

        let entities = vec![
            Entity {
                id: 100,
                name: "VoxNet".to_string(),
                kind: EntityKind::Class,
                concept_tags: vec![4],
                semantic_role: "network".to_string(),
                file: PathBuf::from("src/networks.py"),
                line: 10,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: 101,
                name: "forward".to_string(),
                kind: EntityKind::Method,
                concept_tags: vec![],
                semantic_role: "entry point".to_string(),
                file: PathBuf::from("src/networks.py"),
                line: 15,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: 102,
                name: "spatial_transform".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1, 2],
                semantic_role: "transform".to_string(),
                file: PathBuf::from("src/networks.py"),
                line: 40,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: 103,
                name: "dice_loss".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![3],
                semantic_role: "loss".to_string(),
                file: PathBuf::from("src/losses.py"),
                line: 5,
                signature_idx: None,
                class_info_idx: None,
            },
        ];

        let analysis = AnalysisResult {
            concepts,
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures,
            classes,
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        ConceptGraph::build_with_entities(
            analysis,
            EmbeddingIndex::empty(),
            entities,
            Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn test_describe_file_exact_match() {
        let graph = make_describe_file_graph();
        let results = graph.describe_file("src/losses.py");
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].file,
            PathBuf::from("src/losses.py")
        );
        assert_eq!(results[0].symbols.len(), 1);
        let sym = &results[0].symbols[0];
        assert_eq!(sym.name, "dice_loss");
        assert!(matches!(sym.kind, SymbolKind::Function));
        assert_eq!(sym.concepts, vec!["loss".to_string()]);
        assert_eq!(sym.params, vec!["pred: Tensor".to_string()]);
        assert_eq!(sym.return_type.as_deref(), Some("float"));
    }

    #[test]
    fn test_describe_file_partial_match() {
        let graph = make_describe_file_graph();
        let results = graph.describe_file("networks.py");
        assert_eq!(results.len(), 1);
        assert_eq!(
            results[0].file,
            PathBuf::from("src/networks.py")
        );
        // Should have VoxNet class + spatial_transform function
        assert_eq!(results[0].symbols.len(), 2);
    }

    #[test]
    fn test_describe_file_no_match() {
        let graph = make_describe_file_graph();
        let results = graph.describe_file("nonexistent.py");
        assert!(results.is_empty());
    }

    #[test]
    fn test_describe_file_classes_with_methods() {
        let graph = make_describe_file_graph();
        let results = graph.describe_file("src/networks.py");
        assert_eq!(results.len(), 1);

        let symbols = &results[0].symbols;
        // Sorted by line: VoxNet (10), spatial_transform (40)
        let class_sym = symbols
            .iter()
            .find(|s| s.name == "VoxNet")
            .expect("VoxNet must be present");
        assert!(matches!(class_sym.kind, SymbolKind::Class));
        assert_eq!(class_sym.line, 10);
        assert_eq!(class_sym.concepts, vec!["network".to_string()]);
        assert_eq!(
            class_sym.bases,
            vec!["nn.Module".to_string()]
        );
        assert_eq!(class_sym.role.as_deref(), Some("network"));

        // Methods nested under class
        assert_eq!(class_sym.methods.len(), 2);
        let forward = class_sym
            .methods
            .iter()
            .find(|m| m.name == "forward")
            .expect("forward must be a method");
        assert!(matches!(forward.kind, SymbolKind::Method));
        assert_eq!(forward.params, vec!["x: Tensor".to_string()]);
        assert_eq!(forward.return_type.as_deref(), Some("Tensor"));
        assert_eq!(forward.role.as_deref(), Some("entry point"));

        // Standalone function
        let st = symbols
            .iter()
            .find(|s| s.name == "spatial_transform")
            .expect("spatial_transform must be present");
        assert!(matches!(st.kind, SymbolKind::Function));
        assert_eq!(st.line, 40);
        assert!(st.concepts.contains(&"transform".to_string()));
        assert!(st.concepts.contains(&"spatial".to_string()));
        assert_eq!(
            st.params,
            vec!["vol: Tensor".to_string(), "trf".to_string()]
        );
    }

    fn make_concept_map_graph() -> ConceptGraph {
        let analysis = AnalysisResult {
            concepts: vec![
                make_concept(1, "transform", &["spatial_transform"]),
                make_concept(2, "spatial", &["spatial_transform"]),
                make_concept(3, "loss", &["dice_loss", "ncc_loss"]),
            ],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
        };
        let entities = vec![
            Entity {
                id: Entity::hash_id(
                    "SpatialTransformer",
                    std::path::Path::new("proj/nn/transform.py"),
                    10,
                ),
                name: "SpatialTransformer".to_string(),
                kind: EntityKind::Class,
                concept_tags: vec![1, 2],
                semantic_role: "module".to_string(),
                file: PathBuf::from("proj/nn/transform.py"),
                line: 10,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: Entity::hash_id(
                    "apply_transform",
                    std::path::Path::new("proj/nn/transform.py"),
                    50,
                ),
                name: "apply_transform".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "utility".to_string(),
                file: PathBuf::from("proj/nn/transform.py"),
                line: 50,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: Entity::hash_id(
                    "DiceLoss",
                    std::path::Path::new("proj/losses/dice.py"),
                    5,
                ),
                name: "DiceLoss".to_string(),
                kind: EntityKind::Class,
                concept_tags: vec![3],
                semantic_role: "module".to_string(),
                file: PathBuf::from("proj/losses/dice.py"),
                line: 5,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: Entity::hash_id(
                    "NccLoss",
                    std::path::Path::new("proj/losses/ncc.py"),
                    5,
                ),
                name: "NccLoss".to_string(),
                kind: EntityKind::Class,
                concept_tags: vec![3],
                semantic_role: "module".to_string(),
                file: PathBuf::from("proj/losses/ncc.py"),
                line: 5,
                signature_idx: None,
                class_info_idx: None,
            },
        ];
        ConceptGraph::build_with_entities(
            analysis,
            EmbeddingIndex::empty(),
            entities,
            Vec::new(),
        )
        .unwrap()
    }

    #[test]
    fn test_concept_map_returns_modules() {
        let graph = make_concept_map_graph();
        let map = graph.concept_map();
        assert!(!map.modules.is_empty(), "concept_map must return modules");
        assert_eq!(map.total_entities, 4);
        assert_eq!(map.total_concepts, 3);
    }

    #[test]
    fn test_concept_map_sorted_by_entity_count() {
        let graph = make_concept_map_graph();
        let map = graph.concept_map();
        for i in 1..map.modules.len() {
            assert!(
                map.modules[i - 1].nb_entities
                    >= map.modules[i].nb_entities,
                "modules must be sorted by entity count descending"
            );
        }
    }

    #[test]
    fn test_concept_map_dominant_concepts_present() {
        let graph = make_concept_map_graph();
        let map = graph.concept_map();
        // nn/transform.py has transform and spatial concepts
        let nn_module = map
            .modules
            .iter()
            .find(|m| m.path.contains("nn"))
            .expect("nn module must exist");
        assert!(
            nn_module
                .dominant_concepts
                .iter()
                .any(|c| c == "transform"),
            "nn module must have 'transform' as dominant concept"
        );
        // losses dir has loss concept
        let losses_module = map
            .modules
            .iter()
            .find(|m| m.path.contains("losses"))
            .expect("losses module must exist");
        assert!(
            losses_module
                .dominant_concepts
                .iter()
                .any(|c| c == "loss"),
            "losses module must have 'loss' as dominant concept"
        );
    }

    #[test]
    fn test_concept_map_empty_graph() {
        let graph = ConceptGraph::empty();
        let map = graph.concept_map();
        assert!(map.modules.is_empty());
        assert_eq!(map.total_entities, 0);
        assert_eq!(map.total_concepts, 0);
    }

    #[test]
    fn test_concept_map_entity_counts() {
        let graph = make_concept_map_graph();
        let map = graph.concept_map();
        let nn_module = map
            .modules
            .iter()
            .find(|m| m.path.contains("nn"))
            .expect("nn module must exist");
        assert_eq!(nn_module.nb_classes, 1);
        assert_eq!(nn_module.nb_functions, 1);
        assert_eq!(nn_module.nb_entities, 2);
    }

    #[test]
    fn test_nesting_tree_lookup() {
        use crate::types::{FileNestingTree, NestingKind, NestingNode};

        let mut graph = make_test_graph();
        graph.nesting_trees = vec![FileNestingTree {
            file: PathBuf::from("src/model.py"),
            root: NestingNode {
                name: "model.py".to_string(),
                kind: NestingKind::Module,
                line: 0,
                children: vec![
                    NestingNode {
                        name: "train".to_string(),
                        kind: NestingKind::Function,
                        line: 1,
                        children: Vec::new(),
                    },
                    NestingNode {
                        name: "MyModel".to_string(),
                        kind: NestingKind::Class,
                        line: 10,
                        children: vec![NestingNode {
                            name: "forward".to_string(),
                            kind: NestingKind::Method,
                            line: 15,
                            children: Vec::new(),
                        }],
                    },
                ],
            },
        }];

        // Exact suffix match
        let tree = graph.nesting_tree("src/model.py");
        assert!(tree.is_some());
        let tree = tree.unwrap();
        assert_eq!(tree.root.kind, NestingKind::Module);
        assert_eq!(tree.root.children.len(), 2);

        // Partial suffix match
        assert!(graph.nesting_tree("model.py").is_some());

        // No match
        assert!(graph.nesting_tree("nonexistent.py").is_none());
    }

    #[test]
    fn test_nesting_tree_class_nesting() {
        use crate::types::{FileNestingTree, NestingKind, NestingNode};

        let mut graph = make_test_graph();
        graph.nesting_trees = vec![FileNestingTree {
            file: PathBuf::from("layers.py"),
            root: NestingNode {
                name: "layers.py".to_string(),
                kind: NestingKind::Module,
                line: 0,
                children: vec![NestingNode {
                    name: "ConvBlock".to_string(),
                    kind: NestingKind::Class,
                    line: 5,
                    children: vec![
                        NestingNode {
                            name: "__init__".to_string(),
                            kind: NestingKind::Method,
                            line: 6,
                            children: Vec::new(),
                        },
                        NestingNode {
                            name: "forward".to_string(),
                            kind: NestingKind::Method,
                            line: 20,
                            children: Vec::new(),
                        },
                    ],
                }],
            },
        }];

        let tree = graph.nesting_tree("layers.py").unwrap();
        let class_node = &tree.root.children[0];
        assert_eq!(class_node.name, "ConvBlock");
        assert_eq!(class_node.kind, NestingKind::Class);
        assert_eq!(class_node.children.len(), 2);
        assert_eq!(class_node.children[0].kind, NestingKind::Method);
        assert_eq!(class_node.children[1].kind, NestingKind::Method);
    }

    // --- Type flow tests ---

    #[test]
    fn test_type_flows_basic() {
        let analysis = AnalysisResult {
            concepts: vec![make_concept(1, "transform", &["transform"])],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: vec![
                Signature {
                    name: "transform".to_string(),
                    params: vec![Param {
                        name: "image".to_string(),
                        type_annotation: Some("Tensor".to_string()),
                        default: None,
                    }],
                    return_type: Some("Tensor".to_string()),
                    decorators: Vec::new(),
                    docstring_first_line: None,
                    file: PathBuf::from("ops.py"),
                    line: 10,
                    scope: None,
                    body: None,
                },
                Signature {
                    name: "warp".to_string(),
                    params: vec![Param {
                        name: "field".to_string(),
                        type_annotation: Some("Tensor".to_string()),
                        default: None,
                    }],
                    return_type: Some("Tensor".to_string()),
                    decorators: Vec::new(),
                    docstring_first_line: None,
                    file: PathBuf::from("warp.py"),
                    line: 5,
                    scope: None,
                    body: None,
                },
            ],
            classes: Vec::new(),
            call_sites: vec![CallSite {
                caller_scope: Some("warp".to_string()),
                callee: "transform".to_string(),
                file: PathBuf::from("warp.py"),
                line: 8,
            }],
            nesting_trees: Vec::new(),
        };
        let graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();
        let result = graph.type_flows();

        // Call from warp -> transform should produce:
        // 1. return type flow: transform -> warp (Tensor)
        // 2. param type flow: warp -> transform (Tensor)
        assert!(result.flows.len() >= 2);
        assert!(result.total_typed_edges >= 1);

        // All flows should have type "Tensor"
        assert!(result.flows.iter().all(|f| f.type_name == "Tensor"));

        // Dominant types should list Tensor
        assert!(!result.dominant_types.is_empty());
        assert_eq!(result.dominant_types[0].type_name, "Tensor");
    }

    #[test]
    fn test_type_flows_untyped_signatures() {
        let analysis = AnalysisResult {
            concepts: Vec::new(),
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: vec![Signature {
                name: "foo".to_string(),
                params: vec![Param {
                    name: "x".to_string(),
                    type_annotation: None,
                    default: None,
                }],
                return_type: None,
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("a.py"),
                line: 1,
                scope: None,
                body: None,
            }],
            classes: Vec::new(),
            call_sites: vec![CallSite {
                caller_scope: Some("bar".to_string()),
                callee: "foo".to_string(),
                file: PathBuf::from("b.py"),
                line: 5,
            }],
            nesting_trees: Vec::new(),
        };
        let graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();
        let result = graph.type_flows();

        // No type annotations -> no flows, one untyped edge
        assert!(result.flows.is_empty());
        assert_eq!(result.total_typed_edges, 0);
        assert_eq!(result.total_untyped_edges, 1);
    }

    #[test]
    fn test_type_flows_dominant_types_ranking() {
        let analysis = AnalysisResult {
            concepts: Vec::new(),
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: vec![
                Signature {
                    name: "load".to_string(),
                    params: vec![Param {
                        name: "path".to_string(),
                        type_annotation: Some("str".to_string()),
                        default: None,
                    }],
                    return_type: Some("Tensor".to_string()),
                    decorators: Vec::new(),
                    docstring_first_line: None,
                    file: PathBuf::from("io.py"),
                    line: 1,
                    scope: None,
                    body: None,
                },
                Signature {
                    name: "save".to_string(),
                    params: vec![
                        Param {
                            name: "data".to_string(),
                            type_annotation: Some("Tensor".to_string()),
                            default: None,
                        },
                        Param {
                            name: "path".to_string(),
                            type_annotation: Some("str".to_string()),
                            default: None,
                        },
                    ],
                    return_type: None,
                    decorators: Vec::new(),
                    docstring_first_line: None,
                    file: PathBuf::from("io.py"),
                    line: 10,
                    scope: None,
                    body: None,
                },
            ],
            classes: Vec::new(),
            call_sites: vec![
                CallSite {
                    caller_scope: Some("main".to_string()),
                    callee: "load".to_string(),
                    file: PathBuf::from("run.py"),
                    line: 5,
                },
                CallSite {
                    caller_scope: Some("main".to_string()),
                    callee: "save".to_string(),
                    file: PathBuf::from("run.py"),
                    line: 8,
                },
            ],
            nesting_trees: Vec::new(),
        };
        let graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();
        let result = graph.type_flows();

        // Should have flows for: load->main (Tensor ret), main->load (str param),
        // main->save (Tensor param), main->save (str param)
        assert!(!result.flows.is_empty());
        assert!(result.dominant_types.len() >= 2);

        // Dominant types are sorted by count descending
        for i in 1..result.dominant_types.len() {
            assert!(
                result.dominant_types[i - 1].count
                    >= result.dominant_types[i].count
            );
        }
    }

    #[test]
    fn test_trace_type_filters() {
        let analysis = AnalysisResult {
            concepts: Vec::new(),
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: vec![
                Signature {
                    name: "process".to_string(),
                    params: vec![Param {
                        name: "img".to_string(),
                        type_annotation: Some("torch.Tensor".to_string()),
                        default: None,
                    }],
                    return_type: Some("np.ndarray".to_string()),
                    decorators: Vec::new(),
                    docstring_first_line: None,
                    file: PathBuf::from("proc.py"),
                    line: 1,
                    scope: None,
                    body: None,
                },
            ],
            classes: Vec::new(),
            call_sites: vec![CallSite {
                caller_scope: Some("main".to_string()),
                callee: "process".to_string(),
                file: PathBuf::from("run.py"),
                line: 5,
            }],
            nesting_trees: Vec::new(),
        };
        let graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();

        // Trace "Tensor" should find the param flow
        let tensor_flows = graph.trace_type("Tensor");
        assert!(!tensor_flows.is_empty());
        assert!(tensor_flows.iter().all(|f| f.type_name.contains("Tensor")));

        // Trace "ndarray" should find the return flow
        let array_flows = graph.trace_type("ndarray");
        assert!(!array_flows.is_empty());
        assert!(array_flows.iter().all(|f| f.type_name.contains("ndarray")));

        // Trace nonexistent type -> empty
        let empty = graph.trace_type("DataFrame");
        assert!(empty.is_empty());
    }

    #[test]
    fn test_normalize_type_annotation_optional() {
        assert_eq!(
            super::normalize_type_annotation("Optional[Tensor]"),
            "Tensor"
        );
    }

    #[test]
    fn test_normalize_type_annotation_union_with_none() {
        assert_eq!(
            super::normalize_type_annotation("Union[Tensor, None]"),
            "Tensor"
        );
    }

    #[test]
    fn test_normalize_type_annotation_plain() {
        assert_eq!(
            super::normalize_type_annotation("torch.Tensor"),
            "torch.Tensor"
        );
    }

    #[test]
    fn test_normalize_type_annotation_nested_optional() {
        assert_eq!(
            super::normalize_type_annotation("Optional[List[int]]"),
            "List[int]"
        );
    }

}

use super::ConceptGraph;
use crate::tokenizer::{find_abbreviation, split_identifier};
use crate::types::{
    CallSite, Concept, ConceptQueryResult, DescribeSymbolResult, Entity,
    EntitySummary, LocateConceptResult, QueryConceptParams, RelatedConcept,
    RelationshipKind, Signature, SymbolKind,
};
use std::collections::{HashMap, HashSet};

impl ConceptGraph {
    /// Compute domain density score: sum of concept occurrence counts / subtoken count.
    pub(super) fn domain_density(&self, entity: &Entity) -> f64 {
        let subtokens = split_identifier(&entity.name).len().max(1);
        let occ_sum: usize = entity
            .concept_tags
            .iter()
            .filter_map(|id| self.concepts.get(id))
            .map(|c| c.occurrences.len())
            .sum();
        occ_sum as f64 / subtokens as f64
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
        let conventions: Vec<crate::types::Convention> = self
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
        let matching_classes: Vec<crate::types::ClassInfo> = self
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
    pub(super) fn entity_cluster(&self, entity: &Entity) -> Option<usize> {
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
                (label, super::cosine_similarity(&mean, centroid))
            })
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(label, _)| label)
    }

    /// Select up to `top_k` entities via cluster-aware round-robin.
    /// Largest cluster first, highest domain-density entity per cluster,
    /// cycling until top_k is filled. Falls back to domain density
    /// sorting when no clusters exist.
    pub(super) fn select_by_cluster(
        &self,
        candidates: &[&Entity],
        top_k: usize,
    ) -> Vec<EntitySummary> {
        let mut clustered: HashMap<usize, Vec<&Entity>> = HashMap::new();
        let mut unclustered: Vec<&Entity> = Vec::new();
        for &entity in candidates {
            match self.entity_cluster(entity) {
                Some(label) => clustered.entry(label).or_default().push(entity),
                None => unclustered.push(entity),
            }
        }

        // Sort within each cluster: Functions and Classes before Methods (to
        // ensure top-level domain entities are not crowded out by method
        // entities when top_k is a binding constraint), then by domain density.
        let rank_kind = |e: &Entity| {
            if e.kind == crate::types::EntityKind::Method { 0u8 } else { 1u8 }
        };
        let entity_sort = |a: &&Entity, b: &&Entity| {
            rank_kind(b).cmp(&rank_kind(a)).then_with(|| {
                self.domain_density(b)
                    .partial_cmp(&self.domain_density(a))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        };
        for entities in clustered.values_mut() {
            entities.sort_by(entity_sort);
        }
        unclustered.sort_by(entity_sort);

        let mut cluster_labels: Vec<usize> = clustered.keys().copied().collect();
        cluster_labels.sort_by(|a, b| {
            clustered[b].len().cmp(&clustered[a].len())
        });

        let mut result: Vec<EntitySummary> =
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
    ) -> Vec<EntitySummary> {
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
                            || r.kind == RelationshipKind::Uses
                            || r.kind == RelationshipKind::MemberOf)
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

        let mut doc_context = Vec::new();
        if let Some(ref sig) = signature {
            if let Some(ref doc) = sig.docstring_first_line {
                doc_context.push(doc.clone());
            }
        }
        if let Some(ref ci) = class_info {
            if let Some(ref doc) = ci.docstring_first_line {
                if !doc_context.contains(doc) {
                    doc_context.push(doc.clone());
                }
            }
        }
        for concept_name in &concepts {
            if let Some(c) = self
                .concepts
                .values()
                .find(|c| c.canonical == *concept_name)
            {
                for doc in &c.doc_context {
                    if !doc_context.contains(doc) {
                        doc_context.push(doc.clone());
                    }
                }
            }
        }

        Some(DescribeSymbolResult {
            name: name.to_string(),
            kind,
            signature,
            class_info,
            callers,
            callees,
            concepts,
            related_entities,
            doc_context,
        })
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
        let mut scored_cls: Vec<(&crate::types::ClassInfo, i32)> = self
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
        let exemplar_classes: Vec<crate::types::ClassInfo> = scored_cls
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

    /// Helper: build an EntitySummary from an Entity.
    pub(super) fn entity_summary(entity: &Entity) -> EntitySummary {
        EntitySummary {
            name: entity.name.clone(),
            kind: entity.kind.clone(),
            file: entity.file.clone(),
            line: entity.line,
        }
    }

    /// Resolve entity concept tags to canonical concept names.
    pub(super) fn resolve_concept_tags(
        &self,
        entity: Option<&Entity>,
    ) -> Vec<String> {
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
    pub(super) fn format_params(
        params: &[crate::types::Param],
    ) -> Vec<String> {
        params
            .iter()
            .map(|p| match &p.type_annotation {
                Some(ty) => format!("{}: {}", p.name, ty),
                None => p.name.clone(),
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
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
            cluster_id: None,
            subconcepts: Vec::new(),
            doc_context: Vec::new(),
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
            doc_texts: Vec::new(),
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
    fn test_describe_symbol_not_found() {
        let graph = make_test_graph();
        assert!(graph.describe_symbol("nonexistent").is_none());
    }

    #[test]
    fn test_describe_symbol_function() {
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
            doc_texts: Vec::new(),
        };
        analysis.call_sites.push(CallSite {
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
    fn test_list_entities_cluster_round_robin() {
        // Two clusters: transform-domain near [1,0,0], loss-domain near [0,1,0]
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
            doc_texts: Vec::new(),
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
            Vec::new(),
        )
        .unwrap();
        graph.cluster_and_add_similarity_edges(0.75);

        let results = graph.list_entities(None, None, None, 10);
        assert_eq!(results.len(), 3);

        assert_eq!(results[0].name, "spatial_transform");
        assert_eq!(results[1].name, "dice_loss");
        assert_eq!(results[2].name, "compute_spatial_transform");
    }

    #[test]
    fn test_list_entities_no_clusters_fallback() {
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
            doc_texts: Vec::new(),
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
            Vec::new(),
        )
        .unwrap();

        let results = graph.list_entities(None, None, None, 10);
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].name, "spatial_transform");
        assert_eq!(results[1].name, "dice_loss");
    }
}

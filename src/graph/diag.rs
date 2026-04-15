use super::ConceptGraph;
use crate::config::HealthConfig;
use crate::types::{
    CompactContext, ConceptMap, Entity, InconsistencyPair,
    ModuleMapEntry, PatternKind, RelationshipKind, SessionBriefing, Verdict,
    VocabularyHealth,
};
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

impl ConceptGraph {
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

    /// Average per-concept spelling consistency.
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

    /// Average cosine similarity of cluster members to their centroid.
    fn compute_cluster_cohesion(&self) -> f32 {
        if self.cluster_centroids.is_empty() {
            return 1.0;
        }
        let mut cluster_sims: Vec<f32> = Vec::new();
        for (&cluster_id, centroid) in &self.cluster_centroids {
            let members: Vec<&crate::types::Concept> = self
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
                    sims.push(super::cosine_similarity(vec, centroid));
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
    pub(super) fn identifier_matches_any_convention(
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

    /// Find concepts with multiple identifier spellings.
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

    /// Collect identifiers that match no convention pattern.
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

    /// Generate a session briefing from current graph state.
    pub fn session_briefing(&self) -> SessionBriefing {
        let conventions = self.conventions.clone();

        let abbreviations: Vec<(String, String)> = self
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::AbbreviationOf)
            .filter_map(|r| {
                let short = self.concepts.get(&r.source)
                    .map(|c| c.canonical.clone())?;
                let long = self.concepts.get(&r.target)
                    .map(|c| c.canonical.clone())?;
                Some((short, long))
            })
            .collect();

        let mut concepts_sorted: Vec<&crate::types::Concept> =
            self.concepts.values().collect();
        concepts_sorted.sort_by(|a, b| {
            b.occurrences.len().cmp(&a.occurrences.len())
        });
        let top_concepts: Vec<(String, usize)> = concepts_sorted
            .iter()
            .take(20)
            .map(|c| (c.canonical.clone(), c.occurrences.len()))
            .collect();

        let contrastive_pairs: Vec<(String, String)> = self
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::Contrastive)
            .filter_map(|r| {
                let a = self.concepts.get(&r.source)
                    .map(|c| c.canonical.clone())?;
                let b = self.concepts.get(&r.target)
                    .map(|c| c.canonical.clone())?;
                Some((a, b))
            })
            .collect();

        let vocabulary_warnings: Vec<String> = self
            .concepts
            .values()
            .filter_map(|c| {
                let check = self.check_naming(&c.canonical);
                if check.verdict == Verdict::Inconsistent {
                    Some(format!(
                        "use '{}', not '{}'",
                        check.suggestion.as_deref().unwrap_or("?"),
                        c.canonical,
                    ))
                } else {
                    None
                }
            })
            .collect();

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

    /// Semantic topology of the codebase.
    pub fn concept_map(&self) -> ConceptMap {
        use crate::types::EntityKind;

        if self.entities.is_empty() {
            return ConceptMap {
                modules: Vec::new(),
                total_entities: 0,
                total_concepts: self.concepts.len(),
            };
        }

        let all_files: Vec<&std::path::Path> = self
            .entities
            .values()
            .map(|e| e.file.as_path())
            .collect();
        let common_prefix = longest_common_directory(&all_files);

        let mut dir_groups: HashMap<String, Vec<&Entity>> = HashMap::new();
        for entity in self.entities.values() {
            let rel = entity.file
                .strip_prefix(&common_prefix)
                .unwrap_or(&entity.file);
            let dir = rel.parent()
                .map(|p| p.to_string_lossy().to_string())
                .unwrap_or_default();
            let dir = if dir.is_empty() { ".".to_string() } else { dir };
            dir_groups.entry(dir).or_default().push(entity);
        }

        let mut modules: Vec<ModuleMapEntry> = dir_groups
            .into_iter()
            .map(|(path, entities)| {
                let nb_classes = entities.iter()
                    .filter(|e| e.kind == EntityKind::Class).count();
                let nb_functions = entities.iter()
                    .filter(|e| {
                        e.kind == EntityKind::Function
                            || e.kind == EntityKind::Method
                    }).count();
                let nb_entities = entities.len();

                let mut concept_freq: HashMap<u64, usize> = HashMap::new();
                let mut total_tags = 0usize;
                for entity in &entities {
                    for &cid in &entity.concept_tags {
                        *concept_freq.entry(cid).or_insert(0) += 1;
                        total_tags += 1;
                    }
                }

                let mut freq_vec: Vec<(u64, usize)> =
                    concept_freq.into_iter().collect();
                freq_vec.sort_by(|a, b| b.1.cmp(&a.1));
                let dominant_concepts: Vec<String> = freq_vec.iter()
                    .take(5)
                    .filter_map(|(id, _)| {
                        self.concepts.get(id).map(|c| c.canonical.clone())
                    })
                    .collect();

                let concept_density = if nb_entities > 0 {
                    Some(total_tags as f64 / nb_entities as f64)
                } else {
                    None
                };

                ModuleMapEntry {
                    path, dominant_concepts, nb_classes, nb_functions,
                    nb_entities, concept_density,
                }
            })
            .collect();

        modules.sort_by(|a, b| b.nb_entities.cmp(&a.nb_entities));

        ConceptMap {
            total_entities: self.entities.len(),
            total_concepts: self.concepts.len(),
            modules,
        }
    }

    /// L4: Assemble minimal context for a concept or entity.
    pub fn compact_context(
        &self,
        scope: &str,
        max_tokens: usize,
    ) -> Option<CompactContext> {
        struct Section {
            priority: u8,
            text: String,
        }

        let scope_lower = scope.to_lowercase();

        // 1. Try entity name first
        let mut entity = self.entities.values().find(|e| {
            e.name.to_lowercase() == scope_lower
        });

        // 2. If no entity match, try file path resolution.
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
            file_matches.sort_by(|a, b| {
                let pr_a = self.centrality.get(&a.id)
                    .map(|c| c.pagerank).unwrap_or(0.0);
                let pr_b = self.centrality.get(&b.id)
                    .map(|c| c.pagerank).unwrap_or(0.0);
                pr_b.partial_cmp(&pr_a).unwrap_or(std::cmp::Ordering::Equal)
            });
            entity = file_matches.into_iter().next();
            entity?;
        }

        // 3. If no entity, try matching a concept
        let concept = if entity.is_none() {
            self.concepts.values().find(|c| c.canonical == scope_lower)
        } else {
            None
        };

        if entity.is_none() && concept.is_none() {
            return None;
        }

        let header = Section {
            priority: 1,
            text: format!("# {} \u{2014} compact context", scope),
        };

        let mut tiered: Vec<Section> = vec![header];

        if let Some(ent) = entity {
            let mut structure = format!(
                "## Structure\n{} ({:?})", ent.name, ent.kind
            );
            let class_info = self.classes.iter().find(|c| {
                c.name == ent.name && c.file == ent.file
            });
            if let Some(cls) = class_info {
                if !cls.bases.is_empty() {
                    structure.push_str(&format!(
                        "\n  inherits: {}", cls.bases.join(", ")
                    ));
                }
            }

            let mut depends_on: Vec<&str> = self.relationships.iter()
                .filter(|r| {
                    (r.kind == RelationshipKind::InheritsFrom
                        || r.kind == RelationshipKind::Uses)
                        && r.source == ent.id
                })
                .filter_map(|r| {
                    self.entities.get(&r.target).map(|e| e.name.as_str())
                })
                .collect();
            depends_on.sort_unstable();
            depends_on.dedup();
            if !depends_on.is_empty() {
                structure.push_str(&format!(
                    "\n  depends on: {}", depends_on.join(", ")
                ));
            }

            let mut depended_on_by: Vec<&str> = self.relationships.iter()
                .filter(|r| {
                    (r.kind == RelationshipKind::InheritsFrom
                        || r.kind == RelationshipKind::Uses)
                        && r.target == ent.id
                })
                .filter_map(|r| {
                    self.entities.get(&r.source).map(|e| e.name.as_str())
                })
                .collect();
            depended_on_by.sort_unstable();
            depended_on_by.dedup();
            if !depended_on_by.is_empty() {
                structure.push_str(&format!(
                    "\n  depended on by: {}", depended_on_by.join(", "),
                ));
            }

            if let Some(cs) = self.centrality.get(&ent.id) {
                structure.push_str(&format!(
                    "\n  centrality: pagerank={:.3}, in={}, out={}",
                    cs.pagerank, cs.in_degree, cs.out_degree
                ));
            }
            tiered.push(Section { priority: 2, text: structure });

            if let Some(body) = self.signatures.iter()
                .find(|sig| sig.name == ent.name && sig.file == ent.file)
                .and_then(|sig| sig.body.as_ref())
            {
                if !body.body_text.is_empty() {
                    tiered.push(Section {
                        priority: 3,
                        text: format!("## Behavior\n{}", body.body_text),
                    });
                }
            }

            let mut domain_parts: Vec<String> = Vec::new();
            let concept_names: Vec<&str> = ent.concept_tags.iter()
                .filter_map(|id| {
                    self.concepts.get(id).map(|c| c.canonical.as_str())
                })
                .collect();
            if !concept_names.is_empty() {
                domain_parts.push(format!(
                    "Concepts: {}", concept_names.join(", ")
                ));
            }

            let matching_convs: Vec<&str> = self.conventions.iter()
                .filter(|conv| {
                    conv.examples.iter().any(|ex| {
                        ex.to_lowercase().contains(&ent.name.to_lowercase())
                    })
                })
                .map(|conv| conv.semantic_role.as_str())
                .collect();
            if !matching_convs.is_empty() {
                domain_parts.push(format!(
                    "Conventions: {}", matching_convs.join(", ")
                ));
            }

            if let Some(lc) = self.logic_clusters.iter().find(|lc| {
                lc.entity_ids.contains(&ent.id)
            }) {
                let label = lc.behavioral_label.as_deref().unwrap_or("unlabeled");
                let members: Vec<String> = lc.entity_ids.iter()
                    .filter(|id| **id != ent.id)
                    .filter_map(|id| {
                        self.entities.get(id).map(|e| e.name.clone())
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
                    text: format!("## Domain\n{}", domain_parts.join("\n")),
                });
            }

            let similar = self.logic_index
                .find_similar_to_entity(ent.id, 3);
            if !similar.is_empty() {
                let related_lines: Vec<String> = similar.iter()
                    .filter_map(|(id, score)| {
                        self.entities.get(id).map(|e| {
                            format!("- {} (similarity: {:.2})", e.name, score)
                        })
                    })
                    .collect();
                if !related_lines.is_empty() {
                    tiered.push(Section {
                        priority: 5,
                        text: format!("## Related\n{}", related_lines.join("\n")),
                    });
                }
            }
        } else if let Some(concept) = concept {
            let occ_count = concept.occurrences.len();
            let files: HashSet<&str> = concept.occurrences.iter()
                .filter_map(|o| o.file.to_str()).collect();
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
                    text: format!("## Entities\n{}", entity_lines.join("\n")),
                });
            }
        }

        // Tiered truncation
        let estimate = |sections: &[Section]| -> usize {
            let total_chars: usize = sections.iter()
                .map(|s| s.text.len() + 2).sum();
            total_chars / 4
        };

        if estimate(&tiered) > max_tokens {
            for s in &mut tiered {
                if s.priority != 3 { continue; }
                let lines: Vec<&str> = s.text.lines()
                    .filter(|l| !l.starts_with("... (") || !l.ends_with("omitted) ..."))
                    .collect();
                if lines.len() > 5 {
                    let header_line = lines[0];
                    let pc_lines = &lines[1..];
                    let omitted = pc_lines.len() - 4;
                    let trimmed = format!(
                        "{}\n{}\n{}\n... ({} lines omitted) ...\n{}\n{}",
                        header_line, pc_lines[0], pc_lines[1], omitted,
                        pc_lines[pc_lines.len() - 2], pc_lines[pc_lines.len() - 1],
                    );
                    s.text = trimmed;
                }
            }
        }

        if estimate(&tiered) > max_tokens {
            for s in &mut tiered {
                if s.priority != 5 || !s.text.starts_with("## Related") { continue; }
                let lines: Vec<&str> = s.text.lines().collect();
                if lines.len() > 2 {
                    s.text = format!("{}\n{}", lines[0], lines[1]);
                }
            }
        }
        if estimate(&tiered) > max_tokens {
            tiered.retain(|s| s.priority != 5 || !s.text.starts_with("## Related"));
        }

        if estimate(&tiered) > max_tokens {
            for s in &mut tiered {
                if s.priority != 4 { continue; }
                let new_lines: Vec<String> = s.text.lines()
                    .map(|line| {
                        if line.starts_with("Logic cluster: ") {
                            if let Some(paren_end) = line.find(" members)") {
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

        if estimate(&tiered) > max_tokens {
            tiered.retain(|s| s.priority != 4);
        }

        if estimate(&tiered) > max_tokens {
            if let Some(ent) = entity {
                for s in &mut tiered {
                    if s.priority != 2 || !s.text.starts_with("## Structure") { continue; }
                    s.text = format!("## Structure\n{} ({:?})", ent.name, ent.kind);
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
}

#[cfg(test)]
mod tests {
    use crate::embeddings::EmbeddingIndex;
    use crate::graph::ConceptGraph;
    use crate::types::*;
    use std::collections::{HashMap, HashSet};
    use std::path::PathBuf;

    fn make_concept(
        id: u64, canonical: &str, identifiers: &[&str],
    ) -> Concept {
        Concept {
            id, canonical: canonical.to_string(),
            subtokens: vec![canonical.to_string()],
            occurrences: identifiers.iter().map(|name| Occurrence {
                file: PathBuf::from("test.py"), line: 1,
                identifier: name.to_string(), entity_type: EntityType::Function,
            }).collect(),
            entity_types: HashSet::from([EntityType::Function]),
            embedding: None, cluster_id: None,
            subconcepts: Vec::new(), doc_context: Vec::new(),
        }
    }

    fn make_entity(id: u64, name: &str) -> Entity {
        Entity {
            id, name: name.to_string(), kind: EntityKind::Function,
            concept_tags: Vec::new(), semantic_role: "utility".to_string(),
            file: PathBuf::from("test.py"), line: 1,
            signature_idx: None, class_info_idx: None,
        }
    }

    fn make_sig_with_body(name: &str, body_text: &str) -> Signature {
        Signature {
            name: name.to_string(), params: vec![], return_type: None,
            decorators: vec![], docstring_first_line: None,
            file: PathBuf::from("test.py"), line: 1, scope: None,
            body: Some(FunctionBody {
                entity_name: name.to_string(), scope: None,
                body_text: body_text.to_string(), language: "python".to_string(),
                file: PathBuf::from("test.py"), start_line: 2, end_line: 5,
                was_truncated: false,
            }),
        }
    }

    fn make_graph_with_l4(
        entities: Vec<Entity>, signatures: Vec<Signature>,
        centrality: HashMap<u64, CentralityScore>,
    ) -> ConceptGraph {
        let mut graph = ConceptGraph::empty();
        for entity in entities { graph.entities.insert(entity.id, entity); }
        graph.signatures = signatures;
        graph.centrality = centrality;
        graph
    }

    fn make_test_graph() -> ConceptGraph {
        let analysis = AnalysisResult {
            concepts: vec![
                make_concept(1, "transform",
                    &["spatial_transform", "apply_transform", "transform"]),
                make_concept(2, "spatial", &["spatial_transform"]),
                make_concept(3, "ndim", &["ndim", "ndim", "ndim"]),
                make_concept(4, "nb",
                    &["nb_features", "nb_bins", "nb_steps", "nb_dims"]),
                make_concept(5, "features", &["nb_features", "features"]),
            ],
            conventions: vec![Convention {
                pattern: PatternKind::Prefix("nb_".to_string()),
                entity_type: EntityType::Parameter,
                semantic_role: "count".to_string(),
                examples: vec![
                    "nb_features".into(), "nb_bins".into(),
                    "nb_steps".into(), "nb_dims".into(),
                ],
                frequency: 4,
            }],
            co_occurrence_matrix: vec![((1, 2), 1.0)],
            signatures: Vec::new(), classes: Vec::new(),
            call_sites: Vec::new(), nesting_trees: Vec::new(),
            doc_texts: Vec::new(),
        };
        ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap()
    }

    #[test]
    fn test_vocabulary_health_empty_graph() {
        use crate::config::HealthConfig;
        let graph = ConceptGraph::empty();
        let health = graph.vocabulary_health(&HealthConfig::default());
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
        let health = graph.vocabulary_health(&HealthConfig::default());
        assert!(health.convention_coverage > 0.0);
        assert!(health.convention_coverage <= 1.0);
        assert!(health.consistency_ratio > 0.0 && health.consistency_ratio <= 1.0);
        assert_eq!(health.cluster_cohesion, 1.0);
        assert!(health.overall > 0.0 && health.overall <= 1.0);
    }

    #[test]
    fn test_vocabulary_health_inconsistencies() {
        use crate::config::HealthConfig;
        let analysis = AnalysisResult {
            concepts: vec![make_concept(1, "transform",
                &["transform", "transform", "transform", "trf"])],
            conventions: Vec::new(), co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(), classes: Vec::new(),
            call_sites: Vec::new(), nesting_trees: Vec::new(),
            doc_texts: Vec::new(),
        };
        let graph = ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();
        let health = graph.vocabulary_health(&HealthConfig::default());
        assert_eq!(health.top_inconsistencies.len(), 1);
        assert_eq!(health.top_inconsistencies[0].dominant, "transform");
        assert_eq!(health.top_inconsistencies[0].minority, "trf");
        assert_eq!(health.top_inconsistencies[0].dominant_count, 3);
        assert_eq!(health.top_inconsistencies[0].minority_count, 1);
    }

    #[test]
    fn test_vocabulary_health_uncovered() {
        use crate::config::HealthConfig;
        let analysis = AnalysisResult {
            concepts: vec![make_concept(1, "foo", &["foo_bar", "foo_baz"])],
            conventions: Vec::new(), co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(), classes: Vec::new(),
            call_sites: Vec::new(), nesting_trees: Vec::new(),
            doc_texts: Vec::new(),
        };
        let graph = ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();
        let health = graph.vocabulary_health(&HealthConfig::default());
        assert_eq!(health.uncovered_identifiers.len(), 2);
        assert_eq!(health.convention_coverage, 0.0);
    }

    #[test]
    fn test_concept_map_returns_modules() {
        let graph = make_concept_map_graph();
        let map = graph.concept_map();
        assert!(!map.modules.is_empty());
        assert_eq!(map.total_entities, 4);
        assert_eq!(map.total_concepts, 3);
    }

    #[test]
    fn test_concept_map_sorted_by_entity_count() {
        let graph = make_concept_map_graph();
        let map = graph.concept_map();
        for i in 1..map.modules.len() {
            assert!(map.modules[i - 1].nb_entities >= map.modules[i].nb_entities);
        }
    }

    #[test]
    fn test_concept_map_dominant_concepts_present() {
        let graph = make_concept_map_graph();
        let map = graph.concept_map();
        let nn_module = map.modules.iter()
            .find(|m| m.path.contains("nn")).expect("nn module must exist");
        assert!(nn_module.dominant_concepts.iter().any(|c| c == "transform"));
        let losses_module = map.modules.iter()
            .find(|m| m.path.contains("losses")).expect("losses module must exist");
        assert!(losses_module.dominant_concepts.iter().any(|c| c == "loss"));
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
        let nn_module = map.modules.iter()
            .find(|m| m.path.contains("nn")).expect("nn module must exist");
        assert_eq!(nn_module.nb_classes, 1);
        assert_eq!(nn_module.nb_functions, 1);
        assert_eq!(nn_module.nb_entities, 2);
    }

    fn make_concept_map_graph() -> ConceptGraph {
        let analysis = AnalysisResult {
            concepts: vec![
                make_concept(1, "transform", &["spatial_transform"]),
                make_concept(2, "spatial", &["spatial_transform"]),
                make_concept(3, "loss", &["dice_loss", "ncc_loss"]),
            ],
            conventions: Vec::new(), co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(), classes: Vec::new(),
            call_sites: Vec::new(), nesting_trees: Vec::new(),
            doc_texts: Vec::new(),
        };
        let entities = vec![
            Entity {
                id: Entity::hash_id("SpatialTransformer",
                    std::path::Path::new("proj/nn/transform.py"), 10),
                name: "SpatialTransformer".to_string(), kind: EntityKind::Class,
                concept_tags: vec![1, 2], semantic_role: "module".to_string(),
                file: PathBuf::from("proj/nn/transform.py"), line: 10,
                signature_idx: None, class_info_idx: None,
            },
            Entity {
                id: Entity::hash_id("apply_transform",
                    std::path::Path::new("proj/nn/transform.py"), 50),
                name: "apply_transform".to_string(), kind: EntityKind::Function,
                concept_tags: vec![1], semantic_role: "utility".to_string(),
                file: PathBuf::from("proj/nn/transform.py"), line: 50,
                signature_idx: None, class_info_idx: None,
            },
            Entity {
                id: Entity::hash_id("DiceLoss",
                    std::path::Path::new("proj/losses/dice.py"), 5),
                name: "DiceLoss".to_string(), kind: EntityKind::Class,
                concept_tags: vec![3], semantic_role: "module".to_string(),
                file: PathBuf::from("proj/losses/dice.py"), line: 5,
                signature_idx: None, class_info_idx: None,
            },
            Entity {
                id: Entity::hash_id("NccLoss",
                    std::path::Path::new("proj/losses/ncc.py"), 5),
                name: "NccLoss".to_string(), kind: EntityKind::Class,
                concept_tags: vec![3], semantic_role: "module".to_string(),
                file: PathBuf::from("proj/losses/ncc.py"), line: 5,
                signature_idx: None, class_info_idx: None,
            },
        ];
        ConceptGraph::build_with_entities(
            analysis, EmbeddingIndex::empty(), entities, Vec::new(), Vec::new(),
        ).unwrap()
    }

    #[test]
    fn test_compact_context_entity_scope_has_structure_and_behavior() {
        let entity = make_entity(7, "transform_vol");
        let sig = make_sig_with_body("transform_vol", "return apply(vol, trf)");
        let cs = CentralityScore {
            entity_id: 7, in_degree: 1, out_degree: 2, pagerank: 0.1,
        };
        let mut centrality = HashMap::new();
        centrality.insert(7, cs);

        let graph = make_graph_with_l4(vec![entity], vec![sig], centrality);
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
        let concept = make_concept(99, "segment", &["segment"]);
        graph.concepts.insert(99, concept);
        let mut e1 = make_entity(100, "apply_segment");
        e1.concept_tags = vec![99];
        let mut e2 = make_entity(101, "segment_volume");
        e2.concept_tags = vec![99];
        graph.entities.insert(100, e1);
        graph.entities.insert(101, e2);

        let result = graph.compact_context("segment", 500);
        assert!(result.is_some());
        let ctx = result.unwrap();
        assert!(ctx.text.contains("## Entities"));
        assert!(ctx.scope == "segment");
    }

    #[test]
    fn test_compact_context_not_found_returns_none() {
        let graph = ConceptGraph::empty();
        assert!(graph.compact_context("no_such_scope", 500).is_none());
    }

    #[test]
    fn test_compact_context_tiered_truncation_drops_low_priority_sections() {
        let long_body = (0..10)
            .map(|i| format!("result_{i} = step_{i}(data)"))
            .collect::<Vec<_>>().join("\n");
        let sig = make_sig_with_body("fn_a", &long_body);
        let cs = CentralityScore {
            entity_id: 1, in_degree: 5, out_degree: 3, pagerank: 0.5,
        };
        let mut entity = make_entity(1, "fn_a");
        let concept = make_concept(10, "volume", &["volume"]);
        entity.concept_tags = vec![10];
        let cluster = LogicCluster {
            id: 0, entity_ids: vec![1, 2], centroid: vec![],
            behavioral_label: Some("transform".into()),
        };
        let mut centrality = HashMap::new();
        centrality.insert(1, cs);
        let peer = make_entity(2, "fn_b");
        let mut graph = make_graph_with_l4(vec![entity, peer], vec![sig], centrality);
        graph.concepts.insert(10, concept);
        graph.logic_clusters.push(cluster);
        graph.logic_index.insert_vector(1, vec![1.0, 0.0]);
        graph.logic_index.insert_vector(2, vec![0.9, 0.1]);

        let result = graph.compact_context("fn_a", 10);
        assert!(result.is_some());
        let ctx = result.unwrap();
        assert!(ctx.text.contains("fn_a"));
        assert!(!ctx.text.contains("## Related"));
        assert!(!ctx.text.contains("## Domain"));

        let full_ctx = graph.compact_context("fn_a", 10_000).unwrap();
        assert!(ctx.token_estimate < full_ctx.token_estimate);
        assert!(ctx.token_estimate <= 50);
    }

    #[test]
    fn test_compact_context_resolves_file_path_scope() {
        let entity = make_entity(20, "my_func");
        let graph = make_graph_with_l4(vec![entity], vec![], HashMap::new());
        let result = graph.compact_context("test.py", 500);
        assert!(result.is_some());
        let ctx = result.unwrap();
        assert_eq!(ctx.scope, "test.py");
        assert!(ctx.text.contains("my_func"));
    }

    #[test]
    fn test_compact_context_shows_dependency_names() {
        let entity_a = make_entity(30, "entity_a");
        let entity_b = make_entity(31, "entity_b");
        let rel = Relationship {
            source: 30, target: 31,
            kind: RelationshipKind::Uses, weight: 1.0,
        };
        let mut graph = make_graph_with_l4(
            vec![entity_a, entity_b], vec![], HashMap::new(),
        );
        graph.relationships.push(rel);
        let result = graph.compact_context("entity_a", 500);
        assert!(result.is_some());
        let ctx = result.unwrap();
        assert!(ctx.text.contains("depends on: entity_b"),
            "outgoing Uses edge must appear as 'depends on: entity_b', got:\n{}", ctx.text);
    }
}

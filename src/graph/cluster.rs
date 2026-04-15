use super::ConceptGraph;
use crate::tokenizer::{find_abbreviation, split_identifier};
use crate::types::{
    Relationship, RelationshipKind, Subconcept,
};
use std::collections::{HashMap, HashSet};

impl ConceptGraph {
    /// Cluster concepts by embedding similarity using agglomerative clustering
    /// with average linkage, then add SimilarTo edges between cluster members.
    pub fn cluster_and_add_similarity_edges(&mut self, threshold: f32) {
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

        for (&concept_id, &cluster_label) in &result.assignments {
            if let Some(concept) = self.concepts.get_mut(&concept_id) {
                concept.cluster_id = Some(cluster_label);
            }
        }

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
                    let sim = super::cosine_similarity(&vec_a, vec_b);
                    self.relationships.push(Relationship {
                        source: id_a,
                        target: id_b,
                        kind: RelationshipKind::SimilarTo,
                        weight: sim,
                    });
                }
            }
        }

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

    /// Detect abbreviation relationships between concepts.
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
    pub fn add_contrastive_edges(&mut self) {
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

        let mut contrastive_pairs: Vec<(u64, u64)> = Vec::new();
        for (&pair, &score) in &pair_scores {
            if score >= 3 {
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

            let n = unique_ids.len();
            let mut affinity = vec![vec![0.0f32; n]; n];

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

                    let scopes_i: HashSet<&str> = concept
                        .occurrences
                        .iter()
                        .filter(|o| o.identifier == unique_ids[i])
                        .filter_map(|o| o.file.to_str())
                        .collect();
                    let scopes_j: HashSet<&str> = concept
                        .occurrences
                        .iter()
                        .filter(|o| o.identifier == unique_ids[j])
                        .filter_map(|o| o.file.to_str())
                        .collect();
                    let shared = scopes_i.intersection(&scopes_j).count();
                    if shared > 0 {
                        aff += 0.3 * (shared.min(3) as f32 / 3.0);
                    }

                    if let (Some(ei), Some(ej)) =
                        (&id_embeddings[i], &id_embeddings[j])
                    {
                        let sim = super::cosine_similarity(ei, ej);
                        aff += 0.3 * sim.max(0.0);
                    }

                    let sig_i = self.signatures.iter().find(|s| {
                        s.name == unique_ids[i]
                    });
                    let sig_j = self.signatures.iter().find(|s| {
                        s.name == unique_ids[j]
                    });
                    if let (Some(si), Some(sj)) = (sig_i, sig_j) {
                        let params_i: HashSet<String> = si
                            .params.iter()
                            .flat_map(|p| split_identifier(&p.name))
                            .collect();
                        let params_j: HashSet<String> = sj
                            .params.iter()
                            .flat_map(|p| split_identifier(&p.name))
                            .collect();
                        let union_len = params_i.union(&params_j).count();
                        if union_len > 0 {
                            let intersection_len = params_i
                                .intersection(&params_j).count();
                            let jaccard = intersection_len as f32
                                / union_len as f32;
                            aff += 0.25 * jaccard;
                        }
                    }

                    affinity[i][j] = aff;
                    affinity[j][i] = aff;
                }
            }

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

            let mut clusters: HashMap<usize, Vec<usize>> = HashMap::new();
            for i in 0..n {
                let root = find(&mut parent, i);
                clusters.entry(root).or_default().push(i);
            }

            let valid_clusters: Vec<Vec<usize>> = clusters
                .into_values()
                .filter(|c| c.len() >= 2)
                .collect();

            if valid_clusters.len() < 2 {
                continue;
            }

            let mut subconcepts = Vec::new();
            let all_cluster_subtokens: Vec<HashSet<String>> =
                valid_clusters
                    .iter()
                    .map(|cluster| {
                        cluster.iter()
                            .flat_map(|&idx| split_identifier(&unique_ids[idx]))
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
                        my_subtokens.iter()
                            .find(|st| **st != concept.canonical)
                            .cloned()
                            .unwrap_or_else(|| format!("group_{ci}"))
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
                    canonical: format!("{}.{}", concept.canonical, qualifier),
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
}

#[cfg(test)]
mod tests {
    use crate::embeddings::EmbeddingIndex;
    use crate::graph::ConceptGraph;
    use crate::types::*;
    use std::collections::HashSet;
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

    fn make_graph_with_embeddings(
        concepts: Vec<Concept>,
        embedding_pairs: Vec<(u64, Vec<f32>)>,
    ) -> ConceptGraph {
        let analysis = AnalysisResult {
            concepts, conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(), signatures: Vec::new(),
            classes: Vec::new(), call_sites: Vec::new(),
            nesting_trees: Vec::new(), doc_texts: Vec::new(),
        };
        let mut embeddings = EmbeddingIndex::empty();
        for (id, vec) in embedding_pairs {
            embeddings.insert_vector(id, vec);
        }
        ConceptGraph::build(analysis, embeddings).unwrap()
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
            conventions: Vec::new(), co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(), classes: Vec::new(),
            call_sites: Vec::new(), nesting_trees: Vec::new(),
            doc_texts: Vec::new(),
        };
        let mut graph = ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();

        assert!(graph.relationships.iter()
            .all(|r| r.kind != RelationshipKind::AbbreviationOf));

        graph.add_abbreviation_edges();

        let abbrevs: Vec<_> = graph.relationships.iter()
            .filter(|r| r.kind == RelationshipKind::AbbreviationOf).collect();
        assert_eq!(abbrevs.len(), 2);
        assert!(abbrevs.iter().any(|r| r.source == 1 && r.target == 2));
        assert!(abbrevs.iter().any(|r| r.source == 3 && r.target == 4));
    }

    #[test]
    fn test_cluster_and_add_similarity_edges_basic() {
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

        for id in 1u64..=6 {
            assert!(graph.concepts[&id].cluster_id.is_some(),
                "concept {id} missing cluster_id");
        }

        let cid = |id: u64| graph.concepts[&id].cluster_id.unwrap();
        assert_eq!(cid(1), cid(2));
        assert_eq!(cid(1), cid(3));
        assert_eq!(cid(4), cid(5));
        assert_eq!(cid(4), cid(6));
        assert_ne!(cid(1), cid(4));

        let similar_edges: Vec<_> = graph.relationships.iter()
            .filter(|r| r.kind == RelationshipKind::SimilarTo).collect();
        assert!(!similar_edges.is_empty());

        let group1: HashSet<u64> = [1, 2, 3].into();
        let group2: HashSet<u64> = [4, 5, 6].into();
        for edge in &similar_edges {
            let src_in_g1 = group1.contains(&edge.source);
            let tgt_in_g1 = group1.contains(&edge.target);
            let src_in_g2 = group2.contains(&edge.source);
            let tgt_in_g2 = group2.contains(&edge.target);
            let within_group = (src_in_g1 && tgt_in_g1) || (src_in_g2 && tgt_in_g2);
            assert!(within_group, "cross-group SimilarTo edge: {} \u{2192} {}",
                edge.source, edge.target);
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
        let edge_count_first = graph.relationships.iter()
            .filter(|r| r.kind == RelationshipKind::SimilarTo).count();
        let cluster_ids_first: Vec<Option<usize>> = (1u64..=3)
            .map(|id| graph.concepts[&id].cluster_id).collect();

        graph.cluster_and_add_similarity_edges(0.75);
        let edge_count_second = graph.relationships.iter()
            .filter(|r| r.kind == RelationshipKind::SimilarTo).count();
        let cluster_ids_second: Vec<Option<usize>> = (1u64..=3)
            .map(|id| graph.concepts[&id].cluster_id).collect();

        assert_eq!(edge_count_first, edge_count_second);
        assert_eq!(cluster_ids_first, cluster_ids_second);
    }

    #[test]
    fn test_cluster_contrastive_interaction() {
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

        let cid = |id: u64| graph.concepts[&id].cluster_id.unwrap();
        assert_eq!(cid(1), cid(2));
        assert_eq!(cid(3), cid(4));

        let similar_before: HashSet<(u64, u64)> = graph.relationships.iter()
            .filter(|r| r.kind == RelationshipKind::SimilarTo)
            .map(|r| (r.source.min(r.target), r.source.max(r.target)))
            .collect();
        assert!(similar_before.contains(&(1, 2)));
        assert!(similar_before.contains(&(3, 4)));

        graph.add_contrastive_edges();

        let similar_after: HashSet<(u64, u64)> = graph.relationships.iter()
            .filter(|r| r.kind == RelationshipKind::SimilarTo)
            .map(|r| (r.source.min(r.target), r.source.max(r.target)))
            .collect();

        assert!(!similar_after.contains(&(1, 2)));
        let has_contrastive = graph.relationships.iter().any(|r| {
            r.kind == RelationshipKind::Contrastive
                && ((r.source == 1 && r.target == 2)
                    || (r.source == 2 && r.target == 1))
        });
        assert!(has_contrastive);
        assert!(similar_after.contains(&(3, 4)));
    }
}

use super::ConceptGraph;
use crate::tokenizer::split_identifier;
use crate::types::{
    Convention, NameSuggestion, NamingCheckResult, PatternKind,
    RelationshipKind, Verdict,
};
use std::collections::{HashMap, HashSet};

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

impl ConceptGraph {
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
                        format!("'{}' \u{2194} '{}'", canon, word),
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
    fn test_check_naming_consistent() {
        let graph = make_test_graph();
        let result = graph.check_naming("nb_features");
        assert_eq!(result.verdict, Verdict::Consistent);
    }

    #[test]
    fn test_check_naming_inconsistent() {
        let graph = make_test_graph();
        let result = graph.check_naming("n_dims");
        assert_eq!(result.verdict, Verdict::Inconsistent);
        assert!(result.suggestion.is_some());
    }

    #[test]
    fn test_suggest_name_count() {
        let graph = make_test_graph();
        let suggestions = graph.suggest_name("count of features");
        assert!(!suggestions.is_empty());
        assert!(suggestions
            .iter()
            .any(|s| s.name.contains("nb_features")));
    }

    #[test]
    fn test_confidence_bounded_zero_to_one() {
        let graph = make_test_graph();
        let cases = [
            "nb_features",
            "ndim",
            "n_dims",
            "xyzzy_foo",
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
        assert!(result.matching_convention.is_some());
        assert_eq!(result.confidence, 0.8);
    }

    #[test]
    fn test_confidence_corpus_presence() {
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
            doc_texts: Vec::new(),
        };
        let graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty())
                .unwrap();
        let result = graph.check_naming("ndim");
        assert_eq!(result.verdict, Verdict::Consistent);
        assert!(result.matching_convention.is_none());
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
            doc_texts: Vec::new(),
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
}

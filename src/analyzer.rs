use crate::tokenizer::split_identifier;
use crate::types::*;
use anyhow::Result;
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

fn hash_string(s: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    s.hash(&mut hasher);
    hasher.finish()
}

/// Per-file subtoken tracking for TF-IDF computation.
struct Corpus {
    /// subtoken -> { file -> count_in_file }
    subtoken_file_counts: HashMap<String, HashMap<PathBuf, usize>>,
    /// subtoken -> total count across all files
    subtoken_total_counts: HashMap<String, usize>,
    /// subtoken -> set of entity types seen
    subtoken_entity_types: HashMap<String, HashSet<EntityType>>,
    /// subtoken -> all occurrences (identifier name, file, line, entity_type)
    subtoken_occurrences: HashMap<String, Vec<Occurrence>>,
    /// Total number of distinct files
    nb_files: usize,
}

fn build_corpus(parse_results: &[ParseResult]) -> Corpus {
    let mut subtoken_file_counts: HashMap<String, HashMap<PathBuf, usize>> =
        HashMap::new();
    let mut subtoken_total_counts: HashMap<String, usize> = HashMap::new();
    let mut subtoken_entity_types: HashMap<String, HashSet<EntityType>> =
        HashMap::new();
    let mut subtoken_occurrences: HashMap<String, Vec<Occurrence>> =
        HashMap::new();
    let mut all_files: HashSet<PathBuf> = HashSet::new();

    for pr in parse_results {
        for ident in &pr.identifiers {
            all_files.insert(ident.file.clone());
            let subtokens = split_identifier(&ident.name);

            for st in &subtokens {
                *subtoken_file_counts
                    .entry(st.clone())
                    .or_default()
                    .entry(ident.file.clone())
                    .or_insert(0) += 1;

                *subtoken_total_counts.entry(st.clone()).or_insert(0) += 1;

                subtoken_entity_types
                    .entry(st.clone())
                    .or_default()
                    .insert(ident.entity_type.clone());

                subtoken_occurrences
                    .entry(st.clone())
                    .or_default()
                    .push(Occurrence {
                        file: ident.file.clone(),
                        line: ident.line,
                        identifier: ident.name.clone(),
                        entity_type: ident.entity_type.clone(),
                    });
            }
        }
    }

    // If no files found in identifiers, count distinct files from parse_results
    let nb_files = if all_files.is_empty() {
        parse_results.len().max(1)
    } else {
        all_files.len()
    };

    Corpus {
        subtoken_file_counts,
        subtoken_total_counts,
        subtoken_entity_types,
        subtoken_occurrences,
        nb_files,
    }
}

/// Compute TF-IDF score for each subtoken.
///
/// TF = count of subtoken in file / total subtokens in file (averaged across files)
/// IDF = log(N / df) where df = number of files containing the subtoken
///
/// We use a simplified version: max TF across files * IDF.
fn compute_tfidf(corpus: &Corpus) -> HashMap<String, f64> {
    let nb_files = corpus.nb_files as f64;
    let mut scores = HashMap::new();

    for (subtoken, file_counts) in &corpus.subtoken_file_counts {
        let df = file_counts.len() as f64;
        let idf = (nb_files / df).ln();

        // Max term frequency across files, normalized
        let max_tf = file_counts.values().copied().max().unwrap_or(0) as f64;

        // Simple TF-IDF: max_tf * idf
        // For single-file corpora, idf = ln(1) = 0, so use raw frequency
        let score = if nb_files <= 1.0 {
            max_tf
        } else {
            max_tf * idf
        };

        scores.insert(subtoken.clone(), score);
    }

    scores
}

const MIN_OCCURRENCES: usize = 2;
const TFIDF_THRESHOLD: f64 = 0.1;

fn build_concepts(
    corpus: &Corpus,
    tfidf_scores: &HashMap<String, f64>,
) -> Vec<Concept> {
    let mut concepts = Vec::new();

    for (subtoken, total_count) in &corpus.subtoken_total_counts {
        if *total_count < MIN_OCCURRENCES {
            continue;
        }

        let score = tfidf_scores.get(subtoken).copied().unwrap_or(0.0);
        if score < TFIDF_THRESHOLD {
            continue;
        }

        let canonical = subtoken.to_lowercase();
        let id = hash_string(&canonical);

        let occurrences = corpus
            .subtoken_occurrences
            .get(subtoken)
            .cloned()
            .unwrap_or_default();

        let entity_types = corpus
            .subtoken_entity_types
            .get(subtoken)
            .cloned()
            .unwrap_or_default();

        concepts.push(Concept {
            id,
            canonical,
            subtokens: vec![subtoken.clone()],
            occurrences,
            entity_types,
            embedding: None,
        });
    }

    concepts
}

/// Detect naming conventions (prefix, suffix, conversion patterns).
fn detect_conventions(parse_results: &[ParseResult]) -> Vec<Convention> {
    let all_identifiers: Vec<&RawIdentifier> = parse_results
        .iter()
        .flat_map(|pr| pr.identifiers.iter())
        .collect();

    let mut conventions = Vec::new();
    conventions.extend(detect_prefix_conventions(&all_identifiers));
    conventions.extend(detect_suffix_conventions(&all_identifiers));
    conventions.extend(detect_conversion_conventions(&all_identifiers));
    conventions
}

const CONVENTION_THRESHOLD: usize = 3;

fn detect_prefix_conventions(
    identifiers: &[&RawIdentifier],
) -> Vec<Convention> {
    let prefix_roles: &[(&str, &str)] = &[
        ("nb_", "count"),
        ("is_", "boolean predicate"),
        ("has_", "boolean predicate"),
        ("n_", "count"),
        ("num_", "count"),
    ];

    let mut conventions = Vec::new();

    for &(prefix, role) in prefix_roles {
        let matching: Vec<&RawIdentifier> = identifiers
            .iter()
            .filter(|id| id.name.starts_with(prefix))
            .copied()
            .collect();

        if matching.len() < CONVENTION_THRESHOLD {
            continue;
        }

        let entity_type = most_common_entity_type(&matching);
        let examples: Vec<String> =
            matching.iter().map(|id| id.name.clone()).collect();

        conventions.push(Convention {
            pattern: PatternKind::Prefix(prefix.to_string()),
            entity_type,
            semantic_role: role.to_string(),
            examples,
            frequency: matching.len(),
        });
    }

    conventions
}

fn detect_suffix_conventions(
    identifiers: &[&RawIdentifier],
) -> Vec<Convention> {
    let suffix_roles: &[(&str, &str)] = &[
        ("_count", "count"),
        ("_mask", "binary mask"),
        ("_map", "mapping"),
        ("_list", "collection"),
        ("_path", "file path"),
        ("_size", "dimension size"),
        ("_shape", "tensor shape"),
        ("_idx", "index"),
        ("_fn", "callable"),
    ];

    let mut conventions = Vec::new();

    for &(suffix, role) in suffix_roles {
        let matching: Vec<&RawIdentifier> = identifiers
            .iter()
            .filter(|id| id.name.ends_with(suffix))
            .copied()
            .collect();

        if matching.len() < CONVENTION_THRESHOLD {
            continue;
        }

        let entity_type = most_common_entity_type(&matching);
        let examples: Vec<String> =
            matching.iter().map(|id| id.name.clone()).collect();

        conventions.push(Convention {
            pattern: PatternKind::Suffix(suffix.to_string()),
            entity_type,
            semantic_role: role.to_string(),
            examples,
            frequency: matching.len(),
        });
    }

    conventions
}

fn detect_conversion_conventions(
    identifiers: &[&RawIdentifier],
) -> Vec<Convention> {
    let matching: Vec<&RawIdentifier> = identifiers
        .iter()
        .filter(|id| {
            let parts: Vec<&str> = id.name.splitn(3, "_to_").collect();
            // Must have exactly a_to_b form (non-empty parts on both sides)
            parts.len() == 2 && !parts[0].is_empty() && !parts[1].is_empty()
        })
        .copied()
        .collect();

    if matching.len() < CONVENTION_THRESHOLD {
        return Vec::new();
    }

    let entity_type = most_common_entity_type(&matching);
    let examples: Vec<String> =
        matching.iter().map(|id| id.name.clone()).collect();

    vec![Convention {
        pattern: PatternKind::Conversion("_to_".to_string()),
        entity_type,
        semantic_role: "conversion".to_string(),
        examples,
        frequency: matching.len(),
    }]
}

/// Return the most common entity type from a slice of identifiers.
fn most_common_entity_type(identifiers: &[&RawIdentifier]) -> EntityType {
    let mut counts: HashMap<&EntityType, usize> = HashMap::new();
    for id in identifiers {
        *counts.entry(&id.entity_type).or_insert(0) += 1;
    }
    counts
        .into_iter()
        .max_by_key(|&(_, count)| count)
        .map(|(et, _)| et.clone())
        // Safe: only called when identifiers is non-empty
        .unwrap_or(EntityType::Variable)
}

/// Co-occurrence weight for subtokens sharing the same identifier name.
const CO_WEIGHT_IDENTIFIER: f32 = 1.0;
/// Co-occurrence weight for concepts sharing the same scope (function/class).
const CO_WEIGHT_SCOPE: f32 = 0.5;

/// Build co-occurrence matrix from two sources:
/// 1. Per-identifier: subtokens in the same identifier name (weight 1.0)
/// 2. Per-scope: identifiers sharing the same function/class scope (weight 0.5)
fn build_co_occurrence(
    concepts: &[Concept],
    parse_results: &[ParseResult],
) -> Vec<((u64, u64), f32)> {
    let mut subtoken_to_concept: HashMap<&str, u64> = HashMap::new();
    for concept in concepts {
        for st in &concept.subtokens {
            subtoken_to_concept.insert(st.as_str(), concept.id);
        }
    }

    let mut co_counts: HashMap<(u64, u64), f32> = HashMap::new();

    // Per-identifier co-occurrence: subtokens in the same identifier name
    for pr in parse_results {
        for ident in &pr.identifiers {
            let subtokens = split_identifier(&ident.name);
            let concept_ids = sorted_unique_concept_ids(
                &subtokens,
                &subtoken_to_concept,
            );
            add_pairwise(&mut co_counts, &concept_ids, CO_WEIGHT_IDENTIFIER);
        }
    }

    // Per-scope co-occurrence: identifiers sharing the same scope
    let mut scope_concepts: HashMap<&str, HashSet<u64>> = HashMap::new();
    for pr in parse_results {
        for ident in &pr.identifiers {
            if let Some(ref s) = ident.scope {
                let subtokens = split_identifier(&ident.name);
                for st in &subtokens {
                    if let Some(&cid) = subtoken_to_concept.get(st.as_str())
                    {
                        scope_concepts
                            .entry(s.as_str())
                            .or_default()
                            .insert(cid);
                    }
                }
            }
        }
    }
    for concept_ids_set in scope_concepts.values() {
        let mut ids: Vec<u64> = concept_ids_set.iter().copied().collect();
        ids.sort();
        add_pairwise(&mut co_counts, &ids, CO_WEIGHT_SCOPE);
    }

    co_counts.into_iter().collect()
}

/// Collect sorted, deduplicated concept IDs for a list of subtokens.
fn sorted_unique_concept_ids(
    subtokens: &[String],
    subtoken_to_concept: &HashMap<&str, u64>,
) -> Vec<u64> {
    let mut ids: Vec<u64> = subtokens
        .iter()
        .filter_map(|st| subtoken_to_concept.get(st.as_str()))
        .copied()
        .collect::<HashSet<u64>>()
        .into_iter()
        .collect();
    ids.sort();
    ids
}

/// Add `weight` to every pair (i, j) where i < j in sorted concept IDs.
fn add_pairwise(
    co_counts: &mut HashMap<(u64, u64), f32>,
    concept_ids: &[u64],
    weight: f32,
) {
    for i in 0..concept_ids.len() {
        for j in (i + 1)..concept_ids.len() {
            let key = (concept_ids[i], concept_ids[j]);
            *co_counts.entry(key).or_insert(0.0) += weight;
        }
    }
}

/// Run full analysis pipeline on parsed identifiers.
///
/// 1. Aggregate subtokens across all files
/// 2. TF-IDF to distinguish domain terms from generic terms
/// 3. Build concepts from high-TF-IDF subtokens
/// 4. Detect prefix/suffix/conversion naming conventions
/// 5. Build co-occurrence weights from shared identifiers
pub fn analyze(parse_results: &[ParseResult]) -> Result<AnalysisResult> {
    let corpus = build_corpus(parse_results);
    let tfidf_scores = compute_tfidf(&corpus);
    let concepts = build_concepts(&corpus, &tfidf_scores);
    let conventions = detect_conventions(parse_results);
    let co_occurrence_matrix =
        build_co_occurrence(&concepts, parse_results);

    Ok(AnalysisResult {
        concepts,
        conventions,
        co_occurrence_matrix,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EntityType, ParseResult, RawIdentifier};
    use std::path::PathBuf;

    fn make_id(name: &str, entity_type: EntityType) -> RawIdentifier {
        RawIdentifier {
            name: name.to_string(),
            entity_type,
            file: PathBuf::from("test.py"),
            line: 1,
            scope: None,
        }
    }

    fn make_parse_result(
        names: &[(&str, EntityType)],
    ) -> ParseResult {
        ParseResult {
            identifiers: names
                .iter()
                .map(|(n, et)| make_id(n, et.clone()))
                .collect(),
            doc_texts: Vec::new(),
        }
    }

    #[test]
    fn test_concepts_built_from_subtokens() {
        let pr = make_parse_result(&[
            ("spatial_transform", EntityType::Function),
            ("apply_transform", EntityType::Function),
            ("transform_layer", EntityType::Class),
        ]);
        let result = analyze(&[pr]).unwrap();
        assert!(
            result.concepts.iter().any(|c| c.canonical == "transform")
        );
    }

    #[test]
    fn test_convention_detection_nb_prefix() {
        let pr = make_parse_result(&[
            ("nb_features", EntityType::Parameter),
            ("nb_bins", EntityType::Parameter),
            ("nb_steps", EntityType::Parameter),
            ("nb_dims", EntityType::Parameter),
        ]);
        let result = analyze(&[pr]).unwrap();
        assert!(result.conventions.iter().any(|c| {
            matches!(&c.pattern, PatternKind::Prefix(p) if p == "nb_")
        }));
    }

    #[test]
    fn test_convention_detection_to_conversion() {
        let pr = make_parse_result(&[
            ("disp_to_trf", EntityType::Function),
            ("params_to_affine", EntityType::Function),
            ("seg_to_mask", EntityType::Function),
        ]);
        let result = analyze(&[pr]).unwrap();
        assert!(result.conventions.iter().any(|c| {
            matches!(&c.pattern, PatternKind::Conversion(p) if p == "_to_")
        }));
    }

    #[test]
    fn test_no_convention_below_threshold() {
        let pr = make_parse_result(&[
            ("nb_features", EntityType::Parameter),
            ("nb_bins", EntityType::Parameter),
        ]);
        let result = analyze(&[pr]).unwrap();
        assert!(!result.conventions.iter().any(|c| {
            matches!(&c.pattern, PatternKind::Prefix(p) if p == "nb_")
        }));
    }

    #[test]
    fn test_concepts_exclude_low_frequency() {
        // "unique" appears only once — should not become a concept
        let pr = make_parse_result(&[
            ("unique_thing", EntityType::Variable),
            ("spatial_transform", EntityType::Function),
            ("apply_transform", EntityType::Function),
        ]);
        let result = analyze(&[pr]).unwrap();
        assert!(
            !result.concepts.iter().any(|c| c.canonical == "unique")
        );
        assert!(
            !result.concepts.iter().any(|c| c.canonical == "thing")
        );
    }

    #[test]
    fn test_co_occurrence_between_subtokens() {
        // "spatial" and "transform" co-occur in "spatial_transform"
        let pr = make_parse_result(&[
            ("spatial_transform", EntityType::Function),
            ("spatial_transform", EntityType::Function),
            ("apply_transform", EntityType::Function),
        ]);
        let result = analyze(&[pr]).unwrap();

        let spatial = result
            .concepts
            .iter()
            .find(|c| c.canonical == "spatial");
        let transform = result
            .concepts
            .iter()
            .find(|c| c.canonical == "transform");

        // Both should exist as concepts
        if let (Some(s), Some(t)) = (spatial, transform) {
            let pair = result.co_occurrence_matrix.iter().find(
                |((a, b), _)| {
                    (*a == s.id && *b == t.id)
                        || (*a == t.id && *b == s.id)
                },
            );
            assert!(
                pair.is_some(),
                "spatial and transform should co-occur"
            );
        }
    }

    #[test]
    fn test_empty_input() {
        let result = analyze(&[]).unwrap();
        assert!(result.concepts.is_empty());
        assert!(result.conventions.is_empty());
        assert!(result.co_occurrence_matrix.is_empty());
    }

    #[test]
    fn test_hash_string_deterministic() {
        let a = hash_string("transform");
        let b = hash_string("transform");
        assert_eq!(a, b);
        assert_ne!(hash_string("transform"), hash_string("spatial"));
    }
}

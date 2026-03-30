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

fn build_concepts(
    corpus: &Corpus,
    tfidf_scores: &HashMap<String, f64>,
    min_frequency: usize,
    tfidf_threshold: f64,
) -> Vec<Concept> {
    let mut concepts = Vec::new();

    for (subtoken, total_count) in &corpus.subtoken_total_counts {
        if *total_count < min_frequency {
            continue;
        }

        let score = tfidf_scores.get(subtoken).copied().unwrap_or(0.0);
        if score < tfidf_threshold {
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
            subconcepts: Vec::new(),
        });
    }

    concepts
}

/// Detect naming conventions (prefix, suffix, conversion patterns).
fn detect_conventions(
    parse_results: &[ParseResult],
    convention_threshold: usize,
) -> Vec<Convention> {
    let all_identifiers: Vec<&RawIdentifier> = parse_results
        .iter()
        .flat_map(|pr| pr.identifiers.iter())
        .collect();

    let mut conventions = Vec::new();
    conventions.extend(detect_prefix_conventions(
        &all_identifiers,
        convention_threshold,
    ));
    conventions.extend(detect_suffix_conventions(
        &all_identifiers,
        convention_threshold,
    ));
    conventions.extend(detect_conversion_conventions(
        &all_identifiers,
        convention_threshold,
    ));
    // Higher threshold for camelCase conventions to avoid false positives
    // in Python codebases that happen to have a few camelCase names
    conventions.extend(detect_camelcase_prefix_conventions(
        &all_identifiers,
        convention_threshold.max(5),
    ));
    conventions.extend(detect_special_suffix_conventions(
        &all_identifiers,
        convention_threshold,
    ));
    conventions
}

/// Well-known prefix roles for readability hints.
/// Unknown prefixes that meet the frequency threshold still get detected
/// with a generic "prefix pattern" role.
fn prefix_role(prefix: &str) -> &'static str {
    match prefix {
        "nb_" => "count",
        "is_" => "boolean predicate",
        "has_" => "boolean predicate",
        "n_" => "count",
        "num_" => "count",
        _ => "prefix pattern",
    }
}
fn detect_prefix_conventions(
    identifiers: &[&RawIdentifier],
    convention_threshold: usize,
) -> Vec<Convention> {
    // Collect all prefixes of 1-4 lowercase chars followed by '_'
    let mut prefix_ids: HashMap<String, Vec<&RawIdentifier>> = HashMap::new();

    for &id in identifiers {
        if let Some(underscore_pos) = id.name.find('_') {
            let candidate = &id.name[..underscore_pos];
            if (1..=4).contains(&candidate.len())
                && candidate.chars().all(|c| c.is_ascii_lowercase())
            {
                let prefix = format!("{}_", candidate);
                prefix_ids.entry(prefix).or_default().push(id);
            }
        }
    }

    // Deduplicate: count distinct identifier names per prefix
    let mut conventions = Vec::new();
    for (prefix, matching) in &prefix_ids {
        let distinct: HashSet<&str> =
            matching.iter().map(|id| id.name.as_str()).collect();
        if distinct.len() < convention_threshold {
            continue;
        }

        let entity_type = most_common_entity_type(matching);
        let examples: Vec<String> =
            matching.iter().map(|id| id.name.clone()).collect();
        let role = prefix_role(prefix);

        conventions.push(Convention {
            pattern: PatternKind::Prefix(prefix.clone()),
            entity_type,
            semantic_role: role.to_string(),
            examples,
            frequency: distinct.len(),
        });
    }

    conventions
}

/// Well-known suffix roles for readability hints.
/// Unknown suffixes that meet the frequency threshold still get detected
/// with a generic "suffix pattern" role.
fn suffix_role(suffix: &str) -> &'static str {
    match suffix {
        "_count" => "count",
        "_mask" => "binary mask",
        "_map" => "mapping",
        "_list" => "collection",
        "_path" => "file path",
        "_size" => "dimension size",
        "_shape" => "tensor shape",
        "_idx" => "index",
        "_fn" => "callable",
        _ => "suffix pattern",
    }
}

fn detect_suffix_conventions(
    identifiers: &[&RawIdentifier],
    convention_threshold: usize,
) -> Vec<Convention> {
    // Collect all suffixes of '_' + 1-6 lowercase chars at end
    let mut suffix_ids: HashMap<String, Vec<&RawIdentifier>> = HashMap::new();

    for &id in identifiers {
        if let Some(underscore_pos) = id.name.rfind('_') {
            let candidate = &id.name[underscore_pos + 1..];
            if (1..=6).contains(&candidate.len())
                && candidate.chars().all(|c| c.is_ascii_lowercase())
                // Ensure there's something before the suffix
                && underscore_pos > 0
            {
                let suffix = format!("_{}", candidate);
                suffix_ids.entry(suffix).or_default().push(id);
            }
        }
    }

    // Deduplicate: count distinct identifier names per suffix
    let mut conventions = Vec::new();
    for (suffix, matching) in &suffix_ids {
        let distinct: HashSet<&str> =
            matching.iter().map(|id| id.name.as_str()).collect();
        if distinct.len() < convention_threshold {
            continue;
        }

        let entity_type = most_common_entity_type(matching);
        let examples: Vec<String> =
            matching.iter().map(|id| id.name.clone()).collect();
        let role = suffix_role(suffix);

        conventions.push(Convention {
            pattern: PatternKind::Suffix(suffix.clone()),
            entity_type,
            semantic_role: role.to_string(),
            examples,
            frequency: distinct.len(),
        });
    }

    conventions
}

fn detect_conversion_conventions(
    identifiers: &[&RawIdentifier],
    convention_threshold: usize,
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

    if matching.len() < convention_threshold {
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

/// Detect camelCase prefix conventions (e.g. `useAuth`, `onClick`,
/// `handleSubmit`). Looks for a lowercase prefix before the first
/// uppercase letter in identifiers.
fn detect_camelcase_prefix_conventions(
    identifiers: &[&RawIdentifier],
    convention_threshold: usize,
) -> Vec<Convention> {
    let mut prefix_ids: HashMap<String, Vec<&RawIdentifier>> = HashMap::new();

    for &id in identifiers {
        // Only consider identifiers that start with lowercase and contain
        // an uppercase letter (i.e. camelCase).
        let name = &id.name;
        if name.is_empty() || !name.starts_with(|c: char| c.is_ascii_lowercase()) {
            continue;
        }
        // Find first uppercase letter after position 0
        if let Some(pos) = name[1..].find(|c: char| c.is_uppercase()) {
            let prefix = &name[..pos + 1];
            if (2..=6).contains(&prefix.len())
                && prefix.chars().all(|c| c.is_ascii_lowercase())
            {
                prefix_ids.entry(prefix.to_string()).or_default().push(id);
            }
        }
    }

    let mut conventions = Vec::new();
    for (prefix, matching) in &prefix_ids {
        let distinct: HashSet<&str> =
            matching.iter().map(|id| id.name.as_str()).collect();
        if distinct.len() < convention_threshold {
            continue;
        }
        let entity_type = most_common_entity_type(matching);
        let examples: Vec<String> =
            matching.iter().map(|id| id.name.clone()).collect();
        let role = camelcase_prefix_role(prefix);
        conventions.push(Convention {
            pattern: PatternKind::Prefix(prefix.clone()),
            entity_type,
            semantic_role: role.to_string(),
            examples,
            frequency: distinct.len(),
        });
    }
    conventions
}

/// Well-known camelCase prefix roles for TS/JS conventions.
fn camelcase_prefix_role(prefix: &str) -> &'static str {
    match prefix {
        "use" => "React hook",
        "on" => "event handler",
        "handle" => "event handler implementation",
        "get" => "accessor",
        "set" => "mutator",
        "to" => "conversion",
        _ => "prefix pattern",
    }
}

/// Detect special non-underscore suffix conventions like `$` for
/// Observables and `I` prefix for interfaces.
fn detect_special_suffix_conventions(
    identifiers: &[&RawIdentifier],
    convention_threshold: usize,
) -> Vec<Convention> {
    let mut conventions = Vec::new();

    // Detect `$` suffix (RxJS Observables)
    let dollar_ids: Vec<&RawIdentifier> = identifiers
        .iter()
        .filter(|id| id.name.ends_with('$') && id.name.len() > 1)
        .copied()
        .collect();
    if dollar_ids.len() >= convention_threshold {
        let distinct: HashSet<&str> =
            dollar_ids.iter().map(|id| id.name.as_str()).collect();
        if distinct.len() >= convention_threshold {
            let entity_type = most_common_entity_type(&dollar_ids);
            let examples: Vec<String> =
                dollar_ids.iter().map(|id| id.name.clone()).collect();
            conventions.push(Convention {
                pattern: PatternKind::Suffix("$".to_string()),
                entity_type,
                semantic_role: "Observable".to_string(),
                examples,
                frequency: distinct.len(),
            });
        }
    }

    // Detect `I` prefix on PascalCase identifiers (TS interfaces)
    let i_prefix_ids: Vec<&RawIdentifier> = identifiers
        .iter()
        .filter(|id| {
            id.name.starts_with('I')
                && id.name.len() > 1
                && id.name.chars().nth(1).is_some_and(|c| c.is_uppercase())
        })
        .copied()
        .collect();
    if i_prefix_ids.len() >= convention_threshold {
        let distinct: HashSet<&str> =
            i_prefix_ids.iter().map(|id| id.name.as_str()).collect();
        if distinct.len() >= convention_threshold {
            let entity_type = most_common_entity_type(&i_prefix_ids);
            let examples: Vec<String> =
                i_prefix_ids.iter().map(|id| id.name.clone()).collect();
            conventions.push(Convention {
                pattern: PatternKind::Prefix("I".to_string()),
                entity_type,
                semantic_role: "interface".to_string(),
                examples,
                frequency: distinct.len(),
            });
        }
    }

    conventions
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

/// Thresholds for the analysis pipeline, sourced from config.
pub struct AnalysisParams {
    pub min_frequency: usize,
    pub tfidf_threshold: f64,
    pub convention_threshold: usize,
}

impl Default for AnalysisParams {
    fn default() -> Self {
        Self {
            min_frequency: 2,
            tfidf_threshold: 0.1,
            convention_threshold: 3,
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
pub fn analyze(
    parse_results: &[ParseResult],
    params: &AnalysisParams,
) -> Result<AnalysisResult> {
    let corpus = build_corpus(parse_results);
    let tfidf_scores = compute_tfidf(&corpus);
    let concepts = build_concepts(
        &corpus,
        &tfidf_scores,
        params.min_frequency,
        params.tfidf_threshold,
    );
    let conventions =
        detect_conventions(parse_results, params.convention_threshold);
    let co_occurrence_matrix =
        build_co_occurrence(&concepts, parse_results);

    // L2 pass-through: collect from all parse results
    let signatures = parse_results
        .iter()
        .flat_map(|pr| pr.signatures.clone())
        .collect();
    let classes = parse_results
        .iter()
        .flat_map(|pr| pr.classes.clone())
        .collect();
    let call_sites = parse_results
        .iter()
        .flat_map(|pr| pr.call_sites.clone())
        .collect();

    Ok(AnalysisResult {
        concepts,
        conventions,
        co_occurrence_matrix,
        signatures,
        classes,
        call_sites,
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
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
        }
    }

    #[test]
    fn test_concepts_built_from_subtokens() {
        let pr = make_parse_result(&[
            ("spatial_transform", EntityType::Function),
            ("apply_transform", EntityType::Function),
            ("transform_layer", EntityType::Class),
        ]);
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
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
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
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
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
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
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
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
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
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
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();

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
    fn test_discovers_novel_prefix() {
        let pr = make_parse_result(&[
            ("vx_grid", EntityType::Variable),
            ("vx_data", EntityType::Variable),
            ("vx_size", EntityType::Variable),
        ]);
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
        let conv = result.conventions.iter().find(|c| {
            matches!(&c.pattern, PatternKind::Prefix(p) if p == "vx_")
        });
        assert!(conv.is_some(), "should discover novel prefix vx_");
        assert_eq!(conv.unwrap().semantic_role, "prefix pattern");
    }

    #[test]
    fn test_discovers_novel_suffix() {
        let pr = make_parse_result(&[
            ("input_vol", EntityType::Variable),
            ("output_vol", EntityType::Variable),
            ("target_vol", EntityType::Variable),
        ]);
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
        let conv = result.conventions.iter().find(|c| {
            matches!(&c.pattern, PatternKind::Suffix(s) if s == "_vol")
        });
        assert!(conv.is_some(), "should discover novel suffix _vol");
        assert_eq!(conv.unwrap().semantic_role, "suffix pattern");
    }

    #[test]
    fn test_known_suffix_has_semantic_role() {
        let pr = make_parse_result(&[
            ("loss_mask", EntityType::Variable),
            ("roi_mask", EntityType::Variable),
            ("seg_mask", EntityType::Variable),
        ]);
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
        let conv = result.conventions.iter().find(|c| {
            matches!(&c.pattern, PatternKind::Suffix(s) if s == "_mask")
        });
        assert!(conv.is_some(), "should detect _mask suffix");
        assert_eq!(conv.unwrap().semantic_role, "binary mask");
    }

    #[test]
    fn test_empty_input() {
        let result = analyze(&[], &AnalysisParams::default()).unwrap();
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

    // --- camelCase convention tests ---

    #[test]
    fn test_camelcase_use_prefix_detected() {
        let pr = make_parse_result(&[
            ("useAuth", EntityType::Function),
            ("useRouter", EntityType::Function),
            ("useState", EntityType::Function),
            ("useEffect", EntityType::Function),
            ("useMemo", EntityType::Function),
        ]);
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
        let conv = result.conventions.iter().find(|c| {
            matches!(&c.pattern, PatternKind::Prefix(p) if p == "use")
        });
        assert!(conv.is_some(), "should detect camelCase prefix 'use'");
        assert_eq!(conv.unwrap().semantic_role, "React hook");
    }

    #[test]
    fn test_camelcase_handle_prefix_detected() {
        let pr = make_parse_result(&[
            ("handleSubmit", EntityType::Function),
            ("handleClick", EntityType::Function),
            ("handleChange", EntityType::Function),
            ("handleInput", EntityType::Function),
            ("handleBlur", EntityType::Function),
        ]);
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
        let conv = result.conventions.iter().find(|c| {
            matches!(&c.pattern, PatternKind::Prefix(p) if p == "handle")
        });
        assert!(conv.is_some(), "should detect camelCase prefix 'handle'");
        assert_eq!(
            conv.unwrap().semantic_role,
            "event handler implementation"
        );
    }

    #[test]
    fn test_camelcase_on_prefix_detected() {
        let pr = make_parse_result(&[
            ("onClick", EntityType::Variable),
            ("onChange", EntityType::Variable),
            ("onSubmit", EntityType::Variable),
            ("onBlur", EntityType::Variable),
            ("onFocus", EntityType::Variable),
        ]);
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
        let conv = result.conventions.iter().find(|c| {
            matches!(&c.pattern, PatternKind::Prefix(p) if p == "on")
        });
        assert!(conv.is_some(), "should detect camelCase prefix 'on'");
        assert_eq!(conv.unwrap().semantic_role, "event handler");
    }

    #[test]
    fn test_camelcase_below_threshold_not_detected() {
        let pr = make_parse_result(&[
            ("useAuth", EntityType::Function),
            ("useRouter", EntityType::Function),
            // Only 2 — below default threshold of 3
        ]);
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
        let conv = result.conventions.iter().find(|c| {
            matches!(&c.pattern, PatternKind::Prefix(p) if p == "use")
        });
        assert!(conv.is_none(), "should not detect below threshold");
    }

    #[test]
    fn test_dollar_suffix_detected() {
        let pr = make_parse_result(&[
            ("user$", EntityType::Variable),
            ("data$", EntityType::Variable),
            ("click$", EntityType::Variable),
        ]);
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
        let conv = result.conventions.iter().find(|c| {
            matches!(&c.pattern, PatternKind::Suffix(s) if s == "$")
        });
        assert!(conv.is_some(), "should detect $ suffix (Observable)");
        assert_eq!(conv.unwrap().semantic_role, "Observable");
    }

    #[test]
    fn test_i_prefix_interface_detected() {
        let pr = make_parse_result(&[
            ("IPatient", EntityType::Class),
            ("IConfig", EntityType::Class),
            ("IService", EntityType::Class),
        ]);
        let result = analyze(&[pr], &AnalysisParams::default()).unwrap();
        let conv = result.conventions.iter().find(|c| {
            matches!(&c.pattern, PatternKind::Prefix(p) if p == "I")
        });
        assert!(conv.is_some(), "should detect I prefix for interfaces");
        assert_eq!(conv.unwrap().semantic_role, "interface");
    }
}

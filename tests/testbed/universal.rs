use super::expectations::TestbedExpectations;
use super::skip_if_missing;
use ontomics::domain_pack;
use ontomics::types::{PatternKind, QueryConceptParams, Verdict};


/// Estimate token count from JSON string (rough: 1 token ≈ 4 chars).
fn estimate_tokens(json: &str) -> usize {
    json.len() / 4
}

const MCP_TOKEN_LIMIT: usize = 10_000;

// ── list_concepts ──────────────────────────────────────────────────────────

pub fn run_list_concepts(exp: &TestbedExpectations) {
    let graph = skip_if_missing!(exp);
    let concepts = graph.list_concepts();

    // Structural
    assert!(
        concepts.len() >= exp.min_concepts,
        "{}: expected >= {} concepts, got {}",
        exp.name, exp.min_concepts, concepts.len()
    );

    // Every concept: occurrences > 0, canonical is lowercase, no dupes
    let mut seen = std::collections::HashSet::new();
    for c in &concepts {
        assert!(!c.occurrences.is_empty(), "{}: concept '{}' has 0 occurrences", exp.name, c.canonical);
        assert_eq!(c.canonical, c.canonical.to_lowercase(), "{}: canonical '{}' not lowercase", exp.name, c.canonical);
        assert!(seen.insert(&c.canonical), "{}: duplicate canonical '{}'", exp.name, c.canonical);
    }

    // Sorted by occurrence count descending
    for pair in concepts.windows(2) {
        assert!(
            pair[0].occurrences.len() >= pair[1].occurrences.len(),
            "{}: concepts not sorted: '{}' ({}) before '{}' ({})",
            exp.name, pair[0].canonical, pair[0].occurrences.len(),
            pair[1].canonical, pair[1].occurrences.len()
        );
    }

    // Must-contain concepts in top N
    let top_canonicals: Vec<&str> = concepts
        .iter()
        .take(exp.top_n_for_must_contain)
        .map(|c| c.canonical.as_str())
        .collect();
    for expected in &exp.must_contain_concepts {
        assert!(
            top_canonicals.contains(expected),
            "{}: must_contain concept '{}' not found in top {} concepts. Got: {:?}",
            exp.name, expected, exp.top_n_for_must_contain,
            &top_canonicals[..top_canonicals.len().min(20)]
        );
    }

    // MCP response size limit: the MCP tool returns concept summaries
    // (canonical, occurrences count, entity_types, entity_count) — not
    // the full Concept struct. Simulate that lightweight serialization.
    let summaries: Vec<serde_json::Value> = concepts.iter().map(|c| {
        serde_json::json!({
            "canonical": c.canonical,
            "occurrences": c.occurrences.len(),
            "entity_types": c.entity_types,
            "entity_count": 0
        })
    }).collect();
    let json = serde_json::to_string(&summaries).unwrap_or_default();
    assert!(
        estimate_tokens(&json) < MCP_TOKEN_LIMIT,
        "{}: list_concepts response exceeds {} token limit (~{} tokens for {} concepts)",
        exp.name, MCP_TOKEN_LIMIT, estimate_tokens(&json), concepts.len()
    );
}

// ── query_concept ──────────────────────────────────────────────────────────

pub fn run_query_concept(exp: &TestbedExpectations) {
    let graph = skip_if_missing!(exp);

    // Structural: top-5 concepts must be queryable
    let top5: Vec<String> = graph
        .list_concepts()
        .iter()
        .take(5)
        .map(|c| c.canonical.clone())
        .collect();
    for canonical in &top5 {
        let params = QueryConceptParams::default();
        let result = graph.query_concept(canonical, &params);
        assert!(
            result.is_some(),
            "{}: query_concept('{}') returned None for top-5 concept",
            exp.name, canonical
        );
        let r = result.unwrap();
        assert!(!r.variants.is_empty(), "{}: query_concept('{}') has empty variants", exp.name, canonical);

        // All occurrences have valid file paths
        for occ in &r.top_occurrences {
            let path = occ.file.to_string_lossy();
            assert!(!path.is_empty(), "{}: empty file path in occurrence", exp.name);
            assert!(
                !path.contains("site-packages") && !path.contains("node_modules"),
                "{}: occurrence file path contains site-packages or node_modules: {}",
                exp.name, path
            );
        }
    }

    // Per-codebase checks
    for check in &exp.query_concept_checks {
        let params = QueryConceptParams::default();
        let result = graph.query_concept(check.term, &params);
        assert!(
            result.is_some(),
            "{}: query_concept('{}') returned None",
            exp.name, check.term
        );
        let r = result.unwrap();
        assert!(
            r.variants.len() >= check.min_variants,
            "{}: query_concept('{}') expected >= {} variants, got {} ({:?})",
            exp.name, check.term, check.min_variants, r.variants.len(),
            &r.variants[..r.variants.len().min(10)]
        );

        if let Some(rel_substr) = check.related_contains {
            let has_match = r.related.iter().any(|rc| rc.canonical.contains(rel_substr));
            assert!(
                has_match,
                "{}: query_concept('{}') related concepts don't contain '{}'. Got: {:?}",
                exp.name, check.term, rel_substr,
                r.related.iter().map(|rc| &rc.canonical).collect::<Vec<_>>()
            );
        }

        // MCP token limit
        let json = serde_json::to_string(&r).unwrap_or_default();
        assert!(
            estimate_tokens(&json) < MCP_TOKEN_LIMIT,
            "{}: query_concept('{}') response exceeds {} token limit (~{} tokens)",
            exp.name, check.term, MCP_TOKEN_LIMIT, estimate_tokens(&json)
        );
    }
}

// ── check_naming ───────────────────────────────────────────────────────────

pub fn run_check_naming(exp: &TestbedExpectations) {
    let graph = skip_if_missing!(exp);

    for check in &exp.naming_checks {
        let result = graph.check_naming(check.identifier);

        // Input echoes back exactly
        assert_eq!(result.input, check.identifier, "{}: input mismatch", exp.name);

        // Verdict matches
        let verdict_str = match result.verdict {
            Verdict::Consistent => "Consistent",
            Verdict::Inconsistent => "Inconsistent",
            Verdict::Unknown => "Unknown",
        };
        assert_eq!(
            verdict_str, check.expected_verdict,
            "{}: check_naming('{}') expected verdict {}, got {}. Reason: {}",
            exp.name, check.identifier, check.expected_verdict, verdict_str, result.reason
        );

        // If Inconsistent, suggestion must exist
        if matches!(result.verdict, Verdict::Inconsistent) {
            assert!(
                result.suggestion.is_some(),
                "{}: check_naming('{}') is Inconsistent but has no suggestion",
                exp.name, check.identifier
            );
        }

        // If suggestion_contains specified, check it
        if let Some(substr) = check.suggestion_contains {
            if let Some(ref suggestion) = result.suggestion {
                assert!(
                    suggestion.to_lowercase().contains(&substr.to_lowercase()),
                    "{}: check_naming('{}') suggestion '{}' doesn't contain '{}'",
                    exp.name, check.identifier, suggestion, substr
                );
            }
        }
    }
}

// ── suggest_name ───────────────────────────────────────────────────────────

pub fn run_suggest_name(exp: &TestbedExpectations) {
    let graph = skip_if_missing!(exp);

    for check in &exp.suggest_name_checks {
        let suggestions = graph.suggest_name(check.description);

        assert!(
            !suggestions.is_empty(),
            "{}: suggest_name('{}') returned empty",
            exp.name, check.description
        );
        assert!(
            suggestions.len() <= 5,
            "{}: suggest_name returned {} suggestions (max 5)",
            exp.name, suggestions.len()
        );

        // Structural checks
        for s in &suggestions {
            assert!(!s.name.is_empty(), "{}: empty suggestion name", exp.name);
            assert!(!s.based_on.is_empty(), "{}: empty based_on for '{}'", exp.name, s.name);
            assert!(
                s.confidence > 0.0 && s.confidence <= 1.0,
                "{}: confidence {} out of range for '{}'",
                exp.name, s.confidence, s.name
            );
        }

        // Sorted by confidence descending
        for pair in suggestions.windows(2) {
            assert!(
                pair[0].confidence >= pair[1].confidence,
                "{}: suggestions not sorted by confidence",
                exp.name
            );
        }

        // Must-contain check
        let has_match = suggestions.iter().any(|s| {
            s.name.to_lowercase().contains(&check.must_contain.to_lowercase())
        });
        assert!(
            has_match,
            "{}: suggest_name('{}') no suggestion contains '{}'. Got: {:?}",
            exp.name, check.description, check.must_contain,
            suggestions.iter().map(|s| &s.name).collect::<Vec<_>>()
        );
    }
}

// ── list_conventions ───────────────────────────────────────────────────────

pub fn run_list_conventions(exp: &TestbedExpectations) {
    let graph = skip_if_missing!(exp);
    let conventions = graph.list_conventions();

    // Structural
    for conv in conventions {
        assert!(conv.frequency > 0, "{}: convention with frequency 0", exp.name);
        assert!(!conv.examples.is_empty(), "{}: convention with empty examples", exp.name);
        assert!(!conv.semantic_role.is_empty(), "{}: convention with empty semantic_role", exp.name);
    }

    // Must-have conventions
    for check in &exp.must_have_conventions {
        let found = conventions.iter().any(|c| {
            let (kind_str, value) = match &c.pattern {
                PatternKind::Prefix(v) => ("Prefix", v.as_str()),
                PatternKind::Suffix(v) => ("Suffix", v.as_str()),
                PatternKind::Conversion(v) => ("Conversion", v.as_str()),
                PatternKind::Compound(v) => ("Compound", v.as_str()),
            };
            kind_str == check.pattern_kind && value == check.pattern_value
        });
        assert!(
            found,
            "{}: expected convention {}('{}') not found. Got: {:?}",
            exp.name, check.pattern_kind, check.pattern_value,
            conventions.iter().map(|c| format!("{:?}", c.pattern)).collect::<Vec<_>>()
        );
    }
}

// ── describe_symbol ────────────────────────────────────────────────────────

pub fn run_describe_symbol(exp: &TestbedExpectations) {
    let graph = skip_if_missing!(exp);

    for check in &exp.describe_symbol_checks {
        let result = graph.describe_symbol(check.name);
        assert!(
            result.is_some(),
            "{}: describe_symbol('{}') returned None",
            exp.name, check.name
        );
        let r = result.unwrap();

        let kind_str = format!("{:?}", r.kind);
        assert_eq!(
            kind_str, check.expected_kind,
            "{}: describe_symbol('{}') expected kind {}, got {}",
            exp.name, check.name, check.expected_kind, kind_str
        );

        // MCP token limit
        let json = serde_json::to_string(&r).unwrap_or_default();
        assert!(
            estimate_tokens(&json) < MCP_TOKEN_LIMIT,
            "{}: describe_symbol('{}') response exceeds token limit",
            exp.name, check.name
        );
    }
}

// ── locate_concept ─────────────────────────────────────────────────────────

pub fn run_locate_concept(exp: &TestbedExpectations) {
    let graph = skip_if_missing!(exp);

    // Structural: top-5 concepts must be locatable
    let top5: Vec<String> = graph
        .list_concepts()
        .iter()
        .take(5)
        .map(|c| c.canonical.clone())
        .collect();
    for canonical in &top5 {
        let result = graph.locate_concept(canonical);
        assert!(
            result.is_some(),
            "{}: locate_concept('{}') returned None for top-5 concept",
            exp.name, canonical
        );
        let r = result.unwrap();
        assert!(!r.files.is_empty(), "{}: locate_concept('{}') has no files", exp.name, canonical);
    }

    // Per-codebase checks
    for check in &exp.locate_concept_checks {
        let result = graph.locate_concept(check.term);
        assert!(
            result.is_some(),
            "{}: locate_concept('{}') returned None",
            exp.name, check.term
        );
        let r = result.unwrap();
        let file_paths: Vec<String> = r.files.iter().map(|(p, _)| p.to_string_lossy().to_string()).collect();
        let has_match = file_paths.iter().any(|p| p.contains(check.file_contains));
        assert!(
            has_match,
            "{}: locate_concept('{}') files don't contain '{}'. Got: {:?}",
            exp.name, check.term, check.file_contains,
            &file_paths[..file_paths.len().min(10)]
        );
    }
}

// ── list_entities ──────────────────────────────────────────────────────────

pub fn run_list_entities(exp: &TestbedExpectations) {
    let graph = skip_if_missing!(exp);

    let entities = graph.list_entities(None, None, None, 1000);

    // Structural
    assert!(
        entities.len() >= exp.min_entities,
        "{}: expected >= {} entities, got {}",
        exp.name, exp.min_entities, entities.len()
    );
    for ent in &entities {
        assert!(!ent.name.is_empty(), "{}: entity with empty name", exp.name);
        assert!(ent.line > 0, "{}: entity '{}' has line 0", exp.name, ent.name);
        let path = ent.file.to_string_lossy();
        assert!(
            !path.contains("site-packages") && !path.contains("node_modules"),
            "{}: entity '{}' file contains site-packages/node_modules: {}",
            exp.name, ent.name, path
        );
    }

    // Must-have entities
    let entity_names: Vec<&str> = entities.iter().map(|e| e.name.as_str()).collect();
    for check in &exp.must_have_entities {
        assert!(
            entity_names.contains(&check.name),
            "{}: expected entity '{}' not found. Sample entities: {:?}",
            exp.name, check.name,
            &entity_names[..entity_names.len().min(30)]
        );
    }

    // Respects top_k
    let limited = graph.list_entities(None, None, None, 5);
    assert!(limited.len() <= 5, "{}: top_k=5 returned {} entities", exp.name, limited.len());
}

// ── export_domain_pack ─────────────────────────────────────────────────────

pub fn run_export_domain_pack(exp: &TestbedExpectations) {
    let graph = skip_if_missing!(exp);
    let pack = domain_pack::export_domain_pack(&graph);

    // Valid YAML
    let yaml = serde_yaml::to_string(&pack).expect("domain pack should serialize to YAML");
    assert!(!yaml.is_empty(), "{}: empty YAML", exp.name);

    // Domain terms
    assert!(
        pack.domain_terms.len() >= exp.min_domain_terms,
        "{}: expected >= {} domain terms, got {}",
        exp.name, exp.min_domain_terms, pack.domain_terms.len()
    );

    // Abbreviation structural checks
    for abbr in &pack.abbreviations {
        assert!(!abbr.short.is_empty(), "{}: abbreviation with empty short", exp.name);
        assert!(!abbr.long.is_empty(), "{}: abbreviation with empty long", exp.name);
    }

    // MCP token limit
    let json = serde_json::to_string(&pack).unwrap_or_default();
    assert!(
        estimate_tokens(&json) < MCP_TOKEN_LIMIT,
        "{}: export_domain_pack response exceeds {} token limit (~{} tokens)",
        exp.name, MCP_TOKEN_LIMIT, estimate_tokens(&json)
    );
}

// ── ontology_diff (structural only) ────────────────────────────────────────

pub fn run_ontology_diff(exp: &TestbedExpectations) {
    let graph = skip_if_missing!(exp);

    // ontology_diff requires a git repo — just verify the repo is a git repo
    let repo_path = std::path::Path::new(exp.repo_path);
    let git_dir = repo_path.join(".git");
    if !git_dir.exists() {
        eprintln!("{}: not a git repo, skipping ontology_diff", exp.name);
        return;
    }

    // We can't easily call diff::ontology_diff from here without a language
    // parser, so just verify the structural requirement: git repo exists
    // and the concepts are non-empty (diff would operate on these).
    assert!(
        !graph.concepts.is_empty(),
        "{}: graph has no concepts for diff",
        exp.name
    );
}

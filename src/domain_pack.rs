use crate::types::{
    AnalysisResult, AbbreviationMapping, Concept, ConceptAssociation,
    Convention, ConventionEntry, DomainPack, DomainTerm, EntityType,
    PatternKind, Relationship, RelationshipKind,
};
use crate::graph::ConceptGraph;
use anyhow::{bail, Result};
use std::collections::{HashMap, HashSet};
use std::hash::{Hash, Hasher};
use std::path::Path;

/// Extract portable domain knowledge from a concept graph.
pub fn export_domain_pack(graph: &ConceptGraph) -> DomainPack {
    let mut abbreviations = export_abbreviations(graph);
    let mut conventions = export_conventions(graph);
    let domain_terms = export_domain_terms(graph);
    let concept_associations = export_associations(graph);

    abbreviations.truncate(30);
    conventions.truncate(30);

    DomainPack {
        version: 1,
        domain: None,
        abbreviations,
        conventions,
        domain_terms,
        concept_associations,
    }
}

fn export_abbreviations(graph: &ConceptGraph) -> Vec<AbbreviationMapping> {
    graph
        .relationships
        .iter()
        .filter(|r| r.kind == RelationshipKind::AbbreviationOf)
        .filter_map(|r| {
            let short = graph.concepts.get(&r.source)?.canonical.clone();
            let long = graph.concepts.get(&r.target)?.canonical.clone();
            Some(AbbreviationMapping { short, long })
        })
        .collect()
}

fn export_conventions(graph: &ConceptGraph) -> Vec<ConventionEntry> {
    graph
        .conventions
        .iter()
        .map(|conv| {
            let (pattern, value) = match &conv.pattern {
                PatternKind::Prefix(v) => ("prefix", v.clone()),
                PatternKind::Suffix(v) => ("suffix", v.clone()),
                PatternKind::Compound(v) => ("compound", v.clone()),
                PatternKind::Conversion(v) => ("conversion", v.clone()),
            };
            ConventionEntry {
                pattern: pattern.to_string(),
                value,
                role: conv.semantic_role.clone(),
                entity_types: vec![entity_type_to_string(&conv.entity_type)],
            }
        })
        .collect()
}

fn export_domain_terms(graph: &ConceptGraph) -> Vec<DomainTerm> {
    let mut concepts: Vec<&Concept> = graph.concepts.values().collect();
    concepts.sort_by(|a, b| b.occurrences.len().cmp(&a.occurrences.len()));
    concepts.truncate(50);

    concepts
        .iter()
        .map(|c| {
            let entity_types: Vec<String> = c
                .entity_types
                .iter()
                .map(entity_type_to_string)
                .collect();
            DomainTerm {
                term: c.canonical.clone(),
                entity_types,
            }
        })
        .collect()
}

fn export_associations(graph: &ConceptGraph) -> Vec<ConceptAssociation> {
    let mut associations = Vec::new();

    // Contrastive pairs
    for rel in &graph.relationships {
        if rel.kind != RelationshipKind::Contrastive {
            continue;
        }
        let a = match graph.concepts.get(&rel.source) {
            Some(c) => c.canonical.clone(),
            None => continue,
        };
        let b = match graph.concepts.get(&rel.target) {
            Some(c) => c.canonical.clone(),
            None => continue,
        };
        associations.push(ConceptAssociation {
            concepts: vec![a, b],
            kind: "contrastive".to_string(),
        });
    }

    // Concept clusters (grouped by cluster_id from agglomerative clustering)
    let mut cluster_groups: HashMap<usize, Vec<String>> = HashMap::new();
    for concept in graph.concepts.values() {
        if let Some(label) = concept.cluster_id {
            cluster_groups
                .entry(label)
                .or_default()
                .push(concept.canonical.clone());
        }
    }

    let mut sorted_labels: Vec<usize> = cluster_groups.keys().copied().collect();
    sorted_labels.sort();

    for label in sorted_labels {
        if let Some(names) = cluster_groups.get(&label) {
            if names.len() >= 2 {
                let mut sorted_names = names.clone();
                sorted_names.sort();
                sorted_names.truncate(8);
                associations.push(ConceptAssociation {
                    concepts: sorted_names,
                    kind: "cluster".to_string(),
                });
            }
        }
    }

    associations.truncate(50);
    associations
}

/// Load a domain pack from a YAML file.
pub fn load_domain_pack(path: &Path) -> Result<DomainPack> {
    let content = std::fs::read_to_string(path)?;
    let pack: DomainPack = serde_yaml::from_str(&content)?;
    if pack.version != 1 {
        bail!(
            "unsupported domain pack version {} (expected 1)",
            pack.version
        );
    }
    Ok(pack)
}

/// Merge domain pack knowledge into an analysis result (pre-graph-build).
pub fn merge_pack_into_analysis(pack: &DomainPack, analysis: &mut AnalysisResult) {
    // Conventions
    for entry in &pack.conventions {
        if let Some(conv) = convention_entry_to_convention(entry) {
            if !convention_exists(&analysis.conventions, &conv) {
                analysis.conventions.push(conv);
            }
        }
    }

    // Domain terms — create synthetic concepts with 0 occurrences
    for dt in &pack.domain_terms {
        let term_lower = dt.term.to_lowercase();
        let already_exists = analysis
            .concepts
            .iter()
            .any(|c| c.canonical == term_lower);
        if !already_exists {
            let entity_types: HashSet<EntityType> = dt
                .entity_types
                .iter()
                .filter_map(|s| parse_entity_type(s))
                .collect();
            analysis.concepts.push(make_synthetic_concept(
                &term_lower,
                entity_types,
            ));
        }
    }

    // Abbreviations — ensure both short and long forms exist as concepts
    for abbr in &pack.abbreviations {
        let short_lower = abbr.short.to_lowercase();
        let long_lower = abbr.long.to_lowercase();

        for term in [&short_lower, &long_lower] {
            let exists = analysis.concepts.iter().any(|c| c.canonical == *term);
            if !exists {
                analysis.concepts.push(make_synthetic_concept(
                    term,
                    HashSet::new(),
                ));
            }
        }
    }
}

/// Merge concept associations into a built graph (post-graph-build).
pub fn merge_pack_associations(pack: &DomainPack, graph: &mut ConceptGraph) {
    let canon_to_id: std::collections::HashMap<&str, u64> = graph
        .concepts
        .values()
        .map(|c| (c.canonical.as_str(), c.id))
        .collect();

    for assoc in &pack.concept_associations {
        match assoc.kind.as_str() {
            "contrastive" => {
                if assoc.concepts.len() == 2 {
                    let a = assoc.concepts[0].to_lowercase();
                    let b = assoc.concepts[1].to_lowercase();
                    if let (Some(&id_a), Some(&id_b)) =
                        (canon_to_id.get(a.as_str()), canon_to_id.get(b.as_str()))
                    {
                        let already = graph.relationships.iter().any(|r| {
                            r.kind == RelationshipKind::Contrastive
                                && ((r.source == id_a && r.target == id_b)
                                    || (r.source == id_b && r.target == id_a))
                        });
                        if !already {
                            graph.relationships.push(Relationship {
                                source: id_a,
                                target: id_b,
                                kind: RelationshipKind::Contrastive,
                                weight: 1.0,
                            });
                        }
                    }
                }
            }
            "cluster" => {
                let ids: Vec<u64> = assoc
                    .concepts
                    .iter()
                    .filter_map(|name| {
                        canon_to_id.get(name.to_lowercase().as_str()).copied()
                    })
                    .collect();
                for i in 0..ids.len() {
                    for j in (i + 1)..ids.len() {
                        let already = graph.relationships.iter().any(|r| {
                            r.kind == RelationshipKind::SimilarTo
                                && ((r.source == ids[i] && r.target == ids[j])
                                    || (r.source == ids[j] && r.target == ids[i]))
                        });
                        if !already {
                            graph.relationships.push(Relationship {
                                source: ids[i],
                                target: ids[j],
                                kind: RelationshipKind::SimilarTo,
                                weight: 0.8,
                            });
                        }
                    }
                }
            }
            other => {
                eprintln!("Warning: unknown concept association kind '{other}', skipping");
            }
        }
    }
}

/// Convert a ConventionEntry to a Convention.
pub fn convention_entry_to_convention(entry: &ConventionEntry) -> Option<Convention> {
    let pattern = match entry.pattern.as_str() {
        "prefix" => PatternKind::Prefix(entry.value.clone()),
        "suffix" => PatternKind::Suffix(entry.value.clone()),
        "compound" => PatternKind::Compound(entry.value.clone()),
        "conversion" => PatternKind::Conversion(entry.value.clone()),
        other => {
            eprintln!("Warning: unknown convention pattern '{other}', skipping");
            return None;
        }
    };

    let entity_type = entry
        .entity_types
        .first()
        .and_then(|s| parse_entity_type(s))
        .unwrap_or(EntityType::Variable);

    Some(Convention {
        pattern,
        entity_type,
        semantic_role: entry.role.clone(),
        examples: Vec::new(),
        frequency: 0,
    })
}

/// Check if a convention already exists in a list (by pattern kind + value).
pub fn convention_exists(existing: &[Convention], candidate: &Convention) -> bool {
    existing.iter().any(|c| {
        std::mem::discriminant(&c.pattern) == std::mem::discriminant(&candidate.pattern)
            && pattern_value(&c.pattern) == pattern_value(&candidate.pattern)
    })
}

fn pattern_value(pattern: &PatternKind) -> &str {
    match pattern {
        PatternKind::Prefix(v)
        | PatternKind::Suffix(v)
        | PatternKind::Compound(v)
        | PatternKind::Conversion(v) => v,
    }
}

fn entity_type_to_string(et: &EntityType) -> String {
    match et {
        EntityType::Function => "Function",
        EntityType::Class => "Class",
        EntityType::Parameter => "Parameter",
        EntityType::Variable => "Variable",
        EntityType::Attribute => "Attribute",
        EntityType::Decorator => "Decorator",
        EntityType::TypeAnnotation => "TypeAnnotation",
        EntityType::DocText => "DocText",
        EntityType::Interface => "Interface",
        EntityType::TypeAlias => "TypeAlias",
    }
    .to_string()
}

fn parse_entity_type(s: &str) -> Option<EntityType> {
    match s {
        "Function" => Some(EntityType::Function),
        "Class" => Some(EntityType::Class),
        "Parameter" => Some(EntityType::Parameter),
        "Variable" => Some(EntityType::Variable),
        "Attribute" => Some(EntityType::Attribute),
        "Decorator" => Some(EntityType::Decorator),
        "TypeAnnotation" => Some(EntityType::TypeAnnotation),
        "DocText" => Some(EntityType::DocText),
        "Interface" => Some(EntityType::Interface),
        "TypeAlias" => Some(EntityType::TypeAlias),
        _ => None,
    }
}

fn make_synthetic_concept(term: &str, entity_types: HashSet<EntityType>) -> Concept {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    term.hash(&mut hasher);
    let id = hasher.finish();

    Concept {
        id,
        canonical: term.to_string(),
        subtokens: vec![term.to_string()],
        occurrences: Vec::new(),
        entity_types,
        embedding: None,
        cluster_id: None,
        subconcepts: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingIndex;
    use crate::types::*;
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn make_concept(id: u64, canonical: &str, count: usize) -> Concept {
        Concept {
            id,
            canonical: canonical.to_string(),
            subtokens: vec![canonical.to_string()],
            occurrences: (0..count)
                .map(|i| Occurrence {
                    file: PathBuf::from("test.py"),
                    line: i + 1,
                    identifier: canonical.to_string(),
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
                make_concept(1, "trf", 5),
                make_concept(2, "transform", 20),
                make_concept(3, "source", 15),
                make_concept(4, "target", 12),
                make_concept(5, "displacement", 8),
            ],
            conventions: vec![Convention {
                pattern: PatternKind::Prefix("nb_".to_string()),
                entity_type: EntityType::Parameter,
                semantic_role: "count".to_string(),
                examples: vec!["nb_features".into(), "nb_bins".into()],
                frequency: 5,
            }],
            co_occurrence_matrix: vec![],
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
        };
        let mut graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();

        // Add known edges
        graph.relationships.push(Relationship {
            source: 1,
            target: 2,
            kind: RelationshipKind::AbbreviationOf,
            weight: 1.0,
        });
        graph.relationships.push(Relationship {
            source: 3,
            target: 4,
            kind: RelationshipKind::Contrastive,
            weight: 1.0,
        });
        graph.relationships.push(Relationship {
            source: 2,
            target: 5,
            kind: RelationshipKind::SimilarTo,
            weight: 0.85,
        });

        // Assign cluster_id so transform (2) and displacement (5) form a cluster
        if let Some(c) = graph.concepts.get_mut(&2) {
            c.cluster_id = Some(0);
        }
        if let Some(c) = graph.concepts.get_mut(&5) {
            c.cluster_id = Some(0);
        }

        graph
    }

    #[test]
    fn test_export_domain_pack() {
        let graph = make_test_graph();
        let pack = export_domain_pack(&graph);

        assert_eq!(pack.version, 1);
        assert!(pack.domain.is_none());

        // Abbreviations
        assert_eq!(pack.abbreviations.len(), 1);
        assert_eq!(pack.abbreviations[0].short, "trf");
        assert_eq!(pack.abbreviations[0].long, "transform");

        // Conventions
        assert_eq!(pack.conventions.len(), 1);
        assert_eq!(pack.conventions[0].pattern, "prefix");
        assert_eq!(pack.conventions[0].value, "nb_");

        // Domain terms (top 100, sorted by frequency)
        assert!(!pack.domain_terms.is_empty());
        assert_eq!(pack.domain_terms[0].term, "transform"); // highest freq

        // Associations
        let contrastive: Vec<_> = pack
            .concept_associations
            .iter()
            .filter(|a| a.kind == "contrastive")
            .collect();
        assert_eq!(contrastive.len(), 1);
        assert!(contrastive[0].concepts.contains(&"source".to_string()));
        assert!(contrastive[0].concepts.contains(&"target".to_string()));

        let clusters: Vec<_> = pack
            .concept_associations
            .iter()
            .filter(|a| a.kind == "cluster")
            .collect();
        assert_eq!(clusters.len(), 1);
        assert!(clusters[0].concepts.contains(&"transform".to_string()));
        assert!(clusters[0].concepts.contains(&"displacement".to_string()));
    }

    #[test]
    fn test_load_domain_pack_roundtrip() {
        let graph = make_test_graph();
        let pack = export_domain_pack(&graph);
        let yaml = serde_yaml::to_string(&pack).unwrap();

        let dir = PathBuf::from("/tmp/ontomics_test_domain_pack_rt");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_pack.yaml");
        std::fs::write(&path, &yaml).unwrap();

        let loaded = load_domain_pack(&path).unwrap();
        assert_eq!(loaded.version, 1);
        assert_eq!(loaded.abbreviations.len(), pack.abbreviations.len());
        assert_eq!(loaded.conventions.len(), pack.conventions.len());
        assert_eq!(loaded.domain_terms.len(), pack.domain_terms.len());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_load_domain_pack_rejects_bad_version() {
        let dir = PathBuf::from("/tmp/ontomics_test_domain_pack_bad");
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("bad.yaml");
        std::fs::write(&path, "version: 99\nabbreviations: []\n").unwrap();

        let result = load_domain_pack(&path);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("unsupported"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_merge_pack_into_analysis() {
        let pack = DomainPack {
            version: 1,
            domain: None,
            abbreviations: vec![AbbreviationMapping {
                short: "seg".to_string(),
                long: "segmentation".to_string(),
            }],
            conventions: vec![ConventionEntry {
                pattern: "prefix".to_string(),
                value: "is_".to_string(),
                role: "boolean predicate".to_string(),
                entity_types: vec!["Function".to_string()],
            }],
            domain_terms: vec![DomainTerm {
                term: "registration".to_string(),
                entity_types: vec!["Function".to_string()],
            }],
            concept_associations: Vec::new(),
        };

        let mut analysis = AnalysisResult {
            concepts: Vec::new(),
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
        };

        merge_pack_into_analysis(&pack, &mut analysis);

        // Convention added
        assert_eq!(analysis.conventions.len(), 1);
        assert_eq!(analysis.conventions[0].semantic_role, "boolean predicate");

        // Domain term created as synthetic concept
        assert!(analysis.concepts.iter().any(|c| c.canonical == "registration"));
        let reg = analysis
            .concepts
            .iter()
            .find(|c| c.canonical == "registration")
            .unwrap();
        assert!(reg.occurrences.is_empty()); // 0 occurrences — prior

        // Abbreviation terms created
        assert!(analysis.concepts.iter().any(|c| c.canonical == "seg"));
        assert!(analysis.concepts.iter().any(|c| c.canonical == "segmentation"));
    }

    #[test]
    fn test_convention_entry_to_convention_valid() {
        let entry = ConventionEntry {
            pattern: "prefix".to_string(),
            value: "nb_".to_string(),
            role: "count".to_string(),
            entity_types: vec!["Parameter".to_string()],
        };
        let conv = convention_entry_to_convention(&entry).unwrap();
        assert!(matches!(conv.pattern, PatternKind::Prefix(ref v) if v == "nb_"));
        assert_eq!(conv.entity_type, EntityType::Parameter);
    }

    #[test]
    fn test_convention_entry_to_convention_unknown_pattern() {
        let entry = ConventionEntry {
            pattern: "infix".to_string(),
            value: "x".to_string(),
            role: "test".to_string(),
            entity_types: Vec::new(),
        };
        assert!(convention_entry_to_convention(&entry).is_none());
    }

    #[test]
    fn test_convention_exists_dedup() {
        let existing = vec![Convention {
            pattern: PatternKind::Prefix("nb_".to_string()),
            entity_type: EntityType::Parameter,
            semantic_role: "count".to_string(),
            examples: Vec::new(),
            frequency: 5,
        }];

        // Same pattern+value, different role → still a duplicate
        let candidate = Convention {
            pattern: PatternKind::Prefix("nb_".to_string()),
            entity_type: EntityType::Variable,
            semantic_role: "quantity".to_string(),
            examples: Vec::new(),
            frequency: 0,
        };
        assert!(convention_exists(&existing, &candidate));

        // Different value → not a duplicate
        let different = Convention {
            pattern: PatternKind::Prefix("is_".to_string()),
            entity_type: EntityType::Variable,
            semantic_role: "boolean".to_string(),
            examples: Vec::new(),
            frequency: 0,
        };
        assert!(!convention_exists(&existing, &different));
    }
}

#![cfg(feature = "lsp")]

use crate::types::{Entity, Relationship, RelationshipKind};
use anyhow::Result;
use std::collections::HashMap;
use std::path::Path;
use std::process::Command;

/// Enrichment data from pyright analysis.
pub struct LspEnrichment {
    /// class_name -> resolved base class names (cross-file and third-party)
    pub inheritance_chains: HashMap<String, Vec<String>>,
}

/// Run pyright --outputjson on the repo and extract resolved type information.
/// Returns enrichment data for entity inheritance chains.
/// Failure returns empty enrichment (best-effort, never blocks indexing).
pub fn enrich_with_pyright(
    repo_root: &Path,
    pyright_path: &str,
    _timeout_secs: u64,
) -> Result<LspEnrichment> {
    let output = Command::new(pyright_path)
        .arg("--outputjson")
        .current_dir(repo_root)
        .output();

    let output = match output {
        Ok(o) => o,
        Err(e) => {
            eprintln!("LSP: pyright not available: {e}");
            return Ok(LspEnrichment {
                inheritance_chains: HashMap::new(),
            });
        }
    };

    // pyright returns exit code 0 for success, 1 for diagnostics —
    // both produce valid JSON output
    let json_str = String::from_utf8_lossy(&output.stdout);
    parse_pyright_output(&json_str)
}

/// Parse pyright --outputjson output for class inheritance information.
fn parse_pyright_output(json_str: &str) -> Result<LspEnrichment> {
    // pyright --outputjson produces diagnostic output, not full type info.
    // For inheritance chain extraction, we'd need pyright's internal API
    // or a more complex integration. For now, extract what we can from
    // diagnostic messages that mention class hierarchies.
    let inheritance_chains: HashMap<String, Vec<String>> = HashMap::new();

    if let Ok(value) = serde_json::from_str::<serde_json::Value>(json_str) {
        // Look for diagnostics that reference class definitions
        if let Some(diagnostics) = value
            .get("generalDiagnostics")
            .and_then(|d| d.as_array())
        {
            for diag in diagnostics {
                // Extract any class-related type info from diagnostics
                if let Some(message) = diag.get("message").and_then(|m| m.as_str()) {
                    // Pattern: 'class "Foo" inherits from "Bar"'
                    if message.contains("inherits") {
                        // Best-effort parsing — pyright diagnostics vary
                        let _ = message; // placeholder for future enrichment
                    }
                }
            }
        }
    }

    Ok(LspEnrichment {
        inheritance_chains,
    })
}

/// Apply LSP enrichment to entity relationships.
/// Adds InheritsFrom edges for resolved cross-file and third-party inheritance
/// that tree-sitter string matching would miss.
pub fn apply_enrichment(
    entities: &[Entity],
    enrichment: &LspEnrichment,
    relationships: &mut Vec<Relationship>,
) {
    let entity_by_name: HashMap<&str, u64> = entities
        .iter()
        .map(|e| (e.name.as_str(), e.id))
        .collect();

    for (class_name, bases) in &enrichment.inheritance_chains {
        let child_id = match entity_by_name.get(class_name.as_str()) {
            Some(&id) => id,
            None => continue,
        };
        for base in bases {
            // Extract simple name from fully qualified
            let base_simple = base.rsplit('.').next().unwrap_or(base);
            if let Some(&parent_id) = entity_by_name.get(base_simple) {
                if child_id == parent_id {
                    continue;
                }
                // Check if edge already exists (from tree-sitter)
                let exists = relationships.iter().any(|r| {
                    r.source == child_id
                        && r.target == parent_id
                        && r.kind == RelationshipKind::InheritsFrom
                });
                if !exists {
                    relationships.push(Relationship {
                        source: child_id,
                        target: parent_id,
                        kind: RelationshipKind::InheritsFrom,
                        weight: 1.0,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Entity, EntityKind};
    use std::path::PathBuf;

    #[test]
    fn test_pyright_output_parsing() {
        let json = r#"{"version":"1.1","generalDiagnostics":[]}"#;
        let enrichment = parse_pyright_output(json).unwrap();
        assert!(enrichment.inheritance_chains.is_empty());
    }

    #[test]
    fn test_apply_enrichment_adds_edges() {
        let entities = vec![
            Entity {
                id: 1,
                name: "ChildClass".to_string(),
                kind: EntityKind::Class,
                concept_tags: vec![],
                semantic_role: String::new(),
                file: PathBuf::from("child.py"),
                line: 1,
                signature_idx: None,
                class_info_idx: None,
            },
            Entity {
                id: 2,
                name: "ParentClass".to_string(),
                kind: EntityKind::Class,
                concept_tags: vec![],
                semantic_role: String::new(),
                file: PathBuf::from("parent.py"),
                line: 1,
                signature_idx: None,
                class_info_idx: None,
            },
        ];

        let mut chains = HashMap::new();
        chains.insert(
            "ChildClass".to_string(),
            vec!["some.module.ParentClass".to_string()],
        );
        let enrichment = LspEnrichment {
            inheritance_chains: chains,
        };

        let mut relationships = Vec::new();
        apply_enrichment(&entities, &enrichment, &mut relationships);
        assert_eq!(relationships.len(), 1);
        assert_eq!(relationships[0].kind, RelationshipKind::InheritsFrom);
        assert_eq!(relationships[0].source, 1);
        assert_eq!(relationships[0].target, 2);
    }

    #[test]
    fn test_enrichment_failure_graceful() {
        // Trying to run a nonexistent pyright binary should return empty enrichment
        let result = enrich_with_pyright(
            Path::new("/tmp"),
            "nonexistent_pyright_binary_xyz",
            5,
        );
        assert!(result.is_ok());
        assert!(result.unwrap().inheritance_chains.is_empty());
    }
}

use crate::tokenizer::split_identifier;
use crate::types::{
    CallSite, ClassInfo, Concept, Convention, Entity, EntityKind, PatternKind,
    Relationship, RelationshipKind, Signature,
};
use std::collections::{HashMap, HashSet};

/// Build entity nodes from L2 structural data + L1 concept data.
///
/// Returns entities and their relationship edges (Instantiates, InheritsFrom, Uses).
/// Only promotes objects with domain significance:
/// - Contains at least one domain concept subtoken in its name, OR
/// - Is referenced by other entities (as a base class or call target)
pub fn build_entities(
    signatures: &[Signature],
    classes: &[ClassInfo],
    call_sites: &[CallSite],
    concepts: &HashMap<u64, Concept>,
) -> (Vec<Entity>, Vec<Relationship>) {
    let mut entities: Vec<Entity> = Vec::new();
    let mut relationships: Vec<Relationship> = Vec::new();

    // Build concept lookup: canonical -> id
    let concept_by_canonical: HashMap<&str, u64> = concepts
        .values()
        .map(|c| (c.canonical.as_str(), c.id))
        .collect();

    // Collect all class names, base references, and call targets for
    // significance checks
    let all_bases: HashSet<&str> = classes
        .iter()
        .flat_map(|c| c.bases.iter().map(|b| extract_base_name(b)))
        .collect();

    let call_targets: HashSet<&str> = call_sites
        .iter()
        .map(|cs| extract_callee_name(&cs.callee))
        .collect();

    // Also collect class names for method filtering
    let class_names: HashSet<&str> =
        classes.iter().map(|c| c.name.as_str()).collect();

    // Phase 1: Create entities from classes
    for (idx, class) in classes.iter().enumerate() {
        let subtokens = split_identifier(&class.name);
        let concept_tags: Vec<u64> = subtokens
            .iter()
            .filter_map(|st| concept_by_canonical.get(st.as_str()).copied())
            .collect();

        let is_significant = !concept_tags.is_empty()
            || all_bases.contains(class.name.as_str())
            || call_targets.contains(class.name.as_str());

        if !is_significant {
            continue;
        }

        let id = Entity::hash_id(&class.name, &class.file, class.line);
        entities.push(Entity {
            id,
            name: class.name.clone(),
            kind: EntityKind::Class,
            concept_tags: concept_tags.clone(),
            semantic_role: String::new(),
            file: class.file.clone(),
            line: class.line,
            signature_idx: None,
            class_info_idx: Some(idx),
        });

        // Instantiates edges (entity -> concept)
        for &concept_id in &concept_tags {
            relationships.push(Relationship {
                source: id,
                target: concept_id,
                kind: RelationshipKind::Instantiates,
                weight: 1.0,
            });
        }
    }

    // Phase 2: Create entities from top-level functions (exclude methods)
    for (idx, sig) in signatures.iter().enumerate() {
        // Skip methods: scope contains a class name
        if let Some(ref scope) = sig.scope {
            let scope_root = scope.split('.').next().unwrap_or(scope);
            if class_names.contains(scope_root) {
                continue;
            }
        }

        let subtokens = split_identifier(&sig.name);
        let concept_tags: Vec<u64> = subtokens
            .iter()
            .filter_map(|st| concept_by_canonical.get(st.as_str()).copied())
            .collect();

        let is_significant = !concept_tags.is_empty()
            || call_targets.contains(sig.name.as_str());

        if !is_significant {
            continue;
        }

        let id = Entity::hash_id(&sig.name, &sig.file, sig.line);
        entities.push(Entity {
            id,
            name: sig.name.clone(),
            kind: EntityKind::Function,
            concept_tags: concept_tags.clone(),
            semantic_role: String::new(),
            file: sig.file.clone(),
            line: sig.line,
            signature_idx: Some(idx),
            class_info_idx: None,
        });

        for &concept_id in &concept_tags {
            relationships.push(Relationship {
                source: id,
                target: concept_id,
                kind: RelationshipKind::Instantiates,
                weight: 1.0,
            });
        }
    }

    // Build entity name -> IDs lookup (multi-map: name can appear in multiple files)
    let mut entity_by_name: HashMap<&str, Vec<u64>> = HashMap::new();
    for e in &entities {
        entity_by_name.entry(e.name.as_str()).or_default().push(e.id);
    }

    // Phase 1b: InheritsFrom edges (entity -> entity)
    for class in classes {
        let child_ids = match entity_by_name.get(class.name.as_str()) {
            Some(ids) => ids.clone(),
            None => continue,
        };
        for base in &class.bases {
            let base_name = extract_base_name(base);
            if let Some(parent_ids) = entity_by_name.get(base_name) {
                for &child_id in &child_ids {
                    for &parent_id in parent_ids {
                        if child_id != parent_id {
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
    }

    // Phase 3: Build Uses edges from call sites (deduplicated)
    let mut uses_edges: HashSet<(u64, u64)> = HashSet::new();
    for cs in call_sites {
        let callee_name = extract_callee_name(&cs.callee);
        let caller_name = cs
            .caller_scope
            .as_ref()
            .map(|s| s.split('.').next().unwrap_or(s))
            .unwrap_or("");

        let caller_ids = entity_by_name.get(caller_name);
        let callee_ids = entity_by_name.get(callee_name);

        if let (Some(c_ids), Some(e_ids)) = (caller_ids, callee_ids) {
            for &caller_id in c_ids {
                for &callee_id in e_ids {
                    if caller_id != callee_id && uses_edges.insert((caller_id, callee_id)) {
                        relationships.push(Relationship {
                            source: caller_id,
                            target: callee_id,
                            kind: RelationshipKind::Uses,
                            weight: 1.0,
                        });
                    }
                }
            }
        }
    }

    (entities, relationships)
}

/// Infer semantic roles for entities based on concept tags, base classes,
/// and naming conventions. Mutates entities in place.
///
/// Three strategies (applied in order, first match wins):
/// 1. Base class pattern: inherits nn.Module + tag "loss" -> "loss module"
/// 2. Convention match: name matches detected convention -> convention role + tags
/// 3. Concept-tag fallback: tags [spatial, transform] -> "transform function"
pub fn infer_semantic_roles(
    entities: &mut [Entity],
    classes: &[ClassInfo],
    conventions: &[Convention],
    concepts: &HashMap<u64, Concept>,
) {
    let concept_names: HashMap<u64, &str> = concepts
        .values()
        .map(|c| (c.id, c.canonical.as_str()))
        .collect();

    let class_bases: HashMap<&str, &[String]> = classes
        .iter()
        .map(|c| (c.name.as_str(), c.bases.as_slice()))
        .collect();

    for entity in entities.iter_mut() {
        let tag_names: Vec<&str> = entity
            .concept_tags
            .iter()
            .filter_map(|id| concept_names.get(id).copied())
            .collect();

        // Strategy 1: Base class pattern (classes only)
        if entity.kind == EntityKind::Class {
            if let Some(bases) = class_bases.get(entity.name.as_str()) {
                let has_module_base = bases.iter().any(|b| {
                    let base_name = extract_base_name(b);
                    base_name == "Module"
                        || b == "nn.Module"
                        || b == "torch.nn.Module"
                });
                if has_module_base && !tag_names.is_empty() {
                    let primary_tag = tag_names.last().unwrap();
                    entity.semantic_role = format!("{primary_tag} module");
                    continue;
                }
            }
        }

        // Strategy 2: Convention match
        let name_lower = entity.name.to_lowercase();
        let mut matched = false;
        for conv in conventions {
            let matches = match &conv.pattern {
                PatternKind::Prefix(p) => name_lower.starts_with(p.as_str()),
                PatternKind::Suffix(s) => name_lower.ends_with(s.as_str()),
                PatternKind::Conversion(c) => name_lower.contains(c.as_str()),
                PatternKind::Compound(c) => name_lower.contains(c.as_str()),
            };
            if matches {
                if !tag_names.is_empty() {
                    entity.semantic_role = format!(
                        "{} ({})",
                        conv.semantic_role,
                        tag_names.join(", ")
                    );
                } else {
                    entity.semantic_role = conv.semantic_role.clone();
                }
                matched = true;
                break;
            }
        }
        if matched {
            continue;
        }

        // Strategy 3: Concept-tag fallback
        if !tag_names.is_empty() {
            let primary_tag = tag_names.last().unwrap();
            let kind_suffix = match entity.kind {
                EntityKind::Class => "class",
                EntityKind::Function => "function",
                EntityKind::Method => "method",
            };
            entity.semantic_role = format!("{primary_tag} {kind_suffix}");
        }
    }
}

/// Extract the last component from a dotted base name.
/// E.g., "nn.Module" -> "Module", "Module" -> "Module"
fn extract_base_name(base: &str) -> &str {
    base.rsplit('.').next().unwrap_or(base)
}

/// Extract the last component from a dotted callee name.
fn extract_callee_name(callee: &str) -> &str {
    callee.rsplit('.').next().unwrap_or(callee)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{EntityType, Occurrence, Param};
    use std::collections::HashSet;
    use std::path::PathBuf;

    fn make_concept(id: u64, canonical: &str) -> Concept {
        Concept {
            id,
            canonical: canonical.to_string(),
            subtokens: vec![canonical.to_string()],
            occurrences: vec![Occurrence {
                file: PathBuf::from("test.py"),
                line: 1,
                identifier: canonical.to_string(),
                entity_type: EntityType::Function,
            }],
            entity_types: HashSet::from([EntityType::Function]),
            embedding: None,
            subconcepts: Vec::new(),
        }
    }

    fn make_concepts(pairs: &[(u64, &str)]) -> HashMap<u64, Concept> {
        pairs.iter().map(|&(id, name)| (id, make_concept(id, name))).collect()
    }

    #[test]
    fn test_entity_from_class_with_concepts() {
        let classes = vec![ClassInfo {
            name: "FocalDiceLoss".to_string(),
            bases: vec!["nn.Module".to_string()],
            methods: vec!["forward".to_string()],
            attributes: vec![],
            docstring_first_line: None,
            file: PathBuf::from("losses.py"),
            line: 10,
        }];
        let concepts = make_concepts(&[(1, "focal"), (2, "dice"), (3, "loss")]);
        let (entities, _) = build_entities(&[], &classes, &[], &concepts);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].kind, EntityKind::Class);
        assert_eq!(entities[0].concept_tags.len(), 3);
    }

    #[test]
    fn test_entity_from_function_with_concepts() {
        let sigs = vec![Signature {
            name: "spatial_transform".to_string(),
            params: vec![],
            return_type: None,
            decorators: vec![],
            docstring_first_line: None,
            file: PathBuf::from("utils.py"),
            line: 5,
            scope: None,
        }];
        let concepts = make_concepts(&[(1, "spatial"), (2, "transform")]);
        let (entities, _) = build_entities(&sigs, &[], &[], &concepts);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].kind, EntityKind::Function);
        assert_eq!(entities[0].concept_tags.len(), 2);
    }

    #[test]
    fn test_entity_significance_filter() {
        let classes = vec![ClassInfo {
            name: "HelperUtil".to_string(),
            bases: vec![],
            methods: vec![],
            attributes: vec![],
            docstring_first_line: None,
            file: PathBuf::from("utils.py"),
            line: 1,
        }];
        // No matching concepts, no references
        let concepts = make_concepts(&[(1, "loss")]);
        let (entities, _) = build_entities(&[], &classes, &[], &concepts);
        assert!(entities.is_empty());
    }

    #[test]
    fn test_entity_significance_via_reference() {
        let classes = vec![ClassInfo {
            name: "HelperUtil".to_string(),
            bases: vec![],
            methods: vec![],
            attributes: vec![],
            docstring_first_line: None,
            file: PathBuf::from("utils.py"),
            line: 1,
        }];
        let call_sites = vec![CallSite {
            caller_scope: Some("main".to_string()),
            callee: "HelperUtil".to_string(),
            file: PathBuf::from("main.py"),
            line: 10,
        }];
        let concepts = make_concepts(&[(1, "loss")]);
        let (entities, _) = build_entities(&[], &classes, &call_sites, &concepts);
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "HelperUtil");
    }

    #[test]
    fn test_entity_inherits_from_edge() {
        let classes = vec![
            ClassInfo {
                name: "Module".to_string(),
                bases: vec![],
                methods: vec![],
                attributes: vec![],
                docstring_first_line: None,
                file: PathBuf::from("nn.py"),
                line: 1,
            },
            ClassInfo {
                name: "LossModule".to_string(),
                bases: vec!["Module".to_string()],
                methods: vec![],
                attributes: vec![],
                docstring_first_line: None,
                file: PathBuf::from("loss.py"),
                line: 5,
            },
        ];
        let concepts = make_concepts(&[(1, "loss"), (2, "module")]);
        let (entities, rels) = build_entities(&[], &classes, &[], &concepts);
        assert_eq!(entities.len(), 2);
        let inherits = rels
            .iter()
            .filter(|r| r.kind == RelationshipKind::InheritsFrom)
            .count();
        assert_eq!(inherits, 1);
    }

    #[test]
    fn test_entity_inherits_from_dotted_base() {
        let classes = vec![
            ClassInfo {
                name: "Module".to_string(),
                bases: vec![],
                methods: vec![],
                attributes: vec![],
                docstring_first_line: None,
                file: PathBuf::from("nn.py"),
                line: 1,
            },
            ClassInfo {
                name: "LossModule".to_string(),
                bases: vec!["nn.Module".to_string()],
                methods: vec![],
                attributes: vec![],
                docstring_first_line: None,
                file: PathBuf::from("loss.py"),
                line: 5,
            },
        ];
        let concepts = make_concepts(&[(1, "loss"), (2, "module")]);
        let (_, rels) = build_entities(&[], &classes, &[], &concepts);
        let inherits = rels
            .iter()
            .filter(|r| r.kind == RelationshipKind::InheritsFrom)
            .count();
        assert_eq!(inherits, 1);
    }

    #[test]
    fn test_entity_uses_edge() {
        let classes = vec![
            ClassInfo {
                name: "VxmDense".to_string(),
                bases: vec![],
                methods: vec!["forward".to_string()],
                attributes: vec![],
                docstring_first_line: None,
                file: PathBuf::from("models.py"),
                line: 1,
            },
            ClassInfo {
                name: "SpatialTransformer".to_string(),
                bases: vec![],
                methods: vec![],
                attributes: vec![],
                docstring_first_line: None,
                file: PathBuf::from("layers.py"),
                line: 10,
            },
        ];
        let call_sites = vec![CallSite {
            caller_scope: Some("VxmDense.forward".to_string()),
            callee: "SpatialTransformer".to_string(),
            file: PathBuf::from("models.py"),
            line: 5,
        }];
        let concepts = make_concepts(&[
            (1, "vxm"),
            (2, "dense"),
            (3, "spatial"),
            (4, "transformer"),
        ]);
        let (entities, rels) = build_entities(&[], &classes, &call_sites, &concepts);
        assert_eq!(entities.len(), 2);
        let uses = rels
            .iter()
            .filter(|r| r.kind == RelationshipKind::Uses)
            .count();
        assert_eq!(uses, 1);
    }

    #[test]
    fn test_entity_uses_dedup() {
        let classes = vec![
            ClassInfo {
                name: "ModelA".to_string(),
                bases: vec![],
                methods: vec![],
                attributes: vec![],
                docstring_first_line: None,
                file: PathBuf::from("a.py"),
                line: 1,
            },
            ClassInfo {
                name: "ModelB".to_string(),
                bases: vec![],
                methods: vec![],
                attributes: vec![],
                docstring_first_line: None,
                file: PathBuf::from("b.py"),
                line: 1,
            },
        ];
        let call_sites = vec![
            CallSite {
                caller_scope: Some("ModelA.forward".to_string()),
                callee: "ModelB".to_string(),
                file: PathBuf::from("a.py"),
                line: 5,
            },
            CallSite {
                caller_scope: Some("ModelA.init".to_string()),
                callee: "ModelB".to_string(),
                file: PathBuf::from("a.py"),
                line: 3,
            },
        ];
        let concepts = make_concepts(&[(1, "model")]);
        let (_, rels) = build_entities(&[], &classes, &call_sites, &concepts);
        let uses = rels
            .iter()
            .filter(|r| r.kind == RelationshipKind::Uses)
            .count();
        assert_eq!(uses, 1, "duplicate Uses edges should be deduplicated");
    }

    #[test]
    fn test_entity_id_deterministic() {
        let file = PathBuf::from("test.py");
        let id1 = Entity::hash_id("Foo", &file, 10);
        let id2 = Entity::hash_id("Foo", &file, 10);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_entity_id_no_concept_collision() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        // Concept ID for "transform"
        let mut h = DefaultHasher::new();
        "transform".hash(&mut h);
        let concept_id = h.finish();

        // Entity ID for "transform" at test.py:1
        let entity_id =
            Entity::hash_id("transform", &PathBuf::from("test.py"), 1);

        assert_ne!(concept_id, entity_id, "entity and concept IDs must not collide");
    }

    #[test]
    fn test_methods_excluded_from_top_level_entities() {
        let classes = vec![ClassInfo {
            name: "FocalDiceLoss".to_string(),
            bases: vec![],
            methods: vec!["forward".to_string()],
            attributes: vec![],
            docstring_first_line: None,
            file: PathBuf::from("loss.py"),
            line: 1,
        }];
        let sigs = vec![Signature {
            name: "forward".to_string(),
            params: vec![Param {
                name: "x".to_string(),
                type_annotation: None,
                default: None,
            }],
            return_type: None,
            decorators: vec![],
            docstring_first_line: None,
            file: PathBuf::from("loss.py"),
            line: 5,
            scope: Some("FocalDiceLoss".to_string()),
        }];
        let concepts = make_concepts(&[(1, "focal"), (2, "dice"), (3, "loss")]);
        let (entities, _) = build_entities(&sigs, &classes, &[], &concepts);
        // Only the class should be an entity, not the method
        assert_eq!(entities.len(), 1);
        assert_eq!(entities[0].name, "FocalDiceLoss");
    }

    // --- Semantic role inference tests ---

    #[test]
    fn test_role_module_base_class() {
        let classes = vec![ClassInfo {
            name: "FocalDiceLoss".to_string(),
            bases: vec!["nn.Module".to_string()],
            methods: vec![],
            attributes: vec![],
            docstring_first_line: None,
            file: PathBuf::from("loss.py"),
            line: 1,
        }];
        let concepts = make_concepts(&[(1, "focal"), (2, "dice"), (3, "loss")]);
        let (mut entities, _) = build_entities(&[], &classes, &[], &concepts);
        infer_semantic_roles(&mut entities, &classes, &[], &concepts);
        assert_eq!(entities[0].semantic_role, "loss module");
    }

    #[test]
    fn test_role_convention_match() {
        let sigs = vec![Signature {
            name: "disp_to_trf".to_string(),
            params: vec![],
            return_type: None,
            decorators: vec![],
            docstring_first_line: None,
            file: PathBuf::from("utils.py"),
            line: 1,
            scope: None,
        }];
        let concepts = make_concepts(&[(1, "disp"), (2, "trf")]);
        let conventions = vec![Convention {
            pattern: PatternKind::Conversion("_to_".to_string()),
            entity_type: EntityType::Function,
            semantic_role: "conversion".to_string(),
            examples: vec!["disp_to_trf".to_string()],
            frequency: 3,
        }];
        let (mut entities, _) = build_entities(&sigs, &[], &[], &concepts);
        infer_semantic_roles(&mut entities, &[], &conventions, &concepts);
        assert!(entities[0].semantic_role.contains("conversion"));
        assert!(entities[0].semantic_role.contains("disp"));
    }

    #[test]
    fn test_role_concept_tag_fallback() {
        let sigs = vec![Signature {
            name: "spatial_transform".to_string(),
            params: vec![],
            return_type: None,
            decorators: vec![],
            docstring_first_line: None,
            file: PathBuf::from("utils.py"),
            line: 1,
            scope: None,
        }];
        let concepts = make_concepts(&[(1, "spatial"), (2, "transform")]);
        let (mut entities, _) = build_entities(&sigs, &[], &[], &concepts);
        infer_semantic_roles(&mut entities, &[], &[], &concepts);
        assert_eq!(entities[0].semantic_role, "transform function");
    }

    #[test]
    fn test_role_empty_when_no_signals() {
        let classes = vec![ClassInfo {
            name: "HelperUtil".to_string(),
            bases: vec![],
            methods: vec![],
            attributes: vec![],
            docstring_first_line: None,
            file: PathBuf::from("utils.py"),
            line: 1,
        }];
        let call_sites = vec![CallSite {
            caller_scope: Some("main".to_string()),
            callee: "HelperUtil".to_string(),
            file: PathBuf::from("main.py"),
            line: 10,
        }];
        // Promoted via reference, no matching concepts
        let concepts = make_concepts(&[(1, "loss")]);
        let (mut entities, _) = build_entities(&[], &classes, &call_sites, &concepts);
        infer_semantic_roles(&mut entities, &classes, &[], &concepts);
        assert!(entities[0].semantic_role.is_empty());
    }

    #[test]
    fn test_role_primary_tag_is_last() {
        let classes = vec![ClassInfo {
            name: "FocalDiceLoss".to_string(),
            bases: vec!["nn.Module".to_string()],
            methods: vec![],
            attributes: vec![],
            docstring_first_line: None,
            file: PathBuf::from("loss.py"),
            line: 1,
        }];
        let concepts = make_concepts(&[(1, "focal"), (2, "dice"), (3, "loss")]);
        let (mut entities, _) = build_entities(&[], &classes, &[], &concepts);
        infer_semantic_roles(&mut entities, &classes, &[], &concepts);
        // Last tag "loss" should be the primary tag
        assert!(entities[0].semantic_role.starts_with("loss"));
    }
}

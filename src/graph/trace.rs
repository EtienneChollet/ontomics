use super::ConceptGraph;
use crate::tokenizer::split_identifier;
use crate::types::{
    ClassInfo, ConceptTrace, Entity, MethodSummary, RelationshipKind,
    Signature, TraceEdge, TraceNode, TraceRole, TypeFlow, TypeFlowResult,
    TypeFrequency,
};
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;

/// Normalize a type annotation for type-flow tracking.
/// Strips `Optional[...]` to its inner type and `Union[..., None]`
/// to its first non-None variant.
fn normalize_type_annotation(ann: &str) -> String {
    let trimmed = ann.trim();

    // Optional[X] -> X
    if let Some(inner) = trimmed
        .strip_prefix("Optional[")
        .and_then(|s| s.strip_suffix(']'))
    {
        return normalize_type_annotation(inner);
    }

    // Union[X, None] or Union[X, Y, ...] -> first non-None type
    if let Some(inner) = trimmed
        .strip_prefix("Union[")
        .and_then(|s| s.strip_suffix(']'))
    {
        // Simple split on top-level commas (handles nested brackets)
        let parts = split_top_level_commas(inner);
        for part in &parts {
            let p = part.trim();
            if !p.eq_ignore_ascii_case("none") {
                return normalize_type_annotation(p);
            }
        }
    }

    trimmed.to_string()
}

/// Split a string by commas, but only at the top level
/// (not inside brackets).
fn split_top_level_commas(s: &str) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut depth = 0usize;
    let mut start = 0;
    for (i, ch) in s.char_indices() {
        match ch {
            '[' | '(' => depth += 1,
            ']' | ')' => depth = depth.saturating_sub(1),
            ',' if depth == 0 => {
                parts.push(&s[start..i]);
                start = i + 1;
            }
            _ => {}
        }
    }
    parts.push(&s[start..]);
    parts
}

impl ConceptGraph {
    /// Trace a concept through the call graph, showing which
    /// entities produce it, consume it, and bridge between them.
    pub fn trace_concept(
        &self,
        concept_name: &str,
        max_depth: usize,
    ) -> Option<ConceptTrace> {
        use petgraph::graph::{DiGraph, NodeIndex};

        let term_lower = concept_name.to_lowercase();
        let subtokens = split_identifier(concept_name);

        // 1. Resolve: entity-first, then concept fallback.
        let entity_match: Option<&Entity> = self
            .entities
            .values()
            .find(|e| e.name.to_lowercase() == term_lower);

        let is_entity_trace = entity_match.is_some();

        let (concept_label, seed_ids) =
            if let Some(matched) = entity_match {
                let label = self
                    .concepts
                    .get(
                        matched
                            .concept_tags
                            .first()
                            .unwrap_or(&0),
                    )
                    .map(|c| c.canonical.clone())
                    .unwrap_or_else(|| {
                        matched.name.to_lowercase()
                    });
                let ids: HashSet<u64> =
                    std::iter::once(matched.id).collect();
                (label, ids)
            } else {
                // Concept-based fallback
                let concept = self
                    .concepts
                    .values()
                    .find(|c| c.canonical == term_lower)
                    .or_else(|| {
                        subtokens.iter().find_map(|st| {
                            self.concepts
                                .values()
                                .find(|c| c.canonical == *st)
                        })
                    })
                    .or_else(|| {
                        self.concepts.values().find(|c| {
                            c.occurrences.iter().any(|o| {
                                o.identifier
                                    .to_lowercase()
                                    .contains(&term_lower)
                            })
                        })
                    })?;
                let ids: HashSet<u64> = self
                    .entities
                    .values()
                    .filter(|e| {
                        e.concept_tags.contains(&concept.id)
                    })
                    .map(|e| e.id)
                    .collect();
                (concept.canonical.clone(), ids)
            };

        if seed_ids.is_empty() {
            return Some(ConceptTrace {
                concept: concept_label,
                producers: Vec::new(),
                consumers: Vec::new(),
                call_chain: Vec::new(),
                edges: Vec::new(),
            });
        }

        // Collect concept subtokens for role classification
        let role_subtokens: Vec<String> = seed_ids
            .iter()
            .filter_map(|id| self.entities.get(id))
            .flat_map(|e| {
                e.concept_tags.iter().filter_map(|cid| {
                    self.concepts.get(cid)
                })
            })
            .flat_map(|c| c.subtokens.clone())
            .collect();

        // 2. Classify seeds as producer/consumer/both
        let classify =
            |entity: &Entity| -> TraceRole {
                let sig = entity
                    .signature_idx
                    .and_then(|i| self.signatures.get(i));
                let Some(sig) = sig else {
                    return TraceRole::Both;
                };
                let is_producer = sig
                    .return_type
                    .as_ref()
                    .map(|rt| {
                        let rt_lower = rt.to_lowercase();
                        role_subtokens.iter().any(|s| {
                            rt_lower
                                .contains(&s.to_lowercase())
                        })
                    })
                    .unwrap_or(false);
                let is_consumer =
                    sig.params.iter().any(|p| {
                        let name_lower =
                            p.name.to_lowercase();
                        let type_lower = p
                            .type_annotation
                            .as_deref()
                            .unwrap_or("")
                            .to_lowercase();
                        role_subtokens.iter().any(|s| {
                            let s_lower = s.to_lowercase();
                            name_lower.contains(&s_lower)
                                || type_lower
                                    .contains(&s_lower)
                        })
                    });
                match (is_producer, is_consumer) {
                    (true, true) => TraceRole::Both,
                    (true, false) => TraceRole::Producer,
                    (false, true) => TraceRole::Consumer,
                    (false, false) => TraceRole::Both,
                }
            };

        let mut seed_roles: HashMap<u64, TraceRole> =
            HashMap::new();
        for &sid in &seed_ids {
            if let Some(e) = self.entities.get(&sid) {
                seed_roles.insert(sid, classify(e));
            }
        }

        // 3. Build entity-level graph from call sites + entity
        //    relationships.
        let mut call_graph: DiGraph<String, ()> =
            DiGraph::new();
        let mut node_map: HashMap<String, NodeIndex> =
            HashMap::new();

        let ensure_node =
            |g: &mut DiGraph<String, ()>,
             m: &mut HashMap<String, NodeIndex>,
             name: &str|
             -> NodeIndex {
                if let Some(&idx) = m.get(name) {
                    idx
                } else {
                    let idx = g.add_node(name.to_string());
                    m.insert(name.to_string(), idx);
                    idx
                }
            };

        for cs in &self.call_sites {
            let Some(ref caller) = cs.caller_scope else {
                continue;
            };
            let caller_name = caller
                .split('.')
                .next()
                .unwrap_or(caller);
            let callee_name = cs
                .callee
                .rsplit('.')
                .next()
                .unwrap_or(&cs.callee);
            if caller_name == callee_name {
                continue;
            }
            let caller_idx = ensure_node(
                &mut call_graph,
                &mut node_map,
                caller_name,
            );
            let callee_idx = ensure_node(
                &mut call_graph,
                &mut node_map,
                callee_name,
            );
            call_graph.add_edge(caller_idx, callee_idx, ());
        }

        // Inject entity-level relationships
        let entity_id_to_name: HashMap<u64, &str> = self
            .entities
            .values()
            .map(|e| (e.id, e.name.as_str()))
            .collect();

        for rel in &self.relationships {
            let is_entity_rel = matches!(
                rel.kind,
                RelationshipKind::Uses
                    | RelationshipKind::InheritsFrom
            );
            if !is_entity_rel {
                continue;
            }
            let (Some(src), Some(tgt)) = (
                entity_id_to_name.get(&rel.source),
                entity_id_to_name.get(&rel.target),
            ) else {
                continue;
            };
            if src == tgt {
                continue;
            }
            let src_idx = ensure_node(
                &mut call_graph,
                &mut node_map,
                src,
            );
            let tgt_idx = ensure_node(
                &mut call_graph,
                &mut node_map,
                tgt,
            );
            call_graph.add_edge(src_idx, tgt_idx, ());
        }

        // Map seed entity names to node indices
        let seed_node_indices: HashMap<String, NodeIndex> =
            self.entities
                .values()
                .filter(|e| seed_ids.contains(&e.id))
                .filter_map(|e| {
                    node_map.get(&e.name).map(|&idx| {
                        (e.name.clone(), idx)
                    })
                })
                .collect();

        // 4. Find bridges via BFS forward/backward from seeds
        let bfs_reachable = |starts: &[NodeIndex],
                             forward: bool|
         -> HashSet<NodeIndex> {
            let mut visited = HashSet::new();
            let mut queue = VecDeque::new();
            for &s in starts {
                visited.insert(s);
                queue.push_back((s, 0usize));
            }
            while let Some((node, depth)) =
                queue.pop_front()
            {
                if depth >= max_depth {
                    continue;
                }
                let neighbors: Vec<NodeIndex> = if forward
                {
                    call_graph
                        .neighbors_directed(
                            node,
                            petgraph::Direction::Outgoing,
                        )
                        .collect()
                } else {
                    call_graph
                        .neighbors_directed(
                            node,
                            petgraph::Direction::Incoming,
                        )
                        .collect()
                };
                for nb in neighbors {
                    if visited.insert(nb) {
                        queue.push_back((nb, depth + 1));
                    }
                }
            }
            visited
        };

        let seed_entity_by_name: HashMap<&str, &Entity> =
            self.entities
                .values()
                .filter(|e| seed_ids.contains(&e.id))
                .map(|e| (e.name.as_str(), e))
                .collect();

        let producer_indices: Vec<NodeIndex> =
            seed_node_indices
                .iter()
                .filter(|(name, _)| {
                    seed_entity_by_name
                        .get(name.as_str())
                        .and_then(|e| {
                            seed_roles.get(&e.id)
                        })
                        .map(|r| {
                            matches!(
                                r,
                                TraceRole::Producer
                                    | TraceRole::Both
                            )
                        })
                        .unwrap_or(false)
                })
                .map(|(_, &idx)| idx)
                .collect();
        let consumer_indices: Vec<NodeIndex> =
            seed_node_indices
                .iter()
                .filter(|(name, _)| {
                    seed_entity_by_name
                        .get(name.as_str())
                        .and_then(|e| {
                            seed_roles.get(&e.id)
                        })
                        .map(|r| {
                            matches!(
                                r,
                                TraceRole::Consumer
                                    | TraceRole::Both
                            )
                        })
                        .unwrap_or(false)
                })
                .map(|(_, &idx)| idx)
                .collect();
        let forward_set =
            bfs_reachable(&producer_indices, true);
        let backward_set =
            bfs_reachable(&consumer_indices, false);

        let seed_idx_set: HashSet<NodeIndex> =
            seed_node_indices.values().copied().collect();

        let entity_name_set: HashSet<&str> =
            entity_id_to_name.values().copied().collect();

        // 5. Build subgraph nodes.
        let subgraph_nodes: HashSet<NodeIndex> =
            if is_entity_trace {
                let all_seeds: Vec<NodeIndex> =
                    seed_idx_set.iter().copied().collect();
                let reachable_fwd =
                    bfs_reachable(&all_seeds, true);
                let reachable_bwd =
                    bfs_reachable(&all_seeds, false);
                reachable_fwd
                    .union(&reachable_bwd)
                    .copied()
                    .filter(|n| {
                        seed_idx_set.contains(n)
                            || entity_name_set.contains(
                                call_graph[*n].as_str(),
                            )
                    })
                    .collect()
            } else {
                let bridge_indices: HashSet<NodeIndex> =
                    forward_set
                        .intersection(&backward_set)
                        .copied()
                        .filter(|n| {
                            !seed_idx_set.contains(n)
                        })
                        .collect();
                let mut neighbor_indices: HashSet<NodeIndex> =
                    HashSet::new();
                for &seed_idx in seed_idx_set.iter() {
                    for nb in call_graph
                        .neighbors_directed(
                            seed_idx,
                            petgraph::Direction::Outgoing,
                        )
                        .chain(
                            call_graph.neighbors_directed(
                                seed_idx,
                                petgraph::Direction::Incoming,
                            ),
                        )
                    {
                        if entity_name_set.contains(
                            call_graph[nb].as_str(),
                        ) {
                            neighbor_indices.insert(nb);
                        }
                    }
                }
                seed_idx_set
                    .iter()
                    .chain(bridge_indices.iter())
                    .chain(neighbor_indices.iter())
                    .copied()
                    .collect()
            };

        let mut sub: DiGraph<String, ()> = DiGraph::new();
        let mut sub_map: HashMap<NodeIndex, NodeIndex> =
            HashMap::new();
        for &orig in &subgraph_nodes {
            let new =
                sub.add_node(call_graph[orig].clone());
            sub_map.insert(orig, new);
        }
        for &orig in &subgraph_nodes {
            for neighbor in call_graph.neighbors_directed(
                orig,
                petgraph::Direction::Outgoing,
            ) {
                if let (Some(&from), Some(&to)) = (
                    sub_map.get(&orig),
                    sub_map.get(&neighbor),
                ) {
                    sub.add_edge(from, to, ());
                }
            }
        }

        // Topological sort; fall back to BFS from roots
        let ordered: Vec<NodeIndex> =
            match petgraph::algo::toposort(&sub, None) {
                Ok(sorted) => sorted,
                Err(_) => {
                    let roots: Vec<NodeIndex> = sub
                        .node_indices()
                        .filter(|n| {
                            sub.neighbors_directed(
                                *n,
                                petgraph::Direction::Incoming,
                            )
                            .next()
                            .is_none()
                        })
                        .collect();
                    let mut visited = HashSet::new();
                    let mut queue = VecDeque::new();
                    let mut result = Vec::new();
                    for r in roots {
                        if visited.insert(r) {
                            queue.push_back(r);
                        }
                    }
                    if queue.is_empty() {
                        if let Some(n) =
                            sub.node_indices().next()
                        {
                            visited.insert(n);
                            queue.push_back(n);
                        }
                    }
                    while let Some(n) = queue.pop_front() {
                        result.push(n);
                        for nb in sub.neighbors_directed(
                            n,
                            petgraph::Direction::Outgoing,
                        ) {
                            if visited.insert(nb) {
                                queue.push_back(nb);
                            }
                        }
                        if queue.is_empty() {
                            for c in sub.node_indices() {
                                if visited.insert(c) {
                                    queue.push_back(c);
                                    break;
                                }
                            }
                        }
                    }
                    result
                }
            };

        // 6. Assemble result
        let entity_by_name: HashMap<&str, &Entity> = self
            .entities
            .values()
            .map(|e| (e.name.as_str(), e))
            .collect();

        let concept_name_for_id = |cid: &u64| -> String {
            self.concepts
                .get(cid)
                .map(|c| c.canonical.clone())
                .unwrap_or_default()
        };

        let class_by_name: HashMap<&str, &ClassInfo> = self
            .classes
            .iter()
            .map(|c| (c.name.as_str(), c))
            .collect();

        let method_sigs_by_class: HashMap<&str, Vec<&Signature>> = {
            let mut map: HashMap<&str, Vec<&Signature>> =
                HashMap::new();
            for sig in &self.signatures {
                if let Some(ref scope) = sig.scope {
                    let class_name = scope
                        .split('.')
                        .next()
                        .unwrap_or(scope);
                    if class_by_name.contains_key(class_name)
                    {
                        map.entry(class_name)
                            .or_default()
                            .push(sig);
                    }
                }
            }
            map
        };

        let make_trace_node =
            |name: &str, role: TraceRole| -> TraceNode {
                if let Some(entity) =
                    entity_by_name.get(name)
                {
                    let (bases, methods) =
                        if entity.kind
                            == crate::types::EntityKind::Class
                        {
                            let bases = class_by_name
                                .get(name)
                                .map(|c| c.bases.clone())
                                .unwrap_or_default();
                            let methods = method_sigs_by_class
                                .get(name)
                                .map(|sigs| {
                                    sigs.iter()
                                        .map(|s| MethodSummary {
                                            name: s.name.clone(),
                                            params: s
                                                .params
                                                .iter()
                                                .filter(|p| {
                                                    p.name != "self"
                                                })
                                                .map(|p| {
                                                    match &p.type_annotation {
                                                        Some(t) => format!("{}: {}", p.name, t),
                                                        None => p.name.clone(),
                                                    }
                                                })
                                                .collect(),
                                            return_type: s
                                                .return_type
                                                .clone(),
                                            line: s.line,
                                        })
                                        .collect()
                                })
                                .unwrap_or_default();
                            (bases, methods)
                        } else {
                            (Vec::new(), Vec::new())
                        };

                    TraceNode {
                        entity_name: entity.name.clone(),
                        kind: entity.kind.clone(),
                        file: entity.file.clone(),
                        line: entity.line,
                        role,
                        concept_tags: entity
                            .concept_tags
                            .iter()
                            .map(concept_name_for_id)
                            .collect(),
                        bases,
                        methods,
                    }
                } else {
                    TraceNode {
                        entity_name: name.to_string(),
                        kind: crate::types::EntityKind::Function,
                        file: PathBuf::new(),
                        line: 0,
                        role,
                        concept_tags: Vec::new(),
                        bases: Vec::new(),
                        methods: Vec::new(),
                    }
                }
            };

        let mut call_chain = Vec::new();
        for &sub_idx in &ordered {
            let name = &sub[sub_idx];
            let orig_idx = sub_map.iter().find_map(
                |(&orig, &mapped)| {
                    if mapped == sub_idx {
                        Some(orig)
                    } else {
                        None
                    }
                },
            );
            let role = if let Some(oi) = orig_idx {
                if !seed_idx_set.contains(&oi) {
                    TraceRole::Bridge
                } else if let Some(entity) =
                    entity_by_name.get(name.as_str())
                {
                    seed_roles
                        .get(&entity.id)
                        .cloned()
                        .unwrap_or(TraceRole::Both)
                } else {
                    TraceRole::Both
                }
            } else {
                TraceRole::Both
            };
            call_chain.push(make_trace_node(name, role));
        }

        // Include orphan seeds
        let chain_names: HashSet<&str> = call_chain
            .iter()
            .map(|n| n.entity_name.as_str())
            .collect();
        let orphans: Vec<TraceNode> = seed_roles
            .iter()
            .filter_map(|(&sid, role)| {
                let entity = self.entities.get(&sid)?;
                if chain_names
                    .contains(entity.name.as_str())
                {
                    return None;
                }
                Some(make_trace_node(
                    &entity.name,
                    role.clone(),
                ))
            })
            .collect();
        call_chain.extend(orphans);

        // Build edges from the subgraph
        let subgraph_names: HashSet<&str> = subgraph_nodes
            .iter()
            .map(|&n| call_graph[n].as_str())
            .collect();

        let mut edges: Vec<TraceEdge> = Vec::new();
        let mut seen_edges: HashSet<(String, String)> =
            HashSet::new();

        for cs in &self.call_sites {
            let Some(ref caller) = cs.caller_scope else {
                continue;
            };
            let caller_name = caller
                .split('.')
                .next()
                .unwrap_or(caller);
            let callee_name = cs
                .callee
                .rsplit('.')
                .next()
                .unwrap_or(&cs.callee);
            if subgraph_names.contains(caller_name)
                && subgraph_names.contains(callee_name)
                && caller_name != callee_name
            {
                let key = (
                    caller_name.to_string(),
                    callee_name.to_string(),
                );
                if seen_edges.insert(key) {
                    edges.push(TraceEdge {
                        caller: caller_name.to_string(),
                        callee: callee_name.to_string(),
                        file: cs.file.clone(),
                        line: cs.line,
                    });
                }
            }
        }

        for rel in &self.relationships {
            let is_entity_rel = matches!(
                rel.kind,
                RelationshipKind::Uses
                    | RelationshipKind::InheritsFrom
            );
            if !is_entity_rel {
                continue;
            }
            let (Some(src), Some(tgt)) = (
                entity_id_to_name.get(&rel.source),
                entity_id_to_name.get(&rel.target),
            ) else {
                continue;
            };
            if subgraph_names.contains(src)
                && subgraph_names.contains(tgt)
            {
                let key =
                    (src.to_string(), tgt.to_string());
                if seen_edges.insert(key) {
                    let file = self
                        .entities
                        .get(&rel.source)
                        .map(|e| e.file.clone())
                        .unwrap_or_default();
                    let line = self
                        .entities
                        .get(&rel.source)
                        .map(|e| e.line)
                        .unwrap_or(0);
                    edges.push(TraceEdge {
                        caller: src.to_string(),
                        callee: tgt.to_string(),
                        file,
                        line,
                    });
                }
            }
        }

        let producers: Vec<TraceNode> = call_chain
            .iter()
            .filter(|n| {
                matches!(
                    n.role,
                    TraceRole::Producer | TraceRole::Both
                ) && n.role != TraceRole::Bridge
            })
            .cloned()
            .collect();

        let consumers: Vec<TraceNode> = call_chain
            .iter()
            .filter(|n| {
                matches!(
                    n.role,
                    TraceRole::Consumer | TraceRole::Both
                ) && n.role != TraceRole::Bridge
            })
            .cloned()
            .collect();

        Some(ConceptTrace {
            concept: concept_label,
            producers,
            consumers,
            call_chain,
            edges,
        })
    }

    /// Trace how declared type annotations propagate along call
    /// edges. Builds TypeFlow edges from callee return types and
    /// parameter types using existing signatures and call sites.
    pub fn type_flows(&self) -> TypeFlowResult {
        let mut sig_by_name: HashMap<&str, Vec<&Signature>> = HashMap::new();
        for sig in &self.signatures {
            sig_by_name
                .entry(sig.name.as_str())
                .or_default()
                .push(sig);
        }

        let mut flows: Vec<TypeFlow> = Vec::new();
        let mut typed_edges = 0usize;
        let mut untyped_edges = 0usize;

        for cs in &self.call_sites {
            let callee_sig = sig_by_name
                .get(cs.callee.as_str())
                .and_then(|sigs| sigs.first())
                .copied();

            let caller_name = cs
                .caller_scope
                .as_deref()
                .unwrap_or("<module>");

            let callee_sig = match callee_sig {
                Some(s) => s,
                None => {
                    untyped_edges += 1;
                    continue;
                }
            };

            let mut found_type = false;

            if let Some(ref ret) = callee_sig.return_type {
                let normalized = normalize_type_annotation(ret);
                if !normalized.is_empty() {
                    flows.push(TypeFlow {
                        from_entity: cs.callee.clone(),
                        to_entity: caller_name.to_string(),
                        type_name: normalized,
                        from_file: callee_sig.file.clone(),
                        to_file: cs.file.clone(),
                    });
                    found_type = true;
                }
            }

            for param in &callee_sig.params {
                if let Some(ref ann) = param.type_annotation {
                    let normalized = normalize_type_annotation(ann);
                    if !normalized.is_empty() {
                        flows.push(TypeFlow {
                            from_entity: caller_name.to_string(),
                            to_entity: cs.callee.clone(),
                            type_name: normalized,
                            from_file: cs.file.clone(),
                            to_file: callee_sig.file.clone(),
                        });
                        found_type = true;
                    }
                }
            }

            if found_type {
                typed_edges += 1;
            } else {
                untyped_edges += 1;
            }
        }

        let mut type_counts: HashMap<String, usize> = HashMap::new();
        for flow in &flows {
            *type_counts.entry(flow.type_name.clone()).or_insert(0) += 1;
        }
        let mut dominant_types: Vec<TypeFrequency> = type_counts
            .into_iter()
            .map(|(type_name, count)| TypeFrequency { type_name, count })
            .collect();
        dominant_types.sort_by(|a, b| b.count.cmp(&a.count));

        TypeFlowResult {
            flows,
            dominant_types,
            total_typed_edges: typed_edges,
            total_untyped_edges: untyped_edges,
        }
    }

    /// Filter type flows to those involving a specific type name.
    /// Matches by substring (case-insensitive).
    pub fn trace_type(&self, type_name: &str) -> Vec<TypeFlow> {
        let needle = type_name.to_lowercase();
        self.type_flows()
            .flows
            .into_iter()
            .filter(|f| f.type_name.to_lowercase().contains(&needle))
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
                    &["spatial_transform", "apply_transform", "transform"],
                ),
                make_concept(2, "spatial", &["spatial_transform"]),
                make_concept(3, "ndim", &["ndim", "ndim", "ndim"]),
                make_concept(
                    4, "nb",
                    &["nb_features", "nb_bins", "nb_steps", "nb_dims"],
                ),
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
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
            nesting_trees: Vec::new(),
            doc_texts: Vec::new(),
        };
        ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap()
    }

    #[test]
    fn test_trace_concept_basic() {
        let concept = make_concept(
            1,
            "transform",
            &["apply_transform", "transform"],
        );
        let sigs = vec![
            Signature {
                name: "load_data".to_string(),
                params: Vec::new(),
                return_type: Some("Data".to_string()),
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("test.py"),
                line: 1,
                scope: None,
                body: None,
            },
            Signature {
                name: "apply_transform".to_string(),
                params: vec![Param {
                    name: "data".to_string(),
                    type_annotation: Some("Data".to_string()),
                    default: None,
                }],
                return_type: Some("Transform".to_string()),
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("test.py"),
                line: 10,
                scope: None,
                body: None,
            },
            Signature {
                name: "save_result".to_string(),
                params: vec![Param {
                    name: "trf".to_string(),
                    type_annotation: Some("Transform".to_string()),
                    default: None,
                }],
                return_type: None,
                decorators: Vec::new(),
                docstring_first_line: None,
                file: PathBuf::from("test.py"),
                line: 20,
                scope: None,
                body: None,
            },
        ];
        let entities = vec![
            Entity {
                id: 10,
                name: "apply_transform".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "transform".to_string(),
                file: PathBuf::from("test.py"),
                line: 10,
                signature_idx: Some(1),
                class_info_idx: None,
            },
            Entity {
                id: 11,
                name: "save_result".to_string(),
                kind: EntityKind::Function,
                concept_tags: vec![1],
                semantic_role: "io".to_string(),
                file: PathBuf::from("test.py"),
                line: 20,
                signature_idx: Some(2),
                class_info_idx: None,
            },
        ];
        let call_sites = vec![
            CallSite {
                caller_scope: Some("load_data".to_string()),
                callee: "apply_transform".to_string(),
                file: PathBuf::from("test.py"),
                line: 5,
            },
            CallSite {
                caller_scope: Some("apply_transform".to_string()),
                callee: "save_result".to_string(),
                file: PathBuf::from("test.py"),
                line: 15,
            },
        ];
        let analysis = AnalysisResult {
            concepts: vec![concept],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: sigs,
            classes: Vec::new(),
            call_sites,
            nesting_trees: Vec::new(),
            doc_texts: Vec::new(),
        };
        let graph = ConceptGraph::build_with_entities(
            analysis, EmbeddingIndex::empty(), entities, Vec::new(), Vec::new(),
        ).unwrap();

        let result = graph.trace_concept("transform", 5);
        assert!(result.is_some());
        let trace = result.unwrap();
        assert_eq!(trace.concept, "transform");
        assert!(!trace.call_chain.is_empty());
        let chain_names: Vec<&str> = trace
            .call_chain.iter().map(|n| n.entity_name.as_str()).collect();
        assert!(chain_names.contains(&"apply_transform"));
        assert!(chain_names.contains(&"save_result"));
        assert!(!trace.producers.is_empty());
        assert!(!trace.consumers.is_empty());
    }

    #[test]
    fn test_trace_concept_bridge_entities() {
        let concept = make_concept(
            1, "transform",
            &["produce_transform", "consume_transform"],
        );
        let entities = vec![
            Entity {
                id: 10, name: "produce_transform".to_string(),
                kind: EntityKind::Function, concept_tags: vec![1],
                semantic_role: "transform".to_string(),
                file: PathBuf::from("test.py"), line: 1,
                signature_idx: None, class_info_idx: None,
            },
            Entity {
                id: 11, name: "helper".to_string(),
                kind: EntityKind::Function, concept_tags: Vec::new(),
                semantic_role: "util".to_string(),
                file: PathBuf::from("test.py"), line: 10,
                signature_idx: None, class_info_idx: None,
            },
            Entity {
                id: 12, name: "consume_transform".to_string(),
                kind: EntityKind::Function, concept_tags: vec![1],
                semantic_role: "transform".to_string(),
                file: PathBuf::from("test.py"), line: 20,
                signature_idx: None, class_info_idx: None,
            },
        ];
        let call_sites = vec![
            CallSite {
                caller_scope: Some("produce_transform".to_string()),
                callee: "helper".to_string(),
                file: PathBuf::from("test.py"), line: 5,
            },
            CallSite {
                caller_scope: Some("helper".to_string()),
                callee: "consume_transform".to_string(),
                file: PathBuf::from("test.py"), line: 15,
            },
        ];
        let analysis = AnalysisResult {
            concepts: vec![concept], conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(), signatures: Vec::new(),
            classes: Vec::new(), call_sites,
            nesting_trees: Vec::new(), doc_texts: Vec::new(),
        };
        let graph = ConceptGraph::build_with_entities(
            analysis, EmbeddingIndex::empty(), entities, Vec::new(), Vec::new(),
        ).unwrap();

        let result = graph.trace_concept("transform", 5);
        assert!(result.is_some());
        let trace = result.unwrap();
        let bridge = trace.call_chain.iter().find(|n| n.entity_name == "helper");
        assert!(bridge.is_some());
        assert_eq!(bridge.unwrap().role, TraceRole::Bridge);
    }

    #[test]
    fn test_trace_concept_not_found() {
        let graph = make_test_graph();
        assert!(graph.trace_concept("nonexistent", 5).is_none());
    }

    #[test]
    fn test_trace_concept_no_call_sites__orphan_seeds__appear_in_chain_with_empty_edges() {
        let concept = make_concept(1, "loss", &["dice_loss", "ncc_loss"]);
        let entities = vec![
            Entity {
                id: 20, name: "dice_loss".to_string(),
                kind: EntityKind::Function, concept_tags: vec![1],
                semantic_role: "loss".to_string(),
                file: PathBuf::from("losses.py"), line: 1,
                signature_idx: None, class_info_idx: None,
            },
            Entity {
                id: 21, name: "ncc_loss".to_string(),
                kind: EntityKind::Function, concept_tags: vec![1],
                semantic_role: "loss".to_string(),
                file: PathBuf::from("losses.py"), line: 10,
                signature_idx: None, class_info_idx: None,
            },
        ];
        let analysis = AnalysisResult {
            concepts: vec![concept], conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(), signatures: Vec::new(),
            classes: Vec::new(), call_sites: Vec::new(),
            nesting_trees: Vec::new(), doc_texts: Vec::new(),
        };
        let graph = ConceptGraph::build_with_entities(
            analysis, EmbeddingIndex::empty(), entities, Vec::new(), Vec::new(),
        ).unwrap();

        let trace = graph.trace_concept("loss", 5).expect("concept exists");
        let chain_names: Vec<&str> = trace.call_chain.iter()
            .map(|n| n.entity_name.as_str()).collect();
        assert!(chain_names.contains(&"dice_loss"));
        assert!(chain_names.contains(&"ncc_loss"));
        assert!(trace.edges.is_empty());
        assert!(!trace.producers.is_empty());
        assert!(!trace.consumers.is_empty());
    }

    #[test]
    fn test_trace_concept_cyclic_calls__toposort_fallback__all_nodes_in_chain() {
        let concept = make_concept(1, "segment", &["seg_a", "seg_b", "seg_c"]);
        let entities = vec![
            Entity {
                id: 30, name: "seg_a".to_string(),
                kind: EntityKind::Function, concept_tags: vec![1],
                semantic_role: "segment".to_string(),
                file: PathBuf::from("seg.py"), line: 1,
                signature_idx: None, class_info_idx: None,
            },
            Entity {
                id: 31, name: "seg_b".to_string(),
                kind: EntityKind::Function, concept_tags: vec![1],
                semantic_role: "segment".to_string(),
                file: PathBuf::from("seg.py"), line: 10,
                signature_idx: None, class_info_idx: None,
            },
            Entity {
                id: 32, name: "seg_c".to_string(),
                kind: EntityKind::Function, concept_tags: vec![1],
                semantic_role: "segment".to_string(),
                file: PathBuf::from("seg.py"), line: 20,
                signature_idx: None, class_info_idx: None,
            },
        ];
        let call_sites = vec![
            CallSite { caller_scope: Some("seg_a".to_string()),
                callee: "seg_b".to_string(),
                file: PathBuf::from("seg.py"), line: 5 },
            CallSite { caller_scope: Some("seg_b".to_string()),
                callee: "seg_c".to_string(),
                file: PathBuf::from("seg.py"), line: 15 },
            CallSite { caller_scope: Some("seg_c".to_string()),
                callee: "seg_a".to_string(),
                file: PathBuf::from("seg.py"), line: 25 },
        ];
        let analysis = AnalysisResult {
            concepts: vec![concept], conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(), signatures: Vec::new(),
            classes: Vec::new(), call_sites,
            nesting_trees: Vec::new(), doc_texts: Vec::new(),
        };
        let graph = ConceptGraph::build_with_entities(
            analysis, EmbeddingIndex::empty(), entities, Vec::new(), Vec::new(),
        ).unwrap();

        let trace = graph.trace_concept("segment", 5).expect("concept exists");
        let chain_names: Vec<&str> = trace.call_chain.iter()
            .map(|n| n.entity_name.as_str()).collect();
        assert!(chain_names.contains(&"seg_a"));
        assert!(chain_names.contains(&"seg_b"));
        assert!(chain_names.contains(&"seg_c"));
    }

    #[test]
    fn test_trace_concept_producer_consumer_classification__roles_match_signature() {
        let concept = make_concept(1, "vol", &["make_vol", "apply_vol", "process_vol"]);
        let sigs = vec![
            Signature {
                name: "make_vol".to_string(), params: Vec::new(),
                return_type: Some("Vol".to_string()), decorators: Vec::new(),
                docstring_first_line: None, file: PathBuf::from("vol.py"),
                line: 1, scope: None, body: None,
            },
            Signature {
                name: "apply_vol".to_string(),
                params: vec![Param {
                    name: "src".to_string(),
                    type_annotation: Some("Vol".to_string()), default: None,
                }],
                return_type: None, decorators: Vec::new(),
                docstring_first_line: None, file: PathBuf::from("vol.py"),
                line: 10, scope: None, body: None,
            },
            Signature {
                name: "process_vol".to_string(),
                params: vec![Param {
                    name: "vol".to_string(), type_annotation: None, default: None,
                }],
                return_type: Some("Vol".to_string()), decorators: Vec::new(),
                docstring_first_line: None, file: PathBuf::from("vol.py"),
                line: 20, scope: None, body: None,
            },
        ];
        let entities = vec![
            Entity { id: 40, name: "make_vol".to_string(), kind: EntityKind::Function,
                concept_tags: vec![1], semantic_role: "vol".to_string(),
                file: PathBuf::from("vol.py"), line: 1,
                signature_idx: Some(0), class_info_idx: None },
            Entity { id: 41, name: "apply_vol".to_string(), kind: EntityKind::Function,
                concept_tags: vec![1], semantic_role: "vol".to_string(),
                file: PathBuf::from("vol.py"), line: 10,
                signature_idx: Some(1), class_info_idx: None },
            Entity { id: 42, name: "process_vol".to_string(), kind: EntityKind::Function,
                concept_tags: vec![1], semantic_role: "vol".to_string(),
                file: PathBuf::from("vol.py"), line: 20,
                signature_idx: Some(2), class_info_idx: None },
        ];
        let analysis = AnalysisResult {
            concepts: vec![concept], conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(), signatures: sigs,
            classes: Vec::new(), call_sites: Vec::new(),
            nesting_trees: Vec::new(), doc_texts: Vec::new(),
        };
        let graph = ConceptGraph::build_with_entities(
            analysis, EmbeddingIndex::empty(), entities, Vec::new(), Vec::new(),
        ).unwrap();

        let trace = graph.trace_concept("vol", 5).expect("concept exists");
        let role_of = |name: &str| -> TraceRole {
            trace.call_chain.iter().find(|n| n.entity_name == name)
                .unwrap_or_else(|| panic!("{name} must be in call_chain"))
                .role.clone()
        };
        assert_eq!(role_of("make_vol"), TraceRole::Producer);
        assert_eq!(role_of("apply_vol"), TraceRole::Consumer);
        assert_eq!(role_of("process_vol"), TraceRole::Both);
    }

    #[test]
    fn test_type_flows_basic() {
        let analysis = AnalysisResult {
            concepts: vec![make_concept(1, "transform", &["transform"])],
            conventions: Vec::new(), co_occurrence_matrix: Vec::new(),
            signatures: vec![
                Signature {
                    name: "transform".to_string(),
                    params: vec![Param {
                        name: "image".to_string(),
                        type_annotation: Some("Tensor".to_string()), default: None,
                    }],
                    return_type: Some("Tensor".to_string()), decorators: Vec::new(),
                    docstring_first_line: None, file: PathBuf::from("ops.py"),
                    line: 10, scope: None, body: None,
                },
                Signature {
                    name: "warp".to_string(),
                    params: vec![Param {
                        name: "field".to_string(),
                        type_annotation: Some("Tensor".to_string()), default: None,
                    }],
                    return_type: Some("Tensor".to_string()), decorators: Vec::new(),
                    docstring_first_line: None, file: PathBuf::from("warp.py"),
                    line: 5, scope: None, body: None,
                },
            ],
            classes: Vec::new(),
            call_sites: vec![CallSite {
                caller_scope: Some("warp".to_string()),
                callee: "transform".to_string(),
                file: PathBuf::from("warp.py"), line: 8,
            }],
            nesting_trees: Vec::new(), doc_texts: Vec::new(),
        };
        let graph = ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();
        let result = graph.type_flows();
        assert!(result.flows.len() >= 2);
        assert!(result.total_typed_edges >= 1);
        assert!(result.flows.iter().all(|f| f.type_name == "Tensor"));
        assert!(!result.dominant_types.is_empty());
        assert_eq!(result.dominant_types[0].type_name, "Tensor");
    }

    #[test]
    fn test_type_flows_untyped_signatures() {
        let analysis = AnalysisResult {
            concepts: Vec::new(), conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: vec![Signature {
                name: "foo".to_string(),
                params: vec![Param {
                    name: "x".to_string(), type_annotation: None, default: None,
                }],
                return_type: None, decorators: Vec::new(),
                docstring_first_line: None, file: PathBuf::from("a.py"),
                line: 1, scope: None, body: None,
            }],
            classes: Vec::new(),
            call_sites: vec![CallSite {
                caller_scope: Some("bar".to_string()),
                callee: "foo".to_string(),
                file: PathBuf::from("b.py"), line: 5,
            }],
            nesting_trees: Vec::new(), doc_texts: Vec::new(),
        };
        let graph = ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();
        let result = graph.type_flows();
        assert!(result.flows.is_empty());
        assert_eq!(result.total_typed_edges, 0);
        assert_eq!(result.total_untyped_edges, 1);
    }

    #[test]
    fn test_type_flows_dominant_types_ranking() {
        let analysis = AnalysisResult {
            concepts: Vec::new(), conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: vec![
                Signature {
                    name: "load".to_string(),
                    params: vec![Param {
                        name: "path".to_string(),
                        type_annotation: Some("str".to_string()), default: None,
                    }],
                    return_type: Some("Tensor".to_string()), decorators: Vec::new(),
                    docstring_first_line: None, file: PathBuf::from("io.py"),
                    line: 1, scope: None, body: None,
                },
                Signature {
                    name: "save".to_string(),
                    params: vec![
                        Param { name: "data".to_string(),
                            type_annotation: Some("Tensor".to_string()), default: None },
                        Param { name: "path".to_string(),
                            type_annotation: Some("str".to_string()), default: None },
                    ],
                    return_type: None, decorators: Vec::new(),
                    docstring_first_line: None, file: PathBuf::from("io.py"),
                    line: 10, scope: None, body: None,
                },
            ],
            classes: Vec::new(),
            call_sites: vec![
                CallSite { caller_scope: Some("main".to_string()),
                    callee: "load".to_string(),
                    file: PathBuf::from("run.py"), line: 5 },
                CallSite { caller_scope: Some("main".to_string()),
                    callee: "save".to_string(),
                    file: PathBuf::from("run.py"), line: 8 },
            ],
            nesting_trees: Vec::new(), doc_texts: Vec::new(),
        };
        let graph = ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();
        let result = graph.type_flows();
        assert!(!result.flows.is_empty());
        assert!(result.dominant_types.len() >= 2);
        for i in 1..result.dominant_types.len() {
            assert!(result.dominant_types[i - 1].count >= result.dominant_types[i].count);
        }
    }

    #[test]
    fn test_trace_type_filters() {
        let analysis = AnalysisResult {
            concepts: Vec::new(), conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: vec![Signature {
                name: "process".to_string(),
                params: vec![Param {
                    name: "img".to_string(),
                    type_annotation: Some("torch.Tensor".to_string()), default: None,
                }],
                return_type: Some("np.ndarray".to_string()), decorators: Vec::new(),
                docstring_first_line: None, file: PathBuf::from("proc.py"),
                line: 1, scope: None, body: None,
            }],
            classes: Vec::new(),
            call_sites: vec![CallSite {
                caller_scope: Some("main".to_string()),
                callee: "process".to_string(),
                file: PathBuf::from("run.py"), line: 5,
            }],
            nesting_trees: Vec::new(), doc_texts: Vec::new(),
        };
        let graph = ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap();

        let tensor_flows = graph.trace_type("Tensor");
        assert!(!tensor_flows.is_empty());
        assert!(tensor_flows.iter().all(|f| f.type_name.contains("Tensor")));

        let array_flows = graph.trace_type("ndarray");
        assert!(!array_flows.is_empty());
        assert!(array_flows.iter().all(|f| f.type_name.contains("ndarray")));

        assert!(graph.trace_type("DataFrame").is_empty());
    }

    #[test]
    fn test_normalize_type_annotation_optional() {
        assert_eq!(super::normalize_type_annotation("Optional[Tensor]"), "Tensor");
    }

    #[test]
    fn test_normalize_type_annotation_union_with_none() {
        assert_eq!(super::normalize_type_annotation("Union[Tensor, None]"), "Tensor");
    }

    #[test]
    fn test_normalize_type_annotation_plain() {
        assert_eq!(super::normalize_type_annotation("torch.Tensor"), "torch.Tensor");
    }

    #[test]
    fn test_normalize_type_annotation_nested_optional() {
        assert_eq!(super::normalize_type_annotation("Optional[List[int]]"), "List[int]");
    }
}

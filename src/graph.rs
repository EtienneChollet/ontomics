use crate::embeddings::EmbeddingIndex;
use crate::tokenizer::{find_abbreviation, split_identifier};
use crate::types::{
    AnalysisResult, CallSite, ClassInfo, Concept, ConceptQueryResult,
    Convention, DescribeSymbolResult, Entity, LocateConceptResult,
    NameSuggestion, NamingCheckResult, PatternKind, QueryConceptParams,
    RelatedConcept, Relationship, RelationshipKind, SessionBriefing,
    Signature, Subconcept, SymbolKind, Verdict,
};
use anyhow::Result;
use std::collections::{HashMap, HashSet};

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

pub struct ConceptGraph {
    pub concepts: HashMap<u64, Concept>,
    pub relationships: Vec<Relationship>,
    pub conventions: Vec<Convention>,
    pub embeddings: EmbeddingIndex,
    pub signatures: Vec<Signature>,
    pub classes: Vec<ClassInfo>,
    pub call_sites: Vec<CallSite>,
    pub entities: HashMap<u64, Entity>,
}

impl ConceptGraph {
    /// Build graph from analysis results + embeddings + entities.
    pub fn build(
        analysis: AnalysisResult,
        embeddings: EmbeddingIndex,
    ) -> Result<Self> {
        Self::build_with_entities(analysis, embeddings, Vec::new(), Vec::new())
    }

    /// Build graph with pre-built entities and their relationships.
    pub fn build_with_entities(
        analysis: AnalysisResult,
        embeddings: EmbeddingIndex,
        entities: Vec<Entity>,
        entity_relationships: Vec<Relationship>,
    ) -> Result<Self> {
        let concepts: HashMap<u64, Concept> = analysis
            .concepts
            .into_iter()
            .map(|c| (c.id, c))
            .collect();

        let mut relationships: Vec<Relationship> = analysis
            .co_occurrence_matrix
            .into_iter()
            .map(|((src, tgt), weight)| Relationship {
                source: src,
                target: tgt,
                kind: RelationshipKind::CoOccurs,
                weight,
            })
            .collect();

        relationships.extend(entity_relationships);

        let entity_map: HashMap<u64, Entity> =
            entities.into_iter().map(|e| (e.id, e)).collect();

        Ok(Self {
            concepts,
            relationships,
            conventions: analysis.conventions,
            embeddings,
            signatures: analysis.signatures,
            classes: analysis.classes,
            call_sites: analysis.call_sites,
            entities: entity_map,
        })
    }

    /// Look up a concept by name (exact or fuzzy match via embeddings).
    pub fn query_concept(
        &self,
        term: &str,
        params: &QueryConceptParams,
    ) -> Option<ConceptQueryResult> {
        let term_lower = term.to_lowercase();
        let subtokens = split_identifier(term);

        // 1. Exact match on canonical name
        let matched = self
            .concepts
            .values()
            .find(|c| c.canonical == term_lower)
            .or_else(|| {
                // Match any concept whose canonical equals one of the
                // query's subtokens
                subtokens.iter().find_map(|st| {
                    self.concepts.values().find(|c| c.canonical == *st)
                })
            })
            .or_else(|| {
                // 2. Match by occurrence identifier
                self.concepts.values().find(|c| {
                    c.occurrences
                        .iter()
                        .any(|o| o.identifier.to_lowercase() == term_lower)
                })
            });

        // 3. Embedding-based fuzzy search (if no exact/occurrence match)
        let matched = matched.or_else(|| {
            // Try to find a concept whose embedding is close to the term.
            // We need an embedding for the query — look for a concept whose
            // canonical is a subtoken of the query as a proxy.
            let query_vec = subtokens.iter().find_map(|st| {
                self.concepts.values().find_map(|c| {
                    if c.canonical == *st {
                        self.embeddings.get_vector(c.id)
                    } else {
                        None
                    }
                })
            });
            if let Some(qv) = query_vec {
                let similar = self.embeddings.find_similar(qv, 1);
                similar
                    .first()
                    .and_then(|(id, score)| {
                        if *score > 0.5 {
                            self.concepts.get(id)
                        } else {
                            None
                        }
                    })
            } else {
                None
            }
        });

        let concept = matched?;

        // Collect variants: all unique identifier names from this concept's
        // occurrences, PLUS identifiers from other concepts that contain the
        // search term in their identifiers.
        let mut variants: Vec<String> = concept
            .occurrences
            .iter()
            .map(|o| o.identifier.clone())
            .collect();

        // Also gather identifiers from ALL concepts that contain the term
        for other in self.concepts.values() {
            if other.id == concept.id {
                continue;
            }
            for occ in &other.occurrences {
                let id_lower = occ.identifier.to_lowercase();
                if id_lower.contains(&term_lower) {
                    variants.push(occ.identifier.clone());
                }
            }
        }

        // Check for abbreviation relationships: find concepts whose
        // canonical is an abbreviation of the search term (or vice versa)
        let term_as_slice = [term_lower.clone()];
        for other in self.concepts.values() {
            if other.id == concept.id {
                continue;
            }
            let canon_as_slice = [other.canonical.clone()];
            // Is other's canonical an abbreviation of the search term?
            let is_abbrev = find_abbreviation(
                &other.canonical,
                &term_as_slice,
            )
            .is_some();
            // Is the search term an abbreviation of other's canonical?
            let is_expansion =
                find_abbreviation(&term_lower, &canon_as_slice)
                    .is_some();
            if is_abbrev || is_expansion {
                for occ in &other.occurrences {
                    variants.push(occ.identifier.clone());
                }
            }
        }

        variants.sort();
        variants.dedup();
        variants.truncate(params.max_variants);

        // Related concepts via relationships (lightweight summaries)
        let mut related: Vec<RelatedConcept> = self
            .relationships
            .iter()
            .filter_map(|rel| {
                let other_id = if rel.source == concept.id {
                    Some(rel.target)
                } else if rel.target == concept.id {
                    Some(rel.source)
                } else {
                    None
                }?;
                let other = self.concepts.get(&other_id)?;
                Some(RelatedConcept {
                    canonical: other.canonical.clone(),
                    kind: rel.kind.clone(),
                    weight: rel.weight,
                    occurrences: other.occurrences.len(),
                })
            })
            .collect();
        // Contrastive first, then by weight descending
        related.sort_by(|a, b| {
            let a_contrastive =
                a.kind == RelationshipKind::Contrastive;
            let b_contrastive =
                b.kind == RelationshipKind::Contrastive;
            b_contrastive
                .cmp(&a_contrastive)
                .then(b.weight.partial_cmp(&a.weight).unwrap_or(std::cmp::Ordering::Equal))
        });
        related.truncate(params.max_related);

        // Conventions that match any of this concept's identifiers
        let concept_identifiers: Vec<String> = concept
            .occurrences
            .iter()
            .map(|o| o.identifier.clone())
            .collect();
        let conventions: Vec<Convention> = self
            .conventions
            .iter()
            .filter(|conv| {
                conv.examples
                    .iter()
                    .any(|ex| concept_identifiers.contains(ex))
            })
            .cloned()
            .collect();

        let top_occurrences: Vec<_> = concept
            .occurrences
            .iter()
            .take(params.max_occurrences)
            .cloned()
            .collect();

        // L2: filter signatures matching this concept
        let mut matching_signatures: Vec<Signature> = self
            .signatures
            .iter()
            .filter(|sig| {
                concept_identifiers.contains(&sig.name)
                    || concept
                        .subtokens
                        .iter()
                        .any(|st| sig.name.to_lowercase().contains(st.as_str()))
            })
            .cloned()
            .collect();
        matching_signatures.truncate(params.max_signatures);

        // L2: filter classes matching this concept
        let matching_classes: Vec<ClassInfo> = self
            .classes
            .iter()
            .filter(|cls| {
                concept_identifiers.contains(&cls.name)
                    || cls.methods.iter().any(|m| {
                        concept_identifiers.iter().any(|id| id == m)
                    })
                    || concept.subtokens.iter().any(|st| {
                        cls.name.to_lowercase().contains(st.as_str())
                    })
            })
            .cloned()
            .collect();

        // L2: build call graph pairs involving matching signatures
        let sig_names: Vec<&str> =
            matching_signatures.iter().map(|s| s.name.as_str()).collect();
        let call_graph: Vec<(String, String)> = self
            .call_sites
            .iter()
            .filter(|cs| {
                sig_names.iter().any(|sn| {
                    cs.callee.contains(sn)
                        || cs.caller_scope.as_deref() == Some(sn)
                        || cs
                            .caller_scope
                            .as_deref()
                            .is_some_and(|s| s.ends_with(&format!(".{sn}")))
                })
            })
            .map(|cs| {
                (
                    cs.caller_scope.clone().unwrap_or_default(),
                    cs.callee.clone(),
                )
            })
            .collect();

        // Collect entities that instantiate this concept
        let entities: Vec<_> = self
            .entities
            .values()
            .filter(|e| e.concept_tags.contains(&concept.id))
            .take(params.max_entities)
            .map(|e| e.summary())
            .collect();

        Some(ConceptQueryResult {
            concept: concept.clone(),
            variants,
            related,
            conventions,
            top_occurrences,
            signatures: matching_signatures,
            classes: matching_classes,
            call_graph,
            subconcepts: concept.subconcepts.clone(),
            entities,
        })
    }

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

        // If a related identifier has strictly higher frequency, suggest it
        if let Some((best_name, best_freq)) = related_ids.first() {
            if *best_freq > input_freq {
                return NamingCheckResult {
                    input: identifier.to_string(),
                    subtokens,
                    verdict: Verdict::Inconsistent,
                    reason: format!(
                        "'{}' appears {} times vs '{}' appears {} times",
                        best_name, best_freq, id_lower, input_freq,
                    ),
                    suggestion: Some(best_name.clone()),
                    matching_convention: None,
                    similar_identifiers: similar_from_related(
                        Some(best_name.as_str()),
                    ),
                };
            }
        }

        // 2. Check prefix conventions — if the identifier uses a prefix
        // that conflicts with a convention for the same semantic role
        let known_count_prefixes = ["n_", "nb_", "num_"];
        for conv in &self.conventions {
            if let PatternKind::Prefix(conv_prefix) = &conv.pattern {
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
                            conv_prefix, conv.semantic_role, alt_prefix
                        ),
                        suggestion: Some(suggested),
                        matching_convention: Some(conv.clone()),
                        similar_identifiers: find_similar_identifiers(
                            identifier,
                            &id_freq,
                        ),
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
            };
        }

        // 5. Unknown — identifier not found, no convention match
        NamingCheckResult {
            input: identifier.to_string(),
            subtokens,
            verdict: Verdict::Unknown,
            reason: "identifier not found in corpus and matches no convention"
                .to_string(),
            suggestion: None,
            matching_convention: None,
            similar_identifiers: similar_from_related(None),
        }
    }

    /// Suggest an identifier name given a natural language description.
    pub fn suggest_name(
        &self,
        description: &str,
    ) -> Vec<NameSuggestion> {
        let words: Vec<String> = description
            .split_whitespace()
            .map(|w| w.to_lowercase())
            .collect();

        let mut suggestions = Vec::new();

        // Find matching conventions by semantic role
        let role_keywords: &[(&str, &str)] = &[
            ("count", "count"),
            ("number", "count"),
            ("boolean", "boolean predicate"),
            ("flag", "boolean predicate"),
            ("convert", "conversion"),
            ("conversion", "conversion"),
        ];

        let mut matched_convention: Option<&Convention> = None;
        for word in &words {
            for &(keyword, role) in role_keywords {
                if word == keyword {
                    matched_convention = self.conventions.iter().find(
                        |c| c.semantic_role == role,
                    );
                    break;
                }
            }
            if matched_convention.is_some() {
                break;
            }
        }

        // Find matching concepts by word overlap
        let matched_concepts: Vec<&Concept> = self
            .concepts
            .values()
            .filter(|c| {
                words.iter().any(|w| {
                    c.canonical == *w
                        || c.occurrences.iter().any(|o| {
                            o.identifier.to_lowercase().contains(w.as_str())
                        })
                })
            })
            .collect();

        // Generate suggestions by combining convention prefix with concept
        if let Some(conv) = matched_convention {
            if let PatternKind::Prefix(prefix) = &conv.pattern {
                for concept in &matched_concepts {
                    let name = format!("{}{}", prefix, concept.canonical);
                    // Higher confidence if this exact name exists in corpus
                    let exists = concept
                        .occurrences
                        .iter()
                        .any(|o| o.identifier == name);
                    let confidence = if exists { 0.9 } else { 0.6 };

                    suggestions.push(NameSuggestion {
                        name,
                        confidence,
                        based_on: conv.examples.clone(),
                    });
                }
            }
            if let PatternKind::Conversion(sep) = &conv.pattern {
                // For conversion, look for two concept words
                if matched_concepts.len() >= 2 {
                    let name = format!(
                        "{}{}{}",
                        matched_concepts[0].canonical,
                        sep,
                        matched_concepts[1].canonical,
                    );
                    suggestions.push(NameSuggestion {
                        name,
                        confidence: 0.5,
                        based_on: conv.examples.clone(),
                    });
                }
            }
        }

        // If no convention matched but we have concepts, suggest based on
        // the most common identifier form for each concept
        if suggestions.is_empty() {
            for concept in &matched_concepts {
                // Find most common identifier for this concept
                let mut id_counts: HashMap<&str, usize> = HashMap::new();
                for occ in &concept.occurrences {
                    *id_counts.entry(&occ.identifier).or_insert(0) += 1;
                }
                if let Some((&best, _)) =
                    id_counts.iter().max_by_key(|(_, &c)| c)
                {
                    suggestions.push(NameSuggestion {
                        name: best.to_string(),
                        confidence: 0.4,
                        based_on: vec![format!(
                            "concept '{}'",
                            concept.canonical
                        )],
                    });
                }
            }
        }

        // Sort by confidence descending, take top 5
        suggestions
            .sort_by(|a, b| b.confidence.partial_cmp(&a.confidence).unwrap());
        suggestions.truncate(5);
        suggestions
    }

    /// List all detected conventions.
    pub fn list_conventions(&self) -> &[Convention] {
        &self.conventions
    }

    /// Add SimilarTo relationship edges between concepts whose embeddings
    /// have cosine similarity above the given threshold.
    pub fn add_similarity_edges(&mut self, threshold: f32) {
        // Clear existing similarity edges so this is idempotent.
        self.relationships
            .retain(|r| r.kind != RelationshipKind::SimilarTo);
        let ids: Vec<u64> = self.concepts.keys().copied().collect();
        for (i, &id_a) in ids.iter().enumerate() {
            let vec_a = match self.embeddings.get_vector(id_a) {
                Some(v) => v.clone(),
                None => continue,
            };
            for &id_b in &ids[i + 1..] {
                let vec_b = match self.embeddings.get_vector(id_b) {
                    Some(v) => v,
                    None => continue,
                };
                let sim = cosine_similarity(&vec_a, vec_b);
                if sim > threshold {
                    self.relationships.push(Relationship {
                        source: id_a,
                        target: id_b,
                        kind: RelationshipKind::SimilarTo,
                        weight: sim,
                    });
                }
            }
        }
    }

    /// List all concepts, ordered by frequency (descending).
    pub fn list_concepts(&self) -> Vec<&Concept> {
        let mut concepts: Vec<&Concept> =
            self.concepts.values().collect();
        concepts.sort_by(|a, b| b.occurrences.len().cmp(&a.occurrences.len()));
        concepts
    }

    /// List entities with optional filtering by concept, semantic role, and kind.
    pub fn list_entities(
        &self,
        concept_filter: Option<&str>,
        role_filter: Option<&str>,
        kind_filter: Option<&crate::types::EntityKind>,
        top_k: usize,
    ) -> Vec<crate::types::EntitySummary> {
        let concept_id: Option<u64> = concept_filter.and_then(|term| {
            let term_lower = term.to_lowercase();
            self.concepts
                .values()
                .find(|c| c.canonical == term_lower)
                .map(|c| c.id)
        });

        let mut matched: Vec<_> = self
            .entities
            .values()
            .filter(|e| {
                if let Some(cid) = concept_id {
                    if !e.concept_tags.contains(&cid) {
                        return false;
                    }
                }
                if let Some(role) = role_filter {
                    let role_lower = role.to_lowercase();
                    if !e.semantic_role.to_lowercase().contains(&role_lower) {
                        return false;
                    }
                }
                if let Some(kind) = kind_filter {
                    if e.kind != *kind {
                        return false;
                    }
                }
                true
            })
            .collect();
        matched.sort_by(|a, b| a.name.cmp(&b.name));
        matched
            .into_iter()
            .take(top_k)
            .map(|e| e.summary())
            .collect()
    }

    /// Describe a symbol (function or class) by name.
    pub fn describe_symbol(
        &self,
        name: &str,
    ) -> Option<DescribeSymbolResult> {
        let name_lower = name.to_lowercase();

        let signature = self
            .signatures
            .iter()
            .find(|s| s.name.to_lowercase() == name_lower)
            .cloned();

        let class_info = self
            .classes
            .iter()
            .find(|c| c.name.to_lowercase() == name_lower)
            .cloned();

        if signature.is_none() && class_info.is_none() {
            return None;
        }

        let kind = if class_info.is_some() {
            SymbolKind::Class
        } else if let Some(ref sig) = signature {
            if sig.scope.as_ref().is_some_and(|s| {
                self.classes.iter().any(|c| s.contains(&c.name))
            }) {
                SymbolKind::Method
            } else {
                SymbolKind::Function
            }
        } else {
            SymbolKind::Function
        };

        let callers: Vec<CallSite> = self
            .call_sites
            .iter()
            .filter(|cs| {
                cs.callee.to_lowercase() == name_lower
                    || cs
                        .callee
                        .to_lowercase()
                        .ends_with(&format!(".{name_lower}"))
            })
            .cloned()
            .collect();

        let callees: Vec<CallSite> = self
            .call_sites
            .iter()
            .filter(|cs| {
                cs.caller_scope.as_ref().is_some_and(|s| {
                    s.to_lowercase() == name_lower
                        || s.to_lowercase()
                            .ends_with(&format!(".{name_lower}"))
                })
            })
            .cloned()
            .collect();

        let subtokens = split_identifier(name);
        let concepts: Vec<String> = self
            .concepts
            .values()
            .filter(|c| subtokens.contains(&c.canonical))
            .map(|c| c.canonical.clone())
            .collect();

        // Entity enrichment — prefer exact case match, then case-insensitive,
        // and prefer matching SymbolKind to avoid class/function confusion.
        let entity = self
            .entities
            .values()
            .find(|e| e.name == name)
            .or_else(|| {
                let kind_match = |e: &&Entity| {
                    matches!(
                        (&kind, &e.kind),
                        (SymbolKind::Class, crate::types::EntityKind::Class)
                            | (SymbolKind::Function, crate::types::EntityKind::Function)
                            | (SymbolKind::Method, crate::types::EntityKind::Method)
                    )
                };
                self.entities
                    .values()
                    .filter(|e| e.name.to_lowercase() == name_lower)
                    .find(|e| kind_match(e))
                    .or_else(|| {
                        self.entities
                            .values()
                            .find(|e| e.name.to_lowercase() == name_lower)
                    })
            });

        let semantic_role = entity
            .map(|e| e.semantic_role.clone())
            .unwrap_or_default();

        let concept_tags: Vec<String> = entity
            .map(|e| {
                e.concept_tags
                    .iter()
                    .filter_map(|id| {
                        self.concepts.get(id).map(|c| c.canonical.clone())
                    })
                    .collect()
            })
            .unwrap_or_default();

        let related_entities: Vec<_> = entity
            .map(|e| {
                self.relationships
                    .iter()
                    .filter(|r| {
                        (r.kind == RelationshipKind::InheritsFrom
                            || r.kind == RelationshipKind::Uses)
                            && (r.source == e.id || r.target == e.id)
                    })
                    .filter_map(|r| {
                        let other_id = if r.source == e.id {
                            r.target
                        } else {
                            r.source
                        };
                        self.entities.get(&other_id).map(|e| e.summary())
                    })
                    .collect()
            })
            .unwrap_or_default();

        Some(DescribeSymbolResult {
            name: name.to_string(),
            kind,
            signature,
            class_info,
            callers,
            callees,
            concepts,
            semantic_role,
            concept_tags,
            related_entities,
        })
    }

    /// Detect abbreviation relationships between concepts and persist them
    /// as AbbreviationOf edges. For each pair of concepts where one canonical
    /// is shorter, calls find_abbreviation(short, &[long]). Must run BEFORE
    /// add_contrastive_edges (which checks for AbbreviationOf to suppress
    /// false contrastives).
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

                // Skip very short abbreviations — too noisy
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
    /// Must run AFTER add_similarity_edges. Suppresses conflicting
    /// SimilarTo edges.
    pub fn add_contrastive_edges(&mut self) {
        // Clear existing contrastive edges so this is idempotent.
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

        // Signal 1: param co-occurrence — concepts appearing as separate
        // params in the same signature
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

        // Signal 3: known patterns
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

        // Add contrastive edges for pairs scoring >= 3
        let mut contrastive_pairs: Vec<(u64, u64)> = Vec::new();
        for (&pair, &score) in &pair_scores {
            if score >= 3 {
                // Check no AbbreviationOf edge between them
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

        // Suppress SimilarTo edges and add Contrastive edges
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
    /// Must run AFTER embeddings and co-occurrence are built.
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

            // Build affinity matrix using scope co-occurrence +
            // embedding similarity
            let n = unique_ids.len();
            let mut affinity = vec![vec![0.0f32; n]; n];

            // Batch-embed all unique identifiers up front
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

                    // Signal 1: scope co-occurrence (weight 0.3)
                    let scopes_i: HashSet<&str> = concept
                        .occurrences
                        .iter()
                        .filter(|o| o.identifier == unique_ids[i])
                        .filter_map(|o| {
                            o.file.to_str()
                        })
                        .collect();
                    let scopes_j: HashSet<&str> = concept
                        .occurrences
                        .iter()
                        .filter(|o| o.identifier == unique_ids[j])
                        .filter_map(|o| {
                            o.file.to_str()
                        })
                        .collect();
                    let shared = scopes_i.intersection(&scopes_j).count();
                    if shared > 0 {
                        aff += 0.3
                            * (shared.min(3) as f32 / 3.0);
                    }

                    // Signal 2: embedding similarity (weight 0.3)
                    if let (Some(ei), Some(ej)) =
                        (&id_embeddings[i], &id_embeddings[j])
                    {
                        let sim = cosine_similarity(ei, ej);
                        aff += 0.3 * sim.max(0.0);
                    }

                    // Signal 3: structural context (weight 0.25)
                    let sig_i = self.signatures.iter().find(|s| {
                        s.name == unique_ids[i]
                    });
                    let sig_j = self.signatures.iter().find(|s| {
                        s.name == unique_ids[j]
                    });
                    if let (Some(si), Some(sj)) = (sig_i, sig_j) {
                        let params_i: HashSet<String> = si
                            .params
                            .iter()
                            .flat_map(|p| split_identifier(&p.name))
                            .collect();
                        let params_j: HashSet<String> = sj
                            .params
                            .iter()
                            .flat_map(|p| split_identifier(&p.name))
                            .collect();
                        let union_len =
                            params_i.union(&params_j).count();
                        if union_len > 0 {
                            let intersection_len = params_i
                                .intersection(&params_j)
                                .count();
                            let jaccard = intersection_len as f32
                                / union_len as f32;
                            aff += 0.25 * jaccard;
                        }
                    }

                    affinity[i][j] = aff;
                    affinity[j][i] = aff;
                }
            }

            // Connected components with threshold
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

            // Group into clusters
            let mut clusters: HashMap<usize, Vec<usize>> =
                HashMap::new();
            for i in 0..n {
                let root = find(&mut parent, i);
                clusters.entry(root).or_default().push(i);
            }

            // Filter clusters with >= 2 members
            let valid_clusters: Vec<Vec<usize>> = clusters
                .into_values()
                .filter(|c| c.len() >= 2)
                .collect();

            if valid_clusters.len() < 2 {
                continue;
            }

            // Determine qualifiers and build subconcepts
            let mut subconcepts = Vec::new();
            let all_cluster_subtokens: Vec<HashSet<String>> =
                valid_clusters
                    .iter()
                    .map(|cluster| {
                        cluster
                            .iter()
                            .flat_map(|&idx| {
                                split_identifier(&unique_ids[idx])
                            })
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
                        my_subtokens
                            .iter()
                            .find(|st| **st != concept.canonical)
                            .cloned()
                            .unwrap_or_else(|| {
                                format!("group_{ci}")
                            })
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
                    canonical: format!(
                        "{}.{}",
                        concept.canonical, qualifier
                    ),
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

    /// Locate the best entry points for working with a concept.
    pub fn locate_concept(
        &self,
        term: &str,
    ) -> Option<LocateConceptResult> {
        // Handle dotted subconcept query: "transform.spatial"
        let (base_term, subconcept_filter) =
            if let Some((base, sub)) = term.split_once('.') {
                (base, Some(sub))
            } else {
                (term, None)
            };

        let term_lower = base_term.to_lowercase();
        let concept = self
            .concepts
            .values()
            .find(|c| c.canonical == term_lower)?;

        // If subconcept filter, only use identifiers from that cluster
        let filter_ids: Option<HashSet<&str>> =
            subconcept_filter.and_then(|sub| {
                concept.subconcepts.iter().find_map(|sc| {
                    if sc.qualifier == sub
                        || sc.canonical == term
                    {
                        Some(
                            sc.identifiers
                                .iter()
                                .map(|s| s.as_str())
                                .collect(),
                        )
                    } else {
                        None
                    }
                })
            });

        // Rank signatures
        let mut scored_sigs: Vec<(&Signature, i32)> = self
            .signatures
            .iter()
            .filter_map(|sig| {
                let mut score = 0i32;
                let name_lower = sig.name.to_lowercase();

                // Check filter
                if let Some(ref ids) = filter_ids {
                    if !ids.contains(sig.name.as_str()) {
                        return None;
                    }
                }

                if concept
                    .occurrences
                    .iter()
                    .any(|o| o.identifier == sig.name)
                {
                    score += 3;
                } else if concept
                    .subtokens
                    .iter()
                    .any(|st| name_lower.contains(st.as_str()))
                {
                    score += 2;
                }

                // Param match
                for param in &sig.params {
                    let param_subtokens =
                        split_identifier(&param.name);
                    if param_subtokens
                        .iter()
                        .any(|st| st == &concept.canonical)
                    {
                        score += 1;
                    }
                }

                if score > 0 {
                    Some((sig, score))
                } else {
                    None
                }
            })
            .collect();
        scored_sigs.sort_by(|a, b| b.1.cmp(&a.1));
        let exemplar_signatures: Vec<Signature> = scored_sigs
            .into_iter()
            .take(5)
            .map(|(s, _)| s.clone())
            .collect();

        // Rank classes
        let mut scored_cls: Vec<(&ClassInfo, i32)> = self
            .classes
            .iter()
            .filter_map(|cls| {
                let mut score = 0i32;
                if concept
                    .occurrences
                    .iter()
                    .any(|o| o.identifier == cls.name)
                {
                    score += 3;
                }
                for method in &cls.methods {
                    if concept
                        .occurrences
                        .iter()
                        .any(|o| o.identifier == *method)
                    {
                        score += 1;
                    }
                }
                if score > 0 {
                    Some((cls, score))
                } else {
                    None
                }
            })
            .collect();
        scored_cls.sort_by(|a, b| b.1.cmp(&a.1));
        let exemplar_classes: Vec<ClassInfo> = scored_cls
            .into_iter()
            .take(3)
            .map(|(c, _)| c.clone())
            .collect();

        // Rank files by concept density
        let mut file_counts: HashMap<&std::path::Path, usize> =
            HashMap::new();
        for occ in &concept.occurrences {
            if let Some(ref ids) = filter_ids {
                if !ids.contains(occ.identifier.as_str()) {
                    continue;
                }
            }
            *file_counts.entry(&occ.file).or_insert(0) += 1;
        }
        let mut files: Vec<(std::path::PathBuf, usize)> = file_counts
            .into_iter()
            .map(|(p, c)| (p.to_path_buf(), c))
            .collect();
        files.sort_by(|a, b| b.1.cmp(&a.1));

        // Contrastive concepts
        let contrastive_concepts: Vec<String> = self
            .relationships
            .iter()
            .filter(|r| {
                r.kind == RelationshipKind::Contrastive
                    && (r.source == concept.id
                        || r.target == concept.id)
            })
            .filter_map(|r| {
                let other_id = if r.source == concept.id {
                    r.target
                } else {
                    r.source
                };
                self.concepts
                    .get(&other_id)
                    .map(|c| c.canonical.clone())
            })
            .collect();

        // Key entities sorted by specificity (fewer concept tags = more specific)
        let mut key_entities_vec: Vec<_> = self
            .entities
            .values()
            .filter(|e| e.concept_tags.contains(&concept.id))
            .collect::<Vec<_>>();
        key_entities_vec.sort_by_key(|e| e.concept_tags.len());
        let key_entities: Vec<_> = key_entities_vec
            .into_iter()
            .take(5)
            .map(|e| e.summary())
            .collect();

        Some(LocateConceptResult {
            concept: concept.canonical.clone(),
            exemplar_signatures,
            exemplar_classes,
            files,
            contrastive_concepts,
            key_entities,
        })
    }

    /// Generate a session briefing from current graph state.
    pub fn session_briefing(&self) -> SessionBriefing {
        let conventions = self.conventions.clone();

        // Abbreviations from AbbreviationOf relationships
        let abbreviations: Vec<(String, String)> = self
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::AbbreviationOf)
            .filter_map(|r| {
                let short = self
                    .concepts
                    .get(&r.source)
                    .map(|c| c.canonical.clone())?;
                let long = self
                    .concepts
                    .get(&r.target)
                    .map(|c| c.canonical.clone())?;
                Some((short, long))
            })
            .collect();

        // Top 20 concepts
        let mut concepts_sorted: Vec<&Concept> =
            self.concepts.values().collect();
        concepts_sorted
            .sort_by(|a, b| b.occurrences.len().cmp(&a.occurrences.len()));
        let top_concepts: Vec<(String, usize)> = concepts_sorted
            .iter()
            .take(20)
            .map(|c| (c.canonical.clone(), c.occurrences.len()))
            .collect();

        // Contrastive pairs
        let contrastive_pairs: Vec<(String, String)> = self
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::Contrastive)
            .filter_map(|r| {
                let a = self
                    .concepts
                    .get(&r.source)
                    .map(|c| c.canonical.clone())?;
                let b = self
                    .concepts
                    .get(&r.target)
                    .map(|c| c.canonical.clone())?;
                Some((a, b))
            })
            .collect();

        // Vocabulary warnings
        let vocabulary_warnings: Vec<String> = self
            .concepts
            .values()
            .filter_map(|c| {
                let check = self.check_naming(&c.canonical);
                if check.verdict == Verdict::Inconsistent {
                    Some(format!(
                        "use '{}', not '{}'",
                        check
                            .suggestion
                            .as_deref()
                            .unwrap_or("?"),
                        c.canonical,
                    ))
                } else {
                    None
                }
            })
            .collect();

        // Entity clusters grouped by semantic role (min 2 per cluster)
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
}

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

#[cfg(test)]
mod tests {
    use super::*;
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
            subconcepts: Vec::new(),
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
        };
        ConceptGraph::build(analysis, EmbeddingIndex::empty()).unwrap()
    }

    #[test]
    fn test_query_concept_exact() {
        let graph = make_test_graph();
        let result =
            graph.query_concept("transform", &QueryConceptParams::default());
        assert!(result.is_some());
        let result = result.unwrap();
        assert!(result
            .variants
            .contains(&"spatial_transform".to_string()));
        assert!(
            result.variants.contains(&"apply_transform".to_string())
        );
    }

    #[test]
    fn test_check_naming_consistent() {
        let graph = make_test_graph();
        let result = graph.check_naming("nb_features");
        // nb_features follows the nb_ convention -> Consistent
        assert_eq!(result.verdict, Verdict::Consistent);
    }

    #[test]
    fn test_check_naming_inconsistent() {
        let graph = make_test_graph();
        let result = graph.check_naming("n_dims");
        assert_eq!(result.verdict, Verdict::Inconsistent);
        // Should suggest ndim or nb_dims
        assert!(result.suggestion.is_some());
    }

    #[test]
    fn test_list_concepts_sorted() {
        let graph = make_test_graph();
        let concepts = graph.list_concepts();
        assert!(!concepts.is_empty());
        // Should be sorted by occurrence count descending
        for i in 1..concepts.len() {
            assert!(
                concepts[i - 1].occurrences.len()
                    >= concepts[i].occurrences.len()
            );
        }
    }

    #[test]
    fn test_suggest_name_count() {
        let graph = make_test_graph();
        let suggestions = graph.suggest_name("count of features");
        assert!(!suggestions.is_empty());
        // Should suggest nb_features
        assert!(suggestions
            .iter()
            .any(|s| s.name.contains("nb_features")));
    }

    #[test]
    fn test_describe_symbol_not_found() {
        let graph = make_test_graph();
        assert!(graph.describe_symbol("nonexistent").is_none());
    }

    #[test]
    fn test_describe_symbol_function() {
        use crate::types::{Param, Signature};
        let mut analysis = AnalysisResult {
            concepts: vec![make_concept(
                1,
                "transform",
                &["spatial_transform"],
            )],
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: vec![Signature {
                name: "spatial_transform".to_string(),
                params: vec![Param {
                    name: "vol".to_string(),
                    type_annotation: Some("Tensor".to_string()),
                    default: None,
                }],
                return_type: Some("Tensor".to_string()),
                decorators: Vec::new(),
                docstring_first_line: Some(
                    "Apply spatial transform.".to_string(),
                ),
                file: PathBuf::from("utils.py"),
                line: 10,
                scope: None,
            }],
            classes: Vec::new(),
            call_sites: Vec::new(),
        };
        analysis.call_sites.push(crate::types::CallSite {
            caller_scope: Some("register".to_string()),
            callee: "spatial_transform".to_string(),
            file: PathBuf::from("reg.py"),
            line: 20,
        });
        let graph = ConceptGraph::build(analysis, EmbeddingIndex::empty())
            .unwrap();
        let result = graph
            .describe_symbol("spatial_transform")
            .expect("should find symbol");
        assert!(result.signature.is_some());
        assert_eq!(result.callers.len(), 1);
        assert!(result.concepts.contains(&"transform".to_string()));
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
            conventions: Vec::new(),
            co_occurrence_matrix: Vec::new(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
        };
        let mut graph =
            ConceptGraph::build(analysis, EmbeddingIndex::empty())
                .unwrap();

        assert!(graph
            .relationships
            .iter()
            .all(|r| r.kind != RelationshipKind::AbbreviationOf));

        graph.add_abbreviation_edges();

        let abbrevs: Vec<_> = graph
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::AbbreviationOf)
            .collect();
        assert_eq!(abbrevs.len(), 2);

        // trf → transform
        assert!(abbrevs.iter().any(|r| r.source == 1 && r.target == 2));
        // seg → segmentation
        assert!(abbrevs.iter().any(|r| r.source == 3 && r.target == 4));
    }
}

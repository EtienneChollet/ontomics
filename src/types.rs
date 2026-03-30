use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EntityType {
    Function,
    Class,
    Parameter,
    Variable,
    Attribute,
    Decorator,
    TypeAnnotation,
    DocText,
    Interface,
    TypeAlias,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Occurrence {
    pub file: PathBuf,
    pub line: usize,
    pub identifier: String,
    pub entity_type: EntityType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Concept {
    pub id: u64,
    pub canonical: String,
    pub subtokens: Vec<String>,
    pub occurrences: Vec<Occurrence>,
    pub entity_types: HashSet<EntityType>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub embedding: Option<Vec<f32>>,
    #[serde(default)]
    pub subconcepts: Vec<Subconcept>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Subconcept {
    pub qualifier: String,
    pub canonical: String,
    pub occurrences: Vec<Occurrence>,
    pub identifiers: Vec<String>,
    pub embedding: Option<Vec<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Relationship {
    pub source: u64,
    pub target: u64,
    pub kind: RelationshipKind,
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum RelationshipKind {
    CoOccurs,
    SimilarTo,
    AbbreviationOf,
    SharedPattern,
    Contrastive,
    Instantiates,
    InheritsFrom,
    Uses,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Convention {
    pub pattern: PatternKind,
    pub entity_type: EntityType,
    pub semantic_role: String,
    pub examples: Vec<String>,
    pub frequency: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PatternKind {
    Prefix(String),
    Suffix(String),
    Compound(String),
    Conversion(String),
}

// --- Parser output types ---

#[derive(Debug, Clone)]
pub struct RawIdentifier {
    pub name: String,
    pub entity_type: EntityType,
    pub file: PathBuf,
    pub line: usize,
    /// Enclosing scope, e.g. "MyClass", "MyClass.method", "function_name"
    pub scope: Option<String>,
}

#[derive(Debug, Clone)]
pub struct ParseResult {
    pub identifiers: Vec<RawIdentifier>,
    #[allow(dead_code)]
    pub doc_texts: Vec<(PathBuf, usize, String)>,
    pub signatures: Vec<Signature>,
    pub classes: Vec<ClassInfo>,
    pub call_sites: Vec<CallSite>,
}

// --- Analysis output types ---

#[derive(Debug, Clone)]
pub struct AnalysisResult {
    pub concepts: Vec<Concept>,
    pub conventions: Vec<Convention>,
    pub co_occurrence_matrix: Vec<((u64, u64), f32)>,
    pub signatures: Vec<Signature>,
    pub classes: Vec<ClassInfo>,
    pub call_sites: Vec<CallSite>,
}

// --- Query result types ---

/// Lightweight summary of a related concept (avoids serializing full
/// occurrence lists for every neighbor).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RelatedConcept {
    pub canonical: String,
    pub kind: RelationshipKind,
    pub weight: f32,
    pub occurrences: usize,
}

/// Optional limits for `query_concept` responses.
pub struct QueryConceptParams {
    pub max_related: usize,
    pub max_occurrences: usize,
    pub max_variants: usize,
    pub max_signatures: usize,
    pub max_entities: usize,
}

impl Default for QueryConceptParams {
    fn default() -> Self {
        Self {
            max_related: 10,
            max_occurrences: 5,
            max_variants: 20,
            max_signatures: 5,
            max_entities: 5,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptQueryResult {
    pub concept: Concept,
    pub variants: Vec<String>,
    pub related: Vec<RelatedConcept>,
    pub conventions: Vec<Convention>,
    pub top_occurrences: Vec<Occurrence>,
    pub signatures: Vec<Signature>,
    pub classes: Vec<ClassInfo>,
    pub call_graph: Vec<(String, String)>,
    #[serde(default)]
    pub subconcepts: Vec<Subconcept>,
    #[serde(default)]
    pub entities: Vec<EntitySummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamingCheckResult {
    pub input: String,
    pub subtokens: Vec<String>,
    pub verdict: Verdict,
    pub reason: String,
    pub suggestion: Option<String>,
    pub matching_convention: Option<Convention>,
    pub similar_identifiers: Vec<(String, usize)>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Verdict {
    Consistent,
    Inconsistent,
    Unknown,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NameSuggestion {
    pub name: String,
    pub confidence: f32,
    pub based_on: Vec<String>,
}

// --- Locate and briefing result types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocateConceptResult {
    pub concept: String,
    pub exemplar_signatures: Vec<Signature>,
    pub exemplar_classes: Vec<ClassInfo>,
    pub files: Vec<(PathBuf, usize)>,
    pub contrastive_concepts: Vec<String>,
    #[serde(default)]
    pub key_entities: Vec<EntitySummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionBriefing {
    pub conventions: Vec<Convention>,
    pub abbreviations: Vec<(String, String)>,
    pub top_concepts: Vec<(String, usize)>,
    pub contrastive_pairs: Vec<(String, String)>,
    pub vocabulary_warnings: Vec<String>,
    #[serde(default)]
    pub entity_clusters: Vec<EntityCluster>,
}

// --- L2: Structural types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Signature {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<String>,
    pub decorators: Vec<String>,
    pub docstring_first_line: Option<String>,
    pub file: PathBuf,
    pub line: usize,
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Param {
    pub name: String,
    pub type_annotation: Option<String>,
    pub default: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClassInfo {
    pub name: String,
    pub bases: Vec<String>,
    pub methods: Vec<String>,
    pub attributes: Vec<String>,
    pub docstring_first_line: Option<String>,
    pub file: PathBuf,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallSite {
    pub caller_scope: Option<String>,
    pub callee: String,
    pub file: PathBuf,
    pub line: usize,
}

// --- L3: Entity types ---

/// A Python object promoted to a first-class node in the concept graph.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Entity {
    pub id: u64,
    pub name: String,
    pub kind: EntityKind,
    pub concept_tags: Vec<u64>,
    pub semantic_role: String,
    pub file: PathBuf,
    pub line: usize,
    pub signature_idx: Option<usize>,
    pub class_info_idx: Option<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntityKind {
    Function,
    Class,
    Method,
}

/// Lightweight entity summary for inclusion in query results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySummary {
    pub name: String,
    pub kind: EntityKind,
    pub semantic_role: String,
    pub file: PathBuf,
    pub line: usize,
}

/// Entity cluster for session briefing — groups entities by semantic role.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntityCluster {
    pub role: String,
    pub count: usize,
    pub examples: Vec<String>,
}

impl Entity {
    /// Compute a stable entity ID from its identity triple.
    /// Uses "entity:{name}:{file}:{line}" to avoid collision with concept IDs.
    pub fn hash_id(name: &str, file: &std::path::Path, line: usize) -> u64 {
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        format!("entity:{}:{}:{}", name, file.display(), line).hash(&mut hasher);
        hasher.finish()
    }

    pub fn summary(&self) -> EntitySummary {
        EntitySummary {
            name: self.name.clone(),
            kind: self.kind.clone(),
            semantic_role: self.semantic_role.clone(),
            file: self.file.clone(),
            line: self.line,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeSymbolResult {
    pub name: String,
    pub kind: SymbolKind,
    pub signature: Option<Signature>,
    pub class_info: Option<ClassInfo>,
    pub callers: Vec<CallSite>,
    pub callees: Vec<CallSite>,
    pub concepts: Vec<String>,
    #[serde(default)]
    pub semantic_role: String,
    #[serde(default)]
    pub concept_tags: Vec<String>,
    #[serde(default)]
    pub related_entities: Vec<EntitySummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolKind {
    Function,
    Class,
    Method,
}

// --- Ontology diff types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyDiff {
    pub base_ref: String,
    pub head_ref: String,
    pub added_concepts: Vec<Concept>,
    pub removed_concepts: Vec<Concept>,
    pub changed_concepts: Vec<ConceptDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptDelta {
    pub concept: Concept,
    pub frequency_change: i64,
    pub new_variants: Vec<String>,
    pub removed_variants: Vec<String>,
}

// --- Domain pack types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainPack {
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub domain: Option<String>,
    #[serde(default)]
    pub abbreviations: Vec<AbbreviationMapping>,
    #[serde(default)]
    pub conventions: Vec<ConventionEntry>,
    #[serde(default)]
    pub domain_terms: Vec<DomainTerm>,
    #[serde(default)]
    pub concept_associations: Vec<ConceptAssociation>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AbbreviationMapping {
    pub short: String,
    pub long: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConventionEntry {
    pub pattern: String,
    pub value: String,
    pub role: String,
    #[serde(default)]
    pub entity_types: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DomainTerm {
    pub term: String,
    #[serde(default)]
    pub entity_types: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptAssociation {
    pub concepts: Vec<String>,
    pub kind: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_entity_type_serialization() {
        let et = EntityType::Function;
        let json = serde_json::to_string(&et).unwrap();
        let deserialized: EntityType = serde_json::from_str(&json).unwrap();
        assert_eq!(et, deserialized);
    }

    #[test]
    fn test_verdict_variants() {
        assert_eq!(Verdict::Consistent, Verdict::Consistent);
        assert_ne!(Verdict::Consistent, Verdict::Inconsistent);
        assert_ne!(Verdict::Inconsistent, Verdict::Unknown);
    }
}

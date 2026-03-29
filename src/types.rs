use serde::{Deserialize, Serialize};
use std::collections::HashSet;
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawIdentifier {
    pub name: String,
    pub entity_type: EntityType,
    pub file: PathBuf,
    pub line: usize,
    /// Enclosing scope, e.g. "MyClass", "MyClass.method", "function_name"
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParseResult {
    pub identifiers: Vec<RawIdentifier>,
    pub doc_texts: Vec<(PathBuf, usize, String)>,
    pub signatures: Vec<Signature>,
    pub classes: Vec<ClassInfo>,
    pub call_sites: Vec<CallSite>,
}

// --- Analysis output types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisResult {
    pub concepts: Vec<Concept>,
    pub conventions: Vec<Convention>,
    pub co_occurrence_matrix: Vec<((u64, u64), f32)>,
    #[serde(default)]
    pub signatures: Vec<Signature>,
    #[serde(default)]
    pub classes: Vec<ClassInfo>,
    #[serde(default)]
    pub call_sites: Vec<CallSite>,
}

// --- Query result types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptQueryResult {
    pub concept: Concept,
    pub variants: Vec<String>,
    pub related: Vec<(Concept, RelationshipKind, f32)>,
    pub conventions: Vec<Convention>,
    pub top_occurrences: Vec<Occurrence>,
    pub signatures: Vec<Signature>,
    pub classes: Vec<ClassInfo>,
    pub call_graph: Vec<(String, String)>,
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

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeSymbolResult {
    pub name: String,
    pub kind: SymbolKind,
    pub signature: Option<Signature>,
    pub class_info: Option<ClassInfo>,
    pub callers: Vec<CallSite>,
    pub callees: Vec<CallSite>,
    pub concepts: Vec<String>,
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

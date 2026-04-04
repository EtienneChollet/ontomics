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
    pub identifier: String,
    pub entity_type: EntityType,
    pub file: PathBuf,
    pub line: usize,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cluster_id: Option<usize>,
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

// --- Nesting tree types ---

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NestingKind {
    Module,
    Class,
    Function,
    Method,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NestingNode {
    pub name: String,
    pub kind: NestingKind,
    pub line: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub children: Vec<NestingNode>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileNestingTree {
    pub file: PathBuf,
    pub root: NestingNode,
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
    #[allow(dead_code)]
    pub doc_texts: Vec<(PathBuf, usize, String)>,
    pub signatures: Vec<Signature>,
    pub classes: Vec<ClassInfo>,
    pub call_sites: Vec<CallSite>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nesting_trees: Vec<FileNestingTree>,
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
    pub nesting_trees: Vec<FileNestingTree>,
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
    #[serde(default)]
    pub confidence: f32,
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
pub struct VocabularyHealth {
    pub convention_coverage: f32,
    pub consistency_ratio: f32,
    pub cluster_cohesion: f32,
    pub overall: f32,
    pub top_inconsistencies: Vec<InconsistencyPair>,
    pub uncovered_identifiers: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InconsistencyPair {
    pub dominant: String,
    pub minority: String,
    pub dominant_count: usize,
    pub minority_count: usize,
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
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<FunctionBody>,
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

/// Lightweight entity reference for drill-down query results.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntitySummary {
    pub name: String,
    pub kind: EntityKind,
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
            file: self.file.clone(),
            line: self.line,
        }
    }
}

// --- Concept trace types ---

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum TraceRole {
    Producer,
    Consumer,
    Both,
    Bridge,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodSummary {
    pub name: String,
    pub params: Vec<String>,
    pub return_type: Option<String>,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceNode {
    pub entity_name: String,
    pub kind: EntityKind,
    pub file: PathBuf,
    pub line: usize,
    pub role: TraceRole,
    pub concept_tags: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bases: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub methods: Vec<MethodSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceEdge {
    pub caller: String,
    pub callee: String,
    pub file: PathBuf,
    pub line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptTrace {
    pub concept: String,
    pub producers: Vec<TraceNode>,
    pub consumers: Vec<TraceNode>,
    pub call_chain: Vec<TraceNode>,
    pub edges: Vec<TraceEdge>,
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
    pub related_entities: Vec<EntitySummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymbolKind {
    Function,
    Class,
    Method,
}

// --- File description types ---

/// A symbol within a file overview — compact, concept-annotated.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileSymbol {
    pub name: String,
    pub kind: SymbolKind,
    pub line: usize,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub concepts: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub params: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub return_type: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub bases: Vec<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub methods: Vec<FileSymbol>,
}

/// Result of `describe_file` — one entry per matching file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DescribeFileResult {
    pub file: PathBuf,
    pub symbols: Vec<FileSymbol>,
}

// --- Ontology diff types ---

/// Lightweight concept summary for diffs (no occurrences or embeddings).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffConceptSummary {
    pub canonical: String,
    pub frequency: usize,
}

impl DiffConceptSummary {
    pub fn from_concept(c: &Concept) -> Self {
        Self {
            canonical: c.canonical.clone(),
            frequency: c.occurrences.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OntologyDiff {
    pub base_ref: String,
    pub head_ref: String,
    pub added_concepts: Vec<DiffConceptSummary>,
    pub removed_concepts: Vec<DiffConceptSummary>,
    pub changed_concepts: Vec<ConceptDelta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptDelta {
    pub concept: DiffConceptSummary,
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

// --- L4: Logic types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionBody {
    pub entity_name: String,
    pub scope: Option<String>,
    pub body_text: String,
    pub file: PathBuf,
    pub start_line: usize,
    pub end_line: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pseudocode {
    pub entity_id: u64,
    pub steps: Vec<PseudocodeStep>,
    pub body_hash: u64,
    #[serde(default)]
    pub omitted_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PseudocodeStep {
    Call { callee: String, args: Vec<String> },
    Conditional { branches: Vec<ConditionalBranch> },
    Loop { kind: LoopKind, body: Vec<PseudocodeStep> },
    Return { value: Option<String> },
    Assignment { target: String, source: String },
    Yield { value: Option<String> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConditionalBranch {
    pub condition: Option<String>,
    pub body: Vec<PseudocodeStep>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoopKind {
    For { variable: String, iterable: String },
    While { condition: String },
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct LogicCluster {
    pub id: usize,
    pub entity_ids: Vec<u64>,
    #[serde(default)]
    pub centroid: Vec<f32>,
    pub behavioral_label: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CentralityScore {
    pub entity_id: u64,
    pub in_degree: usize,
    pub out_degree: usize,
    pub pagerank: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicDescription {
    pub entity: EntitySummary,
    pub pseudocode_text: String,
    pub logic_cluster: Option<LogicClusterSummary>,
    pub centrality: CentralityScore,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogicClusterSummary {
    pub id: usize,
    pub size: usize,
    pub behavioral_label: Option<String>,
    pub members: Vec<EntitySummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimilarLogicResult {
    pub query_entity: EntitySummary,
    pub similar: Vec<(EntitySummary, f32)>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactContext {
    pub scope: String,
    pub text: String,
    pub token_estimate: usize,
}

// --- Concept map types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleMapEntry {
    pub path: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub dominant_concepts: Vec<String>,
    pub nb_classes: usize,
    pub nb_functions: usize,
    pub nb_entities: usize,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub concept_density: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConceptMap {
    pub modules: Vec<ModuleMapEntry>,
    pub total_entities: usize,
    pub total_concepts: usize,
}

// --- Type flow types ---

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeFlow {
    pub from_entity: String,
    pub to_entity: String,
    pub type_name: String,
    pub from_file: PathBuf,
    pub to_file: PathBuf,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeFlowResult {
    pub flows: Vec<TypeFlow>,
    pub dominant_types: Vec<TypeFrequency>,
    pub total_typed_edges: usize,
    pub total_untyped_edges: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TypeFrequency {
    pub type_name: String,
    pub count: usize,
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

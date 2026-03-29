# `semex` v2 project specs — Structural Layer

# Goals and non-goals

## Goals

1. **Two-layer concept understanding for LLMs**: Provide vocabulary (L1) and structure (L2) so LLM coding tools can reduce agentic search overhead by querying pre-computed domain knowledge instead of grepping and reading files. The calling LLM synthesizes its own semantic understanding from this rich structured data.
2. **L1 — Vocabulary index** (existing): Extract domain ontology from Python codebases. Concept clusters, naming conventions, abbreviation mappings, occurrence locations. Pure static analysis, always cheap.
3. **L2 — Structural index** (new, always-on): Extract function signatures, class hierarchies, and call-graph edges from the same tree-sitter pass. Group structural information by domain concept so an LLM can understand interfaces without reading files.
4. **Contrastive pair detection**: Identify concepts that co-occur but are semantically opposed (e.g., `source`/`target`, `input`/`output`). Embedding similarity incorrectly clusters these; explicit contrastive detection prevents this (IdBench 2019 finding).
5. **Subconcept disambiguation**: When a high-frequency concept (e.g., `transform`) has distinct usage clusters that never co-occur, split it into subconcepts (e.g., `transform.spatial`, `transform.fourier`, `transform.augmentation`). Prevents polysemy from collapsing distinct domain meanings into one blob.
6. **`locate_concept` tool**: Given a concept name, return the minimal set of exemplar files and signatures an LLM needs to read — optimized for reducing context consumption rather than exhaustive listing.
7. **Session briefing**: Expose an MCP resource (`semex://briefing`) that proactively provides a compact convention + vocabulary summary at session start, without the LLM needing to know to call a tool. Research shows proactive injection outperforms reactive querying.
8. **Unified query interface**: Enrich existing MCP tools with L2 data transparently. An LLM that calls `query_concept` gets vocabulary + structure in a single response.
9. **Backward compatibility**: All v1 functionality continues to work. L2 adds zero configuration.
10. **No remote dependencies**: Everything runs locally. No API keys, no network calls. The calling LLM (e.g. Claude Code) is already capable of synthesizing understanding from the structured data semex provides — there is no need for semex to call an LLM itself.

## Non-Goals

1. **No auto-refactoring**: semex identifies patterns but never modifies source code.
2. **No cross-repo analysis**: Single repository at a time.
3. **No code structure indexing as a primary goal**: L2 extracts structure *organized by domain concept*, not a generic code index. This is not a replacement for LSP or code search — it's structural data in service of semantic understanding.
4. **No language server protocol**: semex is not an LSP. No completions, diagnostics, or hover.
5. **No LLM integration**: semex does not call any LLM API. It provides data; the calling LLM interprets it.

# Architecture and design

## Coding style

- Idiomatic Rust: `Result<T, E>` everywhere, `thiserror` for error types, no `unwrap()` outside tests
- Modules map 1:1 to files (no `mod.rs` nesting beyond one level)
- Builder pattern when constructors take >3 parameters
- `Serialize`/`Deserialize` on all types that cross the MCP boundary

## Environment management

- Rust toolchain via `rustup` (stable channel)
- Build: `cargo build --release`
- Dependencies managed via `Cargo.toml`
- Tree-sitter grammars fetched at build time

## Dependencies

All existing from v1 — no new dependencies required.

* `tree-sitter` + `tree-sitter-python`: Python parsing and identifier/structure extraction.
* `fastembed`: Local ONNX embeddings (`BAAI/bge-small-en-v1.5`). No API key, ~30MB model, CPU.
* `rmcp`: Rust MCP SDK. JSON-RPC transport (stdio), tool registration, request/response lifecycle.
* `rusqlite`: SQLite (used for DB schema creation; cache itself is a JSON file due to SQLite blob reliability issues with large graphs).
* `git2`: libgit2 bindings for `ontology_diff`.
* `notify`: Cross-platform filesystem watcher for incremental re-indexing.
* `serde` + `serde_json`: Serialization for MCP protocol, cache, and tool outputs.
* `anyhow`: Error handling in application code.
* `clap`: CLI argument parsing.
* `rayon`: Parallel file parsing.
* `tokio`: Async runtime for MCP event loop.
* `ignore`: File walking that respects `.gitignore`.
* `toml`: Configuration parsing.
* `tracing` + `tracing-subscriber`: Structured logging.

## Data source / domain context

### Input

Any Python codebase. semex reads `.py` files, extracts identifiers, signatures, classes, and call sites from the AST.

### Test corpora

- `/home/eti/projects/voxelmorph/` — medical image registration. Rich domain vocabulary, strong conventions.
- `/home/eti/projects/neurite/` — neural network utilities for medical imaging. Overlapping domain.

### What gets extracted from each file

#### L1 (existing — vocabulary)

| AST node type | Extracted identifiers | Entity type |
|---|---|---|
| `function_definition` | function name, parameter names | `Function`, `Parameter` |
| `class_definition` | class name | `Class` |
| `assignment` | target variable name | `Variable` |
| `attribute` | attribute name | `Attribute` |
| `decorated_definition` | decorator name | `Decorator` |
| `type` annotations | type names | `TypeAnnotation` |
| `comment` / `string` (docstrings) | free text | `DocText` |

#### L2 (new — structure)

| AST node type | Extracted structure | Output type |
|---|---|---|
| `function_definition` | name, params with type annotations, return type annotation, decorators, docstring first line | `Signature` |
| `class_definition` | name, base classes, method names, class-level attributes, docstring first line | `ClassInfo` |
| `call` expressions | caller scope, callee name, file, line | `CallSite` |

### Key definitions

- A **concept** is a cluster of related subtokens that recur across identifiers. Example: `transform` unifies `spatial_transform`, `apply_transform`, `TransformLayer`, `trf`.
- A **subtoken** is a single word extracted from an identifier by splitting on `_` or camelCase boundaries.
- A **convention** is a detected naming pattern with a semantic role. Example: prefix `nb_` means "count of".
- An **occurrence** is a specific location where a concept appears: file, line, identifier, entity type.
- The **concept graph** is the set of all concepts with weighted relationships (co-occurrence, embedding similarity, shared patterns).
- A **signature** is the structural interface of a function: name, parameters (with optional type annotations), return type, decorators, and docstring first line.
- A **call site** is a location where one function invokes another: caller scope, callee name, file, line.
- A **contrastive pair** is two concepts that consistently co-occur in the same scopes (e.g., same function signature) but fill complementary roles. Examples: `source`/`target`, `input`/`output`, `pred`/`true`, `before`/`after`. These should NOT be clustered as similar despite high co-occurrence.
- A **subconcept** is a distinct usage cluster within a polysemous concept. When `transform` has identifiers that cluster into two non-overlapping groups — `spatial_transform`, `apply_transform`, `trf` (cluster A) vs. `fourier_transform`, `fft`, `dft` (cluster B) — the concept is split into `transform.spatial` and `transform.fourier`. The parent concept remains as a navigation hub; the subconcepts carry the actual occurrences and signatures.

## Component breakdown

### Project structure

```
semex/
|--- Cargo.toml
|--- SPEC.md            # v1 spec (historical)
|--- SPEC2.md           # v2 spec (this document)
|--- src/
| |--- main.rs          # Entry point: CLI subcommands + MCP server
| |--- config.rs        # .semex/config.toml loading and defaults
| |--- parser.rs        # Tree-sitter extraction: identifiers (L1) + signatures/classes/calls (L2)
| |--- tokenizer.rs     # Identifier splitting into subtokens
| |--- analyzer.rs      # TF-IDF, convention detection, co-occurrence
| |--- embeddings.rs    # Local embedding computation + cosine similarity
| |--- graph.rs         # Concept graph: nodes, edges, queries (L1+L2)
| |--- diff.rs          # git2-based ontology diff between refs
| |--- cache.rs         # JSON file persistence + file watcher
| |--- tools.rs         # MCP tool definitions + handlers
| |--- types.rs         # Shared types: Concept, Occurrence, Convention, Signature, etc.
```

# Functional requirements

## API contracts

### `types.rs` — New types (additions to existing v1 types)

```rust
// --- L1 extension: Contrastive pairs ---

// Add to existing RelationshipKind enum:
pub enum RelationshipKind {
    CoOccurs,
    SimilarTo,
    AbbreviationOf,
    SharedPattern,
    Contrastive,  // NEW — concepts that co-occur but are semantically opposed
}

// --- L1 extension: Subconcept disambiguation ---

pub struct Subconcept {
    pub qualifier: String,           // e.g. "spatial", "fourier", "augmentation"
    pub canonical: String,           // e.g. "transform.spatial"
    pub occurrences: Vec<Occurrence>,
    pub identifiers: Vec<String>,    // the identifier names in this cluster
    pub embedding: Option<Vec<f32>>, // cluster-specific embedding
}

// Concept gains optional subconcepts field
// (added to existing Concept struct):
//   pub subconcepts: Vec<Subconcept>,  // empty if concept is not polysemous

// --- L2: Structural types ---

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

pub struct Param {
    pub name: String,
    pub type_annotation: Option<String>,
    pub default: Option<String>,
}

pub struct ClassInfo {
    pub name: String,
    pub bases: Vec<String>,
    pub methods: Vec<String>,
    pub attributes: Vec<String>,
    pub docstring_first_line: Option<String>,
    pub file: PathBuf,
    pub line: usize,
}

pub struct CallSite {
    pub caller_scope: Option<String>,
    pub callee: String,
    pub file: PathBuf,
    pub line: usize,
}
```

### `types.rs` — Extended existing types

```rust
// ParseResult gains L2 fields
pub struct ParseResult {
    pub identifiers: Vec<RawIdentifier>,        // L1
    pub doc_texts: Vec<(PathBuf, usize, String)>, // L1
    pub signatures: Vec<Signature>,              // L2 — NEW
    pub classes: Vec<ClassInfo>,                 // L2 — NEW
    pub call_sites: Vec<CallSite>,              // L2 — NEW
}

// RelatedConcept — lightweight summary (avoids serializing full occurrence lists)
pub struct RelatedConcept {
    pub canonical: String,
    pub kind: RelationshipKind,
    pub weight: f32,
    pub occurrences: usize,
}

// QueryConceptParams — optional limits for response size
pub struct QueryConceptParams {
    pub max_related: usize,      // default: 10
    pub max_occurrences: usize,  // default: 5
    pub max_variants: usize,     // default: 20
    pub max_signatures: usize,   // default: 5
}

// ConceptQueryResult gains L2 + subconcept fields
pub struct ConceptQueryResult {
    pub concept: Concept,
    pub variants: Vec<String>,                   // capped by max_variants
    pub related: Vec<RelatedConcept>,            // CHANGED: lightweight summaries, capped by max_related
    pub conventions: Vec<Convention>,
    pub top_occurrences: Vec<Occurrence>,         // capped by max_occurrences
    pub signatures: Vec<Signature>,              // L2 — NEW, capped by max_signatures
    pub classes: Vec<ClassInfo>,                 // L2 — NEW
    pub call_graph: Vec<(String, String)>,       // L2 — NEW: (caller, callee) pairs
    pub subconcepts: Vec<Subconcept>,            // NEW: non-empty if concept is polysemous
}

// DescribeSymbolResult — NEW
pub struct DescribeSymbolResult {
    pub name: String,
    pub kind: SymbolKind,
    pub signature: Option<Signature>,
    pub class_info: Option<ClassInfo>,
    pub callers: Vec<CallSite>,
    pub callees: Vec<CallSite>,
    pub concepts: Vec<String>,                   // concept canonicals this symbol belongs to
}

pub enum SymbolKind {
    Function,
    Class,
    Method,
}

// LocateConceptResult — NEW
pub struct LocateConceptResult {
    pub concept: String,                         // canonical concept name
    pub exemplar_signatures: Vec<Signature>,      // top signatures by relevance (max 5)
    pub exemplar_classes: Vec<ClassInfo>,         // top classes by relevance (max 3)
    pub files: Vec<(PathBuf, usize)>,            // (file, occurrence_count) ranked by density
    pub contrastive_concepts: Vec<String>,        // concepts that often appear alongside but are distinct
}

// SessionBriefing — NEW (for MCP resource)
pub struct SessionBriefing {
    pub conventions: Vec<Convention>,
    pub abbreviations: Vec<(String, String)>,     // (short, expanded) e.g. ("trf", "transform")
    pub top_concepts: Vec<(String, usize)>,       // (canonical, frequency) top 20
    pub contrastive_pairs: Vec<(String, String)>,  // pairs the LLM should NOT conflate
    pub vocabulary_warnings: Vec<String>,          // e.g. "use ndim, not n_dims or num_dims"
}
```

### `parser.rs` — Extended extraction

The existing `visit_node` function is extended to populate `signatures`, `classes`, and `call_sites` during the same tree-sitter walk.

```rust
// Existing API unchanged
pub fn parse_file(path: &Path) -> Result<ParseResult>;
pub fn parse_content(source: &str, path: &Path) -> Result<ParseResult>;
pub fn parse_directory(root: &Path, opts: &ParseOptions) -> Result<Vec<ParseResult>>;
```

New extraction in `visit_node`:

```
ON function_definition NODE:
    (existing) extract function name, parameters as RawIdentifiers
    (NEW) extract Signature:
        name = name node text
        params = for each parameter child:
            name = identifier text
            type_annotation = type child text (if present)
            default = default child text (if present)
        return_type = return_type child text (if present)
        decorators = from parent decorated_definition (if present)
        docstring_first_line = first string in body block (if present)
        file, line, scope from context

ON class_definition NODE:
    (existing) extract class name as RawIdentifier
    (NEW) extract ClassInfo:
        name = name node text
        bases = argument_list children text (if present)
        methods = names of function_definition children in body
        attributes = names of assignment targets in body (self.X patterns)
        docstring_first_line = first string in body block (if present)
        file, line from context

ON call EXPRESSION (within function/method bodies):
    (NEW) extract CallSite:
        callee = function child text (handles simple `foo()` and `obj.method()`)
        caller_scope = current scope from context
        file, line from context
```

### `graph.rs` — Extended graph

```rust
pub struct ConceptGraph {
    pub concepts: HashMap<u64, Concept>,
    pub relationships: Vec<Relationship>,
    pub conventions: Vec<Convention>,
    pub embeddings: EmbeddingIndex,
    pub signatures: Vec<Signature>,       // NEW — all signatures in corpus
    pub classes: Vec<ClassInfo>,          // NEW — all classes in corpus
    pub call_sites: Vec<CallSite>,        // NEW — all call sites in corpus
}

impl ConceptGraph {
    pub fn build(analysis: AnalysisResult, embeddings: EmbeddingIndex) -> Result<Self>;

    // Existing (enriched with L2 data in response)
    pub fn query_concept(&self, term: &str) -> Option<ConceptQueryResult>;
    pub fn check_naming(&self, identifier: &str) -> NamingCheckResult;
    pub fn suggest_name(&self, description: &str) -> Vec<NameSuggestion>;
    pub fn list_conventions(&self) -> &[Convention];
    pub fn list_concepts(&self) -> Vec<&Concept>;
    pub fn add_similarity_edges(&mut self, threshold: f32);

    // NEW — L2
    pub fn describe_symbol(&self, name: &str) -> Option<DescribeSymbolResult>;

    // NEW — locate_concept
    pub fn locate_concept(&self, term: &str) -> Option<LocateConceptResult>;

    // NEW — contrastive pair detection
    pub fn add_contrastive_edges(&mut self);

    // NEW — subconcept disambiguation
    pub fn detect_subconcepts(&mut self);

    // NEW — session briefing
    pub fn session_briefing(&self) -> SessionBriefing;
}
```

#### `query_concept` enrichment logic

```
EXISTING L1 LOGIC: (unchanged)
    match concept by canonical / occurrence / embedding
    collect variants, related concepts, conventions, top_occurrences

NEW L2 ENRICHMENT:
    signatures = self.signatures.iter()
        .filter(|sig| concept's identifiers contain sig.name
                      OR any concept subtoken appears in sig.name)
        .cloned().collect()

    classes = self.classes.iter()
        .filter(|cls| concept's identifiers contain cls.name
                      OR cls.methods contain any concept identifier)
        .cloned().collect()

    call_graph = for each signature matching this concept:
        find call_sites where caller_scope matches sig.name
        OR callee matches sig.name
        collect as (caller, callee) string pairs

RETURN ConceptQueryResult with all fields populated
```

#### `describe_symbol` logic

```
INPUT: name (e.g. "SpatialTransformer", "spatial_transform")

FIND SYMBOL:
    1. exact match in self.signatures by name
    2. exact match in self.classes by name
    3. if class match: also find Signature for methods matching class.methods

DETERMINE KIND:
    if found in classes -> SymbolKind::Class
    if found in signatures with scope containing a class -> SymbolKind::Method
    else -> SymbolKind::Function

FIND CALLERS:
    call_sites where callee == name

FIND CALLEES:
    call_sites where caller_scope == name (or ends with .name for methods)

FIND CONCEPTS:
    split name into subtokens
    find concepts whose canonical matches any subtoken

RETURN DescribeSymbolResult
```

#### `locate_concept` logic

`locate_concept` is optimized for the LLM use case: "I need to work with concept X — what's the minimum I should read?" It returns exemplar signatures and files ranked by concept density, not an exhaustive list.

```
INPUT: term (e.g. "transform")

FIND CONCEPT:
    same matching logic as query_concept (exact → subtoken → embedding)

RANK SIGNATURES:
    for each signature in self.signatures:
        score = 0
        if sig.name matches a concept identifier: score += 3
        if any concept subtoken appears in sig.name: score += 2
        if any param name matches a concept identifier: score += 1
    sort by score descending, take top 5

RANK CLASSES:
    for each class in self.classes:
        score = 0
        if cls.name matches a concept identifier: score += 3
        if any cls.method matches a concept identifier: score += 1 each
    sort by score descending, take top 3

RANK FILES:
    count occurrences per file for this concept
    sort by count descending

FIND CONTRASTIVE CONCEPTS:
    collect concepts linked by Contrastive relationship edges

RETURN LocateConceptResult
```

#### Contrastive pair detection logic

Detects concept pairs that consistently co-occur in the same scopes but fill complementary roles. Run after graph construction, alongside `add_similarity_edges`.

Three detection signals, merged into a single score:

```
KNOWN CONTRASTIVE PATTERNS:
    POSITIONAL_PAIRS = [
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
    ]

SIGNAL 1 — PARAMETER CO-OCCURRENCE:
    for each signature with >= 2 params:
        param_concepts = for each param, find matching concept canonicals
        for each pair (A, B) in param_concepts where A != B:
            increment param_co_count[(A, B)]

    // Two concepts appearing as separate params in the same function
    // is strong evidence of a contrastive relationship.
    // e.g. def register(source, target) — source and target are contrastive.

SIGNAL 2 — SCOPE CO-OCCURRENCE WITHOUT MERGING:
    // Concepts that appear as separate variables/attributes in the same
    // scope (function body or class), but are never part of the same
    // identifier. e.g. source_vol and target_vol in the same function.

    for each scope in the codebase:
        scope_concepts = set of concept canonicals appearing in this scope
            (via identifiers: source_vol → concept "source", target_vol → concept "target")
        for each pair (A, B) in scope_concepts where A != B:
            // Only count if A and B appear in SEPARATE identifiers
            // (not in the same compound identifier like source_to_target)
            if no identifier in this scope contains both A and B as subtokens:
                increment scope_co_count[(A, B)]

    // Weaker signal than param co-occurrence but catches:
    // - Variable pairs: source_vol / target_vol
    // - Attribute pairs: self.source / self.target
    // - Class-level pairs: SourceEncoder / TargetEncoder

SIGNAL 3 — KNOWN PATTERNS:
    for each (short_a, short_b) in POSITIONAL_PAIRS:
        if concept matching short_a exists AND concept matching short_b exists:
            mark as known_contrastive[(short_a, short_b)]

MERGE AND DECIDE:
    for each concept pair (A, B):
        score = 0
        if param_co_count[(A, B)] >= 3:  score += 2   // moderate signal (tuned: was >=2/+3)
        if scope_co_count[(A, B)] >= 3:  score += 2   // moderate signal
        if known_contrastive[(A, B)]:    score += 3   // known pattern

        if score >= 3:
            // Verify they are NOT the same concept (not abbreviation-linked)
            if no AbbreviationOf edge between A and B:
                add Contrastive edge

SUPPRESS SIMILARITY:
    for each Contrastive edge (A, B):
        remove any SimilarTo edge between A and B
    // Contrastive always wins over SimilarTo. A pair cannot be both.
```

#### Subconcept disambiguation logic

Detects polysemous concepts — single subtokens (like `transform`) that are used in multiple distinct domain contexts. Uses four signals: scope co-occurrence, embedding similarity, structural context (L2 signatures), and call graph connectivity.

Run after contrastive detection (so contrastive pairs are already marked), before session briefing.

```
CANDIDATE SELECTION:
    for each concept with >= 6 occurrences:
        // Only high-frequency concepts are worth splitting.
        // Below 6, there isn't enough evidence for 2+ clusters of 3.
        collect all unique identifiers for this concept
        if unique_identifiers.len() < 4:
            skip  // not enough distinct names to cluster

BUILD AFFINITY MATRIX:
    // For each pair of identifiers (i, j) belonging to this concept,
    // compute a multi-signal affinity score. High affinity = likely same
    // subconcept. Low affinity = likely different subconcepts.

    for each pair (id_i, id_j) in unique_identifiers:
        affinity = 0.0

        // Signal 1: Scope co-occurrence (weight 0.3)
        // Identifiers sharing a scope (same function, same class) are
        // likely the same subconcept. Scope-level is more precise than
        // file-level — a large file can contain multiple subconcepts.
        shared_scopes = count of scopes containing both id_i and id_j
        if shared_scopes > 0:
            affinity += 0.3 * min(shared_scopes, 3) / 3  // cap at 0.3

        // Signal 2: Embedding similarity (weight 0.3)
        // Embed each identifier (not just the concept canonical).
        // "spatial_transform" and "apply_transform" embed similarly.
        // "spatial_transform" and "AugmentTransform" embed differently.
        emb_i = embed_text(id_i)
        emb_j = embed_text(id_j)
        if both embeddings available:
            sim = cosine_similarity(emb_i, emb_j)
            affinity += 0.3 * max(0, sim)  // negative similarity → 0

        // Signal 3: Structural context overlap (weight 0.25)
        // If two identifiers are functions/methods, compare their param
        // vocabularies. spatial_transform(vol, trf) and apply_transform(src, trf)
        // share "trf" → high overlap. fourier_transform(signal, freq) shares
        // nothing → low overlap.
        sig_i = find signature matching id_i
        sig_j = find signature matching id_j
        if both signatures found:
            params_i = set of param name subtokens for sig_i
            params_j = set of param name subtokens for sig_j
            if params_i.union(params_j).len() > 0:
                jaccard = params_i.intersection(params_j).len()
                         / params_i.union(params_j).len()
                affinity += 0.25 * jaccard

        // Signal 4: Call graph connectivity (weight 0.15)
        // If id_i calls id_j (or vice versa), or they share callers/callees,
        // they are likely the same subconcept.
        shared_calls = count of (
            call_sites where caller contains id_i AND callee contains id_j
            OR caller contains id_j AND callee contains id_i
            OR both are called by the same function
        )
        if shared_calls > 0:
            affinity += 0.15 * min(shared_calls, 3) / 3  // cap at 0.15

        affinity_matrix[(id_i, id_j)] = affinity

CLUSTER VIA CONNECTED COMPONENTS WITH THRESHOLD:
    // Convert affinity matrix into a graph where edges exist only
    // between identifiers with affinity >= threshold.
    // Connected components of this thresholded graph are the subconcepts.

    AFFINITY_THRESHOLD = 0.25
    // Chosen so that identifiers need at least moderate evidence from
    // 2+ signals to be grouped. Pure file co-occurrence (0.3 max from
    // one signal) barely passes; embedding similarity + scope co-occurrence
    // easily passes.

    edges = [(i, j) for (i, j) in affinity_matrix
             if affinity_matrix[(i, j)] >= AFFINITY_THRESHOLD]

    clusters = connected_components(identifiers, edges)
    // Standard union-find algorithm — O(n * alpha(n)), no external deps.

    // Discard clusters with < 2 identifiers (noise / singletons)
    clusters = clusters.filter(|c| c.len() >= 2)

    if clusters.len() < 2:
        skip  // concept is not polysemous — all identifiers are connected

VALIDATE CLUSTERS:
    // Each cluster must be internally coherent: average pairwise affinity
    // within the cluster must be above a minimum. This catches cases where
    // connected components produces a cluster from a chain of weak edges
    // (A↔B↔C where A and C have zero affinity).

    COHERENCE_THRESHOLD = 0.2

    for each cluster:
        if cluster.len() <= 2:
            continue  // pairs are coherent by definition
        avg_affinity = mean of affinity_matrix[(i, j)]
                       for all pairs (i, j) in cluster
        if avg_affinity < COHERENCE_THRESHOLD:
            // Cluster is not coherent — try splitting it further.
            // Recursively apply the same algorithm with a higher threshold
            // (AFFINITY_THRESHOLD + 0.1). If this produces subclusters,
            // replace the incoherent cluster with them.
            // If not, keep the original cluster as-is (it's weak but the
            // best we can do).
            sub_clusters = re_cluster(cluster, threshold + 0.1)
            if sub_clusters.len() >= 2:
                replace cluster with sub_clusters

DETERMINE QUALIFIERS:
    for each cluster:
        // The qualifier is the most distinguishing subtoken in this cluster
        // that does NOT appear in other clusters.
        all_subtokens = flatten(split_identifier(id) for id in cluster)
        other_subtokens = flatten(split_identifier(id) for id in other clusters)
        distinguishing = all_subtokens - other_subtokens - {concept.canonical}
        qualifier = most_frequent(distinguishing)
        if qualifier is empty:
            // Fallback: use the most common non-parent subtoken in this cluster
            qualifier = most_frequent(all_subtokens - {concept.canonical})
        if qualifier is still empty:
            qualifier = format!("group_{cluster_index}")  // last resort

BUILD SUBCONCEPTS:
    for each (cluster, qualifier):
        subconcept = Subconcept {
            qualifier,
            canonical: format!("{}.{}", concept.canonical, qualifier),
            occurrences: concept.occurrences filtered to cluster identifiers,
            identifiers: cluster identifiers,
            embedding: embed the subconcept canonical + cluster identifiers,
        }
    concept.subconcepts = all subconcepts
```

**Why these signals and weights:**

| Signal | Weight | Rationale |
|---|---|---|
| Scope co-occurrence | 0.30 | Strongest structural signal — if two identifiers share a function/class scope, they're almost certainly the same subconcept. Scope-level (not file-level) avoids the "large utility file" problem. |
| Embedding similarity | 0.30 | Captures semantic similarity that naming alone misses. `trf` and `spatial_transform` embed similarly despite different surface forms. `spatial_transform` and `fourier_transform` embed differently despite sharing `transform`. |
| Structural context (L2) | 0.25 | Parameter vocabulary overlap is a strong signal. Functions operating on the same types of data belong together. This is the signal that only L2 data provides. |
| Call graph connectivity | 0.15 | Weaker signal — functions may call each other for many reasons. But mutual calling within a concept cluster confirms they're the same subconcept. |

**Why connected components (not greedy or k-means):**

- **Deterministic** — same input always produces the same output. No seed dependency.
- **No k parameter** — we don't know how many subconcepts exist. Connected components discovers the natural number.
- **Robust to noise** — singleton identifiers that connect to nothing form their own (discarded) components instead of being forced into a cluster.
- **Coherence validation** catches weak chains — a connected component where A↔B↔C exists but A and C have zero affinity gets recursively split.
- **O(n · α(n))** via union-find — fast enough for any realistic identifier set.

**Example on voxelmorph:**

```
concept "transform" has identifiers:
    spatial_transform (12x), apply_transform (5x), trf (8x),
    SpatialTransformer (4x), compose_transforms (3x),
    AugmentTransform (2x), RandomTransform (2x),
    fourier_transform (1x)

Affinity matrix (abbreviated):
    spatial_transform ↔ apply_transform:  0.72
        scope: share utils.py (0.3), embedding: 0.85 (0.26),
        params: both take (vol, trf) → jaccard 1.0 (0.25),
        calls: apply_transform calls spatial_transform (0.15)
    spatial_transform ↔ trf:              0.55
        scope: share layers.py (0.3), embedding: 0.61 (0.18),
        params: trf has no signature (0.0), calls: none (0.0)
        + abbreviation link boosts affinity
    spatial_transform ↔ AugmentTransform: 0.08
        scope: different files (0.0), embedding: 0.31 (0.09),
        params: no overlap (0.0), calls: none (0.0)
    AugmentTransform ↔ RandomTransform:   0.65
        scope: share augment.py (0.3), embedding: 0.78 (0.23),
        params: similar augmentation params (0.12), calls: none (0.0)

Thresholded graph (>= 0.25):
    {spatial_transform — apply_transform — trf — SpatialTransformer — compose_transforms}
    {AugmentTransform — RandomTransform}
    {fourier_transform}  ← singleton, discarded

Connected components:
    Cluster A: {spatial_transform, apply_transform, trf, SpatialTransformer, compose_transforms}
    Cluster B: {AugmentTransform, RandomTransform}

Validation:
    Cluster A avg pairwise affinity: 0.51 ✓ (above 0.2)
    Cluster B avg pairwise affinity: 0.65 ✓ (above 0.2)

Qualifiers:
    Cluster A: distinguishing = {spatial, apply, compose} → "spatial" (most frequent)
    Cluster B: distinguishing = {augment, random} → "augment" (most frequent)

Result:
    concept.subconcepts = [
        Subconcept { qualifier: "spatial", canonical: "transform.spatial",
                     identifiers: [spatial_transform, apply_transform, trf, ...] },
        Subconcept { qualifier: "augment", canonical: "transform.augment",
                     identifiers: [AugmentTransform, RandomTransform] },
    ]
```

**How subconcepts affect queries:**

When `query_concept("transform")` is called on a concept with subconcepts:
- The response includes the parent concept as before (all occurrences, all variants)
- The `subconcepts` field is populated with the distinct clusters
- Each subconcept has its own occurrences, identifiers, and embedding
- The LLM can see: "transform has two distinct usages: spatial (displacement-based warping) and augment (data augmentation)" — even without reading any files

When `locate_concept("transform")` is called:
- Exemplar signatures are drawn from across subconcepts (at least one per subconcept)
- The response indicates which subconcept each signature belongs to

When `locate_concept("transform.spatial")` is called:
- The dotted form is recognized as a subconcept query
- Only signatures/files from the spatial cluster are returned

When `check_naming("fourier_transform")` is called:
- If `fourier_transform` has only 1 occurrence, it may not form its own subconcept
- But the `Unknown` verdict (too few examples) is still correct
- If more examples accumulate, a third subconcept emerges automatically

#### Session briefing logic

Generated from the current graph state. This is a read-only summary — no new computation.

```
BUILD BRIEFING:
    conventions = graph.list_conventions()

    abbreviations = collect all AbbreviationOf relationships
        format as (short_form, expanded_form)

    top_concepts = graph.list_concepts().take(20)
        format as (canonical, occurrence_count)

    contrastive_pairs = collect all Contrastive relationship edges
        format as (concept_a.canonical, concept_b.canonical)

    vocabulary_warnings = for each concept:
        if check_naming(concept.canonical) == Inconsistent:
            format as "use {suggestion}, not {canonical}"
        // Also: if multiple canonical forms exist for same concept,
        // warn about the less-frequent one

RETURN SessionBriefing
```

### `tools.rs` — MCP tool definitions

#### Existing tools (enriched responses)

```rust
// query_concept — response includes L2 data; optional limit params control response size
// Params: term (required), max_related (default 10), max_occurrences (default 5),
//         max_variants (default 20), max_signatures (default 5)
// `related` field returns lightweight RelatedConcept summaries, not full Concept objects.
// Default response is ~3.7k tokens (was ~36k before limits).
Tool::new("query_concept", ...)

// check_naming — unchanged
Tool::new("check_naming", ...)

// suggest_name — unchanged
Tool::new("suggest_name", ...)

// ontology_diff — unchanged
Tool::new("ontology_diff", ...)

// list_concepts — unchanged
Tool::new("list_concepts", ...)

// list_conventions — unchanged
Tool::new("list_conventions", ...)
```

#### New tools

```rust
// L2: describe_symbol
Tool::new(
    "describe_symbol",
    "Get full structural information for a function or class: signature,
     callers, callees, and related domain concepts.",
    schema: { "name": { "type": "string", "description": "Function or class name" } },
    required: ["name"],
)

// locate_concept — optimized for "what should I read?"
Tool::new(
    "locate_concept",
    "Find the best entry points for working with a domain concept.
     Returns exemplar signatures, key classes, and files ranked by
     concept density — the minimum context an LLM needs to understand
     and work with this concept. Also flags contrastive concepts that
     should not be conflated.",
    schema: { "term": { "type": "string", "description": "Concept to locate (e.g. 'transform')" } },
    required: ["term"],
)
```

#### MCP resource

```rust
// Session briefing — proactive context injection
// Exposed as an MCP resource, not a tool. The MCP client reads this
// automatically when connecting, without the LLM needing to call a tool.
Resource {
    uri: "semex://briefing",
    name: "Session Briefing",
    description: "Project conventions, abbreviations, top concepts, contrastive
                  pairs, and vocabulary warnings. Read this at session start to
                  ground your understanding of the project's domain vocabulary.",
    mime_type: "application/json",
}
// Returns: SessionBriefing serialized as JSON
```

### CLI

```
semex --repo <PATH> [COMMAND]

Commands:
  serve          Start MCP server on stdio (default)
  query          Look up a domain concept by name
  check          Check an identifier against naming conventions
  suggest        Suggest an identifier name from a description
  diff           Compare domain ontology between git refs
  concepts       List all detected domain concepts
  conventions    List all detected naming conventions
  describe       Describe a function or class (L2)          # NEW
  locate         Find entry points for a concept (L2)       # NEW
  briefing       Print session briefing to stdout            # NEW
```

New subcommands:

```rust
/// Describe a function or class — structural info (L2)
Describe {
    /// Function or class name
    name: String,
}

/// Find the best entry points for working with a concept
Locate {
    /// Concept to locate (e.g. "transform")
    term: String,
}

/// Print session briefing (conventions, abbreviations, warnings)
Briefing,
```

## Logic flow

### Indexing pipeline (startup) — extended

```
INIT:
    repo_root = resolve from CLI arg or cwd
    config = load .semex/config.toml (or defaults)
    cache = IndexCache::open(repo_root/.semex/index.json)

CHECK CACHE:
    if cache.load() returns valid graph AND no files changed:
        graph = cached graph
        skip to SERVE

PARSE (L1 + L2 in single pass):
    py_files = glob repo_root for **/*.py (respecting .gitignore)
    parse_results = parse each file via tree-sitter (parallel with rayon)
        each parse produces: identifiers + doc_texts (L1)
                            + signatures + classes + call_sites (L2)

ANALYZE (L1):
    aggregate subtokens, compute TF-IDF, detect conventions
    build co-occurrence matrix

EMBED:
    for each concept with frequency >= 2:
        compute embedding, add SimilarTo edges

BUILD GRAPH (L1 + L2):
    construct ConceptGraph from analysis + embeddings + relationships
    attach signatures, classes, call_sites to graph

DETECT CONTRASTIVE PAIRS:
    graph.add_contrastive_edges()
    // Must run AFTER similarity edges — contrastive suppresses SimilarTo

DETECT SUBCONCEPTS:
    graph.detect_subconcepts()
    // Must run AFTER embeddings (uses embedding similarity within clusters)
    // and AFTER co-occurrence matrix is built

PERSIST:
    cache.save(graph)

SERVE:
    start file watcher
    register all MCP tools (L1 + L2)
    register MCP resource: semex://briefing
    enter MCP event loop
```

### Incremental update (on file change)

```
ON FILE CHANGE (paths: Vec<PathBuf>):
    re-parse changed files (produces L1 + L2 data)
    rebuild full graph (recompute concepts, attach new signatures/classes/calls)
    cache.save(graph)
```

## Configuration

No new configuration required. All existing v1 config applies unchanged.

```toml
# .semex/config.toml — unchanged from v1

[index]
include = ["**/*.py"]
exclude = ["**/test_*", "**/__pycache__"]
min_frequency = 2
respect_gitignore = true

[analysis]
domain_specificity_threshold = 0.3
co_occurrence_scope = "function"
convention_threshold = 3

[embeddings]
enabled = true
model = "BAAI/bge-small-en-v1.5"
similarity_threshold = 0.75
model_cache_dir = "~/.cache/semex"

[cache]
enabled = true
path = ".semex/index.db"

[[conventions]]
pattern = "prefix"
value = "nb_"
role = "count"
entity_types = ["Parameter", "Variable"]

[[conventions]]
pattern = "prefix"
value = "is_"
role = "boolean predicate"
entity_types = ["Function", "Variable"]

[[conventions]]
pattern = "conversion"
value = "_to_"
role = "type/format conversion"
entity_types = ["Function"]
```

# Testing

## Unit tests

### `parser.rs` — L2 extraction

- `test_signatures_extracted`: fixture with typed and untyped functions → correct `Signature` structs with params, types, return types
- `test_class_info_extracted`: fixture with class hierarchy → correct `ClassInfo` with bases, methods, attributes
- `test_call_sites_extracted`: fixture with function calls → correct `CallSite` structs with caller scope and callee name
- `test_method_call_sites`: `self.method()` and `obj.method()` calls → callee includes attribute chain
- `test_signatures_include_decorators`: decorated functions → decorators captured in `Signature`
- `test_docstring_first_line_in_signature`: functions with docstrings → first line captured

### `graph.rs` — L2 query enrichment

- `test_query_concept_includes_signatures`: query "transform" on graph with `spatial_transform` signature → response includes matching signature
- `test_query_concept_includes_classes`: query "transform" on graph with `SpatialTransformer` class → response includes matching class
- `test_query_concept_includes_call_graph`: query with call sites → response includes relevant (caller, callee) pairs
- `test_describe_symbol_function`: describe "spatial_transform" → returns signature, callers, callees, concepts
- `test_describe_symbol_class`: describe "SpatialTransformer" → returns class info, method signatures, concepts
- `test_describe_symbol_not_found`: describe nonexistent name → returns None

### `graph.rs` — Contrastive pair detection

- `test_contrastive_from_params`: `source` and `target` appearing as separate params in >= 2 signatures → Contrastive edge
- `test_contrastive_from_scope_variables`: `source_vol` and `target_vol` as variables in same function body → Contrastive edge between `source` and `target` concepts
- `test_contrastive_from_attributes`: `self.source` and `self.target` as attributes in same class → Contrastive edge
- `test_contrastive_suppresses_similar`: if both SimilarTo and Contrastive would apply, only Contrastive survives
- `test_contrastive_known_patterns`: `input`/`output` pair detected from known patterns even below co-occurrence threshold
- `test_contrastive_not_triggered_for_same_identifier`: `source` and `target` appearing as subtokens of the SAME identifier (e.g. `source_to_target`) do NOT contribute to contrastive score
- `test_contrastive_not_triggered_for_abbreviations`: if `src` is an abbreviation of `source`, they do NOT get a contrastive edge

### `graph.rs` — Subconcept disambiguation

- `test_subconcept_splits_polysemous`: concept with two clusters of identifiers that have low cross-cluster affinity → two subconcepts with correct qualifiers
- `test_subconcept_not_triggered_below_threshold`: concept with < 6 occurrences → no subconcepts
- `test_subconcept_single_cluster_no_split`: concept where all identifiers have high pairwise affinity → no subconcepts (not polysemous)
- `test_subconcept_uses_embedding_signal`: two identifiers in different files but with high embedding similarity → grouped into same subconcept
- `test_subconcept_uses_structural_context`: two functions with overlapping param vocabularies (both take `vol, trf`) → grouped; function with different params (`signal, freq`) → separate subconcept
- `test_subconcept_coherence_validation`: cluster formed by weak chain (A↔B↔C where A↔C affinity is 0) → recursively split into coherent subclusters
- `test_subconcept_qualifier_selection`: distinguishing subtoken chosen as qualifier, not the shared parent canonical
- `test_subconcept_dotted_query`: `locate_concept("transform.spatial")` returns only spatial cluster results
- `test_query_concept_includes_subconcepts`: query on polysemous concept → response includes populated subconcepts field

### `graph.rs` — locate_concept

- `test_locate_concept_ranks_signatures`: locate "transform" → top signatures are those with "transform" in name, not just any signature
- `test_locate_concept_includes_contrastive`: locate "source" → contrastive_concepts includes "target"
- `test_locate_concept_files_ranked_by_density`: files with more occurrences ranked higher

### `graph.rs` — session briefing

- `test_briefing_includes_conventions`: briefing contains detected conventions
- `test_briefing_includes_abbreviations`: briefing contains AbbreviationOf pairs
- `test_briefing_includes_contrastive_pairs`: briefing lists contrastive pairs

### `tools.rs` — new tool handlers

- `test_describe_symbol_tool`: valid name → returns structured result
- `test_describe_symbol_tool_not_found`: unknown name → returns not_found error
- `test_locate_concept_tool`: valid concept → returns exemplar signatures and files
- `test_locate_concept_tool_not_found`: unknown concept → returns not_found error
- `test_briefing_resource`: MCP resource read → returns valid SessionBriefing JSON

### `cache.rs` — L2 persistence

- `test_cache_roundtrip_with_signatures`: save/load preserves signatures and classes
- `test_cache_roundtrip_with_call_sites`: save/load preserves call sites

## Integration tests

- Full pipeline on voxelmorph: verify `query_concept("transform")` returns signatures for `spatial_transform`, class info for relevant classes, and call graph edges
- `describe_symbol("spatial_transform")` on voxelmorph: returns signature with params, callers, callees
- `locate_concept("transform")` on voxelmorph: returns ranked signatures and files, not exhaustive list
- Contrastive detection on voxelmorph: `source`/`target` detected as contrastive pair (they appear as complementary params in registration functions)
- Subconcept detection: if voxelmorph has both spatial transform and augmentation transform identifiers in separate files, `transform` concept gains subconcepts
- Session briefing on voxelmorph: briefing includes `nb_` convention, `trf`→`transform` abbreviation, `source`/`target` contrastive pair
- MCP tool round-trip for all 8 tools + 1 resource: send JSON-RPC requests, verify response schemas
- Empty repo: all tools return sensible empty responses

# Definition of done

- `cargo build --release` compiles with no warnings
- `cargo test` passes all unit and integration tests (existing + new L2 tests)
- `cargo clippy` produces no warnings
- `ParseResult` includes `signatures`, `classes`, `call_sites` fields populated by tree-sitter
- `query_concept` response includes matching signatures, classes, and call graph edges
- `describe_symbol` MCP tool returns structural info for known functions and classes
- `describe_symbol("spatial_transform")` on voxelmorph returns correct signature with params and types
- All 8 MCP tools (`query_concept`, `check_naming`, `suggest_name`, `ontology_diff`, `list_concepts`, `list_conventions`, `describe_symbol`, `locate_concept`) return correctly shaped JSON
- MCP resource `semex://briefing` returns valid SessionBriefing JSON
- Contrastive pair detection: `source`/`target` detected as contrastive on voxelmorph
- `locate_concept("transform")` on voxelmorph returns ranked exemplar signatures (not exhaustive)
- Subconcept disambiguation: polysemous concepts with distinct usage clusters are split with correct qualifiers
- `query_concept` on a polysemous concept returns populated `subconcepts` field
- Session briefing includes conventions, abbreviations, contrastive pairs, and vocabulary warnings
- Cache round-trip preserves all L2 data including contrastive edges
- No performance regression: indexing voxelmorph completes within 2x of v1 time
- All existing v1 acceptance tests still pass (zero regressions)

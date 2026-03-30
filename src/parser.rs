use crate::types::{
    CallSite, ClassInfo, EntityType, Param, ParseResult, RawIdentifier,
    Signature,
};
use anyhow::Result;
use ignore::WalkBuilder;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

// ---------------------------------------------------------------------------
// LanguageParser trait
// ---------------------------------------------------------------------------

/// Trait abstracting language-specific parsing behavior.
///
/// Each implementation handles one language family (Python, TypeScript, JS).
/// The shared tree-walking infrastructure calls `visit_node` at each AST
/// node and lets the implementation decide what to extract.
#[allow(dead_code)]
pub trait LanguageParser: Send + Sync {
    /// File extensions this parser handles (without the dot).
    fn extensions(&self) -> &[&str];

    /// Create a tree-sitter parser with the language grammar loaded.
    fn make_parser(&self) -> Result<tree_sitter::Parser>;

    /// Create a tree-sitter parser appropriate for a specific file path.
    ///
    /// Default delegates to `make_parser()`. Override when a single parser
    /// handles multiple grammars (e.g. TS vs TSX).
    fn make_parser_for_path(&self, _path: &Path) -> Result<tree_sitter::Parser> {
        self.make_parser()
    }

    /// Parameter names to skip (e.g. `["self", "cls"]` for Python).
    fn skip_params(&self) -> &[&str];

    /// Strip language-specific doc-comment syntax from raw text.
    fn strip_doc_syntax<'a>(&self, raw: &'a str) -> &'a str;

    /// Visit one AST node, extracting identifiers, docs, signatures, etc.
    #[allow(clippy::too_many_arguments)]
    fn visit_node(
        &self,
        node: tree_sitter::Node,
        source: &[u8],
        path: &Path,
        scope: &Option<String>,
        identifiers: &mut Vec<RawIdentifier>,
        doc_texts: &mut Vec<(PathBuf, usize, String)>,
        signatures: &mut Vec<Signature>,
        classes: &mut Vec<ClassInfo>,
        call_sites: &mut Vec<CallSite>,
    );
}

// ---------------------------------------------------------------------------
// Public API — backward-compatible (defaults to Python)
// ---------------------------------------------------------------------------

/// Parse a single Python file and extract identifiers and docstrings.
#[allow(dead_code)]
pub fn parse_file(path: &Path) -> Result<ParseResult> {
    parse_file_with(path, &PythonParser)
}

/// Parse Python source content and extract identifiers and docstrings.
///
/// Like `parse_file`, but takes source content directly instead of reading
/// from disk. Used by the diff module to parse files from git tree objects.
#[cfg(test)]
pub fn parse_content(source: &str, path: &Path) -> Result<ParseResult> {
    parse_content_with(source, path, &PythonParser)
}

// ---------------------------------------------------------------------------
// Public API — language-parametric
// ---------------------------------------------------------------------------

/// Parse a single file using the given language parser.
pub fn parse_file_with(
    path: &Path,
    lang: &dyn LanguageParser,
) -> Result<ParseResult> {
    let source = std::fs::read_to_string(path)?;
    parse_content_with(&source, path, lang)
}

/// Parse source content using the given language parser.
pub fn parse_content_with(
    source: &str,
    path: &Path,
    lang: &dyn LanguageParser,
) -> Result<ParseResult> {
    let mut parser = lang.make_parser_for_path(path)?;

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("Failed to parse {}", path.display()))?;

    let source_bytes = source.as_bytes();
    let mut identifiers = Vec::new();
    let mut doc_texts = Vec::new();
    let mut signatures = Vec::new();
    let mut classes = Vec::new();
    let mut call_sites = Vec::new();

    lang.visit_node(
        tree.root_node(),
        source_bytes,
        path,
        &None,
        &mut identifiers,
        &mut doc_texts,
        &mut signatures,
        &mut classes,
        &mut call_sites,
    );

    Ok(ParseResult {
        identifiers,
        doc_texts,
        signatures,
        classes,
        call_sites,
    })
}

/// Options controlling which files are parsed.
pub struct ParseOptions {
    /// Glob patterns to include (e.g. `["**/*.py"]`).
    pub include: Vec<String>,
    /// Glob patterns to exclude (e.g. `["**/test_*"]`).
    pub exclude: Vec<String>,
    /// Whether to honor `.gitignore` rules.
    pub respect_gitignore: bool,
}

/// Parse all matching files in a directory tree using the given language
/// parser and parallel iteration.
pub fn parse_directory_with(
    root: &Path,
    opts: &ParseOptions,
    lang: &dyn LanguageParser,
) -> Result<Vec<ParseResult>> {
    let mut builder = WalkBuilder::new(root);
    builder.standard_filters(opts.respect_gitignore);

    let mut overrides = ignore::overrides::OverrideBuilder::new(root);
    for pat in &opts.include {
        overrides.add(pat)?;
    }
    for pat in &opts.exclude {
        overrides.add(&format!("!{pat}"))?;
    }
    builder.overrides(overrides.build()?);

    let paths: Vec<_> = builder
        .build()
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.file_type().is_some_and(|ft| ft.is_file()))
        .map(|entry| entry.into_path())
        .collect();

    let results: Vec<ParseResult> = paths
        .par_iter()
        .filter_map(|p| parse_file_with(p, lang).ok())
        .collect();

    Ok(results)
}

// ---------------------------------------------------------------------------
// Constructor helpers
// ---------------------------------------------------------------------------

#[allow(dead_code)]
pub fn python_parser() -> PythonParser {
    PythonParser
}

#[allow(dead_code)]
pub fn typescript_parser() -> TypeScriptParser {
    TypeScriptParser
}

#[allow(dead_code)]
pub fn javascript_parser() -> JavaScriptParser {
    JavaScriptParser
}

// ---------------------------------------------------------------------------
// Shared helpers used across language parsers
// ---------------------------------------------------------------------------

/// Recurse into all children, delegating each to the parser's `visit_node`.
#[allow(clippy::too_many_arguments)]
fn visit_children(
    lang: &dyn LanguageParser,
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
    doc_texts: &mut Vec<(PathBuf, usize, String)>,
    signatures: &mut Vec<Signature>,
    classes: &mut Vec<ClassInfo>,
    call_sites: &mut Vec<CallSite>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        lang.visit_node(
            child,
            source,
            path,
            scope,
            identifiers,
            doc_texts,
            signatures,
            classes,
            call_sites,
        );
    }
}

/// Find the first named child with a given node kind.
fn find_named_child_by_kind<'a>(
    node: tree_sitter::Node<'a>,
    kind: &str,
) -> Option<tree_sitter::Node<'a>> {
    for i in 0..node.named_child_count() {
        if let Some(child) = node.named_child(i) {
            if child.kind() == kind {
                return Some(child);
            }
        }
    }
    None
}

/// Find the first `identifier` child of a node.
fn find_first_identifier_child(
    node: tree_sitter::Node,
) -> Option<tree_sitter::Node> {
    find_named_child_by_kind(node, "identifier")
}

/// Extract a `CallSite` from a call expression node.
///
/// Works for both Python (`call`, function field = `function`) and TS/JS
/// (`call_expression`, function field = `function`).
fn extract_call_site(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
) -> Option<CallSite> {
    let func_node = node.child_by_field_name("function")?;
    let callee = func_node.utf8_text(source).ok()?.to_string();
    let line = node.start_position().row + 1;

    Some(CallSite {
        caller_scope: scope.clone(),
        callee,
        file: path.to_path_buf(),
        line,
    })
}

/// Extract type identifier from a type annotation node (Python).
fn extract_type_annotation(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    match node.kind() {
        "identifier" | "type_identifier" => {
            if let Ok(name) = node.utf8_text(source) {
                identifiers.push(RawIdentifier {
                    name: name.to_string(),
                    entity_type: EntityType::TypeAnnotation,
                    file: path.to_path_buf(),
                    line: node.start_position().row + 1,
                    scope: scope.clone(),
                });
            }
        }
        // `type` node wraps the actual type expression in assignments
        "type" => {
            if let Some(child) = node.named_child(0) {
                extract_type_annotation(
                    child, source, path, scope, identifiers,
                );
            }
        }
        // For `Optional[X]`, `List[X]`, etc. -- extract the outer type
        "subscript" => {
            if let Some(value) = node.child_by_field_name("value") {
                extract_type_annotation(
                    value, source, path, scope, identifiers,
                );
            }
        }
        "attribute" => {
            // e.g., `torch.Tensor` -- extract as-is
            if let Ok(name) = node.utf8_text(source) {
                identifiers.push(RawIdentifier {
                    name: name.to_string(),
                    entity_type: EntityType::TypeAnnotation,
                    file: path.to_path_buf(),
                    line: node.start_position().row + 1,
                    scope: scope.clone(),
                });
            }
        }
        _ => {}
    }
}

// =========================================================================
// PythonParser
// =========================================================================

pub struct PythonParser;

const PY_SKIP_PARAMS: &[&str] = &["self", "cls"];

impl LanguageParser for PythonParser {
    fn extensions(&self) -> &[&str] {
        &["py"]
    }

    fn make_parser(&self) -> Result<tree_sitter::Parser> {
        let mut parser = tree_sitter::Parser::new();
        let language = tree_sitter_python::LANGUAGE;
        parser
            .set_language(&language.into())
            .map_err(|e| anyhow::anyhow!("Error loading Python grammar: {e}"))?;
        Ok(parser)
    }

    fn skip_params(&self) -> &[&str] {
        PY_SKIP_PARAMS
    }

    fn strip_doc_syntax<'a>(&self, raw: &'a str) -> &'a str {
        strip_docstring_quotes(raw)
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_node(
        &self,
        node: tree_sitter::Node,
        source: &[u8],
        path: &Path,
        scope: &Option<String>,
        identifiers: &mut Vec<RawIdentifier>,
        doc_texts: &mut Vec<(PathBuf, usize, String)>,
        signatures: &mut Vec<Signature>,
        classes: &mut Vec<ClassInfo>,
        call_sites: &mut Vec<CallSite>,
    ) {
        match node.kind() {
            "function_definition" => {
                let func_name = py_extract_function(
                    node, source, path, scope, identifiers,
                );
                if let Some(sig) =
                    py_extract_signature(node, source, path, scope)
                {
                    signatures.push(sig);
                }
                let child_scope = func_name.map(|name| match scope {
                    Some(parent) => format!("{parent}.{name}"),
                    None => name,
                });
                visit_children(
                    self, node, source, path, &child_scope,
                    identifiers, doc_texts, signatures, classes,
                    call_sites,
                );
                return;
            }
            "class_definition" => {
                let class_name = py_extract_class(
                    node, source, path, scope, identifiers,
                );
                if let Some(info) = py_extract_class_info(node, source, path)
                {
                    classes.push(info);
                }
                let child_scope = class_name.or_else(|| scope.clone());
                visit_children(
                    self, node, source, path, &child_scope,
                    identifiers, doc_texts, signatures, classes,
                    call_sites,
                );
                return;
            }
            "decorated_definition" => {
                py_extract_decorators(
                    node, source, path, scope, identifiers,
                );
            }
            "assignment" => {
                py_extract_assignment(
                    node, source, path, scope, identifiers,
                );
            }
            "expression_statement" => {
                py_try_extract_docstring(node, source, path, doc_texts);
            }
            "call" => {
                if let Some(cs) =
                    extract_call_site(node, source, path, scope)
                {
                    call_sites.push(cs);
                }
            }
            _ => {}
        }

        visit_children(
            self, node, source, path, scope, identifiers, doc_texts,
            signatures, classes, call_sites,
        );
    }
}

// --- Python-specific helpers ---

fn py_extract_function(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) -> Option<String> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?;

    identifiers.push(RawIdentifier {
        name: name.to_string(),
        entity_type: EntityType::Function,
        file: path.to_path_buf(),
        line: name_node.start_position().row + 1,
        scope: scope.clone(),
    });

    if let Some(params_node) = node.child_by_field_name("parameters") {
        let func_scope = Some(match scope {
            Some(parent) => format!("{parent}.{name}"),
            None => name.to_string(),
        });
        py_extract_parameters(
            params_node, source, path, &func_scope, identifiers,
        );
    }

    Some(name.to_string())
}

fn py_extract_parameters(
    params_node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    let mut cursor = params_node.walk();
    for child in params_node.named_children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                py_push_param_if_valid(
                    child, source, path, scope, identifiers,
                );
            }
            "typed_parameter" | "typed_default_parameter" => {
                if let Some(name_node) = find_first_identifier_child(child) {
                    py_push_param_if_valid(
                        name_node, source, path, scope, identifiers,
                    );
                }
                if let Some(type_node) = child.child_by_field_name("type")
                {
                    extract_type_annotation(
                        type_node, source, path, scope, identifiers,
                    );
                }
            }
            "default_parameter" => {
                if let Some(name_node) = find_first_identifier_child(child) {
                    py_push_param_if_valid(
                        name_node, source, path, scope, identifiers,
                    );
                }
            }
            _ => {}
        }
    }
}

fn py_push_param_if_valid(
    name_node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    if let Ok(name) = name_node.utf8_text(source) {
        if !PY_SKIP_PARAMS.contains(&name) {
            identifiers.push(RawIdentifier {
                name: name.to_string(),
                entity_type: EntityType::Parameter,
                file: path.to_path_buf(),
                line: name_node.start_position().row + 1,
                scope: scope.clone(),
            });
        }
    }
}

fn py_extract_class(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) -> Option<String> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?;

    identifiers.push(RawIdentifier {
        name: name.to_string(),
        entity_type: EntityType::Class,
        file: path.to_path_buf(),
        line: name_node.start_position().row + 1,
        scope: scope.clone(),
    });

    Some(match scope {
        Some(parent) => format!("{parent}.{name}"),
        None => name.to_string(),
    })
}

fn py_extract_decorators(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "decorator" {
            if let Some(name) = py_extract_decorator_name(child, source) {
                identifiers.push(RawIdentifier {
                    name,
                    entity_type: EntityType::Decorator,
                    file: path.to_path_buf(),
                    line: child.start_position().row + 1,
                    scope: scope.clone(),
                });
            }
        }
    }
}

fn py_extract_decorator_name(
    decorator_node: tree_sitter::Node,
    source: &[u8],
) -> Option<String> {
    let mut cursor = decorator_node.walk();
    for child in decorator_node.named_children(&mut cursor) {
        match child.kind() {
            "identifier" | "attribute" => {
                return child
                    .utf8_text(source)
                    .ok()
                    .map(|s| s.to_string());
            }
            "call" => {
                if let Some(func_node) =
                    child.child_by_field_name("function")
                {
                    return func_node
                        .utf8_text(source)
                        .ok()
                        .map(|s| s.to_string());
                }
            }
            _ => {}
        }
    }
    None
}

fn py_extract_assignment(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    if let Some(left) = node.child_by_field_name("left") {
        match left.kind() {
            "identifier" => {
                if let Ok(name) = left.utf8_text(source) {
                    identifiers.push(RawIdentifier {
                        name: name.to_string(),
                        entity_type: EntityType::Variable,
                        file: path.to_path_buf(),
                        line: left.start_position().row + 1,
                        scope: scope.clone(),
                    });
                }
            }
            "attribute" => {
                py_extract_attribute(
                    left, source, path, scope, identifiers,
                );
            }
            _ => {}
        }
    }
    if let Some(type_node) = node.child_by_field_name("type") {
        extract_type_annotation(
            type_node, source, path, scope, identifiers,
        );
    }
}

fn py_extract_attribute(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    if let Some(attr_node) = node.child_by_field_name("attribute") {
        if let Ok(name) = attr_node.utf8_text(source) {
            identifiers.push(RawIdentifier {
                name: name.to_string(),
                entity_type: EntityType::Attribute,
                file: path.to_path_buf(),
                line: attr_node.start_position().row + 1,
                scope: scope.clone(),
            });
        }
    }
}

fn py_try_extract_docstring(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    doc_texts: &mut Vec<(PathBuf, usize, String)>,
) {
    let string_node = match find_named_child_by_kind(node, "string") {
        Some(n) => n,
        None => return,
    };

    if !py_is_first_statement_in_block(node) {
        return;
    }

    if let Ok(text) = string_node.utf8_text(source) {
        let stripped = strip_docstring_quotes(text);
        if !stripped.is_empty() {
            doc_texts.push((
                path.to_path_buf(),
                string_node.start_position().row + 1,
                stripped.to_string(),
            ));
        }
    }
}

fn py_is_first_statement_in_block(node: tree_sitter::Node) -> bool {
    let parent = match node.parent() {
        Some(p) => p,
        None => return false,
    };

    match parent.kind() {
        "module" => parent
            .named_child(0)
            .is_some_and(|first| first.id() == node.id()),
        "block" => {
            let is_body_block = parent.parent().is_some_and(|gp| {
                matches!(
                    gp.kind(),
                    "function_definition" | "class_definition"
                )
            });
            if !is_body_block {
                return false;
            }
            parent
                .named_child(0)
                .is_some_and(|first| first.id() == node.id())
        }
        _ => false,
    }
}

/// Strip surrounding docstring quotes (`"""`, `'''`, `"`, `'`, and r/b
/// prefixes).
fn strip_docstring_quotes(s: &str) -> &str {
    let s = s.trim_start_matches(|c: char| "rRbBuUfF".contains(c));

    if let Some(inner) =
        s.strip_prefix("\"\"\"").and_then(|s| s.strip_suffix("\"\"\""))
    {
        return inner.trim();
    }
    if let Some(inner) =
        s.strip_prefix("'''").and_then(|s| s.strip_suffix("'''"))
    {
        return inner.trim();
    }
    if let Some(inner) =
        s.strip_prefix('"').and_then(|s| s.strip_suffix('"'))
    {
        return inner.trim();
    }
    if let Some(inner) =
        s.strip_prefix('\'').and_then(|s| s.strip_suffix('\''))
    {
        return inner.trim();
    }
    s
}

// --- Python L2 extraction helpers ---

fn py_extract_signature(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
) -> Option<Signature> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?.to_string();
    let line = name_node.start_position().row + 1;

    let mut params = Vec::new();
    if let Some(params_node) = node.child_by_field_name("parameters") {
        let mut cursor = params_node.walk();
        for child in params_node.named_children(&mut cursor) {
            if let Some(param) = py_extract_param(child, source) {
                params.push(param);
            }
        }
    }

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|n| n.utf8_text(source).ok())
        .map(|s| s.to_string());

    let decorators = py_extract_decorator_names_for_sig(node, source);
    let docstring_first_line =
        py_extract_body_docstring_first_line(node, source);

    Some(Signature {
        name,
        params,
        return_type,
        decorators,
        docstring_first_line,
        file: path.to_path_buf(),
        line,
        scope: scope.clone(),
    })
}

fn py_extract_param(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<Param> {
    match node.kind() {
        "identifier" => {
            let name = node.utf8_text(source).ok()?.to_string();
            if PY_SKIP_PARAMS.contains(&name.as_str()) {
                return None;
            }
            Some(Param {
                name,
                type_annotation: None,
                default: None,
            })
        }
        "typed_parameter" => {
            let name = find_first_identifier_child(node)?
                .utf8_text(source)
                .ok()?
                .to_string();
            if PY_SKIP_PARAMS.contains(&name.as_str()) {
                return None;
            }
            let type_ann = node
                .child_by_field_name("type")
                .and_then(|n| n.utf8_text(source).ok())
                .map(|s| s.to_string());
            Some(Param {
                name,
                type_annotation: type_ann,
                default: None,
            })
        }
        "default_parameter" => {
            let name = find_first_identifier_child(node)?
                .utf8_text(source)
                .ok()?
                .to_string();
            if PY_SKIP_PARAMS.contains(&name.as_str()) {
                return None;
            }
            let default = node
                .child_by_field_name("value")
                .and_then(|n| n.utf8_text(source).ok())
                .map(|s| s.to_string());
            Some(Param {
                name,
                type_annotation: None,
                default,
            })
        }
        "typed_default_parameter" => {
            let name = find_first_identifier_child(node)?
                .utf8_text(source)
                .ok()?
                .to_string();
            if PY_SKIP_PARAMS.contains(&name.as_str()) {
                return None;
            }
            let type_ann = node
                .child_by_field_name("type")
                .and_then(|n| n.utf8_text(source).ok())
                .map(|s| s.to_string());
            let default = node
                .child_by_field_name("value")
                .and_then(|n| n.utf8_text(source).ok())
                .map(|s| s.to_string());
            Some(Param {
                name,
                type_annotation: type_ann,
                default,
            })
        }
        _ => None,
    }
}

fn py_extract_decorator_names_for_sig(
    func_node: tree_sitter::Node,
    source: &[u8],
) -> Vec<String> {
    let parent = match func_node.parent() {
        Some(p) if p.kind() == "decorated_definition" => p,
        _ => return Vec::new(),
    };
    let mut decorators = Vec::new();
    let mut cursor = parent.walk();
    for child in parent.children(&mut cursor) {
        if child.kind() == "decorator" {
            if let Some(name) = py_extract_decorator_name(child, source) {
                decorators.push(name);
            }
        }
    }
    decorators
}

fn py_extract_body_docstring_first_line(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<String> {
    let body = node.child_by_field_name("body")?;
    let first_stmt = body.named_child(0)?;
    if first_stmt.kind() != "expression_statement" {
        return None;
    }
    let string_node = find_named_child_by_kind(first_stmt, "string")?;
    let text = string_node.utf8_text(source).ok()?;
    let stripped = strip_docstring_quotes(text);
    if stripped.is_empty() {
        return None;
    }
    Some(stripped.lines().next().unwrap_or("").to_string())
}

fn py_extract_class_info(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
) -> Option<ClassInfo> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?.to_string();
    let line = name_node.start_position().row + 1;

    let mut bases = Vec::new();
    if let Some(args) = node.child_by_field_name("superclasses") {
        let mut cursor = args.walk();
        for child in args.named_children(&mut cursor) {
            if let Ok(text) = child.utf8_text(source) {
                bases.push(text.to_string());
            }
        }
    }

    let mut methods = Vec::new();
    let mut attributes = Vec::new();
    let docstring_first_line =
        py_extract_body_docstring_first_line(node, source);

    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.named_children(&mut cursor) {
            match child.kind() {
                "function_definition" => {
                    if let Some(n) = child.child_by_field_name("name") {
                        if let Ok(method_name) = n.utf8_text(source) {
                            methods.push(method_name.to_string());
                        }
                    }
                }
                "decorated_definition" => {
                    if let Some(func) = find_named_child_by_kind(
                        child,
                        "function_definition",
                    ) {
                        if let Some(n) = func.child_by_field_name("name") {
                            if let Ok(method_name) = n.utf8_text(source) {
                                methods.push(method_name.to_string());
                            }
                        }
                    }
                }
                _ => {}
            }
        }
        py_collect_self_attributes(body, source, &mut attributes);
    }

    Some(ClassInfo {
        name,
        bases,
        methods,
        attributes,
        docstring_first_line,
        file: path.to_path_buf(),
        line,
    })
}

/// Recursively scan a Python class body for `self.X = ...` attributes.
fn py_collect_self_attributes(
    node: tree_sitter::Node,
    source: &[u8],
    attributes: &mut Vec<String>,
) {
    if node.kind() == "assignment" {
        if let Some(left) = node.child_by_field_name("left") {
            if left.kind() == "attribute" {
                if let Some(obj) = left.child_by_field_name("object") {
                    if obj.utf8_text(source).ok() == Some("self") {
                        if let Some(attr) =
                            left.child_by_field_name("attribute")
                        {
                            if let Ok(attr_name) = attr.utf8_text(source) {
                                let attr_str = attr_name.to_string();
                                if !attributes.contains(&attr_str) {
                                    attributes.push(attr_str);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        py_collect_self_attributes(child, source, attributes);
    }
}

// =========================================================================
// TypeScriptParser
// =========================================================================

/// Parser for TypeScript (`.ts`, `.tsx`) files.
///
/// Wired up in Phase 1.3; currently used via tests and the `parse_*_with`
/// API.
#[allow(dead_code)]
pub struct TypeScriptParser;

impl LanguageParser for TypeScriptParser {
    fn extensions(&self) -> &[&str] {
        &["ts", "tsx"]
    }

    fn make_parser(&self) -> Result<tree_sitter::Parser> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(
                &tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            )
            .map_err(|e| {
                anyhow::anyhow!("Error loading TypeScript grammar: {e}")
            })?;
        Ok(parser)
    }

    fn make_parser_for_path(
        &self,
        path: &Path,
    ) -> Result<tree_sitter::Parser> {
        let mut parser = tree_sitter::Parser::new();
        let is_tsx = path.extension().is_some_and(|e| e == "tsx");
        let language = if is_tsx {
            tree_sitter_typescript::LANGUAGE_TSX
        } else {
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT
        };
        parser
            .set_language(&language.into())
            .map_err(|e| {
                anyhow::anyhow!("Error loading TypeScript grammar: {e}")
            })?;
        Ok(parser)
    }

    fn skip_params(&self) -> &[&str] {
        &[]
    }

    fn strip_doc_syntax<'a>(&self, raw: &'a str) -> &'a str {
        strip_jsdoc_borrow(raw)
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_node(
        &self,
        node: tree_sitter::Node,
        source: &[u8],
        path: &Path,
        scope: &Option<String>,
        identifiers: &mut Vec<RawIdentifier>,
        doc_texts: &mut Vec<(PathBuf, usize, String)>,
        signatures: &mut Vec<Signature>,
        classes: &mut Vec<ClassInfo>,
        call_sites: &mut Vec<CallSite>,
    ) {
        match node.kind() {
            "function_declaration" => {
                let func_name = ts_extract_function(
                    node, source, path, scope, identifiers,
                );
                if let Some(sig) =
                    ts_extract_function_signature(node, source, path, scope)
                {
                    signatures.push(sig);
                }
                try_extract_jsdoc(node, source, path, doc_texts);
                let child_scope = func_name.map(|name| match scope {
                    Some(parent) => format!("{parent}.{name}"),
                    None => name,
                });
                visit_children(
                    self, node, source, path, &child_scope,
                    identifiers, doc_texts, signatures, classes,
                    call_sites,
                );
                return;
            }
            "method_definition" => {
                let func_name = ts_extract_method(
                    node, source, path, scope, identifiers,
                );
                if let Some(sig) =
                    ts_extract_method_signature(node, source, path, scope)
                {
                    signatures.push(sig);
                }
                try_extract_jsdoc(node, source, path, doc_texts);
                let child_scope = func_name.map(|name| match scope {
                    Some(parent) => format!("{parent}.{name}"),
                    None => name,
                });
                visit_children(
                    self, node, source, path, &child_scope,
                    identifiers, doc_texts, signatures, classes,
                    call_sites,
                );
                return;
            }
            "class_declaration" => {
                let class_name = ts_extract_class(
                    node, source, path, scope, identifiers,
                );
                if let Some(info) =
                    ts_extract_class_info(node, source, path)
                {
                    classes.push(info);
                }
                try_extract_jsdoc(node, source, path, doc_texts);
                let child_scope = class_name.or_else(|| scope.clone());
                visit_children(
                    self, node, source, path, &child_scope,
                    identifiers, doc_texts, signatures, classes,
                    call_sites,
                );
                return;
            }
            "interface_declaration" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(source) {
                        identifiers.push(RawIdentifier {
                            name: name.to_string(),
                            entity_type: EntityType::Interface,
                            file: path.to_path_buf(),
                            line: name_node.start_position().row + 1,
                            scope: scope.clone(),
                        });
                    }
                }
                try_extract_jsdoc(node, source, path, doc_texts);
            }
            "type_alias_declaration" => {
                if let Some(name_node) = node.child_by_field_name("name") {
                    if let Ok(name) = name_node.utf8_text(source) {
                        identifiers.push(RawIdentifier {
                            name: name.to_string(),
                            entity_type: EntityType::TypeAlias,
                            file: path.to_path_buf(),
                            line: name_node.start_position().row + 1,
                            scope: scope.clone(),
                        });
                    }
                }
                try_extract_jsdoc(node, source, path, doc_texts);
            }
            "variable_declarator" => {
                ts_extract_variable_declarator(
                    node, source, path, scope, identifiers, signatures,
                );
            }
            "public_field_definition" => {
                // Class field: `private threshold: number;`
                if let Some(name_node) = find_named_child_by_kind(
                    node, "property_identifier",
                ) {
                    if let Ok(name) = name_node.utf8_text(source) {
                        identifiers.push(RawIdentifier {
                            name: name.to_string(),
                            entity_type: EntityType::Attribute,
                            file: path.to_path_buf(),
                            line: name_node.start_position().row + 1,
                            scope: scope.clone(),
                        });
                    }
                }
                try_extract_jsdoc(node, source, path, doc_texts);
            }
            "decorator" => {
                if let Some(name) = ts_extract_decorator_name(node, source)
                {
                    identifiers.push(RawIdentifier {
                        name,
                        entity_type: EntityType::Decorator,
                        file: path.to_path_buf(),
                        line: node.start_position().row + 1,
                        scope: scope.clone(),
                    });
                }
            }
            "export_statement" => {
                // Unwrap export: visit inner declaration, don't emit "export"
                visit_children(
                    self, node, source, path, scope, identifiers,
                    doc_texts, signatures, classes, call_sites,
                );
                return;
            }
            "assignment_expression" => {
                // Handle `this.foo = bar` inside methods
                ts_extract_this_attribute_from_assignment(
                    node, source, path, scope, identifiers,
                );
            }
            "call_expression" => {
                if let Some(cs) =
                    extract_call_site(node, source, path, scope)
                {
                    call_sites.push(cs);
                }
            }
            _ => {}
        }

        visit_children(
            self, node, source, path, scope, identifiers, doc_texts,
            signatures, classes, call_sites,
        );
    }
}

// =========================================================================
// JavaScriptParser
// =========================================================================

/// Parser for JavaScript (`.js`, `.jsx`, `.mjs`, `.cjs`) files.
///
/// Wired up in Phase 1.3; currently used via tests and the `parse_*_with`
/// API.
#[allow(dead_code)]
pub struct JavaScriptParser;

impl LanguageParser for JavaScriptParser {
    fn extensions(&self) -> &[&str] {
        &["js", "jsx", "mjs", "cjs"]
    }

    fn make_parser(&self) -> Result<tree_sitter::Parser> {
        let mut parser = tree_sitter::Parser::new();
        parser
            .set_language(&tree_sitter_javascript::LANGUAGE.into())
            .map_err(|e| {
                anyhow::anyhow!("Error loading JavaScript grammar: {e}")
            })?;
        Ok(parser)
    }

    fn skip_params(&self) -> &[&str] {
        &[]
    }

    fn strip_doc_syntax<'a>(&self, raw: &'a str) -> &'a str {
        strip_jsdoc_borrow(raw)
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_node(
        &self,
        node: tree_sitter::Node,
        source: &[u8],
        path: &Path,
        scope: &Option<String>,
        identifiers: &mut Vec<RawIdentifier>,
        doc_texts: &mut Vec<(PathBuf, usize, String)>,
        signatures: &mut Vec<Signature>,
        classes: &mut Vec<ClassInfo>,
        call_sites: &mut Vec<CallSite>,
    ) {
        match node.kind() {
            "function_declaration" => {
                let func_name = js_extract_function(
                    node, source, path, scope, identifiers,
                );
                if let Some(sig) =
                    js_extract_function_signature(node, source, path, scope)
                {
                    signatures.push(sig);
                }
                try_extract_jsdoc(node, source, path, doc_texts);
                let child_scope = func_name.map(|name| match scope {
                    Some(parent) => format!("{parent}.{name}"),
                    None => name,
                });
                visit_children(
                    self, node, source, path, &child_scope,
                    identifiers, doc_texts, signatures, classes,
                    call_sites,
                );
                return;
            }
            "method_definition" => {
                let func_name = js_extract_method(
                    node, source, path, scope, identifiers,
                );
                if let Some(sig) =
                    js_extract_method_signature(node, source, path, scope)
                {
                    signatures.push(sig);
                }
                try_extract_jsdoc(node, source, path, doc_texts);
                let child_scope = func_name.map(|name| match scope {
                    Some(parent) => format!("{parent}.{name}"),
                    None => name,
                });
                visit_children(
                    self, node, source, path, &child_scope,
                    identifiers, doc_texts, signatures, classes,
                    call_sites,
                );
                return;
            }
            "class_declaration" => {
                let class_name = js_extract_class(
                    node, source, path, scope, identifiers,
                );
                if let Some(info) =
                    js_extract_class_info(node, source, path)
                {
                    classes.push(info);
                }
                try_extract_jsdoc(node, source, path, doc_texts);
                let child_scope = class_name.or_else(|| scope.clone());
                visit_children(
                    self, node, source, path, &child_scope,
                    identifiers, doc_texts, signatures, classes,
                    call_sites,
                );
                return;
            }
            "variable_declarator" => {
                js_extract_variable_declarator(
                    node, source, path, scope, identifiers, signatures,
                );
            }
            "export_statement" => {
                visit_children(
                    self, node, source, path, scope, identifiers,
                    doc_texts, signatures, classes, call_sites,
                );
                return;
            }
            "assignment_expression" => {
                ts_extract_this_attribute_from_assignment(
                    node, source, path, scope, identifiers,
                );
            }
            "call_expression" => {
                if let Some(cs) =
                    extract_call_site(node, source, path, scope)
                {
                    call_sites.push(cs);
                }
            }
            _ => {}
        }

        visit_children(
            self, node, source, path, scope, identifiers, doc_texts,
            signatures, classes, call_sites,
        );
    }
}

// =========================================================================
// Shared TS/JS helpers
// =========================================================================

/// Strip JSDoc syntax for borrowed return (simple trim of `/**` / `*/`).
fn strip_jsdoc_borrow(raw: &str) -> &str {
    let s = raw.trim();
    let s = s.strip_prefix("/**").unwrap_or(s);
    let s = s.strip_suffix("*/").unwrap_or(s);
    s.trim()
}

/// Strip JSDoc to a fully-cleaned string (removes `/**`, `*/`, and per-line
/// leading `* `).
fn strip_jsdoc_to_string(raw: &str) -> String {
    let s = raw.trim();
    let s = s.strip_prefix("/**").unwrap_or(s);
    let s = s.strip_suffix("*/").unwrap_or(s);
    s.lines()
        .map(|line| {
            let trimmed = line.trim();
            trimmed
                .strip_prefix("* ")
                .or_else(|| trimmed.strip_prefix('*'))
                .unwrap_or(trimmed)
        })
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

/// Try to extract a JSDoc comment from the previous sibling of a node.
fn try_extract_jsdoc(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    doc_texts: &mut Vec<(PathBuf, usize, String)>,
) {
    let prev = match node.prev_named_sibling() {
        Some(n) if n.kind() == "comment" => n,
        _ => return,
    };
    let text = match prev.utf8_text(source) {
        Ok(t) if t.starts_with("/**") => t,
        _ => return,
    };
    let stripped = strip_jsdoc_to_string(text);
    if !stripped.is_empty() {
        doc_texts.push((
            path.to_path_buf(),
            prev.start_position().row + 1,
            stripped,
        ));
    }
}

/// Extract `this.foo` from an assignment expression (`this.foo = bar`).
fn ts_extract_this_attribute_from_assignment(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    let left = match node.child_by_field_name("left") {
        Some(n) if n.kind() == "member_expression" => n,
        _ => return,
    };
    let obj = match left.child_by_field_name("object") {
        Some(n) => n,
        None => return,
    };
    if obj.utf8_text(source).ok() != Some("this") {
        return;
    }
    if let Some(prop) = left.child_by_field_name("property") {
        if let Ok(name) = prop.utf8_text(source) {
            identifiers.push(RawIdentifier {
                name: name.to_string(),
                entity_type: EntityType::Attribute,
                file: path.to_path_buf(),
                line: prop.start_position().row + 1,
                scope: scope.clone(),
            });
        }
    }
}

/// Extract decorator name from a TS/JS `decorator` node (`@foo` or `@foo()`).
fn ts_extract_decorator_name(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<String> {
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                return child
                    .utf8_text(source)
                    .ok()
                    .map(|s| s.to_string());
            }
            "call_expression" => {
                if let Some(func) = child.child_by_field_name("function") {
                    return func
                        .utf8_text(source)
                        .ok()
                        .map(|s| s.to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract TS param from a `required_parameter` or `optional_parameter` node.
fn ts_extract_param(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<Param> {
    match node.kind() {
        "required_parameter" | "optional_parameter" => {
            // The `pattern` field holds the identifier
            let name_node = node.child_by_field_name("pattern")
                .or_else(|| find_first_identifier_child(node))?;
            let name = name_node.utf8_text(source).ok()?.to_string();
            let type_ann = node
                .child_by_field_name("type")
                .and_then(|n| {
                    // type_annotation wraps the actual type; get the inner type text
                    n.named_child(0)
                        .and_then(|inner| inner.utf8_text(source).ok())
                })
                .map(|s| s.to_string());
            let default = node
                .child_by_field_name("value")
                .and_then(|n| n.utf8_text(source).ok())
                .map(|s| s.to_string());
            Some(Param {
                name,
                type_annotation: type_ann,
                default,
            })
        }
        _ => None,
    }
}

/// Extract JS param from children of `formal_parameters`.
///
/// JS params are either plain `identifier` or `assignment_pattern` (with
/// default). No type annotations.
fn js_extract_param(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<Param> {
    match node.kind() {
        "identifier" => {
            let name = node.utf8_text(source).ok()?.to_string();
            Some(Param {
                name,
                type_annotation: None,
                default: None,
            })
        }
        "assignment_pattern" => {
            let name_node = node.child_by_field_name("left")
                .or_else(|| find_first_identifier_child(node))?;
            let name = name_node.utf8_text(source).ok()?.to_string();
            let default = node
                .child_by_field_name("right")
                .and_then(|n| n.utf8_text(source).ok())
                .map(|s| s.to_string());
            Some(Param {
                name,
                type_annotation: None,
                default,
            })
        }
        _ => None,
    }
}

// =========================================================================
// TypeScript-specific extraction
// =========================================================================

fn ts_extract_function(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) -> Option<String> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?;

    identifiers.push(RawIdentifier {
        name: name.to_string(),
        entity_type: EntityType::Function,
        file: path.to_path_buf(),
        line: name_node.start_position().row + 1,
        scope: scope.clone(),
    });

    // Extract parameters as identifiers
    if let Some(params_node) = node.child_by_field_name("parameters") {
        let func_scope = Some(match scope {
            Some(parent) => format!("{parent}.{name}"),
            None => name.to_string(),
        });
        ts_extract_param_identifiers(
            params_node, source, path, &func_scope, identifiers,
        );
    }

    Some(name.to_string())
}

fn ts_extract_method(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) -> Option<String> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?;

    identifiers.push(RawIdentifier {
        name: name.to_string(),
        entity_type: EntityType::Function,
        file: path.to_path_buf(),
        line: name_node.start_position().row + 1,
        scope: scope.clone(),
    });

    if let Some(params_node) = node.child_by_field_name("parameters") {
        let func_scope = Some(match scope {
            Some(parent) => format!("{parent}.{name}"),
            None => name.to_string(),
        });
        ts_extract_param_identifiers(
            params_node, source, path, &func_scope, identifiers,
        );
    }

    Some(name.to_string())
}

/// Extract parameter names as `RawIdentifier`s from TS `formal_parameters`.
fn ts_extract_param_identifiers(
    params_node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    let mut cursor = params_node.walk();
    for child in params_node.named_children(&mut cursor) {
        match child.kind() {
            "required_parameter" | "optional_parameter" => {
                let name_node = child
                    .child_by_field_name("pattern")
                    .or_else(|| find_first_identifier_child(child));
                if let Some(n) = name_node {
                    if let Ok(name) = n.utf8_text(source) {
                        identifiers.push(RawIdentifier {
                            name: name.to_string(),
                            entity_type: EntityType::Parameter,
                            file: path.to_path_buf(),
                            line: n.start_position().row + 1,
                            scope: scope.clone(),
                        });
                    }
                }
                // Extract type annotation
                if let Some(type_node) = child.child_by_field_name("type")
                {
                    if let Some(inner) = type_node.named_child(0) {
                        extract_type_annotation(
                            inner, source, path, scope, identifiers,
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

fn ts_extract_class(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) -> Option<String> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?;

    identifiers.push(RawIdentifier {
        name: name.to_string(),
        entity_type: EntityType::Class,
        file: path.to_path_buf(),
        line: name_node.start_position().row + 1,
        scope: scope.clone(),
    });

    Some(match scope {
        Some(parent) => format!("{parent}.{name}"),
        None => name.to_string(),
    })
}

fn ts_extract_class_info(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
) -> Option<ClassInfo> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?.to_string();
    let line = name_node.start_position().row + 1;

    // Base classes from heritage clauses (extends / implements)
    let mut bases = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "class_heritage" {
            let mut inner_cursor = child.walk();
            for inner in child.named_children(&mut inner_cursor) {
                // Skip the "extends" / "implements" keywords
                if inner.kind() == "extends_clause"
                    || inner.kind() == "implements_clause"
                {
                    let mut type_cursor = inner.walk();
                    for type_child in inner.named_children(&mut type_cursor)
                    {
                        if let Ok(text) = type_child.utf8_text(source) {
                            let text = text.trim();
                            if !text.is_empty()
                                && text != "extends"
                                && text != "implements"
                            {
                                bases.push(text.to_string());
                            }
                        }
                    }
                } else if let Ok(text) = inner.utf8_text(source) {
                    let text = text.trim();
                    if !text.is_empty()
                        && text != "extends"
                        && text != "implements"
                    {
                        bases.push(text.to_string());
                    }
                }
            }
        }
    }

    let mut methods = Vec::new();
    let mut attributes = Vec::new();

    if let Some(body) = node.child_by_field_name("body") {
        let mut body_cursor = body.walk();
        for child in body.named_children(&mut body_cursor) {
            match child.kind() {
                "method_definition" => {
                    if let Some(n) = child.child_by_field_name("name") {
                        if let Ok(method_name) = n.utf8_text(source) {
                            methods.push(method_name.to_string());
                        }
                    }
                }
                "public_field_definition" => {
                    if let Some(n) = find_named_child_by_kind(
                        child,
                        "property_identifier",
                    ) {
                        if let Ok(attr_name) = n.utf8_text(source) {
                            attributes.push(attr_name.to_string());
                        }
                    }
                }
                _ => {}
            }
        }
        // Collect this.X attributes from the class body
        ts_collect_this_attributes(body, source, &mut attributes);
    }

    Some(ClassInfo {
        name,
        bases,
        methods,
        attributes,
        docstring_first_line: None,
        file: path.to_path_buf(),
        line,
    })
}

/// Recursively scan for `this.X = ...` assignments in a TS/JS class body.
fn ts_collect_this_attributes(
    node: tree_sitter::Node,
    source: &[u8],
    attributes: &mut Vec<String>,
) {
    if node.kind() == "assignment_expression" {
        if let Some(left) = node.child_by_field_name("left") {
            if left.kind() == "member_expression" {
                if let Some(obj) = left.child_by_field_name("object") {
                    if obj.utf8_text(source).ok() == Some("this") {
                        if let Some(prop) =
                            left.child_by_field_name("property")
                        {
                            if let Ok(attr_name) = prop.utf8_text(source) {
                                let attr_str = attr_name.to_string();
                                if !attributes.contains(&attr_str) {
                                    attributes.push(attr_str);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        ts_collect_this_attributes(child, source, attributes);
    }
}

fn ts_extract_variable_declarator(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
    signatures: &mut Vec<Signature>,
) {
    let name_node = match node.child_by_field_name("name") {
        Some(n) => n,
        None => return,
    };
    let name = match name_node.utf8_text(source) {
        Ok(n) => n.to_string(),
        Err(_) => return,
    };

    let value_node = node.child_by_field_name("value");
    let is_arrow =
        value_node.as_ref().is_some_and(|v| v.kind() == "arrow_function");

    if is_arrow {
        identifiers.push(RawIdentifier {
            name: name.clone(),
            entity_type: EntityType::Function,
            file: path.to_path_buf(),
            line: name_node.start_position().row + 1,
            scope: scope.clone(),
        });
        if let Some(arrow) = value_node {
            // Extract arrow params as RawIdentifiers
            let func_scope = Some(match scope {
                Some(parent) => format!("{parent}.{name}"),
                None => name.clone(),
            });
            if let Some(params_node) =
                arrow.child_by_field_name("parameters")
            {
                ts_extract_param_identifiers(
                    params_node, source, path, &func_scope, identifiers,
                );
            }
            if let Some(sig) = ts_extract_arrow_signature(
                &name, arrow, source, path, scope,
            ) {
                signatures.push(sig);
            }
        }
    } else {
        identifiers.push(RawIdentifier {
            name,
            entity_type: EntityType::Variable,
            file: path.to_path_buf(),
            line: name_node.start_position().row + 1,
            scope: scope.clone(),
        });
    }
}

fn ts_extract_function_signature(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
) -> Option<Signature> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?.to_string();
    let line = name_node.start_position().row + 1;

    let mut params = Vec::new();
    if let Some(params_node) = node.child_by_field_name("parameters") {
        let mut cursor = params_node.walk();
        for child in params_node.named_children(&mut cursor) {
            if let Some(param) = ts_extract_param(child, source) {
                params.push(param);
            }
        }
    }

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|n| n.named_child(0))
        .and_then(|n| n.utf8_text(source).ok())
        .map(|s| s.to_string());

    // Check for JSDoc docstring in previous sibling
    let docstring_first_line = node
        .prev_named_sibling()
        .filter(|n| n.kind() == "comment")
        .and_then(|n| n.utf8_text(source).ok())
        .filter(|t| t.starts_with("/**"))
        .map(strip_jsdoc_to_string)
        .and_then(|s| {
            s.lines().next().map(|l| l.to_string())
        })
        .filter(|s| !s.is_empty());

    Some(Signature {
        name,
        params,
        return_type,
        decorators: Vec::new(),
        docstring_first_line,
        file: path.to_path_buf(),
        line,
        scope: scope.clone(),
    })
}

fn ts_extract_method_signature(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
) -> Option<Signature> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?.to_string();
    let line = name_node.start_position().row + 1;

    let mut params = Vec::new();
    if let Some(params_node) = node.child_by_field_name("parameters") {
        let mut cursor = params_node.walk();
        for child in params_node.named_children(&mut cursor) {
            if let Some(param) = ts_extract_param(child, source) {
                params.push(param);
            }
        }
    }

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|n| n.named_child(0))
        .and_then(|n| n.utf8_text(source).ok())
        .map(|s| s.to_string());

    // Check for decorators on the method (parent or previous sibling)
    let mut decorators = Vec::new();
    if let Some(prev) = node.prev_named_sibling() {
        if prev.kind() == "decorator" {
            if let Some(dec_name) = ts_extract_decorator_name(prev, source)
            {
                decorators.push(dec_name);
            }
        }
    }

    let docstring_first_line = node
        .prev_named_sibling()
        .filter(|n| n.kind() == "comment")
        .and_then(|n| n.utf8_text(source).ok())
        .filter(|t| t.starts_with("/**"))
        .map(strip_jsdoc_to_string)
        .and_then(|s| s.lines().next().map(|l| l.to_string()))
        .filter(|s| !s.is_empty());

    Some(Signature {
        name,
        params,
        return_type,
        decorators,
        docstring_first_line,
        file: path.to_path_buf(),
        line,
        scope: scope.clone(),
    })
}

fn ts_extract_arrow_signature(
    name: &str,
    arrow_node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
) -> Option<Signature> {
    let line = arrow_node.start_position().row + 1;

    let mut params = Vec::new();
    if let Some(params_node) = arrow_node.child_by_field_name("parameters")
    {
        let mut cursor = params_node.walk();
        for child in params_node.named_children(&mut cursor) {
            if let Some(param) = ts_extract_param(child, source) {
                params.push(param);
            }
        }
    }

    let return_type = arrow_node
        .child_by_field_name("return_type")
        .and_then(|n| n.named_child(0))
        .and_then(|n| n.utf8_text(source).ok())
        .map(|s| s.to_string());

    Some(Signature {
        name: name.to_string(),
        params,
        return_type,
        decorators: Vec::new(),
        docstring_first_line: None,
        file: path.to_path_buf(),
        line,
        scope: scope.clone(),
    })
}

// =========================================================================
// JavaScript-specific extraction
// =========================================================================

fn js_extract_function(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) -> Option<String> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?;

    identifiers.push(RawIdentifier {
        name: name.to_string(),
        entity_type: EntityType::Function,
        file: path.to_path_buf(),
        line: name_node.start_position().row + 1,
        scope: scope.clone(),
    });

    if let Some(params_node) = node.child_by_field_name("parameters") {
        let func_scope = Some(match scope {
            Some(parent) => format!("{parent}.{name}"),
            None => name.to_string(),
        });
        js_extract_param_identifiers(
            params_node, source, path, &func_scope, identifiers,
        );
    }

    Some(name.to_string())
}

fn js_extract_method(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) -> Option<String> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?;

    identifiers.push(RawIdentifier {
        name: name.to_string(),
        entity_type: EntityType::Function,
        file: path.to_path_buf(),
        line: name_node.start_position().row + 1,
        scope: scope.clone(),
    });

    if let Some(params_node) = node.child_by_field_name("parameters") {
        let func_scope = Some(match scope {
            Some(parent) => format!("{parent}.{name}"),
            None => name.to_string(),
        });
        js_extract_param_identifiers(
            params_node, source, path, &func_scope, identifiers,
        );
    }

    Some(name.to_string())
}

fn js_extract_class(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) -> Option<String> {
    // In JS, the class name is an `identifier` (not `type_identifier`)
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?;

    identifiers.push(RawIdentifier {
        name: name.to_string(),
        entity_type: EntityType::Class,
        file: path.to_path_buf(),
        line: name_node.start_position().row + 1,
        scope: scope.clone(),
    });

    Some(match scope {
        Some(parent) => format!("{parent}.{name}"),
        None => name.to_string(),
    })
}

/// Extract parameter names as `RawIdentifier`s from JS `formal_parameters`.
fn js_extract_param_identifiers(
    params_node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    let mut cursor = params_node.walk();
    for child in params_node.named_children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                if let Ok(name) = child.utf8_text(source) {
                    identifiers.push(RawIdentifier {
                        name: name.to_string(),
                        entity_type: EntityType::Parameter,
                        file: path.to_path_buf(),
                        line: child.start_position().row + 1,
                        scope: scope.clone(),
                    });
                }
            }
            "assignment_pattern" => {
                if let Some(left) = child.child_by_field_name("left") {
                    if let Ok(name) = left.utf8_text(source) {
                        identifiers.push(RawIdentifier {
                            name: name.to_string(),
                            entity_type: EntityType::Parameter,
                            file: path.to_path_buf(),
                            line: left.start_position().row + 1,
                            scope: scope.clone(),
                        });
                    }
                }
            }
            _ => {}
        }
    }
}

fn js_extract_variable_declarator(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
    signatures: &mut Vec<Signature>,
) {
    let name_node = match node.child_by_field_name("name") {
        Some(n) => n,
        None => return,
    };
    let name = match name_node.utf8_text(source) {
        Ok(n) => n.to_string(),
        Err(_) => return,
    };

    let value_node = node.child_by_field_name("value");
    let is_arrow =
        value_node.as_ref().is_some_and(|v| v.kind() == "arrow_function");

    if is_arrow {
        identifiers.push(RawIdentifier {
            name: name.clone(),
            entity_type: EntityType::Function,
            file: path.to_path_buf(),
            line: name_node.start_position().row + 1,
            scope: scope.clone(),
        });
        if let Some(arrow) = value_node {
            // Extract arrow params as RawIdentifiers
            let func_scope = Some(match scope {
                Some(parent) => format!("{parent}.{name}"),
                None => name.clone(),
            });
            if let Some(params_node) =
                arrow.child_by_field_name("parameters")
            {
                js_extract_param_identifiers(
                    params_node, source, path, &func_scope, identifiers,
                );
            }
            if let Some(sig) = js_extract_arrow_signature(
                &name, arrow, source, path, scope,
            ) {
                signatures.push(sig);
            }
        }
    } else {
        identifiers.push(RawIdentifier {
            name,
            entity_type: EntityType::Variable,
            file: path.to_path_buf(),
            line: name_node.start_position().row + 1,
            scope: scope.clone(),
        });
    }
}

fn js_extract_class_info(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
) -> Option<ClassInfo> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?.to_string();
    let line = name_node.start_position().row + 1;

    // Base classes from heritage (extends only in JS, no implements)
    let mut bases = Vec::new();
    let mut cursor = node.walk();
    for child in node.named_children(&mut cursor) {
        if child.kind() == "class_heritage" {
            let mut inner_cursor = child.walk();
            for inner in child.named_children(&mut inner_cursor) {
                if let Ok(text) = inner.utf8_text(source) {
                    let text = text.trim();
                    if !text.is_empty() && text != "extends" {
                        bases.push(text.to_string());
                    }
                }
            }
        }
    }

    let mut methods = Vec::new();
    let mut attributes = Vec::new();

    if let Some(body) = node.child_by_field_name("body") {
        let mut body_cursor = body.walk();
        for child in body.named_children(&mut body_cursor) {
            if child.kind() == "method_definition" {
                if let Some(n) = child.child_by_field_name("name") {
                    if let Ok(method_name) = n.utf8_text(source) {
                        methods.push(method_name.to_string());
                    }
                }
            }
        }
        ts_collect_this_attributes(body, source, &mut attributes);
    }

    Some(ClassInfo {
        name,
        bases,
        methods,
        attributes,
        docstring_first_line: None,
        file: path.to_path_buf(),
        line,
    })
}

fn js_extract_function_signature(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
) -> Option<Signature> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?.to_string();
    let line = name_node.start_position().row + 1;

    let mut params = Vec::new();
    if let Some(params_node) = node.child_by_field_name("parameters") {
        let mut cursor = params_node.walk();
        for child in params_node.named_children(&mut cursor) {
            if let Some(param) = js_extract_param(child, source) {
                params.push(param);
            }
        }
    }

    let docstring_first_line = node
        .prev_named_sibling()
        .filter(|n| n.kind() == "comment")
        .and_then(|n| n.utf8_text(source).ok())
        .filter(|t| t.starts_with("/**"))
        .map(strip_jsdoc_to_string)
        .and_then(|s| s.lines().next().map(|l| l.to_string()))
        .filter(|s| !s.is_empty());

    Some(Signature {
        name,
        params,
        return_type: None,
        decorators: Vec::new(),
        docstring_first_line,
        file: path.to_path_buf(),
        line,
        scope: scope.clone(),
    })
}

fn js_extract_method_signature(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
) -> Option<Signature> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?.to_string();
    let line = name_node.start_position().row + 1;

    let mut params = Vec::new();
    if let Some(params_node) = node.child_by_field_name("parameters") {
        let mut cursor = params_node.walk();
        for child in params_node.named_children(&mut cursor) {
            if let Some(param) = js_extract_param(child, source) {
                params.push(param);
            }
        }
    }

    let docstring_first_line = node
        .prev_named_sibling()
        .filter(|n| n.kind() == "comment")
        .and_then(|n| n.utf8_text(source).ok())
        .filter(|t| t.starts_with("/**"))
        .map(strip_jsdoc_to_string)
        .and_then(|s| s.lines().next().map(|l| l.to_string()))
        .filter(|s| !s.is_empty());

    Some(Signature {
        name,
        params,
        return_type: None,
        decorators: Vec::new(),
        docstring_first_line,
        file: path.to_path_buf(),
        line,
        scope: scope.clone(),
    })
}

fn js_extract_arrow_signature(
    name: &str,
    arrow_node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
) -> Option<Signature> {
    let line = arrow_node.start_position().row + 1;

    let mut params = Vec::new();
    if let Some(params_node) = arrow_node.child_by_field_name("parameters")
    {
        let mut cursor = params_node.walk();
        for child in params_node.named_children(&mut cursor) {
            if let Some(param) = js_extract_param(child, source) {
                params.push(param);
            }
        }
    }

    Some(Signature {
        name: name.to_string(),
        params,
        return_type: None,
        decorators: Vec::new(),
        docstring_first_line: None,
        file: path.to_path_buf(),
        line,
        scope: scope.clone(),
    })
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::EntityType;
    use std::sync::OnceLock;

    const FIXTURE: &str = r#""""Module docstring."""

class MyClass:
    """Class docstring."""
    class_var = 42

    def __init__(self, nb_features: int, is_valid: bool = True):
        self.nb_features = nb_features
        self.is_valid = is_valid

    @staticmethod
    def compute_transform(source, target):
        displacement = source - target
        return displacement

def spatial_transform(vol: Tensor, trf, nb_dims=3):
    """Apply spatial transform."""
    ndim: int = len(vol.shape)
    return ndim
"#;

    fn fixture_result() -> &'static ParseResult {
        static RESULT: OnceLock<ParseResult> = OnceLock::new();
        RESULT.get_or_init(|| {
            let path = Path::new("/tmp/ontomics_test_parser.py");
            std::fs::write(path, FIXTURE).unwrap();
            parse_file(path).unwrap()
        })
    }

    fn names_of(
        result: &ParseResult,
        entity_type: EntityType,
    ) -> Vec<String> {
        result
            .identifiers
            .iter()
            .filter(|id| id.entity_type == entity_type)
            .map(|id| id.name.clone())
            .collect()
    }

    #[test]
    fn test_functions_found() {
        let result = fixture_result();
        let functions = names_of(result, EntityType::Function);
        assert!(functions.contains(&"__init__".to_string()));
        assert!(functions.contains(&"compute_transform".to_string()));
        assert!(functions.contains(&"spatial_transform".to_string()));
    }

    #[test]
    fn test_parameters_found() {
        let result = fixture_result();
        let params = names_of(result, EntityType::Parameter);
        for expected in &[
            "nb_features",
            "is_valid",
            "source",
            "target",
            "vol",
            "trf",
            "nb_dims",
        ] {
            assert!(
                params.contains(&expected.to_string()),
                "Missing parameter: {expected}"
            );
        }
        assert!(
            !params.contains(&"self".to_string()),
            "self should be excluded"
        );
    }

    #[test]
    fn test_class_found() {
        let result = fixture_result();
        let classes = names_of(result, EntityType::Class);
        assert!(classes.contains(&"MyClass".to_string()));
    }

    #[test]
    fn test_variables_found() {
        let result = fixture_result();
        let vars = names_of(result, EntityType::Variable);
        assert!(
            vars.contains(&"class_var".to_string()),
            "Missing variable: class_var"
        );
        assert!(
            vars.contains(&"displacement".to_string()),
            "Missing variable: displacement"
        );
        assert!(
            vars.contains(&"ndim".to_string()),
            "Missing variable: ndim"
        );
    }

    #[test]
    fn test_docstrings_captured() {
        let result = fixture_result();
        assert!(
            !result.doc_texts.is_empty(),
            "Expected at least one docstring"
        );
        let texts: Vec<&str> = result
            .doc_texts
            .iter()
            .map(|(_, _, t)| t.as_str())
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("Module docstring")),
            "Missing module docstring"
        );
        assert!(
            texts.iter().any(|t| t.contains("Class docstring")),
            "Missing class docstring"
        );
        assert!(
            texts.iter().any(|t| t.contains("Apply spatial transform")),
            "Missing function docstring"
        );
    }

    #[test]
    fn test_attributes_found() {
        let result = fixture_result();
        let attrs = names_of(result, EntityType::Attribute);
        assert!(
            attrs.contains(&"nb_features".to_string()),
            "Missing attribute: nb_features"
        );
        assert!(
            attrs.contains(&"is_valid".to_string()),
            "Missing attribute: is_valid"
        );
    }

    #[test]
    fn test_type_annotations_found() {
        let result = fixture_result();
        let types = names_of(result, EntityType::TypeAnnotation);
        assert!(
            types.contains(&"int".to_string()),
            "Missing type annotation: int"
        );
        assert!(
            types.contains(&"bool".to_string()),
            "Missing type annotation: bool"
        );
        assert!(
            types.contains(&"Tensor".to_string()),
            "Missing type annotation: Tensor"
        );
    }

    #[test]
    fn test_decorator_found() {
        let result = fixture_result();
        let decorators = names_of(result, EntityType::Decorator);
        assert!(
            decorators.contains(&"staticmethod".to_string()),
            "Missing decorator: staticmethod"
        );
    }

    #[test]
    fn test_line_numbers_are_1_indexed() {
        let result = fixture_result();
        for id in &result.identifiers {
            assert!(id.line >= 1, "Line numbers must be 1-indexed");
        }
        for (_, line, _) in &result.doc_texts {
            assert!(*line >= 1, "Docstring line numbers must be 1-indexed");
        }
    }

    #[test]
    fn test_scope_tracking() {
        let result = fixture_result();

        let spatial = result
            .identifiers
            .iter()
            .find(|id| id.name == "spatial_transform")
            .unwrap();
        assert_eq!(spatial.scope, None);

        let class = result
            .identifiers
            .iter()
            .find(|id| id.name == "MyClass")
            .unwrap();
        assert_eq!(class.scope, None);

        let init = result
            .identifiers
            .iter()
            .find(|id| {
                id.name == "__init__"
                    && id.entity_type == EntityType::Function
            })
            .unwrap();
        assert_eq!(init.scope, Some("MyClass".to_string()));

        let nb_feat = result
            .identifiers
            .iter()
            .find(|id| {
                id.name == "nb_features"
                    && id.entity_type == EntityType::Parameter
            })
            .unwrap();
        assert_eq!(nb_feat.scope, Some("MyClass.__init__".to_string()));

        let disp = result
            .identifiers
            .iter()
            .find(|id| id.name == "displacement")
            .unwrap();
        assert_eq!(
            disp.scope,
            Some("MyClass.compute_transform".to_string())
        );

        let ndim = result
            .identifiers
            .iter()
            .find(|id| id.name == "ndim")
            .unwrap();
        assert_eq!(ndim.scope, Some("spatial_transform".to_string()));

        let vol = result
            .identifiers
            .iter()
            .find(|id| {
                id.name == "vol"
                    && id.entity_type == EntityType::Parameter
            })
            .unwrap();
        assert_eq!(vol.scope, Some("spatial_transform".to_string()));
    }

    // --- L2 extraction tests ---

    #[test]
    fn test_signatures_extracted() {
        let result = fixture_result();
        assert!(
            !result.signatures.is_empty(),
            "Expected at least one signature"
        );
        let spatial = result
            .signatures
            .iter()
            .find(|s| s.name == "spatial_transform")
            .expect("Missing signature: spatial_transform");
        assert_eq!(spatial.params.len(), 3);
        assert_eq!(spatial.params[0].name, "vol");
        assert_eq!(
            spatial.params[0].type_annotation,
            Some("Tensor".to_string())
        );
        assert_eq!(spatial.params[1].name, "trf");
        assert!(spatial.params[1].type_annotation.is_none());
        assert_eq!(spatial.params[2].name, "nb_dims");
        assert_eq!(spatial.params[2].default, Some("3".to_string()));
        assert_eq!(
            spatial.docstring_first_line,
            Some("Apply spatial transform.".to_string())
        );
    }

    #[test]
    fn test_signatures_include_decorators() {
        let result = fixture_result();
        let compute = result
            .signatures
            .iter()
            .find(|s| s.name == "compute_transform")
            .expect("Missing signature: compute_transform");
        assert!(
            compute
                .decorators
                .contains(&"staticmethod".to_string()),
            "Missing decorator: staticmethod"
        );
    }

    #[test]
    fn test_class_info_extracted() {
        let result = fixture_result();
        assert!(
            !result.classes.is_empty(),
            "Expected at least one class"
        );
        let cls = result
            .classes
            .iter()
            .find(|c| c.name == "MyClass")
            .expect("Missing class: MyClass");
        assert!(cls.methods.contains(&"__init__".to_string()));
        assert!(cls.methods.contains(&"compute_transform".to_string()));
        assert!(cls.attributes.contains(&"nb_features".to_string()));
        assert!(cls.attributes.contains(&"is_valid".to_string()));
        assert_eq!(
            cls.docstring_first_line,
            Some("Class docstring.".to_string())
        );
    }

    #[test]
    fn test_call_sites_extracted() {
        let source = r#"
def foo():
    bar()
    self.method()
    obj.other()
"#;
        let path = Path::new("test_calls.py");
        let result = parse_content(source, path).unwrap();
        assert!(
            !result.call_sites.is_empty(),
            "Expected call sites"
        );
        let callees: Vec<&str> = result
            .call_sites
            .iter()
            .map(|c| c.callee.as_str())
            .collect();
        assert!(callees.contains(&"bar"), "Missing callee: bar");
        assert!(
            callees.contains(&"self.method"),
            "Missing callee: self.method"
        );
        assert!(
            callees.contains(&"obj.other"),
            "Missing callee: obj.other"
        );
    }

    #[test]
    fn test_call_sites_have_scope() {
        let source = r#"
def outer():
    inner()

class Cls:
    def method(self):
        helper()
"#;
        let path = Path::new("test_scope_calls.py");
        let result = parse_content(source, path).unwrap();
        let inner_call = result
            .call_sites
            .iter()
            .find(|c| c.callee == "inner")
            .expect("Missing call site: inner");
        assert_eq!(
            inner_call.caller_scope,
            Some("outer".to_string())
        );
        let helper_call = result
            .call_sites
            .iter()
            .find(|c| c.callee == "helper")
            .expect("Missing call site: helper");
        assert_eq!(
            helper_call.caller_scope,
            Some("Cls.method".to_string())
        );
    }
}

// =========================================================================
// TypeScript tests
// =========================================================================

#[cfg(test)]
mod ts_tests {
    use super::*;
    use crate::types::EntityType;
    use std::sync::OnceLock;

    const TS_FIXTURE: &str = r#"/** Module-level JSDoc comment */

interface PatientRecord {
  name: string;
  age: number;
}

type SegmentationMask = Float32Array;

export class ImageProcessor {
  /** Process medical images */
  private threshold: number;

  constructor(threshold: number = 0.5) {
    this.threshold = threshold;
  }

  @deprecated
  processImage(source: ImageData, target: ImageData): SegmentationMask {
    const displacement = this.computeDisplacement(source, target);
    return displacement;
  }
}

export const useSegmentation = (nbFeatures: number): SegmentationMask => {
  const isValid = nbFeatures > 0;
  return new Float32Array(nbFeatures);
};

function spatialTransform(vol: Tensor, trf: Transform, nbDims: number = 3): Tensor {
  /** Apply spatial transform */
  const ndim = vol.shape.length;
  return vol;
}
"#;

    fn ts_fixture_result() -> &'static ParseResult {
        static RESULT: OnceLock<ParseResult> = OnceLock::new();
        RESULT.get_or_init(|| {
            let parser = TypeScriptParser;
            parse_content_with(
                TS_FIXTURE,
                Path::new("test.ts"),
                &parser,
            )
            .unwrap()
        })
    }

    fn names_of(
        result: &ParseResult,
        entity_type: EntityType,
    ) -> Vec<String> {
        result
            .identifiers
            .iter()
            .filter(|id| id.entity_type == entity_type)
            .map(|id| id.name.clone())
            .collect()
    }

    #[test]
    fn test_ts_functions_found() {
        let result = ts_fixture_result();
        let functions = names_of(result, EntityType::Function);
        assert!(
            functions.contains(&"spatialTransform".to_string()),
            "Missing: spatialTransform"
        );
        assert!(
            functions.contains(&"useSegmentation".to_string()),
            "Missing arrow fn: useSegmentation"
        );
        assert!(
            functions.contains(&"processImage".to_string()),
            "Missing method: processImage"
        );
        assert!(
            functions.contains(&"constructor".to_string()),
            "Missing: constructor"
        );
    }

    #[test]
    fn test_ts_interfaces_found() {
        let result = ts_fixture_result();
        let interfaces = names_of(result, EntityType::Interface);
        assert!(
            interfaces.contains(&"PatientRecord".to_string()),
            "Missing interface: PatientRecord"
        );
    }

    #[test]
    fn test_ts_type_aliases_found() {
        let result = ts_fixture_result();
        let aliases = names_of(result, EntityType::TypeAlias);
        assert!(
            aliases.contains(&"SegmentationMask".to_string()),
            "Missing type alias: SegmentationMask"
        );
    }

    #[test]
    fn test_ts_classes_found() {
        let result = ts_fixture_result();
        let classes = names_of(result, EntityType::Class);
        assert!(
            classes.contains(&"ImageProcessor".to_string()),
            "Missing class: ImageProcessor"
        );
    }

    #[test]
    fn test_ts_variables_found() {
        let result = ts_fixture_result();
        let vars = names_of(result, EntityType::Variable);
        assert!(
            vars.contains(&"displacement".to_string()),
            "Missing variable: displacement"
        );
        assert!(
            vars.contains(&"isValid".to_string()),
            "Missing variable: isValid"
        );
        assert!(
            vars.contains(&"ndim".to_string()),
            "Missing variable: ndim"
        );
    }

    #[test]
    fn test_ts_arrow_function_named() {
        let result = ts_fixture_result();
        let functions = names_of(result, EntityType::Function);
        assert!(
            functions.contains(&"useSegmentation".to_string()),
            "Arrow function should be named useSegmentation"
        );
        // Should NOT appear as a variable
        let vars = names_of(result, EntityType::Variable);
        assert!(
            !vars.contains(&"useSegmentation".to_string()),
            "Arrow function should not be a Variable"
        );
    }

    #[test]
    fn test_ts_jsdoc_extracted() {
        let result = ts_fixture_result();
        assert!(
            !result.doc_texts.is_empty(),
            "Expected at least one JSDoc comment"
        );
        let texts: Vec<&str> = result
            .doc_texts
            .iter()
            .map(|(_, _, t)| t.as_str())
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("Module-level JSDoc")),
            "Missing module JSDoc. Got: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("Process medical images")),
            "Missing class JSDoc. Got: {texts:?}"
        );
    }

    #[test]
    fn test_ts_class_info() {
        let result = ts_fixture_result();
        let cls = result
            .classes
            .iter()
            .find(|c| c.name == "ImageProcessor")
            .expect("Missing class info: ImageProcessor");
        assert!(
            cls.methods.contains(&"processImage".to_string()),
            "Missing method: processImage"
        );
        assert!(
            cls.methods.contains(&"constructor".to_string()),
            "Missing method: constructor"
        );
        assert!(
            cls.attributes.contains(&"threshold".to_string()),
            "Missing attribute: threshold. Got: {:?}",
            cls.attributes
        );
    }

    #[test]
    fn test_ts_export_unwrapped() {
        let result = ts_fixture_result();
        let functions = names_of(result, EntityType::Function);
        let classes = names_of(result, EntityType::Class);
        assert!(
            classes.contains(&"ImageProcessor".to_string()),
            "Exported class should be extracted"
        );
        assert!(
            functions.contains(&"useSegmentation".to_string()),
            "Exported arrow fn should be extracted"
        );
    }

    #[test]
    fn test_ts_signatures_extracted() {
        let result = ts_fixture_result();
        assert!(
            !result.signatures.is_empty(),
            "Expected at least one signature"
        );
        let spatial = result
            .signatures
            .iter()
            .find(|s| s.name == "spatialTransform")
            .expect("Missing signature: spatialTransform");
        assert_eq!(spatial.params.len(), 3);
        assert_eq!(spatial.params[0].name, "vol");
        assert_eq!(
            spatial.params[0].type_annotation,
            Some("Tensor".to_string())
        );
        assert_eq!(spatial.params[1].name, "trf");
        assert_eq!(
            spatial.params[1].type_annotation,
            Some("Transform".to_string())
        );
        assert_eq!(spatial.params[2].name, "nbDims");
        assert_eq!(
            spatial.params[2].type_annotation,
            Some("number".to_string())
        );
        assert_eq!(spatial.params[2].default, Some("3".to_string()));
        assert_eq!(
            spatial.return_type,
            Some("Tensor".to_string())
        );
    }

    #[test]
    fn test_ts_arrow_signature() {
        let result = ts_fixture_result();
        let seg = result
            .signatures
            .iter()
            .find(|s| s.name == "useSegmentation")
            .expect("Missing signature: useSegmentation");
        assert_eq!(seg.params.len(), 1);
        assert_eq!(seg.params[0].name, "nbFeatures");
        assert_eq!(
            seg.params[0].type_annotation,
            Some("number".to_string())
        );
        assert_eq!(
            seg.return_type,
            Some("SegmentationMask".to_string())
        );
    }

    #[test]
    fn test_ts_decorator_found() {
        let result = ts_fixture_result();
        let decorators = names_of(result, EntityType::Decorator);
        assert!(
            decorators.contains(&"deprecated".to_string()),
            "Missing decorator: deprecated. Got: {decorators:?}"
        );
    }

    #[test]
    fn test_ts_parameters_found() {
        let result = ts_fixture_result();
        let params = names_of(result, EntityType::Parameter);
        assert!(
            params.contains(&"vol".to_string()),
            "Missing param: vol. Got: {params:?}"
        );
        assert!(
            params.contains(&"trf".to_string()),
            "Missing param: trf"
        );
        assert!(
            params.contains(&"nbDims".to_string()),
            "Missing param: nbDims"
        );
        assert!(
            params.contains(&"nbFeatures".to_string()),
            "Missing param: nbFeatures"
        );
    }

    #[test]
    fn test_ts_call_sites() {
        let result = ts_fixture_result();
        let callees: Vec<&str> = result
            .call_sites
            .iter()
            .map(|c| c.callee.as_str())
            .collect();
        assert!(
            callees.iter().any(|c| c.contains("computeDisplacement")),
            "Missing call: computeDisplacement. Got: {callees:?}"
        );
    }

    #[test]
    fn test_ts_scope_tracking() {
        let result = ts_fixture_result();

        // spatialTransform is at module scope
        let spatial = result
            .identifiers
            .iter()
            .find(|id| id.name == "spatialTransform")
            .unwrap();
        assert_eq!(spatial.scope, None);

        // processImage is scoped to ImageProcessor
        let process = result
            .identifiers
            .iter()
            .find(|id| {
                id.name == "processImage"
                    && id.entity_type == EntityType::Function
            })
            .unwrap();
        assert_eq!(
            process.scope,
            Some("ImageProcessor".to_string())
        );

        // displacement is scoped to ImageProcessor.processImage
        let disp = result
            .identifiers
            .iter()
            .find(|id| id.name == "displacement")
            .unwrap();
        assert_eq!(
            disp.scope,
            Some("ImageProcessor.processImage".to_string())
        );
    }

    #[test]
    fn test_ts_type_annotations_extracted() {
        let result = ts_fixture_result();
        let types = names_of(result, EntityType::TypeAnnotation);
        assert!(
            types.contains(&"Tensor".to_string()),
            "Missing type: Tensor. Got: {types:?}"
        );
        assert!(
            types.contains(&"Transform".to_string()),
            "Missing type: Transform. Got: {types:?}"
        );
    }
}

// =========================================================================
// JavaScript tests
// =========================================================================

#[cfg(test)]
mod js_tests {
    use super::*;
    use crate::types::EntityType;
    use std::sync::OnceLock;

    const JS_FIXTURE: &str = r#"/** Module-level JSDoc */

class ImageProcessor {
  /** Process images */
  constructor(threshold) {
    this.threshold = threshold;
  }

  processImage(source, target) {
    const displacement = this.computeDisplacement(source, target);
    return displacement;
  }
}

const useSegmentation = (nbFeatures) => {
  const isValid = nbFeatures > 0;
  return new Float32Array(nbFeatures);
};

function spatialTransform(vol, trf, nbDims = 3) {
  /** Apply spatial transform */
  const ndim = vol.shape.length;
  return vol;
}
"#;

    fn js_fixture_result() -> &'static ParseResult {
        static RESULT: OnceLock<ParseResult> = OnceLock::new();
        RESULT.get_or_init(|| {
            let parser = JavaScriptParser;
            parse_content_with(
                JS_FIXTURE,
                Path::new("test.js"),
                &parser,
            )
            .unwrap()
        })
    }

    fn names_of(
        result: &ParseResult,
        entity_type: EntityType,
    ) -> Vec<String> {
        result
            .identifiers
            .iter()
            .filter(|id| id.entity_type == entity_type)
            .map(|id| id.name.clone())
            .collect()
    }

    #[test]
    fn test_js_functions_found() {
        let result = js_fixture_result();
        let functions = names_of(result, EntityType::Function);
        assert!(
            functions.contains(&"spatialTransform".to_string()),
            "Missing: spatialTransform"
        );
        assert!(
            functions.contains(&"useSegmentation".to_string()),
            "Missing arrow fn: useSegmentation"
        );
        assert!(
            functions.contains(&"processImage".to_string()),
            "Missing method: processImage"
        );
        assert!(
            functions.contains(&"constructor".to_string()),
            "Missing: constructor"
        );
    }

    #[test]
    fn test_js_classes_found() {
        let result = js_fixture_result();
        let classes = names_of(result, EntityType::Class);
        assert!(
            classes.contains(&"ImageProcessor".to_string()),
            "Missing class: ImageProcessor"
        );
    }

    #[test]
    fn test_js_variables_found() {
        let result = js_fixture_result();
        let vars = names_of(result, EntityType::Variable);
        assert!(
            vars.contains(&"displacement".to_string()),
            "Missing variable: displacement"
        );
        assert!(
            vars.contains(&"isValid".to_string()),
            "Missing variable: isValid"
        );
        assert!(
            vars.contains(&"ndim".to_string()),
            "Missing variable: ndim"
        );
    }

    #[test]
    fn test_js_arrow_function_named() {
        let result = js_fixture_result();
        let functions = names_of(result, EntityType::Function);
        assert!(
            functions.contains(&"useSegmentation".to_string()),
            "Arrow fn should be Function, not Variable"
        );
    }

    #[test]
    fn test_js_jsdoc_extracted() {
        let result = js_fixture_result();
        assert!(
            !result.doc_texts.is_empty(),
            "Expected JSDoc comments"
        );
        let texts: Vec<&str> = result
            .doc_texts
            .iter()
            .map(|(_, _, t)| t.as_str())
            .collect();
        assert!(
            texts.iter().any(|t| t.contains("Module-level JSDoc")),
            "Missing module JSDoc. Got: {texts:?}"
        );
        assert!(
            texts.iter().any(|t| t.contains("Process images")),
            "Missing class JSDoc. Got: {texts:?}"
        );
    }

    #[test]
    fn test_js_class_info() {
        let result = js_fixture_result();
        let cls = result
            .classes
            .iter()
            .find(|c| c.name == "ImageProcessor")
            .expect("Missing class info: ImageProcessor");
        assert!(
            cls.methods.contains(&"processImage".to_string()),
            "Missing method: processImage"
        );
        assert!(
            cls.methods.contains(&"constructor".to_string()),
            "Missing method: constructor"
        );
        assert!(
            cls.attributes.contains(&"threshold".to_string()),
            "Missing attribute: threshold. Got: {:?}",
            cls.attributes
        );
    }

    #[test]
    fn test_js_signatures_extracted() {
        let result = js_fixture_result();
        let spatial = result
            .signatures
            .iter()
            .find(|s| s.name == "spatialTransform")
            .expect("Missing signature: spatialTransform");
        assert_eq!(spatial.params.len(), 3);
        assert_eq!(spatial.params[0].name, "vol");
        assert!(spatial.params[0].type_annotation.is_none());
        assert_eq!(spatial.params[2].name, "nbDims");
        assert_eq!(spatial.params[2].default, Some("3".to_string()));
        assert!(spatial.return_type.is_none());
    }

    #[test]
    fn test_js_parameters_found() {
        let result = js_fixture_result();
        let params = names_of(result, EntityType::Parameter);
        assert!(
            params.contains(&"vol".to_string()),
            "Missing param: vol. Got: {params:?}"
        );
        assert!(
            params.contains(&"trf".to_string()),
            "Missing param: trf"
        );
        assert!(
            params.contains(&"nbDims".to_string()),
            "Missing param: nbDims"
        );
        assert!(
            params.contains(&"nbFeatures".to_string()),
            "Missing param: nbFeatures"
        );
    }

    #[test]
    fn test_js_call_sites() {
        let result = js_fixture_result();
        let callees: Vec<&str> = result
            .call_sites
            .iter()
            .map(|c| c.callee.as_str())
            .collect();
        assert!(
            callees.iter().any(|c| c.contains("computeDisplacement")),
            "Missing call: computeDisplacement. Got: {callees:?}"
        );
    }

    #[test]
    fn test_js_scope_tracking() {
        let result = js_fixture_result();

        let spatial = result
            .identifiers
            .iter()
            .find(|id| id.name == "spatialTransform")
            .unwrap();
        assert_eq!(spatial.scope, None);

        let process = result
            .identifiers
            .iter()
            .find(|id| {
                id.name == "processImage"
                    && id.entity_type == EntityType::Function
            })
            .unwrap();
        assert_eq!(
            process.scope,
            Some("ImageProcessor".to_string())
        );
    }

    #[test]
    fn test_js_no_interfaces_or_type_aliases() {
        let result = js_fixture_result();
        let interfaces = names_of(result, EntityType::Interface);
        let aliases = names_of(result, EntityType::TypeAlias);
        assert!(interfaces.is_empty(), "JS should have no interfaces");
        assert!(aliases.is_empty(), "JS should have no type aliases");
    }
}

// =========================================================================
// Cross-language parity tests
// =========================================================================

#[cfg(test)]
mod cross_language_tests {
    use super::*;
    use crate::types::EntityType;
    use std::path::Path;

    fn names_of(
        result: &ParseResult,
        entity_type: EntityType,
    ) -> Vec<String> {
        result
            .identifiers
            .iter()
            .filter(|id| id.entity_type == entity_type)
            .map(|id| id.name.clone())
            .collect()
    }

    // The three equivalent source snippets define the same domain concepts.
    const PY_SOURCE: &str = r#"
class SpatialTransform:
    def __init__(self, source, target):
        self.displacement = source - target

    def apply(self, volume):
        return volume

def compute_displacement(source, target):
    displacement = source - target
    return displacement
"#;

    const TS_SOURCE: &str = r#"
class SpatialTransform {
    displacement: number;
    constructor(source: any, target: any) {
        this.displacement = source - target;
    }
    apply(volume: any): any {
        return volume;
    }
}

function computeDisplacement(source: any, target: any): number {
    const displacement = source - target;
    return displacement;
}
"#;

    const JS_SOURCE: &str = r#"
class SpatialTransform {
    constructor(source, target) {
        this.displacement = source - target;
    }
    apply(volume) {
        return volume;
    }
}

function computeDisplacement(source, target) {
    const displacement = source - target;
    return displacement;
}
"#;

    #[test]
    fn test_cross_language_parity() {
        let py = parse_content_with(
            PY_SOURCE, Path::new("test.py"), &PythonParser,
        )
        .unwrap();
        let ts = parse_content_with(
            TS_SOURCE, Path::new("test.ts"), &TypeScriptParser,
        )
        .unwrap();
        let js = parse_content_with(
            JS_SOURCE, Path::new("test.js"), &JavaScriptParser,
        )
        .unwrap();

        for (label, result) in
            [("Python", &py), ("TypeScript", &ts), ("JavaScript", &js)]
        {
            let classes = names_of(result, EntityType::Class);
            assert!(
                classes.contains(&"SpatialTransform".to_string()),
                "{label}: missing class SpatialTransform. Got: {classes:?}"
            );

            let fns = names_of(result, EntityType::Function);
            assert!(
                fns.contains(&"apply".to_string()),
                "{label}: missing method 'apply'. Got: {fns:?}"
            );

            let vars = names_of(result, EntityType::Variable);
            assert!(
                vars.contains(&"displacement".to_string()),
                "{label}: missing variable displacement. Got: {vars:?}"
            );

            let attrs = names_of(result, EntityType::Attribute);
            assert!(
                attrs.contains(&"displacement".to_string()),
                "{label}: missing attribute displacement \
                 (self/this.displacement). Got: {attrs:?}"
            );
        }

        // Standalone function names differ by convention (snake_case vs
        // camelCase), but both must be classified as Function.
        assert!(
            names_of(&py, EntityType::Function)
                .contains(&"compute_displacement".to_string()),
            "Python: missing fn compute_displacement"
        );
        assert!(
            names_of(&ts, EntityType::Function)
                .contains(&"computeDisplacement".to_string()),
            "TypeScript: missing fn computeDisplacement"
        );
        assert!(
            names_of(&js, EntityType::Function)
                .contains(&"computeDisplacement".to_string()),
            "JavaScript: missing fn computeDisplacement"
        );
    }

    // -----------------------------------------------------------------------
    // TS-specific features
    // -----------------------------------------------------------------------

    #[test]
    fn test_ts_interface_extraction() {
        let source = r#"
interface VolumeTransform {
    source: Float32Array;
    displacement: number[];
    apply(volume: any): any;
}
"#;
        let result = parse_content_with(
            source, Path::new("test.ts"), &TypeScriptParser,
        )
        .unwrap();
        let interfaces = names_of(&result, EntityType::Interface);
        assert!(
            interfaces.contains(&"VolumeTransform".to_string()),
            "Missing interface VolumeTransform. Got: {interfaces:?}"
        );
    }

    #[test]
    fn test_ts_type_alias_extraction() {
        let source = r#"
type DisplacementField = Float32Array;
type RegistrationResult = { source: any; target: any };
"#;
        let result = parse_content_with(
            source, Path::new("test.ts"), &TypeScriptParser,
        )
        .unwrap();
        let aliases = names_of(&result, EntityType::TypeAlias);
        assert!(
            aliases.contains(&"DisplacementField".to_string()),
            "Missing type alias DisplacementField. Got: {aliases:?}"
        );
        assert!(
            aliases.contains(&"RegistrationResult".to_string()),
            "Missing type alias RegistrationResult. Got: {aliases:?}"
        );
    }

    #[test]
    fn test_ts_jsdoc_with_param_tags() {
        let source = r#"
/**
 * Apply a spatial transform to a volume.
 * @param source - the source volume
 * @param displacement - the displacement field
 * @returns transformed volume
 */
function applyTransform(source: any, displacement: any): any {
    return source;
}
"#;
        let result = parse_content_with(
            source, Path::new("test.ts"), &TypeScriptParser,
        )
        .unwrap();
        assert!(
            !result.doc_texts.is_empty(),
            "Expected JSDoc to be extracted"
        );
        let combined: String = result
            .doc_texts
            .iter()
            .map(|(_, _, t)| t.as_str())
            .collect::<Vec<_>>()
            .join("\n");
        assert!(
            combined.contains("spatial transform"),
            "Expected 'spatial transform' in doc. Got: {combined:?}"
        );
        assert!(
            combined.contains("@param"),
            "Expected @param tags in doc. Got: {combined:?}"
        );
    }

    #[test]
    fn test_ts_decorators_extracted() {
        let source = r#"
class Model {
    @injectable
    private service: any;

    @deprecated
    oldMethod(): void {}
}
"#;
        let result = parse_content_with(
            source, Path::new("test.ts"), &TypeScriptParser,
        )
        .unwrap();
        let decorators = names_of(&result, EntityType::Decorator);
        assert!(
            decorators.contains(&"injectable".to_string()),
            "Missing decorator: injectable. Got: {decorators:?}"
        );
        assert!(
            decorators.contains(&"deprecated".to_string()),
            "Missing decorator: deprecated. Got: {decorators:?}"
        );
    }

    #[test]
    fn test_ts_export_default_class() {
        let source = r#"
export default class VolumeRenderer {
    render(volume: any): void {}
}
"#;
        let result = parse_content_with(
            source, Path::new("test.ts"), &TypeScriptParser,
        )
        .unwrap();
        let classes = names_of(&result, EntityType::Class);
        assert!(
            classes.contains(&"VolumeRenderer".to_string()),
            "Missing class from export default class. Got: {classes:?}"
        );
    }

    #[test]
    fn test_ts_export_const_arrow_function() {
        let source = r#"
export const computeNorm = (vec: number[]): number => {
    return Math.sqrt(vec.reduce((a, b) => a + b * b, 0));
};
"#;
        let result = parse_content_with(
            source, Path::new("test.ts"), &TypeScriptParser,
        )
        .unwrap();
        let fns = names_of(&result, EntityType::Function);
        assert!(
            fns.contains(&"computeNorm".to_string()),
            "export const arrow fn should be Function. Got: {fns:?}"
        );
        let vars = names_of(&result, EntityType::Variable);
        assert!(
            !vars.contains(&"computeNorm".to_string()),
            "Arrow fn computeNorm must not be a Variable. Got: {vars:?}"
        );
    }

    #[test]
    fn test_ts_arrow_function_with_complex_types() {
        let source = r#"
const processVolume = (
    source: Float32Array,
    target: Float32Array,
    options: { normalize: boolean; nbDims: number }
): Promise<Float32Array> => {
    return Promise.resolve(source);
};
"#;
        let result = parse_content_with(
            source, Path::new("test.ts"), &TypeScriptParser,
        )
        .unwrap();
        let fns = names_of(&result, EntityType::Function);
        assert!(
            fns.contains(&"processVolume".to_string()),
            "Missing arrow fn with complex types. Got: {fns:?}"
        );
        let params = names_of(&result, EntityType::Parameter);
        assert!(
            params.contains(&"source".to_string()),
            "Missing param source. Got: {params:?}"
        );
        assert!(
            params.contains(&"target".to_string()),
            "Missing param target. Got: {params:?}"
        );
    }

    // -----------------------------------------------------------------------
    // JS edge cases: patterns that must not crash
    // -----------------------------------------------------------------------

    #[test]
    fn test_js_module_exports_no_crash() {
        let source = r#"
function computeTransform(source, target) {
    return source - target;
}

module.exports = {
    computeTransform,
};
"#;
        let result = parse_content_with(
            source, Path::new("test.js"), &JavaScriptParser,
        );
        assert!(result.is_ok(), "module.exports should not crash the parser");
        let fns = names_of(&result.unwrap(), EntityType::Function);
        assert!(
            fns.contains(&"computeTransform".to_string()),
            "Missing fn with module.exports. Got: {fns:?}"
        );
    }

    #[test]
    fn test_js_require_calls_no_crash() {
        let source = r#"
const fs = require('fs');
const path = require('path');

function loadVolume(filepath) {
    return fs.readFileSync(filepath);
}
"#;
        let result = parse_content_with(
            source, Path::new("test.js"), &JavaScriptParser,
        );
        assert!(result.is_ok(), "require() calls should not crash the parser");
        let fns = names_of(&result.unwrap(), EntityType::Function);
        assert!(
            fns.contains(&"loadVolume".to_string()),
            "Missing fn with require. Got: {fns:?}"
        );
    }

    #[test]
    fn test_js_prototype_pattern_no_crash() {
        let source = r#"
function VolumeProcessor(threshold) {
    this.threshold = threshold;
}

VolumeProcessor.prototype.process = function(volume) {
    return volume;
};
"#;
        let result = parse_content_with(
            source, Path::new("test.js"), &JavaScriptParser,
        );
        assert!(
            result.is_ok(),
            "Prototype patterns should not crash the parser"
        );
        let fns = names_of(&result.unwrap(), EntityType::Function);
        assert!(
            fns.contains(&"VolumeProcessor".to_string()),
            "Missing constructor fn VolumeProcessor. Got: {fns:?}"
        );
    }
}

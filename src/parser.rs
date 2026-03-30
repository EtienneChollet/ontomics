use crate::types::{
    CallSite, ClassInfo, EntityType, Param, ParseResult, RawIdentifier,
    Signature,
};
use anyhow::Result;
use ignore::WalkBuilder;
use rayon::prelude::*;
use std::path::{Path, PathBuf};

/// Language-specific parsing strategy. Each implementation knows how to
/// traverse its tree-sitter grammar and extract identifiers, docstrings,
/// signatures, classes, and call sites into the shared types.
#[allow(dead_code)]
pub trait LanguageParser: Send + Sync {
    /// File extensions this parser handles (without dots).
    fn extensions(&self) -> &[&str];

    /// Initialize a tree-sitter parser with the correct grammar.
    fn make_parser(&self) -> Result<tree_sitter::Parser>;

    /// Parameters to skip (e.g., `["self", "cls"]` for Python).
    fn skip_params(&self) -> &[&str];

    /// Strip language-specific doc comment syntax.
    fn strip_doc_syntax<'a>(&self, raw: &'a str) -> &'a str;

    /// Walk the AST and extract identifiers, doc text, signatures,
    /// classes, and call sites.
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

pub struct PythonParser;

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
        &["self", "cls"]
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
                let skip = self.skip_params();
                let func_name = extract_function(
                    node, source, path, scope, identifiers, skip,
                );
                if let Some(sig) = extract_signature(
                    node, source, path, scope, skip,
                ) {
                    signatures.push(sig);
                }
                let child_scope = func_name.map(|name| match scope {
                    Some(parent) => format!("{parent}.{name}"),
                    None => name,
                });
                visit_children(
                    self,
                    node,
                    source,
                    path,
                    &child_scope,
                    identifiers,
                    doc_texts,
                    signatures,
                    classes,
                    call_sites,
                );
                return;
            }
            "class_definition" => {
                let class_name =
                    extract_class(node, source, path, scope, identifiers);
                if let Some(info) = extract_class_info(node, source, path) {
                    classes.push(info);
                }
                let child_scope = class_name.or_else(|| scope.clone());
                visit_children(
                    self,
                    node,
                    source,
                    path,
                    &child_scope,
                    identifiers,
                    doc_texts,
                    signatures,
                    classes,
                    call_sites,
                );
                return;
            }
            "decorated_definition" => {
                extract_decorators(node, source, path, scope, identifiers);
            }
            "assignment" => {
                extract_assignment(node, source, path, scope, identifiers);
            }
            "expression_statement" => {
                try_extract_docstring(node, source, path, doc_texts);
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
            self,
            node,
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

/// Convenience constructor for a Python parser.
pub fn python_parser() -> PythonParser {
    PythonParser
}

/// Parse a single file and extract identifiers and docstrings.
pub fn parse_file(path: &Path, parser: &dyn LanguageParser) -> Result<ParseResult> {
    let source = std::fs::read_to_string(path)?;
    parse_content(&source, path, parser)
}

/// Parse source content and extract identifiers and docstrings.
///
/// Like `parse_file`, but takes source content directly instead of reading
/// from disk. Used by the diff module to parse files from git tree objects.
pub fn parse_content(
    source: &str,
    path: &Path,
    parser: &dyn LanguageParser,
) -> Result<ParseResult> {
    let mut ts_parser = parser.make_parser()?;

    let tree = ts_parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("Failed to parse {}", path.display()))?;

    let source_bytes = source.as_bytes();
    let mut identifiers = Vec::new();
    let mut doc_texts = Vec::new();
    let mut signatures = Vec::new();
    let mut classes = Vec::new();
    let mut call_sites = Vec::new();

    parser.visit_node(
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

/// Parse all matching files in a directory tree using parallel iteration.
///
/// Uses `ignore::WalkBuilder` to natively respect `.gitignore` files
/// and skip hidden files/directories.
pub fn parse_directory(
    root: &Path,
    opts: &ParseOptions,
    parser: &dyn LanguageParser,
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
        .filter(|entry| {
            entry.file_type().is_some_and(|ft| ft.is_file())
        })
        .map(|entry| entry.into_path())
        .collect();

    let results: Vec<ParseResult> = paths
        .par_iter()
        .filter_map(|p| parse_file(p, parser).ok())
        .collect();

    Ok(results)
}

// --- Private helpers ---

/// Recurse into all children of a node, dispatching through the parser trait.
#[allow(clippy::too_many_arguments)]
fn visit_children(
    parser: &dyn LanguageParser,
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
        parser.visit_node(
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

/// Extract function name and its parameters. Returns the function name
/// (used to build scope for children).
fn extract_function(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
    skip_params: &[&str],
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
        extract_parameters(
            params_node,
            source,
            path,
            &func_scope,
            identifiers,
            skip_params,
        );
    }

    Some(name.to_string())
}

/// Extract parameter names from a `parameters` node.
fn extract_parameters(
    params_node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
    skip_params: &[&str],
) {
    let mut cursor = params_node.walk();
    for child in params_node.named_children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                push_param_if_valid(
                    child, source, path, scope, identifiers, skip_params,
                );
            }
            "typed_parameter" | "typed_default_parameter" => {
                if let Some(name_node) = find_first_identifier_child(child) {
                    push_param_if_valid(
                        name_node, source, path, scope, identifiers,
                        skip_params,
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
                    push_param_if_valid(
                        name_node, source, path, scope, identifiers,
                        skip_params,
                    );
                }
            }
            "list_splat_pattern" | "dictionary_splat_pattern" => {}
            _ => {}
        }
    }
}

/// Push a parameter identifier if it's not in the skip list.
fn push_param_if_valid(
    name_node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
    skip_params: &[&str],
) {
    if let Ok(name) = name_node.utf8_text(source) {
        if !skip_params.contains(&name) {
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

/// Find the first `identifier` child of a node (for typed/default params).
fn find_first_identifier_child(
    node: tree_sitter::Node,
) -> Option<tree_sitter::Node> {
    find_named_child_by_kind(node, "identifier")
}

/// Extract the class name. Returns the class scope string for children.
fn extract_class(
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

/// Extract decorator names from a `decorated_definition` node.
fn extract_decorators(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "decorator" {
            if let Some(name) = extract_decorator_name(child, source) {
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

/// Extract the name from a decorator node. Handles simple `@foo` and
/// callable `@foo(...)` forms.
fn extract_decorator_name(
    decorator_node: tree_sitter::Node,
    source: &[u8],
) -> Option<String> {
    let mut cursor = decorator_node.walk();
    for child in decorator_node.named_children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                return child
                    .utf8_text(source)
                    .ok()
                    .map(|s| s.to_string());
            }
            "attribute" => {
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

/// Extract variable name from a simple assignment (`x = ...`).
/// Also extracts attribute names for `self.foo = ...` patterns,
/// and type annotation identifiers for annotated assignments.
fn extract_assignment(
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
                extract_attribute(left, source, path, scope, identifiers);
            }
            _ => {}
        }
    }
    // Handle type annotation on the assignment (e.g., `x: int = 5`)
    if let Some(type_node) = node.child_by_field_name("type") {
        extract_type_annotation(type_node, source, path, scope, identifiers);
    }
}

/// Extract attribute name from `self.foo` or `obj.bar` patterns.
/// Only the final attribute name is extracted (e.g., `foo` from `self.foo`).
fn extract_attribute(
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

/// Extract type identifier from a type annotation node.
/// Handles simple types (`int`, `str`), custom types (`Tensor`),
/// and the wrapping `type` node that tree-sitter uses.
fn extract_type_annotation(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    match node.kind() {
        "identifier" => {
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
        // For `Optional[X]`, `List[X]`, etc. — extract the outer type
        "subscript" => {
            if let Some(value) = node.child_by_field_name("value") {
                extract_type_annotation(
                    value, source, path, scope, identifiers,
                );
            }
        }
        "attribute" => {
            // e.g., `torch.Tensor` — extract as-is
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

/// If this `expression_statement` is a docstring (string literal that is the
/// first statement in a function, class, or module body), extract it.
fn try_extract_docstring(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    doc_texts: &mut Vec<(std::path::PathBuf, usize, String)>,
) {
    let string_node = match find_named_child_by_kind(node, "string") {
        Some(n) => n,
        None => return,
    };

    if !is_first_statement_in_block(node) {
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

/// Check if a node is the first statement in a block body (function, class,
/// or module).
fn is_first_statement_in_block(node: tree_sitter::Node) -> bool {
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

// --- L2 extraction helpers ---

/// Extract a `Signature` from a `function_definition` node.
fn extract_signature(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    skip_params: &[&str],
) -> Option<Signature> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?.to_string();
    let line = name_node.start_position().row + 1;

    let mut params = Vec::new();
    if let Some(params_node) = node.child_by_field_name("parameters") {
        let mut cursor = params_node.walk();
        for child in params_node.named_children(&mut cursor) {
            if let Some(param) = extract_param(child, source, skip_params) {
                params.push(param);
            }
        }
    }

    let return_type = node
        .child_by_field_name("return_type")
        .and_then(|n| n.utf8_text(source).ok())
        .map(|s| s.to_string());

    let decorators = extract_decorator_names_for_sig(node, source);
    let docstring_first_line =
        extract_body_docstring_first_line(node, source);

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

/// Extract a `Param` from a parameter node in a function's parameter list.
fn extract_param(
    node: tree_sitter::Node,
    source: &[u8],
    skip_params: &[&str],
) -> Option<Param> {
    match node.kind() {
        "identifier" => {
            let name = node.utf8_text(source).ok()?.to_string();
            if skip_params.contains(&name.as_str()) {
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
            if skip_params.contains(&name.as_str()) {
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
            if skip_params.contains(&name.as_str()) {
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
            if skip_params.contains(&name.as_str()) {
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

/// Collect decorator names from a function's parent `decorated_definition`.
fn extract_decorator_names_for_sig(
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
            if let Some(name) = extract_decorator_name(child, source) {
                decorators.push(name);
            }
        }
    }
    decorators
}

/// Extract the first line of a docstring from a function/class body.
fn extract_body_docstring_first_line(
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

/// Extract a `ClassInfo` from a `class_definition` node.
fn extract_class_info(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
) -> Option<ClassInfo> {
    let name_node = node.child_by_field_name("name")?;
    let name = name_node.utf8_text(source).ok()?.to_string();
    let line = name_node.start_position().row + 1;

    // Base classes
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
        extract_body_docstring_first_line(node, source);

    if let Some(body) = node.child_by_field_name("body") {
        let mut cursor = body.walk();
        for child in body.named_children(&mut cursor) {
            // Methods: function_definition or decorated function
            match child.kind() {
                "function_definition" => {
                    if let Some(n) = child.child_by_field_name("name") {
                        if let Ok(method_name) = n.utf8_text(source) {
                            methods.push(method_name.to_string());
                        }
                    }
                }
                "decorated_definition" => {
                    if let Some(func) =
                        find_named_child_by_kind(child, "function_definition")
                    {
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

        // Collect self.X attributes by walking the entire class body
        collect_self_attributes(body, source, &mut attributes);
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

/// Recursively scan a class body for `self.X = ...` attribute assignments.
fn collect_self_attributes(
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
        collect_self_attributes(child, source, attributes);
    }
}

/// Extract a `CallSite` from a `call` expression node.
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
            let path = Path::new("/tmp/semex_test_parser.py");
            std::fs::write(path, FIXTURE).unwrap();
            let parser = PythonParser;
            parse_file(path, &parser).unwrap()
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

        // Module-level function has no scope
        let spatial = result
            .identifiers
            .iter()
            .find(|id| id.name == "spatial_transform")
            .unwrap();
        assert_eq!(spatial.scope, None);

        // Class itself has no scope (defined at module level)
        let class = result
            .identifiers
            .iter()
            .find(|id| id.name == "MyClass")
            .unwrap();
        assert_eq!(class.scope, None);

        // Method __init__ is scoped to MyClass
        let init = result
            .identifiers
            .iter()
            .find(|id| {
                id.name == "__init__"
                    && id.entity_type == EntityType::Function
            })
            .unwrap();
        assert_eq!(init.scope, Some("MyClass".to_string()));

        // Parameter nb_features is scoped to MyClass.__init__
        let nb_feat = result
            .identifiers
            .iter()
            .find(|id| {
                id.name == "nb_features"
                    && id.entity_type == EntityType::Parameter
            })
            .unwrap();
        assert_eq!(nb_feat.scope, Some("MyClass.__init__".to_string()));

        // Variable displacement is inside MyClass.compute_transform
        let disp = result
            .identifiers
            .iter()
            .find(|id| id.name == "displacement")
            .unwrap();
        assert_eq!(
            disp.scope,
            Some("MyClass.compute_transform".to_string())
        );

        // Variable ndim is inside spatial_transform (module-level)
        let ndim = result
            .identifiers
            .iter()
            .find(|id| id.name == "ndim")
            .unwrap();
        assert_eq!(ndim.scope, Some("spatial_transform".to_string()));

        // Parameter vol is scoped to spatial_transform
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
        let parser = PythonParser;
        let result = parse_content(source, path, &parser).unwrap();
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
        let parser = PythonParser;
        let result = parse_content(source, path, &parser).unwrap();
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

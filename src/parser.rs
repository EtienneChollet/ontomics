use crate::types::{EntityType, ParseResult, RawIdentifier};
use anyhow::Result;
use rayon::prelude::*;
use std::path::Path;

/// Parse a single Python file and extract identifiers and docstrings.
pub fn parse_file(path: &Path) -> Result<ParseResult> {
    let source = std::fs::read_to_string(path)?;
    parse_content(&source, path)
}

/// Parse Python source content and extract identifiers and docstrings.
///
/// Like `parse_file`, but takes source content directly instead of reading
/// from disk. Used by the diff module to parse files from git tree objects.
pub fn parse_content(source: &str, path: &Path) -> Result<ParseResult> {
    let mut parser = tree_sitter::Parser::new();
    let language = tree_sitter_python::LANGUAGE;
    parser
        .set_language(&language.into())
        .expect("Error loading Python grammar");

    let tree = parser
        .parse(source, None)
        .ok_or_else(|| anyhow::anyhow!("Failed to parse {}", path.display()))?;

    let source_bytes = source.as_bytes();
    let mut identifiers = Vec::new();
    let mut doc_texts = Vec::new();

    visit_node(
        tree.root_node(),
        source_bytes,
        path,
        &mut identifiers,
        &mut doc_texts,
    );

    Ok(ParseResult {
        identifiers,
        doc_texts,
    })
}

/// Parse all `.py` files in a directory tree using parallel iteration.
pub fn parse_directory(root: &Path) -> Result<Vec<ParseResult>> {
    let pattern = format!("{}/**/*.py", root.display());
    let paths: Vec<_> = glob::glob(&pattern)?
        .filter_map(|entry| entry.ok())
        .filter(|p| {
            let s = p.to_string_lossy();
            !s.contains("__pycache__")
                && !s.contains("/venv/")
                && !s.contains("/venv.")
                && !s.contains("/.venv/")
                && !s.contains("/node_modules/")
                && !s.contains("/.git/")
                && !s.contains("/site-packages/")
                && !s.contains("/.tox/")
                && !s.contains("/.eggs/")
                && !s.contains("/egg-info/")
        })
        .collect();

    let results: Vec<ParseResult> = paths
        .par_iter()
        .filter_map(|p| parse_file(p).ok())
        .collect();

    Ok(results)
}

// --- Private helpers ---

const SKIP_PARAMS: &[&str] = &["self", "cls"];

/// Recursively visit a tree-sitter node, extracting identifiers and docstrings.
fn visit_node(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    identifiers: &mut Vec<RawIdentifier>,
    doc_texts: &mut Vec<(std::path::PathBuf, usize, String)>,
) {
    match node.kind() {
        "function_definition" => {
            extract_function(node, source, path, identifiers);
        }
        "class_definition" => {
            extract_class(node, source, path, identifiers);
        }
        "decorated_definition" => {
            extract_decorators(node, source, path, identifiers);
        }
        "assignment" => {
            extract_assignment(node, source, path, identifiers);
        }
        "expression_statement" => {
            try_extract_docstring(node, source, path, doc_texts);
        }
        _ => {}
    }

    // Recurse into children
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        visit_node(child, source, path, identifiers, doc_texts);
    }
}

/// Extract function name and its parameters.
fn extract_function(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    identifiers: &mut Vec<RawIdentifier>,
) {
    if let Some(name_node) = node.child_by_field_name("name") {
        if let Ok(name) = name_node.utf8_text(source) {
            identifiers.push(RawIdentifier {
                name: name.to_string(),
                entity_type: EntityType::Function,
                file: path.to_path_buf(),
                line: name_node.start_position().row + 1,
            });
        }
    }

    if let Some(params_node) = node.child_by_field_name("parameters") {
        extract_parameters(params_node, source, path, identifiers);
    }
}

/// Extract parameter names from a `parameters` node.
fn extract_parameters(
    params_node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    identifiers: &mut Vec<RawIdentifier>,
) {
    let mut cursor = params_node.walk();
    for child in params_node.named_children(&mut cursor) {
        match child.kind() {
            // Simple parameter: just an identifier
            "identifier" => {
                push_param_if_valid(child, source, path, identifiers);
            }
            // `name: type` or `name = default` or `name: type = default`
            "typed_parameter" | "default_parameter" | "typed_default_parameter" => {
                // The param name is the first identifier child
                if let Some(name_node) = find_first_identifier_child(child) {
                    push_param_if_valid(name_node, source, path, identifiers);
                }
            }
            // *args, **kwargs — skip
            "list_splat_pattern" | "dictionary_splat_pattern" => {}
            _ => {}
        }
    }
}

/// Push a parameter identifier if it's not `self` or `cls`.
fn push_param_if_valid(
    name_node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    identifiers: &mut Vec<RawIdentifier>,
) {
    if let Ok(name) = name_node.utf8_text(source) {
        if !SKIP_PARAMS.contains(&name) {
            identifiers.push(RawIdentifier {
                name: name.to_string(),
                entity_type: EntityType::Parameter,
                file: path.to_path_buf(),
                line: name_node.start_position().row + 1,
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
fn find_first_identifier_child(node: tree_sitter::Node) -> Option<tree_sitter::Node> {
    find_named_child_by_kind(node, "identifier")
}

/// Extract the class name.
fn extract_class(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    identifiers: &mut Vec<RawIdentifier>,
) {
    if let Some(name_node) = node.child_by_field_name("name") {
        if let Ok(name) = name_node.utf8_text(source) {
            identifiers.push(RawIdentifier {
                name: name.to_string(),
                entity_type: EntityType::Class,
                file: path.to_path_buf(),
                line: name_node.start_position().row + 1,
            });
        }
    }
}

/// Extract decorator names from a `decorated_definition` node.
fn extract_decorators(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    identifiers: &mut Vec<RawIdentifier>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "decorator" {
            // The decorator node's children: "@" + expression
            // We want the identifier name (could be a dotted name or call)
            if let Some(name) = extract_decorator_name(child, source) {
                identifiers.push(RawIdentifier {
                    name,
                    entity_type: EntityType::Decorator,
                    file: path.to_path_buf(),
                    line: child.start_position().row + 1,
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
                return child.utf8_text(source).ok().map(|s| s.to_string());
            }
            "attribute" => {
                // Dotted name like `module.decorator` — use full text
                return child.utf8_text(source).ok().map(|s| s.to_string());
            }
            "call" => {
                // `@decorator(args)` — get the function part
                if let Some(func_node) = child.child_by_field_name("function") {
                    return func_node.utf8_text(source).ok().map(|s| s.to_string());
                }
            }
            _ => {}
        }
    }
    None
}

/// Extract variable name from a simple assignment (`x = ...`).
fn extract_assignment(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    identifiers: &mut Vec<RawIdentifier>,
) {
    if let Some(left) = node.child_by_field_name("left") {
        if left.kind() == "identifier" {
            if let Ok(name) = left.utf8_text(source) {
                identifiers.push(RawIdentifier {
                    name: name.to_string(),
                    entity_type: EntityType::Variable,
                    file: path.to_path_buf(),
                    line: left.start_position().row + 1,
                });
            }
        }
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
    // Must contain a string child
    let string_node = find_named_child_by_kind(node, "string");
    let string_node = match string_node {
        Some(n) => n,
        None => return,
    };

    // Check: is this the first statement in a block (body of a
    // function/class/module)?
    if !is_first_statement_in_block(node) {
        return;
    }

    if let Ok(text) = string_node.utf8_text(source) {
        // Strip surrounding quotes (""", ''', ", ')
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
    // Strip string prefixes (r, b, u, f, and combinations)
    let s = s.trim_start_matches(|c: char| "rRbBuUfF".contains(c));

    if let Some(inner) = s.strip_prefix("\"\"\"").and_then(|s| s.strip_suffix("\"\"\"")) {
        return inner.trim();
    }
    if let Some(inner) = s.strip_prefix("'''").and_then(|s| s.strip_suffix("'''")) {
        return inner.trim();
    }
    if let Some(inner) = s.strip_prefix('"').and_then(|s| s.strip_suffix('"')) {
        return inner.trim();
    }
    if let Some(inner) = s.strip_prefix('\'').and_then(|s| s.strip_suffix('\'')) {
        return inner.trim();
    }
    s
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

    @staticmethod
    def compute_transform(source, target):
        displacement = source - target
        return displacement

def spatial_transform(vol, trf, nb_dims=3):
    """Apply spatial transform."""
    ndim = len(vol.shape)
    return ndim
"#;

    fn fixture_result() -> &'static ParseResult {
        static RESULT: OnceLock<ParseResult> = OnceLock::new();
        RESULT.get_or_init(|| {
            let path = Path::new("/tmp/semex_test_parser.py");
            std::fs::write(path, FIXTURE).unwrap();
            parse_file(path).unwrap()
        })
    }

    fn names_of(result: &ParseResult, entity_type: EntityType) -> Vec<String> {
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
}

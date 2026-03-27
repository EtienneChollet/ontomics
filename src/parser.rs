use crate::types::{EntityType, ParseResult, RawIdentifier};
use anyhow::Result;
use ignore::WalkBuilder;
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
        &None,
        &mut identifiers,
        &mut doc_texts,
    );

    Ok(ParseResult {
        identifiers,
        doc_texts,
    })
}

/// Parse all `.py` files in a directory tree using parallel iteration.
///
/// Uses `ignore::WalkBuilder` to natively respect `.gitignore` files
/// and skip hidden files/directories.
pub fn parse_directory(root: &Path) -> Result<Vec<ParseResult>> {
    let paths: Vec<_> = WalkBuilder::new(root)
        .standard_filters(true)
        .build()
        .filter_map(|entry| entry.ok())
        .filter(|entry| {
            entry.file_type().is_some_and(|ft| ft.is_file())
                && entry.path().extension().is_some_and(|ext| ext == "py")
        })
        .map(|entry| entry.into_path())
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
///
/// `scope` tracks the enclosing class/function for co-occurrence analysis.
fn visit_node(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
    doc_texts: &mut Vec<(std::path::PathBuf, usize, String)>,
) {
    match node.kind() {
        "function_definition" => {
            let func_name =
                extract_function(node, source, path, scope, identifiers);
            // Build new scope for children inside the function body
            let child_scope = func_name.map(|name| match scope {
                Some(parent) => format!("{parent}.{name}"),
                None => name,
            });
            visit_children(
                node,
                source,
                path,
                &child_scope,
                identifiers,
                doc_texts,
            );
            return;
        }
        "class_definition" => {
            let class_name =
                extract_class(node, source, path, scope, identifiers);
            let child_scope = class_name.or_else(|| scope.clone());
            visit_children(
                node,
                source,
                path,
                &child_scope,
                identifiers,
                doc_texts,
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
        _ => {}
    }

    visit_children(node, source, path, scope, identifiers, doc_texts);
}

/// Recurse into all children of a node.
fn visit_children(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
    doc_texts: &mut Vec<(std::path::PathBuf, usize, String)>,
) {
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        visit_node(child, source, path, scope, identifiers, doc_texts);
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
        // Parameters belong to the function's own scope
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
) {
    let mut cursor = params_node.walk();
    for child in params_node.named_children(&mut cursor) {
        match child.kind() {
            "identifier" => {
                push_param_if_valid(child, source, path, scope, identifiers);
            }
            "typed_parameter" | "default_parameter"
            | "typed_default_parameter" => {
                if let Some(name_node) = find_first_identifier_child(child) {
                    push_param_if_valid(
                        name_node,
                        source,
                        path,
                        scope,
                        identifiers,
                    );
                }
            }
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
    scope: &Option<String>,
    identifiers: &mut Vec<RawIdentifier>,
) {
    if let Ok(name) = name_node.utf8_text(source) {
        if !SKIP_PARAMS.contains(&name) {
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
fn extract_assignment(
    node: tree_sitter::Node,
    source: &[u8],
    path: &Path,
    scope: &Option<String>,
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
                    scope: scope.clone(),
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
}

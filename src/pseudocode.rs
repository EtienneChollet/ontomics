// L4: AST-based pseudocode generation from function bodies.

use crate::parser::LanguageParser;
use crate::types::{
    ConditionalBranch, Entity, FunctionBody, LoopKind, Pseudocode,
    PseudocodeStep, Signature,
};
use anyhow::Result;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

/// Generate pseudocode for a function body by re-parsing with
/// tree-sitter.
///
/// Wraps the body text in a dummy function definition so
/// tree-sitter can parse the fragment, then walks the body's
/// AST children to extract structured `PseudocodeStep` items.
pub fn generate_pseudocode(
    body: &FunctionBody,
    parser: &dyn LanguageParser,
    max_lines: usize,
) -> Result<Pseudocode> {
    let entity_id = crate::types::Entity::hash_id(
        &body.entity_name,
        &body.file,
        body.start_line,
    );
    let body_hash = hash_body(&body.body_text);

    let wrapped = wrap_body(body, parser);
    let mut ts_parser = parser.make_parser()?;

    let tree = ts_parser.parse(&wrapped, None).ok_or_else(|| {
        anyhow::anyhow!(
            "Failed to parse body of {}",
            body.entity_name
        )
    })?;

    let source = wrapped.as_bytes();
    let root = tree.root_node();

    // Find the function body node inside the wrapper.
    let exts = parser.extensions();
    let body_node = find_body_node(root, parser);
    let steps = match body_node {
        Some(node) => extract_steps(node, source, exts),
        None => vec![],
    };

    let original_len = steps.len();
    let steps = truncate_steps(steps, max_lines);
    let omitted_count = original_len.saturating_sub(steps.len());

    Ok(Pseudocode {
        entity_id,
        steps,
        body_hash,
        omitted_count,
    })
}

/// Format structured pseudocode as human-readable indented text.
pub fn format_pseudocode(pseudocode: &Pseudocode) -> String {
    let mut lines = Vec::new();
    let step_count = pseudocode.steps.len();
    let midpoint = step_count / 2;
    for (i, step) in pseudocode.steps.iter().enumerate() {
        format_step(step, 0, &mut lines);
        if pseudocode.omitted_count > 0 && i + 1 == midpoint {
            lines.push(format!(
                "... ({} steps omitted) ...",
                pseudocode.omitted_count
            ));
        }
    }
    lines.join("\n")
}

/// Generate pseudocode for all entities that have function bodies.
///
/// Matches entities to signatures by name and file path. Entities
/// whose signatures lack a body are skipped.
pub fn generate_all_pseudocode(
    entities: &HashMap<u64, Entity>,
    signatures: &[Signature],
    parsers: &[(&dyn LanguageParser, &str)],
    max_lines: usize,
) -> HashMap<u64, Pseudocode> {
    let mut result = HashMap::new();

    for entity in entities.values() {
        let sig = signatures.iter().find(|s| {
            s.name == entity.name && s.file == entity.file
        });
        let body = sig.and_then(|s| s.body.as_ref());

        if let Some(b) = body {
            let ext = b.file.extension()
                .and_then(|e| e.to_str())
                .unwrap_or("");
            let parser = parsers.iter()
                .find(|(p, _)| p.extensions().contains(&ext))
                .map(|(p, _)| *p);
            let Some(parser) = parser else { continue };

            if let Ok(mut pc) = generate_pseudocode(b, parser, max_lines)
            {
                // Override: entity_id in the struct must match entity.id.
                // generate_pseudocode derives it from body.start_line which
                // differs from sig.line (the entity's identity line).
                pc.entity_id = entity.id;
                result.insert(entity.id, pc);
            }
        }
    }

    result
}

// -------------------------------------------------------------------
// Wrapping helpers
// -------------------------------------------------------------------

/// Wrap body text in a language-appropriate dummy function so
/// tree-sitter can parse the fragment.
fn wrap_body(body: &FunctionBody, parser: &dyn LanguageParser) -> String {
    let exts = parser.extensions();
    if exts.contains(&"py") {
        format!("def _():\n{}", ensure_indented(&body.body_text))
    } else if exts.contains(&"rs") {
        format!("fn _() {{\n{}\n}}", body.body_text)
    } else {
        // TypeScript / JavaScript
        format!("function _() {{\n{}\n}}", body.body_text)
    }
}

/// Ensure each line of Python body text is indented by at least
/// 4 spaces (tree-sitter expects indented block content).
fn ensure_indented(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    if lines.is_empty() {
        return "    pass".to_string();
    }

    // Find minimum non-empty indentation.
    let min_indent = lines
        .iter()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.len() - l.trim_start().len())
        .min()
        .unwrap_or(0);

    if min_indent >= 4 {
        text.to_string()
    } else {
        let pad = " ".repeat(4 - min_indent);
        lines
            .iter()
            .map(|l| {
                if l.trim().is_empty() {
                    String::new()
                } else {
                    format!("{pad}{l}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Walk down from root to find the function body/block node.
fn find_body_node<'a>(
    root: tree_sitter::Node<'a>,
    parser: &dyn LanguageParser,
) -> Option<tree_sitter::Node<'a>> {
    let exts = parser.extensions();
    let is_python = exts.contains(&"py");
    let is_rust = exts.contains(&"rs");

    // Walk the root looking for the wrapper function definition.
    let func_node = find_first_descendant(root, |n| {
        if is_python {
            n.kind() == "function_definition"
        } else if is_rust {
            n.kind() == "function_item"
        } else {
            n.kind() == "function_declaration"
                || n.kind() == "function"
        }
    })?;

    func_node.child_by_field_name("body")
}

fn find_first_descendant<'a>(
    node: tree_sitter::Node<'a>,
    predicate: impl Fn(tree_sitter::Node) -> bool + Copy,
) -> Option<tree_sitter::Node<'a>> {
    if predicate(node) {
        return Some(node);
    }
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if let Some(found) = find_first_descendant(child, predicate) {
            return Some(found);
        }
    }
    None
}

// -------------------------------------------------------------------
// Step extraction
// -------------------------------------------------------------------

fn extract_steps(
    body_node: tree_sitter::Node,
    source: &[u8],
    exts: &[&str],
) -> Vec<PseudocodeStep> {
    let mut steps = Vec::new();
    let mut cursor = body_node.walk();

    for child in body_node.children(&mut cursor) {
        if let Some(step) = node_to_step(child, source, exts) {
            steps.push(step);
        }
    }
    steps
}

fn node_to_step(
    node: tree_sitter::Node,
    source: &[u8],
    exts: &[&str],
) -> Option<PseudocodeStep> {
    let kind = node.kind();
    let is_python = exts.contains(&"py");
    let is_rust = exts.contains(&"rs");

    match kind {
        // --- Conditionals ---
        "if_statement" if is_python => {
            Some(extract_py_conditional(node, source, exts))
        }
        "if_statement" if !is_python && !is_rust => {
            Some(extract_js_conditional(node, source, exts))
        }
        "if_expression" if is_rust => {
            Some(extract_rust_conditional(node, source, exts))
        }

        // --- Loops ---
        "for_statement" if is_python => {
            Some(extract_py_for(node, source, exts))
        }
        "while_statement" => {
            Some(extract_while(node, source, exts, is_python))
        }
        "for_statement" | "for_in_statement" if !is_python && !is_rust => {
            Some(extract_js_for(node, source, exts))
        }
        "for_expression" if is_rust => {
            Some(extract_rust_for(node, source, exts))
        }
        "loop_expression" if is_rust => {
            let body_steps = node
                .child_by_field_name("body")
                .map(|b| extract_steps(b, source, &["rs"]))
                .unwrap_or_default();
            Some(PseudocodeStep::Loop {
                kind: LoopKind::While {
                    condition: "true".to_string(),
                },
                body: body_steps,
            })
        }

        // --- Return ---
        "return_statement" => {
            let value = first_child_text(node, source);
            Some(PseudocodeStep::Return { value })
        }
        // Rust: return_expression
        "return_expression" if is_rust => {
            let value = first_child_text(node, source);
            Some(PseudocodeStep::Return { value })
        }

        // --- Yield ---
        "yield" if is_python => {
            let value = first_child_text(node, source);
            Some(PseudocodeStep::Yield { value })
        }

        // --- Assignments ---
        "assignment" if is_python => extract_assignment(node, source),
        "augmented_assignment" if is_python => {
            extract_assignment(node, source)
        }
        "let_declaration" if is_rust => {
            extract_let_declaration(node, source)
        }
        "lexical_declaration" | "variable_declaration"
            if !is_python && !is_rust =>
        {
            extract_js_declaration(node, source)
        }

        // --- Expression statements (may contain calls) ---
        "expression_statement" => {
            extract_expression_step(node, source, is_python)
        }

        _ => None,
    }
}

// -------------------------------------------------------------------
// Python-specific extractors
// -------------------------------------------------------------------

fn extract_py_conditional(
    node: tree_sitter::Node,
    source: &[u8],
    exts: &[&str],
) -> PseudocodeStep {
    let mut branches = Vec::new();
    let mut cursor = node.walk();

    for child in node.children(&mut cursor) {
        match child.kind() {
            // Main if: condition is first named child, consequence
            // follows.
            "block" => {
                // This is the if-body — attach to previous branch or
                // create one. Handled below via condition tracking.
            }
            "elif_clause" => {
                let cond = child
                    .child_by_field_name("condition")
                    .and_then(|c| node_text(c, source));
                let body_steps = child
                    .child_by_field_name("consequence")
                    .map(|b| extract_steps(b, source, exts))
                    .unwrap_or_default();
                branches.push(ConditionalBranch {
                    condition: cond,
                    body: body_steps,
                });
            }
            "else_clause" => {
                let body_steps = child
                    .child_by_field_name("body")
                    .map(|b| extract_steps(b, source, exts))
                    .unwrap_or_default();
                branches.push(ConditionalBranch {
                    condition: None,
                    body: body_steps,
                });
            }
            _ => {}
        }
    }

    // The if-branch itself: condition + consequence as first branch.
    let if_cond = node
        .child_by_field_name("condition")
        .and_then(|c| node_text(c, source));
    let if_body = node
        .child_by_field_name("consequence")
        .map(|b| extract_steps(b, source, exts))
        .unwrap_or_default();

    branches.insert(
        0,
        ConditionalBranch {
            condition: if_cond,
            body: if_body,
        },
    );

    PseudocodeStep::Conditional { branches }
}

fn extract_py_for(
    node: tree_sitter::Node,
    source: &[u8],
    exts: &[&str],
) -> PseudocodeStep {
    let variable = node
        .child_by_field_name("left")
        .and_then(|n| node_text(n, source))
        .unwrap_or_else(|| "item".to_string());
    let iterable = node
        .child_by_field_name("right")
        .and_then(|n| node_text(n, source))
        .unwrap_or_else(|| "iterable".to_string());
    let body_steps = node
        .child_by_field_name("body")
        .map(|b| extract_steps(b, source, exts))
        .unwrap_or_default();

    PseudocodeStep::Loop {
        kind: LoopKind::For { variable, iterable },
        body: body_steps,
    }
}

fn extract_while(
    node: tree_sitter::Node,
    source: &[u8],
    exts: &[&str],
    _is_python: bool,
) -> PseudocodeStep {
    let condition = node
        .child_by_field_name("condition")
        .and_then(|c| node_text(c, source))
        .unwrap_or_else(|| "true".to_string());
    let body_steps = node
        .child_by_field_name("body")
        .map(|b| extract_steps(b, source, exts))
        .unwrap_or_default();

    PseudocodeStep::Loop {
        kind: LoopKind::While { condition },
        body: body_steps,
    }
}

// -------------------------------------------------------------------
// JS/TS-specific extractors
// -------------------------------------------------------------------

fn extract_js_conditional(
    node: tree_sitter::Node,
    source: &[u8],
    exts: &[&str],
) -> PseudocodeStep {
    let mut branches = Vec::new();

    let cond = node
        .child_by_field_name("condition")
        .and_then(|c| node_text(c, source));
    let if_body = node
        .child_by_field_name("consequence")
        .map(|b| extract_steps(b, source, exts))
        .unwrap_or_default();
    branches.push(ConditionalBranch {
        condition: cond,
        body: if_body,
    });

    if let Some(alt) = node.child_by_field_name("alternative") {
        if alt.kind() == "if_statement" {
            // else-if chain: recurse
            if let PseudocodeStep::Conditional {
                branches: sub,
            } = extract_js_conditional(alt, source, exts)
            {
                branches.extend(sub);
            }
        } else {
            let body_steps = extract_steps(alt, source, exts);
            branches.push(ConditionalBranch {
                condition: None,
                body: body_steps,
            });
        }
    }

    PseudocodeStep::Conditional { branches }
}

fn extract_js_for(
    node: tree_sitter::Node,
    source: &[u8],
    exts: &[&str],
) -> PseudocodeStep {
    // for..in / for..of
    let variable = node
        .child_by_field_name("left")
        .and_then(|n| node_text(n, source))
        .unwrap_or_else(|| "item".to_string());
    let iterable = node
        .child_by_field_name("right")
        .and_then(|n| node_text(n, source))
        .unwrap_or_else(|| "iterable".to_string());
    let body_steps = node
        .child_by_field_name("body")
        .map(|b| extract_steps(b, source, exts))
        .unwrap_or_default();

    PseudocodeStep::Loop {
        kind: LoopKind::For { variable, iterable },
        body: body_steps,
    }
}

// -------------------------------------------------------------------
// Rust-specific extractors
// -------------------------------------------------------------------

fn extract_rust_conditional(
    node: tree_sitter::Node,
    source: &[u8],
    exts: &[&str],
) -> PseudocodeStep {
    let mut branches = Vec::new();

    let cond = node
        .child_by_field_name("condition")
        .and_then(|c| node_text(c, source));
    let if_body = node
        .child_by_field_name("consequence")
        .map(|b| extract_steps(b, source, exts))
        .unwrap_or_default();
    branches.push(ConditionalBranch {
        condition: cond,
        body: if_body,
    });

    if let Some(alt) = node.child_by_field_name("alternative") {
        // else { ... } or else if { ... }
        let mut inner_cursor = alt.walk();
        let first_child = alt.children(&mut inner_cursor).next();
        if let Some(child) = first_child {
            if child.kind() == "if_expression" {
                if let PseudocodeStep::Conditional {
                    branches: sub,
                } = extract_rust_conditional(child, source, exts)
                {
                    branches.extend(sub);
                }
            } else {
                let body_steps = extract_steps(alt, source, exts);
                branches.push(ConditionalBranch {
                    condition: None,
                    body: body_steps,
                });
            }
        }
    }

    PseudocodeStep::Conditional { branches }
}

fn extract_rust_for(
    node: tree_sitter::Node,
    source: &[u8],
    exts: &[&str],
) -> PseudocodeStep {
    let variable = node
        .child_by_field_name("pattern")
        .and_then(|n| node_text(n, source))
        .unwrap_or_else(|| "item".to_string());
    let iterable = node
        .child_by_field_name("value")
        .and_then(|n| node_text(n, source))
        .unwrap_or_else(|| "iterable".to_string());
    let body_steps = node
        .child_by_field_name("body")
        .map(|b| extract_steps(b, source, exts))
        .unwrap_or_default();

    PseudocodeStep::Loop {
        kind: LoopKind::For { variable, iterable },
        body: body_steps,
    }
}

// -------------------------------------------------------------------
// Shared extractors
// -------------------------------------------------------------------

fn extract_assignment(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<PseudocodeStep> {
    let target = node
        .child_by_field_name("left")
        .and_then(|n| node_text(n, source))?;
    let rhs = node.child_by_field_name("right");
    let src = simplify_rhs(rhs, source);
    Some(PseudocodeStep::Assignment {
        target,
        source: src,
    })
}

fn extract_let_declaration(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<PseudocodeStep> {
    let target = node
        .child_by_field_name("pattern")
        .and_then(|n| node_text(n, source))?;
    let rhs = node.child_by_field_name("value");
    let src = simplify_rhs(rhs, source);
    Some(PseudocodeStep::Assignment {
        target,
        source: src,
    })
}

fn extract_js_declaration(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<PseudocodeStep> {
    // lexical_declaration contains variable_declarator children.
    let mut cursor = node.walk();
    for child in node.children(&mut cursor) {
        if child.kind() == "variable_declarator" {
            let target = child
                .child_by_field_name("name")
                .and_then(|n| node_text(n, source));
            let rhs = child.child_by_field_name("value");
            if let Some(t) = target {
                return Some(PseudocodeStep::Assignment {
                    target: t,
                    source: simplify_rhs(rhs, source),
                });
            }
        }
    }
    None
}

fn extract_expression_step(
    node: tree_sitter::Node,
    source: &[u8],
    is_python: bool,
) -> Option<PseudocodeStep> {
    // The expression_statement wraps one child expression.
    let expr = node.named_child(0)?;

    match expr.kind() {
        "call" | "call_expression" => extract_call(expr, source),
        "assignment" if is_python => extract_assignment(expr, source),
        "augmented_assignment" if is_python => {
            extract_assignment(expr, source)
        }
        "assignment_expression" => {
            // JS/TS: a = b
            let target = expr
                .child_by_field_name("left")
                .and_then(|n| node_text(n, source))?;
            let rhs = expr.child_by_field_name("right");
            Some(PseudocodeStep::Assignment {
                target,
                source: simplify_rhs(rhs, source),
            })
        }
        "yield" => {
            let value = first_child_text(expr, source);
            Some(PseudocodeStep::Yield { value })
        }
        "await_expression" => {
            // Unwrap: the interesting part is the awaited expression.
            let inner = expr.named_child(0)?;
            if inner.kind() == "call" || inner.kind() == "call_expression"
            {
                extract_call(inner, source)
            } else {
                None
            }
        }
        _ => None,
    }
}

fn extract_call(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<PseudocodeStep> {
    let func_node = node.child_by_field_name("function")?;
    let callee = simplify_callee(func_node, source)?;

    let args = node
        .child_by_field_name("arguments")
        .map(|a| extract_arg_names(a, source))
        .unwrap_or_default();

    Some(PseudocodeStep::Call { callee, args })
}

/// Simplify a callee expression: `self.method` -> `self.method`,
/// `module.func` -> `module.func`, plain `func` -> `func`.
fn simplify_callee(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<String> {
    node_text(node, source)
}

/// Extract argument names/placeholders from an argument list.
fn extract_arg_names(
    args_node: tree_sitter::Node,
    source: &[u8],
) -> Vec<String> {
    let mut names = Vec::new();
    let mut cursor = args_node.walk();

    for child in args_node.children(&mut cursor) {
        if child.is_named() {
            match child.kind() {
                "identifier" | "string" | "integer" | "float"
                | "true" | "false" | "none" | "null" => {
                    if let Some(t) = node_text(child, source) {
                        names.push(t);
                    }
                }
                _ => {
                    // Complex expression — just note its presence.
                    names.push("<expr>".to_string());
                }
            }
        }
    }
    names
}

/// Simplify a right-hand side expression for display.
fn simplify_rhs(
    rhs: Option<tree_sitter::Node>,
    source: &[u8],
) -> String {
    let Some(node) = rhs else {
        return "<unknown>".to_string();
    };

    simplify_node(node, source)
}

/// Classify a single AST node into a pseudocode-friendly placeholder.
fn simplify_node(
    node: tree_sitter::Node,
    source: &[u8],
) -> String {
    match node.kind() {
        "call" | "call_expression" => {
            let callee = node
                .child_by_field_name("function")
                .and_then(|n| node_text(n, source))
                .unwrap_or_else(|| "?".to_string());
            format!("{callee}(...)")
        }
        "identifier" => node_text(node, source)
            .unwrap_or_else(|| "<expr>".to_string()),
        "string" | "integer" | "float" | "true" | "false"
        | "none" | "null" | "string_literal"
        | "integer_literal" | "float_literal"
        | "boolean_literal" | "none_type" => {
            "<literal>".to_string()
        }
        "binary_operator" | "unary_operator"
        | "binary_expression" | "unary_expression"
        | "augmented_assignment_expression" => {
            "<arithmetic>".to_string()
        }
        "list_comprehension" => "[<comprehension>]".to_string(),
        "dictionary_comprehension"
        | "set_comprehension" => "{<comprehension>}".to_string(),
        "generator_expression" => {
            "(<comprehension>)".to_string()
        }
        _ => {
            // For complex expressions, use the raw text if short.
            let text = node_text(node, source)
                .unwrap_or_else(|| "<expr>".to_string());
            if text.len() <= 60 {
                text
            } else {
                "<expr>".to_string()
            }
        }
    }
}

// -------------------------------------------------------------------
// Formatting
// -------------------------------------------------------------------

fn format_step(
    step: &PseudocodeStep,
    indent: usize,
    lines: &mut Vec<String>,
) {
    let pad = " ".repeat(indent);
    match step {
        PseudocodeStep::Call { callee, args } => {
            if args.is_empty() {
                lines.push(format!("{pad}CALL {callee}()"));
            } else {
                let args_str = args.join(", ");
                lines.push(format!("{pad}CALL {callee}({args_str})"));
            }
        }
        PseudocodeStep::Conditional { branches } => {
            for (i, branch) in branches.iter().enumerate() {
                let keyword = if i == 0 {
                    "IF"
                } else if branch.condition.is_some() {
                    "ELIF"
                } else {
                    "ELSE"
                };
                match &branch.condition {
                    Some(cond) => {
                        lines.push(format!("{pad}{keyword} {cond}:"));
                    }
                    None => {
                        lines.push(format!("{pad}{keyword}:"));
                    }
                }
                for s in &branch.body {
                    format_step(s, indent + 2, lines);
                }
            }
        }
        PseudocodeStep::Loop { kind, body } => match kind {
            LoopKind::For { variable, iterable } => {
                lines.push(format!(
                    "{pad}FOR {variable} IN {iterable}:"
                ));
                for s in body {
                    format_step(s, indent + 2, lines);
                }
            }
            LoopKind::While { condition } => {
                lines.push(format!("{pad}WHILE {condition}:"));
                for s in body {
                    format_step(s, indent + 2, lines);
                }
            }
        },
        PseudocodeStep::Return { value } => match value {
            Some(v) => lines.push(format!("{pad}RETURN {v}")),
            None => lines.push(format!("{pad}RETURN")),
        },
        PseudocodeStep::Assignment { target, source } => {
            lines.push(format!("{pad}ASSIGN {target} = {source}"));
        }
        PseudocodeStep::Yield { value } => match value {
            Some(v) => lines.push(format!("{pad}YIELD {v}")),
            None => lines.push(format!("{pad}YIELD")),
        },
    }
}

// -------------------------------------------------------------------
// Utility helpers
// -------------------------------------------------------------------

fn node_text(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<String> {
    node.utf8_text(source).ok().map(|s| s.to_string())
}

/// Simplify the first named child of a node (used for return/yield
/// values).
fn first_child_text(
    node: tree_sitter::Node,
    source: &[u8],
) -> Option<String> {
    let child = node.named_child(0)?;
    Some(simplify_node(child, source))
}

/// Truncate steps to at most `max_lines`, keeping first and last
/// halves.
fn truncate_steps(
    steps: Vec<PseudocodeStep>,
    max_lines: usize,
) -> Vec<PseudocodeStep> {
    if max_lines == 0 || steps.len() <= max_lines {
        return steps;
    }
    let first_half = max_lines / 2;
    let last_half = max_lines - first_half;
    let skip = steps.len() - last_half;

    let mut result = Vec::with_capacity(max_lines);
    result.extend_from_slice(&steps[..first_half]);
    result.extend_from_slice(&steps[skip..]);
    result
}

fn hash_body(text: &str) -> u64 {
    let mut hasher =
        std::collections::hash_map::DefaultHasher::new();
    text.hash(&mut hasher);
    hasher.finish()
}

// -------------------------------------------------------------------
// Tests
// -------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::python_parser;
    use std::path::PathBuf;

    fn make_body(text: &str) -> FunctionBody {
        FunctionBody {
            entity_name: "test_func".to_string(),
            scope: None,
            body_text: text.to_string(),
            file: PathBuf::from("test.py"),
            start_line: 2,
            end_line: 10,
        }
    }

    #[test]
    fn empty_body() {
        let body = make_body("pass");
        let parser = python_parser();
        let pc =
            generate_pseudocode(&body, &parser, 100).unwrap();
        // `pass` is not a recognized step type, so no steps.
        assert!(pc.steps.is_empty());
    }

    #[test]
    fn python_return() {
        let body = make_body("return x + 1");
        let parser = python_parser();
        let pc =
            generate_pseudocode(&body, &parser, 100).unwrap();
        assert_eq!(pc.steps.len(), 1);
        match &pc.steps[0] {
            PseudocodeStep::Return { value } => {
                assert!(value.is_some());
            }
            other => panic!("Expected Return, got {:?}", other),
        }
    }

    #[test]
    fn python_assignment() {
        let body = make_body("x = compute()");
        let parser = python_parser();
        let pc =
            generate_pseudocode(&body, &parser, 100).unwrap();
        assert_eq!(pc.steps.len(), 1);
        match &pc.steps[0] {
            PseudocodeStep::Assignment { target, source } => {
                assert_eq!(target, "x");
                assert!(source.contains("compute"));
            }
            other => {
                panic!("Expected Assignment, got {:?}", other)
            }
        }
    }

    #[test]
    fn python_call() {
        let body = make_body("process(item)");
        let parser = python_parser();
        let pc =
            generate_pseudocode(&body, &parser, 100).unwrap();
        assert_eq!(pc.steps.len(), 1);
        match &pc.steps[0] {
            PseudocodeStep::Call { callee, args } => {
                assert_eq!(callee, "process");
                assert_eq!(args, &["item"]);
            }
            other => panic!("Expected Call, got {:?}", other),
        }
    }

    #[test]
    fn python_if_elif_else() {
        let body = make_body(
            "if x > 0:\n    return x\nelif x == 0:\n    return 0\nelse:\n    return -x",
        );
        let parser = python_parser();
        let pc =
            generate_pseudocode(&body, &parser, 100).unwrap();
        assert_eq!(pc.steps.len(), 1);
        match &pc.steps[0] {
            PseudocodeStep::Conditional { branches } => {
                assert_eq!(branches.len(), 3);
                // if branch has condition
                assert!(branches[0].condition.is_some());
                // elif branch has condition
                assert!(branches[1].condition.is_some());
                // else branch has no condition
                assert!(branches[2].condition.is_none());
                // Each branch has a return step
                assert_eq!(branches[0].body.len(), 1);
                assert_eq!(branches[1].body.len(), 1);
                assert_eq!(branches[2].body.len(), 1);
            }
            other => {
                panic!("Expected Conditional, got {:?}", other)
            }
        }
    }

    #[test]
    fn python_for_loop() {
        let body = make_body(
            "for item in items:\n    process(item)",
        );
        let parser = python_parser();
        let pc =
            generate_pseudocode(&body, &parser, 100).unwrap();
        assert_eq!(pc.steps.len(), 1);
        match &pc.steps[0] {
            PseudocodeStep::Loop { kind, body } => {
                match kind {
                    LoopKind::For {
                        variable,
                        iterable,
                    } => {
                        assert_eq!(variable, "item");
                        assert_eq!(iterable, "items");
                    }
                    other => {
                        panic!(
                            "Expected For loop kind, got {:?}",
                            other
                        )
                    }
                }
                assert_eq!(body.len(), 1);
                assert!(
                    matches!(&body[0], PseudocodeStep::Call { .. })
                );
            }
            other => panic!("Expected Loop, got {:?}", other),
        }
    }

    #[test]
    fn python_while_loop() {
        let body = make_body(
            "while not converged:\n    step()",
        );
        let parser = python_parser();
        let pc =
            generate_pseudocode(&body, &parser, 100).unwrap();
        assert_eq!(pc.steps.len(), 1);
        match &pc.steps[0] {
            PseudocodeStep::Loop { kind, body } => {
                match kind {
                    LoopKind::While { condition } => {
                        assert!(condition.contains("converged"));
                    }
                    other => {
                        panic!(
                            "Expected While loop kind, got {:?}",
                            other
                        )
                    }
                }
                assert_eq!(body.len(), 1);
            }
            other => panic!("Expected Loop, got {:?}", other),
        }
    }

    #[test]
    fn nested_control_flow() {
        // if inside for
        let body = make_body(
            "for item in items:\n    if item > 0:\n        process(item)",
        );
        let parser = python_parser();
        let pc =
            generate_pseudocode(&body, &parser, 100).unwrap();
        assert_eq!(pc.steps.len(), 1);
        match &pc.steps[0] {
            PseudocodeStep::Loop { body, .. } => {
                assert_eq!(body.len(), 1);
                assert!(matches!(
                    &body[0],
                    PseudocodeStep::Conditional { .. }
                ));
            }
            other => panic!("Expected Loop, got {:?}", other),
        }
    }

    #[test]
    fn format_produces_readable_text() {
        let pc = Pseudocode {
            entity_id: 0,
            steps: vec![
                PseudocodeStep::Call {
                    callee: "init".to_string(),
                    args: vec![],
                },
                PseudocodeStep::Conditional {
                    branches: vec![
                        ConditionalBranch {
                            condition: Some("x > 0".to_string()),
                            body: vec![PseudocodeStep::Return {
                                value: Some("x".to_string()),
                            }],
                        },
                        ConditionalBranch {
                            condition: None,
                            body: vec![PseudocodeStep::Return {
                                value: Some("-x".to_string()),
                            }],
                        },
                    ],
                },
                PseudocodeStep::Loop {
                    kind: LoopKind::For {
                        variable: "item".to_string(),
                        iterable: "items".to_string(),
                    },
                    body: vec![PseudocodeStep::Call {
                        callee: "process".to_string(),
                        args: vec!["item".to_string()],
                    }],
                },
            ],
            body_hash: 0,
            omitted_count: 0,
        };

        let text = format_pseudocode(&pc);
        assert!(text.contains("CALL init()"));
        assert!(text.contains("IF x > 0:"));
        assert!(text.contains("  RETURN x"));
        assert!(text.contains("ELSE:"));
        assert!(text.contains("  RETURN -x"));
        assert!(text.contains("FOR item IN items:"));
        assert!(text.contains("  CALL process(item)"));
    }

    #[test]
    fn truncation_at_max_lines() {
        let steps: Vec<PseudocodeStep> = (0..10)
            .map(|i| PseudocodeStep::Return {
                value: Some(format!("{i}")),
            })
            .collect();

        let truncated = truncate_steps(steps, 4);
        assert_eq!(truncated.len(), 4);

        // First 2 from beginning
        match &truncated[0] {
            PseudocodeStep::Return { value } => {
                assert_eq!(value.as_deref(), Some("0"));
            }
            _ => panic!("Wrong step"),
        }
        match &truncated[1] {
            PseudocodeStep::Return { value } => {
                assert_eq!(value.as_deref(), Some("1"));
            }
            _ => panic!("Wrong step"),
        }
        // Last 2 from end
        match &truncated[2] {
            PseudocodeStep::Return { value } => {
                assert_eq!(value.as_deref(), Some("8"));
            }
            _ => panic!("Wrong step"),
        }
        match &truncated[3] {
            PseudocodeStep::Return { value } => {
                assert_eq!(value.as_deref(), Some("9"));
            }
            _ => panic!("Wrong step"),
        }
    }

    #[test]
    fn body_hash_deterministic() {
        let h1 = hash_body("x = 1");
        let h2 = hash_body("x = 1");
        let h3 = hash_body("x = 2");
        assert_eq!(h1, h2);
        assert_ne!(h1, h3);
    }
}

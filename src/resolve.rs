use crate::types::ImportStatement;
use std::path::{Path, PathBuf};

/// Resolve `resolved_file` for each import by probing the filesystem.
///
/// For imports within the repo, sets `resolved_file` to the absolute path
/// of the target file. For external packages (e.g. `torch`, `react`) that
/// cannot be found within `repo_root`, leaves `resolved_file` as `None`.
pub fn resolve_imports(imports: &mut Vec<ImportStatement>, repo_root: &Path) {
    for import in imports.iter_mut() {
        import.resolved_file = resolve_module(
            &import.source_module,
            &import.file,
            repo_root,
        );
    }
}

fn resolve_module(
    source_module: &str,
    caller_file: &Path,
    repo_root: &Path,
) -> Option<PathBuf> {
    let ext = caller_file.extension().and_then(|e| e.to_str()).unwrap_or("");
    match ext {
        "py" => resolve_python(source_module, caller_file, repo_root),
        "ts" | "tsx" | "js" | "jsx" => resolve_js(source_module, caller_file),
        "rs" => resolve_rust(source_module, repo_root),
        _ => None,
    }
}

fn resolve_python(
    source_module: &str,
    caller_file: &Path,
    repo_root: &Path,
) -> Option<PathBuf> {
    let module_path = source_module.replace('.', "/");
    let caller_dir = caller_file.parent().unwrap_or(repo_root);

    // Try relative to caller directory first
    for candidate in python_candidates(caller_dir, &module_path) {
        if candidate.exists() {
            return Some(candidate);
        }
    }

    // Try from repo root
    for candidate in python_candidates(repo_root, &module_path) {
        if candidate.exists() {
            return Some(candidate);
        }
    }

    None
}

fn python_candidates(base: &Path, module_path: &str) -> [PathBuf; 2] {
    [
        base.join(format!("{module_path}.py")),
        base.join(module_path).join("__init__.py"),
    ]
}

fn resolve_js(source_module: &str, caller_file: &Path) -> Option<PathBuf> {
    // Only resolve relative imports; bare module names are external
    if !source_module.starts_with("./") && !source_module.starts_with("../") {
        return None;
    }
    let caller_dir = caller_file.parent()?;
    let base = caller_dir.join(source_module);

    for candidate in js_candidates(&base) {
        if candidate.exists() {
            return Some(candidate);
        }
    }
    None
}

fn js_candidates(base: &Path) -> [PathBuf; 6] {
    [
        base.with_extension("ts"),
        base.with_extension("tsx"),
        base.with_extension("js"),
        base.with_extension("jsx"),
        base.join("index.ts"),
        base.join("index.js"),
    ]
}

fn resolve_rust(source_module: &str, repo_root: &Path) -> Option<PathBuf> {
    // Strip standard leading segment qualifiers
    let stripped = source_module
        .trim_start_matches("crate::")
        .trim_start_matches("super::")
        .trim_start_matches("self::");

    let rel_path = stripped.replace("::", "/");
    let src_dir = repo_root.join("src");

    let as_file = src_dir.join(format!("{rel_path}.rs"));
    if as_file.exists() {
        return Some(as_file);
    }
    let as_mod = src_dir.join(&rel_path).join("mod.rs");
    if as_mod.exists() {
        return Some(as_mod);
    }
    None
}

// =========================================================================
// Tests
// =========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    fn make_temp_repo() -> TempDir {
        tempfile::tempdir().expect("tempdir")
    }

    #[test]
    fn test_python_relative_resolution() {
        let dir = make_temp_repo();
        let repo_root = dir.path();

        // Create neurite/utils.py
        fs::create_dir_all(repo_root.join("neurite")).unwrap();
        fs::write(repo_root.join("neurite/utils.py"), "").unwrap();

        let caller = repo_root.join("train.py");
        let mut imports = vec![ImportStatement {
            source_module: "neurite.utils".to_string(),
            imported_symbols: vec!["resize".to_string()],
            alias: None,
            resolved_file: None,
            file: caller,
            line: 1,
        }];
        resolve_imports(&mut imports, repo_root);

        assert_eq!(
            imports[0].resolved_file,
            Some(repo_root.join("neurite/utils.py"))
        );
    }

    #[test]
    fn test_python_package_init_resolution() {
        let dir = make_temp_repo();
        let repo_root = dir.path();

        fs::create_dir_all(repo_root.join("mypkg")).unwrap();
        fs::write(repo_root.join("mypkg/__init__.py"), "").unwrap();

        let caller = repo_root.join("main.py");
        let mut imports = vec![ImportStatement {
            source_module: "mypkg".to_string(),
            imported_symbols: vec![],
            alias: None,
            resolved_file: None,
            file: caller,
            line: 1,
        }];
        resolve_imports(&mut imports, repo_root);

        assert_eq!(
            imports[0].resolved_file,
            Some(repo_root.join("mypkg/__init__.py"))
        );
    }

    #[test]
    fn test_external_package_resolves_to_none() {
        let dir = make_temp_repo();
        let repo_root = dir.path();
        let caller = repo_root.join("train.py");

        let mut imports = vec![ImportStatement {
            source_module: "torch".to_string(),
            imported_symbols: vec![],
            alias: Some("torch".to_string()),
            resolved_file: None,
            file: caller,
            line: 1,
        }];
        resolve_imports(&mut imports, repo_root);

        assert!(imports[0].resolved_file.is_none());
    }

    #[test]
    fn test_js_relative_resolution() {
        let dir = make_temp_repo();
        let repo_root = dir.path();

        fs::write(repo_root.join("utils.ts"), "").unwrap();

        let caller = repo_root.join("app.ts");
        let mut imports = vec![ImportStatement {
            source_module: "./utils".to_string(),
            imported_symbols: vec!["helper".to_string()],
            alias: None,
            resolved_file: None,
            file: caller,
            line: 1,
        }];
        resolve_imports(&mut imports, repo_root);

        assert_eq!(
            imports[0].resolved_file,
            Some(repo_root.join("utils.ts"))
        );
    }

    #[test]
    fn test_rust_module_resolution() {
        let dir = make_temp_repo();
        let repo_root = dir.path();

        fs::create_dir_all(repo_root.join("src")).unwrap();
        fs::write(repo_root.join("src/parser.rs"), "").unwrap();

        let caller = repo_root.join("src/main.rs");
        let mut imports = vec![ImportStatement {
            source_module: "crate::parser".to_string(),
            imported_symbols: vec!["PythonParser".to_string()],
            alias: None,
            resolved_file: None,
            file: caller,
            line: 1,
        }];
        resolve_imports(&mut imports, repo_root);

        assert_eq!(
            imports[0].resolved_file,
            Some(repo_root.join("src/parser.rs"))
        );
    }
}

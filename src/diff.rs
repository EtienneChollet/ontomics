use crate::analyzer;
use crate::parser;
use crate::types::{Concept, ConceptDelta, OntologyDiff, ParseResult};
use anyhow::{Context, Result};
use std::collections::{HashMap, HashSet};
use std::path::Path;

/// Compute ontology diff between a base git ref and the current concept graph.
///
/// Resolves `base_ref` in the git repo at `repo_root`, parses all Python files
/// from that commit's tree, runs analysis, and compares against
/// `current_concepts` (the HEAD state already built by the server).
pub fn ontology_diff(
    repo_root: &Path,
    base_ref: &str,
    current_concepts: &HashMap<u64, Concept>,
) -> Result<OntologyDiff> {
    let repo = git2::Repository::open(repo_root)
        .context("failed to open git repository")?;

    let base_obj = repo
        .revparse_single(base_ref)
        .with_context(|| format!("failed to resolve ref '{base_ref}'"))?;
    let base_commit = base_obj
        .peel_to_commit()
        .with_context(|| format!("'{base_ref}' does not resolve to a commit"))?;
    let base_tree = base_commit.tree()?;

    let base_parse_results = parse_git_tree(&repo, &base_tree, repo_root)?;
    let base_analysis = analyzer::analyze(&base_parse_results, &analyzer::AnalysisParams::default())?;
    let base_concepts: HashMap<u64, Concept> = base_analysis
        .concepts
        .into_iter()
        .map(|c| (c.id, c))
        .collect();

    let mut added = Vec::new();
    let mut removed = Vec::new();
    let mut changed = Vec::new();

    for (id, concept) in current_concepts {
        if !base_concepts.contains_key(id) {
            added.push(concept.clone());
        }
    }

    for (id, concept) in &base_concepts {
        if !current_concepts.contains_key(id) {
            removed.push(concept.clone());
        }
    }

    for (id, current) in current_concepts {
        let base = match base_concepts.get(id) {
            Some(b) => b,
            None => continue,
        };

        let freq_change =
            current.occurrences.len() as i64 - base.occurrences.len() as i64;

        let current_variants: HashSet<&str> = current
            .occurrences
            .iter()
            .map(|o| o.identifier.as_str())
            .collect();
        let base_variants: HashSet<&str> = base
            .occurrences
            .iter()
            .map(|o| o.identifier.as_str())
            .collect();

        let new_variants: Vec<String> = current_variants
            .difference(&base_variants)
            .map(|s| s.to_string())
            .collect();
        let removed_variants: Vec<String> = base_variants
            .difference(&current_variants)
            .map(|s| s.to_string())
            .collect();

        if freq_change != 0
            || !new_variants.is_empty()
            || !removed_variants.is_empty()
        {
            changed.push(ConceptDelta {
                concept: current.clone(),
                frequency_change: freq_change,
                new_variants,
                removed_variants,
            });
        }
    }

    Ok(OntologyDiff {
        base_ref: base_ref.to_string(),
        head_ref: "HEAD".to_string(),
        added_concepts: added,
        removed_concepts: removed,
        changed_concepts: changed,
    })
}

/// Skip directories that are not source code.
const SKIP_DIRS: &[&str] = &[
    "__pycache__",
    ".git/",
    "venv/",
    ".venv/",
    "node_modules/",
    "site-packages/",
    ".tox/",
    ".eggs/",
    "egg-info/",
];

/// Parse all Python files from a git tree object (without touching the
/// working directory).
fn parse_git_tree(
    repo: &git2::Repository,
    tree: &git2::Tree,
    repo_root: &Path,
) -> Result<Vec<ParseResult>> {
    let mut results = Vec::new();

    tree.walk(git2::TreeWalkMode::PreOrder, |dir, entry| {
        let name = match entry.name() {
            Some(n) => n,
            None => return git2::TreeWalkResult::Ok,
        };

        if !name.ends_with(".py") {
            return git2::TreeWalkResult::Ok;
        }

        if SKIP_DIRS.iter().any(|skip| dir.contains(skip)) {
            return git2::TreeWalkResult::Ok;
        }

        if entry.kind() != Some(git2::ObjectType::Blob) {
            return git2::TreeWalkResult::Ok;
        }

        let blob = match repo.find_blob(entry.id()) {
            Ok(b) => b,
            Err(_) => return git2::TreeWalkResult::Ok,
        };

        let content = match std::str::from_utf8(blob.content()) {
            Ok(s) => s,
            // Binary or non-utf8 file, skip
            Err(_) => return git2::TreeWalkResult::Ok,
        };

        let file_path = repo_root.join(dir).join(name);
        if let Ok(parse_result) = parser::parse_content(content, &file_path) {
            results.push(parse_result);
        }

        git2::TreeWalkResult::Ok
    })?;

    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::path::PathBuf;

    /// Create a temp git repo with an initial commit containing `files`,
    /// returning the repo path.
    fn make_test_repo(
        test_name: &str,
        files: &[(&str, &str)],
    ) -> (PathBuf, git2::Repository) {
        let dir = std::env::temp_dir().join(format!(
            "semex_diff_test_{}_{}",
            std::process::id(),
            test_name,
        ));
        if dir.exists() {
            fs::remove_dir_all(&dir).unwrap();
        }
        fs::create_dir_all(&dir).unwrap();

        let repo = git2::Repository::init(&dir).unwrap();
        let mut index = repo.index().unwrap();

        for (name, content) in files {
            let file_path = dir.join(name);
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&file_path, content).unwrap();
            index.add_path(Path::new(name)).unwrap();
        }

        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        {
            let tree = repo.find_tree(tree_id).unwrap();
            let sig =
                git2::Signature::now("test", "test@test.com").unwrap();
            repo.commit(
                Some("HEAD"),
                &sig,
                &sig,
                "initial commit",
                &tree,
                &[],
            )
            .unwrap();
        }

        (dir, repo)
    }

    fn commit_files(
        repo: &git2::Repository,
        dir: &Path,
        files: &[(&str, &str)],
        message: &str,
    ) {
        let mut index = repo.index().unwrap();
        for (name, content) in files {
            let file_path = dir.join(name);
            if let Some(parent) = file_path.parent() {
                fs::create_dir_all(parent).unwrap();
            }
            fs::write(&file_path, content).unwrap();
            index.add_path(Path::new(name)).unwrap();
        }
        index.write().unwrap();
        let tree_id = index.write_tree().unwrap();
        let tree = repo.find_tree(tree_id).unwrap();
        let sig = git2::Signature::now("test", "test@test.com").unwrap();
        let head = repo.head().unwrap().peel_to_commit().unwrap();
        repo.commit(
            Some("HEAD"),
            &sig,
            &sig,
            message,
            &tree,
            &[&head],
        )
        .unwrap();
    }

    #[test]
    fn test_no_diff_same_content() {
        let base_py = r#"
def spatial_transform(vol, trf):
    return vol

def spatial_transform(vol, trf):
    return vol
"#;
        let (dir, _repo) =
            make_test_repo("no_diff", &[("module.py", base_py)]);

        // Parse HEAD to get current concepts
        let parse_results = parser::parse_content(
            base_py,
            &dir.join("module.py"),
        )
        .unwrap();
        let analysis = analyzer::analyze(&[parse_results], &analyzer::AnalysisParams::default()).unwrap();
        let current: HashMap<u64, Concept> = analysis
            .concepts
            .into_iter()
            .map(|c| (c.id, c))
            .collect();

        let diff = ontology_diff(&dir, "HEAD", &current).unwrap();
        assert!(diff.added_concepts.is_empty());
        assert!(diff.removed_concepts.is_empty());
        assert!(diff.changed_concepts.is_empty());

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_added_concept() {
        let base_py = r#"
def spatial_transform(vol, trf):
    return vol

def spatial_transform(vol, trf):
    return vol
"#;
        let (dir, repo) =
            make_test_repo("added", &[("module.py", base_py)]);

        // Add a new concept in a second commit
        let head_py = r#"
def spatial_transform(vol, trf):
    return vol

def spatial_transform(vol, trf):
    return vol

def compute_displacement(source, target):
    displacement = source - target
    return displacement
"#;
        commit_files(&repo, &dir, &[("module.py", head_py)], "add displacement");

        // Parse HEAD to get current concepts
        let parse_results = parser::parse_content(
            head_py,
            &dir.join("module.py"),
        )
        .unwrap();
        let analysis = analyzer::analyze(&[parse_results], &analyzer::AnalysisParams::default()).unwrap();
        let current: HashMap<u64, Concept> = analysis
            .concepts
            .into_iter()
            .map(|c| (c.id, c))
            .collect();

        let diff = ontology_diff(&dir, "HEAD~1", &current).unwrap();

        // "displacement" should be added (appears in HEAD but not in base)
        let added_names: Vec<&str> = diff
            .added_concepts
            .iter()
            .map(|c| c.canonical.as_str())
            .collect();
        assert!(
            added_names.contains(&"displacement"),
            "expected 'displacement' in added, got: {added_names:?}"
        );

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_removed_concept() {
        let base_py = r#"
def spatial_transform(vol, trf):
    return vol

def spatial_transform(vol, trf):
    return vol

def compute_displacement(source, target):
    displacement = source - target
    return displacement
"#;
        let (dir, repo) =
            make_test_repo("removed", &[("module.py", base_py)]);

        // Remove displacement in second commit
        let head_py = r#"
def spatial_transform(vol, trf):
    return vol

def spatial_transform(vol, trf):
    return vol
"#;
        commit_files(
            &repo,
            &dir,
            &[("module.py", head_py)],
            "remove displacement",
        );

        let parse_results = parser::parse_content(
            head_py,
            &dir.join("module.py"),
        )
        .unwrap();
        let analysis = analyzer::analyze(&[parse_results], &analyzer::AnalysisParams::default()).unwrap();
        let current: HashMap<u64, Concept> = analysis
            .concepts
            .into_iter()
            .map(|c| (c.id, c))
            .collect();

        let diff = ontology_diff(&dir, "HEAD~1", &current).unwrap();

        let removed_names: Vec<&str> = diff
            .removed_concepts
            .iter()
            .map(|c| c.canonical.as_str())
            .collect();
        assert!(
            removed_names.contains(&"displacement"),
            "expected 'displacement' in removed, got: {removed_names:?}"
        );

        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn test_invalid_ref() {
        let base_py = "x = 1\n";
        let (dir, _repo) =
            make_test_repo("invalid_ref", &[("module.py", base_py)]);

        let result = ontology_diff(&dir, "nonexistent_ref", &HashMap::new());
        assert!(result.is_err());

        fs::remove_dir_all(&dir).unwrap();
    }
}

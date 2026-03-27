use crate::graph::ConceptGraph;
use anyhow::Result;
use std::path::{Path, PathBuf};

pub struct IndexCache {
    db_path: PathBuf,
}

impl IndexCache {
    /// Open or create cache at `<repo_root>/.semex/index.db`.
    pub fn open(repo_root: &Path) -> Result<Self> {
        todo!()
    }

    /// Save full graph to cache.
    pub fn save(&self, graph: &ConceptGraph) -> Result<()> {
        todo!()
    }

    /// Load cached graph. Returns None if cache is missing or stale.
    pub fn load(&self) -> Result<Option<ConceptGraph>> {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_placeholder() {
        // Will be replaced with real tests
    }
}

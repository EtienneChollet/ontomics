use crate::embeddings::EmbeddingIndex;
use crate::graph::ConceptGraph;
use crate::types::{Concept, Convention, Relationship};
use anyhow::Result;
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

pub struct IndexCache {
    db_path: PathBuf,
}

/// Serializable proxy for ConceptGraph (which may not derive Serialize).
#[derive(Serialize, Deserialize)]
struct CachedGraph {
    concepts: HashMap<u64, Concept>,
    relationships: Vec<Relationship>,
    conventions: Vec<Convention>,
    embeddings: EmbeddingIndex,
}

impl IndexCache {
    /// Open or create cache at `<repo_root>/.semex/index.db`.
    pub fn open(repo_root: &Path) -> Result<Self> {
        let semex_dir = repo_root.join(".semex");
        std::fs::create_dir_all(&semex_dir)?;

        let db_path = semex_dir.join("index.db");
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS cache (
                key TEXT PRIMARY KEY,
                value BLOB,
                updated_at TEXT
            );",
        )?;

        Ok(Self { db_path })
    }

    /// Save full graph to cache.
    pub fn save(&self, graph: &ConceptGraph) -> Result<()> {
        let cached = CachedGraph {
            concepts: graph.concepts.clone(),
            relationships: graph.relationships.clone(),
            conventions: graph.conventions.clone(),
            embeddings: serde_json::from_value(serde_json::to_value(&graph.embeddings)?)?,
        };
        let json = serde_json::to_vec(&cached)?;

        let conn = Connection::open(&self.db_path)?;
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, updated_at)
             VALUES (?1, ?2, datetime('now'))",
            rusqlite::params!["graph", json],
        )?;

        Ok(())
    }

    /// Start watching for Python file changes. Returns the watcher (must be
    /// kept alive) and a receiver that emits batches of changed `.py` paths.
    pub fn watch(
        &self,
        root: &Path,
    ) -> Result<(RecommendedWatcher, mpsc::Receiver<Vec<PathBuf>>)> {
        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(
            move |res: std::result::Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    let py_paths: Vec<PathBuf> = event
                        .paths
                        .into_iter()
                        .filter(|p| {
                            p.extension().is_some_and(|e| e == "py")
                        })
                        .collect();
                    if !py_paths.is_empty() {
                        let _ = tx.send(py_paths);
                    }
                }
            },
        )?;
        watcher.watch(root, RecursiveMode::Recursive)?;
        Ok((watcher, rx))
    }

    /// Load cached graph. Returns None if cache is missing or corrupt.
    pub fn load(&self) -> Result<Option<ConceptGraph>> {
        if !self.db_path.exists() {
            return Ok(None);
        }

        let conn = Connection::open(&self.db_path)?;
        let result: rusqlite::Result<Vec<u8>> = conn.query_row(
            "SELECT value FROM cache WHERE key = ?1",
            rusqlite::params!["graph"],
            |row| row.get(0),
        );

        let blob = match result {
            Ok(b) => b,
            Err(rusqlite::Error::QueryReturnedNoRows) => return Ok(None),
            Err(_) => return Ok(None),
        };

        let cached: CachedGraph = match serde_json::from_slice(&blob) {
            Ok(c) => c,
            Err(_) => return Ok(None), // corrupt cache — caller rebuilds
        };

        Ok(Some(ConceptGraph {
            concepts: cached.concepts,
            relationships: cached.relationships,
            conventions: cached.conventions,
            embeddings: cached.embeddings,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embeddings::EmbeddingIndex;
    use crate::graph::ConceptGraph;
    use crate::types::Concept;
    use std::collections::{HashMap, HashSet};

    fn make_test_graph() -> ConceptGraph {
        let mut concepts = HashMap::new();
        concepts.insert(
            1,
            Concept {
                id: 1,
                canonical: "transform".to_string(),
                subtokens: vec!["transform".to_string()],
                occurrences: vec![],
                entity_types: HashSet::new(),
                embedding: None,
            },
        );
        ConceptGraph {
            concepts,
            relationships: Vec::new(),
            conventions: Vec::new(),
            embeddings: EmbeddingIndex::empty(),
        }
    }

    #[test]
    fn test_cache_roundtrip() {
        let dir = std::path::Path::new("/tmp/semex_test_cache");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let cache = IndexCache::open(dir).unwrap();
        let graph = make_test_graph();
        cache.save(&graph).unwrap();

        let loaded = cache.load().unwrap();
        assert!(loaded.is_some());
        let loaded = loaded.unwrap();
        assert!(loaded.concepts.contains_key(&1));
        assert_eq!(loaded.concepts[&1].canonical, "transform");

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_cache_load_missing() {
        let dir = std::path::Path::new("/tmp/semex_test_cache_missing");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let cache = IndexCache::open(dir).unwrap();
        let loaded = cache.load().unwrap();
        assert!(loaded.is_none());

        let _ = std::fs::remove_dir_all(dir);
    }
}

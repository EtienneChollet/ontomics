use crate::embeddings::EmbeddingIndex;
use crate::graph::ConceptGraph;
use crate::types::{
    CallSite, ClassInfo, Concept, Convention, Relationship, Signature,
};
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
    #[serde(default)]
    signatures: Vec<Signature>,
    #[serde(default)]
    classes: Vec<ClassInfo>,
    #[serde(default)]
    call_sites: Vec<CallSite>,
}

impl IndexCache {
    /// Open or create cache at `<repo_root>/.semex/index.db`.
    pub fn open(repo_root: &Path) -> Result<Self> {
        let semex_dir = repo_root.join(".semex");
        std::fs::create_dir_all(&semex_dir)?;

        let db_path = semex_dir.join("index.db");
        {
            let conn = Connection::open(&db_path)?;
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    updated_at TEXT
                );",
            )?;
            // Explicitly drop before proceeding
        }

        Ok(Self { db_path })
    }

    /// Save full graph to cache.
    pub fn save(&self, graph: &ConceptGraph) -> Result<()> {
        let cached = CachedGraph {
            concepts: graph.concepts.clone(),
            relationships: graph.relationships.clone(),
            conventions: graph.conventions.clone(),
            embeddings: serde_json::from_value(serde_json::to_value(&graph.embeddings)?)?,
            signatures: graph.signatures.clone(),
            classes: graph.classes.clone(),
            call_sites: graph.call_sites.clone(),
        };
        let json = serde_json::to_vec(&cached)?;

        let conn = Connection::open(&self.db_path)?;
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, updated_at)
             VALUES (?1, ?2, datetime('now'))",
            rusqlite::params!["graph", &json],
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
            signatures: cached.signatures,
            classes: cached.classes,
            call_sites: cached.call_sites,
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
                subconcepts: Vec::new(),
            },
        );
        ConceptGraph {
            concepts,
            relationships: Vec::new(),
            conventions: Vec::new(),
            embeddings: EmbeddingIndex::empty(),
            signatures: Vec::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
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
    fn test_sqlite_blob_roundtrip_sizes() {
        let dir = std::path::Path::new("/tmp/semex_test_blob_sizes");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let sizes: &[usize] = &[
            100_000,   // 100KB
            500_000,   // 500KB
            1_000_000, // 1MB
            1_500_000, // 1.5MB
            1_700_000, // 1.7MB — the size that failed in practice
            2_000_000, // 2MB
        ];

        for &size in sizes {
            let db_path = dir.join(format!("test_{size}.db"));
            let _ = std::fs::remove_file(&db_path);

            // Write in one connection scope
            {
                let conn = Connection::open(&db_path).unwrap();
                conn.execute_batch(
                    "CREATE TABLE cache (
                        key TEXT PRIMARY KEY,
                        value BLOB
                    );",
                )
                .unwrap();

                let blob = vec![0x42u8; size];
                conn.execute(
                    "INSERT INTO cache (key, value) VALUES (?1, ?2)",
                    rusqlite::params!["data", &blob],
                )
                .unwrap();
            } // conn dropped

            // Read from a NEW connection
            {
                let conn = Connection::open(&db_path).unwrap();
                let stored_len: usize = conn
                    .query_row(
                        "SELECT length(value) FROM cache WHERE key = ?1",
                        rusqlite::params!["data"],
                        |row| row.get(0),
                    )
                    .unwrap();

                assert_eq!(
                    stored_len, size,
                    "Blob size mismatch: wrote {size}, read {stored_len}"
                );
            }

            let _ = std::fs::remove_file(&db_path);
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_sqlite_save_load_pattern() {
        let dir = std::path::Path::new("/tmp/semex_test_save_pattern");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let db_path = dir.join(".semex").join("index.db");
        std::fs::create_dir_all(db_path.parent().unwrap()).unwrap();

        // Step 1: open() — create table
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute_batch(
                "CREATE TABLE IF NOT EXISTS cache (
                    key TEXT PRIMARY KEY,
                    value BLOB,
                    updated_at TEXT
                );",
            )
            .unwrap();
        }

        // Step 2: save() — open new conn, INSERT OR REPLACE
        let blob_size = 1_700_000usize;
        let blob = vec![0x42u8; blob_size];
        {
            let conn = Connection::open(&db_path).unwrap();
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, updated_at)
                 VALUES (?1, ?2, datetime('now'))",
                rusqlite::params!["graph", &blob],
            )
            .unwrap();
        }

        // Step 3: load() — open new conn, SELECT
        {
            let conn = Connection::open(&db_path).unwrap();
            let stored: Vec<u8> = conn
                .query_row(
                    "SELECT value FROM cache WHERE key = ?1",
                    rusqlite::params!["graph"],
                    |row| row.get(0),
                )
                .unwrap();

            let journal = db_path.with_extension("db-journal");
            let wal = db_path.with_extension("db-wal");
            eprintln!(
                "save_load_pattern: wrote={}, read={}, journal={}, wal={}",
                blob_size,
                stored.len(),
                journal.exists(),
                wal.exists(),
            );

            assert_eq!(
                stored.len(),
                blob_size,
                "save/load pattern: wrote {blob_size}, read {}",
                stored.len(),
            );
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Test with a realistic-sized graph that has L2 data and
    /// multiple relationship types — mirrors the actual build_graph output.
    #[test]
    fn test_cache_roundtrip_realistic() {
        use crate::types::{
            CallSite, ClassInfo, Occurrence, Relationship, RelationshipKind,
            Signature, Param,
        };

        let dir = std::path::Path::new("/tmp/semex_test_cache_realistic");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        // Build a graph with multiple relationship types and L2 data
        let mut concepts = HashMap::new();
        for i in 0..50 {
            let mut occs = Vec::new();
            for j in 0..20 {
                occs.push(Occurrence {
                    file: std::path::PathBuf::from(format!("file_{j}.py")),
                    line: j + 1,
                    identifier: format!("id_{i}_{j}"),
                    entity_type: crate::types::EntityType::Function,
                });
            }
            concepts.insert(
                i as u64,
                Concept {
                    id: i as u64,
                    canonical: format!("concept_{i}"),
                    subtokens: vec![format!("concept_{i}")],
                    occurrences: occs,
                    entity_types: HashSet::new(),
                    embedding: None,
                    subconcepts: Vec::new(),
                },
            );
        }

        let mut relationships = Vec::new();
        // Add CoOccurs
        for i in 0..40 {
            relationships.push(Relationship {
                source: i,
                target: i + 1,
                kind: RelationshipKind::CoOccurs,
                weight: 1.0,
            });
        }
        // Add SimilarTo
        for i in 0..10 {
            relationships.push(Relationship {
                source: i,
                target: i + 10,
                kind: RelationshipKind::SimilarTo,
                weight: 0.8,
            });
        }
        // Add Contrastive
        relationships.push(Relationship {
            source: 0,
            target: 1,
            kind: RelationshipKind::Contrastive,
            weight: 1.0,
        });

        let signatures: Vec<Signature> = (0..20)
            .map(|i| Signature {
                name: format!("func_{i}"),
                params: vec![
                    Param { name: "x".into(), type_annotation: Some("Tensor".into()), default: None },
                    Param { name: "y".into(), type_annotation: None, default: Some("3".into()) },
                ],
                return_type: Some("Tensor".into()),
                decorators: Vec::new(),
                docstring_first_line: Some(format!("Doc for func_{i}")),
                file: std::path::PathBuf::from(format!("file_{i}.py")),
                line: i + 1,
                scope: None,
            })
            .collect();

        let classes: Vec<ClassInfo> = (0..5)
            .map(|i| ClassInfo {
                name: format!("Class_{i}"),
                bases: vec!["Module".into()],
                methods: vec!["forward".into(), "__init__".into()],
                attributes: vec!["weight".into()],
                docstring_first_line: Some(format!("Class {i}")),
                file: std::path::PathBuf::from(format!("cls_{i}.py")),
                line: 1,
            })
            .collect();

        let call_sites: Vec<CallSite> = (0..30)
            .map(|i| CallSite {
                caller_scope: Some(format!("caller_{i}")),
                callee: format!("func_{}", i % 20),
                file: std::path::PathBuf::from("calls.py"),
                line: i + 1,
            })
            .collect();

        let graph = ConceptGraph {
            concepts,
            relationships,
            conventions: Vec::new(),
            embeddings: EmbeddingIndex::empty(),
            signatures,
            classes,
            call_sites,
        };

        let cache = IndexCache::open(dir).unwrap();
        cache.save(&graph).unwrap();

        // Check for leftover journal/wal files
        let semex_dir = dir.join(".semex");
        let journal = semex_dir.join("index.db-journal");
        let wal = semex_dir.join("index.db-wal");
        assert!(
            !journal.exists(),
            "Journal file should not exist after save"
        );
        assert!(
            !wal.exists(),
            "WAL file should not exist after save"
        );

        // Load from cache
        let loaded = cache.load().unwrap().expect("cache should load");

        // Verify all data roundtripped
        assert_eq!(loaded.concepts.len(), 50);
        assert_eq!(loaded.signatures.len(), 20);
        assert_eq!(loaded.classes.len(), 5);
        assert_eq!(loaded.call_sites.len(), 30);

        // Verify relationship types survived
        let co_occurs = loaded
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::CoOccurs)
            .count();
        let similar = loaded
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::SimilarTo)
            .count();
        let contrastive = loaded
            .relationships
            .iter()
            .filter(|r| r.kind == RelationshipKind::Contrastive)
            .count();

        assert_eq!(co_occurs, 40, "CoOccurs edges lost");
        assert_eq!(similar, 10, "SimilarTo edges lost");
        assert_eq!(contrastive, 1, "Contrastive edges lost");

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

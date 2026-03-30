use crate::embeddings::EmbeddingIndex;
use crate::graph::ConceptGraph;
use crate::types::{
    CallSite, ClassInfo, Concept, Convention, Entity, Relationship, Signature,
};
use anyhow::{Context, Result};
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

/// Auto-generated hash of indexing source files. Changes whenever
/// entity.rs, analyzer.rs, parser.rs, cache.rs, tokenizer.rs, or types.rs
/// are modified, triggering automatic re-indexing of stale caches.
const CACHE_VERSION: &str = env!("SEMEX_CACHE_VERSION");

pub struct IndexCache {
    db_path: PathBuf,
}

/// Serializable proxy for ConceptGraph.
/// `#[serde(default)]` on the struct ensures old caches missing new
/// fields deserialize gracefully (fields default to empty/zero).
#[derive(Default, Serialize, Deserialize)]
#[serde(default)]
struct CachedGraph {
    concepts: HashMap<u64, Concept>,
    relationships: Vec<Relationship>,
    conventions: Vec<Convention>,
    embeddings: EmbeddingIndex,
    signatures: Vec<Signature>,
    classes: Vec<ClassInfo>,
    call_sites: Vec<CallSite>,
    entities: HashMap<u64, Entity>,
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

    fn lock_path(&self) -> PathBuf {
        self.db_path.with_extension("lock")
    }

    /// Acquire an exclusive lock file. Returns an error if another
    /// instance holds the lock (stale locks older than 5 minutes are
    /// broken automatically).
    fn acquire_lock(&self) -> Result<()> {
        let lock = self.lock_path();
        if lock.exists() {
            let metadata = fs::metadata(&lock)?;
            let age = metadata
                .modified()?
                .elapsed()
                .unwrap_or_default();
            if age.as_secs() < 300 {
                anyhow::bail!(
                    "another semex instance holds the cache lock \
                     (pid in {:?}, age {:?})",
                    lock,
                    age,
                );
            }
            eprintln!(
                "Warning: breaking stale cache lock ({:?} old)",
                age,
            );
        }
        fs::write(
            &lock,
            format!("{}", std::process::id()),
        )?;
        Ok(())
    }

    fn release_lock(&self) {
        let _ = fs::remove_file(self.lock_path());
    }

    // --- Embedding lock: long-lived, held for the entire background
    //     embedding run to prevent duplicate work across instances. ---

    fn embedding_lock_path(&self) -> PathBuf {
        self.db_path.with_extension("embedding.lock")
    }

    /// Try to acquire the embedding lock. Returns `true` if we got it,
    /// `false` if another instance is already embedding (lock exists and
    /// the PID in it is still alive, or lock is <10 min old).
    pub fn try_acquire_embedding_lock(&self) -> bool {
        let lock = self.embedding_lock_path();
        if lock.exists() {
            // Check if the holder is still alive
            if let Ok(contents) = fs::read_to_string(&lock) {
                if let Ok(pid) = contents.trim().parse::<i32>() {
                    let holder = nix::unistd::Pid::from_raw(pid);
                    if nix::sys::signal::kill(holder, None).is_ok() {
                        // Process is alive — another instance is embedding
                        return false;
                    }
                    // Process is dead — stale lock, fall through to claim
                }
            }
            // Can't parse or process dead — break stale lock
            eprintln!("Warning: breaking stale embedding lock");
        }
        fs::write(&lock, format!("{}", std::process::id())).is_ok()
    }

    /// Release the embedding lock.
    pub fn release_embedding_lock(&self) {
        let _ = fs::remove_file(self.embedding_lock_path());
    }

    /// Save full graph to cache, tagged with the active language.
    pub fn save(&self, graph: &ConceptGraph, language: &str) -> Result<()> {
        self.acquire_lock()
            .context("failed to acquire cache lock")?;

        let result = self.save_inner(graph, language);
        self.release_lock();
        result
    }

    fn save_inner(&self, graph: &ConceptGraph, language: &str) -> Result<()> {
        let cached = CachedGraph {
            concepts: graph.concepts.clone(),
            relationships: graph.relationships.clone(),
            conventions: graph.conventions.clone(),
            embeddings: serde_json::from_value(serde_json::to_value(&graph.embeddings)?)?,
            signatures: graph.signatures.clone(),
            classes: graph.classes.clone(),
            call_sites: graph.call_sites.clone(),
            entities: graph.entities.clone(),
        };
        let json = serde_json::to_vec(&cached)?;

        let conn = Connection::open(&self.db_path)?;
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, updated_at)
             VALUES (?1, ?2, datetime('now'))",
            rusqlite::params!["graph", &json],
        )?;
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, updated_at)
             VALUES (?1, ?2, datetime('now'))",
            rusqlite::params![
                "cache_version",
                CACHE_VERSION.as_bytes()
            ],
        )?;
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, updated_at)
             VALUES (?1, ?2, datetime('now'))",
            rusqlite::params!["language", language.as_bytes()],
        )?;

        Ok(())
    }

    /// Start watching for source file changes matching the given extensions.
    /// Returns the watcher (must be kept alive) and a receiver that emits
    /// batches of changed paths.
    pub fn watch(
        &self,
        root: &Path,
        extensions: &[&str],
    ) -> Result<(RecommendedWatcher, mpsc::Receiver<Vec<PathBuf>>)> {
        let exts: Vec<String> = extensions.iter().map(|e| e.to_string()).collect();
        let (tx, rx) = mpsc::channel();
        let mut watcher = notify::recommended_watcher(
            move |res: std::result::Result<Event, notify::Error>| {
                if let Ok(event) = res {
                    let matched: Vec<PathBuf> = event
                        .paths
                        .into_iter()
                        .filter(|p| {
                            p.extension().is_some_and(|e| {
                                exts.iter().any(|ext| e == ext.as_str())
                            }) && !p.components().any(|c| {
                                c.as_os_str() == ".semex"
                            })
                        })
                        .collect();
                    if !matched.is_empty() {
                        let _ = tx.send(matched);
                    }
                }
            },
        )?;
        watcher.watch(root, RecursiveMode::Recursive)?;
        Ok((watcher, rx))
    }

    /// Load cached graph. Returns None if cache is missing, corrupt, stale,
    /// or was built for a different language.
    pub fn load(&self, language: &str) -> Result<Option<ConceptGraph>> {
        if !self.db_path.exists() {
            return Ok(None);
        }
        let conn = Connection::open(&self.db_path)?;

        // Check cache version — stale cache triggers re-index
        let stored_version: Option<String> = conn
            .query_row(
                "SELECT value FROM cache WHERE key = ?1",
                rusqlite::params!["cache_version"],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .ok()
            .and_then(|b| String::from_utf8(b).ok());
        if stored_version.as_deref() != Some(CACHE_VERSION) {
            eprintln!("Cache version mismatch — re-indexing");
            return Ok(None);
        }

        let stored_language: Option<String> = conn
            .query_row(
                "SELECT value FROM cache WHERE key = ?1",
                rusqlite::params!["language"],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .ok()
            .and_then(|b| String::from_utf8(b).ok());
        if stored_language.as_deref() != Some(language) {
            eprintln!(
                "Cache language mismatch (cached {:?}, active {:?}) — re-indexing",
                stored_language.as_deref().unwrap_or("unknown"),
                language,
            );
            return Ok(None);
        }

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
            entities: cached.entities,
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
            entities: HashMap::new(),
        }
    }

    #[test]
    fn test_cache_roundtrip() {
        let dir = std::path::Path::new("/tmp/semex_test_cache");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let cache = IndexCache::open(dir).unwrap();
        let graph = make_test_graph();
        cache.save(&graph, "python").unwrap();

        let loaded = cache.load("python").unwrap();
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
            entities: HashMap::new(),
        };

        let cache = IndexCache::open(dir).unwrap();
        cache.save(&graph, "python").unwrap();

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
        let loaded = cache.load("python").unwrap().expect("cache should load");

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
        let loaded = cache.load("python").unwrap();
        assert!(loaded.is_none());

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Save a graph with only some concepts embedded, load back,
    /// verify partial embeddings survive and concept count is intact.
    #[test]
    fn test_cache_roundtrip_partial_embeddings() {
        let dir = std::path::Path::new(
            "/tmp/semex_test_cache_partial_embed",
        );
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let mut concepts = HashMap::new();
        for i in 0..10u64 {
            concepts.insert(
                i,
                Concept {
                    id: i,
                    canonical: format!("concept_{i}"),
                    subtokens: vec![format!("concept_{i}")],
                    occurrences: vec![],
                    entity_types: HashSet::new(),
                    embedding: None,
                    subconcepts: Vec::new(),
                },
            );
        }

        let mut embeddings = EmbeddingIndex::empty();
        for i in 0..5u64 {
            embeddings.insert_vector(i, vec![i as f32; 3]);
        }

        let graph = ConceptGraph {
            concepts,
            relationships: Vec::new(),
            conventions: Vec::new(),
            embeddings,
            signatures: Vec::new(),
            classes: Vec::new(),
            entities: HashMap::new(),
            call_sites: Vec::new(),
        };

        let cache = IndexCache::open(dir).unwrap();
        cache.save(&graph, "python").unwrap();

        let loaded = cache.load("python").unwrap().expect("cache should load");
        assert_eq!(loaded.concepts.len(), 10);
        assert_eq!(loaded.embeddings.nb_vectors(), 5);

        for i in 0..5u64 {
            assert!(
                loaded.embeddings.get_vector(i).is_some(),
                "vector {i} should be present"
            );
        }
        for i in 5..10u64 {
            assert!(
                loaded.embeddings.get_vector(i).is_none(),
                "vector {i} should be absent"
            );
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Simulate checkpoint accumulation: save with 3 embeddings,
    /// load, add 4 more, save again, load — verify 7 total.
    #[test]
    fn test_cache_roundtrip_incremental_embeddings() {
        let dir = std::path::Path::new(
            "/tmp/semex_test_cache_incr_embed",
        );
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let mut concepts = HashMap::new();
        for i in 0..10u64 {
            concepts.insert(
                i,
                Concept {
                    id: i,
                    canonical: format!("concept_{i}"),
                    subtokens: vec![format!("concept_{i}")],
                    occurrences: vec![],
                    entity_types: HashSet::new(),
                    embedding: None,
                    subconcepts: Vec::new(),
                },
            );
        }

        // First save: 3 embeddings
        let mut embeddings = EmbeddingIndex::empty();
        for i in 0..3u64 {
            embeddings.insert_vector(i, vec![i as f32; 3]);
        }

        let graph = ConceptGraph {
            concepts: concepts.clone(),
            relationships: Vec::new(),
            conventions: Vec::new(),
            embeddings,
            signatures: Vec::new(),
            entities: HashMap::new(),
            classes: Vec::new(),
            call_sites: Vec::new(),
        };

        let cache = IndexCache::open(dir).unwrap();
        cache.save(&graph, "python").unwrap();

        // Load, add 4 more embeddings, save again
        let mut loaded = cache.load("python").unwrap().expect("first load");
        assert_eq!(loaded.embeddings.nb_vectors(), 3);

        for i in 3..7u64 {
            loaded.embeddings.insert_vector(i, vec![i as f32; 3]);
        }
        cache.save(&loaded, "python").unwrap();

        // Second load: should have 7
        let reloaded = cache.load("python").unwrap().expect("second load");
        assert_eq!(reloaded.concepts.len(), 10);
        assert_eq!(reloaded.embeddings.nb_vectors(), 7);

        for i in 0..7u64 {
            assert!(
                reloaded.embeddings.get_vector(i).is_some(),
                "vector {i} should be present after incremental save"
            );
        }

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_embedding_lock_acquire_release() {
        let dir = std::path::Path::new(
            "/tmp/semex_test_embed_lock_basic",
        );
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let cache = IndexCache::open(dir).unwrap();

        assert!(cache.try_acquire_embedding_lock());
        cache.release_embedding_lock();
        assert!(cache.try_acquire_embedding_lock());
        cache.release_embedding_lock();

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_embedding_lock_same_process_blocks() {
        let dir = std::path::Path::new(
            "/tmp/semex_test_embed_lock_block",
        );
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let cache = IndexCache::open(dir).unwrap();

        assert!(cache.try_acquire_embedding_lock());
        // Same PID is alive → second acquire should fail
        assert!(!cache.try_acquire_embedding_lock());
        cache.release_embedding_lock();

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_embedding_lock_stale_pid_breaks() {
        let dir = std::path::Path::new(
            "/tmp/semex_test_embed_lock_stale",
        );
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let cache = IndexCache::open(dir).unwrap();

        // Write a lock file with a PID that almost certainly doesn't exist
        let lock_path = dir.join(".semex").join("index.embedding.lock");
        std::fs::write(&lock_path, "999999999").unwrap();

        // Should break the stale lock and acquire
        assert!(cache.try_acquire_embedding_lock());
        cache.release_embedding_lock();

        let _ = std::fs::remove_dir_all(dir);
    }
}

use crate::embeddings::EmbeddingIndex;
use crate::graph::ConceptGraph;
use crate::types::{
    CallSite, ClassInfo, Concept, Convention, Entity, ParseResult,
    Relationship, Signature,
};
use anyhow::{Context, Result};
use notify::{Event, RecommendedWatcher, RecursiveMode, Watcher};
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::UNIX_EPOCH;

/// Auto-generated hash of indexing source files. Changes whenever
/// entity.rs, analyzer.rs, parser.rs, cache.rs, tokenizer.rs, or types.rs
/// are modified, triggering automatic re-indexing of stale caches.
const CACHE_VERSION: &str = env!("ONTOMICS_CACHE_VERSION");

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
    /// Open or create cache at `<repo_root>/.ontomics/index.db`.
    pub fn open(repo_root: &Path) -> Result<Self> {
        let ontomics_dir = repo_root.join(".ontomics");
        std::fs::create_dir_all(&ontomics_dir).with_context(|| {
            format!(
                "Cannot create cache directory '{}'. Check permissions on '{}'.",
                ontomics_dir.display(),
                repo_root.display(),
            )
        })?;

        let db_path = ontomics_dir.join("index.db");
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
                    "another ontomics instance holds the cache lock \
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

    /// Repo root derived from db_path (<repo>/.ontomics/index.db).
    fn repo_root(&self) -> &Path {
        self.db_path
            .parent()
            .and_then(|p| p.parent())
            .expect("db_path must be <repo>/.ontomics/index.db")
    }

    /// Collect all unique source files referenced by the graph.
    fn referenced_files(graph: &ConceptGraph) -> HashSet<PathBuf> {
        let mut files = HashSet::new();
        for concept in graph.concepts.values() {
            for occ in &concept.occurrences {
                files.insert(occ.file.clone());
            }
        }
        for sig in &graph.signatures {
            files.insert(sig.file.clone());
        }
        for cls in &graph.classes {
            files.insert(cls.file.clone());
        }
        for cs in &graph.call_sites {
            files.insert(cs.file.clone());
        }
        files
    }

    /// Build manifest: relative path → mtime as unix seconds.
    fn build_manifest(
        repo_root: &Path,
        files: &HashSet<PathBuf>,
    ) -> HashMap<PathBuf, u64> {
        let mut manifest = HashMap::new();
        for file in files {
            let abs = if file.is_relative() {
                repo_root.join(file)
            } else {
                file.clone()
            };
            if let Ok(meta) = fs::metadata(&abs) {
                let mtime = meta
                    .modified()
                    .ok()
                    .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                    .map(|d| d.as_secs())
                    .unwrap_or(0);
                manifest.insert(file.clone(), mtime);
            }
        }
        manifest
    }

    /// Save full graph to cache, tagged with the active language.
    /// For single-language mode — delegates to `save_multi`.
    pub fn save(&self, graph: &ConceptGraph, language: &str) -> Result<()> {
        self.save_multi(graph, &[language], &[])
    }

    /// Save graph with per-language parse results and file manifests.
    ///
    /// `languages` is the sorted list of active language names.
    /// `per_lang_results` contains `(lang_name, parse_results)` pairs
    /// for each language — stored so stale languages can be selectively
    /// re-parsed without touching fresh partitions.
    pub fn save_multi(
        &self,
        graph: &ConceptGraph,
        languages: &[&str],
        per_lang_results: &[(&str, &[ParseResult])],
    ) -> Result<()> {
        self.acquire_lock()
            .context("failed to acquire cache lock")?;

        let result =
            self.save_multi_inner(graph, languages, per_lang_results);
        self.release_lock();
        result
    }

    fn save_multi_inner(
        &self,
        graph: &ConceptGraph,
        languages: &[&str],
        per_lang_results: &[(&str, &[ParseResult])],
    ) -> Result<()> {
        let cached = CachedGraph {
            concepts: graph.concepts.clone(),
            relationships: graph.relationships.clone(),
            conventions: graph.conventions.clone(),
            embeddings: serde_json::from_value(
                serde_json::to_value(&graph.embeddings)?,
            )?,
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

        // Store sorted, comma-joined language key
        let lang_key = languages.join(",");
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, updated_at)
             VALUES (?1, ?2, datetime('now'))",
            rusqlite::params!["languages", lang_key.as_bytes()],
        )?;

        // Per-language parse results and file manifests
        let repo_root = self.repo_root().to_path_buf();
        for &(lang, results) in per_lang_results {
            let pr_json = serde_json::to_vec(results)?;
            let pr_key = format!("parse:{lang}");
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, updated_at)
                 VALUES (?1, ?2, datetime('now'))",
                rusqlite::params![pr_key, &pr_json],
            )?;

            let mut files = HashSet::new();
            for pr in results {
                for ident in &pr.identifiers {
                    files.insert(ident.file.clone());
                }
                for sig in &pr.signatures {
                    files.insert(sig.file.clone());
                }
                for cls in &pr.classes {
                    files.insert(cls.file.clone());
                }
                for cs in &pr.call_sites {
                    files.insert(cs.file.clone());
                }
            }
            let manifest = Self::build_manifest(&repo_root, &files);
            let manifest_json = serde_json::to_vec(&manifest)?;
            let mf_key = format!("manifest:{lang}");
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, updated_at)
                 VALUES (?1, ?2, datetime('now'))",
                rusqlite::params![mf_key, &manifest_json],
            )?;
        }

        // Fallback: store combined file manifest for single-language
        // callers (save() with no per_lang_results)
        if per_lang_results.is_empty() {
            let files = Self::referenced_files(graph);
            let manifest = Self::build_manifest(&repo_root, &files);
            let manifest_json = serde_json::to_vec(&manifest)?;
            conn.execute(
                "INSERT OR REPLACE INTO cache (key, value, updated_at)
                 VALUES (?1, ?2, datetime('now'))",
                rusqlite::params!["file_manifest", &manifest_json],
            )?;
        }

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
                                c.as_os_str() == ".ontomics"
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

    /// Load cached graph (single-language). Returns None if cache is
    /// missing, corrupt, stale, or was built for a different language.
    pub fn load(&self, language: &str) -> Result<Option<ConceptGraph>> {
        self.load_multi(&[language])
    }

    /// Load cached graph for a set of languages. Returns `Ok(None)` if
    /// the cache is stale, missing, or the language set doesn't match.
    pub fn load_multi(
        &self,
        languages: &[&str],
    ) -> Result<Option<ConceptGraph>> {
        if !self.db_path.exists() {
            return Ok(None);
        }
        let conn = Connection::open(&self.db_path)?;

        // Check cache version
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

        // Check language set match
        let lang_key = languages.join(",");
        let stored_langs: Option<String> = conn
            .query_row(
                "SELECT value FROM cache WHERE key = ?1",
                rusqlite::params!["languages"],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .ok()
            .and_then(|b| String::from_utf8(b).ok());
        if stored_langs.as_deref() != Some(&lang_key) {
            eprintln!(
                "Cache language mismatch (cached {:?}, active {:?}) \
                 — re-indexing",
                stored_langs.as_deref().unwrap_or("unknown"),
                lang_key,
            );
            return Ok(None);
        }

        // Check per-language file manifests
        if !self.check_manifests(&conn, languages)? {
            return Ok(None);
        }

        // Load graph blob
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
            Err(_) => return Ok(None),
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
            cluster_centroids: HashMap::new(),
        }))
    }

    /// Check which languages have stale file manifests. Returns the list
    /// of language names that need re-parsing.
    pub fn stale_languages(
        &self,
        languages: &[&str],
    ) -> Result<Vec<String>> {
        if !self.db_path.exists() {
            return Ok(languages.iter().map(|s| s.to_string()).collect());
        }
        let conn = Connection::open(&self.db_path)?;

        // Version mismatch → all stale
        let stored_version: Option<String> = conn
            .query_row(
                "SELECT value FROM cache WHERE key = ?1",
                rusqlite::params!["cache_version"],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .ok()
            .and_then(|b| String::from_utf8(b).ok());
        if stored_version.as_deref() != Some(CACHE_VERSION) {
            return Ok(languages.iter().map(|s| s.to_string()).collect());
        }

        let mut stale = Vec::new();
        let repo_root = self.repo_root().to_path_buf();
        for &lang in languages {
            let mf_key = format!("manifest:{lang}");
            let manifest_blob: Option<Vec<u8>> = conn
                .query_row(
                    "SELECT value FROM cache WHERE key = ?1",
                    rusqlite::params![mf_key],
                    |row| row.get(0),
                )
                .ok();
            match manifest_blob {
                None => {
                    stale.push(lang.to_string());
                }
                Some(blob) => {
                    if let Ok(manifest) = serde_json::from_slice::<
                        HashMap<PathBuf, u64>,
                    >(&blob)
                    {
                        if !Self::is_manifest_fresh(&repo_root, &manifest) {
                            stale.push(lang.to_string());
                        }
                    } else {
                        stale.push(lang.to_string());
                    }
                }
            }
        }
        Ok(stale)
    }

    /// Load cached parse results for a single language. Returns None if
    /// not present or corrupt.
    pub fn load_parse_results(
        &self,
        language: &str,
    ) -> Result<Option<Vec<ParseResult>>> {
        if !self.db_path.exists() {
            return Ok(None);
        }
        let conn = Connection::open(&self.db_path)?;
        let pr_key = format!("parse:{language}");
        let blob: Option<Vec<u8>> = conn
            .query_row(
                "SELECT value FROM cache WHERE key = ?1",
                rusqlite::params![pr_key],
                |row| row.get(0),
            )
            .ok();
        match blob {
            None => Ok(None),
            Some(b) => match serde_json::from_slice(&b) {
                Ok(results) => Ok(Some(results)),
                Err(_) => Ok(None),
            },
        }
    }

    /// Check per-language manifests. If per-language manifests exist, use
    /// them. Otherwise fall back to the combined `file_manifest` key
    /// (backward compatibility with single-language caches).
    fn check_manifests(
        &self,
        conn: &Connection,
        languages: &[&str],
    ) -> Result<bool> {
        let repo_root = self.repo_root().to_path_buf();

        // Try per-language manifests first
        let mut found_any = false;
        for &lang in languages {
            let mf_key = format!("manifest:{lang}");
            let manifest_blob: Option<Vec<u8>> = conn
                .query_row(
                    "SELECT value FROM cache WHERE key = ?1",
                    rusqlite::params![mf_key],
                    |row| row.get(0),
                )
                .ok();
            if let Some(blob) = manifest_blob {
                found_any = true;
                if let Ok(manifest) = serde_json::from_slice::<
                    HashMap<PathBuf, u64>,
                >(&blob)
                {
                    if !Self::is_manifest_fresh(&repo_root, &manifest) {
                        return Ok(false);
                    }
                }
            }
        }

        // Fall back to combined manifest (single-language cache compat)
        if !found_any {
            let manifest_blob: Option<Vec<u8>> = conn
                .query_row(
                    "SELECT value FROM cache WHERE key = ?1",
                    rusqlite::params!["file_manifest"],
                    |row| row.get(0),
                )
                .ok();
            if let Some(blob) = manifest_blob {
                if let Ok(manifest) = serde_json::from_slice::<
                    HashMap<PathBuf, u64>,
                >(&blob)
                {
                    if !Self::is_manifest_fresh(&repo_root, &manifest) {
                        return Ok(false);
                    }
                }
            }
        }

        Ok(true)
    }

    /// Check whether all files in a manifest still exist with same mtime.
    fn is_manifest_fresh(
        repo_root: &Path,
        manifest: &HashMap<PathBuf, u64>,
    ) -> bool {
        for (file, cached_mtime) in manifest {
            let abs = if file.is_relative() {
                repo_root.join(file)
            } else {
                file.clone()
            };
            let current_mtime = fs::metadata(&abs)
                .ok()
                .and_then(|m| m.modified().ok())
                .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
                .map(|d| d.as_secs());
            match current_mtime {
                None => {
                    eprintln!(
                        "Cached file deleted: {} — re-indexing",
                        file.display(),
                    );
                    return false;
                }
                Some(mtime) if mtime != *cached_mtime => {
                    eprintln!(
                        "Cached file modified: {} — re-indexing",
                        file.display(),
                    );
                    return false;
                }
                _ => {}
            }
        }
        true
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
                cluster_id: None,
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
            cluster_centroids: HashMap::new(),
        }
    }

    #[test]
    fn test_cache_roundtrip() {
        let dir = std::path::Path::new("/tmp/ontomics_test_cache");
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
        let dir = std::path::Path::new("/tmp/ontomics_test_blob_sizes");
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
        let dir = std::path::Path::new("/tmp/ontomics_test_save_pattern");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let db_path = dir.join(".ontomics").join("index.db");
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

        let dir = std::path::Path::new("/tmp/ontomics_test_cache_realistic");
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
            // Assign cluster_id to concepts 0-9 (3 clusters) to test round-trip
            let cluster_id = if i < 4 {
                Some(0)
            } else if i < 8 {
                Some(1)
            } else if i < 10 {
                Some(2)
            } else {
                None
            };
            concepts.insert(
                i as u64,
                Concept {
                    id: i as u64,
                    canonical: format!("concept_{i}"),
                    subtokens: vec![format!("concept_{i}")],
                    occurrences: occs,
                    entity_types: HashSet::new(),
                    embedding: None,
                    cluster_id,
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
            cluster_centroids: HashMap::new(),
        };

        let cache = IndexCache::open(dir).unwrap();
        cache.save(&graph, "python").unwrap();

        // Check for leftover journal/wal files
        let ontomics_dir = dir.join(".ontomics");
        let journal = ontomics_dir.join("index.db-journal");
        let wal = ontomics_dir.join("index.db-wal");
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

        // Verify cluster_id survives round-trip
        assert_eq!(loaded.concepts[&0].cluster_id, Some(0));
        assert_eq!(loaded.concepts[&5].cluster_id, Some(1));
        assert_eq!(loaded.concepts[&9].cluster_id, Some(2));
        assert_eq!(loaded.concepts[&10].cluster_id, None);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_cache_load_missing() {
        let dir = std::path::Path::new("/tmp/ontomics_test_cache_missing");
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
            "/tmp/ontomics_test_cache_partial_embed",
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
                    cluster_id: None,
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
            cluster_centroids: HashMap::new(),
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
            "/tmp/ontomics_test_cache_incr_embed",
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
                    cluster_id: None,
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
            cluster_centroids: HashMap::new(),
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
            "/tmp/ontomics_test_embed_lock_basic",
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
            "/tmp/ontomics_test_embed_lock_block",
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
            "/tmp/ontomics_test_embed_lock_stale",
        );
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let cache = IndexCache::open(dir).unwrap();

        // Write a lock file with a PID that almost certainly doesn't exist
        let lock_path = dir.join(".ontomics").join("index.embedding.lock");
        std::fs::write(&lock_path, "999999999").unwrap();

        // Should break the stale lock and acquire
        assert!(cache.try_acquire_embedding_lock());
        cache.release_embedding_lock();

        let _ = std::fs::remove_dir_all(dir);
    }

    /// Helper: build a graph whose occurrences reference real files on disk.
    fn make_graph_with_files(files: &[PathBuf]) -> ConceptGraph {
        use crate::types::Occurrence;
        let mut concepts = HashMap::new();
        let occs: Vec<Occurrence> = files
            .iter()
            .enumerate()
            .map(|(i, f)| Occurrence {
                identifier: format!("id_{i}"),
                entity_type: crate::types::EntityType::Function,
                file: f.clone(),
                line: 1,
            })
            .collect();
        concepts.insert(
            1,
            Concept {
                id: 1,
                canonical: "test".to_string(),
                subtokens: vec!["test".to_string()],
                occurrences: occs,
                entity_types: HashSet::new(),
                embedding: None,
                cluster_id: None,
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
            cluster_centroids: HashMap::new(),
        }
    }

    #[test]
    fn test_manifest_invalidates_on_deleted_file() {
        let dir = Path::new("/tmp/ontomics_test_manifest_delete");
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();

        // Create a source file the graph references
        let src = dir.join("example.py");
        fs::write(&src, "def foo(): pass").unwrap();

        let graph = make_graph_with_files(&[src.clone()]);
        let cache = IndexCache::open(dir).unwrap();
        cache.save(&graph, "python").unwrap();

        // Cache loads fine while file exists
        assert!(cache.load("python").unwrap().is_some());

        // Delete the file — cache should invalidate
        fs::remove_file(&src).unwrap();
        assert!(cache.load("python").unwrap().is_none());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_manifest_invalidates_on_modified_file() {
        let dir = Path::new("/tmp/ontomics_test_manifest_modify");
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();

        let src = dir.join("example.py");
        fs::write(&src, "def foo(): pass").unwrap();

        let graph = make_graph_with_files(&[src.clone()]);
        let cache = IndexCache::open(dir).unwrap();
        cache.save(&graph, "python").unwrap();

        assert!(cache.load("python").unwrap().is_some());

        // Modify the file (bump mtime by writing new content after a delay)
        std::thread::sleep(std::time::Duration::from_secs(1));
        fs::write(&src, "def foo(): return 42").unwrap();
        assert!(cache.load("python").unwrap().is_none());

        let _ = fs::remove_dir_all(dir);
    }

    #[test]
    fn test_manifest_missing_is_graceful() {
        // Old caches without a manifest should still load (no regression)
        let dir = Path::new("/tmp/ontomics_test_manifest_missing");
        let _ = fs::remove_dir_all(dir);
        fs::create_dir_all(dir).unwrap();

        let graph = make_test_graph();
        let cache = IndexCache::open(dir).unwrap();
        cache.save(&graph, "python").unwrap();

        // Manually delete the manifest row to simulate an old cache
        let db_path = dir.join(".ontomics").join("index.db");
        let conn = Connection::open(&db_path).unwrap();
        conn.execute(
            "DELETE FROM cache WHERE key = ?1",
            rusqlite::params!["file_manifest"],
        )
        .unwrap();

        // Should still load — no manifest means skip check
        assert!(cache.load("python").unwrap().is_some());

        let _ = fs::remove_dir_all(dir);
    }
}

use serde::Deserialize;
use std::path::Path;

/// Language to analyze.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    #[default]
    Auto,
    Python,
    #[serde(alias = "ts")]
    TypeScript,
    #[serde(alias = "js")]
    JavaScript,
}

/// Directories to skip when counting file extensions for auto-detection.
const DETECT_SKIP_DIRS: &[&str] = &[
    "node_modules",
    "__pycache__",
    ".git",
    "dist",
    ".next",
    "venv",
    ".venv",
    ".tox",
    "target",
    ".semex",
];

impl Language {
    /// Auto-detect language from file extensions in the repo root.
    /// Walks up to `max_depth` levels, skipping common non-source dirs.
    pub fn detect(repo_root: &Path) -> Language {
        let mut py_count = 0u32;
        let mut ts_count = 0u32;
        let mut js_count = 0u32;
        count_extensions(repo_root, &mut py_count, &mut ts_count, &mut js_count, 3);

        if ts_count > py_count && ts_count > js_count {
            Language::TypeScript
        } else if js_count > py_count && js_count > ts_count {
            Language::JavaScript
        } else {
            Language::Python
        }
    }

    /// Stable string name for cache tagging.
    pub fn name(&self) -> &'static str {
        match self {
            Language::Auto | Language::Python => "python",
            Language::TypeScript => "typescript",
            Language::JavaScript => "javascript",
        }
    }

    /// Resolve auto-detection: if `Auto`, detect from repo. Otherwise
    /// return self.
    pub fn resolve(&self, repo_root: &Path) -> Language {
        match self {
            Language::Auto => Language::detect(repo_root),
            other => other.clone(),
        }
    }

    /// Default include globs for this language.
    pub fn default_include(&self) -> Vec<String> {
        match self {
            Language::Python | Language::Auto => {
                vec!["**/*.py".to_string()]
            }
            Language::TypeScript => {
                vec!["**/*.ts".to_string(), "**/*.tsx".to_string()]
            }
            Language::JavaScript => {
                vec!["**/*.js".to_string(), "**/*.jsx".to_string()]
            }
        }
    }

    /// Default exclude globs for this language.
    pub fn default_exclude(&self) -> Vec<String> {
        match self {
            Language::Python | Language::Auto => vec![
                "**/test_*".to_string(),
                "**/__pycache__".to_string(),
            ],
            Language::TypeScript => vec![
                "**/node_modules".to_string(),
                "**/dist".to_string(),
                "**/.next".to_string(),
                "**/*.d.ts".to_string(),
                "**/*.test.ts".to_string(),
                "**/*.spec.ts".to_string(),
            ],
            Language::JavaScript => vec![
                "**/node_modules".to_string(),
                "**/dist".to_string(),
                "**/.next".to_string(),
                "**/*.test.js".to_string(),
                "**/*.spec.js".to_string(),
            ],
        }
    }
}

/// Recursively count file extensions up to `remaining_depth`.
fn count_extensions(
    dir: &Path,
    py: &mut u32,
    ts: &mut u32,
    js: &mut u32,
    remaining_depth: u32,
) {
    if remaining_depth == 0 {
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(_) => return,
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name.starts_with('.') {
                continue;
            }
            if path.is_dir() {
                if DETECT_SKIP_DIRS.contains(&name) {
                    continue;
                }
                count_extensions(&path, py, ts, js, remaining_depth - 1);
                continue;
            }
        }
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            match ext {
                "py" => *py += 1,
                "ts" | "tsx" => *ts += 1,
                "js" | "jsx" | "mjs" | "cjs" => *js += 1,
                _ => {}
            }
        }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    /// Language to analyze (default: auto-detect).
    #[serde(default)]
    pub language: Language,
    pub index: IndexConfig,
    pub analysis: AnalysisConfig,
    pub embeddings: EmbeddingsConfig,
    pub cache: CacheConfig,
    pub resources: ResourcesConfig,
    /// Bootstrap conventions to seed detection (optional).
    #[serde(default)]
    pub conventions: Vec<ConventionConfig>,
    /// Paths to domain pack YAML files, resolved relative to repo root.
    #[serde(default)]
    pub domain_packs: Vec<String>,
    #[cfg(feature = "lsp")]
    #[serde(default)]
    pub lsp: LspConfig,
}

#[cfg(feature = "lsp")]
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct LspConfig {
    pub enabled: bool,
    pub pyright_path: String,
    pub timeout_secs: u64,
}

#[cfg(feature = "lsp")]
impl Default for LspConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            pyright_path: "pyright".to_string(),
            timeout_secs: 120,
        }
    }
}

/// A convention entry in the config file, matching the TOML `[[conventions]]`
/// array format from the spec.
#[derive(Debug, Clone, Deserialize)]
pub struct ConventionConfig {
    /// Pattern type: "prefix", "suffix", "compound", or "conversion".
    pub pattern: String,
    /// The pattern value (e.g., "nb_", "_count", "_to_").
    pub value: String,
    /// Semantic role (e.g., "count", "boolean predicate").
    pub role: String,
    /// Entity types this convention applies to.
    #[serde(default)]
    pub entity_types: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct IndexConfig {
    pub include: Vec<String>,
    pub exclude: Vec<String>,
    pub min_frequency: usize,
    pub respect_gitignore: bool,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct AnalysisConfig {
    pub domain_specificity_threshold: f64,
    pub co_occurrence_scope: String,
    pub convention_threshold: usize,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingsConfig {
    pub enabled: bool,
    pub model: String,
    pub similarity_threshold: f32,
    pub model_cache_dir: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CacheConfig {
    pub enabled: bool,
    pub path: String,
}

impl Default for IndexConfig {
    fn default() -> Self {
        Self {
            include: vec!["**/*.py".to_string()],
            exclude: vec![
                "**/test_*".to_string(),
                "**/__pycache__".to_string(),
            ],
            min_frequency: 2,
            respect_gitignore: true,
        }
    }
}

impl IndexConfig {
    /// Apply language-specific defaults if the user hasn't customized them.
    /// Detects customization by comparing against the Python defaults.
    pub fn resolve_for_language(&mut self, lang: &Language) {
        let py_defaults = IndexConfig::default();
        if self.include == py_defaults.include {
            self.include = lang.default_include();
        }
        if self.exclude == py_defaults.exclude {
            self.exclude = lang.default_exclude();
        }
    }
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            domain_specificity_threshold: 0.3,
            co_occurrence_scope: "function".to_string(),
            convention_threshold: 3,
        }
    }
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            model: "BAAI/bge-small-en-v1.5".to_string(),
            similarity_threshold: 0.75,
            model_cache_dir: "~/.cache/semex".to_string(),
        }
    }
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            path: ".semex/index.db".to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ResourcesConfig {
    /// Max rayon threads for parallel parsing (default: half CPU count).
    pub max_threads: usize,
    /// Concepts per embedding batch (default: 64). Smaller = less peak
    /// memory, more frequent cache checkpoints.
    pub embedding_batch_size: usize,
}

impl ResourcesConfig {
    /// Ensure values are within sane bounds after deserialization.
    pub fn clamp(&mut self) {
        if self.max_threads == 0 {
            self.max_threads = 1;
        }
        if self.embedding_batch_size == 0 {
            self.embedding_batch_size = 1;
        }
    }
}

impl Default for ResourcesConfig {
    fn default() -> Self {
        let cpus = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4);
        Self {
            max_threads: (cpus / 2).max(1),
            embedding_batch_size: 64,
        }
    }
}

impl Config {
    /// Load config from `.semex/config.toml` relative to repo_root.
    /// Returns default config if file doesn't exist.
    pub fn load(repo_root: &Path) -> anyhow::Result<Self> {
        let config_path = repo_root.join(".semex").join("config.toml");
        if !config_path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(&config_path)?;
        let mut config: Config = toml::from_str(&content)?;
        config.resources.clamp();
        Ok(config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.index.min_frequency, 2);
        assert_eq!(config.analysis.convention_threshold, 3);
        assert!(config.embeddings.enabled);
        assert!((config.embeddings.similarity_threshold - 0.75).abs() < f32::EPSILON);
        assert!(config.cache.enabled);
    }

    #[test]
    fn test_load_missing_file_returns_default() {
        let dir = std::path::Path::new("/tmp/semex_test_config_missing");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();

        let config = Config::load(dir).unwrap();
        assert_eq!(config.index.min_frequency, 2);
        assert_eq!(config.analysis.convention_threshold, 3);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_load_from_toml_string() {
        let toml_str = r#"
[index]
min_frequency = 5

[analysis]
convention_threshold = 10
domain_specificity_threshold = 0.5

[embeddings]
enabled = false
similarity_threshold = 0.8

[cache]
enabled = false
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.min_frequency, 5);
        assert_eq!(config.analysis.convention_threshold, 10);
        assert!(!config.embeddings.enabled);
        assert!((config.embeddings.similarity_threshold - 0.8).abs() < f32::EPSILON);
        assert!(!config.cache.enabled);
    }

    #[test]
    fn test_partial_toml_uses_defaults() {
        let toml_str = r#"
[analysis]
convention_threshold = 7
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        // Overridden
        assert_eq!(config.analysis.convention_threshold, 7);
        // Defaults preserved
        assert_eq!(config.index.min_frequency, 2);
        assert!(config.embeddings.enabled);
    }

    #[test]
    fn test_conventions_config() {
        let toml_str = r#"
[[conventions]]
pattern = "prefix"
value = "nb_"
role = "count"
entity_types = ["Parameter", "Variable"]

[[conventions]]
pattern = "conversion"
value = "_to_"
role = "type/format conversion"
entity_types = ["Function"]
"#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.conventions.len(), 2);
        assert_eq!(config.conventions[0].pattern, "prefix");
        assert_eq!(config.conventions[0].value, "nb_");
        assert_eq!(config.conventions[0].role, "count");
        assert_eq!(config.conventions[1].pattern, "conversion");
        assert_eq!(config.conventions[1].value, "_to_");
    }

    #[test]
    fn test_load_from_file() {
        let dir = std::path::Path::new("/tmp/semex_test_config_load");
        let _ = std::fs::remove_dir_all(dir);
        let semex_dir = dir.join(".semex");
        std::fs::create_dir_all(&semex_dir).unwrap();

        std::fs::write(
            semex_dir.join("config.toml"),
            "[analysis]\nconvention_threshold = 4\n",
        )
        .unwrap();

        let config = Config::load(dir).unwrap();
        assert_eq!(config.analysis.convention_threshold, 4);
        assert_eq!(config.index.min_frequency, 2); // default

        let _ = std::fs::remove_dir_all(dir);
    }

    // --- Language tests ---

    #[test]
    fn test_language_default_is_auto() {
        let config = Config::default();
        assert_eq!(config.language, Language::Auto);
    }

    #[test]
    fn test_language_from_toml() {
        let toml_str = r#"language = "typescript""#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.language, Language::TypeScript);
    }

    #[test]
    fn test_language_alias_ts() {
        let toml_str = r#"language = "ts""#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.language, Language::TypeScript);
    }

    #[test]
    fn test_language_alias_js() {
        let toml_str = r#"language = "js""#;
        let config: Config = toml::from_str(toml_str).unwrap();
        assert_eq!(config.language, Language::JavaScript);
    }

    #[test]
    fn test_language_detect_python() {
        let dir = std::path::Path::new("/tmp/semex_test_detect_py");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("foo.py"), "").unwrap();
        std::fs::write(dir.join("bar.py"), "").unwrap();
        std::fs::write(dir.join("baz.py"), "").unwrap();

        let detected = Language::detect(dir);
        assert_eq!(detected, Language::Python);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_language_detect_typescript() {
        let dir = std::path::Path::new("/tmp/semex_test_detect_ts");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("index.ts"), "").unwrap();
        std::fs::write(dir.join("app.tsx"), "").unwrap();
        std::fs::write(dir.join("utils.ts"), "").unwrap();

        let detected = Language::detect(dir);
        assert_eq!(detected, Language::TypeScript);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_language_detect_javascript() {
        let dir = std::path::Path::new("/tmp/semex_test_detect_js");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("index.js"), "").unwrap();
        std::fs::write(dir.join("app.jsx"), "").unwrap();
        std::fs::write(dir.join("utils.js"), "").unwrap();

        let detected = Language::detect(dir);
        assert_eq!(detected, Language::JavaScript);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_language_detect_skips_node_modules() {
        let dir = std::path::Path::new("/tmp/semex_test_detect_skip_nm");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir.join("node_modules")).unwrap();
        // Many JS files in node_modules should not tip the scale
        for i in 0..20 {
            std::fs::write(
                dir.join("node_modules").join(format!("lib_{i}.js")),
                "",
            )
            .unwrap();
        }
        std::fs::write(dir.join("main.py"), "").unwrap();
        std::fs::write(dir.join("utils.py"), "").unwrap();

        let detected = Language::detect(dir);
        assert_eq!(detected, Language::Python);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_language_resolve_auto_detects() {
        let dir = std::path::Path::new("/tmp/semex_test_resolve_auto");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        std::fs::write(dir.join("a.ts"), "").unwrap();
        std::fs::write(dir.join("b.ts"), "").unwrap();

        let resolved = Language::Auto.resolve(dir);
        assert_eq!(resolved, Language::TypeScript);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_language_resolve_explicit_unchanged() {
        let dir = std::path::Path::new("/tmp/semex_test_resolve_explicit");
        let _ = std::fs::remove_dir_all(dir);
        std::fs::create_dir_all(dir).unwrap();
        // Even though there are TS files, explicit Python stays Python
        std::fs::write(dir.join("a.ts"), "").unwrap();

        let resolved = Language::Python.resolve(dir);
        assert_eq!(resolved, Language::Python);

        let _ = std::fs::remove_dir_all(dir);
    }

    #[test]
    fn test_typescript_default_globs() {
        let include = Language::TypeScript.default_include();
        assert_eq!(include, vec!["**/*.ts", "**/*.tsx"]);
        let exclude = Language::TypeScript.default_exclude();
        assert!(exclude.contains(&"**/node_modules".to_string()));
        assert!(exclude.contains(&"**/*.d.ts".to_string()));
    }

    #[test]
    fn test_javascript_default_globs() {
        let include = Language::JavaScript.default_include();
        assert_eq!(include, vec!["**/*.js", "**/*.jsx"]);
        let exclude = Language::JavaScript.default_exclude();
        assert!(exclude.contains(&"**/node_modules".to_string()));
    }

    #[test]
    fn test_resolve_for_language_overrides_defaults() {
        let mut index = IndexConfig::default();
        assert_eq!(index.include, vec!["**/*.py"]);

        index.resolve_for_language(&Language::TypeScript);
        assert_eq!(index.include, vec!["**/*.ts", "**/*.tsx"]);
        assert!(index.exclude.contains(&"**/node_modules".to_string()));
    }

    #[test]
    fn test_resolve_for_language_preserves_custom() {
        let mut index = IndexConfig {
            include: vec!["src/**/*.py".to_string()],
            exclude: vec!["**/vendor/**".to_string()],
            ..Default::default()
        };

        index.resolve_for_language(&Language::TypeScript);
        // Custom values preserved (not replaced with TS defaults)
        assert_eq!(index.include, vec!["src/**/*.py"]);
        assert_eq!(index.exclude, vec!["**/vendor/**"]);
    }
}

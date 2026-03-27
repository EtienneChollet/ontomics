use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Default, Deserialize)]
#[serde(default)]
pub struct Config {
    pub index: IndexConfig,
    pub analysis: AnalysisConfig,
    pub embeddings: EmbeddingsConfig,
    pub cache: CacheConfig,
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

impl Config {
    /// Load config from `.semex/config.toml` relative to repo_root.
    /// Returns default config if file doesn't exist.
    pub fn load(repo_root: &Path) -> anyhow::Result<Self> {
        let config_path = repo_root.join(".semex").join("config.toml");
        if !config_path.exists() {
            return Ok(Self::default());
        }
        let content = std::fs::read_to_string(&config_path)?;
        let config: Config = toml::from_str(&content)?;
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
}

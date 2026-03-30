use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Files whose logic affects the cached index. When any of these change,
/// the cache version changes automatically and stale caches are discarded.
const INDEX_FILES: &[&str] = &[
    "src/entity.rs",
    "src/analyzer.rs",
    "src/parser.rs",
    "src/cache.rs",
    "src/tokenizer.rs",
    "src/types.rs",
];

fn main() {
    let mut hasher = DefaultHasher::new();
    for path in INDEX_FILES {
        println!("cargo:rerun-if-changed={path}");
        if let Ok(contents) = std::fs::read_to_string(path) {
            contents.hash(&mut hasher);
        }
    }
    let hash = hasher.finish();
    println!("cargo:rustc-env=ONTOMICS_CACHE_VERSION={hash}");
}

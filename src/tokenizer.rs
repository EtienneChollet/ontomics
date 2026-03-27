/// Split an identifier into word-level subtokens.
/// "spatial_transform" -> ["spatial", "transform"]
/// "SpatialTransform" -> ["spatial", "transform"]
/// "nb_features" -> ["nb", "features"]
/// "ndim" -> ["ndim"]  (single token — no split point)
pub fn split_identifier(name: &str) -> Vec<String> {
    todo!()
}

/// Detect if a subtoken is likely an abbreviation of a full word.
/// Uses edit distance + common abbreviation patterns.
pub fn find_abbreviation(short: &str, candidates: &[String]) -> Option<String> {
    todo!()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_snake_case() {
        assert_eq!(
            split_identifier("spatial_transform"),
            vec!["spatial", "transform"]
        );
    }

    #[test]
    fn test_split_camel_case() {
        assert_eq!(
            split_identifier("SpatialTransform"),
            vec!["spatial", "transform"]
        );
    }

    #[test]
    fn test_split_single_word() {
        assert_eq!(split_identifier("ndim"), vec!["ndim"]);
    }
}

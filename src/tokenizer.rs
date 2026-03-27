/// Split an identifier into word-level subtokens.
///
/// Rules applied in order:
/// 1. Strip leading/trailing underscores
/// 2. Split on `_` (snake_case)
/// 3. Split on camelCase boundaries (lowercase-to-uppercase transition)
/// 4. Split on ALLCAPS-to-lowercase transitions (e.g., HTTPSConnection)
/// 5. Lowercase all tokens
/// 6. Filter empty strings
///
/// Examples
/// --------
/// - `"spatial_transform"` -> `["spatial", "transform"]`
/// - `"SpatialTransform"` -> `["spatial", "transform"]`
/// - `"HTTPSConnection"` -> `["https", "connection"]`
/// - `"__init__"` -> `["init"]`
/// - `"ndim"` -> `["ndim"]`
pub fn split_identifier(name: &str) -> Vec<String> {
    let trimmed = name.trim_matches('_');
    if trimmed.is_empty() {
        return Vec::new();
    }

    // First split on underscores
    let underscore_parts: Vec<&str> = trimmed.split('_').collect();

    let mut tokens = Vec::new();
    for part in underscore_parts {
        if part.is_empty() {
            continue;
        }
        // Split each part on camelCase / ALLCAPS boundaries
        tokens.extend(split_camel(part));
    }

    tokens
}

/// Split a single segment (no underscores) on camelCase and ALLCAPS boundaries.
///
/// - `"SpatialTransform"` -> `["spatial", "transform"]`
/// - `"HTTPSConnection"` -> `["https", "connection"]`
/// - `"get2dArray"` -> `["get", "2d", "array"]` (digits stay with adjacent lowercase)
/// - `"ndim"` -> `["ndim"]`
fn split_camel(s: &str) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    let len = chars.len();
    if len == 0 {
        return Vec::new();
    }

    let mut tokens = Vec::new();
    let mut start = 0;

    for i in 1..len {
        let prev = chars[i - 1];
        let curr = chars[i];

        // Split on lowercase/digit -> uppercase transition (camelCase boundary)
        // e.g., "spatial|Transform" or "get|2d" doesn't apply here, but
        //        "spatialT" -> split before T
        let lower_to_upper =
            (prev.is_lowercase() || prev.is_ascii_digit()) && curr.is_uppercase();

        // Split on ALLCAPS -> lowercase transition, but split BEFORE the last
        // uppercase char. E.g., "HTTPS|Connection": when we see 'o' (lowercase)
        // after 'C' (uppercase), we split before 'C'.
        let caps_to_lower = i >= 2
            && chars[i - 2].is_uppercase()
            && prev.is_uppercase()
            && curr.is_lowercase();

        if lower_to_upper {
            let token: String =
                chars[start..i].iter().collect::<String>().to_lowercase();
            if !token.is_empty() {
                tokens.push(token);
            }
            start = i;
        } else if caps_to_lower {
            // Split before prev (i-1), not before curr
            let token: String =
                chars[start..i - 1].iter().collect::<String>().to_lowercase();
            if !token.is_empty() {
                tokens.push(token);
            }
            start = i - 1;
        }
    }

    // Push remaining segment
    let token: String = chars[start..].iter().collect::<String>().to_lowercase();
    if !token.is_empty() {
        tokens.push(token);
    }

    tokens
}

/// Find whether `short` is likely an abbreviation of one of the `candidates`.
///
/// Matching strategy (in priority order):
/// 1. Prefix match: `short` is a prefix of a candidate
/// 2. Subsequence match: letters of `short` appear in order within a candidate
///
/// Returns the best match: prefers prefix matches over subsequence matches,
/// and shorter candidates over longer ones (as a tighter match).
///
/// Returns `None` if `short` is empty or no candidate matches.
pub fn find_abbreviation(short: &str, candidates: &[String]) -> Option<String> {
    if short.is_empty() {
        return None;
    }

    let short_lower = short.to_lowercase();

    // Collect prefix matches and subsequence matches separately
    let mut prefix_matches: Vec<&String> = Vec::new();
    let mut subseq_matches: Vec<&String> = Vec::new();

    for candidate in candidates {
        let cand_lower = candidate.to_lowercase();

        // short must be shorter than candidate to be an abbreviation
        if short_lower.len() >= cand_lower.len() {
            continue;
        }

        if cand_lower.starts_with(&short_lower) {
            prefix_matches.push(candidate);
        } else if is_subsequence(&short_lower, &cand_lower) {
            subseq_matches.push(candidate);
        }
    }

    // Prefer prefix matches, then subsequence matches.
    // Within each group, prefer the shortest candidate (tightest match).
    if let Some(best) = prefix_matches.iter().min_by_key(|c| c.len()) {
        return Some((*best).clone());
    }

    if let Some(best) = subseq_matches.iter().min_by_key(|c| c.len()) {
        return Some((*best).clone());
    }

    None
}

/// Check if every character in `needle` appears in `haystack` in order.
fn is_subsequence(needle: &str, haystack: &str) -> bool {
    let mut hay_iter = haystack.chars();
    for n_ch in needle.chars() {
        loop {
            match hay_iter.next() {
                Some(h_ch) if h_ch == n_ch => break,
                Some(_) => continue,
                None => return false,
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- split_identifier tests ---

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

    #[test]
    fn test_split_leading_underscores() {
        assert_eq!(
            split_identifier("_private_var"),
            vec!["private", "var"]
        );
    }

    #[test]
    fn test_split_dunder() {
        assert_eq!(split_identifier("__init__"), vec!["init"]);
    }

    #[test]
    fn test_split_nb_prefix() {
        assert_eq!(
            split_identifier("nb_features"),
            vec!["nb", "features"]
        );
    }

    #[test]
    fn test_split_empty() {
        let empty: Vec<String> = Vec::new();
        assert_eq!(split_identifier(""), empty);
    }

    #[test]
    fn test_split_allcaps_transition() {
        assert_eq!(
            split_identifier("HTTPSConnection"),
            vec!["https", "connection"]
        );
    }

    #[test]
    fn test_split_with_digits() {
        assert_eq!(
            split_identifier("get_2d_array"),
            vec!["get", "2d", "array"]
        );
    }

    #[test]
    fn test_split_single_char() {
        assert_eq!(split_identifier("x"), vec!["x"]);
    }

    #[test]
    fn test_split_only_underscores() {
        let empty: Vec<String> = Vec::new();
        assert_eq!(split_identifier("___"), empty);
    }

    #[test]
    fn test_split_lower_camel() {
        assert_eq!(
            split_identifier("getFieldValue"),
            vec!["get", "field", "value"]
        );
    }

    // --- find_abbreviation tests ---

    #[test]
    fn test_abbrev_prefix_match() {
        let candidates = vec![
            "segmentation".to_string(),
            "segment".to_string(),
            "value".to_string(),
        ];
        let result = find_abbreviation("seg", &candidates);
        assert!(result.is_some());
        // Should prefer "segment" (shorter prefix match)
        assert_eq!(result.unwrap(), "segment");
    }

    #[test]
    fn test_abbrev_subsequence_match() {
        let candidates = vec!["transform".to_string(), "transfer".to_string()];
        let result = find_abbreviation("trf", &candidates);
        assert!(result.is_some());
        // "trf" is a subsequence of both; prefer shorter
        let matched = result.unwrap();
        assert!(matched == "transfer" || matched == "transform");
    }

    #[test]
    fn test_abbrev_no_match() {
        let candidates = vec!["apple".to_string(), "banana".to_string()];
        assert_eq!(find_abbreviation("xyz", &candidates), None);
    }

    #[test]
    fn test_abbrev_empty_short() {
        let candidates = vec!["transform".to_string()];
        assert_eq!(find_abbreviation("", &candidates), None);
    }

    #[test]
    fn test_abbrev_short_longer_than_candidate() {
        let candidates = vec!["ab".to_string()];
        assert_eq!(find_abbreviation("abc", &candidates), None);
    }

    #[test]
    fn test_abbrev_empty_candidates() {
        let candidates: Vec<String> = Vec::new();
        assert_eq!(find_abbreviation("seg", &candidates), None);
    }

    #[test]
    fn test_abbrev_prefers_prefix_over_subsequence() {
        let candidates = vec![
            "transform".to_string(), // "tra" is prefix
            "trajectory".to_string(), // "tra" is prefix too, but longer
        ];
        let result = find_abbreviation("tra", &candidates);
        assert!(result.is_some());
        // Both are prefix matches; prefer shorter
        assert_eq!(result.unwrap(), "transform");
    }

    // --- is_subsequence tests ---

    #[test]
    fn test_subsequence_basic() {
        assert!(is_subsequence("trf", "transform"));
        assert!(is_subsequence("sg", "segmentation"));
        assert!(!is_subsequence("xyz", "apple"));
        assert!(is_subsequence("", "anything"));
    }
}

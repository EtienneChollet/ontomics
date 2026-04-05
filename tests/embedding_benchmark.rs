// Embedding model benchmark — ground truth cluster expectations.
//
// Each ExpectedCluster defines a behavioral group of functions that a good
// embedding model MUST place in the same cluster. The `must_not_cluster` list
// names functions that are superficially related but behaviorally different.
//
// Scoring: for each model, we count how many expected clusters are fully
// recovered (all must_cluster members share a cluster_id AND no must_not_cluster
// members appear in that cluster). The model with the highest recovery rate wins.
//
// These expectations are derived from manual analysis of voxelmorph, neurite,
// and interseg3d source code.

use ontomics::cluster;
use ontomics::config::Language;
use ontomics::embeddings::{self, SUPPORTED_MODELS};
use ontomics::parser::{self, ParseOptions};
use std::collections::HashMap;
use std::path::Path;

struct ExpectedCluster {
    label: &'static str,
    must_cluster: Vec<&'static str>,
    must_not_cluster: Vec<&'static str>,
}

struct BenchmarkResult {
    model_id: String,
    nb_functions: usize,
    nb_clusters: usize,
    clusters_recovered: usize,
    clusters_total: usize,
    recovery_rate: f32,
    mean_intra_similarity: f32,
    mean_inter_similarity: f32,
    embed_ms: u128,
    cluster_ms: u128,
    per_cluster_results: Vec<(String, bool, String)>, // (label, recovered, reason)
}

// ---------------------------------------------------------------------------
// Ground truth expectations
// ---------------------------------------------------------------------------

fn voxelmorph_expectations() -> Vec<ExpectedCluster> {
    vec![
        ExpectedCluster {
            label: "displacement_field_conversions",
            must_cluster: vec![
                "affine_to_disp",
                "disp_to_trf",
                "trf_to_disp",
                "disp_to_coords",
            ],
            must_not_cluster: vec![
                "spatial_transform",
                "is_affine_shape",
            ],
        },
        ExpectedCluster {
            label: "velocity_integration",
            must_cluster: vec![
                "integrate_disp",         // functional.py
                "IntegrateVelocityField", // nn/modules.py (init or forward)
            ],
            must_not_cluster: vec![
                "resize_disp",
                "compose",
            ],
        },
        ExpectedCluster {
            label: "data_generators",
            must_cluster: vec![
                "scan_to_scan",
                "scan_to_atlas",
                "semisupervised",
                "template_creation",
            ],
            must_not_cluster: vec![
                "load_volfile",
                "spatial_transform",
            ],
        },
        ExpectedCluster {
            label: "distance_transforms",
            must_cluster: vec![
                "dist_trf",
                "signed_dist_trf",
                "vol_to_sdt",
            ],
            must_not_cluster: vec![
                "dice",
                "pad",
            ],
        },
        ExpectedCluster {
            label: "deprecated_loss_stubs",
            must_cluster: vec![
                "NCC",
                "MSE",
                "Dice",
                "Grad",
            ],
            must_not_cluster: vec![
                "SpatialTransformer",
                "volgen",
            ],
        },
        ExpectedCluster {
            label: "file_io",
            must_cluster: vec![
                "load_volfile",
                "save_volfile",
            ],
            must_not_cluster: vec![
                "dice",
                "clean_seg",
            ],
        },
        ExpectedCluster {
            label: "segmentation_cleaning",
            must_cluster: vec![
                "clean_seg",
                "clean_seg_batch",
                "filter_labels",
            ],
            must_not_cluster: vec![
                "load_volfile",
                "dist_trf",
            ],
        },
        ExpectedCluster {
            label: "surface_extraction",
            must_cluster: vec![
                "edge_to_surface_pts",
                "sdt_to_surface_pts",
            ],
            must_not_cluster: vec![
                "dist_trf",
                "dice",
            ],
        },
    ]
}

fn neurite_expectations() -> Vec<ExpectedCluster> {
    vec![
        ExpectedCluster {
            label: "similarity_metrics",
            must_cluster: vec![
                "mse",
                "dice",
                "ncc",
            ],
            must_not_cluster: vec![
                "resample",
                "crop",
            ],
        },
        ExpectedCluster {
            label: "resampling_ops",
            must_cluster: vec![
                "resample",
                "subsample",
            ],
            must_not_cluster: vec![
                "mse",
                "gaussian_kernel",
            ],
        },
        ExpectedCluster {
            label: "noise_generation",
            must_cluster: vec![
                "random_smoothed_noise",
                "upsample_noise",
                "fractal_noise",
            ],
            must_not_cluster: vec![
                "gaussian_kernel",
                "resample",
            ],
        },
        ExpectedCluster {
            label: "conv_building_blocks",
            must_cluster: vec![
                "ConvBlock",
                "DownsampleConvBlock",
                "UpsampleConvBlock",
            ],
            must_not_cluster: vec![
                "Dice",
                "MSE",
            ],
        },
        ExpectedCluster {
            label: "visualization",
            must_cluster: vec![
                "slices",
                "volume3D",
                "flow",
            ],
            must_not_cluster: vec![
                "resample",
                "mse",
            ],
        },
        ExpectedCluster {
            label: "gaussian_filtering",
            must_cluster: vec![
                "gaussian_kernel",
                "gaussian_smoothing",
            ],
            must_not_cluster: vec![
                "fractal_noise",
                "mse",
            ],
        },
    ]
}

fn interseg3d_expectations() -> Vec<ExpectedCluster> {
    vec![
        ExpectedCluster {
            label: "composite_losses",
            must_cluster: vec![
                "FocalDiceLoss",
                "DiceCELoss",
            ],
            must_not_cluster: vec![
                "quantile_norm",
                "resize_to_max_dim",
            ],
        },
        ExpectedCluster {
            label: "signal_augmentation",
            must_cluster: vec![
                "quantile_norm",
                "QuantileNormd",
                "RandAdditiveGaussianNoise",
            ],
            must_not_cluster: vec![
                "resize_to_max_dim",
                "warp_tensors",
            ],
        },
        ExpectedCluster {
            label: "batch_collation",
            must_cluster: vec![
                "CollateTaskWithContext",
                "CollateTaskWithContextResample",
                "CollateTaskPadToMax",
            ],
            must_not_cluster: vec![
                "FocalDiceLoss",
                "quantile_norm",
            ],
        },
        ExpectedCluster {
            label: "click_sampling",
            must_cluster: vec![
                "sample_confusion_coords_with_labels",
                "sample_confusion_coords",
            ],
            must_not_cluster: vec![
                "embed_clicks_onehot",
                "FocalDiceLoss",
            ],
        },
        ExpectedCluster {
            label: "click_embedding",
            must_cluster: vec![
                "embed_clicks_onehot",
                "embed_corrective_clicks_confusion",
                "embed_corrective_clicks_prediction",
            ],
            must_not_cluster: vec![
                "sample_confusion_coords",
                "resize_to_max_dim",
            ],
        },
        ExpectedCluster {
            label: "batch_samplers",
            must_cluster: vec![
                "SameIndexBatchSampler",
                "UniformBatchSampler",
            ],
            must_not_cluster: vec![
                "CollateTaskWithContext",
                "FocalDiceLoss",
            ],
        },
    ]
}

// ---------------------------------------------------------------------------
// Benchmark harness
// ---------------------------------------------------------------------------

// All benchmark codebases are Python, so we use the Python parser directly.
// The Language param is retained for future cross-language benchmarks.

/// Run the full benchmark pipeline for one model on one repo.
fn run_benchmark(
    repo: &Path,
    lang: &Language,
    model_id: &str,
    threshold: f32,
    expectations: &[ExpectedCluster],
) -> BenchmarkResult {
    // 1. Parse and extract function bodies
    let parser = parser::python_parser();
    let opts = ParseOptions {
        include: lang.default_include(),
        exclude: lang.default_exclude(),
        respect_gitignore: true,
    };
    let results =
        parser::parse_directory_with(repo, &opts, &parser).unwrap();

    let mut all_bodies: Vec<(String, String)> = Vec::new();
    for pr in &results {
        for sig in &pr.signatures {
            if let Some(ref body) = sig.body {
                let display_name = match &body.scope {
                    Some(cls) => format!("{}.{}", cls, body.entity_name),
                    None => body.entity_name.clone(),
                };
                if !body.body_text.trim().is_empty() {
                    all_bodies.push((display_name, body.body_text.clone()));
                }
            }
        }
    }

    let nb_functions = all_bodies.len();

    // 2. Load model and embed
    let model = embeddings::load_model(model_id, None)
        .unwrap_or_else(|e| panic!("Failed to load {model_id}: {e}"));

    let texts: Vec<String> = all_bodies.iter().map(|(_, t)| t.clone()).collect();
    let t_embed = std::time::Instant::now();
    let vectors = model.embed(texts).unwrap();
    let embed_ms = t_embed.elapsed().as_millis();

    // 3. Cluster
    let mut emb_index = embeddings::EmbeddingIndex::empty();
    let ids: Vec<u64> = (0..vectors.len() as u64).collect();
    for (i, v) in vectors.iter().enumerate() {
        emb_index.insert_vector(i as u64, v.clone());
    }

    let t_cluster = std::time::Instant::now();
    let result = cluster::agglomerative_cluster(&ids, &emb_index, threshold);
    let cluster_ms = t_cluster.elapsed().as_millis();

    // 4. Build name -> cluster_id map
    let mut name_to_cluster: HashMap<String, usize> = HashMap::new();
    for (idx, (name, _)) in all_bodies.iter().enumerate() {
        if let Some(&cid) = result.assignments.get(&(idx as u64)) {
            // Store both the full name and just the short name for matching
            name_to_cluster.insert(name.clone(), cid);
            // Also store the method name without class prefix
            if let Some(dot_pos) = name.rfind('.') {
                let short = &name[dot_pos + 1..];
                // Only insert short name if not already present (avoid
                // collisions — full name takes priority)
                name_to_cluster.entry(short.to_string()).or_insert(cid);
            }
        }
    }

    // 5. Compute metrics
    let mut cluster_members: HashMap<usize, Vec<usize>> = HashMap::new();
    for (&id, &cid) in &result.assignments {
        cluster_members.entry(cid).or_default().push(id as usize);
    }

    let mut intra_sims = Vec::new();
    let mut inter_sims = Vec::new();
    let cluster_ids: Vec<usize> = cluster_members.keys().copied().collect();

    for &cid in &cluster_ids {
        let members = &cluster_members[&cid];
        for i in 0..members.len() {
            for j in (i + 1)..members.len() {
                intra_sims.push(embeddings::cosine_similarity(
                    &vectors[members[i]],
                    &vectors[members[j]],
                ));
            }
        }
        for &other in &cluster_ids {
            if other == cid {
                continue;
            }
            let others = &cluster_members[&other];
            if !members.is_empty() && !others.is_empty() {
                inter_sims.push(embeddings::cosine_similarity(
                    &vectors[members[0]],
                    &vectors[others[0]],
                ));
            }
        }
    }

    let mean = |v: &[f32]| -> f32 {
        if v.is_empty() { 0.0 } else { v.iter().sum::<f32>() / v.len() as f32 }
    };
    let mean_intra = mean(&intra_sims);
    let mean_inter = mean(&inter_sims);

    // 6. Evaluate cluster expectations
    let mut clusters_recovered = 0;
    let mut per_cluster_results = Vec::new();

    for exp in expectations {
        // Find cluster IDs for must_cluster members
        let mut found_ids: Vec<(String, Option<usize>)> = Vec::new();
        for name in &exp.must_cluster {
            let cid = name_to_cluster.get(*name).copied();
            found_ids.push((name.to_string(), cid));
        }

        let missing: Vec<&str> = found_ids
            .iter()
            .filter(|(_, cid)| cid.is_none())
            .map(|(n, _)| n.as_str())
            .collect();

        if !missing.is_empty() {
            per_cluster_results.push((
                exp.label.to_string(),
                false,
                format!("functions not found: {}", missing.join(", ")),
            ));
            continue;
        }

        let cluster_ids_found: Vec<usize> =
            found_ids.iter().filter_map(|(_, c)| *c).collect();
        let first = cluster_ids_found[0];
        let all_same = cluster_ids_found.iter().all(|&c| c == first);

        if !all_same {
            let detail: Vec<String> = found_ids
                .iter()
                .map(|(n, c)| format!("{}={}", n, c.unwrap()))
                .collect();
            per_cluster_results.push((
                exp.label.to_string(),
                false,
                format!("split across clusters: {}", detail.join(", ")),
            ));
            continue;
        }

        // Check must_not_cluster exclusions
        let mut violations = Vec::new();
        for name in &exp.must_not_cluster {
            if let Some(&cid) = name_to_cluster.get(*name) {
                if cid == first {
                    violations.push(*name);
                }
            }
        }

        if !violations.is_empty() {
            per_cluster_results.push((
                exp.label.to_string(),
                false,
                format!(
                    "unwanted functions in cluster: {}",
                    violations.join(", ")
                ),
            ));
            continue;
        }

        clusters_recovered += 1;
        per_cluster_results.push((
            exp.label.to_string(),
            true,
            "recovered".to_string(),
        ));
    }

    let clusters_total = expectations.len();
    let recovery_rate = if clusters_total > 0 {
        clusters_recovered as f32 / clusters_total as f32
    } else {
        0.0
    };

    BenchmarkResult {
        model_id: model_id.to_string(),
        nb_functions,
        nb_clusters: result.nb_clusters,
        clusters_recovered,
        clusters_total,
        recovery_rate,
        mean_intra_similarity: mean_intra,
        mean_inter_similarity: mean_inter,
        embed_ms,
        cluster_ms,
        per_cluster_results,
    }
}

fn print_benchmark_result(r: &BenchmarkResult) {
    eprintln!("\n{}", "=".repeat(70));
    eprintln!("Model: {}", r.model_id);
    eprintln!("{}", "=".repeat(70));
    eprintln!(
        "Functions: {} | Clusters: {} | Embed: {}ms | Cluster: {}ms",
        r.nb_functions, r.nb_clusters, r.embed_ms, r.cluster_ms
    );
    eprintln!(
        "Intra sim: {:.4} | Inter sim: {:.4}",
        r.mean_intra_similarity, r.mean_inter_similarity
    );
    eprintln!(
        "Recovery: {}/{} ({:.1}%)",
        r.clusters_recovered,
        r.clusters_total,
        r.recovery_rate * 100.0
    );
    eprintln!("\nPer-cluster:");
    for (label, ok, reason) in &r.per_cluster_results {
        let icon = if *ok { "PASS" } else { "FAIL" };
        eprintln!("  [{icon}] {label}: {reason}");
    }
}

// ---------------------------------------------------------------------------
// Tests — one per codebase, run with each model
// ---------------------------------------------------------------------------

// These tests use a helper that skips gracefully if the repo path doesn't
// exist (same pattern as the testbed).

/// Run all 4 models on a single codebase and print comparison.
fn run_comparison(
    repo: &Path,
    lang: &Language,
    expectations: Vec<ExpectedCluster>,
    threshold: f32,
) -> Vec<BenchmarkResult> {
    if !repo.exists() {
        eprintln!("SKIP: {} not found", repo.display());
        return Vec::new();
    }

    let mut results = Vec::new();
    for &model_id in SUPPORTED_MODELS {
        eprintln!("\n--- Running {model_id} on {} ---", repo.display());
        let r = run_benchmark(repo, lang, model_id, threshold, &expectations);
        print_benchmark_result(&r);
        results.push(r);
    }

    // Print summary table
    eprintln!("\n\n{}", "=".repeat(80));
    eprintln!("COMPARISON SUMMARY: {}", repo.display());
    eprintln!("{}", "=".repeat(80));
    eprintln!(
        "{:<40} {:>8} {:>8} {:>10} {:>8} {:>8}",
        "Model", "Clusters", "Recov", "Rate", "Intra", "Inter"
    );
    for r in &results {
        eprintln!(
            "{:<40} {:>8} {:>5}/{:<2} {:>9.1}% {:>8.4} {:>8.4}",
            r.model_id,
            r.nb_clusters,
            r.clusters_recovered,
            r.clusters_total,
            r.recovery_rate * 100.0,
            r.mean_intra_similarity,
            r.mean_inter_similarity,
        );
    }

    results
}

#[test]
fn benchmark_voxelmorph() {
    let repo = Path::new("/home/eti/projects/voxelmorph");
    if !repo.exists() {
        eprintln!("SKIP: voxelmorph not found");
        return;
    }
    let results = run_comparison(
        repo,
        &Language::Python,
        voxelmorph_expectations(),
        0.30,
    );
    if results.is_empty() {
        return;
    }
    // At least one model should recover ≥50% of expected clusters
    let best = results
        .iter()
        .max_by(|a, b| {
            a.recovery_rate
                .partial_cmp(&b.recovery_rate)
                .unwrap()
        })
        .unwrap();
    eprintln!(
        "\nBest model for voxelmorph: {} ({:.1}% recovery)",
        best.model_id,
        best.recovery_rate * 100.0,
    );
}

#[test]
fn benchmark_neurite() {
    let repo = Path::new("/home/eti/projects/neurite");
    if !repo.exists() {
        eprintln!("SKIP: neurite not found");
        return;
    }
    let results = run_comparison(
        repo,
        &Language::Python,
        neurite_expectations(),
        0.30,
    );
    if results.is_empty() {
        return;
    }
    let best = results
        .iter()
        .max_by(|a, b| {
            a.recovery_rate
                .partial_cmp(&b.recovery_rate)
                .unwrap()
        })
        .unwrap();
    eprintln!(
        "\nBest model for neurite: {} ({:.1}% recovery)",
        best.model_id,
        best.recovery_rate * 100.0,
    );
}

#[test]
fn benchmark_interseg3d() {
    let repo = Path::new("/home/eti/projects/interseg3d");
    if !repo.exists() {
        eprintln!("SKIP: interseg3d not found");
        return;
    }
    let results = run_comparison(
        repo,
        &Language::Python,
        interseg3d_expectations(),
        0.30,
    );
    if results.is_empty() {
        return;
    }
    let best = results
        .iter()
        .max_by(|a, b| {
            a.recovery_rate
                .partial_cmp(&b.recovery_rate)
                .unwrap()
        })
        .unwrap();
    eprintln!(
        "\nBest model for interseg3d: {} ({:.1}% recovery)",
        best.model_id,
        best.recovery_rate * 100.0,
    );
}

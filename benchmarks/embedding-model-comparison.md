# Embedding Model Comparison for Behavioral Clustering

## Overview

Compared 4 embedding models for clustering raw function body text from Python codebases. Goal: determine which model produces the best behavioral clusters for ontomics' L4 layer.

## Candidates

| Model | HF ID | Params | Dim | candle module | Status |
|-------|-------|--------|-----|---------------|--------|
| BGE-small | BAAI/bge-small-en-v1.5 | 33M | 384 | bert | Working |
| Jina Code v2 | jinaai/jina-embeddings-v2-base-code | 161M | 768 | jina_bert | Incompatible |
| CodeRankEmbed | nomic-ai/CodeRankEmbed | 137M | 768 | nomic_bert | Working |
| GTE-ModernBERT | Alibaba-NLP/gte-modernbert-base | 149M | 768 | modernbert | Incompatible |

### Compatibility Issues

- **Jina Code v2**: candle's `jina_bert` module expects fused `gated_layers` weight but the model ships `up_gated_layer` + `down_layer` separately. Architecture version mismatch.
- **GTE-ModernBERT**: candle's `modernbert` module expects `model.embeddings.tok_embeddings.weight` but the model uses different weight naming. Prefix/naming mismatch.

Both could be fixed with weight remapping or custom candle module patches, but this is out of scope for the spike.

## Method

1. Parse codebases with tree-sitter, extract function body text (both standalone functions and class methods via `Signature.body`)
2. Embed all bodies with each model
3. Run agglomerative clustering (average linkage) at 6 thresholds: 0.20, 0.25, 0.30, 0.35, 0.40, 0.50
4. Evaluate against ground truth clusters defined by manual code analysis
5. Report best threshold per model

## Ground Truth Clusters

Defined 20 expected clusters across 3 codebases:

- **voxelmorph** (8 clusters): displacement field conversions, velocity integration, data generators, distance transforms, deprecated loss stubs, file I/O, segmentation cleaning, surface extraction
- **neurite** (6 clusters): similarity metrics, resampling, noise generation, conv building blocks, visualization, gaussian filtering
- **interseg3d** (6 clusters): composite losses, signal augmentation, batch collation, click sampling, click embedding, batch samplers

A cluster is "recovered" when all must_cluster members share a cluster_id AND no must_not_cluster members appear in that cluster.

## Results

### Quantitative

| Codebase | Functions | BGE-small | CodeRankEmbed |
|----------|-----------|-----------|---------------|
| voxelmorph | 80 | 4/8 (50%) at t=0.25 | **5/8 (62.5%)** at t=0.50 |
| neurite | 126 | 1/6 (17%) at t=0.25 | **2/6 (33%)** at t=0.30 |
| interseg3d | 1209 | 1/6 (17%) at t=0.20 | 1/6 (17%) at t=0.20 |

### Embedding Quality Metrics

| Metric | BGE-small | CodeRankEmbed |
|--------|-----------|---------------|
| Mean intra-cluster similarity (voxelmorph) | 0.816 | 0.624 |
| Mean inter-cluster similarity (voxelmorph) | 0.638 | 0.229 |
| Separation ratio (intra/inter) | 1.28 | 2.72 |
| Optimal clusters (80 funcs) | 23 | 28 |

### Inference Speed (RTX 3070 GPU)

| Model | voxelmorph (80) | neurite (126) | interseg3d (1209) |
|-------|-----------------|---------------|-------------------|
| BGE-small | 3.4s | 5.7s | 22.9s |
| CodeRankEmbed | 17.2s | 18.7s | 20.4s |

### Failure Analysis

**BGE-small failure mode: over-merging.** At its best threshold (0.25), BGE produces only 23 clusters for 80 functions. Unrelated functions end up in the same cluster because BGE's embedding space compresses code text into a narrow similarity band (inter-cluster similarity 0.638). The "data generators" cluster works because generators have highly distinctive Python syntax (while True, yield).

**CodeRankEmbed failure mode: over-fragmenting at low thresholds, clean at higher thresholds.** At t=0.30 (52 clusters for 80 functions), CodeRank separates too aggressively. At t=0.50 (28 clusters), it finds a sweet spot with good discrimination. Its embedding space has much better separation (inter-cluster similarity 0.229), meaning the distance threshold can be tuned more precisely.

**Shared failures:**
- `displacement_field_conversions`: `spatial_transform` (a warping function) ends up clustered with conversion functions because they share similar code patterns (tensor operations on displacement fields)
- `velocity_integration`: `resize_disp` and `compose` are incorrectly included because they share the `spatial_transform` call pattern
- Class-based expectations in interseg3d: many class names (FocalDiceLoss, QuantileNormd, etc.) aren't found because the parser outputs method names like `FocalDiceLoss.forward`, not class names directly

## Key Findings

1. **CodeRankEmbed is the better model for behavioral clustering.** It wins or ties on all 3 codebases. Its embedding space has 2x better separation between clusters (intra/inter ratio 2.72 vs 1.28), giving more headroom for threshold tuning.

2. **Optimal threshold differs by model.** BGE-small needs 0.25, CodeRankEmbed needs 0.50. This is because code-trained models produce more spread-out similarity distributions. The default 0.75 used by ontomics' concept clustering is too high for both.

3. **The threshold should be model-specific configuration.** Current config `embeddings.similarity_threshold` applies to both concept and behavioral clustering. For v0.3.0, consider separate thresholds for concept embedding (BGE, t~0.75) vs behavioral embedding (CodeRank, t~0.50).

4. **Neither model is perfect at behavioral clustering.** Best recovery is 62.5% on a curated set. Some behavioral similarities are semantic (same purpose, different implementation), which pure text embedding can't capture. Improving beyond ~70% likely requires hybrid approaches (AST structure + text embedding).

5. **candle compatibility is a bottleneck.** 2 of 4 candidates failed due to weight naming mismatches. The candle ecosystem moves faster than model releases — newer model architectures may not have matching candle modules, or module implementations may target different weight versions.

6. **GPU is essential for iteration speed.** CPU embedding: 76-317s per model per codebase. GPU: 2-23s. The `candle-nn/cuda` feature must be enabled alongside `candle-core/cuda` for layer-norm GPU acceleration.

## Recommendation

**Ship v0.3.0 with CodeRankEmbed (nomic-ai/CodeRankEmbed)** as the behavioral embedding model, with threshold 0.50.

Keep BGE-small for concept-level embedding (it's fast and the concept pipeline is already tuned for it).

The `EmbeddingModel` trait from this spike becomes the production abstraction, making future model swaps straightforward.

### Follow-up

- Fix candle weight loading for Jina Code v2 and GTE-ModernBERT to complete the comparison
- Tune interseg3d expectations to use actual method names from parser output
- Investigate hybrid embedding approaches for the ~30% of clusters that text-only embedding misses
- Consider separate `embeddings.behavioral_threshold` config field

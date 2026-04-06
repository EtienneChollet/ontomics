# Embedding Model Comparison for Behavioral Clustering

## Overview

Compared 4 embedding models for clustering raw function body text from Python codebases. Goal: determine which model produces the most discriminative embedding space for ontomics' L4 behavioral layer.

## Candidates

| Model | HF ID | Params | Dim | candle module | Status |
|-------|-------|--------|-----|---------------|--------|
| BGE-small | BAAI/bge-small-en-v1.5 | 33M | 384 | bert | Working |
| Jina Code v2 | jinaai/jina-embeddings-v2-base-code | 161M | 768 | jina_bert | Collapsed embeddings |
| CodeRankEmbed | nomic-ai/CodeRankEmbed | 137M | 768 | nomic_bert | Working |
| GTE-ModernBERT | Alibaba-NLP/gte-modernbert-base | 149M | 768 | modernbert | Working |

### Compatibility Issues (resolved)

Both Jina Code v2 and GTE-ModernBERT initially failed to load due to weight naming mismatches between candle's module implementations and the actual safetensors checkpoints. Both were fixed with key remapping at load time:

- **Jina Code v2**: candle's `jina_bert` expects `gated_layers`, `wo`, `mlp.layernorm` — model ships `up_gated_layer`, `down_layer`, `layer_norm_2`. Shapes are identical; pure name translation via `load_jina_code_remapped()`.
- **GTE-ModernBERT**: candle's `modernbert` hardcodes a `model.` prefix in all weight paths — model ships keys without the prefix. Fixed by prepending `model.` to all backbone keys via `load_gte_modern_remapped()`.

**Jina Code v2 produces degenerate embeddings** despite loading successfully. The model variant (`jina-bert-v2-qk-post-norm`) requires QK post-normalization layers (`layer_norm_q`, `layer_norm_k`) and a pre-attention `layer_norm_1` that candle's `jina_bert` module does not implement. Without these norms, attention outputs are poorly scaled and all embeddings collapse to near-identical vectors (inter-cluster similarity 0.916). Fixing this requires writing a new attention struct, not just key remapping.

## Method

1. Parse codebases with tree-sitter, extract function body text (both standalone functions and class methods via `Signature.body`)
2. Embed all bodies with each model
3. Run agglomerative clustering (average linkage) at 6 distance thresholds: 0.20, 0.25, 0.30, 0.35, 0.40, 0.50
4. Evaluate using intrinsic embedding quality metrics (separation ratio, cluster distribution)

## Evaluation Philosophy

The original experiment included hand-defined ground truth clusters evaluated by recovery rate. This approach was dropped because:

- Ground truth encodes one person's subjective interpretation of behavioral similarity
- The actual use case is query-based (`find_similar_symbols`, L4 layer browsing), not taxonomy-based
- Whether two functions "belong together" depends on what the consumer is asking, not a predefined grouping
- Models were penalized for finding valid similarities the ground truth didn't anticipate

**Evaluation now uses intrinsic metrics only**: separation ratio (intra/inter cluster similarity), cluster count distribution, and embedding speed. These measure whether the embedding space has useful structure without imposing a subjective taxonomy.

## Results

### Embedding Quality Metrics (voxelmorph, 80 functions)

| Metric | BGE-small | Jina Code v2 | CodeRankEmbed | GTE-ModernBERT |
|--------|-----------|--------------|---------------|----------------|
| Mean intra-cluster similarity | 0.816 | 0.976 | 0.624 | 0.858 |
| Mean inter-cluster similarity | 0.633 | 0.916 | 0.231 | 0.516 |
| Separation ratio (intra/inter) | 1.29 | collapsed | **2.70** | 1.66 |
| Clusters at optimal threshold | 23 | 1 | 28 | 42 |

### Embedding Quality Metrics (neurite, 126 functions)

| Metric | BGE-small | CodeRankEmbed | GTE-ModernBERT |
|--------|-----------|---------------|----------------|
| Mean intra-cluster similarity | 0.771 | 0.794 | 0.738 |
| Mean inter-cluster similarity | 0.624 | 0.258 | 0.455 |
| Separation ratio | 1.24 | **3.08** | 1.62 |

(Jina OOM on neurite and interseg3d due to single-text batching + large corpus)

### Inference Speed (RTX 3070 GPU)

| Model | voxelmorph (80) | neurite (126) | interseg3d (1209) |
|-------|-----------------|---------------|-------------------|
| BGE-small | 3.4s | 5.7s | 22.9s |
| Jina Code v2 | 1.6s | OOM | OOM |
| CodeRankEmbed | 2.0s | 18.7s | 24.5s |
| GTE-ModernBERT | 7.5s | 18.0s | OOM |

### Failure Mode Analysis

**BGE-small: over-merging.** Inter-cluster similarity 0.633 means even unrelated functions look 63% similar. The embedding space is compressed into a narrow similarity band, leaving little room for threshold tuning. Works when function syntax is highly distinctive (Python generators with `yield`), fails on semantic distinctions.

**Jina Code v2: embedding collapse.** All vectors near-identical. Even at distance threshold 0.05 (similarity >= 0.95), 70 of 80 functions land in one cluster. Caused by missing QK post-normalization in candle's `jina_bert`. Unusable.

**CodeRankEmbed: best separation.** Inter-cluster similarity 0.231 means unrelated functions genuinely look different. The 2.7x separation ratio gives wide room for threshold tuning — similarity scores are discriminative at any threshold.

**GTE-ModernBERT: decent but weaker than CodeRankEmbed.** Separation ratio 1.66 is better than BGE (1.29) but well below CodeRankEmbed (2.70). Despite having the highest CoIR benchmark score (79.3), its general-purpose training doesn't produce as discriminative an embedding space for code behavior as CodeRankEmbed's contrastive code-specific training.

## Key Findings

1. **CodeRankEmbed produces the most discriminative embedding space for code behavioral similarity.** Separation ratio 2.70 (voxelmorph) and 3.08 (neurite) — roughly 2x better than any other model. This means its similarity scores carry real information: high similarity genuinely indicates shared behavioral patterns.

2. **Separation ratio is the right evaluation metric.** It measures whether the embedding space has useful structure without imposing subjective behavioral categories. A high separation ratio means similarity queries return meaningful results at any threshold.

3. **Optimal distance threshold is model-specific.** BGE needs ~0.25, CodeRankEmbed needs ~0.50, GTE needs ~0.20. Code-trained models produce more spread-out similarity distributions. The threshold should be a configurable parameter, not a constant.

4. **candle compatibility is a bottleneck.** 1 of 4 candidates (Jina) was architecturally incompatible even after weight remapping. The candle ecosystem covers common architectures but newer model variants with novel attention patterns may not work without custom module implementations.

5. **GPU is essential for iteration speed.** CPU embedding: 76-317s per model per codebase. GPU: 2-25s.

## Decision

**Ship v0.3.0 with CodeRankEmbed (`nomic-ai/CodeRankEmbed`)** as the behavioral embedding model, with distance threshold ~0.50.

Keep BGE-small for concept-level embedding (fast, concept pipeline already tuned for it).

The `EmbeddingModel` trait from this experiment becomes the production abstraction, making future model swaps straightforward.

### Follow-up

- Investigate instruction-prefix embedding ("Cluster by behavioral purpose: ...") for potential separation improvement
- Consider separate `embeddings.behavioral_threshold` config field
- Evaluate whether CodeRankEmbed batch size can be increased beyond 1 for faster inference
- Jina Code v2 requires a custom `JinaBertV2Attention` candle struct to be usable — out of scope for v0.3.0

# Benchmark: "What are the main domain concepts in this codebase?"

Repo: [ScribblePrompt](https://github.com/halleewong/ScribblePrompt) | Model: Claude Sonnet

## With ontomics (1 tool call, ~3.7k tokens, ~5s)

```
> What are the main domain concepts in this codebase?

⏺ ontomics - list_concepts (MCP)(top_k: 20)
  ⎿  { ... +118 lines }

⏺ ScribblePrompt is an interactive image segmentation codebase. The top domain
  concepts:

  ┌─────────────┬─────────────┬────────────────────────────────────────────────────┐
  │   Concept   │ Occurrences │                        Role                        │
  ├─────────────┼─────────────┼────────────────────────────────────────────────────┤
  │ mask        │ 171         │ Segmentation output — binary/multi-label masks     │
  ├─────────────┼─────────────┼────────────────────────────────────────────────────┤
  │ scribble    │ 110         │ Core interaction type — drawn stroke annotations   │
  ├─────────────┼─────────────┼────────────────────────────────────────────────────┤
  │ random      │ 109         │ Random prompt generation for training augmentation │
  ├─────────────┼─────────────┼────────────────────────────────────────────────────┤
  │ click       │ 98          │ Point-based interaction (positive/negative clicks) │
  ├─────────────┼─────────────┼────────────────────────────────────────────────────┤
  │ image / img │ 70/59       │ Input 2D images                                   │
  ├─────────────┼─────────────┼────────────────────────────────────────────────────┤
  │ coords      │ 70          │ Spatial coordinates for interactions               │
  ├─────────────┼─────────────┼────────────────────────────────────────────────────┤
  │ box         │ 51          │ Bounding box prompts                               │
  ├─────────────┼─────────────┼────────────────────────────────────────────────────┤
  │ channelwise │ 51          │ Per-channel mask/scribble operations               │
  ├─────────────┼─────────────┼────────────────────────────────────────────────────┤
  │ embed       │ 49          │ Model embeddings (likely SAM-style image encoder)  │
  ├─────────────┼─────────────┼────────────────────────────────────────────────────┤
  │ points      │ 45          │ Point prompt collections                           │
  └─────────────┴─────────────┴────────────────────────────────────────────────────┘

  Core idea: The codebase generates and handles interactive segmentation prompts —
  scribbles, clicks, boxes, and points — applied to images to produce masks. The
  heavy random presence suggests significant synthetic prompt generation for training
  data. This follows the SAM-style (Segment Anything) paradigm but with scribble
  interactions as the primary modality.
```

## Without ontomics (26 tool calls, ~61.6k tokens, 56s)

```
> What are the main domain concepts in this codebase?

⏺ Explore(Explore ScribblePrompt codebase structure)
  ⎿  Done (26 tool uses · 61.6k tokens · 35s)

⏺ Here are the main domain concepts:

  Core Domain: Interactive Medical Image Segmentation

  Problem: Segment arbitrary anatomical structures in medical images using minimal
  user interaction.

  ---
  Concepts

  Prompt types — the 3 interaction modalities users/models work with:
  - Scribbles — freehand strokes on foreground/background regions (WarpScribble)
  - Clicks — point annotations (positive/negative) (RandomClick)
  - Bounding boxes — spatial bounds around a target structure (boxes.py)

  Prompt encoding — raw interactions are converted to 5-channel tensors fed to the
  model: [image | fg_scribble | bg_scribble | click_map | prev_mask] (embed.py)

  Iterative refinement — segmentation improves over rounds of interaction; models
  consume their previous prediction as input, enabling correction loops
  (SuperFlexiblePrompt)

  Prompt simulation — synthetic prompts are generated from ground-truth masks during
  training so the model learns from realistic interaction patterns
  (SuperFlexiblePrompt, WarpScribble)

  ---
  Two Model Architectures

  ┌───────┬────────────────────┬───────────────────┐
  │       │ ScribblePromptUNet │ ScribblePromptSAM │
  ├───────┼────────────────────┼───────────────────┤
  │ Input │ 128×128            │ 1024×1024         │
  ├───────┼────────────────────┼───────────────────┤
  │ Base  │ Custom UNet        │ Segment Anything  │
  ├───────┼────────────────────┼───────────────────┤
  │ Speed │ Real-time          │ Higher accuracy   │
  └───────┴────────────────────┴───────────────────┘

  ---
  Supporting Concepts

  - MedScribble — released benchmark dataset: 14 biomedical tasks, 64 images,
    3 annotators
  - Augmentation pipeline — geometry + label augmentations composed from YAML config
    (SegmentationSequential)
  - FocalDiceLoss — combined loss for class-imbalanced binary segmentation
  - Experiment framework — training loop built on pylot with YAML configs in configs/

✻ Sautéed for 56s
```

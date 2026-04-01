# Benchmark: "What does 'transform' mean in this codebase?"

Repo: [voxelmorph](https://github.com/voxelmorph/voxelmorph) | Model: Claude Sonnet

## With ontomics (1 tool call, ~3.7k tokens, 5s)

```
> What does 'transform' mean in this codebase?

⏺ ontomics - query_concept (MCP)(term: "transform")
  ⎿  { ... +110 lines }

⏺ In this codebase, transform (abbreviated trf) refers to a spatial deformation
  field — a dense vector field that maps every voxel in one image to a location in
  another image's coordinate space. It's the core concept of VoxelMorph.

  Two flavors

  ┌────────────────────┬──────────────────────────────────────────────────────┬────────────────┐
  │        Type        │                       Meaning                        │  isdisp flag   │
  ├────────────────────┼──────────────────────────────────────────────────────┼────────────────┤
  │ Displacement field │ Relative offset from identity (each voxel stores     │ True (default) │
  │                    │ how far to move)                                     │                │
  ├────────────────────┼──────────────────────────────────────────────────────┼────────────────┤
  │ Coordinate field   │ Absolute sampling coordinates                        │ False          │
  └────────────────────┴──────────────────────────────────────────────────────┴────────────────┘

  They're interconvertible via disp_to_coords / coords_to_disp.

  Key functions

  - spatial_transform (voxelmorph/nn/functional.py:16, voxelmorph/functional.py:596)
    — The workhorse. Takes an image (B, C, *spatial) and a trf tensor, warps the
    image via grid_sample. Handles both displacement and affine transforms
    (auto-detected via is_affine_shape).
  - random_transform (voxelmorph/nn/functional.py:352, voxelmorph/functional.py:1399)
    — Generates random spatial transforms by combining affine (translation, rotation,
    scaling) and nonlinear (fractal-based) warps.
  - SpatialTransformer (voxelmorph/nn/modules.py:18) — The nn.Module wrapper around
    spatial_transform, used inside models like VxmPairwise.

  Related concepts

  - affine — A linear transform represented as a matrix; converted to a displacement
    field via affine_to_disp before spatial sampling.
  - disp — Short for displacement, the most common transform representation.
  - vel — Velocity field, integrated (via integrate_disp) to produce a diffeomorphic
    displacement.
  - compose — Chains two transforms by using spatial_transform to warp one field by
    another.

  Naming convention

  The abbreviation trf is the standard parameter/variable name for a transform tensor
  throughout the codebase. Conversion functions follow the x_to_y pattern: disp_to_trf,
  affine_to_disp, params_to_affine, etc.
```

## Without ontomics (19 tool calls, ~76k tokens, 1m 15s)

```
> What does 'transform' mean in this codebase?

⏺ Explore(Explore transform usage in voxelmorph)
  ⎿  Done (19 tool uses · 76.3k tokens · 57s)

⏺ In VoxelMorph, "transform" (trf) has a specific meaning distinct from
  "displacement" (disp). Here's the breakdown:

  Displacement field (disp): A vector field where each voxel stores the offset from
  its original position. A zero displacement field means "don't move anything."
  Shape: (ndim, *spatial).

  Transform / deformation field (trf): A vector field where each voxel stores the
  absolute coordinate it should sample from. An identity transform field equals the
  identity meshgrid (each voxel points to itself). Shape: (ndim, *spatial).

  The relationship is simple:
  - trf = disp + identity_grid (disp_to_trf)
  - disp = trf - identity_grid (trf_to_disp)

  spatial_transform is the core operation — it takes an image and a field
  (displacement, affine, or coordinate field) and warps the image by sampling from
  the specified locations. This is the "pull" / "backward" convention: for each
  output voxel, the field says where to read from in the input.

  SpatialTransformer is the nn.Module wrapper around spatial_transform for use in
  network architectures (e.g., VxmPairwise).

  random_transform generates random affine matrices for data augmentation.

  So "transform" in this codebase means absolute sampling coordinates (as opposed to
  relative offsets), and spatial_transform is the verb form — the act of warping an
  image using any supported field type.

✻ Baked for 1m 6s
```

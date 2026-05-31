# Schema Configs

These JSON files describe fixed nested spatial-index schemas used by the early point-cloud experiments.

The default schema-search candidate set is intentionally finite:

- `quadtree_default`
- `octree_default`
- `karras_octree_default`
- `kdtree_default`
- `bih_default`
- `bvh_default`
- `lbvh_default`
- `regular_grid_default`
- `hgrid_default`
- `quadtree_octree`
- `octree_kdtree`
- `urban_hybrid_quadtree_octree_kdtree`
- `gpu_mixed_all`

Supported `type` values at this stage:

- `QuadTree`
- `Octree`
- `KarrasOctree`
- `KDTree`
- `BIH`
- `BVH`
- `LBVH`
- `RegularGrid`
- `HGrid`

`numLevels` is the number of node-depth levels assigned to that schema block. For example:

```json
[
  { "type": "Octree", "numLevels": 1 },
  { "type": "BVH", "numLevels": 3 }
]
```

maps depth `0` to `Octree`, then depths `1`, `2`, and `3` to `BVH`. Depths beyond the configured total reuse the final block only as a defensive fallback; normal builds stop at the configured total or `buildPolicy.maxDepth`.

`QuadTree` levels support an optional `axisPolicy`. Point-cloud schemas default to `xy`, which splits X/Y and leaves Z unsplit. Explicit values are `xy`, `xz`, `yz`, `ignore_shortest`, `ignore_x`, `ignore_y`, and `ignore_z`.

`KarrasOctree`, `BIH`, and `LBVH` are GPU-flavored schema names that preserve base Octree, KDTree, and BVH CPU fallback families while letting CUDA builders pick more specific split behavior. `RegularGrid` and `HGrid` are implemented by the CPU point index as recursive grid split levels and by CUDA as both MixedTree split levels and standalone global cell-bin evaluators. Point-cloud builds use single-child assignment and should keep `allowOverlapDuplication` set to `false`.

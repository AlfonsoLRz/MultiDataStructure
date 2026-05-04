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

`KarrasOctree`, `BIH`, `LBVH`, `RegularGrid`, and `HGrid` are GPU-flavored schema names that preserve the base Octree, KDTree, and BVH families for CPU fallback while letting the CUDA MixedTree builder pick the more specific split behavior. Inside a mixed schema, `RegularGrid` and `HGrid` are recursive per-node grid split levels; the standalone CUDA `regular_grid` and `hgrid` evaluators still use their own global cell-bin implementations. Point-cloud builds use single-child assignment and should keep `allowOverlapDuplication` set to `false`.

# Schema Configs

These JSON files describe fixed nested spatial-index schemas used by the early point-cloud experiments.

The default schema-search candidate set is intentionally finite:

- `quadtree_default`
- `octree_default`
- `kdtree_default`
- `quadtree_octree`
- `octree_kdtree`
- `urban_hybrid_quadtree_octree_kdtree`

Supported `type` values at this stage:

- `QuadTree`
- `Octree`
- `KDTree`
- `BVH`

`Grid2D` and `Grid3D` are still future schema types. Point-cloud builds use single-child assignment and should keep `allowOverlapDuplication` set to `false`.

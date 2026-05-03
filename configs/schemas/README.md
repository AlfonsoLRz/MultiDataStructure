# Schema Configs

These JSON files describe fixed nested spatial-index schemas used by the early point-cloud experiments.

Supported `type` values at this stage:

- `QuadTree`
- `Octree`
- `KDTree`
- `BVH`

`Grid2D` and `Grid3D` are still future schema types. Point-cloud builds use single-child assignment and should keep `allowOverlapDuplication` set to `false`.


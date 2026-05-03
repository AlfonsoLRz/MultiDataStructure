# Experiment Protocol

Schema-search output is the training table for the first schema selector. Each raw row represents one `(dataset, workload, schema)` candidate evaluation and contains:

- dataset identity and point count,
- deterministic point-cloud features,
- deterministic workload features,
- schema identity,
- build metrics,
- query metrics,
- score components and final score.

## Point-Cloud Features

Point-cloud features are computed once per dataset and copied into every schema candidate row for that dataset.

For large clouds, feature extraction uses a deterministic reservoir sample with a fixed seed and logs `feature_sample_size`. Bounding-box features still come from the full cloud stats.

Current point features:

| Column | Meaning |
|---|---|
| `feature_sample_size` | Number of points used for sampled features. |
| `bbox_x`, `bbox_y`, `bbox_z` | Bounding-box extents. |
| `aspect_xy`, `aspect_xz`, `aspect_yz` | Safe extent ratios. |
| `density_bbox` | Points per bbox volume, or area/extent fallback for flat clouds. |
| `height_mean`, `height_std`, `height_range` | Sampled height distribution plus bbox height range. |
| `cov_eig_0`, `cov_eig_1`, `cov_eig_2` | Descending eigenvalues of sampled XYZ covariance. |
| `linearity` | `(eig0 - eig1) / eig0`. |
| `planarity` | `(eig1 - eig2) / eig0`. |
| `scattering` | `eig2 / eig0`. |
| `occupancy_ratio_8` | Occupied cells in an 8x8x8 grid divided by 512. |
| `occupancy_entropy_8` | Shannon entropy of occupied cell counts. |
| `density_cv_8` | Coefficient of variation across occupied cell counts. |
| `verticality_score` | Heuristic score for tall/vertical structure. |
| `flatness_score` | Heuristic score for terrain-like flat structure. |

## Workload Features

Workload features are computed from the workload profile and score weights:

| Column | Meaning |
|---|---|
| `w_range`, `w_radius`, `w_knn` | Normalized query mix weights. |
| `knn_k` | K for KNN queries. |
| `num_queries` | Total generated queries in the profile. |
| `query_scale_mean`, `query_scale_std` | Deterministic generated range/radius scale summary. |
| `build_weight`, `memory_weight` | Score weights used for build time and memory. |

The best-schema CSV keeps these feature columns so it can be used directly for supervised learning labels.

## Schema Features

The Python training pipeline augments raw rows with schema-composition features parsed from each schema JSON:

| Column | Meaning |
|---|---|
| `schema_has_quadtree` | Candidate includes at least one QuadTree block. |
| `schema_has_octree` | Candidate includes at least one Octree block. |
| `schema_has_kdtree` | Candidate includes at least one KDTree block. |
| `schema_has_grid2d` | Candidate includes at least one Grid2D block. |
| `schema_has_grid3d` | Candidate includes at least one Grid3D block. |
| `schema_num_blocks` | Number of configured structure blocks. |
| `schema_total_levels` | Total scheduled levels across blocks. |
| `schema_max_leaf_capacity` | Maximum leaf capacity across blocks. |
| `schema_min_leaf_capacity` | Minimum nonzero leaf capacity across blocks. |

These features are the first bridge from fixed candidates to generated schema combinations.

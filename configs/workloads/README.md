# Workload Profiles

Schema-search workload profiles define a deterministic mix of generated point queries.

Supported query weights:

- `aabb_range`
- `radius`
- `knn`

`numQueries` is the total number of queries in the generated profile, and `knnK` controls the KNN neighbor count. The schema-search runner resets the random seed for each dataset/workload/schema comparison so candidate schemas see the same query stream.

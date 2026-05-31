# Workload Profiles

Schema-search workload profiles define a deterministic mix of generated point queries.

Supported query weights:

- `aabb_range`
- `radius`
- `knn`

`numQueries` is the total number of queries in the generated profile, and `knnK` controls the KNN neighbor count. The schema-search runner resets the random seed for each dataset/workload/schema comparison so candidate schemas see the same query stream.

Optional `scoreWeights` make the scoring protocol part of the workload definition:

```json
{
  "scoreWeights": {
    "latency": 1.0,
    "buildTime": 0.0,
    "memory": 0.0,
    "imbalance": 0.0,
    "visitedNodes": 0.0,
    "testedPoints": 0.0
  }
}
```

By default the score is query latency only. Positive `visitedNodes` or `testedPoints` switches that workload to the existing visit-proxy score; otherwise `latency`, `buildTime`, `memory`, and `imbalance` are combined with the measured metrics. CLI score flags override workload-local weights.

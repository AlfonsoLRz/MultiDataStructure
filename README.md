# MultiDataStructure

Experimental nested spatial data structures for point-cloud indexing and schema search.

## Current Notes

- The maintained path is the point-cloud workload: load `.las`, `.ply`, `.xyz`, or `.csv`, build a schema-driven CPU index, and run range/count/radius/KNN benchmarks.
- Schema search can use CUDA builders for measured confirmation, but CUDA KNN is currently labeled `bruteforce_gpu_scan`: it scans the GPU point buffer and should not be reported as tree-accelerated KNN traversal.
- More detailed behavior lives in `docs/current_behavior.md` and command examples in `docs/command_reference.md`.

# Linux (headless) build

The root `CMakeLists.txt` builds the command-line executable for Linux: every mode except
`--mode gui` (tests, `points`, `schema-search`, `batch-search`, `evaluate-one`, `engine-check`).
It defines `MDS_HEADLESS`, which makes `stdafx.h` skip GL/GLFW/ImGui/ImGuizmo/SOIL2, drops the
GL interop part of `GPUStructs.h`, and turns `--mode gui` into an error. The GUI, the legacy
triangle/BVH code (`Bvh.cu`, `bvh_kernels.cu`, `morton_encoding_kernels.cu`, `cuda_globals.cu`,
`KdTree.cpp`, `Octree.cpp`, `QuadTree.cpp`, `GPUStructs.cpp`) and ONNX Runtime are left out.
The Windows build (`MultiDataStructure.sln`) is unchanged.

Dependencies: GCC 12 (nvcc 12.2 accepts host GCC up to 12), CUDA 12.2 nvcc + cudart + cub,
Boost >= 1.75 (Boost.JSON), glm (header-only; Windows uses 1.0.1), OpenMP, CMake >= 3.24.

## makinon2 layout

| Path | Contents |
| --- | --- |
| `/home2/alfonsolr/mds/src` | source (shipped with `git archive`, not cloned) |
| `/home2/alfonsolr/mds/build` | CMake build tree; the binary is `build/MultiDataStructure` |
| `/home2/alfonsolr/mds/data`, `/home2/alfonsolr/mds/traces` | datasets and traces |
| conda env `mds-build` | compilers, nvcc, Boost, glm, CMake, Ninja |

The system CUDA 12.x directories on makinon2 only hold runtime libraries (no `nvcc`), so nvcc
comes from conda-forge. Install it through the `cuda-nvcc-impl` package, not the `cuda-nvcc`
metapackage: the metapackage pulls in `gcc_linux-64`, whose activation scripts export
`CFLAGS`/`CXXFLAGS` (`-march=nocona -O2 ...`) that CMake would pick up.

## Create the environment (once)

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda create -y -n mds-build -c conda-forge --override-channels \
    "gcc=12" "gxx=12" "libboost-devel>=1.75" "glm=1.0.1" cmake ninja \
    "cuda-version=12.2" "cuda-nvcc-impl=12.2.140" "cuda-cudart-dev=12.2" "cuda-cccl=12.2"
```

## Ship the source (from Windows)

```powershell
git -c core.autocrlf=false archive --format=tar -o mds-src.tar HEAD
scp mds-src.tar makinon2:/home2/alfonsolr/mds/
ssh makinon2 "mkdir -p /home2/alfonsolr/mds/src && tar -xf /home2/alfonsolr/mds/mds-src.tar -C /home2/alfonsolr/mds/src"
```

## Configure and build

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mds-build
cmake -S /home2/alfonsolr/mds/src -B /home2/alfonsolr/mds/build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER="$CONDA_PREFIX/bin/gcc" \
    -DCMAKE_CXX_COMPILER="$CONDA_PREFIX/bin/g++" \
    -DCMAKE_CUDA_COMPILER="$CONDA_PREFIX/bin/nvcc" \
    -DCMAKE_CUDA_HOST_COMPILER="$CONDA_PREFIX/bin/g++" \
    -DCMAKE_PREFIX_PATH="$CONDA_PREFIX"
cmake --build /home2/alfonsolr/mds/build -j 48
```

CUDA code targets `sm_75` (RTX 2080 Ti, Quadro RTX 6000); pass `-DCMAKE_CUDA_ARCHITECTURES=...`
for other GPUs. The binary has an RPATH into the env, so running it does not need the env
activated.

## Run

Run from the source root: tests and schema/workload paths such as `configs/schemas/octree.json`
resolve against the working directory (and its ancestors).

```bash
cd /home2/alfonsolr/mds/src
/home2/alfonsolr/mds/build/MultiDataStructure --run-tests      # prints "Baseline tests passed"
/home2/alfonsolr/mds/build/MultiDataStructure --mode engine-check \
    --input /path/to/cloud.mdspc \
    --schemas "configs/schemas/octree.json;configs/schemas/kdtree.json" \
    --queries 1500 --query-seed 1337 --warmup 1 --measure-repeats 1 \
    --query-csv q.csv --summary-csv s.csv
```

CUDA numbers devices fastest-first, so device 0 on makinon2 is the Quadro RTX 6000, not the
2080 Ti that `nvidia-smi` lists first. Use `CUDA_VISIBLE_DEVICES` to pin one.

## Why `-ffp-contract=off`

Results must be bit-identical to the Windows build, which compiles with MSVC `/fp:precise`.
Since VS 2022 that mode never fuses `a*b + c` into a fused multiply-add (only `/fp:contract`
allows it). GCC does fuse by default
(`-ffp-contract=fast`) whenever the target ISA has FMA (for example under `-march=native` or
`-march=skylake-avx512`). An FMA rounds once instead of twice, so a squared distance or a split
coordinate can change in the last bit, and a point that sits exactly on a query radius or a split
plane then lands on the other side. That shows up as a different `result_checksum` or
`structure_digest`. With `-mavx2` alone GCC has no FMA instructions to use, but the flag keeps
the guarantee if someone adds `-march=...` later. For the same reason, never add `-ffast-math`.
The flag is also passed to the host compiler for `.cu` files. Device code keeps nvcc's default
`--fmad=true`, as on Windows.

`-mavx2 -mbmi -mpopcnt` match `/arch:AVX2` and enable the AVX2 leaf scans in
`PointSpatialIndex.cpp` (`_tzcnt_u32` needs BMI and `_mm_popcnt_u32` needs POPCNT under GCC).

Cross-platform check (SanAndreas 1M, octree + kdtree, 1500 queries, seed 1337): `engine-check`
on Linux and Windows gave identical `structure_digest`, `returned_points` and
`result_checksum` on 3000/3000 rows. When both binaries came from the same source, the traversal
counters (`visited_nodes`, `tested_boxes`, `tested_points`) were identical too.

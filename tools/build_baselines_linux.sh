#!/bin/bash
# Builds the external replay baselines (testing.md 2r) on Linux with the engine's code-generation
# flags (CMakeLists.txt: AVX2/BMI/POPCNT, no FMA contraction). Header-only libraries (nanoflann,
# unibn, Boost.Geometry) are compiled with these flags; PCL and Open3D link the conda-forge builds
# as distributed.
#
# Needs the user-space conda env `mds-baselines` (pcl, nanoflann, open3d, eigen, boost, gcc 12) and
# the unibn header in third_party/unibn_octree.
#   tools/build_baselines_linux.sh <out_dir> [third_party_dir]
set -e
OUT=${1:?usage: build_baselines_linux.sh <out_dir> [third_party_dir]}
THIRD=${2:-$(dirname "$0")/../../third_party}
SRC=$(cd "$(dirname "$0")" && pwd)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate mds-baselines
P=$CONDA_PREFIX
mkdir -p "$OUT"
BASE="-O3 -DNDEBUG -std=c++17 -ffp-contract=off -pthread -I$SRC -Wl,-rpath,$P/lib -L$P/lib"
FLAGS="$BASE -mavx2 -mbmi -mpopcnt"
# PCL and Open3D pass Eigen types across their library boundary and were built without AVX, so
# their drivers must use the same alignment (PCL warns of an aligned-malloc mismatch otherwise);
# their search code runs in the precompiled libraries either way.
LIBFLAGS="$BASE"

# OpenMP for the --threads throughput mode (testing.md 2x); the per-query mode stays single-threaded.
g++ $FLAGS -fopenmp -I$P/include "$SRC/nanoflann_point_baseline.cpp" -o "$OUT/nanoflann_point_baseline"
g++ $FLAGS -I$P/include "$SRC/boost_rtree_point_baseline.cpp" -o "$OUT/boost_rtree_point_baseline"
g++ $FLAGS -I"$THIRD/unibn_octree" "$SRC/unibn_point_baseline.cpp" -o "$OUT/unibn_point_baseline"
g++ $LIBFLAGS -I$P/include/pcl-1.15 -I$P/include/eigen3 -I$P/include "$SRC/pcl_point_baseline.cpp" \
	-lpcl_octree -lpcl_kdtree -lpcl_common -lflann_cpp -llz4 -o "$OUT/pcl_point_baseline"
g++ $LIBFLAGS -I$P/include -I$P/include/eigen3 -I$P/include/open3d/3rdparty "$SRC/open3d_point_baseline.cpp" \
	-lOpen3D -o "$OUT/open3d_point_baseline"
echo "built into $OUT"

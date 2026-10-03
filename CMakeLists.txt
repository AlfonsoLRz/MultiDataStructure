# Headless Linux build of the MultiDataStructure CLI (tests, schema-search, batch-search,
# evaluate-one, engine-check, points). The GUI and the legacy triangle/BVH code are left out;
# Windows keeps building through MultiDataStructure.sln. See docs/linux_build.md.
cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

project(MultiDataStructure LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

find_package(Boost 1.75 CONFIG REQUIRED COMPONENTS json)
find_package(OpenMP REQUIRED COMPONENTS CXX)
find_package(Threads REQUIRED)
find_package(CUDAToolkit REQUIRED)

# glm is header-only; point GLM_INCLUDE_DIR at it if it is not on the default include path.
find_path(GLM_INCLUDE_DIR glm/glm.hpp HINTS ENV CONDA_PREFIX PATH_SUFFIXES include)
if(NOT GLM_INCLUDE_DIR)
	message(FATAL_ERROR "glm not found; set GLM_INCLUDE_DIR")
endif()

set(MDS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/MultiDataStructure)

set(MDS_CXX_SOURCES
	${MDS_DIR}/main.cpp
	${MDS_DIR}/AABB.cpp
	${MDS_DIR}/AppConfig.cpp
	${MDS_DIR}/CudaHelper.cpp
	${MDS_DIR}/MultiDataStructure.cpp
	${MDS_DIR}/Ray.cpp
	${MDS_DIR}/core/BuildPolicy.cpp
	${MDS_DIR}/core/Config.cpp
	${MDS_DIR}/experiments/BatchSearch.cpp
	${MDS_DIR}/experiments/EngineCheck.cpp
	${MDS_DIR}/experiments/EvaluationCache.cpp
	${MDS_DIR}/experiments/FeatureExtraction.cpp
	${MDS_DIR}/experiments/Metrics.cpp
	${MDS_DIR}/experiments/QueryTrace.cpp
	${MDS_DIR}/experiments/SchemaSearch.cpp
	${MDS_DIR}/experiments/SchemaSelector.cpp
	${MDS_DIR}/experiments/SurrogateAcquisition.cpp
	${MDS_DIR}/experiments/ThresholdRefiner.cpp
	${MDS_DIR}/workloads/points/BIH.cpp
	${MDS_DIR}/workloads/points/HGrid.cpp
	${MDS_DIR}/workloads/points/PointBenchmark.cpp
	${MDS_DIR}/workloads/points/PointCloud.cpp
	${MDS_DIR}/workloads/points/PointPrimitive.cpp
	${MDS_DIR}/workloads/points/PointSpatialIndex.cpp
	${MDS_DIR}/workloads/points/SyntheticPointClouds.cpp
)

file(GLOB MDS_TEST_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)

set(MDS_CUDA_SOURCES
	${MDS_DIR}/workloads/points/KDTree.cu
	${MDS_DIR}/workloads/points/LBVH.cu
	${MDS_DIR}/workloads/points/MixedTree.cu
	${MDS_DIR}/workloads/points/Octree.cu
	${MDS_DIR}/workloads/points/QuadTree.cu
	${MDS_DIR}/workloads/points/RegularGrid.cu
)

add_executable(MultiDataStructure ${MDS_CXX_SOURCES} ${MDS_TEST_SOURCES} ${MDS_CUDA_SOURCES})

target_include_directories(MultiDataStructure PRIVATE ${MDS_DIR} ${GLM_INCLUDE_DIR})
target_compile_definitions(MultiDataStructure PRIVATE MDS_HEADLESS)
target_link_libraries(MultiDataStructure PRIVATE
	Boost::json OpenMP::OpenMP_CXX Threads::Threads CUDA::cudart_static)

# Mirrors the vcxproj: relocatable device code and a 64-register cap for the CUDA kernels.
set_target_properties(MultiDataStructure PROPERTIES
	CUDA_SEPARABLE_COMPILATION ON
	CUDA_RESOLVE_DEVICE_SYMBOLS ON)

# Host float arithmetic must stay bit-identical to MSVC /fp:precise, which (VS 2022+) never fuses a*b+c.
# GCC fuses into FMA by default whenever the target has FMA (-ffp-contract=fast), which would
# move points that sit exactly on a query boundary; hence -ffp-contract=off and no -ffast-math.
# AVX2/BMI/POPCNT match /arch:AVX2 and enable the leaf-scan intrinsics in PointSpatialIndex.cpp.
target_compile_options(MultiDataStructure PRIVATE
	$<$<COMPILE_LANGUAGE:CXX>:-mavx2 -mbmi -mpopcnt -ffp-contract=off>
	$<$<COMPILE_LANGUAGE:CUDA>:--maxrregcount=64 -Xcompiler=-ffp-contract=off>)

#pragma once

#include <cuda.h>
#define GLM_FORCE_CUDA

#define IMGUI_DEFINE_MATH_OPERATORS
#define GLM_ENABLE_EXPERIMENTAL

#include <algorithm>
#include <any>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstring>
#include <ctime>
#include <cstdint>
#include <cstdlib>
#include <execution>
#include <iostream>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <limits.h>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <stdio.h>
#include <time.h>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// IMPORTANTE: El include de GLEW debe estar siempre ANTES de el de GLFW
#include "GL/glew.h"								
#include "GLFW/glfw3.h"
#include "glm/glm.hpp"
#include "glm/gtc/constants.hpp"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/ext/vector_relational.hpp"  
#include "glm/gtc/epsilon.hpp"   
#include "glm/gtx/norm.hpp"

// Gui
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "ImGuizmo.h"

// Textures
#include "SOIL2/SOIL2.h"

// Cuda
#include "cooperative_groups.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda_texture_types.h"
#include "texture_fetch_functions.h"

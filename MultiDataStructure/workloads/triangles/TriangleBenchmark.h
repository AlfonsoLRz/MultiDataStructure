#pragma once

#include "../../stdafx.h"

namespace TriangleBenchmark
{
	inline constexpr const char* DEFAULT_SCENE_PATH = "C:/Datasets/models/CornellBox/CornellKnightDragon.obj";
	inline constexpr glm::uint DEFAULT_RANDOM_SEED = 1337;

	struct Options
	{
		std::string benchmark = "default";
		std::string scenePath = DEFAULT_SCENE_PATH;
		std::string outputPath;
		bool pauseAtEnd = true;
	};

	int run(const Options& options);
}


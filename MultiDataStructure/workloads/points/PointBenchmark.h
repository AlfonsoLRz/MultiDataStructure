#pragma once

#include "../../stdafx.h"

namespace PointBenchmark
{
	inline constexpr const char* DEFAULT_SCHEMA_PATH = "configs/schemas/octree.json";

	struct Options
	{
		std::string inputPath;
		std::string schemaPath = DEFAULT_SCHEMA_PATH;
		std::string outputPath;
		bool useBinaryCache = true;
		bool rebuildBinaryCache = false;
		bool pauseAtEnd = true;
	};

	int run(const Options& options);
}

#pragma once

#include "../../stdafx.h"

namespace PointBenchmark
{
	inline constexpr const char* DEFAULT_SCHEMA_PATH = "configs/schemas/octree.json";

	struct Options
	{
		std::string inputPath;
		std::string schemaPath = DEFAULT_SCHEMA_PATH;
		std::vector<std::string> schemaPaths;
		std::string outputPath;
		std::string csvPath;
		std::string queryTracePath;
		std::string modelPath;
		std::string workloadProfilePath;
		bool useBinaryCache = true;
		bool rebuildBinaryCache = false;
		bool pauseAtEnd = true;
		size_t queryCount = 0;
		size_t queryK = 8;
		uint32_t querySeed = 1337;
		bool enableLeafMicroIndexes = false;
		size_t leafMicroIndexThreshold = 512;
	};

	int run(const Options& options);
}

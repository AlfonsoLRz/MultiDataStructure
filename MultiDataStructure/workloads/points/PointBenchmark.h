#pragma once

#include "../../stdafx.h"

namespace PointBenchmark
{
	inline constexpr const char* DEFAULT_SCHEMA_PATH = "configs/schemas/octree.json";

	struct Options
	{
		std::string _inputPath;
		std::string _schemaPath = DEFAULT_SCHEMA_PATH;
		std::vector<std::string> _schemaPaths;
		std::string _outputPath;
		std::string _csvPath;
		std::string _queryTracePath;
		std::string _modelPath;
		std::string _workloadProfilePath;
		bool _useBinaryCache = true;
		bool _rebuildBinaryCache = false;
		bool _pauseAtEnd = true;
		size_t _queryCount = 0;
		size_t _queryK = 8;
		uint32_t _querySeed = 1337;
		bool _enableLeafMicroIndexes = false;
		size_t _leafMicroIndexThreshold = 512;
	};

	int run(const Options& options);
}

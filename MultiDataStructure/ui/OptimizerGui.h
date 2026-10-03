#pragma once

#include "../stdafx.h"

class AppConfig;

namespace OptimizerGui
{
	// Opens the optimizer window. `config` (optional) pre-fills the input cloud and trace from
	// the command line, e.g. `MultiDataStructure.exe --gui --input cloud.las --input-trace t.csv`.
	int run(const AppConfig* config = nullptr);
}

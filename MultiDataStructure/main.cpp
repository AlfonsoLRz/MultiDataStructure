#include "stdafx.h"

#include "AppConfig.h"
#include "experiments/BatchSearch.h"
#include "experiments/SchemaSearch.h"
#include "ui/OptimizerGui.h"
#include "workloads/points/PointBenchmark.h"
#include "../tests/BaselineTests.h"

int main(int argc, char* argv[])
{
	try
	{
		const AppConfig config = AppConfig::parse(argc, argv);
		if (config._showHelp)
		{
			AppConfig::printHelp(std::cout);
			return 0;
		}

		if (config.wantsTests())
			return runBaselineTests();

		if (config._mode == "points")
			return PointBenchmark::run(config._pointOptions);

		if (config._mode == "schema-search")
			return Experiments::runSchemaSearch(config._schemaSearchOptions);

		if (config._mode == "batch-search")
			return Experiments::runBatchSearch(config._schemaSearchOptions);

		if (config._mode == "evaluate-one")
			return Experiments::runEvaluateOne(config._schemaSearchOptions);

		if (config._mode == "gui")
			return OptimizerGui::run();

		throw std::invalid_argument("Unsupported mode: " + config._mode + ". Use --mode gui, --mode points, --mode schema-search, --mode batch-search, --mode evaluate-one, or --run-tests");
	}
	catch (const std::exception& exception)
	{
		std::cerr << "Error: " << exception.what() << '\n';
		return 1;
	}
}

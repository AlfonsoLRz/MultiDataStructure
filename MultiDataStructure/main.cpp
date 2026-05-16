#include "stdafx.h"

#include "AppConfig.h"
#include "experiments/SchemaSearch.h"
#include "ui/OptimizerGui.h"
#include "workloads/points/PointBenchmark.h"
#include "../tests/BaselineTests.h"

int main(int argc, char* argv[])
{
	try
	{
		const AppConfig config = AppConfig::parse(argc, argv);
		if (config.showHelp)
		{
			AppConfig::printHelp(std::cout);
			return 0;
		}

		if (config.wantsTests())
			return runBaselineTests();

		if (config.mode == "points")
			return PointBenchmark::run(config.pointOptions);

		if (config.mode == "schema-search")
			return Experiments::runSchemaSearch(config.schemaSearchOptions);

		if (config.mode == "evaluate-one")
			return Experiments::runEvaluateOne(config.schemaSearchOptions);

		if (config.mode == "gui")
			return OptimizerGui::run();

		throw std::invalid_argument("Unsupported mode: " + config.mode + ". Use --mode gui, --mode points, --mode schema-search, --mode evaluate-one, or --run-tests");
	}
	catch (const std::exception& exception)
	{
		std::cerr << "Error: " << exception.what() << '\n';
		return 1;
	}
}

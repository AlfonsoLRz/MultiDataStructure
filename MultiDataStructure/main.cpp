#include "stdafx.h"

#include "AppConfig.h"
#include "experiments/BatchSearch.h"
#include "experiments/SchemaSearch.h"
#ifndef MDS_HEADLESS
#include "ui/OptimizerGui.h"
#endif
#include "workloads/points/PointBenchmark.h"
#include "workloads/points/PointSpatialIndex.h"
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

		if (config._nodeCostNs > 0.0)
		{
			PointSpatialIndex::setNodeVisitCost(config._nodeCostNs);
			// stderr: evaluate-one writes its JSON to stdout.
			std::cerr << "Node visit cost: " << config._nodeCostNs << " ns (" << PointSpatialIndex::nodeVisitTicks()
					  << " timestamp-counter ticks) per entered node\n";
		}
		PointSpatialIndex::setHugePages(config._poolHugePages);
		PointSpatialIndex::setPrefetchChildren(config._prefetchChildren);
		if (config._poolHugePages || config._prefetchChildren)
			std::cerr << "Memory layout:" << (config._poolHugePages ? " huge pages" : "") << (config._prefetchChildren ? " child prefetch" : "") << '\n';

		if (config._mode == "points")
			return PointBenchmark::run(config._pointOptions);

		if (config._mode == "schema-search")
			return Experiments::runSchemaSearch(config._schemaSearchOptions);

		if (config._mode == "batch-search")
			return Experiments::runBatchSearch(config._schemaSearchOptions);

		if (config._mode == "engine-check")
			return Experiments::runEngineCheck(config.engineCheckOptions());

		if (config._mode == "batch-check")
			return Experiments::runBatchCheck(config.batchCheckOptions());

		if (config._mode == "evaluate-one")
			return Experiments::runEvaluateOne(config._schemaSearchOptions);

		if (config._mode == "gui")
		{
#ifdef MDS_HEADLESS
			throw std::invalid_argument("--mode gui is not available in this headless build; use --mode points, schema-search, batch-search, evaluate-one, engine-check, batch-check, or --run-tests");
#else
			return OptimizerGui::run(&config);
#endif
		}

		throw std::invalid_argument("Unsupported mode: " + config._mode + ". Use --mode gui, --mode points, --mode schema-search, --mode batch-search, --mode evaluate-one, --mode engine-check, --mode batch-check, or --run-tests");
	}
	catch (const std::exception& exception)
	{
		std::cerr << "Error: " << exception.what() << '\n';
		return 1;
	}
}

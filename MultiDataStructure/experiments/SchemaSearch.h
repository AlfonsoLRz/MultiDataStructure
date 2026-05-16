#pragma once

#include "../stdafx.h"
#include "../core/Config.h"
#include "FeatureExtraction.h"
#include "Metrics.h"

namespace Experiments
{
	class EvaluationCache;
	struct SchemaSearchRecord;

	struct SchemaCandidate
	{
		std::string name;
		std::string path;
		SchemaConfig config;
		bool generated = false;
		// Marks canonical single-block controls (pure QuadTree, Octree, KDTree, BVH, etc.).
		// Baselines are force-promoted through every auto-condition stage so the operator can
		// always see how the naive structures perform on the full cloud, even if their proxy-stage
		// rank is poor.
		bool isBaseline = false;
	};

	struct SchemaGenerationOptions
	{
		size_t count = 0;
		size_t minBlocks = 1;
		size_t maxBlocks = 3;
		size_t maxDepth = 12;
		size_t minLeafCapacity = 32;
		size_t maxLeafCapacity = 32768;
		bool conditionalLevels = false;
		double conditionalProbability = 0.35;
		uint32_t seed = 1337;
		std::string outputDirectory = "results/generated_schemas";
	};

	struct ConditionDomain
	{
		std::vector<size_t> pointThresholds;
		std::vector<double> densityThresholds;
		std::vector<double> heightRatioThresholds;
		std::vector<double> extentXThresholds;
		std::vector<double> extentYThresholds;
		std::vector<double> extentZThresholds;
		size_t samplePoints = 0;
		size_t sketchNodes = 0;
		bool estimatedFromCloud = false;
	};

	struct AutoConditionOptions
	{
		bool enabled = false;
		size_t proxyCandidateCount = 256;
		size_t proxyPointCap = 262144;
		size_t proxyQueryCount = 8;
		size_t finalTopK = 16;
		size_t confirmationTopK = 4;
		std::string outputDirectory = "results/auto_conditions";
		std::string selectorOutputPath = "models/local_schema_selector.json";
	};

	struct WorkloadProfile
	{
		std::string name = "mixed";
		double rangeWeight = 0.4;
		double radiusWeight = 0.3;
		double knnWeight = 0.3;
		double rangeScaleMin = 0.01;
		double rangeScaleMax = 0.05;
		double radiusScaleMin = 0.01;
		double radiusScaleMax = 0.04;
		size_t numQueries = 1000;
		size_t knnK = 16;
		uint32_t querySeed = 1337;
	};

	struct ScoreWeights
	{
		double lambdaBuild = 0.0;
		double lambdaMemory = 0.0;
		double lambdaImbalance = 0.0;
		// When useVisitProxy is set, the score is computed from cheap deterministic kernel counters
		// (averageVisitedNodes + visitProxyAlpha * averageTestedPoints) instead of wall-clock query
		// latency. Useful for screening stages where many candidates must be ranked without
		// committing to a noisy latency measurement.
		bool useVisitProxy = false;
		double visitProxyAlpha = 0.1;
	};

	// One rung in a successive-halving / multi-fidelity schedule. The batch flows R0 -> R1 -> ... ;
	// each rung evaluates its input candidates at its own fidelity, then keeps the top
	// `advanceTopK` to feed the next rung. The final rung's ranked output is the archive update.
	struct RungSpec
	{
		std::string name = "rung";
		// 0 means "use the workload profile's own numQueries". A positive value overrides it for
		// this rung only (e.g. 4 for the cheap proxy, 64 for confirmation).
		size_t queryCountOverride = 0;
		// When true, this rung scores candidates by the cheap deterministic visit-proxy
		// (averageVisitedNodes + alpha * averageTestedPoints) instead of wall-clock latency.
		bool useVisitProxy = false;
		double visitProxyAlpha = 0.1;
		// Number of candidates to promote to the next rung. 0 = promote all (only meaningful for
		// the last rung).
		size_t advanceTopK = 0;
	};

	struct RungSchedule
	{
		// When empty, the optimizer falls back to today's single-rung behavior.
		std::vector<RungSpec> rungs;
		// If set, points to an exported selector model JSON used between rungs to sample fresh
		// genomes via the surrogate (Bayesian acquisition step). Empty disables surrogate
		// proposals; the GA's mutation/immigration path stays untouched in that case.
		std::string surrogateModelPath;
		// Number of fresh candidates the surrogate is asked to propose between rungs. Ignored
		// when surrogateModelPath is empty.
		size_t surrogateProposalsPerStep = 0;
		// Pool size the surrogate ranks down to `surrogateProposalsPerStep`. Larger pool =
		// better acquisition at the cost of cheap surrogate evaluations.
		size_t surrogateCandidatePool = 0;
	};

	struct EvolutionOptions
	{
		bool enabled = false;
		size_t generations = 3;
		size_t populationSize = 64;
		size_t eliteCount = 6;
		double mutationRate = 0.65;
		double randomImmigrationRate = 0.20;
		uint32_t seed = 1337;
		// Multi-fidelity rung schedule applied to each evaluated batch. Empty = today's flat
		// evaluate-all-then-mutate behavior.
		RungSchedule rungSchedule;
	};

	struct CudaEvaluationOptions
	{
		int device = -1;
		std::string builder = "lbvh";
		size_t queryBatchSize = 0;
		size_t memoryBudgetMb = 0;
	};

	struct SchemaSearchOptions
	{
		std::vector<std::string> inputPaths;
		std::vector<std::string> schemaPaths;
		std::vector<std::string> workloadPaths;
		std::string rankModelPath;
		std::string csvPath = "results/schema_search.csv";
		std::string bestCsvPath = "results/schema_search_best.csv";
		bool includeSyntheticDatasets = true;
		bool includeConfiguredSchemas = true;
		bool useBinaryCache = true;
		bool rebuildBinaryCache = false;
		bool pauseAtEnd = false;
		size_t syntheticScale = 512;
		size_t queryCountOverride = 0;
		size_t knnKOverride = 0;
		size_t benchmarkTopK = 0;
		uint32_t querySeed = 1337;
		bool querySeedOverride = false;
		bool deepNestedSearch = false;
		SchemaGenerationOptions generation;
		AutoConditionOptions autoConditions;
		ScoreWeights weights;
		EvolutionOptions evolution;
		std::string evaluator = "cpu";
		CudaEvaluationOptions cuda;
		// Always include canonical single-block schemas (pure QuadTree, Octree, KDTree, BVH, plus
		// GPU-native LBVH, KarrasOctree, RegularGrid, HGrid, BIH) as controls alongside whatever
		// generated/configured candidates are present. Lets the operator confirm that nested
		// candidates actually beat the naive baselines instead of just comparing nested to nested.
		bool includeBaselineSchemas = true;
		std::string scoreCachePath;
		bool rebuildScoreCache = false;
		EvaluationCache* scoreCache = nullptr;
		// Maximum number of CPU candidates to benchmark concurrently in the evolutionary loop.
		// Only honored when the evaluator is "cpu" (the CUDA path is serialised because the GPU
		// state and the per-builder build cache are not thread-safe). Default 1 keeps current
		// behaviour exactly.
		size_t parallelDispatch = 1;
		std::function<void(const SchemaSearchRecord&)> progressCallback;
	};

	struct SchemaSearchRecord
	{
		std::string datasetName;
		std::string datasetSource;
		size_t numPoints = 0;
		std::string workloadName;
		double rangeWeight = 0.0;
		double radiusWeight = 0.0;
		double knnWeight = 0.0;
		size_t numQueries = 0;
		size_t knnK = 0;
		uint32_t querySeed = 0;
		std::string schemaName;
		std::string schemaPath;
		BuildMetrics buildMetrics;
		QueryMetrics queryMetrics;
		size_t rangeQueries = 0;
		size_t countRangeQueries = 0;
		size_t radiusQueries = 0;
		size_t knnQueries = 0;
		double score = 0.0;
		double scoreMemoryMb = 0.0;
		double scoreImbalancePenalty = 0.0;
		ScoreWeights weights;
		PointCloudFeatures pointFeatures;
		WorkloadFeatures workloadFeatures;
		std::string backend = "cpu";
		int cudaDevice = -1;
		std::string cudaBuilder;
		double gpuUploadMs = 0.0;
		double gpuBuildMs = 0.0;
		double gpuQueryMs = 0.0;
		size_t gpuMemoryBytes = 0;
		size_t conditionalLevels = 0;
		size_t conditionFields = 0;
		std::string conditionSummary;
		bool isBaseline = false;
		size_t activeStructureTypes = 0;
		double nestedActiveFraction = 0.0;
		std::string activeStructureSummary;
		std::string bestBaselineSchema;
		double bestBaselineScore = 0.0;
		double relativeSpeedupVsBaseline = 0.0;
	};

	struct EvaluatorResolution
	{
		std::string evaluator = "cpu";
		bool requestedCuda = false;
		bool usingCuda = false;
		bool fellBackToCpu = false;
		std::string warning;
	};

	EvaluatorResolution resolveSchemaSearchEvaluator(
		const std::string& requestedEvaluator,
		bool cudaAvailable,
		const std::string& cudaError = {});
	WorkloadProfile parseWorkloadProfile(const std::string& jsonText, const std::string& sourceName = {});
	WorkloadProfile loadWorkloadProfile(const std::string& filename);
	ConditionDomain estimateConditionDomain(const PointCloud& cloud, size_t maxSamplePoints = 262144);
	std::vector<SchemaCandidate> generateSchemaCandidates(const SchemaGenerationOptions& options);
	std::vector<SchemaCandidate> generateSchemaCandidates(
		const SchemaGenerationOptions& options,
		const ConditionDomain* conditionDomain);
	double computeSchemaSearchScore(
		const BuildMetrics& buildMetrics,
		const QueryMetrics& queryMetrics,
		const ScoreWeights& weights,
		double& memoryMb,
		double& imbalancePenalty);
	std::vector<SchemaSearchRecord> selectBestRecords(const std::vector<SchemaSearchRecord>& records);
	int runSchemaSearch(const SchemaSearchOptions& options);

	// One-shot evaluator: loads exactly one dataset, one schema, and one workload from `options`,
	// runs build + queries, and writes the resulting SchemaSearchRecord as JSON to stdout. Intended
	// to back Python-driven optimizers (Optuna ask/tell, Hyperband) where the C++ exe is launched
	// per candidate. CSV outputs are suppressed regardless of options.csvPath.
	int runEvaluateOne(const SchemaSearchOptions& options);
}

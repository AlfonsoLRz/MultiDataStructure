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
		bool adaptiveLeafCapacity = false;
		double adaptiveLeafProbability = 0.25;
		uint32_t seed = 1337;
		std::string outputDirectory = "results/generated_schemas";
		// "auto" resolves at schema-search startup: CPU discovery gets query_minimal_cpu,
		// CUDA measurement gets cuda_query_full. Explicit values are query_minimal_cpu,
		// cuda_query_full, and all.
		std::string primitiveProfile = "auto";
	};

	struct ConditionDomain
	{
		std::vector<size_t> pointThresholds;
		std::vector<double> densityThresholds;
		std::vector<double> heightRatioThresholds;
		std::vector<double> extentXThresholds;
		std::vector<double> extentYThresholds;
		std::vector<double> extentZThresholds;
		// Phase B3 anisotropy thresholds. Quantiles of `1 - shortExtent / longExtent` computed
		// over the 8x8x8 sketch cells during `estimateConditionDomain`. Empty when the cloud's
		// shape didn't produce enough variation for the quantile estimator (very uniform clouds).
		std::vector<double> anisotropyThresholds;
		// Phase B3 occupancy-entropy thresholds. Per-node entropy is normalized to [0, 1] by
		// dividing by ln(64) (a 4x4x4 sub-grid is used at evaluation time). The domain is
		// populated from a fixed [0.1, 0.9] range plus the cloud's own 8x8x8 normalized entropy
		// so per-cloud tuning still has an anchor.
		std::vector<double> occupancyEntropyThresholds;
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

	struct ScoreWeights
	{
		double lambdaLatency = 1.0;
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
		bool stratifyQueries = false;
		bool hasScoreWeights = false;
		ScoreWeights scoreWeights;
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

	// Forward declaration; full definition lives in ThresholdRefiner.h.
	struct ThresholdRefinementOptions;

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
		// Continuous-threshold post-pass over the GA archive. Disabled by default. The full
		// settings struct (sigma, evaluation budget, etc.) is defined in ThresholdRefiner.h to
		// keep that header self-contained, but the on/off flag and top-K live here so callers
		// don't need to include the refiner header just to wire the GA.
		bool refineThresholds = false;
		size_t refineThresholdsTopK = 4;
		size_t refineThresholdsEvaluations = 60;
		double refineThresholdsSigma0 = 0.3;
		uint32_t refineThresholdsSeed = 1337;
		// Phase B4 diversity controls.
		// Two-parent crossover: probability that a child is built by splicing two elite parents'
		// level lists instead of mutating one. 0 = pure mutation (legacy behavior).
		double crossoverRate = 0.4;
		// NSGA-II elite ranking: when true, elites are picked by non-dominated-sort + crowding
		// distance over the four Pareto objectives instead of the scalar aggregateScore.
		// Intrinsically preserves diversity across the front.
		bool useNsga2Ranking = true;
		// Diagnostic-guided repair mutations. When enabled, each generation first creates a few
		// children from measured archive entries by classifying their build/query bottleneck
		// (high leaf occupancy, high tested-points, high visited-nodes, high full-containment,
		// etc.) and applying a targeted schema edit before falling back to random mutation.
		bool repairMutations = false;
		size_t repairTopK = 4;
		size_t repairPerCandidate = 2;
	};

	struct CudaEvaluationOptions
	{
		int device = -1;
		std::string builder = "lbvh";
		size_t queryBatchSize = 0;
		size_t memoryBudgetMb = 0;
		std::string knnBackend = "auto";
	};

	struct SchemaSearchOptions
	{
		std::vector<std::string> inputPaths;
		std::vector<std::string> schemaPaths;
		std::vector<std::string> workloadPaths;
		std::string rankModelPath;
		std::string csvPath = "results/schema_search.csv";
		std::string bestCsvPath = "results/schema_search_best.csv";
		// Optional Phase C1 Pareto-front CSV. One row per non-dominated candidate per
		// (dataset, workload) group with the `pareto_rank` column. Empty = skip.
		std::string paretoCsvPath;
		// Optional Markdown explanation report. Empty = skip. The report is generated from the
		// measured SchemaSearchRecord rows after baseline annotation.
		std::string explainReportPath;
		// Optional per-query CSV trace. Empty = disabled. When enabled, cached score rows are
		// bypassed so the trace reflects queries actually executed in this run.
		std::string queryTracePath;
		// Phase C2 multi-seed confirmation. When >= 2, the top-K candidates per (dataset, workload)
		// are re-measured with N distinct query seeds and their record fields are updated with
		// mean ± 95% bootstrap CI on (avgLatencyMs, p95LatencyMs, gpuBuildMs). The Pareto
		// dominance check then uses the seed-averaged mean instead of the single-seed point
		// estimate. 0 or 1 disables (one-shot single-seed measurement).
		size_t confirmSeeds = 0;
		size_t confirmTopK = 4;
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
		bool scoreWeightsOverride = false;
		std::string scoreStage = "final";
		bool scoreIsFinalLatency = true;
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
		bool enableLeafMicroIndexes = false;
		size_t leafMicroIndexThreshold = 512;
		// Measurement-reliability repeats. When >= 2, each candidate's query batch is re-timed this
		// many times (same queries, no rebuild) and the per-repeat batch-average latency mean,
		// standard deviation, coefficient of variation, and 95% bootstrap CI are recorded on
		// queryMetrics. 1 keeps the original one-shot timing. This measures timing noise, which is
		// complementary to confirmSeeds (which re-draws the query set).
		size_t measurementRepeats = 1;
		// Optional sidecar CSV for the proxy/latency rank-correlation report (one row per
		// dataset/workload). Empty = stdout summary only. The report is always printed to stdout.
		std::string proxyCorrelationCsvPath;
		// When true (and CUDA is available), after the search build canonical schemas on both the CPU
		// index and the GPU MixedTree and compare per-query returned-point counts on a deterministic
		// range/radius sample. Surfaces CPU<->GPU disagreement that the separate evaluators would
		// otherwise hide. Writes results/parity_report.csv and a stdout summary.
		bool verifyParity = false;
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
		QueryMetrics rangeMetrics;
		QueryMetrics countRangeMetrics;
		QueryMetrics radiusMetrics;
		QueryMetrics knnMetrics;
		size_t rangeQueries = 0;
		size_t countRangeQueries = 0;
		size_t radiusQueries = 0;
		size_t knnQueries = 0;
		std::string queryStrataSummary;
		double score = 0.0;
		double scoreMemoryMb = 0.0;
		double scoreImbalancePenalty = 0.0;
		ScoreWeights weights;
		std::string scoreMode = "latency";
		std::string scoreStage = "final";
		bool scoreIsFinalLatency = true;
		PointCloudFeatures pointFeatures;
		WorkloadFeatures workloadFeatures;
		std::string backend = "cpu";
		std::string knnBackend = "none";
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
		// Phase C1 Pareto-front tagging. -1 = not on the (dataset, workload) front; otherwise the
		// 0-indexed rank by avgLatencyMs among non-dominated peers. Populated by
		// `selectParetoRecords`; unset after a bare measurement run.
		int paretoRank = -1;
		// Phase C2 multi-seed confirmation. `confirmSeedsUsed` > 0 means the three latency / build
		// statistics below were computed across that many distinct query seeds; otherwise they are
		// left at 0.0 and consumers fall back to `queryMetrics.averageLatencyMs` etc.
		size_t confirmSeedsUsed = 0;
		double latencyMean = 0.0;
		double latencyCiLow = 0.0;
		double latencyCiHigh = 0.0;
		double p95LatencyMean = 0.0;
		double p95LatencyCiLow = 0.0;
		double p95LatencyCiHigh = 0.0;
		double gpuBuildMean = 0.0;
		double gpuBuildCiLow = 0.0;
		double gpuBuildCiHigh = 0.0;
		// True when this record's (dataset, workload) winner is separated from its runner-up by
		// non-overlapping latency confidence intervals (see `confidentlyBetter`). Set by
		// `annotateRankingConfidence`. Defaults to false: with single-shot measurement (no
		// --measure-repeats and no --confirm-seeds) confidence cannot be established, which is the
		// honest answer and nudges the operator toward repeated measurement.
		bool rankingConfident = false;
		// True for the Pareto-front knee of this record's (dataset, workload) group: the entry
		// closest to the normalized ideal across (latency, build, memory, imbalance). Set by
		// selectParetoRecords; it is the recommended balanced compromise when no single objective
		// dominates the decision.
		bool paretoKnee = false;
		// Zero-build heuristic estimate of average per-query work (see estimateSchemaQueryCost),
		// computed from the schema + cloud features without constructing the index. Stored so it can
		// be validated against measured latency (reportEstimatedCostCorrelation) and, once trusted,
		// used as a cheap pre-filter on generated candidates.
		double estimatedQueryCost = 0.0;
	};

	struct SchemaRepairDiagnostics
	{
		std::string bottleneck = "balanced";
		double leafOccupancyRatio = 0.0;
		double testedPerVisited = 0.0;
		double visitedPerQuery = 0.0;
		double testedPointFraction = 0.0;
		double fullContainmentRatio = 0.0;
		bool highLeafOccupancy = false;
		bool testedPointDominated = false;
		bool visitedNodeDominated = false;
		bool fullContainmentDominated = false;
		bool likelySingleChildChains = false;
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
	std::string resolvePrimitiveProfile(const std::string& requestedProfile, bool cudaEvaluator);
	WorkloadProfile parseWorkloadProfile(const std::string& jsonText, const std::string& sourceName = {});
	WorkloadProfile loadWorkloadProfile(const std::string& filename);
	ConditionDomain estimateConditionDomain(const PointCloud& cloud, size_t maxSamplePoints = 262144);
	std::vector<SchemaCandidate> generateSchemaCandidates(const SchemaGenerationOptions& options);
	std::vector<SchemaCandidate> generateSchemaCandidates(
		const SchemaGenerationOptions& options,
		const ConditionDomain* conditionDomain);
	SchemaRepairDiagnostics diagnoseSchemaRepair(
		const SchemaCandidate& candidate,
		const std::vector<SchemaSearchRecord>& measuredRecords);
	std::vector<SchemaCandidate> generateSchemaRepairCandidates(
		const SchemaCandidate& candidate,
		const std::vector<SchemaSearchRecord>& measuredRecords,
		const SchemaGenerationOptions& generationOptions,
		const ConditionDomain* conditionDomain,
		size_t maxCandidates,
		uint32_t seed,
		const std::string& outputDirectory);
	double computeSchemaSearchScore(
		const BuildMetrics& buildMetrics,
		const QueryMetrics& queryMetrics,
		const ScoreWeights& weights,
		double& memoryMb,
		double& imbalancePenalty);

	// Writes `schema` as a generated JSON under `outputDirectory` (empty = no file, candidate
	// gets a generated:* pseudo path) and returns a SchemaCandidate with a name derived from the
	// canonical schema signature plus `namePrefix`. Exposed publicly so the threshold refiner
	// (and any future external generators) can produce unique replayable candidates without
	// duplicating the JSON serializer.
	SchemaCandidate materializeSchemaCandidate(
		SchemaConfig schema,
		const std::string& namePrefix,
		const std::string& outputDirectory);
	std::vector<SchemaSearchRecord> selectBestRecords(const std::vector<SchemaSearchRecord>& records);

	// True when candidate `a` is confidently lower-latency than `b`: a's upper latency CI lies
	// strictly below b's lower latency CI, so the ordering is not an artifact of measurement noise.
	// CI source priority: multi-seed confirmation CI (confirmSeedsUsed > 0) first, then measurement-
	// repeat CI (queryMetrics.measurementRepeats > 1). With neither available the intervals collapse
	// to the point estimate and the function returns false (no confidence claimable).
	bool confidentlyBetter(const SchemaSearchRecord& a, const SchemaSearchRecord& b);

	// Sets `rankingConfident` on every record: for each (dataset, workload) group, true iff the
	// group's best candidate is `confidentlyBetter` than its runner-up (or the group has a single
	// candidate). Call before writing CSVs so both raw and best rows carry the flag.
	void annotateRankingConfidence(std::vector<SchemaSearchRecord>& records);

	// Spearman rank correlation of two equal-length series, tie-aware (average ranks). Returns 0.0
	// for fewer than two points or when either series has zero rank variance.
	double spearmanRankCorrelation(const std::vector<double>& a, const std::vector<double>& b);

	// Per (dataset, workload), reports the Spearman correlation between the cheap visit proxy
	// (avgVisitedNodes + alpha * avgTestedPoints) and measured latency over the real-latency records
	// (proxy-stage rows are excluded). Prints a summary to stdout and, when csvPath is non-empty,
	// writes one row per group. High rho means the proxy can be trusted to pre-rank candidates; low
	// means the cheap rungs may be discarding good schemas.
	void reportProxyLatencyCorrelation(const std::vector<SchemaSearchRecord>& records, const std::string& csvPath);

	// Zero-build heuristic estimate of average per-query work for a schema on a cloud, WITHOUT
	// building the index. Returns a "visited-nodes + alpha * tested-points"-style cost. Assumes a
	// roughly uniform distribution and that the structure subdivides until leaves hold about
	// leafCapacity points; the deepest block drives the leaf geometry. Intended as a cheap
	// pre-ranking signal, not an accurate timing predictor — validate with
	// reportEstimatedCostCorrelation before trusting it on a dataset.
	double estimateSchemaQueryCost(
		const SchemaConfig& schema,
		const PointCloudFeatures& features,
		const WorkloadFeatures& workload,
		double visitProxyAlpha = 0.1);

	// Per (dataset, workload), reports the Spearman correlation between the zero-build
	// estimatedQueryCost and measured latency over the real-latency records. High rho means the
	// estimate can pre-filter generated candidates before any build is paid for.
	void reportEstimatedCostCorrelation(const std::vector<SchemaSearchRecord>& records, const std::string& csvPath);

	// Returns the non-dominated set per (dataset, workload) group over the four-tuple
	// (avgLatencyMs, buildTimeMs, memoryMb, imbalancePenalty). Each returned record carries its
	// `paretoRank` (0 = lowest avgLatencyMs on the front, 1 = next, ...). Domination uses the
	// strict definition: a dominates b iff a is component-wise <= b on all four and < on at least
	// one. Records that share all four metrics get distinct ranks but neither dominates the other.
	std::vector<SchemaSearchRecord> selectParetoRecords(const std::vector<SchemaSearchRecord>& records);

	// Prints, per (dataset, workload), the Pareto knee (balanced compromise) alongside the
	// latency-optimal entry with a per-objective breakdown (latency / build / memory / imbalance),
	// so the operator sees *why* a schema was recommended rather than just a scalar score. Memory
	// uses the measured GPU footprint when available.
	void reportParetoKnee(const std::vector<SchemaSearchRecord>& records);

	// Bootstrap mean and 95% percentile-CI from a sample vector. Returns {mean, ciLow, ciHigh}.
	// Empty input yields zeros; single-element input collapses CI to the point value. Uses a
	// fixed-seed RNG so the returned CI is deterministic given the same samples — needed for
	// reproducible publication tables.
	std::tuple<double, double, double> bootstrapMeanCI(
		const std::vector<double>& samples,
		size_t resamples = 1000,
		uint32_t seed = 0xC0FFEEu);
	int runSchemaSearch(const SchemaSearchOptions& options);

	// One-shot evaluator: loads exactly one dataset, one schema, and one workload from `options`,
	// runs build + queries, and writes the resulting SchemaSearchRecord as JSON to stdout. Intended
	// to back Python-driven optimizers (Optuna ask/tell, Hyperband) where the C++ exe is launched
	// per candidate. CSV outputs are suppressed regardless of options.csvPath.
	int runEvaluateOne(const SchemaSearchOptions& options);
}

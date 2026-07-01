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
		std::string		_name;
		std::string		_path;
		SchemaConfig	_config;
		bool			_generated = false;
		// Canonical single-block control; force-promoted through every auto-condition stage.
		bool	_isBaseline = false;
	};

	struct SchemaGenerationOptions
	{
		size_t		_count = 0;
		size_t		_minBlocks = 1;
		size_t		_maxBlocks = 3;
		size_t		_maxDepth = 12;
		size_t		_minLeafCapacity = 32;
		size_t		_maxLeafCapacity = 32768;
		bool		_conditionalLevels = false;
		double		_conditionalProbability = 0.35;
		bool		_adaptiveLeafCapacity = false;
		double		_adaptiveLeafProbability = 0.25;
		uint32_t	_seed = 1337;
		std::string	_outputDirectory = "results/generated_schemas";
		// "auto" resolves to query_minimal_cpu for CPU discovery, cuda_query_full for CUDA.
		std::string	_primitiveProfile = "auto";
		// Under the CUDA evaluator, force kd/BIH levels to center_longest_axis (the only GPU-built split policy) so generated schemas stay GPU-native instead of falling back to the CPU.
		bool		_gpuNativeAxisPolicy = false;
	};

	struct ConditionDomain
	{
		std::vector<size_t>	_pointThresholds;
		std::vector<double>	_densityThresholds;
		std::vector<double>	_heightRatioThresholds;
		std::vector<double>	_extentXThresholds;
		std::vector<double>	_extentYThresholds;
		std::vector<double>	_extentZThresholds;
		// Anisotropy quantiles (1 - shortExtent/longExtent) over the sketch; empty for uniform clouds.
		std::vector<double>	_anisotropyThresholds;
		// Occupancy-entropy thresholds, normalized to [0,1]; seeded from [0.1,0.9] plus the cloud sketch.
		std::vector<double>	_occupancyEntropyThresholds;
		size_t				_samplePoints = 0;
		size_t				_sketchNodes = 0;
		bool				_estimatedFromCloud = false;
	};

	struct AutoConditionOptions
	{
		bool		_enabled = false;
		size_t		_proxyCandidateCount = 256;
		size_t		_proxyPointCap = 262144;
		size_t		_proxyQueryCount = 8;
		size_t		_finalTopK = 16;
		size_t		_confirmationTopK = 4;
		std::string	_outputDirectory = "results/auto_conditions";
		std::string	_selectorOutputPath = "models/local_schema_selector.json";
	};

	struct ScoreWeights
	{
		double	_lambdaLatency = 1.0;
		double	_lambdaBuild = 0.0;
		double	_lambdaMemory = 0.0;
		double	_lambdaImbalance = 0.0;
		// Score from cheap kernel counters (visitedNodes + alpha*testedPoints) instead of wall-clock latency.
		bool	_useVisitProxy = false;
		double	_visitProxyAlpha = 0.1;
	};

	struct WorkloadProfile
	{
		std::string		_name = "mixed";
		// Optional recorded query trace (CSV, the --query-trace format). When set, the
		// evaluator replays these queries instead of generating synthetic ones; the
		// synthetic weight/scale fields below are ignored.
		std::string		_tracePath;
		double			_rangeWeight = 0.4;
		double			_radiusWeight = 0.3;
		double			_knnWeight = 0.3;
		double			_rangeScaleMin = 0.01;
		double			_rangeScaleMax = 0.05;
		double			_radiusScaleMin = 0.01;
		double			_radiusScaleMax = 0.04;
		size_t			_numQueries = 1000;
		size_t			_knnK = 16;
		uint32_t		_querySeed = 1337;
		bool			_stratifyQueries = false;
		bool			_hasScoreWeights = false;
		ScoreWeights	_scoreWeights;
	};

	// One rung in a successive-halving schedule; each rung keeps the top advanceTopK for the next.
	struct RungSpec
	{
		std::string	_name = "rung";
		// 0 = use the workload's numQueries; positive overrides for this rung (e.g. 4 proxy, 64 confirm).
		size_t	_queryCountOverride = 0;
		// Score this rung by the cheap visit-proxy instead of wall-clock latency.
		bool	_useVisitProxy = false;
		double	_visitProxyAlpha = 0.1;
		// Candidates promoted to the next rung; 0 = promote all.
		size_t	_advanceTopK = 0;
	};

	struct RungSchedule
	{
		// When empty, the optimizer falls back to today's single-rung behavior.
		std::vector<RungSpec>	_rungs;
		// If set, points to an exported selector model JSON used between rungs to sample fresh
		// genomes via the surrogate (Bayesian acquisition step). Empty disables surrogate
		// proposals; the GA's mutation/immigration path stays untouched in that case.
		std::string	_surrogateModelPath;
		// Number of fresh candidates the surrogate is asked to propose between rungs. Ignored
		// when surrogateModelPath is empty.
		size_t	_surrogateProposalsPerStep = 0;
		// Pool size the surrogate ranks down to `surrogateProposalsPerStep`. Larger pool =
		// better acquisition at the cost of cheap surrogate evaluations.
		size_t	_surrogateCandidatePool = 0;
	};

	// Forward declaration; full definition lives in ThresholdRefiner.h.
	struct ThresholdRefinementOptions;

	struct EvolutionOptions
	{
		bool		_enabled = false;
		size_t		_generations = 3;
		size_t		_populationSize = 64;
		size_t		_eliteCount = 6;
		double		_mutationRate = 0.65;
		double		_randomImmigrationRate = 0.20;
		uint32_t	_seed = 1337;
		// Multi-fidelity rung schedule per batch; empty = flat evaluate-all-then-mutate.
		RungSchedule	_rungSchedule;
		// Continuous-threshold post-pass over the GA archive (settings in ThresholdRefiner.h).
		bool		_refineThresholds = false;
		size_t		_refineThresholdsTopK = 4;
		size_t		_refineThresholdsEvaluations = 60;
		double		_refineThresholdsSigma0 = 0.3;
		uint32_t	_refineThresholdsSeed = 1337;
		// Two-parent crossover probability; 0 = pure mutation.
		double	_crossoverRate = 0.4;
		// Pick elites by NSGA-II non-dominated-sort + crowding instead of scalar aggregateScore.
		bool	_useNsga2Ranking = true;
		// Diagnostic-guided repair mutations from measured bottlenecks before random mutation.
		bool	_repairMutations = false;
		size_t	_repairTopK = 4;
		size_t	_repairPerCandidate = 2;
		// Directory for per-(dataset,workload) best-schema JSON exports; empty = skip. Set via --optimizer-output-dir. Mirrors AutoConditionOptions._outputDirectory.
		std::string	_outputDirectory;
	};

	struct CudaEvaluationOptions
	{
		int			_device = -1;
		std::string	_builder = "lbvh";
		size_t		_queryBatchSize = 0;
		size_t		_memoryBudgetMb = 0;
		std::string	_knnBackend = "auto";
	};

	struct SchemaSearchOptions
	{
		std::vector<std::string>	_inputPaths;
		std::vector<std::string>	_schemaPaths;
		std::vector<std::string>	_workloadPaths;
		std::string					_rankModelPath;
		std::string					_csvPath = "results/schema_search.csv";
		std::string					_bestCsvPath = "results/schema_search_best.csv";
		// Pareto-front CSV (one non-dominated row per dataset/workload, with pareto_rank); empty = skip.
		std::string	_paretoCsvPath;
		// Markdown explanation report; empty = skip.
		std::string	_explainReportPath;
		// Per-query CSV trace; empty = disabled (bypasses the score cache so the trace is real).
		std::string	_queryTracePath;
		// Recorded query trace (CSV) replayed as the evaluation workload for every loaded
		// workload profile; overrides each profile's own trace/synthetic settings.
		std::string	_inputTracePath;
		// >= 2 re-measures the top-K per dataset/workload with N seeds and records mean + 95% CI.
		size_t					_confirmSeeds = 0;
		size_t					_confirmTopK = 4;
		bool					_includeSyntheticDatasets = true;
		bool					_includeConfiguredSchemas = true;
		bool					_useBinaryCache = true;
		bool					_rebuildBinaryCache = false;
		bool					_pauseAtEnd = false;
		size_t					_syntheticScale = 512;
		size_t					_queryCountOverride = 0;
		size_t					_knnKOverride = 0;
		size_t					_benchmarkTopK = 0;
		uint32_t				_querySeed = 1337;
		bool					_querySeedOverride = false;
		bool					_deepNestedSearch = false;
		SchemaGenerationOptions	_generation;
		AutoConditionOptions	_autoConditions;
		ScoreWeights			_weights;
		bool					_scoreWeightsOverride = false;
		std::string				_scoreStage = "final";
		bool					_scoreIsFinalLatency = true;
		EvolutionOptions		_evolution;
		std::string				_evaluator = "cpu";
		CudaEvaluationOptions	_cuda;
		std::string				_scoreObjective = "latency";
		// Always include canonical single-block schemas (pure QuadTree, Octree, KDTree, BVH, plus
		// GPU-native LBVH, KarrasOctree, RegularGrid, HGrid, BIH) as controls alongside whatever
		// generated/configured candidates are present. Lets the operator confirm that nested
		// candidates actually beat the naive baselines instead of just comparing nested to nested.
		bool				_includeBaselineSchemas = true;
		std::string			_scoreCachePath;
		bool				_rebuildScoreCache = false;
		EvaluationCache*	_scoreCache = nullptr;
		// Maximum number of CPU candidates to benchmark concurrently in the evolutionary loop.
		// Only honored when the evaluator is "cpu" (the CUDA path is serialised because the GPU
		// state and the per-builder build cache are not thread-safe). Default 1 keeps current
		// behaviour exactly.
		size_t	_parallelDispatch = 1;
		bool	_enableLeafMicroIndexes = false;
		size_t	_leafMicroIndexThreshold = 512;
		// >= 2 re-times each candidate's batch N times and records latency mean/stddev/CV/95% CI (timing noise).
		size_t	_measurementRepeats = 1;
		// Sidecar CSV for the proxy/latency correlation report; empty = stdout only.
		std::string	_proxyCorrelationCsvPath;
		// With --benchmark-top and no rank model, keep the K cheapest candidates by the zero-build estimate.
		bool	_estimatePrefilter = false;
		// Build canonical schemas on CPU and GPU and compare per-query returned counts (results/parity_report.csv).
		bool	_verifyParity = false;
		std::function<void(const SchemaSearchRecord&)> progressCallback;
	};

	struct SchemaSearchRecord
	{
		std::string			_datasetName;
		std::string			_datasetSource;
		size_t				_numPoints = 0;
		std::string			_workloadName;
		double				_rangeWeight = 0.0;
		double				_radiusWeight = 0.0;
		double				_knnWeight = 0.0;
		size_t				_numQueries = 0;
		size_t				_knnK = 0;
		uint32_t			_querySeed = 0;
		std::string			_schemaName;
		std::string			_schemaPath;
		BuildMetrics		_buildMetrics;
		QueryMetrics		_queryMetrics;
		QueryMetrics		_rangeMetrics;
		QueryMetrics		_countRangeMetrics;
		QueryMetrics		_radiusMetrics;
		QueryMetrics		_knnMetrics;
		size_t				_rangeQueries = 0;
		size_t				_countRangeQueries = 0;
		size_t				_radiusQueries = 0;
		size_t				_knnQueries = 0;
		std::string			_queryStrataSummary;
		double				_score = 0.0;
		double				_scoreMemoryMb = 0.0;
		double				_scoreImbalancePenalty = 0.0;
		ScoreWeights		_weights;
		std::string			_scoreObjective = "latency";
		std::string			_scoreMode = "latency";
		std::string			_scoreStage = "final";
		bool				_scoreIsFinalLatency = true;
		PointCloudFeatures	_pointFeatures;
		WorkloadFeatures	_workloadFeatures;
		std::string			_backend = "cpu";
		std::string			_gpuSupportStatus = "full";
		std::string			_knnBackend = "none";
		int					_cudaDevice = -1;
		std::string			_cudaBuilder;
		double				_gpuUploadMs = 0.0;
		double				_gpuBuildMs = 0.0;
		double				_gpuQueryMs = 0.0;
		size_t				_gpuMemoryBytes = 0;
		size_t				_conditionalLevels = 0;
		size_t				_conditionFields = 0;
		std::string			_conditionSummary;
		bool				_isBaseline = false;
		size_t				_activeStructureTypes = 0;
		double				_nestedActiveFraction = 0.0;
		std::string			_activeStructureSummary;
		std::string			_bestBaselineSchema;
		double				_bestBaselineScore = 0.0;
		double				_relativeSpeedupVsBaseline = 0.0;
		// Pareto-front rank by avgLatencyMs among non-dominated peers; -1 = not on the front.
		int	_paretoRank = -1;
		// > 0 means the latency/build CI fields below were computed across that many query seeds.
		size_t	_confirmSeedsUsed = 0;
		double	_latencyMean = 0.0;
		double	_latencyCiLow = 0.0;
		double	_latencyCiHigh = 0.0;
		double	_p95LatencyMean = 0.0;
		double	_p95LatencyCiLow = 0.0;
		double	_p95LatencyCiHigh = 0.0;
		double	_gpuBuildMean = 0.0;
		double	_gpuBuildCiLow = 0.0;
		double	_gpuBuildCiHigh = 0.0;
		// Winner's latency CI is separated from the runner-up's (see confidentlyBetter); false when single-shot.
		bool	_rankingConfident = false;
		// Pareto-front knee: entry closest to the normalized ideal across latency/build/memory/imbalance.
		bool	_paretoKnee = false;
		// Zero-build per-query cost estimate from schema + cloud features (see estimateSchemaQueryCost).
		double	_estimatedQueryCost = 0.0;
	};

	struct SchemaRepairDiagnostics
	{
		std::string	_bottleneck = "balanced";
		double		_leafOccupancyRatio = 0.0;
		double		_testedPerVisited = 0.0;
		double		_visitedPerQuery = 0.0;
		double		_testedPointFraction = 0.0;
		double		_fullContainmentRatio = 0.0;
		bool		_highLeafOccupancy = false;
		bool		_testedPointDominated = false;
		bool		_visitedNodeDominated = false;
		bool		_fullContainmentDominated = false;
		bool		_likelySingleChildChains = false;
	};

	struct EvaluatorResolution
	{
		std::string	_evaluator = "cpu";
		bool		_requestedCuda = false;
		bool		_usingCuda = false;
		bool		_fellBackToCpu = false;
		std::string	_warning;
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
	// repeat CI (queryMetrics._measurementRepeats > 1). With neither available the intervals collapse
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
	// per candidate. CSV outputs are suppressed regardless of options._csvPath.
	int runEvaluateOne(const SchemaSearchOptions& options);
}

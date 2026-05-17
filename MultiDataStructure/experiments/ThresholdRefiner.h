#pragma once

#include "../stdafx.h"
#include "SchemaSearch.h"

namespace Experiments
{
	struct ThresholdRefinementOptions
	{
		bool enabled = false;
		// How many candidates from the optimizer archive get refined. 0 keeps the GA's behavior.
		size_t topK = 4;
		// Total evaluation budget per candidate (visit-proxy calls inside the inner loop).
		// 40-80 is the publication-target range; the plan settled on ~60.
		size_t maxEvaluations = 60;
		// Population size lambda for the (1+lambda)-ES. 0 = pick automatically (max(4, 4 + 3*ln(n))).
		size_t populationLambda = 0;
		// Initial step size in normalised [0,1] threshold space.
		double sigma0 = 0.3;
		// Floor / ceiling for sigma during adaptation.
		double sigmaMin = 1.0e-3;
		double sigmaMax = 1.0;
		uint32_t seed = 1337;
		// When set, refined candidate JSONs are written here. Empty = candidate.path is "refined:<name>".
		std::string outputDirectory = "results/refined_schemas";
	};

	// Each entry is the *current* value of one threshold field on one schema level. The refiner
	// optimises the vector of these values. `hi` > `lo` is enforced; degenerate dimensions are
	// dropped before optimization starts.
	struct RefinementDimension
	{
		size_t levelIndex = 0;
		std::string field;          // "minPoints" / "maxPoints" / "minDensity" / ... matches SchemaLevelCondition
		bool integerValued = false; // true for minPoints / maxPoints
		bool logScale = false;      // true for point counts (geometric stretch over orders of magnitude)
		double lo = 0.0;
		double hi = 0.0;
		double current = 0.0;       // value already present on the candidate, used to seed the search
	};

	// Discovers the active threshold dimensions on `candidate` and clips their bounds against
	// `domain`. Returns an empty vector when the candidate has no conditional levels or when no
	// domain bound is available for any active field. Pure function, useful for tests.
	std::vector<RefinementDimension> collectRefinementDimensions(
		const SchemaCandidate& candidate,
		const ConditionDomain& domain);

	// Applies a decoded threshold vector back onto a fresh copy of the candidate's schema. Returns
	// the new schema (caller-side wraps it in a fresh SchemaCandidate via the existing
	// materialise path so it gets a unique signature and JSON file).
	SchemaConfig applyRefinementVector(
		const SchemaConfig& base,
		const std::vector<RefinementDimension>& dimensions,
		const std::vector<double>& values);

	using ThresholdScoreFn = std::function<double(const SchemaCandidate&)>;

	struct ThresholdRefinementResult
	{
		SchemaCandidate refinedCandidate;
		double initialScore = 0.0;
		double refinedScore = 0.0;
		size_t evaluationsUsed = 0;
		size_t dimensions = 0;
	};

	// (1+lambda)-evolution strategy with adaptive sigma. Picked over full CMA-ES because for
	// dimensions <= 6 with only ~60 evaluations there is not enough budget to learn a useful
	// covariance; the cheap sigma-adapted ES converges as fast in practice and is ~3x less code.
	// Returns the candidate unchanged when there are no active threshold dimensions.
	ThresholdRefinementResult refineSchemaThresholds(
		const SchemaCandidate& candidate,
		const ConditionDomain& domain,
		const ThresholdRefinementOptions& options,
		const ThresholdScoreFn& scoreFn);
}

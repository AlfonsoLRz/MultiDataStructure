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
		// Total evaluation budget per candidate (visit-proxy calls in the inner loop); ~60 is the publication target.
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

	// One threshold field on one schema level (its current value); the refiner optimises the vector of these, dropping degenerate dimensions before starting.
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

	// Discovers the active threshold dimensions on candidate, clipping bounds against domain; returns empty when there are no conditional levels or no usable bounds. Pure function.
	std::vector<RefinementDimension> collectRefinementDimensions(
		const SchemaCandidate& candidate,
		const ConditionDomain& domain);

	// Applies a decoded threshold vector onto a fresh copy of the candidate's schema; the caller wraps the result via the materialise path for a unique signature and JSON file.
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

	// (1+lambda)-ES with adaptive sigma, chosen over CMA-ES since ~60 evaluations can't learn a useful covariance; returns the candidate unchanged when no threshold dimensions are active.
	ThresholdRefinementResult refineSchemaThresholds(
		const SchemaCandidate& candidate,
		const ConditionDomain& domain,
		const ThresholdRefinementOptions& options,
		const ThresholdScoreFn& scoreFn);
}

#pragma once

#include "../stdafx.h"
#include "SchemaSearch.h"

namespace Experiments
{
	struct ThresholdRefinementOptions
	{
		bool	_enabled = false;
		// How many candidates from the optimizer archive get refined. 0 keeps the GA's behavior.
		size_t	_topK = 4;
		// Total evaluation budget per candidate (visit-proxy calls in the inner loop); ~60 is the publication target.
		size_t	_maxEvaluations = 60;
		// Population size lambda for the (1+lambda)-ES. 0 = pick automatically (max(4, 4 + 3*ln(n))).
		size_t	_populationLambda = 0;
		// Initial step size in normalised [0,1] threshold space.
		double	_sigma0 = 0.3;
		// Floor / ceiling for sigma during adaptation.
		double		_sigmaMin = 1.0e-3;
		double		_sigmaMax = 1.0;
		uint32_t	_seed = 1337;
		// When set, refined candidate JSONs are written here. Empty = candidate._path is "refined:<name>".
		std::string	_outputDirectory = "results/refined_schemas";
	};

	// One threshold field on one schema level (its current value); the refiner optimises the vector of these, dropping degenerate dimensions before starting.
	struct RefinementDimension
	{
		size_t	_levelIndex = 0;
		std::string field;          // "minPoints" / "maxPoints" / "minDensity" / ... matches SchemaLevelCondition
		bool integerValued = false; // true for minPoints / maxPoints
		bool logScale = false;      // true for point counts (geometric stretch over orders of magnitude)
		double	_lo = 0.0;
		double	_hi = 0.0;
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
		SchemaCandidate	_refinedCandidate;
		double			_initialScore = 0.0;
		double			_refinedScore = 0.0;
		size_t			_evaluationsUsed = 0;
		size_t			_dimensions = 0;
	};

	// (1+lambda)-ES with adaptive sigma, chosen over CMA-ES since ~60 evaluations can't learn a useful covariance; returns the candidate unchanged when no threshold dimensions are active.
	ThresholdRefinementResult refineSchemaThresholds(
		const SchemaCandidate& candidate,
		const ConditionDomain& domain,
		const ThresholdRefinementOptions& options,
		const ThresholdScoreFn& scoreFn);
}

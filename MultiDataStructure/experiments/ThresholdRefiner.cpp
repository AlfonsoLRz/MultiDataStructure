#include "../stdafx.h"
#include "ThresholdRefiner.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>

struct DomainBound
{
	bool	_valid = false;
	double	_lo = 0.0;
	double	_hi = 0.0;
};

template <typename TValue>
static DomainBound boundsOf(const std::vector<TValue>& sortedValues)
{
	DomainBound bound;
	if (sortedValues.empty())
		return bound;
	bound._valid = sortedValues.front() != sortedValues.back();
	bound._lo = static_cast<double>(sortedValues.front());
	bound._hi = static_cast<double>(sortedValues.back());
	// Degenerate bound: skip — there's nothing to refine.
	if (!bound._valid)
		return bound;
	// Defensive ordering: front < back is expected for sorted domains but guard anyway.
	if (bound._lo > bound._hi)
		std::swap(bound._lo, bound._hi);
	return bound;
}

static double clamp01(double x)
{
	return std::clamp(x, 0.0, 1.0);
}

static double encodeLinear(double value, double lo, double hi)
{
	if (hi <= lo)
		return 0.0;
	return clamp01((value - lo) / (hi - lo));
}

static double decodeLinear(double x, double lo, double hi)
{
	return lo + clamp01(x) * (hi - lo);
}

static double encodeLog(double value, double lo, double hi)
{
	if (hi <= lo || value <= 0.0 || lo <= 0.0)
		return 0.0;
	const double logLo = std::log2(lo);
	const double logHi = std::log2(hi);
	if (logHi <= logLo)
		return 0.0;
	return clamp01((std::log2(value) - logLo) / (logHi - logLo));
}

static double decodeLog(double x, double lo, double hi)
{
	if (hi <= lo || lo <= 0.0)
		return lo;
	const double logLo = std::log2(lo);
	const double logHi = std::log2(hi);
	return std::pow(2.0, logLo + clamp01(x) * (logHi - logLo));
}

// Adds a threshold dimension only when the candidate has it set and the domain has a non-degenerate bound, keeping refinement non-destructive.
template <typename TGetter, typename TBoundProvider>
static void addDimensionIfActive(
	std::vector<Experiments::RefinementDimension>& outDims,
	size_t levelIndex,
	const std::string& fieldName,
	bool integerValued,
	bool logScale,
	TGetter getter,
	TBoundProvider bounds)
{
	const auto& optional = getter();
	if (!optional.has_value())
		return;
	const DomainBound bound = bounds();
	if (!bound._valid)
		return;

	Experiments::RefinementDimension dim;
	dim._levelIndex = levelIndex;
	dim.field = fieldName;
	dim.integerValued = integerValued;
	dim.logScale = logScale;
	dim._lo = bound._lo;
	dim._hi = bound._hi;
	dim.current = static_cast<double>(optional.value());
	// Snap a current value outside the tightened domain back into range so the initial encoding lands in [0,1].
	dim.current = std::clamp(dim.current, dim._lo, dim._hi);
	outDims.push_back(dim);
}

static double encode(const Experiments::RefinementDimension& dim, double value)
{
	return dim.logScale ? encodeLog(value, dim._lo, dim._hi) : encodeLinear(value, dim._lo, dim._hi);
}

static double decode(const Experiments::RefinementDimension& dim, double x)
{
	double v = dim.logScale ? decodeLog(x, dim._lo, dim._hi) : decodeLinear(x, dim._lo, dim._hi);
	if (dim.integerValued)
		v = std::round(v);
	return v;
}

static void writeDimensionInto(SchemaLevelCondition& condition, const std::string& field, double value)
{
	if (field == "minPoints")
		condition._minPoints = static_cast<size_t>(std::max(0.0, value));
	else if (field == "maxPoints")
		condition._maxPoints = static_cast<size_t>(std::max(0.0, value));
	else if (field == "minDensity")
		condition._minDensity = value;
	else if (field == "maxDensity")
		condition._maxDensity = value;
	else if (field == "minHeightRatio")
		condition._minHeightRatio = value;
	else if (field == "maxHeightRatio")
		condition._maxHeightRatio = value;
	else if (field == "minExtentX")
		condition._minExtentX = value;
	else if (field == "maxExtentX")
		condition._maxExtentX = value;
	else if (field == "minExtentY")
		condition._minExtentY = value;
	else if (field == "maxExtentY")
		condition._maxExtentY = value;
	else if (field == "minExtentZ")
		condition._minExtentZ = value;
	else if (field == "maxExtentZ")
		condition._maxExtentZ = value;
	else if (field == "minAnisotropy")
		condition._minAnisotropy = std::clamp(value, 0.0, 1.0);
	else if (field == "maxAnisotropy")
		condition._maxAnisotropy = std::clamp(value, 0.0, 1.0);
	else if (field == "minOccupancyEntropy")
		condition._minOccupancyEntropy = std::clamp(value, 0.0, 1.0);
	else if (field == "maxOccupancyEntropy")
		condition._maxOccupancyEntropy = std::clamp(value, 0.0, 1.0);
}

namespace Experiments
{
	std::vector<RefinementDimension> collectRefinementDimensions(
		const SchemaCandidate& candidate,
		const ConditionDomain& domain)
	{
		std::vector<RefinementDimension> dims;
		const DomainBound pointBound = boundsOf(domain._pointThresholds);
		const DomainBound densityBound = boundsOf(domain._densityThresholds);
		const DomainBound heightBound = boundsOf(domain._heightRatioThresholds);
		const DomainBound extentXBound = boundsOf(domain._extentXThresholds);
		const DomainBound extentYBound = boundsOf(domain._extentYThresholds);
		const DomainBound extentZBound = boundsOf(domain._extentZThresholds);
		const DomainBound anisotropyBound = boundsOf(domain._anisotropyThresholds);
		const DomainBound entropyBound = boundsOf(domain._occupancyEntropyThresholds);

		for (size_t levelIndex = 0; levelIndex < candidate._config._levels.size(); ++levelIndex)
		{
			const SchemaLevelCondition& condition = candidate._config._levels[levelIndex]._condition;
			if (condition.empty())
				continue;

			addDimensionIfActive(dims, levelIndex, "minPoints", true, true,
				[&]() -> const std::optional<size_t>& { return condition._minPoints; },
				[&]() { return pointBound; });
			addDimensionIfActive(dims, levelIndex, "maxPoints", true, true,
				[&]() -> const std::optional<size_t>& { return condition._maxPoints; },
				[&]() { return pointBound; });
			addDimensionIfActive(dims, levelIndex, "minDensity", false, false,
				[&]() -> const std::optional<double>& { return condition._minDensity; },
				[&]() { return densityBound; });
			addDimensionIfActive(dims, levelIndex, "maxDensity", false, false,
				[&]() -> const std::optional<double>& { return condition._maxDensity; },
				[&]() { return densityBound; });
			addDimensionIfActive(dims, levelIndex, "minHeightRatio", false, false,
				[&]() -> const std::optional<double>& { return condition._minHeightRatio; },
				[&]() { return heightBound; });
			addDimensionIfActive(dims, levelIndex, "maxHeightRatio", false, false,
				[&]() -> const std::optional<double>& { return condition._maxHeightRatio; },
				[&]() { return heightBound; });
			addDimensionIfActive(dims, levelIndex, "minExtentX", false, false,
				[&]() -> const std::optional<double>& { return condition._minExtentX; },
				[&]() { return extentXBound; });
			addDimensionIfActive(dims, levelIndex, "maxExtentX", false, false,
				[&]() -> const std::optional<double>& { return condition._maxExtentX; },
				[&]() { return extentXBound; });
			addDimensionIfActive(dims, levelIndex, "minExtentY", false, false,
				[&]() -> const std::optional<double>& { return condition._minExtentY; },
				[&]() { return extentYBound; });
			addDimensionIfActive(dims, levelIndex, "maxExtentY", false, false,
				[&]() -> const std::optional<double>& { return condition._maxExtentY; },
				[&]() { return extentYBound; });
			addDimensionIfActive(dims, levelIndex, "minExtentZ", false, false,
				[&]() -> const std::optional<double>& { return condition._minExtentZ; },
				[&]() { return extentZBound; });
			addDimensionIfActive(dims, levelIndex, "maxExtentZ", false, false,
				[&]() -> const std::optional<double>& { return condition._maxExtentZ; },
				[&]() { return extentZBound; });
			addDimensionIfActive(dims, levelIndex, "minAnisotropy", false, false,
				[&]() -> const std::optional<double>& { return condition._minAnisotropy; },
				[&]() { return anisotropyBound; });
			addDimensionIfActive(dims, levelIndex, "maxAnisotropy", false, false,
				[&]() -> const std::optional<double>& { return condition._maxAnisotropy; },
				[&]() { return anisotropyBound; });
			addDimensionIfActive(dims, levelIndex, "minOccupancyEntropy", false, false,
				[&]() -> const std::optional<double>& { return condition._minOccupancyEntropy; },
				[&]() { return entropyBound; });
			addDimensionIfActive(dims, levelIndex, "maxOccupancyEntropy", false, false,
				[&]() -> const std::optional<double>& { return condition._maxOccupancyEntropy; },
				[&]() { return entropyBound; });
		}

		return dims;
	}

	SchemaConfig applyRefinementVector(
		const SchemaConfig& base,
		const std::vector<RefinementDimension>& dimensions,
		const std::vector<double>& values)
	{
		SchemaConfig refined = base;
		const size_t writeCount = std::min(dimensions.size(), values.size());
		for (size_t i = 0; i < writeCount; ++i)
		{
			const RefinementDimension& dim = dimensions[i];
			if (dim._levelIndex >= refined._levels.size())
				continue;
			writeDimensionInto(refined._levels[dim._levelIndex]._condition, dim.field, values[i]);
		}
		return refined;
	}

	ThresholdRefinementResult refineSchemaThresholds(
		const SchemaCandidate& candidate,
		const ConditionDomain& domain,
		const ThresholdRefinementOptions& options,
		const ThresholdScoreFn& scoreFn)
	{
		ThresholdRefinementResult result;
		result._refinedCandidate = candidate;
		result._initialScore = scoreFn ? scoreFn(candidate) : std::numeric_limits<double>::infinity();
		result._refinedScore = result._initialScore;
		result._evaluationsUsed = scoreFn ? 1 : 0;

		const std::vector<RefinementDimension> dims = collectRefinementDimensions(candidate, domain);
		result._dimensions = dims.size();
		if (dims.empty() || !scoreFn || options._maxEvaluations <= 1)
			return result;

		const size_t n = dims.size();
		const size_t autoLambda = static_cast<size_t>(std::max(4.0, 4.0 + 3.0 * std::log(static_cast<double>(n))));
		const size_t lambda = options._populationLambda > 0 ? options._populationLambda : autoLambda;
		if (lambda == 0)
			return result;

		std::mt19937 rng(options._seed);
		std::normal_distribution<double> noise(0.0, 1.0);

		std::vector<double> bestX(n, 0.0);
		for (size_t i = 0; i < n; ++i)
			bestX[i] = encode(dims[i], dims[i].current);

		double sigma = std::clamp(options._sigma0, options._sigmaMin, options._sigmaMax);
		double bestScore = result._initialScore;
		size_t evaluationsUsed = result._evaluationsUsed;

		auto decodeVector = [&](const std::vector<double>& x) {
			std::vector<double> decoded(n);
			for (size_t i = 0; i < n; ++i)
				decoded[i] = decode(dims[i], x[i]);
			return decoded;
		};

		auto scoreVector = [&](const std::vector<double>& x, size_t generation, size_t childIndex) {
			const std::vector<double> decoded = decodeVector(x);
			SchemaConfig refinedConfig = applyRefinementVector(candidate._config, dims, decoded);
			const std::string prefix = "refined_g" + std::to_string(generation) + "_c" + std::to_string(childIndex);
			SchemaCandidate childCandidate = materializeSchemaCandidate(refinedConfig, prefix, options._outputDirectory);
			childCandidate._generated = true;
			return std::pair<double, SchemaCandidate>{ scoreFn(childCandidate), std::move(childCandidate) };
		};

		std::vector<double> xCandidate(n, 0.0);
		size_t generation = 0;
		const auto refinementStart = std::chrono::steady_clock::now();
		auto lastLogTime = refinementStart;
		std::cout << "      refining " << options._maxEvaluations << "-eval budget, "
			<< n << " active dim(s), lambda " << lambda
			<< ", sigma0 " << options._sigma0 << '\n';
		while (evaluationsUsed + lambda <= options._maxEvaluations)
		{
			++generation;
			double bestChildScore = std::numeric_limits<double>::infinity();
			std::vector<double> bestChildX;
			SchemaCandidate bestChildCandidate;

			for (size_t child = 0; child < lambda; ++child)
			{
				for (size_t i = 0; i < n; ++i)
					xCandidate[i] = clamp01(bestX[i] + sigma * noise(rng));

				auto [score, childCandidate] = scoreVector(xCandidate, generation, child);
				++evaluationsUsed;
				if (score < bestChildScore)
				{
					bestChildScore = score;
					bestChildX = xCandidate;
					bestChildCandidate = std::move(childCandidate);
				}
			}

			if (bestChildScore < bestScore)
			{
				bestScore = bestChildScore;
				bestX = bestChildX;
				result._refinedCandidate = std::move(bestChildCandidate);
				// 1/5-rule heuristic: expand sigma when an offspring improved on the parent, contract otherwise.
				sigma = std::min(sigma * 1.5, options._sigmaMax);
			}
			else
			{
				sigma = std::max(sigma * 0.8, options._sigmaMin);
			}

			// Throttled per-generation progress line so the GUI doesn't appear to hang during the refiner's expensive full-build evaluations.
			const auto now = std::chrono::steady_clock::now();
			const double sinceLastLog = std::chrono::duration<double>(now - lastLogTime).count();
			if (sinceLastLog >= 2.0 || evaluationsUsed >= options._maxEvaluations)
			{
				const double elapsedSec = std::chrono::duration<double>(now - refinementStart).count();
				std::cout << "        refine gen " << generation
					<< " eval " << evaluationsUsed << "/" << options._maxEvaluations
					<< " best " << bestScore
					<< " sigma " << sigma
					<< " (" << elapsedSec << " s elapsed)\n";
				lastLogTime = now;
			}
		}

		result._refinedScore = bestScore;
		result._evaluationsUsed = evaluationsUsed;
		return result;
	}
}

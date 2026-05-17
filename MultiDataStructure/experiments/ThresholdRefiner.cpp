#include "../stdafx.h"
#include "ThresholdRefiner.h"

#include <cmath>
#include <random>

namespace
{
	struct DomainBound
	{
		bool valid = false;
		double lo = 0.0;
		double hi = 0.0;
	};

	template <typename TValue>
	DomainBound boundsOf(const std::vector<TValue>& sortedValues)
	{
		DomainBound bound;
		if (sortedValues.empty())
			return bound;
		bound.valid = sortedValues.front() != sortedValues.back();
		bound.lo = static_cast<double>(sortedValues.front());
		bound.hi = static_cast<double>(sortedValues.back());
		// Degenerate bound: skip — there's nothing to refine.
		if (!bound.valid)
			return bound;
		// Defensive ordering: front < back is expected for sorted domains but guard anyway.
		if (bound.lo > bound.hi)
			std::swap(bound.lo, bound.hi);
		return bound;
	}

	double clamp01(double x)
	{
		return std::clamp(x, 0.0, 1.0);
	}

	double encodeLinear(double value, double lo, double hi)
	{
		if (hi <= lo)
			return 0.0;
		return clamp01((value - lo) / (hi - lo));
	}

	double decodeLinear(double x, double lo, double hi)
	{
		return lo + clamp01(x) * (hi - lo);
	}

	double encodeLog(double value, double lo, double hi)
	{
		if (hi <= lo || value <= 0.0 || lo <= 0.0)
			return 0.0;
		const double logLo = std::log2(lo);
		const double logHi = std::log2(hi);
		if (logHi <= logLo)
			return 0.0;
		return clamp01((std::log2(value) - logLo) / (logHi - logLo));
	}

	double decodeLog(double x, double lo, double hi)
	{
		if (hi <= lo || lo <= 0.0)
			return lo;
		const double logLo = std::log2(lo);
		const double logHi = std::log2(hi);
		return std::pow(2.0, logLo + clamp01(x) * (logHi - logLo));
	}

	// Adds an active threshold dimension if both (a) the candidate has it set and (b) the domain
	// has a non-degenerate bound for it. The plan keeps refinement non-destructive: thresholds
	// the candidate did not set stay unset.
	template <typename TGetter, typename TBoundProvider>
	void addDimensionIfActive(
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
		if (!bound.valid)
			return;

		Experiments::RefinementDimension dim;
		dim.levelIndex = levelIndex;
		dim.field = fieldName;
		dim.integerValued = integerValued;
		dim.logScale = logScale;
		dim.lo = bound.lo;
		dim.hi = bound.hi;
		dim.current = static_cast<double>(optional.value());
		// Snap a current value that falls outside the (possibly tightened) domain back into range
		// so the initial mean encoding lands in [0,1].
		dim.current = std::clamp(dim.current, dim.lo, dim.hi);
		outDims.push_back(dim);
	}

	double encode(const Experiments::RefinementDimension& dim, double value)
	{
		return dim.logScale ? encodeLog(value, dim.lo, dim.hi) : encodeLinear(value, dim.lo, dim.hi);
	}

	double decode(const Experiments::RefinementDimension& dim, double x)
	{
		double v = dim.logScale ? decodeLog(x, dim.lo, dim.hi) : decodeLinear(x, dim.lo, dim.hi);
		if (dim.integerValued)
			v = std::round(v);
		return v;
	}

	void writeDimensionInto(SchemaLevelCondition& condition, const std::string& field, double value)
	{
		if (field == "minPoints")
			condition.minPoints = static_cast<size_t>(std::max(0.0, value));
		else if (field == "maxPoints")
			condition.maxPoints = static_cast<size_t>(std::max(0.0, value));
		else if (field == "minDensity")
			condition.minDensity = value;
		else if (field == "maxDensity")
			condition.maxDensity = value;
		else if (field == "minHeightRatio")
			condition.minHeightRatio = value;
		else if (field == "maxHeightRatio")
			condition.maxHeightRatio = value;
		else if (field == "minExtentX")
			condition.minExtentX = value;
		else if (field == "maxExtentX")
			condition.maxExtentX = value;
		else if (field == "minExtentY")
			condition.minExtentY = value;
		else if (field == "maxExtentY")
			condition.maxExtentY = value;
		else if (field == "minExtentZ")
			condition.minExtentZ = value;
		else if (field == "maxExtentZ")
			condition.maxExtentZ = value;
	}
}

namespace Experiments
{
	std::vector<RefinementDimension> collectRefinementDimensions(
		const SchemaCandidate& candidate,
		const ConditionDomain& domain)
	{
		std::vector<RefinementDimension> dims;
		const DomainBound pointBound = boundsOf(domain.pointThresholds);
		const DomainBound densityBound = boundsOf(domain.densityThresholds);
		const DomainBound heightBound = boundsOf(domain.heightRatioThresholds);
		const DomainBound extentXBound = boundsOf(domain.extentXThresholds);
		const DomainBound extentYBound = boundsOf(domain.extentYThresholds);
		const DomainBound extentZBound = boundsOf(domain.extentZThresholds);

		for (size_t levelIndex = 0; levelIndex < candidate.config.levels.size(); ++levelIndex)
		{
			const SchemaLevelCondition& condition = candidate.config.levels[levelIndex].condition;
			if (condition.empty())
				continue;

			addDimensionIfActive(dims, levelIndex, "minPoints", true, true,
				[&]() -> const std::optional<size_t>& { return condition.minPoints; },
				[&]() { return pointBound; });
			addDimensionIfActive(dims, levelIndex, "maxPoints", true, true,
				[&]() -> const std::optional<size_t>& { return condition.maxPoints; },
				[&]() { return pointBound; });
			addDimensionIfActive(dims, levelIndex, "minDensity", false, false,
				[&]() -> const std::optional<double>& { return condition.minDensity; },
				[&]() { return densityBound; });
			addDimensionIfActive(dims, levelIndex, "maxDensity", false, false,
				[&]() -> const std::optional<double>& { return condition.maxDensity; },
				[&]() { return densityBound; });
			addDimensionIfActive(dims, levelIndex, "minHeightRatio", false, false,
				[&]() -> const std::optional<double>& { return condition.minHeightRatio; },
				[&]() { return heightBound; });
			addDimensionIfActive(dims, levelIndex, "maxHeightRatio", false, false,
				[&]() -> const std::optional<double>& { return condition.maxHeightRatio; },
				[&]() { return heightBound; });
			addDimensionIfActive(dims, levelIndex, "minExtentX", false, false,
				[&]() -> const std::optional<double>& { return condition.minExtentX; },
				[&]() { return extentXBound; });
			addDimensionIfActive(dims, levelIndex, "maxExtentX", false, false,
				[&]() -> const std::optional<double>& { return condition.maxExtentX; },
				[&]() { return extentXBound; });
			addDimensionIfActive(dims, levelIndex, "minExtentY", false, false,
				[&]() -> const std::optional<double>& { return condition.minExtentY; },
				[&]() { return extentYBound; });
			addDimensionIfActive(dims, levelIndex, "maxExtentY", false, false,
				[&]() -> const std::optional<double>& { return condition.maxExtentY; },
				[&]() { return extentYBound; });
			addDimensionIfActive(dims, levelIndex, "minExtentZ", false, false,
				[&]() -> const std::optional<double>& { return condition.minExtentZ; },
				[&]() { return extentZBound; });
			addDimensionIfActive(dims, levelIndex, "maxExtentZ", false, false,
				[&]() -> const std::optional<double>& { return condition.maxExtentZ; },
				[&]() { return extentZBound; });
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
			if (dim.levelIndex >= refined.levels.size())
				continue;
			writeDimensionInto(refined.levels[dim.levelIndex].condition, dim.field, values[i]);
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
		result.refinedCandidate = candidate;
		result.initialScore = scoreFn ? scoreFn(candidate) : std::numeric_limits<double>::infinity();
		result.refinedScore = result.initialScore;
		result.evaluationsUsed = scoreFn ? 1 : 0;

		const std::vector<RefinementDimension> dims = collectRefinementDimensions(candidate, domain);
		result.dimensions = dims.size();
		if (dims.empty() || !scoreFn || options.maxEvaluations <= 1)
			return result;

		const size_t n = dims.size();
		const size_t autoLambda = static_cast<size_t>(std::max(4.0, 4.0 + 3.0 * std::log(static_cast<double>(n))));
		const size_t lambda = options.populationLambda > 0 ? options.populationLambda : autoLambda;
		if (lambda == 0)
			return result;

		std::mt19937 rng(options.seed);
		std::normal_distribution<double> noise(0.0, 1.0);

		std::vector<double> bestX(n, 0.0);
		for (size_t i = 0; i < n; ++i)
			bestX[i] = encode(dims[i], dims[i].current);

		double sigma = std::clamp(options.sigma0, options.sigmaMin, options.sigmaMax);
		double bestScore = result.initialScore;
		size_t evaluationsUsed = result.evaluationsUsed;

		auto decodeVector = [&](const std::vector<double>& x) {
			std::vector<double> decoded(n);
			for (size_t i = 0; i < n; ++i)
				decoded[i] = decode(dims[i], x[i]);
			return decoded;
		};

		auto scoreVector = [&](const std::vector<double>& x, size_t generation, size_t childIndex) {
			const std::vector<double> decoded = decodeVector(x);
			SchemaConfig refinedConfig = applyRefinementVector(candidate.config, dims, decoded);
			const std::string prefix = "refined_g" + std::to_string(generation) + "_c" + std::to_string(childIndex);
			SchemaCandidate childCandidate = materializeSchemaCandidate(refinedConfig, prefix, options.outputDirectory);
			childCandidate.generated = true;
			return std::pair<double, SchemaCandidate>{ scoreFn(childCandidate), std::move(childCandidate) };
		};

		std::vector<double> xCandidate(n, 0.0);
		size_t generation = 0;
		while (evaluationsUsed + lambda <= options.maxEvaluations)
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
				result.refinedCandidate = std::move(bestChildCandidate);
				// 1/5-rule heuristic for batched ES: expand search when at least one offspring
				// improved on the parent, contract otherwise.
				sigma = std::min(sigma * 1.5, options.sigmaMax);
			}
			else
			{
				sigma = std::max(sigma * 0.8, options.sigmaMin);
			}
		}

		result.refinedScore = bestScore;
		result.evaluationsUsed = evaluationsUsed;
		return result;
	}
}

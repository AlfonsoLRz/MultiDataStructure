#include "../stdafx.h"
#include "FeatureExtraction.h"

#include "SchemaSearch.h"

#include <array>

namespace
{
	constexpr double EPSILON = 1e-9;
	constexpr double PI = 3.14159265358979323846;

	double safeDivide(double numerator, double denominator)
	{
		return std::abs(denominator) > EPSILON ? numerator / denominator : 0.0;
	}

	double clamp01(double value)
	{
		return std::clamp(value, 0.0, 1.0);
	}

	std::vector<size_t> deterministicSampleIndices(size_t count, size_t maxSampleSize, uint32_t seed)
	{
		const size_t sampleSize = std::min(count, maxSampleSize);
		std::vector<size_t> sample;
		sample.reserve(sampleSize);

		for (size_t i = 0; i < sampleSize; ++i)
			sample.push_back(i);

		if (sampleSize == count)
			return sample;

		std::mt19937 rng(seed);
		for (size_t i = sampleSize; i < count; ++i)
		{
			std::uniform_int_distribution<size_t> distribution(0, i);
			const size_t replacement = distribution(rng);
			if (replacement < sampleSize)
				sample[replacement] = i;
		}

		std::sort(sample.begin(), sample.end());
		return sample;
	}

	std::array<double, 3> covarianceEigenvalues(
		double c00,
		double c01,
		double c02,
		double c11,
		double c12,
		double c22)
	{
		const double p1 = c01 * c01 + c02 * c02 + c12 * c12;
		if (p1 <= EPSILON)
		{
			std::array<double, 3> diagonal = { c00, c11, c22 };
			std::sort(diagonal.begin(), diagonal.end(), std::greater<double>());
			for (double& value : diagonal)
				value = std::max(0.0, value);
			return diagonal;
		}

		const double q = (c00 + c11 + c22) / 3.0;
		const double b00 = c00 - q;
		const double b11 = c11 - q;
		const double b22 = c22 - q;
		const double p2 = b00 * b00 + b11 * b11 + b22 * b22 + 2.0 * p1;
		const double p = std::sqrt(p2 / 6.0);
		if (p <= EPSILON)
			return { 0.0, 0.0, 0.0 };

		const double invP = 1.0 / p;
		const double m00 = b00 * invP;
		const double m01 = c01 * invP;
		const double m02 = c02 * invP;
		const double m11 = b11 * invP;
		const double m12 = c12 * invP;
		const double m22 = b22 * invP;
		const double determinant =
			m00 * (m11 * m22 - m12 * m12) -
			m01 * (m01 * m22 - m12 * m02) +
			m02 * (m01 * m12 - m11 * m02);
		const double r = determinant * 0.5;

		double phi = 0.0;
		if (r <= -1.0)
			phi = PI / 3.0;
		else if (r < 1.0)
			phi = std::acos(r) / 3.0;

		std::array<double, 3> values = {
			q + 2.0 * p * std::cos(phi),
			q + 2.0 * p * std::cos(phi + (2.0 * PI / 3.0)),
			q + 2.0 * p * std::cos(phi + (4.0 * PI / 3.0)),
		};
		std::sort(values.begin(), values.end(), std::greater<double>());
		for (double& value : values)
			value = std::max(0.0, value);
		return values;
	}

	void addOccupancyFeatures(
		Experiments::PointCloudFeatures& features,
		const PointCloud& cloud,
		const std::vector<size_t>& sample)
	{
		if (sample.empty())
			return;

		std::array<size_t, 512> counts{};
		const glm::vec3 min = cloud.bounds().min();
		const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.0f));

		for (const size_t pointIndex : sample)
		{
			const glm::vec3& position = cloud.points()[pointIndex].position;
			glm::uvec3 cell(0);
			for (glm::uint axis = 0; axis < 3; ++axis)
			{
				if (range[axis] > 0.0f)
				{
					const double normalized = std::clamp(
						static_cast<double>((position[axis] - min[axis]) / range[axis]),
						0.0,
						0.999999);
					cell[axis] = static_cast<glm::uint>(normalized * 8.0);
				}
			}

			const size_t flatIndex = static_cast<size_t>(cell.x) * 64 + static_cast<size_t>(cell.y) * 8 + static_cast<size_t>(cell.z);
			++counts[std::min<size_t>(flatIndex, counts.size() - 1)];
		}

		size_t occupied = 0;
		double entropy = 0.0;
		for (const size_t count : counts)
		{
			if (count == 0)
				continue;

			++occupied;
			const double probability = static_cast<double>(count) / static_cast<double>(sample.size());
			entropy -= probability * std::log(probability);
		}

		features.occupancyRatio8 = static_cast<double>(occupied) / static_cast<double>(counts.size());
		features.occupancyEntropy8 = entropy;

		if (occupied == 0)
			return;

		const double mean = static_cast<double>(sample.size()) / static_cast<double>(occupied);
		double variance = 0.0;
		for (const size_t count : counts)
		{
			if (count == 0)
				continue;
			const double delta = static_cast<double>(count) - mean;
			variance += delta * delta;
		}
		variance /= static_cast<double>(occupied);
		features.densityCv8 = mean > EPSILON ? std::sqrt(variance) / mean : 0.0;
	}

	std::array<double, 3> normalizedWorkloadWeights(const Experiments::WorkloadProfile& profile)
	{
		std::array<double, 3> weights = {
			std::max(0.0, profile.rangeWeight),
			std::max(0.0, profile.radiusWeight),
			std::max(0.0, profile.knnWeight),
		};

		const double total = weights[0] + weights[1] + weights[2];
		if (total <= EPSILON)
			return { 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0 };

		weights[0] /= total;
		weights[1] /= total;
		weights[2] /= total;
		return weights;
	}
}

Experiments::PointCloudFeatures Experiments::extractPointCloudFeatures(const PointCloud& cloud, size_t maxSampleSize, uint32_t seed)
{
	PointCloudFeatures features;
	features.numPoints = cloud.size();
	if (cloud.empty() || maxSampleSize == 0)
		return features;

	const std::vector<size_t> sample = deterministicSampleIndices(cloud.size(), maxSampleSize, seed);
	features.sampleSize = sample.size();

	const glm::vec3 range = glm::max(cloud.coordinateRange(), glm::vec3(0.0f));
	features.bboxX = range.x;
	features.bboxY = range.y;
	features.bboxZ = range.z;
	features.aspectXY = safeDivide(features.bboxX, features.bboxY);
	features.aspectXZ = safeDivide(features.bboxX, features.bboxZ);
	features.aspectYZ = safeDivide(features.bboxY, features.bboxZ);
	features.heightRange = features.bboxZ;

	const double volume = features.bboxX * features.bboxY * features.bboxZ;
	if (volume > EPSILON)
	{
		features.densityBbox = static_cast<double>(features.numPoints) / volume;
	}
	else
	{
		const double area = std::max({
			features.bboxX * features.bboxY,
			features.bboxX * features.bboxZ,
			features.bboxY * features.bboxZ,
		});
		const double fallbackExtent = std::max({ features.bboxX, features.bboxY, features.bboxZ, 1.0 });
		features.densityBbox = area > EPSILON
			? static_cast<double>(features.numPoints) / area
			: static_cast<double>(features.numPoints) / fallbackExtent;
	}

	double sumZ = 0.0;
	glm::dvec3 mean(0.0);
	for (const size_t pointIndex : sample)
	{
		const glm::vec3& position = cloud.points()[pointIndex].position;
		sumZ += position.z;
		mean += glm::dvec3(position);
	}

	features.heightMean = sumZ / static_cast<double>(sample.size());
	mean /= static_cast<double>(sample.size());

	double heightVariance = 0.0;
	double c00 = 0.0;
	double c01 = 0.0;
	double c02 = 0.0;
	double c11 = 0.0;
	double c12 = 0.0;
	double c22 = 0.0;

	for (const size_t pointIndex : sample)
	{
		const glm::dvec3 position(cloud.points()[pointIndex].position);
		const double zDelta = position.z - features.heightMean;
		heightVariance += zDelta * zDelta;

		const glm::dvec3 delta = position - mean;
		c00 += delta.x * delta.x;
		c01 += delta.x * delta.y;
		c02 += delta.x * delta.z;
		c11 += delta.y * delta.y;
		c12 += delta.y * delta.z;
		c22 += delta.z * delta.z;
	}

	const double invCount = 1.0 / static_cast<double>(sample.size());
	features.heightStd = std::sqrt(heightVariance * invCount);

	const std::array<double, 3> eigenvalues = covarianceEigenvalues(
		c00 * invCount,
		c01 * invCount,
		c02 * invCount,
		c11 * invCount,
		c12 * invCount,
		c22 * invCount);
	features.covEig0 = eigenvalues[0];
	features.covEig1 = eigenvalues[1];
	features.covEig2 = eigenvalues[2];

	if (features.covEig0 > EPSILON)
	{
		features.linearity = (features.covEig0 - features.covEig1) / features.covEig0;
		features.planarity = (features.covEig1 - features.covEig2) / features.covEig0;
		features.scattering = features.covEig2 / features.covEig0;
	}

	addOccupancyFeatures(features, cloud, sample);

	const double horizontalExtent = std::max({ features.bboxX, features.bboxY, EPSILON });
	const double heightRatio = safeDivide(features.bboxZ, horizontalExtent);
	features.verticalityScore = clamp01(heightRatio * (features.planarity + features.linearity));
	features.flatnessScore = clamp01((1.0 - clamp01(heightRatio)) * (features.planarity + (1.0 - features.scattering)) * 0.5);

	return features;
}

Experiments::WorkloadFeatures Experiments::extractWorkloadFeatures(const WorkloadProfile& profile, const ScoreWeights& weights)
{
	WorkloadFeatures features;
	const std::array<double, 3> normalized = normalizedWorkloadWeights(profile);
	features.wRange = normalized[0];
	features.wRadius = normalized[1];
	features.wKnn = normalized[2];
	features.knnK = profile.knnK;
	features.numQueries = profile.numQueries;
	features.rangeScaleMin = std::min(profile.rangeScaleMin, profile.rangeScaleMax);
	features.rangeScaleMax = std::max(profile.rangeScaleMin, profile.rangeScaleMax);
	features.radiusScaleMin = std::min(profile.radiusScaleMin, profile.radiusScaleMax);
	features.radiusScaleMax = std::max(profile.radiusScaleMin, profile.radiusScaleMax);
	features.buildWeight = weights.lambdaBuild;
	features.memoryWeight = weights.lambdaMemory;

	if (profile.numQueries == 0)
		return features;

	std::mt19937 rng(profile.querySeed);
	std::discrete_distribution<size_t> queryType(normalized.begin(), normalized.end());

	std::vector<double> scales;
	scales.reserve(profile.numQueries);
	for (size_t i = 0; i < profile.numQueries; ++i)
	{
		const size_t type = queryType(rng);
		if (type == 0)
		{
			std::uniform_real_distribution<double> distribution(features.rangeScaleMin, features.rangeScaleMax);
			scales.push_back(distribution(rng));
		}
		else if (type == 1)
		{
			std::uniform_real_distribution<double> distribution(features.radiusScaleMin, features.radiusScaleMax);
			scales.push_back(distribution(rng));
		}
		else
		{
			scales.push_back(0.0);
		}
	}

	features.queryScaleMean = std::accumulate(scales.begin(), scales.end(), 0.0) / static_cast<double>(scales.size());
	double variance = 0.0;
	for (const double scale : scales)
	{
		const double delta = scale - features.queryScaleMean;
		variance += delta * delta;
	}
	features.queryScaleStd = std::sqrt(variance / static_cast<double>(scales.size()));
	return features;
}

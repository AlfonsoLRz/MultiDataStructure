#pragma once

#include "../stdafx.h"
#include "../workloads/points/PointCloud.h"

namespace Experiments
{
	struct WorkloadProfile;
	struct ScoreWeights;

	struct PointCloudFeatures
	{
		size_t numPoints = 0;
		size_t sampleSize = 0;
		double bboxX = 0.0;
		double bboxY = 0.0;
		double bboxZ = 0.0;
		double aspectXY = 0.0;
		double aspectXZ = 0.0;
		double aspectYZ = 0.0;
		double densityBbox = 0.0;
		double heightMean = 0.0;
		double heightStd = 0.0;
		double heightRange = 0.0;
		double covEig0 = 0.0;
		double covEig1 = 0.0;
		double covEig2 = 0.0;
		double linearity = 0.0;
		double planarity = 0.0;
		double scattering = 0.0;
		double occupancyRatio8 = 0.0;
		double occupancyEntropy8 = 0.0;
		double densityCv8 = 0.0;
		double verticalityScore = 0.0;
		double flatnessScore = 0.0;
	};

	struct WorkloadFeatures
	{
		double wRange = 0.0;
		double wRadius = 0.0;
		double wKnn = 0.0;
		size_t knnK = 0;
		size_t numQueries = 0;
		double rangeScaleMin = 0.0;
		double rangeScaleMax = 0.0;
		double radiusScaleMin = 0.0;
		double radiusScaleMax = 0.0;
		double queryScaleMean = 0.0;
		double queryScaleStd = 0.0;
		double buildWeight = 0.0;
		double memoryWeight = 0.0;
	};

	PointCloudFeatures extractPointCloudFeatures(const PointCloud& cloud, size_t maxSampleSize = 8192, uint32_t seed = 1337);
	WorkloadFeatures extractWorkloadFeatures(const WorkloadProfile& profile, const ScoreWeights& weights);
}

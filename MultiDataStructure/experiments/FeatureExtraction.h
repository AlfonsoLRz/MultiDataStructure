#pragma once

#include "../stdafx.h"
#include "../workloads/points/PointCloud.h"

namespace Experiments
{
	struct WorkloadProfile;
	struct ScoreWeights;

	struct PointCloudFeatures
	{
		size_t _numPoints = 0;
		size_t _sampleSize = 0;
		double _bboxX = 0.0;
		double _bboxY = 0.0;
		double _bboxZ = 0.0;
		double _aspectXY = 0.0;
		double _aspectXZ = 0.0;
		double _aspectYZ = 0.0;
		double _densityBbox = 0.0;
		double _heightMean = 0.0;
		double _heightStd = 0.0;
		double _heightRange = 0.0;
		double _covEig0 = 0.0;
		double _covEig1 = 0.0;
		double _covEig2 = 0.0;
		double _linearity = 0.0;
		double _planarity = 0.0;
		double _scattering = 0.0;
		double _occupancyRatio8 = 0.0;
		double _occupancyEntropy8 = 0.0;
		double _densityCv8 = 0.0;
		double _verticalityScore = 0.0;
		double _flatnessScore = 0.0;
	};

	struct WorkloadFeatures
	{
		double _wRange = 0.0;
		double _wRadius = 0.0;
		double _wKnn = 0.0;
		size_t _knnK = 0;
		size_t _numQueries = 0;
		double _rangeScaleMin = 0.0;
		double _rangeScaleMax = 0.0;
		double _radiusScaleMin = 0.0;
		double _radiusScaleMax = 0.0;
		double _queryScaleMean = 0.0;
		double _queryScaleStd = 0.0;
		double _buildWeight = 0.0;
		double _memoryWeight = 0.0;
	};

	PointCloudFeatures extractPointCloudFeatures(const PointCloud& cloud, size_t maxSampleSize = 8192, uint32_t seed = 1337);
	WorkloadFeatures extractWorkloadFeatures(const WorkloadProfile& profile, const ScoreWeights& weights);
}

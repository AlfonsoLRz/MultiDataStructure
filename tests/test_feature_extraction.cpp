#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/experiments/FeatureExtraction.h"
#include "../MultiDataStructure/experiments/SchemaSearch.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		bool nearlyEqual(double left, double right, double epsilon = 0.0001)
		{
			return std::abs(left - right) <= epsilon;
		}

		PointPrimitive makePoint(float x, float y, float z, uint64_t)
		{
			PointPrimitive point;
			point.position = glm::vec3(x, y, z);
			return point;
		}
	}

	void runFeatureExtractionTests()
	{
		PointCloud plane;
		plane.addPoint(makePoint(0.0f, 0.0f, 0.0f, 0));
		plane.addPoint(makePoint(2.0f, 0.0f, 0.0f, 1));
		plane.addPoint(makePoint(0.0f, 2.0f, 0.0f, 2));
		plane.addPoint(makePoint(2.0f, 2.0f, 0.0f, 3));

		const Experiments::PointCloudFeatures planeFeatures = Experiments::extractPointCloudFeatures(plane);
		expect(planeFeatures._numPoints == 4, "point features count points");
		expect(planeFeatures._sampleSize == 4, "point features record sample size");
		expect(nearlyEqual(planeFeatures._bboxX, 2.0), "point features bbox x");
		expect(nearlyEqual(planeFeatures._bboxY, 2.0), "point features bbox y");
		expect(nearlyEqual(planeFeatures._bboxZ, 0.0), "point features bbox z");
		expect(nearlyEqual(planeFeatures._aspectXY, 1.0), "point features aspect xy");
		expect(nearlyEqual(planeFeatures._densityBbox, 1.0), "point features area density fallback");
		expect(nearlyEqual(planeFeatures._heightMean, 0.0), "point features height mean");
		expect(nearlyEqual(planeFeatures._heightStd, 0.0), "point features height std");
		expect(nearlyEqual(planeFeatures._covEig0, 1.0), "point features largest covariance eigenvalue");
		expect(nearlyEqual(planeFeatures._covEig1, 1.0), "point features middle covariance eigenvalue");
		expect(nearlyEqual(planeFeatures._covEig2, 0.0), "point features smallest covariance eigenvalue");
		expect(nearlyEqual(planeFeatures._planarity, 1.0), "point features planarity");
		expect(nearlyEqual(planeFeatures._scattering, 0.0), "point features scattering");
		expect(nearlyEqual(planeFeatures._occupancyRatio8, 4.0 / 512.0), "point features occupancy ratio");
		expect(nearlyEqual(planeFeatures._occupancyEntropy8, std::log(4.0)), "point features occupancy entropy");
		expect(planeFeatures._flatnessScore > planeFeatures._verticalityScore, "flat plane has stronger flatness than verticality");

		PointCloud sampled;
		for (size_t i = 0; i < 20; ++i)
			sampled.addPoint(makePoint(static_cast<float>(i), static_cast<float>(i % 3), static_cast<float>(i % 5), static_cast<uint64_t>(i)));

		const Experiments::PointCloudFeatures sampleA = Experiments::extractPointCloudFeatures(sampled, 5, 44);
		const Experiments::PointCloudFeatures sampleB = Experiments::extractPointCloudFeatures(sampled, 5, 44);
		expect(sampleA._sampleSize == 5, "point features respect max sample size");
		expect(nearlyEqual(sampleA._heightMean, sampleB._heightMean), "point features deterministic sampled height mean");
		expect(nearlyEqual(sampleA._covEig0, sampleB._covEig0), "point features deterministic sampled covariance");
		expect(nearlyEqual(sampleA._occupancyEntropy8, sampleB._occupancyEntropy8), "point features deterministic sampled occupancy");

		Experiments::WorkloadProfile profile;
		profile._rangeWeight = 0.2;
		profile._radiusWeight = 0.1;
		profile._knnWeight = 0.7;
		profile._knnK = 12;
		profile._numQueries = 20;
		profile._querySeed = 99;
		profile._rangeScaleMin = 0.02;
		profile._rangeScaleMax = 0.25;
		profile._radiusScaleMin = 0.03;
		profile._radiusScaleMax = 0.12;

		Experiments::ScoreWeights weights;
		weights._lambdaBuild = 0.002;
		weights._lambdaMemory = 0.03;

		const Experiments::WorkloadFeatures workloadA = Experiments::extractWorkloadFeatures(profile, weights);
		const Experiments::WorkloadFeatures workloadB = Experiments::extractWorkloadFeatures(profile, weights);
		expect(nearlyEqual(workloadA._wRange, 0.2), "workload features normalize range weight");
		expect(nearlyEqual(workloadA._wRadius, 0.1), "workload features normalize radius weight");
		expect(nearlyEqual(workloadA._wKnn, 0.7), "workload features normalize knn weight");
		expect(workloadA._knnK == 12, "workload features record knn k");
		expect(workloadA._numQueries == 20, "workload features record query count");
		expect(nearlyEqual(workloadA._rangeScaleMin, 0.02), "workload features record range scale min");
		expect(nearlyEqual(workloadA._rangeScaleMax, 0.25), "workload features record range scale max");
		expect(nearlyEqual(workloadA._radiusScaleMin, 0.03), "workload features record radius scale min");
		expect(nearlyEqual(workloadA._radiusScaleMax, 0.12), "workload features record radius scale max");
		expect(nearlyEqual(workloadA._buildWeight, 0.002), "workload features record build score weight");
		expect(nearlyEqual(workloadA._memoryWeight, 0.03), "workload features record memory score weight");
		expect(nearlyEqual(workloadA._queryScaleMean, workloadB._queryScaleMean), "workload features deterministic scale mean");
		expect(nearlyEqual(workloadA._queryScaleStd, workloadB._queryScaleStd), "workload features deterministic scale std");
	}
}

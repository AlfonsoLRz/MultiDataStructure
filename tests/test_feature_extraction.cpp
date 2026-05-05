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
		expect(planeFeatures.numPoints == 4, "point features count points");
		expect(planeFeatures.sampleSize == 4, "point features record sample size");
		expect(nearlyEqual(planeFeatures.bboxX, 2.0), "point features bbox x");
		expect(nearlyEqual(planeFeatures.bboxY, 2.0), "point features bbox y");
		expect(nearlyEqual(planeFeatures.bboxZ, 0.0), "point features bbox z");
		expect(nearlyEqual(planeFeatures.aspectXY, 1.0), "point features aspect xy");
		expect(nearlyEqual(planeFeatures.densityBbox, 1.0), "point features area density fallback");
		expect(nearlyEqual(planeFeatures.heightMean, 0.0), "point features height mean");
		expect(nearlyEqual(planeFeatures.heightStd, 0.0), "point features height std");
		expect(nearlyEqual(planeFeatures.covEig0, 1.0), "point features largest covariance eigenvalue");
		expect(nearlyEqual(planeFeatures.covEig1, 1.0), "point features middle covariance eigenvalue");
		expect(nearlyEqual(planeFeatures.covEig2, 0.0), "point features smallest covariance eigenvalue");
		expect(nearlyEqual(planeFeatures.planarity, 1.0), "point features planarity");
		expect(nearlyEqual(planeFeatures.scattering, 0.0), "point features scattering");
		expect(nearlyEqual(planeFeatures.occupancyRatio8, 4.0 / 512.0), "point features occupancy ratio");
		expect(nearlyEqual(planeFeatures.occupancyEntropy8, std::log(4.0)), "point features occupancy entropy");
		expect(planeFeatures.flatnessScore > planeFeatures.verticalityScore, "flat plane has stronger flatness than verticality");

		PointCloud sampled;
		for (size_t i = 0; i < 20; ++i)
			sampled.addPoint(makePoint(static_cast<float>(i), static_cast<float>(i % 3), static_cast<float>(i % 5), static_cast<uint64_t>(i)));

		const Experiments::PointCloudFeatures sampleA = Experiments::extractPointCloudFeatures(sampled, 5, 44);
		const Experiments::PointCloudFeatures sampleB = Experiments::extractPointCloudFeatures(sampled, 5, 44);
		expect(sampleA.sampleSize == 5, "point features respect max sample size");
		expect(nearlyEqual(sampleA.heightMean, sampleB.heightMean), "point features deterministic sampled height mean");
		expect(nearlyEqual(sampleA.covEig0, sampleB.covEig0), "point features deterministic sampled covariance");
		expect(nearlyEqual(sampleA.occupancyEntropy8, sampleB.occupancyEntropy8), "point features deterministic sampled occupancy");

		Experiments::WorkloadProfile profile;
		profile.rangeWeight = 0.2;
		profile.radiusWeight = 0.1;
		profile.knnWeight = 0.7;
		profile.knnK = 12;
		profile.numQueries = 20;
		profile.querySeed = 99;
		profile.rangeScaleMin = 0.02;
		profile.rangeScaleMax = 0.25;
		profile.radiusScaleMin = 0.03;
		profile.radiusScaleMax = 0.12;

		Experiments::ScoreWeights weights;
		weights.lambdaBuild = 0.002;
		weights.lambdaMemory = 0.03;

		const Experiments::WorkloadFeatures workloadA = Experiments::extractWorkloadFeatures(profile, weights);
		const Experiments::WorkloadFeatures workloadB = Experiments::extractWorkloadFeatures(profile, weights);
		expect(nearlyEqual(workloadA.wRange, 0.2), "workload features normalize range weight");
		expect(nearlyEqual(workloadA.wRadius, 0.1), "workload features normalize radius weight");
		expect(nearlyEqual(workloadA.wKnn, 0.7), "workload features normalize knn weight");
		expect(workloadA.knnK == 12, "workload features record knn k");
		expect(workloadA.numQueries == 20, "workload features record query count");
		expect(nearlyEqual(workloadA.rangeScaleMin, 0.02), "workload features record range scale min");
		expect(nearlyEqual(workloadA.rangeScaleMax, 0.25), "workload features record range scale max");
		expect(nearlyEqual(workloadA.radiusScaleMin, 0.03), "workload features record radius scale min");
		expect(nearlyEqual(workloadA.radiusScaleMax, 0.12), "workload features record radius scale max");
		expect(nearlyEqual(workloadA.buildWeight, 0.002), "workload features record build score weight");
		expect(nearlyEqual(workloadA.memoryWeight, 0.03), "workload features record memory score weight");
		expect(nearlyEqual(workloadA.queryScaleMean, workloadB.queryScaleMean), "workload features deterministic scale mean");
		expect(nearlyEqual(workloadA.queryScaleStd, workloadB.queryScaleStd), "workload features deterministic scale std");
	}
}

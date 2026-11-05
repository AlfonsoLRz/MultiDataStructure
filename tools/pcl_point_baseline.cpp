// External replay baseline against the Point Cloud Library (pointclouds.org, BSD): pcl::KdTreeFLANN
// (FLANN's single kd-tree, PCL's default search structure) and pcl::octree::OctreePointCloudSearch,
// single-threaded. Protocol and output: tools/baseline_common.h.
//
// The octree has no default resolution: the caller picks the voxel side. It is swept as the cloud's
// longest extent divided by --octree-divisions (128 is what the older Windows rows used). KdTreeFLANN
// radius results are requested unsorted, as for the other libraries. Points are float in the
// cloud's local frame; a PointXYZ is 16 bytes, so the copy into a pcl::PointCloud is not timed.
//
// Build: tools/build_baselines_linux.sh
// Usage:
//   pcl_point_baseline --input <cloud.mdspc> --query-trace <trace.csv> --summary-csv <out.csv>
//                      [--query-csv <q.csv>] [--max-queries N] [--warmup W] [--repeats R]
//                      [--octree-divisions 128,256,512,1024,2048,4096] [--kdtree 1]

#include "baseline_common.h"

#include <memory>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>
#include <pcl/pcl_config.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

int main(int argc, char** argv)
{
	try
	{
		using Point = pcl::PointXYZ;
		const Baseline::Options options = Baseline::parseOptions(argc, argv, { "--octree-divisions", "--kdtree" });
		const Baseline::Cloud cloud = Baseline::loadMdspc(options.input);
		const std::vector<Baseline::Query> queries = Baseline::loadTrace(options.trace, cloud, options.maxQueries);

		pcl::PointCloud<Point>::Ptr points(new pcl::PointCloud<Point>);
		points->resize(cloud.size());
		float lo[3] = { INFINITY, INFINITY, INFINITY }, hi[3] = { -INFINITY, -INFINITY, -INFINITY };
		for (size_t i = 0; i < cloud.size(); ++i)
		{
			const float* p = cloud.point(i);
			(*points)[i] = Point(p[0], p[1], p[2]);
			for (int a = 0; a < 3; ++a)
			{
				lo[a] = std::min(lo[a], p[a]);
				hi[a] = std::max(hi[a], p[a]);
			}
		}
		const double longest = std::max({ hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2] });

		std::unique_ptr<pcl::KdTreeFLANN<Point>> kdtree;
		std::unique_ptr<pcl::octree::OctreePointCloudSearch<Point>> octree;
		pcl::Indices indices;
		std::vector<float> distances;

		const auto answer = [&](size_t& count, uint64_t& checksum) {
			count = indices.size();
			for (const auto index : indices)
				Baseline::mixChecksum(checksum, static_cast<size_t>(index));
		};
		// Each search returns before the clock stops; `indices` is read after it.
		const auto timedSearch = [&](const Baseline::Query& q, auto&& radiusSearch, auto&& knnSearch) {
			const Point center(q.center[0], q.center[1], q.center[2]);
			if (q.kind == Baseline::Kind::Radius)
			{
				const auto start = Baseline::Clock::now();
				radiusSearch(center, q.radius);
				return Baseline::msSince(start);
			}
			if (q.kind == Baseline::Kind::Knn)
			{
				const auto start = Baseline::Clock::now();
				knnSearch(center, static_cast<int>(q.k));
				return Baseline::msSince(start);
			}
			return -1.0;
		};

		std::vector<Baseline::Method> methods;
		if (options.list("--kdtree", "1").front() != 0.0)
		{
			Baseline::Method method;
			method.name = "pcl_kdtree_flann";
			method.build = [&]() {
				const auto start = Baseline::Clock::now();
				kdtree = std::make_unique<pcl::KdTreeFLANN<Point>>(false);	// unsorted radius results
				kdtree->setInputCloud(points);
				return Baseline::msSince(start);
			};
			method.query = [&](const Baseline::Query& q) {
				return timedSearch(q,
					[&](const Point& c, float r) { kdtree->radiusSearch(c, r, indices, distances); },
					[&](const Point& c, int k) { kdtree->nearestKSearch(c, k, indices, distances); });
			};
			method.answer = answer;
			method.release = [&]() { kdtree.reset(); };
			methods.push_back(std::move(method));
		}
		for (const double divisions : options.list("--octree-divisions", "128,256,512,1024,2048,4096"))
		{
			Baseline::Method method;
			method.name = "pcl_octree_div" + std::to_string(static_cast<size_t>(divisions));
			method.build = [&, divisions]() {
				const auto start = Baseline::Clock::now();
				octree = std::make_unique<pcl::octree::OctreePointCloudSearch<Point>>(longest / divisions);
				octree->setInputCloud(points);
				octree->addPointsFromInputCloud();
				const double ms = Baseline::msSince(start);
				std::cout << "  pcl_octree_div" << static_cast<size_t>(divisions) << ": resolution " << longest / divisions << ", "
						  << octree->getLeafCount() << " leaves, " << static_cast<double>(cloud.size()) / octree->getLeafCount()
						  << " points per leaf, depth " << octree->getTreeDepth() << '\n';
				return ms;
			};
			method.query = [&](const Baseline::Query& q) {
				return timedSearch(q,
					[&](const Point& c, float r) { octree->radiusSearch(c, r, indices, distances); },
					[&](const Point& c, int k) { octree->nearestKSearch(c, k, indices, distances); });
			};
			method.answer = answer;
			method.release = [&]() { octree.reset(); };
			methods.push_back(std::move(method));
		}
		Baseline::runAll(std::string("PCL ") + PCL_VERSION_PRETTY, options, cloud, queries, methods);
		return 0;
	}
	catch (const std::exception& e)
	{
		std::cerr << "error: " << e.what() << '\n';
		return 1;
	}
}

#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/PointCloud.h"
#include "../MultiDataStructure/workloads/points/PointSpatialIndex.h"
#include "../MultiDataStructure/workloads/points/SyntheticPointClouds.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	namespace
	{
		bool nearlyEqual(float left, float right, float epsilon = 0.0001f)
		{
			return std::abs(left - right) <= epsilon;
		}

		bool nearlyEqual(double left, double right, double epsilon = 0.000001)
		{
			return std::abs(left - right) <= epsilon;
		}

		std::filesystem::path tempFile(const std::string& filename)
		{
			return std::filesystem::temp_directory_path() / filename;
		}

		void removePointCloudCache(const std::filesystem::path& sourcePath)
		{
			std::filesystem::remove(PointCloud::cachePathFor(sourcePath.string()));
		}

		template <typename T>
		void writeValueAt(std::fstream& file, std::streamoff offset, T value)
		{
			file.seekp(offset, std::ios::beg);
			file.write(reinterpret_cast<const char*>(&value), sizeof(T));
		}

		void writeLasHeader(
			std::fstream& file,
			uint32_t pointCount,
			double xScale,
			double yScale,
			double zScale,
			double xOffset,
			double yOffset,
			double zOffset,
			double minX,
			double maxX,
			double minY,
			double maxY,
			double minZ,
			double maxZ)
		{
			file.seekp(0, std::ios::beg);
			file.write("LASF", 4);
			writeValueAt<uint8_t>(file, 24, 1);
			writeValueAt<uint8_t>(file, 25, 2);
			writeValueAt<uint16_t>(file, 94, 227);
			writeValueAt<uint32_t>(file, 96, 227);
			writeValueAt<uint8_t>(file, 104, 0);
			writeValueAt<uint16_t>(file, 105, 20);
			writeValueAt<uint32_t>(file, 107, pointCount);
			writeValueAt<double>(file, 131, xScale);
			writeValueAt<double>(file, 139, yScale);
			writeValueAt<double>(file, 147, zScale);
			writeValueAt<double>(file, 155, xOffset);
			writeValueAt<double>(file, 163, yOffset);
			writeValueAt<double>(file, 171, zOffset);
			writeValueAt<double>(file, 179, maxX);
			writeValueAt<double>(file, 187, minX);
			writeValueAt<double>(file, 195, maxY);
			writeValueAt<double>(file, 203, minY);
			writeValueAt<double>(file, 211, maxZ);
			writeValueAt<double>(file, 219, minZ);
		}

		void writeLasPoint(
			std::fstream& file,
			std::streamoff offset,
			int32_t rawX,
			int32_t rawY,
			int32_t rawZ,
			uint16_t intensity,
			uint8_t classification)
		{
			writeValueAt<int32_t>(file, offset + 0, rawX);
			writeValueAt<int32_t>(file, offset + 4, rawY);
			writeValueAt<int32_t>(file, offset + 8, rawZ);
			writeValueAt<uint16_t>(file, offset + 12, intensity);
			writeValueAt<uint8_t>(file, offset + 15, classification);
		}

		SchemaConfig makePrecisionSchema()
		{
			SchemaLevelConfig level;
			level._type = MultiDataStructure::DataStructureLevel::KDTreeNode;
			level._typeName = "KDTree";
			level._numLevels = 4;
			level._leafCapacity = 1;
			level._minPrimitivesToSplit = 2;

			SchemaConfig schema;
			schema._name = "precision_point_query_test";
			schema._levels.push_back(level);
			schema._buildPolicy._maxDepth = 4;
			schema._buildPolicy._leafCapacity = 1;
			schema._buildPolicy._minPrimitivesToSplit = 2;
			schema._buildPolicy._collapseSingleChild = false;
			schema._buildPolicy._removeEmptyNodes = true;
			schema._buildPolicy._allowOverlapDuplication = false;
			return schema;
		}
	}

	void runPointCloudTests()
	{
		const std::filesystem::path xyzPath = tempFile("multids_pointcloud_test.xyz");
		removePointCloudCache(xyzPath);
		{
			std::ofstream file(xyzPath);
			file << "# x y z intensity classification\n";
			file << "0 1 2 0.5 3\n";
			file << "2 3 4 0.7 5\n";
		}

		const PointCloud xyzCloud = PointCloud::load(xyzPath.string());
		expect(xyzCloud.size() == 2, "XYZ loader reads two points");
		expect(nearlyEqual(xyzCloud.points()[0].position.z, 2.0f), "XYZ loader stores z coordinate");
		expect(nearlyEqual(xyzCloud.bounds().min().x, 0.0f), "XYZ bounds min is correct");
		expect(nearlyEqual(xyzCloud.bounds().max().z, 4.0f), "XYZ bounds max is correct");
		expect(std::filesystem::exists(PointCloud::cachePathFor(xyzPath.string())), "XYZ load writes sibling binary cache");

		const PointCloud cachedXyzCloud = PointCloud::load(xyzPath.string());
		expect(cachedXyzCloud.loadedFromCache(), "second XYZ load uses binary cache");
		expect(cachedXyzCloud.size() == xyzCloud.size(), "cached XYZ load preserves point count");
		expect(nearlyEqual(cachedXyzCloud.points()[1].position.y, 3.0f), "cached XYZ load preserves coordinates");

		const std::filesystem::path csvPath = tempFile("multids_pointcloud_test.csv");
		removePointCloudCache(csvPath);
		{
			std::ofstream file(csvPath);
			file << "x,y,z,intensity,classification\n";
			file << "-1,0,1,0.25,2\n";
			file << "3,4,5,0.75,6\n";
		}

		const PointCloud csvCloud = PointCloud::load(csvPath.string());
		expect(csvCloud.size() == 2, "CSV loader skips header and reads two points");
		expect(nearlyEqual(csvCloud.points()[0].position.x, -1.0f), "CSV loader stores x coordinate");
		expect(nearlyEqual(csvCloud.points()[1].position.z, 5.0f), "CSV loader stores z coordinate");

		const std::filesystem::path plyPath = tempFile("multids_pointcloud_test.ply");
		removePointCloudCache(plyPath);
		{
			std::ofstream file(plyPath);
			file << "ply\n";
			file << "format ascii 1.0\n";
			file << "element vertex 2\n";
			file << "property float x\n";
			file << "property float y\n";
			file << "property float z\n";
			file << "property float intensity\n";
			file << "property uchar classification\n";
			file << "end_header\n";
			file << "1 2 3 42 4\n";
			file << "5 6 7 24 8\n";
		}

		const PointCloud plyCloud = PointCloud::load(plyPath.string());
		expect(plyCloud.size() == 2, "PLY loader reads two vertices");
		expect(nearlyEqual(plyCloud.points()[0].position.y, 2.0f), "PLY loader stores y coordinate");
		expect(nearlyEqual(plyCloud.points()[1].position.z, 7.0f), "PLY loader stores z coordinate");

		const std::filesystem::path binaryPlyPath = tempFile("multids_pointcloud_test_binary.ply");
		removePointCloudCache(binaryPlyPath);
		{
			std::ofstream file(binaryPlyPath, std::ios::binary);
			file << "ply\n";
			file << "format binary_little_endian 1.0\n";
			file << "element vertex 2\n";
			file << "property float x\n";
			file << "property float y\n";
			file << "property double z\n";
			file << "property uchar classification\n";
			file << "end_header\n";
			const auto writeFloat = [&](float v) { file.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
			const auto writeDouble = [&](double v) { file.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
			const auto writeByte = [&](uint8_t v) { file.write(reinterpret_cast<const char*>(&v), sizeof(v)); };
			writeFloat(1.5f); writeFloat(2.5f); writeDouble(3.25); writeByte(4);
			writeFloat(-5.0f); writeFloat(6.0f); writeDouble(-7.5); writeByte(9);
		}

		const PointCloud binaryPlyCloud = PointCloud::load(binaryPlyPath.string());
		expect(binaryPlyCloud.size() == 2, "binary PLY loader reads two vertices");
		expect(nearlyEqual(binaryPlyCloud.points()[0].position.x, 1.5f), "binary PLY loader stores x (float)");
		expect(nearlyEqual(binaryPlyCloud.points()[0].position.z, 3.25f), "binary PLY loader stores z (double)");
		expect(nearlyEqual(binaryPlyCloud.points()[1].position.x, -5.0f), "binary PLY loader keeps signed floats");
		expect(nearlyEqual(binaryPlyCloud.points()[1].position.z, -7.5f), "binary PLY loader stores second z (double)");

		const std::filesystem::path lasPath = tempFile("multids_pointcloud_test.las");
		removePointCloudCache(lasPath);
		{
			std::fstream file(lasPath, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
			std::vector<char> zeros(227 + 2 * 20, 0);
			file.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
			writeLasHeader(file, 2, 0.01, 0.01, 0.01, 1000.0, 2000.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);

			const std::streamoff firstPoint = 227;
			writeLasPoint(file, firstPoint, 100, 200, 300, 1500, 4);
			const std::streamoff secondPoint = firstPoint + 20;
			writeLasPoint(file, secondPoint, -50, 0, 125, 2500, 7);
		}

		const PointCloud lasCloud = PointCloud::load(lasPath.string());
		expect(lasCloud.size() == 2, "LAS loader reads two point records");
		expect(nearlyEqual(lasCloud.coordinateFrame().origin.x, 1001.0), "LAS loader chooses first point as fallback local origin");
		expect(nearlyEqual(lasCloud.points()[0].position.x, 0.0f), "LAS loader stores first x coordinate locally");
		expect(nearlyEqual(lasCloud.points()[0].position.z, 0.0f), "LAS loader stores first z coordinate locally");
		expect(nearlyEqual(lasCloud.points()[1].position.x, -1.5f), "LAS loader keeps signed local coordinates");
		expect(nearlyEqual(lasCloud.points()[1].position.z, -1.75f), "LAS loader stores second z coordinate locally");
		expect(nearlyEqual(lasCloud.toWorldPosition(lasCloud.points()[0].position).x, 1001.0), "LAS loader reconstructs first world x coordinate");
		expect(nearlyEqual(lasCloud.toWorldPosition(lasCloud.points()[1].position).z, 11.25), "LAS loader reconstructs second world z coordinate");

		const PointCloud cachedLasCloud = PointCloud::load(lasPath.string());
		expect(cachedLasCloud.loadedFromCache(), "second LAS load uses binary cache");
		expect(nearlyEqual(cachedLasCloud.coordinateFrame().origin.x, lasCloud.coordinateFrame().origin.x), "cached LAS preserves coordinate frame origin");
		expect(nearlyEqual(cachedLasCloud.toWorldPosition(cachedLasCloud.points()[1].position).x, 999.5), "cached LAS reconstructs world coordinates");

		const std::filesystem::path utmLasPath = tempFile("multids_pointcloud_precision_test.las");
		removePointCloudCache(utmLasPath);
		{
			std::fstream file(utmLasPath, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
			std::vector<char> zeros(227 + 3 * 20, 0);
			file.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
			writeLasHeader(
				file,
				3,
				0.01,
				0.01,
				0.01,
				500000.0,
				4500000.0,
				100.0,
				500000.01,
				500000.02,
				4500000.01,
				4500000.02,
				100.01,
				100.01);

			const std::streamoff firstPoint = 227;
			writeLasPoint(file, firstPoint, 1, 1, 1, 1000, 1);
			writeLasPoint(file, firstPoint + 20, 2, 1, 1, 1000, 1);
			writeLasPoint(file, firstPoint + 40, 1, 2, 1, 1000, 1);
		}

		const PointCloud utmCloud = PointCloud::load(utmLasPath.string());
		expect(utmCloud.size() == 3, "UTM-like LAS precision test reads all records");
		expect(nearlyEqual(utmCloud.coordinateFrame().origin.x, 500000.01), "UTM-like LAS uses header min x as local origin");
		expect(nearlyEqual(utmCloud.coordinateFrame().origin.y, 4500000.01), "UTM-like LAS uses header min y as local origin");
		expect(nearlyEqual(utmCloud.points()[1].position.x - utmCloud.points()[0].position.x, 0.01f, 0.00001f), "local coordinates preserve centimeter x separation");
		expect(nearlyEqual(utmCloud.toWorldPosition(utmCloud.points()[1].position).x, 500000.02), "UTM-like LAS reconstructs centimeter world x");

		PointSpatialIndex precisionIndex;
		precisionIndex.build(utmCloud, makePrecisionSchema());
		const glm::vec3 queryCenter = utmCloud.points()[1].position;
		const AABB tightBox(queryCenter - glm::vec3(0.002f), queryCenter + glm::vec3(0.002f));
		const PointSpatialIndex::CountResult tightCount = precisionIndex.countRange(tightBox);
		expect(tightCount._count == 1, "range/count query distinguishes centimeter-separated UTM points");
		const PointSpatialIndex::QueryResult tightKnn = precisionIndex.knnQuery(queryCenter, 1);
		expect(tightKnn._pointIndices.size() == 1 && tightKnn._pointIndices[0] == 1, "KNN distinguishes centimeter-separated UTM points");

		const PointCloud flat = SyntheticPointClouds::generateFlatTerrain(128, 10.0f, 20.0f, 0.1f);
		expect(flat.size() == 128, "flat terrain generator returns requested point count");
		expect(!flat.empty(), "flat terrain generator returns non-empty cloud");

		const PointCloud facade = SyntheticPointClouds::generateFacade(64, 8.0f, 12.0f, 0.2f);
		expect(facade.size() == 64, "facade generator returns requested point count");
		expect(facade.coordinateRange().z > 8.0f, "facade generator has vertical extent");

		const PointCloud urban = SyntheticPointClouds::generateUrbanMixed(50, 75, 3);
		expect(urban.size() >= 125, "urban generator returns at least requested terrain plus building points");

		const PointCloud mixture = SyntheticPointClouds::generateSparseDenseMixture(20, 40);
		expect(mixture.size() == 60, "sparse/dense generator returns requested point count");
		expect(mixture.size() == 60, "point-cloud stats count points");

		std::filesystem::remove(xyzPath);
		removePointCloudCache(xyzPath);
		std::filesystem::remove(csvPath);
		removePointCloudCache(csvPath);
		std::filesystem::remove(plyPath);
		removePointCloudCache(plyPath);
		std::filesystem::remove(lasPath);
		removePointCloudCache(lasPath);
		std::filesystem::remove(utmLasPath);
		removePointCloudCache(utmLasPath);
	}
}

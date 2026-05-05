#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/workloads/points/PointCloud.h"
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

		const std::filesystem::path lasPath = tempFile("multids_pointcloud_test.las");
		removePointCloudCache(lasPath);
		{
			std::fstream file(lasPath, std::ios::binary | std::ios::in | std::ios::out | std::ios::trunc);
			std::vector<char> zeros(227 + 2 * 20, 0);
			file.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
			file.seekp(0, std::ios::beg);
			file.write("LASF", 4);
			writeValueAt<uint8_t>(file, 24, 1);
			writeValueAt<uint8_t>(file, 25, 2);
			writeValueAt<uint16_t>(file, 94, 227);
			writeValueAt<uint32_t>(file, 96, 227);
			writeValueAt<uint8_t>(file, 104, 0);
			writeValueAt<uint16_t>(file, 105, 20);
			writeValueAt<uint32_t>(file, 107, 2);
			writeValueAt<double>(file, 131, 0.01);
			writeValueAt<double>(file, 139, 0.01);
			writeValueAt<double>(file, 147, 0.01);
			writeValueAt<double>(file, 155, 1000.0);
			writeValueAt<double>(file, 163, 2000.0);
			writeValueAt<double>(file, 171, 10.0);

			const std::streamoff firstPoint = 227;
			writeValueAt<int32_t>(file, firstPoint + 0, 100);
			writeValueAt<int32_t>(file, firstPoint + 4, 200);
			writeValueAt<int32_t>(file, firstPoint + 8, 300);
			writeValueAt<uint16_t>(file, firstPoint + 12, 1500);
			writeValueAt<uint8_t>(file, firstPoint + 15, 4);

			const std::streamoff secondPoint = firstPoint + 20;
			writeValueAt<int32_t>(file, secondPoint + 0, -50);
			writeValueAt<int32_t>(file, secondPoint + 4, 0);
			writeValueAt<int32_t>(file, secondPoint + 8, 125);
			writeValueAt<uint16_t>(file, secondPoint + 12, 2500);
			writeValueAt<uint8_t>(file, secondPoint + 15, 7);
		}

		const PointCloud lasCloud = PointCloud::load(lasPath.string());
		expect(lasCloud.size() == 2, "LAS loader reads two point records");
		expect(nearlyEqual(lasCloud.points()[0].position.x, 1001.0f), "LAS loader applies x scale and offset");
		expect(nearlyEqual(lasCloud.points()[0].position.z, 13.0f), "LAS loader applies z scale and offset");
		expect(nearlyEqual(lasCloud.points()[1].position.x, 999.5f), "LAS loader handles signed coordinates");
		expect(nearlyEqual(lasCloud.points()[1].position.z, 11.25f), "LAS loader applies z scale to second point");

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
	}
}

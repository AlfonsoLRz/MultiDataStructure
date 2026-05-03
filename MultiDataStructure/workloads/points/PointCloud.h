#pragma once

#include "../../stdafx.h"
#include "../../AABB.h"
#include "PointPrimitive.h"

class PointCloud
{
public:
	struct LoadOptions
	{
		bool useBinaryCache = true;
		bool rebuildBinaryCache = false;
	};

	static PointCloud load(const std::string& filename, const LoadOptions& options = {});
	static std::filesystem::path cachePathFor(const std::string& filename);

	size_t size() const;
	bool empty() const;
	const std::vector<PointPrimitive>& points() const;

	AABB bounds() const;
	glm::vec3 coordinateRange() const;
	float approximateDensity() const;

	bool loadedFromCache() const;
	const std::string& sourcePath() const;
	const std::string& cachePath() const;

	void reserve(size_t count);
	void addPoint(const PointPrimitive& point);
	void clear();

private:
	struct Stats
	{
		size_t numPoints = 0;
		AABB bounds;
		glm::vec3 coordinateRange = glm::vec3(0.0f);
		float approximateDensity = 0.0f;
	};

	std::vector<PointPrimitive> _points;
	Stats _stats;
	std::string _sourcePath;
	std::string _cachePath;
	bool _loadedFromCache = false;

	void recomputeStats();
	void updateStatsForPoint(const PointPrimitive& point);

	bool tryLoadBinaryCache(const std::string& sourcePath, const std::filesystem::path& cachePath);
	void saveBinaryCache(const std::string& sourcePath, const std::filesystem::path& cachePath) const;

	static PointCloud loadFromSource(const std::string& filename);
	static PointCloud loadXYZ(const std::string& filename);
	static PointCloud loadCSV(const std::string& filename);
	static PointCloud loadPLY(const std::string& filename);
	static PointCloud loadLAS(const std::string& filename);
};

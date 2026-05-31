#pragma once

#include "../../stdafx.h"
#include "../../AABB.h"
#include "PointPrimitive.h"

class PointCloud
{
public:
	struct CoordinateFrame
	{
		glm::dvec3 origin = glm::dvec3(0.0);
		glm::dvec3 scale = glm::dvec3(1.0);
	};

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

	const CoordinateFrame& coordinateFrame() const;
	glm::dvec3 toWorldPosition(const glm::vec3& localPosition) const;
	glm::vec3 toLocalPosition(const glm::dvec3& worldPosition) const;
	// Bounds are expressed in the local coordinate frame used by hot query paths.
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
	CoordinateFrame _coordinateFrame;
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

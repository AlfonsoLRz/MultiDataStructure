#include "../stdafx.h"
#include "Config.h"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

namespace
{
	size_t asSize(const boost::json::object& object, const char* key, size_t fallback)
	{
		if (const boost::json::value* value = object.if_contains(key))
		{
			if (value->is_int64())
				return static_cast<size_t>(value->as_int64());
			if (value->is_uint64())
				return static_cast<size_t>(value->as_uint64());
			if (value->is_double())
				return static_cast<size_t>(value->as_double());
		}

		return fallback;
	}

	std::optional<size_t> optionalSize(const boost::json::object& object, std::initializer_list<const char*> keys)
	{
		for (const char* key : keys)
		{
			if (const boost::json::value* value = object.if_contains(key))
			{
				if (value->is_int64())
					return static_cast<size_t>(std::max<int64_t>(0, value->as_int64()));
				if (value->is_uint64())
					return static_cast<size_t>(value->as_uint64());
				if (value->is_double())
					return static_cast<size_t>(std::max(0.0, value->as_double()));
			}
		}

		return std::nullopt;
	}

	std::optional<double> optionalDouble(const boost::json::object& object, std::initializer_list<const char*> keys)
	{
		for (const char* key : keys)
		{
			if (const boost::json::value* value = object.if_contains(key))
			{
				if (value->is_double())
					return value->as_double();
				if (value->is_int64())
					return static_cast<double>(value->as_int64());
				if (value->is_uint64())
					return static_cast<double>(value->as_uint64());
			}
		}

		return std::nullopt;
	}

	bool asBool(const boost::json::object& object, const char* key, bool fallback)
	{
		if (const boost::json::value* value = object.if_contains(key))
		{
			if (value->is_bool())
				return value->as_bool();
		}

		return fallback;
	}

	std::string asString(const boost::json::object& object, const char* key, const std::string& fallback = {})
	{
		if (const boost::json::value* value = object.if_contains(key))
		{
			if (value->is_string())
				return std::string(value->as_string().c_str());
		}

		return fallback;
	}

	std::string normalizeTypeName(std::string value)
	{
		value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
			return std::isspace(c) || c == '_' || c == '-';
			}), value.end());
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
		return value;
	}

	bool pathExists(const std::filesystem::path& path)
	{
		std::error_code error;
		return std::filesystem::exists(path, error);
	}

	std::filesystem::path resolveConfigPath(const std::string& filename)
	{
		const std::filesystem::path configuredPath(filename);
		if (pathExists(configuredPath) || configuredPath.is_absolute())
			return configuredPath;

		std::error_code error;
		std::filesystem::path current = std::filesystem::absolute(std::filesystem::current_path(), error);
		if (error)
			return configuredPath;

		for (;;)
		{
			const std::filesystem::path candidate = (current / configuredPath).lexically_normal();
			if (pathExists(candidate))
				return candidate;

			if (!current.has_parent_path() || current == current.parent_path())
				break;

			current = current.parent_path();
		}

		return configuredPath;
	}

	SchemaLevelCondition parseLevelCondition(const boost::json::object& object)
	{
		SchemaLevelCondition condition;
		condition.minPoints = optionalSize(object, { "minPoints", "pointCountMin", "minPointCount", "point_count_min" });
		condition.maxPoints = optionalSize(object, { "maxPoints", "pointCountMax", "maxPointCount", "point_count_max" });
		condition.minDensity = optionalDouble(object, { "minDensity", "densityMin", "density_min" });
		condition.maxDensity = optionalDouble(object, { "maxDensity", "densityMax", "density_max" });
		condition.minHeightRatio = optionalDouble(object, { "minHeightRatio", "heightRatioMin", "height_ratio_min" });
		condition.maxHeightRatio = optionalDouble(object, { "maxHeightRatio", "heightRatioMax", "height_ratio_max" });
		condition.minExtentX = optionalDouble(object, { "minExtentX", "extentXMin", "extent_x_min" });
		condition.maxExtentX = optionalDouble(object, { "maxExtentX", "extentXMax", "extent_x_max" });
		condition.minExtentY = optionalDouble(object, { "minExtentY", "extentYMin", "extent_y_min" });
		condition.maxExtentY = optionalDouble(object, { "maxExtentY", "extentYMax", "extent_y_max" });
		condition.minExtentZ = optionalDouble(object, { "minExtentZ", "extentZMin", "extent_z_min" });
		condition.maxExtentZ = optionalDouble(object, { "maxExtentZ", "extentZMax", "extent_z_max" });
		condition.minAnisotropy = optionalDouble(object, { "minAnisotropy", "anisotropyMin", "anisotropy_min" });
		condition.maxAnisotropy = optionalDouble(object, { "maxAnisotropy", "anisotropyMax", "anisotropy_max" });
		condition.minOccupancyEntropy = optionalDouble(object, { "minOccupancyEntropy", "occupancyEntropyMin", "occupancy_entropy_min" });
		condition.maxOccupancyEntropy = optionalDouble(object, { "maxOccupancyEntropy", "occupancyEntropyMax", "occupancy_entropy_max" });
		return condition;
	}

	SchemaLevelConfig parseLevel(const boost::json::object& object)
	{
		SchemaLevelConfig level;
		level.typeName = asString(object, "type", level.typeName);
		level.primitiveKind = Config::parseSchemaPrimitiveKind(level.typeName);
		level.typeName = Config::schemaPrimitiveKindName(level.primitiveKind);
		level.cpuFallbackType = Config::cpuFallbackForPrimitiveKind(level.primitiveKind);
		level.type = level.cpuFallbackType;
		level.numLevels = asSize(object, "numLevels", level.numLevels);
		level.leafCapacity = asSize(object, "leafCapacity", level.leafCapacity);
		level.minPrimitivesToSplit = asSize(object, "minPointsToSplit", level.minPrimitivesToSplit);
		level.minPrimitivesToSplit = asSize(object, "minPrimitivesToSplit", level.minPrimitivesToSplit);
		level.axisPolicy = asString(object, "axisPolicy", level.axisPolicy);
		if (const boost::json::value* conditionValue = object.if_contains("condition"))
		{
			if (!conditionValue->is_object())
				throw std::runtime_error("Schema level condition must be an object");
			level.condition = parseLevelCondition(conditionValue->as_object());
		}

		if (level.numLevels == 0)
			throw std::runtime_error("Schema level numLevels must be greater than zero");

		return level;
	}

	BuildPolicy parseBuildPolicy(const boost::json::object& object)
	{
		BuildPolicy policy;
		policy.maxDepth = asSize(object, "maxDepth", policy.maxDepth);
		policy.leafCapacity = asSize(object, "leafCapacity", policy.leafCapacity);
		policy.minPrimitivesToSplit = asSize(object, "minPointsToSplit", policy.minPrimitivesToSplit);
		policy.minPrimitivesToSplit = asSize(object, "minPrimitivesToSplit", policy.minPrimitivesToSplit);
		policy.collapseSingleChild = asBool(object, "collapseSingleChild", policy.collapseSingleChild);
		policy.removeEmptyNodes = asBool(object, "removeEmptyNodes", policy.removeEmptyNodes);
		policy.allowOverlapDuplication = asBool(object, "allowOverlapDuplication", policy.allowOverlapDuplication);
		return policy;
	}
}

bool SchemaLevelCondition::empty() const
{
	return !minPoints && !maxPoints &&
		!minDensity && !maxDensity &&
		!minHeightRatio && !maxHeightRatio &&
		!minExtentX && !maxExtentX &&
		!minExtentY && !maxExtentY &&
		!minExtentZ && !maxExtentZ &&
		!minAnisotropy && !maxAnisotropy &&
		!minOccupancyEntropy && !maxOccupancyEntropy;
}

size_t SchemaConfig::totalLevels() const
{
	size_t total = 0;
	for (const SchemaLevelConfig& level : levels)
		total += level.numLevels;

	return total;
}

std::vector<MultiDataStructure::LevelConfig> SchemaConfig::toLevelConfigs() const
{
	std::vector<MultiDataStructure::LevelConfig> result;
	result.reserve(levels.size());

	for (const SchemaLevelConfig& level : levels)
	{
		MultiDataStructure::LevelConfig converted;
		converted._levelType = level.cpuFallbackType;
		converted._numLevels = static_cast<glm::uint>(level.numLevels);
		converted._leafCapacity = level.leafCapacity;
		converted._minPrimitivesToSplit = level.minPrimitivesToSplit;
		result.push_back(converted);
	}

	return result;
}

const SchemaLevelConfig& SchemaConfig::levelForDepth(size_t depth) const
{
	if (levels.empty())
		throw std::runtime_error("Schema has no levels");

	size_t cumulative = 0;
	for (const SchemaLevelConfig& level : levels)
	{
		cumulative += level.numLevels;
		if (depth < cumulative)
			return level;
	}

	return levels.back();
}

SchemaConfig Config::loadSchemaConfig(const std::string& filename)
{
	const std::filesystem::path resolvedPath = resolveConfigPath(filename);
	std::ifstream file(resolvedPath);
	if (!file.is_open())
		throw std::runtime_error("Unable to open schema config: " + filename);

	std::stringstream buffer;
	buffer << file.rdbuf();
	return parseSchemaConfig(buffer.str(), resolvedPath.string());
}

SchemaConfig Config::parseSchemaConfig(const std::string& jsonText, const std::string& sourceName)
{
	boost::system::error_code error;
	boost::json::value rootValue = boost::json::parse(jsonText, error);
	if (error)
		throw std::runtime_error("Invalid schema JSON" + (sourceName.empty() ? std::string() : " in " + sourceName) + ": " + error.message());

	if (!rootValue.is_object())
		throw std::runtime_error("Schema JSON root must be an object");

	const boost::json::object& root = rootValue.as_object();

	SchemaConfig schema;
	schema.name = asString(root, "name", sourceName);

	if (const boost::json::value* policyValue = root.if_contains("buildPolicy"))
	{
		if (!policyValue->is_object())
			throw std::runtime_error("buildPolicy must be an object");
		schema.buildPolicy = parseBuildPolicy(policyValue->as_object());
	}

	const boost::json::value* levelsValue = root.if_contains("levels");
	if (!levelsValue || !levelsValue->is_array())
		throw std::runtime_error("Schema config requires a levels array");

	for (const boost::json::value& levelValue : levelsValue->as_array())
	{
		if (!levelValue.is_object())
			throw std::runtime_error("Each schema level must be an object");
		schema.levels.push_back(parseLevel(levelValue.as_object()));
	}

	if (schema.levels.empty())
		throw std::runtime_error("Schema config requires at least one level");

	if (schema.buildPolicy.maxDepth == 0)
		schema.buildPolicy.maxDepth = schema.totalLevels();

	return schema;
}

MultiDataStructure::DataStructureLevel Config::parseDataStructureLevel(const std::string& value)
{
	return cpuFallbackForPrimitiveKind(parseSchemaPrimitiveKind(value));
}

SchemaPrimitiveKind Config::parseSchemaPrimitiveKind(const std::string& value)
{
	const std::string normalized = normalizeTypeName(value);
	if (normalized == "quadtree" || normalized == "quadtreenode")
		return SchemaPrimitiveKind::QuadTree;
	if (normalized == "kdtree" || normalized == "kdtreenode")
		return SchemaPrimitiveKind::KDTree;
	if (normalized == "bih" || normalized == "binaryintervalhierarchy" || normalized == "intervalhierarchy")
		return SchemaPrimitiveKind::BIH;
	if (normalized == "octree" || normalized == "octreenode")
		return SchemaPrimitiveKind::Octree;
	if (normalized == "karrasoctree" || normalized == "mortonoctree" || normalized == "octreekarras" || normalized == "octreemorton")
		return SchemaPrimitiveKind::KarrasOctree;
	if (normalized == "regulargrid" || normalized == "uniformgrid" || normalized == "grid" || normalized == "grid3d")
		return SchemaPrimitiveKind::RegularGrid;
	if (normalized == "hgrid" || normalized == "hierarchicalgrid" || normalized == "hierarchicalgrid3d")
		return SchemaPrimitiveKind::HGrid;
	if (normalized == "bvh" || normalized == "bvhnode")
		return SchemaPrimitiveKind::BVH;
	if (normalized == "lbvh" || normalized == "linearbvh")
		return SchemaPrimitiveKind::LBVH;
	if (normalized == "mixed" || normalized == "mixedtree")
		return SchemaPrimitiveKind::Mixed;

	throw std::runtime_error("Unsupported spatial structure type: " + value);
}

std::string Config::schemaPrimitiveKindName(SchemaPrimitiveKind kind)
{
	switch (kind)
	{
	case SchemaPrimitiveKind::QuadTree:
		return "QuadTree";
	case SchemaPrimitiveKind::Octree:
		return "Octree";
	case SchemaPrimitiveKind::KDTree:
		return "KDTree";
	case SchemaPrimitiveKind::BVH:
		return "BVH";
	case SchemaPrimitiveKind::LBVH:
		return "LBVH";
	case SchemaPrimitiveKind::BIH:
		return "BIH";
	case SchemaPrimitiveKind::KarrasOctree:
		return "KarrasOctree";
	case SchemaPrimitiveKind::RegularGrid:
		return "RegularGrid";
	case SchemaPrimitiveKind::HGrid:
		return "HGrid";
	case SchemaPrimitiveKind::Mixed:
		return "Mixed";
	default:
		return "Octree";
	}
}

MultiDataStructure::DataStructureLevel Config::cpuFallbackForPrimitiveKind(SchemaPrimitiveKind kind)
{
	switch (kind)
	{
	case SchemaPrimitiveKind::QuadTree:
		return MultiDataStructure::DataStructureLevel::QuadTreeNode;
	case SchemaPrimitiveKind::KDTree:
	case SchemaPrimitiveKind::BIH:
		return MultiDataStructure::DataStructureLevel::KDTreeNode;
	case SchemaPrimitiveKind::BVH:
	case SchemaPrimitiveKind::LBVH:
		return MultiDataStructure::DataStructureLevel::BvhNode;
	case SchemaPrimitiveKind::Octree:
	case SchemaPrimitiveKind::KarrasOctree:
	case SchemaPrimitiveKind::RegularGrid:
	case SchemaPrimitiveKind::HGrid:
	case SchemaPrimitiveKind::Mixed:
	default:
		return MultiDataStructure::DataStructureLevel::OctreeNode;
	}
}

std::string Config::dataStructureLevelName(MultiDataStructure::DataStructureLevel level)
{
	switch (level)
	{
	case MultiDataStructure::DataStructureLevel::QuadTreeNode:
		return "QuadTree";
	case MultiDataStructure::DataStructureLevel::KDTreeNode:
		return "KDTree";
	case MultiDataStructure::DataStructureLevel::OctreeNode:
		return "Octree";
	case MultiDataStructure::DataStructureLevel::BvhNode:
		return "BVH";
	default:
		return "Unknown";
	}
}

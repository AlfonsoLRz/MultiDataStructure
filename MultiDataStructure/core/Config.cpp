#include "../stdafx.h"
#include "Config.h"

#include <boost/json.hpp>
#include <boost/system/error_code.hpp>

static size_t asSize(const boost::json::object& object, const char* key, size_t fallback)
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

static std::optional<size_t> optionalSize(const boost::json::object& object, std::initializer_list<const char*> keys)
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

static std::optional<double> optionalDouble(const boost::json::object& object, std::initializer_list<const char*> keys)
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

static double asDouble(const boost::json::object& object, std::initializer_list<const char*> keys, double fallback)
{
	return optionalDouble(object, keys).value_or(fallback);
}

static bool asBool(const boost::json::object& object, const char* key, bool fallback)
{
	if (const boost::json::value* value = object.if_contains(key))
	{
		if (value->is_bool())
			return value->as_bool();
	}

	return fallback;
}

static std::string asString(const boost::json::object& object, const char* key, const std::string& fallback = {})
{
	if (const boost::json::value* value = object.if_contains(key))
	{
		if (value->is_string())
			return std::string(value->as_string().c_str());
	}

	return fallback;
}

static std::string normalizeTypeName(std::string value)
{
	value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
		return std::isspace(c) || c == '_' || c == '-';
		}), value.end());
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return value;
}

static std::string normalizeAxisPolicy(std::string value)
{
	value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
		return std::isspace(c) || c == '_' || c == '-';
		}), value.end());
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
		return static_cast<char>(std::tolower(c));
	});
	if (value == "xy")
		return "xy";
	if (value == "xz")
		return "xz";
	if (value == "yz")
		return "yz";
	if (value == "ignoreshortest" || value == "shortest")
		return "ignore_shortest";
	if (value == "ignorex" || value == "x")
		return "ignore_x";
	if (value == "ignorey" || value == "y")
		return "ignore_y";
	if (value == "ignorez" || value == "z")
		return "ignore_z";
	if (value == "medianlongestaxis" || value == "longestaxis" || value == "longestextent")
		return "median_longest_axis";
	if (value == "roundrobin")
		return "round_robin";
	if (value == "centerlongestaxis")
		return "center_longest_axis";
	return value;
}

static void normalizeLevelAxisPolicy(SchemaLevelConfig& level)
{
	level._axisPolicy = normalizeAxisPolicy(level._axisPolicy);
	if (level._primitiveKind == SchemaPrimitiveKind::QuadTree)
	{
		if (level._axisPolicy.empty())
		{
			level._axisPolicy = "xy";
			return;
		}
		if (level._axisPolicy == "xy" ||
			level._axisPolicy == "xz" ||
			level._axisPolicy == "yz" ||
			level._axisPolicy == "ignore_shortest" ||
			level._axisPolicy == "ignore_x" ||
			level._axisPolicy == "ignore_y" ||
			level._axisPolicy == "ignore_z")
			return;

		throw std::runtime_error("Unsupported QuadTree axisPolicy: " + level._axisPolicy);
	}

	if (level._type == MultiDataStructure::DataStructureLevel::KDTreeNode)
	{
		if (level._axisPolicy.empty())
			level._axisPolicy = "median_longest_axis";
		if (level._axisPolicy == "median_longest_axis" ||
			level._axisPolicy == "round_robin" ||
			level._axisPolicy == "center_longest_axis")
			return;

		throw std::runtime_error("Unsupported KDTree/BIH axisPolicy: " + level._axisPolicy);
	}
}

static bool pathExists(const std::filesystem::path& path)
{
	std::error_code error;
	return std::filesystem::exists(path, error);
}

static std::filesystem::path resolveConfigPath(const std::string& filename)
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

static SchemaLevelCondition parseLevelCondition(const boost::json::object& object)
{
	SchemaLevelCondition condition;
	condition._minPoints = optionalSize(object, { "minPoints", "pointCountMin", "minPointCount", "point_count_min" });
	condition._maxPoints = optionalSize(object, { "maxPoints", "pointCountMax", "maxPointCount", "point_count_max" });
	condition._minDensity = optionalDouble(object, { "minDensity", "densityMin", "density_min" });
	condition._maxDensity = optionalDouble(object, { "maxDensity", "densityMax", "density_max" });
	condition._minHeightRatio = optionalDouble(object, { "minHeightRatio", "heightRatioMin", "height_ratio_min" });
	condition._maxHeightRatio = optionalDouble(object, { "maxHeightRatio", "heightRatioMax", "height_ratio_max" });
	condition._minExtentX = optionalDouble(object, { "minExtentX", "extentXMin", "extent_x_min" });
	condition._maxExtentX = optionalDouble(object, { "maxExtentX", "extentXMax", "extent_x_max" });
	condition._minExtentY = optionalDouble(object, { "minExtentY", "extentYMin", "extent_y_min" });
	condition._maxExtentY = optionalDouble(object, { "maxExtentY", "extentYMax", "extent_y_max" });
	condition._minExtentZ = optionalDouble(object, { "minExtentZ", "extentZMin", "extent_z_min" });
	condition._maxExtentZ = optionalDouble(object, { "maxExtentZ", "extentZMax", "extent_z_max" });
	condition._minAnisotropy = optionalDouble(object, { "minAnisotropy", "anisotropyMin", "anisotropy_min" });
	condition._maxAnisotropy = optionalDouble(object, { "maxAnisotropy", "anisotropyMax", "anisotropy_max" });
	condition._minOccupancyEntropy = optionalDouble(object, { "minOccupancyEntropy", "occupancyEntropyMin", "occupancy_entropy_min" });
	condition._maxOccupancyEntropy = optionalDouble(object, { "maxOccupancyEntropy", "occupancyEntropyMax", "occupancy_entropy_max" });
	return condition;
}

static AdaptiveLeafCapacityConfig parseAdaptiveLeafCapacity(const boost::json::object& object)
{
	AdaptiveLeafCapacityConfig config;
	config._enabled = asBool(object, "enabled", true);
	config._minCapacity = optionalSize(object, { "minCapacity", "minLeafCapacity", "leafCapacityMin", "min_leaf_capacity" }).value_or(0);
	config._maxCapacity = optionalSize(object, { "maxCapacity", "maxLeafCapacity", "leafCapacityMax", "max_leaf_capacity" }).value_or(0);
	config._densityWeight = asDouble(object, { "densityWeight", "densityExponent", "density_factor_weight" }, config._densityWeight);
	config._anisotropyWeight = asDouble(object, { "anisotropyWeight", "anisotropyExponent", "anisotropy_factor_weight" }, config._anisotropyWeight);
	config._heightRatioWeight = asDouble(object, { "heightRatioWeight", "heightRatioExponent", "height_ratio_factor_weight" }, config._heightRatioWeight);
	config._queryMixFactor = asDouble(object, { "queryMixFactor", "queryFactor", "workloadFactor", "query_mix_factor" }, config._queryMixFactor);

	if (config._maxCapacity > 0 && config._minCapacity > 0 && config._maxCapacity < config._minCapacity)
		throw std::runtime_error("adaptiveLeafCapacity maxCapacity must be >= minCapacity");
	if (config._queryMixFactor <= 0.0)
		throw std::runtime_error("adaptiveLeafCapacity queryMixFactor must be positive");

	return config;
}

static SchemaLevelConfig parseLevel(const boost::json::object& object)
{
	SchemaLevelConfig level;
	level._typeName = asString(object, "type", level._typeName);
	level._primitiveKind = Config::parseSchemaPrimitiveKind(level._typeName);
	level._typeName = Config::schemaPrimitiveKindName(level._primitiveKind);
	level._cpuFallbackType = Config::cpuFallbackForPrimitiveKind(level._primitiveKind);
	level._type = level._cpuFallbackType;
	level._numLevels = asSize(object, "numLevels", level._numLevels);
	level._leafCapacity = asSize(object, "leafCapacity", level._leafCapacity);
	level._minPrimitivesToSplit = asSize(object, "minPointsToSplit", level._minPrimitivesToSplit);
	level._minPrimitivesToSplit = asSize(object, "minPrimitivesToSplit", level._minPrimitivesToSplit);
	level._axisPolicy = asString(object, "axisPolicy", level._axisPolicy);
	normalizeLevelAxisPolicy(level);
	if (const boost::json::value* conditionValue = object.if_contains("condition"))
	{
		if (!conditionValue->is_object())
			throw std::runtime_error("Schema level condition must be an object");
		level._condition = parseLevelCondition(conditionValue->as_object());
	}
	for (const char* key : { "adaptiveLeafCapacity", "leafCapacityAdaptation", "adaptiveCapacity" })
	{
		if (const boost::json::value* adaptiveValue = object.if_contains(key))
		{
			if (!adaptiveValue->is_object())
				throw std::runtime_error("adaptiveLeafCapacity must be an object");
			level._adaptiveLeafCapacity = parseAdaptiveLeafCapacity(adaptiveValue->as_object());
			break;
		}
	}

	if (level._numLevels == 0)
		throw std::runtime_error("Schema level numLevels must be greater than zero");

	return level;
}

static BuildPolicy parseBuildPolicy(const boost::json::object& object)
{
	BuildPolicy policy;
	policy._maxDepth = asSize(object, "maxDepth", policy._maxDepth);
	policy._leafCapacity = asSize(object, "leafCapacity", policy._leafCapacity);
	policy._minPrimitivesToSplit = asSize(object, "minPointsToSplit", policy._minPrimitivesToSplit);
	policy._minPrimitivesToSplit = asSize(object, "minPrimitivesToSplit", policy._minPrimitivesToSplit);
	policy._collapseSingleChild = asBool(object, "collapseSingleChild", policy._collapseSingleChild);
	policy._removeEmptyNodes = asBool(object, "removeEmptyNodes", policy._removeEmptyNodes);
	policy._allowOverlapDuplication = asBool(object, "allowOverlapDuplication", policy._allowOverlapDuplication);
	policy._enableLeafMicroIndexes = asBool(object, "enableLeafMicroIndexes", policy._enableLeafMicroIndexes);
	policy._enableLeafMicroIndexes = asBool(object, "leafMicroIndexes", policy._enableLeafMicroIndexes);
	policy._leafMicroIndexThreshold = asSize(object, "leafMicroIndexThreshold", policy._leafMicroIndexThreshold);
	policy._leafMicroIndexThreshold = asSize(object, "microIndexThreshold", policy._leafMicroIndexThreshold);
	return policy;
}

bool SchemaLevelCondition::empty() const
{
	return !_minPoints && !_maxPoints &&
		!_minDensity && !_maxDensity &&
		!_minHeightRatio && !_maxHeightRatio &&
		!_minExtentX && !_maxExtentX &&
		!_minExtentY && !_maxExtentY &&
		!_minExtentZ && !_maxExtentZ &&
		!_minAnisotropy && !_maxAnisotropy &&
		!_minOccupancyEntropy && !_maxOccupancyEntropy;
}

size_t SchemaConfig::totalLevels() const
{
	size_t total = 0;
	for (const SchemaLevelConfig& level : _levels)
		total += level._numLevels;

	return total;
}

std::vector<MultiDataStructure::LevelConfig> SchemaConfig::toLevelConfigs() const
{
	std::vector<MultiDataStructure::LevelConfig> result;
	result.reserve(_levels.size());

	for (const SchemaLevelConfig& level : _levels)
	{
		MultiDataStructure::LevelConfig converted;
		converted._levelType = level._cpuFallbackType;
		converted._numLevels = static_cast<glm::uint>(level._numLevels);
		converted._leafCapacity = level._leafCapacity;
		converted._minPrimitivesToSplit = level._minPrimitivesToSplit;
		result.push_back(converted);
	}

	return result;
}

const SchemaLevelConfig& SchemaConfig::levelForDepth(size_t depth) const
{
	if (_levels.empty())
		throw std::runtime_error("Schema has no levels");

	size_t cumulative = 0;
	for (const SchemaLevelConfig& level : _levels)
	{
		cumulative += level._numLevels;
		if (depth < cumulative)
			return level;
	}

	return _levels.back();
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
	schema._name = asString(root, "name", sourceName);

	if (const boost::json::value* policyValue = root.if_contains("buildPolicy"))
	{
		if (!policyValue->is_object())
			throw std::runtime_error("buildPolicy must be an object");
		schema._buildPolicy = parseBuildPolicy(policyValue->as_object());
	}

	const boost::json::value* levelsValue = root.if_contains("levels");
	if (!levelsValue || !levelsValue->is_array())
		throw std::runtime_error("Schema config requires a levels array");

	for (const boost::json::value& levelValue : levelsValue->as_array())
	{
		if (!levelValue.is_object())
			throw std::runtime_error("Each schema level must be an object");
		schema._levels.push_back(parseLevel(levelValue.as_object()));
	}

	if (schema._levels.empty())
		throw std::runtime_error("Schema config requires at least one level");

	if (schema._buildPolicy._maxDepth == 0)
		schema._buildPolicy._maxDepth = schema.totalLevels();

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

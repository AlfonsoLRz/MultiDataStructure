#include "../../stdafx.h"
#include "PointCloud.h"

static constexpr char CACHE_MAGIC[8] = { 'M', 'D', 'S', 'P', 'C', '0', '1', '\0' };
static constexpr uint32_t CACHE_VERSION = 3;
static constexpr uint32_t POSITION_ONLY_CACHE_VERSION = 3;
static constexpr uint32_t LEGACY_POSITION_ONLY_CACHE_VERSION = 2;
static constexpr uint32_t LEGACY_FULL_POINT_CACHE_VERSION = 1;

struct BinaryHeaderPrefix
{
	char magic[8] = {};
	uint32_t	_version = CACHE_VERSION;
	uint64_t	_sourceSize = 0;
	int64_t		_sourceWriteTime = 0;
	uint64_t	_numPoints = 0;
};

struct BinaryHeaderMetadata
{
	double origin[3] = { 0.0, 0.0, 0.0 };
	double	_scale[3] = { 1.0, 1.0, 1.0 };
};

struct BinaryHeader
{
	BinaryHeaderPrefix		_prefix;
	BinaryHeaderMetadata	_metadata;
};

struct BinaryPoint
{
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
};

struct LegacyBinaryPoint
{
	float x = 0.0f;
	float y = 0.0f;
	float z = 0.0f;
	float		_intensity = 0.0f;
	uint32_t	_classification = 0;
	uint64_t	_id = 0;
};

static_assert(sizeof(BinaryPoint) == sizeof(PointPrimitive), "Binary point cache must match the position-only point payload.");

static std::string trim(const std::string& value)
{
	const auto begin = std::find_if_not(value.begin(), value.end(), [](unsigned char c) { return std::isspace(c); });
	const auto end = std::find_if_not(value.rbegin(), value.rend(), [](unsigned char c) { return std::isspace(c); }).base();
	if (begin >= end)
		return {};

	return std::string(begin, end);
}

static std::string lowerExtension(const std::string& filename)
{
	std::filesystem::path path(filename);
	std::string extension = path.extension().string();
	std::transform(extension.begin(), extension.end(), extension.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return extension;
}

static std::string lowerString(std::string value)
{
	std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
	return value;
}

static bool tryParseFloat(const std::string& token, float& value)
{
	try
	{
		const std::string trimmed = trim(token);
		size_t parsed = 0;
		value = std::stof(trimmed, &parsed);
		return parsed == trimmed.size();
	}
	catch (...)
	{
		return false;
	}
}

static bool tryParseSize(const std::string& token, size_t& value)
{
	try
	{
		const std::string trimmed = trim(token);
		size_t parsed = 0;
		const unsigned long long parsedValue = std::stoull(trimmed, &parsed);
		if (parsed != trimmed.size())
			return false;

		value = static_cast<size_t>(parsedValue);
		return true;
	}
	catch (...)
	{
		return false;
	}
}

static std::vector<std::string> splitCSV(const std::string& line)
{
	std::vector<std::string> tokens;
	std::stringstream stream(line);
	std::string token;
	while (std::getline(stream, token, ','))
		tokens.push_back(trim(token));

	return tokens;
}

static std::vector<std::string> splitWhitespace(const std::string& line)
{
	std::vector<std::string> tokens;
	std::stringstream stream(line);
	std::string token;
	while (stream >> token)
		tokens.push_back(token);

	return tokens;
}

static int findPropertyIndex(const std::vector<std::string>& names, const std::vector<std::string>& candidates)
{
	for (size_t index = 0; index < names.size(); ++index)
	{
		const std::string name = lowerString(names[index]);
		if (std::find(candidates.begin(), candidates.end(), name) != candidates.end())
			return static_cast<int>(index);
	}

	return -1;
}

template <typename T>
static T readScalarAt(std::ifstream& file, std::streamoff offset, const std::string& label)
{
	T value{};
	file.seekg(offset, std::ios::beg);
	file.read(reinterpret_cast<char*>(&value), sizeof(T));
	if (!file)
		throw std::runtime_error("Invalid LAS header while reading " + label);

	return value;
}

template <typename T>
static T readScalarFromRecord(const std::vector<char>& record, size_t offset, const std::string& label)
{
	if (offset + sizeof(T) > record.size())
		throw std::runtime_error("Invalid LAS point record while reading " + label);

	T value{};
	std::memcpy(&value, record.data() + offset, sizeof(T));
	return value;
}

static PointPrimitive makePoint(float x, float y, float z, float = 0.0f, uint32_t = 0, uint64_t = 0)
{
	PointPrimitive point;
	point.position = glm::vec3(x, y, z);
	return point;
}

static uint64_t fileSize(const std::string& filename)
{
	return static_cast<uint64_t>(std::filesystem::file_size(filename));
}

static int64_t fileWriteTime(const std::string& filename)
{
	return static_cast<int64_t>(std::filesystem::last_write_time(filename).time_since_epoch().count());
}

static BinaryHeader makeHeader(const std::string& sourcePath, size_t numPoints, const PointCloud::CoordinateFrame& frame)
{
	BinaryHeader header;
	std::memcpy(header._prefix.magic, CACHE_MAGIC, sizeof(header._prefix.magic));
	header._prefix._version = CACHE_VERSION;
	header._prefix._sourceSize = fileSize(sourcePath);
	header._prefix._sourceWriteTime = fileWriteTime(sourcePath);
	header._prefix._numPoints = static_cast<uint64_t>(numPoints);
	header._metadata.origin[0] = frame.origin.x;
	header._metadata.origin[1] = frame.origin.y;
	header._metadata.origin[2] = frame.origin.z;
	header._metadata._scale[0] = frame._scale.x;
	header._metadata._scale[1] = frame._scale.y;
	header._metadata._scale[2] = frame._scale.z;
	return header;
}

static bool headerMatchesSource(const BinaryHeaderPrefix& header, const std::string& sourcePath)
{
	return std::memcmp(header.magic, CACHE_MAGIC, sizeof(header.magic)) == 0 &&
		(header._version == POSITION_ONLY_CACHE_VERSION ||
		 header._version == LEGACY_POSITION_ONLY_CACHE_VERSION ||
		 header._version == LEGACY_FULL_POINT_CACHE_VERSION) &&
		header._sourceSize == fileSize(sourcePath) &&
		header._sourceWriteTime == fileWriteTime(sourcePath);
}

static PointCloud::CoordinateFrame frameFromMetadata(const BinaryHeaderMetadata& metadata)
{
	PointCloud::CoordinateFrame frame;
	frame.origin = glm::dvec3(metadata.origin[0], metadata.origin[1], metadata.origin[2]);
	frame._scale = glm::dvec3(metadata._scale[0], metadata._scale[1], metadata._scale[2]);
	return frame;
}

static BinaryPoint toBinaryPoint(const PointPrimitive& point)
{
	BinaryPoint binaryPoint;
	binaryPoint.x = point.position.x;
	binaryPoint.y = point.position.y;
	binaryPoint.z = point.position.z;
	return binaryPoint;
}

static PointPrimitive fromBinaryPoint(const BinaryPoint& binaryPoint)
{
	return makePoint(binaryPoint.x, binaryPoint.y, binaryPoint.z);
}

static PointPrimitive fromLegacyBinaryPoint(const LegacyBinaryPoint& binaryPoint)
{
	return makePoint(binaryPoint.x, binaryPoint.y, binaryPoint.z);
}

PointCloud PointCloud::load(const std::string& filename, const LoadOptions& options)
{
	const std::filesystem::path cachePath = cachePathFor(filename);
	if (options._useBinaryCache && !options._rebuildBinaryCache)
	{
		PointCloud cachedCloud;
		if (cachedCloud.tryLoadBinaryCache(filename, cachePath))
			return cachedCloud;
	}

	PointCloud cloud = loadFromSource(filename);
	cloud._sourcePath = filename;
	cloud._cachePath = cachePath.string();
	cloud._loadedFromCache = false;
	cloud.recomputeStats();

	if (options._useBinaryCache)
	{
		try
		{
			cloud.saveBinaryCache(filename, cachePath);
		}
		catch (const std::exception& exception)
		{
			std::cerr << "Warning: unable to write point-cloud cache: " << exception.what() << '\n';
		}
	}

	return cloud;
}

std::filesystem::path PointCloud::cachePathFor(const std::string& filename)
{
	std::filesystem::path cachePath(filename);
	cachePath += ".mdspc";
	return cachePath;
}

size_t PointCloud::size() const
{
	return _points.size();
}

bool PointCloud::empty() const
{
	return _points.empty();
}

const std::vector<PointPrimitive>& PointCloud::points() const
{
	return _points;
}

const PointCloud::CoordinateFrame& PointCloud::coordinateFrame() const
{
	return _coordinateFrame;
}

glm::dvec3 PointCloud::toWorldPosition(const glm::vec3& localPosition) const
{
	return _coordinateFrame.origin + glm::dvec3(localPosition) * _coordinateFrame._scale;
}

glm::vec3 PointCloud::toLocalPosition(const glm::dvec3& worldPosition) const
{
	const glm::dvec3 local = (worldPosition - _coordinateFrame.origin) / _coordinateFrame._scale;
	return glm::vec3(local);
}

AABB PointCloud::bounds() const
{
	return _stats._bounds;
}

glm::vec3 PointCloud::coordinateRange() const
{
	return _stats.coordinateRange;
}

float PointCloud::approximateDensity() const
{
	return _stats._approximateDensity;
}

bool PointCloud::loadedFromCache() const
{
	return _loadedFromCache;
}

const std::string& PointCloud::sourcePath() const
{
	return _sourcePath;
}

const std::string& PointCloud::cachePath() const
{
	return _cachePath;
}

void PointCloud::reserve(size_t count)
{
	_points.reserve(count);
}

void PointCloud::addPoint(const PointPrimitive& point)
{
	_points.push_back(point);
	updateStatsForPoint(point);
}

void PointCloud::clear()
{
	_points.clear();
	_coordinateFrame = CoordinateFrame();
	_stats = Stats();
	_sourcePath.clear();
	_cachePath.clear();
	_loadedFromCache = false;
}

void PointCloud::recomputeStats()
{
	_stats = Stats();
	for (const PointPrimitive& point : _points)
		updateStatsForPoint(point);
}

void PointCloud::updateStatsForPoint(const PointPrimitive& point)
{
	_stats._numPoints = _points.size();
	_stats._bounds.update(point.position);
	_stats.coordinateRange = _stats._bounds.size();

	const float volume = _stats.coordinateRange.x * _stats.coordinateRange.y * _stats.coordinateRange.z;
	_stats._approximateDensity = 0.0f;
	if (volume > std::numeric_limits<float>::epsilon())
		_stats._approximateDensity = static_cast<float>(_stats._numPoints) / volume;
}

bool PointCloud::tryLoadBinaryCache(const std::string& sourcePath, const std::filesystem::path& cachePath)
{
	if (!std::filesystem::exists(cachePath))
		return false;

	std::ifstream file(cachePath, std::ios::binary);
	if (!file.is_open())
		return false;

	BinaryHeaderPrefix header;
	file.read(reinterpret_cast<char*>(&header), sizeof(header));
	if (!file || !headerMatchesSource(header, sourcePath))
		return false;

	if (header._numPoints > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
		return false;

	_coordinateFrame = CoordinateFrame();
	if (header._version == POSITION_ONLY_CACHE_VERSION)
	{
		BinaryHeaderMetadata metadata;
		file.read(reinterpret_cast<char*>(&metadata), sizeof(metadata));
		if (!file)
			return false;
		_coordinateFrame = frameFromMetadata(metadata);
	}

	_points.clear();
	const size_t pointCount = static_cast<size_t>(header._numPoints);
	_points.reserve(pointCount);
	if (header._version == POSITION_ONLY_CACHE_VERSION)
	{
		_points.resize(pointCount);
		if (!_points.empty())
			file.read(reinterpret_cast<char*>(_points.data()), static_cast<std::streamsize>(_points.size() * sizeof(BinaryPoint)));
		if (!file)
			return false;
	}
	else if (header._version == LEGACY_POSITION_ONLY_CACHE_VERSION)
	{
		_points.resize(pointCount);
		if (!_points.empty())
			file.read(reinterpret_cast<char*>(_points.data()), static_cast<std::streamsize>(_points.size() * sizeof(BinaryPoint)));
		if (!file)
			return false;
	}
	else if (header._version == LEGACY_FULL_POINT_CACHE_VERSION)
	{
		constexpr size_t ChunkPoints = 1 << 20;
		std::vector<LegacyBinaryPoint> chunk;
		chunk.resize(std::min(pointCount, ChunkPoints));
		size_t remaining = pointCount;
		while (remaining > 0)
		{
			const size_t current = std::min(remaining, chunk.size());
			file.read(reinterpret_cast<char*>(chunk.data()), static_cast<std::streamsize>(current * sizeof(LegacyBinaryPoint)));
			if (!file)
				return false;

			for (size_t i = 0; i < current; ++i)
				_points.push_back(fromLegacyBinaryPoint(chunk[i]));
			remaining -= current;
		}
	}
	else
	{
		return false;
	}

	_sourcePath = sourcePath;
	_cachePath = cachePath.string();
	_loadedFromCache = true;
	recomputeStats();
	return true;
}

void PointCloud::saveBinaryCache(const std::string& sourcePath, const std::filesystem::path& cachePath) const
{
	const std::filesystem::path parent = cachePath.parent_path();
	if (!parent.empty())
		std::filesystem::create_directories(parent);

	std::filesystem::path tempPath = cachePath;
	tempPath += ".tmp";

	std::ofstream file(tempPath, std::ios::binary | std::ios::trunc);
	if (!file.is_open())
		throw std::runtime_error("unable to open " + tempPath.string());

	const BinaryHeader header = makeHeader(sourcePath, _points.size(), _coordinateFrame);
	file.write(reinterpret_cast<const char*>(&header), sizeof(header));

	if (!_points.empty())
		file.write(reinterpret_cast<const char*>(_points.data()), static_cast<std::streamsize>(_points.size() * sizeof(BinaryPoint)));

	if (!file)
		throw std::runtime_error("failed while writing " + tempPath.string());

	file.close();
	if (!file)
		throw std::runtime_error("failed while closing " + tempPath.string());

	std::filesystem::remove(cachePath);
	std::filesystem::rename(tempPath, cachePath);
}

PointCloud PointCloud::loadFromSource(const std::string& filename)
{
	const std::string extension = lowerExtension(filename);
	if (extension == ".xyz")
		return loadXYZ(filename);
	if (extension == ".csv")
		return loadCSV(filename);
	if (extension == ".ply")
		return loadPLY(filename);
	if (extension == ".las")
		return loadLAS(filename);
	if (extension == ".laz")
		throw std::runtime_error("LAZ is compressed and is not supported by the built-in reader yet; convert to LAS first");

	throw std::runtime_error("Unsupported point-cloud extension: " + extension);
}

PointCloud PointCloud::loadXYZ(const std::string& filename)
{
	std::ifstream file(filename);
	if (!file.is_open())
		throw std::runtime_error("Unable to open XYZ point cloud: " + filename);

	PointCloud cloud;
	std::string line;
	size_t lineNumber = 0;

	while (std::getline(file, line))
	{
		++lineNumber;
		line = trim(line);
		if (line.empty() || line[0] == '#')
			continue;

		std::stringstream stream(line);
		float x = 0.0f, y = 0.0f, z = 0.0f, intensity = 0.0f;
		uint32_t classification = 0;
		if (!(stream >> x >> y >> z))
			throw std::runtime_error("Invalid XYZ point at line " + std::to_string(lineNumber));

		stream >> intensity;
		stream >> classification;
		cloud.addPoint(makePoint(x, y, z, intensity, classification, static_cast<uint64_t>(cloud.size())));
	}

	return cloud;
}

PointCloud PointCloud::loadCSV(const std::string& filename)
{
	std::ifstream file(filename);
	if (!file.is_open())
		throw std::runtime_error("Unable to open CSV point cloud: " + filename);

	PointCloud cloud;
	std::string line;
	size_t lineNumber = 0;
	bool skippedHeader = false;

	while (std::getline(file, line))
	{
		++lineNumber;
		line = trim(line);
		if (line.empty() || line[0] == '#')
			continue;

		const std::vector<std::string> tokens = splitCSV(line);
		if (tokens.size() < 3)
			throw std::runtime_error("CSV point cloud line has fewer than 3 columns at line " + std::to_string(lineNumber));

		float x = 0.0f, y = 0.0f, z = 0.0f;
		if (!tryParseFloat(tokens[0], x) || !tryParseFloat(tokens[1], y) || !tryParseFloat(tokens[2], z))
		{
			if (!skippedHeader && cloud.empty())
			{
				skippedHeader = true;
				continue;
			}

			throw std::runtime_error("Invalid CSV point at line " + std::to_string(lineNumber));
		}

		float intensity = 0.0f;
		if (tokens.size() >= 4)
			tryParseFloat(tokens[3], intensity);

		uint32_t classification = 0;
		if (tokens.size() >= 5)
		{
			float parsedClassification = 0.0f;
			if (tryParseFloat(tokens[4], parsedClassification))
				classification = static_cast<uint32_t>(parsedClassification);
		}

		cloud.addPoint(makePoint(x, y, z, intensity, classification, static_cast<uint64_t>(cloud.size())));
	}

	return cloud;
}

PointCloud PointCloud::loadPLY(const std::string& filename)
{
	std::ifstream file(filename);
	if (!file.is_open())
		throw std::runtime_error("Unable to open PLY point cloud: " + filename);

	std::string line;
	if (!std::getline(file, line) || trim(line) != "ply")
		throw std::runtime_error("Invalid PLY header: missing ply magic");

	bool asciiFormat = false;
	bool foundEndHeader = false;
	bool readingVertexElement = false;
	size_t vertexCount = 0;
	std::vector<std::string> vertexProperties;

	while (std::getline(file, line))
	{
		line = trim(line);
		if (line.empty())
			continue;

		std::stringstream stream(line);
		std::string keyword;
		stream >> keyword;

		if (keyword == "format")
		{
			std::string format;
			stream >> format;
			if (format != "ascii")
				throw std::runtime_error("Only ASCII PLY point clouds are supported by the built-in reader");
			asciiFormat = true;
		}
		else if (keyword == "element")
		{
			std::string elementName;
			std::string countToken;
			stream >> elementName >> countToken;
			readingVertexElement = elementName == "vertex";
			if (readingVertexElement && !tryParseSize(countToken, vertexCount))
				throw std::runtime_error("Invalid PLY vertex count");
		}
		else if (keyword == "property" && readingVertexElement)
		{
			std::string propertyType;
			stream >> propertyType;
			if (propertyType == "list")
				continue;

			std::string propertyName;
			stream >> propertyName;
			if (!propertyName.empty())
				vertexProperties.push_back(propertyName);
		}
		else if (keyword == "end_header")
		{
			foundEndHeader = true;
			break;
		}
	}

	if (!asciiFormat)
		throw std::runtime_error("PLY file does not declare format ascii");
	if (!foundEndHeader)
		throw std::runtime_error("Invalid PLY header: missing end_header");

	const int xIndex = findPropertyIndex(vertexProperties, { "x" });
	const int yIndex = findPropertyIndex(vertexProperties, { "y" });
	const int zIndex = findPropertyIndex(vertexProperties, { "z" });
	if (xIndex < 0 || yIndex < 0 || zIndex < 0)
		throw std::runtime_error("PLY vertex element must include x, y, and z properties");

	const int intensityIndex = findPropertyIndex(vertexProperties, { "intensity", "scalar_intensity" });
	const int classificationIndex = findPropertyIndex(vertexProperties, { "classification", "class", "label" });

	PointCloud cloud;
	cloud.reserve(vertexCount);
	for (size_t vertexIndex = 0; vertexIndex < vertexCount; ++vertexIndex)
	{
		if (!std::getline(file, line))
			throw std::runtime_error("PLY ended before all vertices were read");

		const std::vector<std::string> tokens = splitWhitespace(line);
		if (tokens.size() < vertexProperties.size())
			throw std::runtime_error("PLY vertex row has fewer values than declared properties");

		float x = 0.0f, y = 0.0f, z = 0.0f;
		if (!tryParseFloat(tokens[xIndex], x) || !tryParseFloat(tokens[yIndex], y) || !tryParseFloat(tokens[zIndex], z))
			throw std::runtime_error("Invalid PLY vertex position");

		float intensity = 0.0f;
		if (intensityIndex >= 0)
			tryParseFloat(tokens[intensityIndex], intensity);

		uint32_t classification = 0;
		if (classificationIndex >= 0)
		{
			float parsedClassification = 0.0f;
			if (tryParseFloat(tokens[classificationIndex], parsedClassification))
				classification = static_cast<uint32_t>(parsedClassification);
		}

		cloud.addPoint(makePoint(x, y, z, intensity, classification, static_cast<uint64_t>(cloud.size())));
	}

	return cloud;
}

PointCloud PointCloud::loadLAS(const std::string& filename)
{
	std::ifstream file(filename, std::ios::binary);
	if (!file.is_open())
		throw std::runtime_error("Unable to open LAS point cloud: " + filename);

	char signature[4] = {};
	file.read(signature, sizeof(signature));
	if (!file || std::string(signature, signature + 4) != "LASF")
		throw std::runtime_error("Invalid LAS header: missing LASF signature");

	const uint16_t headerSize = readScalarAt<uint16_t>(file, 94, "header size");
	const uint32_t pointDataOffset = readScalarAt<uint32_t>(file, 96, "point data offset");
	const uint8_t pointFormatRaw = readScalarAt<uint8_t>(file, 104, "point format");
	const uint16_t pointRecordLength = readScalarAt<uint16_t>(file, 105, "point record length");
	const uint32_t legacyPointCount = readScalarAt<uint32_t>(file, 107, "legacy point count");
	const double xScale = readScalarAt<double>(file, 131, "x scale");
	const double yScale = readScalarAt<double>(file, 139, "y scale");
	const double zScale = readScalarAt<double>(file, 147, "z scale");
	const double xOffset = readScalarAt<double>(file, 155, "x offset");
	const double yOffset = readScalarAt<double>(file, 163, "y offset");
	const double zOffset = readScalarAt<double>(file, 171, "z offset");
	const double maxX = readScalarAt<double>(file, 179, "max x");
	const double minX = readScalarAt<double>(file, 187, "min x");
	const double maxY = readScalarAt<double>(file, 195, "max y");
	const double minY = readScalarAt<double>(file, 203, "min y");
	const double maxZ = readScalarAt<double>(file, 211, "max z");
	const double minZ = readScalarAt<double>(file, 219, "min z");

	if ((pointFormatRaw & 0x80) != 0)
		throw std::runtime_error("Compressed LAS/LAZ point records are not supported by the built-in reader");

	const uint8_t pointFormat = pointFormatRaw & 0x3f;
	if (pointFormat > 10)
		throw std::runtime_error("Unsupported LAS point data record format: " + std::to_string(pointFormat));

	uint64_t pointCount = legacyPointCount;
	if (pointCount == 0 && headerSize >= 375)
		pointCount = readScalarAt<uint64_t>(file, 247, "extended point count");

	const size_t classificationOffset = pointFormat <= 5 ? 15 : 16;
	if (pointRecordLength < classificationOffset + 1)
		throw std::runtime_error("LAS point record length is too short for the declared format");

	if (pointCount > static_cast<uint64_t>(std::numeric_limits<size_t>::max()))
		throw std::runtime_error("LAS point count is too large for this process");

	PointCloud cloud;
	cloud.reserve(static_cast<size_t>(pointCount));
	cloud._coordinateFrame._scale = glm::dvec3(1.0);

	bool originInitialized = false;
	const bool headerBoundsLookValid =
		std::isfinite(minX) && std::isfinite(minY) && std::isfinite(minZ) &&
		std::isfinite(maxX) && std::isfinite(maxY) && std::isfinite(maxZ) &&
		maxX >= minX && maxY >= minY && maxZ >= minZ &&
		(maxX > minX || maxY > minY || maxZ > minZ);
	if (headerBoundsLookValid)
	{
		cloud._coordinateFrame.origin = glm::dvec3(minX, minY, minZ);
		originInitialized = true;
	}

	file.seekg(pointDataOffset, std::ios::beg);
	std::vector<char> record(pointRecordLength);
	for (uint64_t pointIndex = 0; pointIndex < pointCount; ++pointIndex)
	{
		file.read(record.data(), record.size());
		if (!file)
			throw std::runtime_error("LAS ended before all point records were read");

		const int32_t rawX = readScalarFromRecord<int32_t>(record, 0, "x");
		const int32_t rawY = readScalarFromRecord<int32_t>(record, 4, "y");
		const int32_t rawZ = readScalarFromRecord<int32_t>(record, 8, "z");
		const uint16_t rawIntensity = readScalarFromRecord<uint16_t>(record, 12, "intensity");
		const uint8_t rawClassification = static_cast<uint8_t>(record[classificationOffset]);
		const uint32_t classification = pointFormat <= 5 ? (rawClassification & 0x1f) : rawClassification;

		const double x = static_cast<double>(rawX) * xScale + xOffset;
		const double y = static_cast<double>(rawY) * yScale + yOffset;
		const double z = static_cast<double>(rawZ) * zScale + zOffset;
		if (!originInitialized)
		{
			cloud._coordinateFrame.origin = glm::dvec3(x, y, z);
			originInitialized = true;
		}
		const glm::vec3 localPosition = cloud.toLocalPosition(glm::dvec3(x, y, z));
		cloud.addPoint(makePoint(
			localPosition.x,
			localPosition.y,
			localPosition.z,
			static_cast<float>(rawIntensity),
			classification,
			static_cast<uint64_t>(cloud.size())));
	}

	return cloud;
}

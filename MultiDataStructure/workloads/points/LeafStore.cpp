#include "../../stdafx.h"
#include "LeafStore.h"

#if defined(__linux__)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cerrno>
#endif

namespace
{
	constexpr char StoreMagic[8] = { 'M', 'D', 'S', 'L', 'S', '0', '1', '\0' };
	constexpr uint32_t StoreVersion = 1;

	struct StoreHeader
	{
		char		_magic[8] = {};
		uint32_t	_version = 0;
		uint32_t	_bytesPerPoint = 0;
		uint64_t	_digest = 0;
		uint64_t	_nodeCount = 0;
		uint64_t	_pointCount = 0;
		uint64_t	_hierarchyOffset = 0;
		uint64_t	_pointsOffset = 0;
		uint64_t	_fileBytes = 0;
	};

	uint64_t alignUp(uint64_t value)
	{
		return (value + LeafStore::PageBytes - 1) & ~static_cast<uint64_t>(LeafStore::PageBytes - 1);
	}

	StoreHeader headerFor(const PointSpatialIndex::PoolView& pool, uint64_t digest)
	{
		StoreHeader header;
		std::memcpy(header._magic, StoreMagic, sizeof(StoreMagic));
		header._version = StoreVersion;
		header._bytesPerPoint = static_cast<uint32_t>(LeafStore::BytesPerPoint);
		header._digest = digest;
		header._nodeCount = pool._nodeCount;
		header._pointCount = pool._pointCount;
		header._hierarchyOffset = LeafStore::PageBytes;
		header._pointsOffset = alignUp(header._hierarchyOffset + pool._nodeCount * sizeof(PointSpatialIndex::HotNode));
		header._fileBytes = alignUp(header._pointsOffset + pool._pointCount * LeafStore::BytesPerPoint);
		return header;
	}

	// Non-empty leaves in tree order. They must tile [0, pointCount): the file has no leaf table and
	// places each leaf by its first tree position.
	std::vector<std::pair<uint32_t, uint32_t>> leavesInTreeOrder(const PointSpatialIndex::PoolView& pool)
	{
		std::vector<std::pair<uint32_t, uint32_t>> leaves;
		for (size_t slot = 0; slot < pool._nodeCount; ++slot)
		{
			const PointSpatialIndex::HotNode& node = pool._nodes[slot];
			if (!node.isInternal() && node.count() > 0)
				leaves.emplace_back(node._first, node.count());
		}
		std::sort(leaves.begin(), leaves.end());
		uint64_t next = 0;
		for (const auto& [first, count] : leaves)
		{
			if (first != next)
				throw std::runtime_error("leaf store: leaves do not tile the tree order");
			next += count;
		}
		if (next != pool._pointCount)
			throw std::runtime_error("leaf store: leaves do not cover every point");
		return leaves;
	}

	bool readHeader(const std::string& path, StoreHeader& header)
	{
		std::ifstream in(path, std::ios::binary);
		if (!in)
			return false;
		in.read(reinterpret_cast<char*>(&header), sizeof(header));
		return static_cast<bool>(in) && std::memcmp(header._magic, StoreMagic, sizeof(StoreMagic)) == 0;
	}
}

void LeafStore::write(const std::string& path, const PointSpatialIndex::PoolView& pool, uint64_t digest)
{
	const StoreHeader header = headerFor(pool, digest);
	const std::vector<std::pair<uint32_t, uint32_t>> leaves = leavesInTreeOrder(pool);

	std::filesystem::create_directories(std::filesystem::absolute(path).parent_path());
	const std::string temporary = path + ".tmp";
	{
		std::vector<char> streamBuffer(4u << 20);
		std::ofstream out;
		out.rdbuf()->pubsetbuf(streamBuffer.data(), static_cast<std::streamsize>(streamBuffer.size()));
		out.open(temporary, std::ios::binary | std::ios::trunc);
		if (!out)
			throw std::runtime_error("leaf store: cannot write " + temporary);

		const std::vector<char> zeros(PageBytes, 0);
		uint64_t written = 0;
		const auto put = [&](const void* data, uint64_t bytes) {
			out.write(static_cast<const char*>(data), static_cast<std::streamsize>(bytes));
			written += bytes;
		};
		const auto padTo = [&](uint64_t offset) {
			while (written < offset)
				put(zeros.data(), std::min<uint64_t>(offset - written, zeros.size()));
		};

		put(&header, sizeof(header));
		padTo(header._hierarchyOffset);
		put(pool._nodes, pool._nodeCount * sizeof(PointSpatialIndex::HotNode));
		padTo(header._pointsOffset);
		for (const auto& [first, count] : leaves)
		{
			put(pool._x + first, count * sizeof(float));
			put(pool._y + first, count * sizeof(float));
			put(pool._z + first, count * sizeof(float));
			put(pool._order + first, count * sizeof(uint32_t));
		}
		padTo(header._fileBytes);
		out.close();
		if (!out)
			throw std::runtime_error("leaf store: failed writing " + temporary);
	}

#if defined(__linux__)
	// On disk before it is renamed into place, and out of the page cache, so the first timed read
	// goes to the device like every later one.
	const int fd = ::open(temporary.c_str(), O_RDONLY);
	if (fd >= 0)
	{
		::fsync(fd);
		::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
		::close(fd);
	}
#endif
	std::filesystem::rename(temporary, path);
}

bool LeafStore::matches(const std::string& path, const PointSpatialIndex::PoolView& pool, uint64_t digest)
{
	StoreHeader stored;
	if (!readHeader(path, stored))
		return false;
	const StoreHeader expected = headerFor(pool, digest);
	std::error_code error;
	const uint64_t size = std::filesystem::file_size(path, error);
	return !error && size == expected._fileBytes && stored._version == expected._version && stored._digest == expected._digest &&
		   stored._nodeCount == expected._nodeCount && stored._pointCount == expected._pointCount &&
		   stored._pointsOffset == expected._pointsOffset;
}

uint64_t LeafStore::deviceSectorsRead(const std::string& device)
{
#if defined(__linux__)
	if (device.empty())
		return 0;
	std::ifstream stats("/proc/diskstats");
	std::string line;
	while (std::getline(stats, line))
	{
		// major minor name reads merged sectors_read ...
		std::istringstream fields(line);
		unsigned major = 0, minor = 0;
		std::string name;
		uint64_t reads = 0, merged = 0, sectors = 0;
		if (fields >> major >> minor >> name >> reads >> merged >> sectors && name == device)
			return sectors;
	}
#else
	(void)device;
#endif
	return 0;
}

LeafStore::LeafStore(const std::string& path, const PointSpatialIndex::PoolView& pool, uint64_t digest, const Options& options)
	: _path(path), _options(options)
{
	StoreHeader header;
	if (!readHeader(path, header) || !matches(path, pool, digest))
		throw std::runtime_error("leaf store: " + path + " is not the store of this tree");
	_fileBytes = header._fileBytes;
	_hierarchyOffset = header._hierarchyOffset;
	_pointsOffset = header._pointsOffset;

	const std::vector<std::pair<uint32_t, uint32_t>> leaves = leavesInTreeOrder(pool);
	_leafFirst.reserve(leaves.size());
	_leafCount.reserve(leaves.size());
	for (const auto& [first, count] : leaves)
	{
		_leafFirst.push_back(first);
		_leafCount.push_back(count);
	}

#if defined(__linux__)
	_fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
	_direct = _fd >= 0;
	if (_fd < 0 && !options._requireDirect)
		_fd = ::open(path.c_str(), O_RDONLY);
	if (_fd < 0)
		throw std::runtime_error("leaf store: cannot open " + path + (options._requireDirect ? " with O_DIRECT" : ""));
#else
	if (options._requireDirect)
		throw std::runtime_error("leaf store: direct reads are Linux only");
	_stream.open(path, std::ios::binary);
	if (!_stream)
		throw std::runtime_error("leaf store: cannot open " + path);
#endif
}

LeafStore::~LeafStore()
{
#if defined(__linux__)
	if (_fd >= 0)
		::close(_fd);
#endif
}

LeafStore::Buffer LeafStore::allocate(size_t bytes)
{
	return Buffer(static_cast<std::byte*>(::operator new[](bytes, std::align_val_t(PageBytes))));
}

void LeafStore::readAt(std::byte* buffer, size_t bytes, uint64_t alignedOffset)
{
	const auto start = std::chrono::steady_clock::now();
#if defined(__linux__)
	size_t done = 0;
	while (done < bytes)
	{
		const ssize_t got = ::pread(_fd, buffer + done, bytes - done, static_cast<off_t>(alignedOffset + done));
		if (got < 0)
		{
			if (errno == EINTR)
				continue;
			throw std::system_error(errno, std::generic_category(), "leaf store: read failed");
		}
		if (got == 0)
			throw std::runtime_error("leaf store: read past the end of " + _path);
		done += static_cast<size_t>(got);
	}
#else
	_stream.seekg(static_cast<std::streamoff>(alignedOffset));
	_stream.read(reinterpret_cast<char*>(buffer), static_cast<std::streamsize>(bytes));
	if (!_stream)
		throw std::runtime_error("leaf store: read failed on " + _path);
#endif
	_counters._ioNs += static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
	++_counters._reads;
	_counters._bytes += bytes;
}

const std::byte* LeafStore::fetch(uint64_t key, size_t bytes, uint64_t alignedOffset)
{
	if (_options._cacheBytes > 0 && bytes <= _options._cacheBytes)
	{
		const auto found = _cached.find(key);
		if (found != _cached.end())
		{
			++_counters._hits;
			_lru.splice(_lru.begin(), _lru, found->second);
			return found->second->_buffer.get();
		}

		Entry entry;
		entry._key = key;
		entry._bytes = bytes;
		entry._buffer = allocate(bytes);
		readAt(entry._buffer.get(), bytes, alignedOffset);
		_lru.push_front(std::move(entry));
		_cached[key] = _lru.begin();
		_cachedBytes += bytes;
		// Never the entry just read: its buffer is what the caller is about to scan.
		while (_cachedBytes > _options._cacheBytes && _lru.size() > 1)
		{
			_cachedBytes -= _lru.back()._bytes;
			_cached.erase(_lru.back()._key);
			_lru.pop_back();
		}
		return _lru.front()._buffer.get();
	}

	if (bytes > _scratchBytes)
	{
		_scratch = allocate(bytes);
		_scratchBytes = bytes;
	}
	readAt(_scratch.get(), bytes, alignedOffset);
	return _scratch.get();
}

LeafStore::LeafView LeafStore::leaf(uint32_t first, uint32_t count)
{
	const uint64_t begin = _pointsOffset + BytesPerPoint * static_cast<uint64_t>(first);
	const uint64_t end = begin + BytesPerPoint * static_cast<uint64_t>(count);
	const uint64_t alignedBegin = begin & ~static_cast<uint64_t>(PageBytes - 1);
	const std::byte* data = fetch(first, static_cast<size_t>(alignUp(end) - alignedBegin), alignedBegin) + (begin - alignedBegin);

	LeafView view;
	view._x = reinterpret_cast<const float*>(data);
	view._y = view._x + count;
	view._z = view._y + count;
	view._order = reinterpret_cast<const uint32_t*>(view._z + count);
	return view;
}

void LeafStore::appendOrder(uint32_t first, uint32_t count, std::vector<size_t>& out)
{
	const uint32_t end = first + count;
	size_t i = static_cast<size_t>(std::lower_bound(_leafFirst.begin(), _leafFirst.end(), first) - _leafFirst.begin());
	for (; i < _leafFirst.size() && _leafFirst[i] < end; ++i)
	{
		const LeafView view = leaf(_leafFirst[i], _leafCount[i]);
		out.insert(out.end(), view._order, view._order + _leafCount[i]);
	}
}

void LeafStore::touchNodePages(uint32_t firstSlot, uint32_t count)
{
	// Hierarchy pages share the cache with leaves; the top bit keeps their keys apart.
	constexpr uint64_t PageKey = 1ull << 63;
	const uint64_t begin = _hierarchyOffset + sizeof(PointSpatialIndex::HotNode) * static_cast<uint64_t>(firstSlot);
	const uint64_t end = begin + sizeof(PointSpatialIndex::HotNode) * static_cast<uint64_t>(std::max<uint32_t>(count, 1));
	for (uint64_t page = begin / PageBytes; page * PageBytes < end; ++page)
		fetch(PageKey | page, PageBytes, page * PageBytes);
}

void LeafStore::clearCache()
{
	_lru.clear();
	_cached.clear();
	_cachedBytes = 0;
}

#pragma once

#include "../../stdafx.h"
#include "PointSpatialIndex.h"

#include <list>

// A built index's pool and tree-ordered points in one file on disk, read back during queries
// (testing.md 2q). It turns the in-memory engine into an out-of-core one without changing a
// traversal decision: the traversals still prune on the in-memory pool, and only where they would
// read a leaf's points (or, with a paged hierarchy, a node's child group) do they ask the store.
//
// File layout, every region 4 KiB aligned:
//   header page | hot pool (32 B nodes, pool order) | points
// The points are the leaves back to back in tree order, each leaf as x[], y[], z[] (float) and the
// cloud index order[] (uint32), 16 B per point. A leaf starting at tree position f therefore sits
// at pointsOffset + 16 f, so no leaf table is stored.
//
// Reads go straight to the device (O_DIRECT on Linux): with 251 GB of RAM on the measuring server,
// the page cache would otherwise hold every store and measure memory again. Each leaf is one read
// of the 4 KiB pages covering it, synchronous, queue depth 1, which is the per-node load of a
// file-per-node store (ISPRS 2023) or of a COPC chunk. An optional LRU cache, sized in bytes,
// holds leaves and hierarchy pages between reads.
class LeafStore
{
public:
	static constexpr size_t PageBytes = 4096;
	static constexpr size_t BytesPerPoint = 16;

	struct Options
	{
		size_t	_cacheBytes = 0;			// 0 = every access is a device read
		bool	_pagedHierarchy = false;	// also read the page of each child group a query tests
		bool	_requireDirect = true;		// fail instead of falling back to buffered reads
	};

	struct Counters
	{
		uint64_t	_reads = 0;
		uint64_t	_bytes = 0;
		uint64_t	_hits = 0;
		uint64_t	_ioNs = 0;
	};

	// A leaf's points, valid until the next read from this store.
	struct LeafView
	{
		const float*	_x = nullptr;
		const float*	_y = nullptr;
		const float*	_z = nullptr;
		const uint32_t*	_order = nullptr;
	};

	// Writes the store through a temporary file and renames it into place, then drops the file
	// from the page cache so the first query does not find it there.
	static void write(const std::string& path, const PointSpatialIndex::PoolView& pool, uint64_t digest);
	// True if `path` holds the store of this tree (same digest, node and point counts).
	static bool matches(const std::string& path, const PointSpatialIndex::PoolView& pool, uint64_t digest);
	// Device sectors read so far, from /proc/diskstats; 0 where unavailable. Lets a run see reads by
	// other processes on the same device during its passes.
	static uint64_t deviceSectorsRead(const std::string& device);

	LeafStore(const std::string& path, const PointSpatialIndex::PoolView& pool, uint64_t digest, const Options& options);
	~LeafStore();
	LeafStore(const LeafStore&) = delete;
	LeafStore& operator=(const LeafStore&) = delete;

	// The leaf whose points are tree positions [first, first + count).
	LeafView leaf(uint32_t first, uint32_t count);
	// Every leaf inside tree positions [first, first + count), one read each, appending their cloud
	// indices: a fully contained subtree of a range query.
	void appendOrder(uint32_t first, uint32_t count, std::vector<size_t>& out);
	// The hierarchy page(s) holding pool slots [firstSlot, firstSlot + count); a no-op unless paged.
	void touchNodes(uint32_t firstSlot, uint32_t count)
	{
		if (_options._pagedHierarchy)
			touchNodePages(firstSlot, count);
	}

	void clearCache();
	void resetCounters() { _counters = {}; }
	const Counters& counters() const { return _counters; }
	uint64_t fileBytes() const { return _fileBytes; }
	bool direct() const { return _direct; }
	const std::string& path() const { return _path; }

private:
	struct AlignedFree
	{
		void operator()(std::byte* pointer) const { ::operator delete[](pointer, std::align_val_t(PageBytes)); }
	};
	using Buffer = std::unique_ptr<std::byte[], AlignedFree>;

	struct Entry
	{
		uint64_t	_key = 0;
		size_t		_bytes = 0;
		Buffer		_buffer;
	};

	static Buffer allocate(size_t bytes);
	// Reads [alignedOffset, alignedOffset + bytes) into `buffer`, counting it.
	void readAt(std::byte* buffer, size_t bytes, uint64_t alignedOffset);
	// The buffer holding [alignedOffset, alignedOffset + bytes), from the cache or the device.
	const std::byte* fetch(uint64_t key, size_t bytes, uint64_t alignedOffset);
	void touchNodePages(uint32_t firstSlot, uint32_t count);

	std::string	_path;
	Options		_options;
	bool		_direct = false;
	uint64_t	_fileBytes = 0;
	uint64_t	_hierarchyOffset = 0;
	uint64_t	_pointsOffset = 0;
	// Leaves in tree order: first tree position and point count, for appendOrder().
	std::vector<uint32_t>	_leafFirst;
	std::vector<uint32_t>	_leafCount;
	Counters	_counters;
	Buffer		_scratch;
	size_t		_scratchBytes = 0;
	std::list<Entry>	_lru;	// most recent first
	std::unordered_map<uint64_t, std::list<Entry>::iterator>	_cached;
	size_t		_cachedBytes = 0;
#if defined(__linux__)
	int			_fd = -1;
#else
	std::ifstream	_stream;
#endif
};

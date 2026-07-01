#include "../stdafx.h"
#include "QueryTrace.h"

namespace
{
	std::vector<std::string> splitCsvLine(const std::string& line)
	{
		std::vector<std::string> cells;
		std::string current;
		for (const char c : line)
		{
			if (c == ',')
			{
				cells.push_back(current);
				current.clear();
				continue;
			}
			if (c == '\r')
				continue;
			current.push_back(c);
		}
		cells.push_back(current);
		return cells;
	}

	std::string lowerCopy(std::string value)
	{
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return value;
	}

	float parseFloatCell(const std::string& value)
	{
		if (value.empty())
			return 0.0f;
		try { return std::stof(value); }
		catch (...) { return 0.0f; }
	}

	size_t parseSizeCell(const std::string& value)
	{
		if (value.empty())
			return 0;
		try
		{
			const long long parsed = std::stoll(value);
			return parsed > 0 ? static_cast<size_t>(parsed) : 0;
		}
		catch (...) { return 0; }
	}
}

std::vector<Experiments::TraceQuery> Experiments::loadQueryTrace(const std::string& path)
{
	std::ifstream input(path);
	if (!input.is_open())
		throw std::runtime_error("Unable to open query trace: " + path);

	std::string line;
	if (!std::getline(input, line))
		throw std::runtime_error("Empty query trace: " + path);

	const std::vector<std::string> header = splitCsvLine(line);
	std::unordered_map<std::string, size_t> columns;
	for (size_t i = 0; i < header.size(); ++i)
		columns[lowerCopy(header[i])] = i;

	const auto cell = [&](const std::vector<std::string>& row, const char* name) -> std::string
	{
		const auto found = columns.find(name);
		if (found == columns.end() || found->second >= row.size())
			return {};
		return row[found->second];
	};

	std::vector<TraceQuery> trace;
	while (std::getline(input, line))
	{
		if (line.empty())
			continue;

		const std::vector<std::string> row = splitCsvLine(line);
		const std::string kind = lowerCopy(cell(row, "query_type"));

		TraceQuery query;
		if (kind == "range" || kind == "aabb_range")
			query._kind = TraceQuery::Kind::Range;
		else if (kind == "count_range")
			query._kind = TraceQuery::Kind::CountRange;
		else if (kind == "radius")
			query._kind = TraceQuery::Kind::Radius;
		else if (kind == "knn")
			query._kind = TraceQuery::Kind::Knn;
		else
			continue;

		query._minBound = glm::vec3(
			parseFloatCell(cell(row, "bounds_min_x")),
			parseFloatCell(cell(row, "bounds_min_y")),
			parseFloatCell(cell(row, "bounds_min_z")));
		query._maxBound = glm::vec3(
			parseFloatCell(cell(row, "bounds_max_x")),
			parseFloatCell(cell(row, "bounds_max_y")),
			parseFloatCell(cell(row, "bounds_max_z")));
		query._center = glm::vec3(
			parseFloatCell(cell(row, "center_x")),
			parseFloatCell(cell(row, "center_y")),
			parseFloatCell(cell(row, "center_z")));
		query._radius = parseFloatCell(cell(row, "radius"));
		query._k = parseSizeCell(cell(row, "k"));
		trace.push_back(query);
	}

	if (trace.empty())
		throw std::runtime_error("Query trace contains no parseable queries: " + path);

	return trace;
}

size_t Experiments::dominantKnnK(const std::vector<TraceQuery>& trace)
{
	std::unordered_map<size_t, size_t> histogram;
	for (const TraceQuery& query : trace)
	{
		if (query._kind == TraceQuery::Kind::Knn && query._k > 0)
			++histogram[query._k];
	}

	size_t bestK = 0, bestCount = 0;
	for (const auto& [k, count] : histogram)
	{
		if (count > bestCount)
		{
			bestK = k;
			bestCount = count;
		}
	}
	return bestK;
}

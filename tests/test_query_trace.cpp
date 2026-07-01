#include "../MultiDataStructure/stdafx.h"
#include "../MultiDataStructure/experiments/QueryTrace.h"
#include "../MultiDataStructure/experiments/SchemaSearch.h"
#include "BaselineTests.h"

namespace BaselineTests
{
	void expect(bool condition, const std::string& message);

	static bool nearlyEqualTrace(float left, float right, float epsilon = 1e-5f)
	{
		return std::abs(left - right) <= epsilon;
	}

	void runQueryTraceTests()
	{
		// --- round-trip: write the --query-trace CSV format, load it back ---
		const std::filesystem::path tracePath = std::filesystem::temp_directory_path() / "mdspc_test_query_trace.csv";
		{
			std::ofstream out(tracePath);
			out << "query_id,query_type,bounds_min_x,bounds_min_y,bounds_min_z,bounds_max_x,bounds_max_y,bounds_max_z,"
				<< "center_x,center_y,center_z,radius,k,returned_points,latency_ms\n";
			out << "0,range,-1,-2,-3,1,2,3,0,0,0,0,0,42,0.5\n";
			out << "1,count_range,0,0,0,4,4,4,2,2,2,0,0,7,0.1\n";
			out << "2,radius,0,0,0,0,0,0,1.5,2.5,3.5,0.75,0,12,0.2\n";
			out << "3,knn,0,0,0,0,0,0,9,8,7,0,16,16,0.3\n";
			out << "4,knn,0,0,0,0,0,0,1,1,1,0,16,16,0.3\n";
			out << "5,unsupported_kind,0,0,0,0,0,0,0,0,0,0,0,0,0\n";
		}

		const std::vector<Experiments::TraceQuery> trace = Experiments::loadQueryTrace(tracePath.string());
		expect(trace.size() == 5, "trace loader keeps the 5 supported queries and skips unknown kinds");

		expect(trace[0]._kind == Experiments::TraceQuery::Kind::Range, "query 0 is a range query");
		expect(nearlyEqualTrace(trace[0]._minBound.x, -1.0f) && nearlyEqualTrace(trace[0]._maxBound.z, 3.0f), "range bounds round-trip");

		expect(trace[1]._kind == Experiments::TraceQuery::Kind::CountRange, "query 1 is a count_range query");
		expect(nearlyEqualTrace(trace[1]._maxBound.x, 4.0f), "count_range bounds round-trip");

		expect(trace[2]._kind == Experiments::TraceQuery::Kind::Radius, "query 2 is a radius query");
		expect(nearlyEqualTrace(trace[2]._center.y, 2.5f) && nearlyEqualTrace(trace[2]._radius, 0.75f), "radius center/radius round-trip");

		expect(trace[3]._kind == Experiments::TraceQuery::Kind::Knn, "query 3 is a knn query");
		expect(trace[3]._k == 16 && nearlyEqualTrace(trace[3]._center.x, 9.0f), "knn k/center round-trip");

		expect(Experiments::dominantKnnK(trace) == 16, "dominant knn k is 16");

		std::filesystem::remove(tracePath);

		// --- workload JSON: the "trace" key reaches WorkloadProfile ---
		const Experiments::WorkloadProfile profile = Experiments::parseWorkloadProfile(
			R"({"name":"replayed","trace":"some/trace.csv","numQueries":128})", "inline-test");
		expect(profile._tracePath == "some/trace.csv", "workload JSON trace key populates _tracePath");
		expect(profile._numQueries == 128, "workload JSON numQueries still parses alongside trace");
	}
}

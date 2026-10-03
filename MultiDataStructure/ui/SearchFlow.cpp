#include "SearchFlow.h"

#include "GuiCommon.h"
#include "../AppConfig.h"

#include <set>

namespace
{
	constexpr size_t MaxLogBytes = 2u << 20;

	// Routes std::cout/std::cerr into the session log while a search runs on the worker thread.
	// Keeps the tail when the log outgrows MaxLogBytes, so a long run cannot exhaust memory.
	class LogBuffer : public std::streambuf
	{
	public:
		explicit LogBuffer(SearchFlow::Session& session) : _session(session) {}

	protected:
		int overflow(int value) override
		{
			if (value != traits_type::eof())
			{
				const char c = traits_type::to_char_type(value);
				append(&c, 1);
			}
			return traits_type::not_eof(value);
		}

		std::streamsize xsputn(const char* text, std::streamsize count) override
		{
			append(text, static_cast<size_t>(count));
			return count;
		}

	private:
		void append(const char* text, size_t count)
		{
			std::lock_guard<std::mutex> lock(_session._mutex);
			_session._log.append(text, count);
			if (_session._log.size() > MaxLogBytes)
				_session._log.erase(0, _session._log.size() - MaxLogBytes / 2);
		}

		SearchFlow::Session& _session;
	};

	class ScopedCapture
	{
	public:
		explicit ScopedCapture(SearchFlow::Session& session)
			: _buffer(session), _out(std::cout.rdbuf(&_buffer)), _err(std::cerr.rdbuf(&_buffer)) {}
		~ScopedCapture()
		{
			std::cout.rdbuf(_out);
			std::cerr.rdbuf(_err);
		}

	private:
		LogBuffer		_buffer;
		std::streambuf*	_out;
		std::streambuf*	_err;
	};

	std::string timestamp()
	{
		const std::time_t now = std::time(nullptr);
		std::tm local{};
#ifdef _WIN32
		localtime_s(&local, &now);
#else
		localtime_r(&now, &local);
#endif
		char text[32];
		std::strftime(text, sizeof(text), "%Y%m%d_%H%M%S", &local);
		return text;
	}

	std::string join(const std::vector<std::string>& values, const char* separator)
	{
		std::string out;
		for (size_t i = 0; i < values.size(); ++i)
		{
			if (i)
				out += separator;
			out += values[i];
		}
		return out;
	}

	// Arguments every arm shares: the cloud, the workload, and how queries are drawn.
	std::vector<std::string> commonArgs(const SearchFlow::Settings& s, const std::string& trace)
	{
		std::vector<std::string> args = { "--mode", "schema-search", "--input", s._inputPath, "--no-synthetic" };
		if (!trace.empty())
		{
			// The workload JSON names the replay profile; the trace supplies the actual queries.
			// Score caching is off for traces (the cache key cannot vouch for a trace's content).
			args.insert(args.end(), { "--workloads", "configs/workloads/pipeline_replay.json",
									  "--input-trace", trace, "--no-score-cache" });
		}
		else
		{
			args.insert(args.end(), { "--workloads", s._workloadPath, "--knn-k", std::to_string(s._knnK),
									  "--query-seed", std::to_string(s._querySeed) });
		}
		args.insert(args.end(), { "--queries", std::to_string(s._queries),
								  "--evaluator", s._useCuda ? "cuda" : "cpu", "--no-pause" });
		return args;
	}
}

void SearchFlow::Settings::applyBudget()
{
	switch (_budget)
	{
	case Budget::Quick: _generatedCount = 24; _generations = 2; _population = 12; _queries = 1000; break;
	case Budget::Standard: _generatedCount = 32; _generations = 3; _population = 20; _queries = 2000; break;
	case Budget::Thorough: _generatedCount = 64; _generations = 4; _population = 32; _queries = 3000; break;
	case Budget::Custom: break;
	}
}

std::string SearchFlow::tunedSinglesList()
{
	std::ifstream index(Gui::resolvePath("configs/schemas/tuned_singles_small/index.txt"));
	std::vector<std::string> paths;
	std::string line;
	while (std::getline(index, line))
	{
		if (!line.empty() && line.back() == '\r')
			line.pop_back();
		if (!line.empty())
			paths.push_back(line);
	}
	return join(paths, ";");
}

std::string SearchFlow::heldOutTraceFor(const std::string& tracePath)
{
	const std::string suffix = "_opt.csv";
	if (tracePath.size() <= suffix.size() || tracePath.compare(tracePath.size() - suffix.size(), suffix.size(), suffix) != 0)
		return {};
	const std::string test = tracePath.substr(0, tracePath.size() - suffix.size()) + "_test.csv";
	return Gui::fileExists(test) ? test : std::string();
}

std::vector<SearchFlow::Arm> SearchFlow::planArms(const Settings& s, std::string* error)
{
	const auto fail = [error](const std::string& message) {
		if (error)
			*error = message;
		return std::vector<Arm>();
	};
	if (s._inputPath.empty() || !Gui::fileExists(s._inputPath))
		return fail("Choose a point cloud that exists (step 1).");
	if (s._useTrace && !Gui::fileExists(s._tracePath))
		return fail("Choose a query trace that exists, or switch to a synthetic workload (step 2).");
	if (!s._useTrace && !Gui::fileExists(Gui::resolvePath(s._workloadPath)))
		return fail("The workload profile was not found (step 2).");
	if (s._strategy == Strategy::FixedSchemas && s._fixedSchemas.empty())
		return fail("Pick at least one schema to measure (step 3).");

	const std::string out = s._outputDir;
	const std::string trace = s._useTrace ? s._tracePath : std::string();
	const std::vector<std::string> common = commonArgs(s, trace);
	std::vector<Arm> arms;

	if (s._strategy == Strategy::Recommended || s._strategy == Strategy::GeneticOnly)
	{
		Arm ga;
		ga._label = "Genetic search";
		ga._csvPath = out + "/ga.csv";
		ga._bestCsvPath = out + "/ga_best.csv";
		ga._args = common;
		ga._args.insert(ga._args.end(), {
			"--generate-schemas", std::to_string(s._generatedCount),
			"--optimizer-generations", std::to_string(s._generations),
			"--optimizer-population", std::to_string(s._population),
			"--parallel", std::to_string(std::max(1, s._parallel)),
			"--generated-schema-dir", out + "/generated_schemas",
			"--optimizer-output-dir", out + "/best_schemas",
			"--csv", ga._csvPath, "--best-csv", ga._bestCsvPath, "--pareto-csv", out + "/ga_pareto.csv" });
		arms.push_back(std::move(ga));
	}

	if (s._strategy == Strategy::Recommended || s._strategy == Strategy::TunedSinglesOnly)
	{
		Arm singles;
		singles._label = "Tuned single structures";
		singles._csvPath = out + "/singles.csv";
		singles._bestCsvPath = out + "/singles_best.csv";
		singles._args = { "--flat-search" };
		singles._args.insert(singles._args.end(), common.begin(), common.end());
		singles._args.insert(singles._args.end(), {
			"--schemas", tunedSinglesList(), "--generate-schemas", "0", "--no-baselines",
			"--csv", singles._csvPath, "--best-csv", singles._bestCsvPath });
		arms.push_back(std::move(singles));
	}

	if (s._strategy == Strategy::FixedSchemas)
	{
		Arm fixed;
		fixed._label = "Selected schemas";
		fixed._csvPath = out + "/fixed.csv";
		fixed._bestCsvPath = out + "/fixed_best.csv";
		fixed._args = { "--flat-search" };
		fixed._args.insert(fixed._args.end(), common.begin(), common.end());
		fixed._args.insert(fixed._args.end(), {
			"--schemas", join(s._fixedSchemas, ";"), "--generate-schemas", "0", "--no-baselines",
			"--measure-repeats", "3", "--csv", fixed._csvPath, "--best-csv", fixed._bestCsvPath });
		arms.push_back(std::move(fixed));
	}
	return arms;
}

std::optional<SearchFlow::Arm> SearchFlow::validationArm(const Settings& s, const std::vector<std::string>& schemaPaths)
{
	const std::string heldOut = s._useTrace ? heldOutTraceFor(s._tracePath) : std::string();
	if (heldOut.empty() || schemaPaths.empty())
		return std::nullopt;

	Arm arm;
	arm._label = "Held-out validation";
	arm._csvPath = s._outputDir + "/heldout.csv";
	arm._bestCsvPath = s._outputDir + "/heldout_best.csv";
	arm._args = { "--flat-search" };
	const std::vector<std::string> common = commonArgs(s, heldOut);
	arm._args.insert(arm._args.end(), common.begin(), common.end());
	arm._args.insert(arm._args.end(), {
		"--schemas", join(schemaPaths, ";"), "--generate-schemas", "0", "--no-baselines",
		"--measure-repeats", "3", "--csv", arm._csvPath, "--best-csv", arm._bestCsvPath });
	return arm;
}

std::string SearchFlow::commandLine(const Arm& arm)
{
	std::string line = "MultiDataStructure.exe";
	for (const std::string& arg : arm._args)
	{
		const bool quote = arg.find_first_of(" ;") != std::string::npos || arg.empty();
		line += quote ? " \"" + arg + "\"" : " " + arg;
	}
	return line;
}

std::vector<SearchFlow::ResultRow> SearchFlow::loadResults(const Arm& arm)
{
	const Gui::CsvTable table = Gui::readCsv(arm._csvPath);
	const bool hasFinalFlag = table.column("score_is_final_latency") != std::numeric_limits<size_t>::max();
	std::map<std::string, ResultRow> bestBySchema;
	for (size_t r = 0; r < table._rows.size(); ++r)
	{
		// Proxy-rung rows score a visit-count proxy, not latency; they must not compete with
		// measured rows. Only final-latency rows enter the table.
		if (hasFinalFlag && table.value(r, "score_is_final_latency") != "1")
			continue;
		ResultRow row;
		row._arm = arm._label;
		row._schemaName = table.value(r, "schema_name");
		row._schemaPath = table.value(r, "schema_path");
		const double repeatMean = table.number(r, "latency_repeat_mean_ms", 0.0);
		row._latencyMs = repeatMean > 0.0 ? repeatMean : table.number(r, "avg_latency_ms");
		row._buildMs = table.number(r, "build_time_ms");
		row._memoryBytes = table.number(r, "memory_estimate_bytes");
		row._visited = table.number(r, "avg_visited_nodes");
		row._tested = table.number(r, "avg_tested_points");
		row._score = table.number(r, "score");
		row._structures = table.value(r, "active_structure_summary");
		row._baseline = table.value(r, "is_baseline") == "1";
		const auto existing = bestBySchema.find(row._schemaName);
		if (existing == bestBySchema.end() || row._latencyMs < existing->second._latencyMs)
			bestBySchema[row._schemaName] = row;
	}
	std::vector<ResultRow> rows;
	for (auto& [name, row] : bestBySchema)
		rows.push_back(std::move(row));
	std::sort(rows.begin(), rows.end(), [](const ResultRow& a, const ResultRow& b) { return a._latencyMs < b._latencyMs; });
	return rows;
}

// ---------------------------------------------------------------- session

SearchFlow::Session::~Session()
{
	cancel();
	if (_worker.joinable())
		_worker.join();
}

void SearchFlow::Session::cancel()
{
	_cancel->store(true);
}

void SearchFlow::Session::poll()
{
	if (_finished.exchange(false) && _worker.joinable())
		_worker.join();
	if (_running)
	{
		std::lock_guard<std::mutex> lock(_mutex);
		_elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - _started).count();
	}
}

void SearchFlow::Session::start(const Settings& settingsIn)
{
	if (_running)
		return;
	if (_worker.joinable())
		_worker.join();

	Settings settings = settingsIn;
	if (settings._outputDir.empty())
	{
		const std::string stem = std::filesystem::path(settings._inputPath).stem().stem().string();
		settings._outputDir = Gui::projectPath("results/gui/" + stem + "_" + timestamp());
	}

	std::string error;
	std::vector<Arm> arms = planArms(settings, &error);
	{
		std::lock_guard<std::mutex> lock(_mutex);
		_error = error;
		_log.clear();
		_live.clear();
		_results.clear();
		_validation.clear();
		_arms = arms;
		_currentArm = 0;
		_stage = Experiments::SearchStage();
		_status = arms.empty() ? "Not started" : "Running";
	}
	if (arms.empty())
		return;

	_cancel = std::make_shared<std::atomic<bool>>(false);
	_started = std::chrono::steady_clock::now();
	_elapsedSeconds = 0.0;
	_running = true;
	_finished = false;
	_worker = std::thread([this, settings, arms]() {
		std::string failure;
		bool cancelled = false;
		std::vector<ResultRow> allResults;
		const auto runArm = [&](const Arm& arm, size_t armIndex) -> std::vector<ResultRow> {
			{
				std::lock_guard<std::mutex> lock(_mutex);
				_currentArm = armIndex;
				_status = arm._label;
				_stage = Experiments::SearchStage{ arm._label, 0, 0 };
				_log += "\n=== " + arm._label + " ===\n" + commandLine(arm) + "\n";
			}
			std::filesystem::create_directories(settings._outputDir);

			std::vector<std::string> argv = { (Gui::executableDirectory() / "MultiDataStructure.exe").string() };
			argv.insert(argv.end(), arm._args.begin(), arm._args.end());
			std::vector<char*> pointers;
			for (std::string& value : argv)
				pointers.push_back(value.data());
			AppConfig config = AppConfig::parse(static_cast<int>(pointers.size()), pointers.data());

			Experiments::SchemaSearchOptions options = config._schemaSearchOptions;
			options.cancelRequested = _cancel;
			options.progressCallback = [this, armIndex](const Experiments::SchemaSearchRecord& record) {
				LiveRecord live;
				live._arm = armIndex;
				live._schemaName = record._schemaName;
				live._schemaPath = record._schemaPath;
				live._score = record._score;
				live._latencyMs = record._queryMetrics._averageLatencyMs;
				live._buildMs = record._buildMetrics._buildTimeMs;
				live._visited = record._queryMetrics._averageVisitedNodes;
				live._tested = record._queryMetrics._averageTestedPoints;
				live._baseline = record._isBaseline;
				std::lock_guard<std::mutex> lock(_mutex);
				live._order = _live.size();
				_live.push_back(std::move(live));
			};
			options.stageCallback = [this](const Experiments::SearchStage& stage) {
				std::lock_guard<std::mutex> lock(_mutex);
				_stage = stage;
			};

			ScopedCapture capture(*this);
			const int code = Experiments::runSchemaSearch(options);
			if (code != 0)
				throw std::runtime_error(arm._label + " exited with code " + std::to_string(code));
			return loadResults(arm);
		};

		try
		{
			for (size_t i = 0; i < arms.size(); ++i)
			{
				std::vector<ResultRow> rows = runArm(arms[i], i);
				allResults.insert(allResults.end(), rows.begin(), rows.end());
				std::lock_guard<std::mutex> lock(_mutex);
				_results = allResults;
			}

			// Winners of each arm, plus the default structures as context, re-measured on queries
			// the search never saw.
			if (settings._validateHeldOut)
			{
				std::vector<std::string> winners;
				std::set<std::string> seenArms;
				for (const ResultRow& row : allResults)
				{
					if (!row._schemaPath.empty() && seenArms.insert(row._arm).second)
						winners.push_back(row._schemaPath);
				}
				for (const char* fallback : { "configs/schemas/octree.json", "configs/schemas/kdtree.json", "configs/schemas/quadtree.json" })
					winners.push_back(Gui::resolvePath(fallback));
				if (std::optional<Arm> validation = validationArm(settings, winners))
				{
					{
						std::lock_guard<std::mutex> lock(_mutex);
						_arms.push_back(*validation);
					}
					std::vector<ResultRow> rows = runArm(*validation, arms.size());
					std::lock_guard<std::mutex> lock(_mutex);
					_validation = std::move(rows);
				}
			}
		}
		catch (const Experiments::SearchCancelled&)
		{
			cancelled = true;
		}
		catch (const std::exception& exception)
		{
			failure = exception.what();
		}

		{
			std::lock_guard<std::mutex> lock(_mutex);
			_error = failure;
			_status = cancelled ? "Cancelled" : (failure.empty() ? "Complete" : "Failed");
			_elapsedSeconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - _started).count();
		}
		_running = false;
		_finished = true;
	});
}

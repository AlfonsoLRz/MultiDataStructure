#pragma once

#include "../stdafx.h"
#include "../experiments/SchemaSearch.h"

// What the GUI runs, and how it runs it.
//
// The GUI never fills SchemaSearchOptions field by field. It builds the exact command line the
// CLI would take for each step ("arm") of a run and feeds that through AppConfig::parse, the same
// code path as a typed command. That makes "Copy command" reproduce a GUI run exactly, and it is
// what keeps the GUI and the CLI from running different algorithms (the old GUI defaulted to the
// auto-condition search while the CLI defaulted to the GA).
namespace SearchFlow
{
	enum class Strategy
	{
		Recommended,	// GA search + tuned single structures, compared side by side
		GeneticOnly,
		TunedSinglesOnly,
		FixedSchemas,	// measure a hand-picked list, no search
	};

	enum class Budget { Quick, Standard, Thorough, Custom };

	struct Settings
	{
		// 1. Data
		std::string	_inputPath;
		std::vector<std::string>	_recentInputs;

		// 2. Workload: a recorded trace (preferred) or a synthetic profile.
		bool		_useTrace = true;
		std::string	_tracePath;
		std::string	_workloadPath = "configs/workloads/pipeline_replay.json";
		int			_queries = 2000;
		int			_knnK = 16;
		int			_querySeed = 1337;

		// 3. Search
		Strategy	_strategy = Strategy::Recommended;
		Budget		_budget = Budget::Standard;
		int			_generatedCount = 32;
		int			_generations = 3;
		int			_population = 20;
		int			_parallel = 8;
		bool		_useCuda = false;
		std::vector<std::string>	_fixedSchemas;
		std::string	_outputDir;
		bool		_validateHeldOut = true;	// re-measure winners on the held-out trace half

		void applyBudget();
	};

	struct Arm
	{
		std::string					_label;
		std::vector<std::string>	_args;		// without argv[0]
		std::string					_csvPath;
		std::string					_bestCsvPath;
	};

	// The arms a run will execute, in order. The held-out validation arm cannot be built until
	// the search arms have produced winners; see validationArm().
	std::vector<Arm> planArms(const Settings& settings, std::string* error);
	// Re-measures the given schemas on the held-out trace half, when there is one.
	std::optional<Arm> validationArm(const Settings& settings, const std::vector<std::string>& schemaPaths);
	// pipeline_trace_opt.csv -> pipeline_trace_test.csv, if that file exists.
	std::string heldOutTraceFor(const std::string& tracePath);
	std::string commandLine(const Arm& arm);
	std::string tunedSinglesList();

	struct LiveRecord
	{
		size_t		_order = 0;
		size_t		_arm = 0;
		std::string	_schemaName;
		std::string	_schemaPath;
		double		_score = 0.0;
		double		_latencyMs = 0.0;
		double		_buildMs = 0.0;
		double		_visited = 0.0;
		double		_tested = 0.0;
		bool		_baseline = false;
	};

	struct ResultRow
	{
		std::string	_arm;
		std::string	_schemaName;
		std::string	_schemaPath;
		double		_latencyMs = 0.0;
		double		_buildMs = 0.0;
		double		_memoryBytes = 0.0;
		double		_visited = 0.0;
		double		_tested = 0.0;
		double		_score = 0.0;
		std::string	_structures;
		bool		_baseline = false;
	};

	struct Session
	{
		std::thread					_worker;
		std::atomic<bool>			_running{ false };
		std::atomic<bool>			_finished{ false };
		std::shared_ptr<std::atomic<bool>>	_cancel = std::make_shared<std::atomic<bool>>(false);
		std::mutex					_mutex;
		std::string					_status = "Idle";
		std::string					_error;
		std::string					_log;
		std::vector<Arm>			_arms;
		size_t						_currentArm = 0;
		Experiments::SearchStage	_stage;
		std::vector<LiveRecord>		_live;
		std::vector<ResultRow>		_results;
		std::vector<ResultRow>		_validation;
		std::chrono::steady_clock::time_point	_started;
		double						_elapsedSeconds = 0.0;

		~Session();
		void start(const Settings& settings);
		void cancel();
		// Call once per frame: joins a finished worker.
		void poll();
	};

	// Best row per arm by score, read from an arm's CSV after it finishes.
	std::vector<ResultRow> loadResults(const Arm& arm);
}

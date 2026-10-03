#include "OptimizerGui.h"

#include "Charts.h"
#include "GuiCommon.h"
#include "SearchFlow.h"
#include "Viewer3D.h"
#include "../AppConfig.h"
#include "../core/Config.h"
#include "../experiments/QueryTrace.h"
#include "../workloads/points/PointCloud.h"

#include "imgui_stdlib.h"
#include <boost/json.hpp>

#include <set>

namespace
{
	enum class Step { Data, Workload, Search, Run, Results, Inspect };
	constexpr const char* StepNames[] = { "1  Data", "2  Workload", "3  Search", "4  Run", "5  Results", "6  Inspect" };

	const ImU32 ArmColors[] = { IM_COL32(140, 212, 184, 255), IM_COL32(242, 199, 82, 255), IM_COL32(150, 178, 255, 255), IM_COL32(255, 122, 140, 255) };

	struct BenchmarkCell
	{
		std::string	_name;
		std::string	_cloud;
		std::string	_trace;
		size_t		_points = 0;
	};

	struct TraceSummary
	{
		std::string	_path;
		size_t		_total = 0;
		std::map<std::string, size_t>	_byKind;
		std::string	_error;
	};

	// Everything the Inspect step owns: a loaded cloud, an index built on it, the GPU-side scene,
	// and the last query probed against it. Loading and building run on a worker thread.
	struct Inspect
	{
		std::string	_schemaPath;
		std::string	_cloudPath;
		std::shared_ptr<PointCloud>			_cloud;
		std::shared_ptr<PointSpatialIndex>	_index;
		SchemaConfig	_schema;
		std::thread		_worker;
		std::atomic<bool>	_busy{ false };
		std::atomic<bool>	_ready{ false };
		std::mutex		_mutex;
		std::string		_status;
		double			_loadMs = 0.0;
		double			_buildMs = 0.0;
		PointSpatialIndex::Stats	_stats;

		Viewer::Renderer		_renderer;
		Viewer::Camera			_camera;
		Viewer::SceneBuilder	_builder;
		bool		_sceneDirty = false;
		bool		_overlayDirty = false;
		float		_sceneRadius = 1.0f;
		float		_pointSize = 2.0f;
		size_t		_boxesShown = 0;

		Viewer::QueryKind	_queryKind = Viewer::QueryKind::Radius;
		float		_radius = 0.5f;
		int			_k = 16;
		float		_halfExtent = 1.0f;
		glm::vec3	_center = glm::vec3(0.0f);
		int			_traceQuery = 0;
		std::vector<Experiments::TraceQuery>	_trace;
		std::string	_tracePath;
		Viewer::QueryOverlay	_overlay;
		uint64_t	_seed = 1;

		~Inspect()
		{
			if (_worker.joinable())
				_worker.join();
		}
	};

	struct App
	{
		SearchFlow::Settings	_settings;
		SearchFlow::Session		_session;
		Step		_step = Step::Data;
		Inspect		_inspect;
		std::vector<BenchmarkCell>	_cells;
		std::vector<std::string>	_workloads;
		std::vector<std::string>	_schemaChoices;
		TraceSummary	_traceSummary;
		std::string		_loadedResultsDir;
		std::vector<SearchFlow::ResultRow>	_shownResults;
		std::vector<SearchFlow::ResultRow>	_shownValidation;
		int			_selectedResult = -1;
		std::string	_logCopy;
		std::vector<size_t>	_logLines;
		bool		_autoScroll = true;
		std::string	_notice;
	};

	std::filesystem::path settingsFile()
	{
		return Gui::executableDirectory() / "mds_gui_settings.json";
	}

	// ------------------------------------------------------------ settings persistence

	void saveSettings(const SearchFlow::Settings& s)
	{
		boost::json::object o;
		o["input"] = s._inputPath;
		boost::json::array recent;
		for (const std::string& r : s._recentInputs)
			recent.push_back(boost::json::value(r));
		o["recent"] = recent;
		o["use_trace"] = s._useTrace;
		o["trace"] = s._tracePath;
		o["workload"] = s._workloadPath;
		o["queries"] = s._queries;
		o["knn_k"] = s._knnK;
		o["query_seed"] = s._querySeed;
		o["strategy"] = static_cast<int>(s._strategy);
		o["budget"] = static_cast<int>(s._budget);
		o["generated"] = s._generatedCount;
		o["generations"] = s._generations;
		o["population"] = s._population;
		o["parallel"] = s._parallel;
		o["cuda"] = s._useCuda;
		o["validate"] = s._validateHeldOut;
		std::ofstream out(settingsFile());
		out << boost::json::serialize(o);
	}

	void loadSettings(SearchFlow::Settings& s)
	{
		std::ifstream in(settingsFile());
		if (!in)
			return;
		const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		boost::system::error_code error;
		const boost::json::value parsed = boost::json::parse(text, error);
		if (error || !parsed.is_object())
			return;
		const boost::json::object& o = parsed.as_object();
		const auto str = [&](const char* key, std::string& target) {
			if (const auto* v = o.if_contains(key); v && v->is_string()) target = std::string(v->as_string());
		};
		const auto num = [&](const char* key, int& target) {
			if (const auto* v = o.if_contains(key); v && v->is_int64()) target = static_cast<int>(v->as_int64());
		};
		const auto flag = [&](const char* key, bool& target) {
			if (const auto* v = o.if_contains(key); v && v->is_bool()) target = v->as_bool();
		};
		str("input", s._inputPath);
		if (const auto* v = o.if_contains("recent"); v && v->is_array())
		{
			s._recentInputs.clear();
			for (const auto& r : v->as_array())
				if (r.is_string())
					s._recentInputs.push_back(std::string(r.as_string()));
		}
		flag("use_trace", s._useTrace);
		str("trace", s._tracePath);
		str("workload", s._workloadPath);
		num("queries", s._queries);
		num("knn_k", s._knnK);
		num("query_seed", s._querySeed);
		int strategy = static_cast<int>(s._strategy), budget = static_cast<int>(s._budget);
		num("strategy", strategy);
		num("budget", budget);
		s._strategy = static_cast<SearchFlow::Strategy>(std::clamp(strategy, 0, 3));
		s._budget = static_cast<SearchFlow::Budget>(std::clamp(budget, 0, 3));
		num("generated", s._generatedCount);
		num("generations", s._generations);
		num("population", s._population);
		num("parallel", s._parallel);
		flag("cuda", s._useCuda);
		flag("validate", s._validateHeldOut);
	}

	// ------------------------------------------------------------ discovery

	std::vector<BenchmarkCell> discoverCells()
	{
		std::vector<BenchmarkCell> cells;
		std::ifstream in(Gui::resolvePath("configs/datasets/curated_cells.json"));
		if (!in)
			return cells;
		const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
		boost::system::error_code error;
		const boost::json::value parsed = boost::json::parse(text, error);
		if (error || !parsed.is_object() || !parsed.as_object().contains("cells"))
			return cells;
		for (const auto& value : parsed.as_object().at("cells").as_array())
		{
			const boost::json::object& c = value.as_object();
			BenchmarkCell cell;
			cell._name = std::string(c.at("cell").as_string());
			cell._cloud = std::string(c.at("mdspc_path").as_string());
			cell._trace = Gui::resolvePath(std::string(c.at("trace_dir").as_string()) + "/pipeline_trace_opt.csv");
			cell._points = c.contains("points") ? static_cast<size_t>(c.at("points").to_number<double>()) : 0;
			if (Gui::fileExists(cell._cloud))
				cells.push_back(std::move(cell));
		}
		return cells;
	}

	std::vector<std::string> listJson(const std::string& relativeDir)
	{
		std::vector<std::string> paths;
		std::error_code error;
		const std::filesystem::path dir(Gui::resolvePath(relativeDir));
		for (const auto& entry : std::filesystem::directory_iterator(dir, error))
		{
			if (entry.is_regular_file() && entry.path().extension() == ".json")
				paths.push_back(entry.path().lexically_normal().string());
		}
		std::sort(paths.begin(), paths.end());
		return paths;
	}

	TraceSummary summarizeTrace(const std::string& path)
	{
		TraceSummary summary;
		summary._path = path;
		std::ifstream in(path);
		if (!in)
		{
			summary._error = "cannot open";
			return summary;
		}
		std::string line;
		std::getline(in, line);
		const std::vector<std::string> header = Gui::splitCsvLine(line);
		const auto kindColumn = std::find(header.begin(), header.end(), "query_type");
		if (kindColumn == header.end())
		{
			summary._error = "no query_type column";
			return summary;
		}
		const size_t column = static_cast<size_t>(kindColumn - header.begin());
		while (std::getline(in, line))
		{
			const std::vector<std::string> cells = Gui::splitCsvLine(line);
			if (column < cells.size())
			{
				++summary._byKind[cells[column]];
				++summary._total;
			}
		}
		return summary;
	}

	std::string stemOf(const std::string& path)
	{
		return std::filesystem::path(path).filename().string();
	}

	void rememberInput(SearchFlow::Settings& s)
	{
		if (s._inputPath.empty())
			return;
		auto& recent = s._recentInputs;
		recent.erase(std::remove(recent.begin(), recent.end(), s._inputPath), recent.end());
		recent.insert(recent.begin(), s._inputPath);
		if (recent.size() > 8)
			recent.resize(8);
	}

	bool nextButton(const char* label)
	{
		ImGui::Spacing();
		ImGui::Separator();
		ImGui::Spacing();
		return ImGui::Button(label, ImVec2(220.0f, 34.0f));
	}

	void sectionTitle(const char* title)
	{
		ImGui::Spacing();
		ImGui::TextUnformatted(title);
		ImGui::Separator();
	}

	// ------------------------------------------------------------ step 1: data

	void drawDataStep(App& app)
	{
		SearchFlow::Settings& s = app._settings;
		sectionTitle("Point cloud");
		Gui::hint("The cloud the index will be built on. LAS, binary or ASCII PLY, XYZ and CSV are read directly; "
				  "a .mdspc cache next to a file (or the .mdspc itself) loads much faster.");

		ImGui::SetNextItemWidth(-110.0f);
		ImGui::InputText("##input", &s._inputPath);
		ImGui::SameLine();
		if (ImGui::Button("Browse...", ImVec2(100.0f, 0.0f)))
		{
			const std::string picked = Gui::openFileDialog("Point clouds\0*.las;*.ply;*.xyz;*.csv;*.mdspc\0All files\0*.*\0", s._inputPath);
			if (!picked.empty())
				s._inputPath = picked;
		}

		if (!s._inputPath.empty())
		{
			std::error_code error;
			if (Gui::fileExists(s._inputPath))
			{
				const double bytes = static_cast<double>(std::filesystem::file_size(s._inputPath, error));
				const bool cached = std::filesystem::path(s._inputPath).extension() == ".mdspc" || Gui::fileExists(s._inputPath + ".mdspc");
				ImGui::TextColored(ImVec4(0.55f, 0.83f, 0.72f, 1.0f), "Found, %s%s", Gui::formatBytes(bytes).c_str(),
								   cached ? " - binary cache available" : " - first load will create a binary cache");
			}
			else
				ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.45f, 1.0f), "File not found.");
		}

		if (!s._recentInputs.empty())
		{
			ImGui::Spacing();
			ImGui::TextDisabled("Recent");
			for (const std::string& recent : s._recentInputs)
			{
				if (ImGui::Selectable(recent.c_str(), recent == s._inputPath))
					s._inputPath = recent;
			}
		}

		if (!app._cells.empty())
		{
			ImGui::Spacing();
			sectionTitle("Curated benchmark");
			Gui::hint("Cells of the curated multi-scale benchmark that are present on this machine. Picking one also selects "
					  "its recorded query trace, so the search can be validated on held-out queries.");
			static ImGuiTextFilter filter;
			filter.Draw("Filter cells", 220.0f);
			ImGui::BeginChild("cells", ImVec2(0.0f, 220.0f), ImGuiChildFlags_Borders);
			for (const BenchmarkCell& cell : app._cells)
			{
				if (!filter.PassFilter(cell._name.c_str()))
					continue;
				const bool selected = cell._cloud == s._inputPath;
				const std::string label = cell._name + "   (" + Gui::formatCount(static_cast<double>(cell._points)) + " points)";
				if (ImGui::Selectable(label.c_str(), selected))
				{
					s._inputPath = cell._cloud;
					if (Gui::fileExists(cell._trace))
					{
						s._useTrace = true;
						s._tracePath = cell._trace;
					}
				}
			}
			ImGui::EndChild();
		}

		if (nextButton("Next: workload  >"))
		{
			rememberInput(s);
			app._step = Step::Workload;
		}
	}

	// ------------------------------------------------------------ step 2: workload

	void drawWorkloadStep(App& app)
	{
		SearchFlow::Settings& s = app._settings;
		sectionTitle("Queries the index must answer");
		Gui::hint("The search tunes the index for a workload. A recorded trace of the application's own queries is the "
				  "faithful choice; synthetic profiles are for exploring without one.");

		if (ImGui::RadioButton("Replay a recorded query trace (recommended)", s._useTrace))
			s._useTrace = true;
		if (ImGui::RadioButton("Generate synthetic queries from a profile", !s._useTrace))
			s._useTrace = false;
		ImGui::Spacing();

		if (s._useTrace)
		{
			ImGui::SetNextItemWidth(-110.0f);
			ImGui::InputText("##trace", &s._tracePath);
			ImGui::SameLine();
			if (ImGui::Button("Browse...##trace", ImVec2(100.0f, 0.0f)))
			{
				const std::string picked = Gui::openFileDialog("Query traces\0*.csv\0All files\0*.*\0", s._tracePath);
				if (!picked.empty())
					s._tracePath = picked;
			}
			if (app._traceSummary._path != s._tracePath && Gui::fileExists(s._tracePath))
				app._traceSummary = summarizeTrace(s._tracePath);
			if (Gui::fileExists(s._tracePath) && app._traceSummary._error.empty())
			{
				std::string kinds;
				for (const auto& [kind, count] : app._traceSummary._byKind)
					kinds += (kinds.empty() ? "" : ", ") + Gui::formatCount(static_cast<double>(count)) + " " + kind;
				ImGui::TextColored(ImVec4(0.55f, 0.83f, 0.72f, 1.0f), "%s queries: %s",
								   Gui::formatCount(static_cast<double>(app._traceSummary._total)).c_str(), kinds.c_str());
				const std::string heldOut = SearchFlow::heldOutTraceFor(s._tracePath);
				if (!heldOut.empty())
					ImGui::TextDisabled("Held-out half found (%s): winners will be re-measured on it.", stemOf(heldOut).c_str());
				else
					ImGui::TextDisabled("No held-out half (*_test.csv) next to this trace; results will not be validated on unseen queries.");
			}
			else if (!s._tracePath.empty())
				ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.45f, 1.0f), "Trace not found or unreadable.");
		}
		else
		{
			if (ImGui::BeginCombo("Profile", stemOf(s._workloadPath).c_str()))
			{
				for (const std::string& path : app._workloads)
				{
					if (ImGui::Selectable(stemOf(path).c_str(), path == s._workloadPath))
						s._workloadPath = path;
				}
				ImGui::EndCombo();
			}
			ImGui::SliderInt("kNN k", &s._knnK, 1, 64);
			ImGui::InputInt("Query seed", &s._querySeed);
		}

		ImGui::SliderInt("Queries per evaluation", &s._queries, 100, 10000, "%d", ImGuiSliderFlags_Logarithmic);
		Gui::helpMarker("How many queries time each candidate. A trace larger than this is subsampled with an even stride, "
						"so the mixture of query kinds is kept. More queries = more reliable rankings, slower search.");

		if (nextButton("Next: search  >"))
			app._step = Step::Search;
	}

	// ------------------------------------------------------------ step 3: search

	void drawSearchStep(App& app)
	{
		SearchFlow::Settings& s = app._settings;
		sectionTitle("How to search");

		const auto strategy = [&](SearchFlow::Strategy value, const char* label, const char* help) {
			if (ImGui::RadioButton(label, s._strategy == value))
				s._strategy = value;
			ImGui::Indent(28.0f);
			Gui::hint(help);
			ImGui::Unindent(28.0f);
		};
		strategy(SearchFlow::Strategy::Recommended, "Search, and compare against tuned single structures (recommended)",
				 "A genetic search over nested schemas, plus every single structure (octree, kd-tree, quadtree, BVH, BIH, "
				 "grids) tuned over depth and leaf size. You see whether nesting actually beats the best single structure. "
				 "This is the protocol the evaluation reports.");
		strategy(SearchFlow::Strategy::GeneticOnly, "Genetic search only", "Faster; no single-structure comparison.");
		strategy(SearchFlow::Strategy::TunedSinglesOnly, "Tuned single structures only", "No search: the best of the tuned single-structure grid.");
		strategy(SearchFlow::Strategy::FixedSchemas, "Measure schemas I choose", "No search: time a hand-picked list of schema files.");

		if (s._strategy == SearchFlow::Strategy::FixedSchemas)
		{
			ImGui::BeginChild("fixed", ImVec2(0.0f, 180.0f), ImGuiChildFlags_Borders);
			for (const std::string& path : app._schemaChoices)
			{
				bool selected = std::find(s._fixedSchemas.begin(), s._fixedSchemas.end(), path) != s._fixedSchemas.end();
				if (ImGui::Checkbox(stemOf(path).c_str(), &selected))
				{
					if (selected)
						s._fixedSchemas.push_back(path);
					else
						s._fixedSchemas.erase(std::remove(s._fixedSchemas.begin(), s._fixedSchemas.end(), path), s._fixedSchemas.end());
				}
			}
			ImGui::EndChild();
		}

		if (s._strategy == SearchFlow::Strategy::Recommended || s._strategy == SearchFlow::Strategy::GeneticOnly)
		{
			ImGui::Spacing();
			sectionTitle("Search budget");
			const char* budgets[] = { "Quick", "Standard", "Thorough", "Custom" };
			int budget = static_cast<int>(s._budget);
			for (int i = 0; i < 4; ++i)
			{
				if (i)
					ImGui::SameLine();
				if (ImGui::RadioButton(budgets[i], budget == i))
				{
					s._budget = static_cast<SearchFlow::Budget>(i);
					s.applyBudget();
				}
			}
			ImGui::BeginDisabled(s._budget != SearchFlow::Budget::Custom);
			ImGui::SliderInt("Initial candidates", &s._generatedCount, 8, 256);
			ImGui::SliderInt("Generations", &s._generations, 1, 10);
			ImGui::SliderInt("Population", &s._population, 4, 64);
			ImGui::EndDisabled();
			Gui::hint("Larger clouds need larger budgets: at 100M points the original budget was the binding constraint. "
					  "Standard matches the benchmark at 50-100M; Quick is for a first look.");
		}

		ImGui::Spacing();
		sectionTitle("Execution");
		const int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
		ImGui::SliderInt("Parallel candidates", &s._parallel, 1, threads);
		Gui::helpMarker("Candidates evaluated concurrently. Each builds its own index, so memory grows with this. "
						"Timing noise grows too; the held-out validation re-measures winners alone.");
		ImGui::Checkbox("Use the GPU evaluator (experimental)", &s._useCuda);
		Gui::helpMarker("The CPU evaluator is the measured, comparable path. Schemas the GPU cannot build fall back to the CPU, "
						"so a GPU ranking can mix backends.");
		ImGui::BeginDisabled(!s._useTrace || SearchFlow::heldOutTraceFor(s._tracePath).empty());
		ImGui::Checkbox("Validate winners on the held-out half of the trace", &s._validateHeldOut);
		ImGui::EndDisabled();
		ImGui::SetNextItemWidth(-110.0f);
		ImGui::InputTextWithHint("##outdir", "output folder (default: results/gui/<cloud>_<time>)", &s._outputDir);
		ImGui::SameLine();
		if (ImGui::Button("Choose...", ImVec2(100.0f, 0.0f)))
		{
			const std::string picked = Gui::selectFolderDialog(s._outputDir);
			if (!picked.empty())
				s._outputDir = picked;
		}

		// What will run, and the exact commands.
		ImGui::Spacing();
		sectionTitle("What will run");
		SearchFlow::Settings preview = s;
		if (preview._outputDir.empty())
			preview._outputDir = "<output folder>";
		std::string error;
		const std::vector<SearchFlow::Arm> arms = SearchFlow::planArms(preview, &error);
		if (!error.empty())
			ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.45f, 1.0f), "%s", error.c_str());
		std::string commands;
		for (size_t i = 0; i < arms.size(); ++i)
		{
			ImGui::BulletText("%s", arms[i]._label.c_str());
			commands += SearchFlow::commandLine(arms[i]) + "\n";
		}
		if (!arms.empty() && s._validateHeldOut && s._useTrace && !SearchFlow::heldOutTraceFor(s._tracePath).empty())
			ImGui::BulletText("Held-out validation of the winners (added when the search finishes)");
		ImGui::BeginDisabled(arms.empty());
		if (ImGui::Button("Copy command line"))
			Gui::copyToClipboard(commands);
		ImGui::EndDisabled();
		Gui::helpMarker("The exact CLI commands this run executes, so a GUI run can be reproduced or scripted.");

		ImGui::Spacing();
		ImGui::Separator();
		ImGui::Spacing();
		ImGui::BeginDisabled(arms.empty() || app._session._running);
		if (ImGui::Button("Start search", ImVec2(220.0f, 38.0f)))
		{
			rememberInput(s);
			saveSettings(s);
			app._session.start(s);
			app._shownResults.clear();
			app._shownValidation.clear();
			app._loadedResultsDir.clear();
			app._step = Step::Run;
		}
		ImGui::EndDisabled();
	}

	// ------------------------------------------------------------ step 4: run

	void drawLog(App& app)
	{
		{
			std::lock_guard<std::mutex> lock(app._session._mutex);
			if (app._session._log.size() != app._logCopy.size())
				app._logCopy = app._session._log;
		}
		app._logLines.clear();
		app._logLines.push_back(0);
		for (size_t i = 0; i < app._logCopy.size(); ++i)
			if (app._logCopy[i] == '\n')
				app._logLines.push_back(i + 1);

		ImGui::Checkbox("Follow", &app._autoScroll);
		ImGui::BeginChild("log", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
		ImGuiListClipper clipper;
		clipper.Begin(static_cast<int>(app._logLines.size()));
		while (clipper.Step())
		{
			for (int line = clipper.DisplayStart; line < clipper.DisplayEnd; ++line)
			{
				const size_t begin = app._logLines[static_cast<size_t>(line)];
				const size_t end = static_cast<size_t>(line) + 1 < app._logLines.size() ? app._logLines[static_cast<size_t>(line) + 1] : app._logCopy.size();
				ImGui::TextUnformatted(app._logCopy.data() + begin, app._logCopy.data() + end);
			}
		}
		if (app._autoScroll && app._session._running)
			ImGui::SetScrollHereY(1.0f);
		ImGui::EndChild();
	}

	void drawRunStep(App& app)
	{
		SearchFlow::Session& session = app._session;
		std::string status, error, stageLabel;
		size_t currentArm = 0, armCount = 0, stageIndex = 0, stageTotal = 0;
		std::vector<SearchFlow::LiveRecord> live;
		std::vector<std::string> armLabels;
		double elapsed = 0.0;
		{
			std::lock_guard<std::mutex> lock(session._mutex);
			status = session._status;
			error = session._error;
			currentArm = session._currentArm;
			armCount = session._arms.size();
			for (const SearchFlow::Arm& arm : session._arms)
				armLabels.push_back(arm._label);
			stageLabel = session._stage._label;
			stageIndex = session._stage._index;
			stageTotal = session._stage._total;
			live = session._live;
			elapsed = session._elapsedSeconds;
		}

		sectionTitle("Run");
		if (armCount == 0)
		{
			Gui::hint("Nothing has run yet. Configure the search in step 3 and press Start.");
			return;
		}
		ImGui::Text("%s   |   step %zu of %zu   |   %zu evaluations   |   %.0f s", status.c_str(),
					std::min(currentArm + 1, armCount), armCount, live.size(), elapsed);
		if (session._running)
		{
			const float fraction = stageTotal > 0 ? static_cast<float>(stageIndex) / static_cast<float>(stageTotal) : -1.0f * static_cast<float>(ImGui::GetTime());
			const std::string overlay = stageLabel + (stageTotal > 0 ? " " + std::to_string(stageIndex) + "/" + std::to_string(stageTotal) : std::string());
			ImGui::ProgressBar(fraction, ImVec2(-120.0f, 0.0f), overlay.c_str());
			ImGui::SameLine();
			if (ImGui::Button("Cancel", ImVec2(110.0f, 0.0f)))
				session.cancel();
		}
		else if (status == "Complete")
		{
			if (ImGui::Button("See results  >", ImVec2(220.0f, 30.0f)))
				app._step = Step::Results;
		}
		if (!error.empty())
			ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.45f, 1.0f), "%s", error.c_str());

		// Convergence: every measured candidate, and the running best, per step of the run.
		std::vector<Charts::Series> series;
		std::vector<std::vector<const SearchFlow::LiveRecord*>> seriesRecords;
		for (size_t arm = 0; arm < armCount; ++arm)
		{
			std::vector<const SearchFlow::LiveRecord*> records;
			Charts::Series points, best;
			points._label = arm < armLabels.size() ? armLabels[arm] : "";
			points._line = false;
			points._color = ArmColors[arm % 4];
			best._step = true;
			best._color = ArmColors[arm % 4];
			double running = std::numeric_limits<double>::max();
			for (const SearchFlow::LiveRecord& record : live)
			{
				if (record._arm != arm || !(record._latencyMs > 0.0))
					continue;
				points._x.push_back(static_cast<double>(record._order));
				points._y.push_back(record._latencyMs * 1000.0);
				records.push_back(&record);
				running = std::min(running, record._latencyMs * 1000.0);
				best._x.push_back(static_cast<double>(record._order));
				best._y.push_back(running);
			}
			if (!points._x.empty())
			{
				series.push_back(points);
				seriesRecords.push_back(records);
				series.push_back(best);
				seriesRecords.push_back(records);
			}
		}
		Charts::PlotOptions options;
		options._xLabel = "evaluation";
		options._yLabel = "mean query latency (us, log)";
		options._logY = true;
		Charts::plot("convergence", series, options, nullptr, [&](int si, int pi) {
			const std::vector<const SearchFlow::LiveRecord*>& records = seriesRecords[static_cast<size_t>(si)];
			if (pi < 0 || static_cast<size_t>(pi) >= records.size())
				return std::string();
			const SearchFlow::LiveRecord* record = records[static_cast<size_t>(pi)];
			return record->_schemaName + "\n" + Gui::formatMs(record->_latencyMs) + " per query, build " + Gui::formatMs(record->_buildMs);
		});

		// Live leaderboard.
		std::vector<const SearchFlow::LiveRecord*> ranked;
		// Search steps only: the held-out validation times a different query set, so its rows
		// are not comparable with the search's and belong on the Results step.
		for (const SearchFlow::LiveRecord& record : live)
			if (record._latencyMs > 0.0 && (record._arm >= armLabels.size() || armLabels[record._arm] != "Held-out validation"))
				ranked.push_back(&record);
		std::sort(ranked.begin(), ranked.end(), [](const auto* a, const auto* b) { return a->_latencyMs < b->_latencyMs; });
		std::set<std::string> seen;
		if (ImGui::BeginTable("leaders", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH | ImGuiTableFlags_SizingStretchProp))
		{
			ImGui::TableSetupColumn("#", ImGuiTableColumnFlags_WidthFixed, 30.0f);
			ImGui::TableSetupColumn("Schema");
			ImGui::TableSetupColumn("Latency", ImGuiTableColumnFlags_WidthFixed, 90.0f);
			ImGui::TableSetupColumn("Build", ImGuiTableColumnFlags_WidthFixed, 90.0f);
			ImGui::TableSetupColumn("Step", ImGuiTableColumnFlags_WidthFixed, 170.0f);
			ImGui::TableHeadersRow();
			int shown = 0;
			for (const SearchFlow::LiveRecord* record : ranked)
			{
				if (shown >= 10 || !seen.insert(record->_schemaName).second)
					continue;
				ImGui::TableNextRow();
				ImGui::TableNextColumn(); ImGui::Text("%d", ++shown);
				ImGui::TableNextColumn(); ImGui::TextUnformatted(record->_schemaName.c_str());
				ImGui::TableNextColumn(); ImGui::TextUnformatted(Gui::formatMs(record->_latencyMs).c_str());
				ImGui::TableNextColumn(); ImGui::TextUnformatted(Gui::formatMs(record->_buildMs).c_str());
				ImGui::TableNextColumn(); ImGui::TextUnformatted(record->_arm < armLabels.size() ? armLabels[record->_arm].c_str() : "");
			}
			ImGui::EndTable();
		}
		ImGui::Spacing();
		drawLog(app);
	}

	// ------------------------------------------------------------ step 5: results

	void loadResultsFolder(App& app, const std::string& folder)
	{
		app._shownResults.clear();
		app._shownValidation.clear();
		const struct { const char* file; const char* label; } known[] = {
			{ "ga.csv", "Genetic search" }, { "singles.csv", "Tuned single structures" }, { "fixed.csv", "Selected schemas" } };
		for (const auto& entry : known)
		{
			SearchFlow::Arm arm;
			arm._label = entry.label;
			arm._csvPath = (std::filesystem::path(folder) / entry.file).string();
			if (Gui::fileExists(arm._csvPath))
			{
				const std::vector<SearchFlow::ResultRow> rows = SearchFlow::loadResults(arm);
				app._shownResults.insert(app._shownResults.end(), rows.begin(), rows.end());
			}
		}
		SearchFlow::Arm heldOut;
		heldOut._label = "Held-out validation";
		heldOut._csvPath = (std::filesystem::path(folder) / "heldout.csv").string();
		if (Gui::fileExists(heldOut._csvPath))
			app._shownValidation = SearchFlow::loadResults(heldOut);
		app._loadedResultsDir = folder;
		app._selectedResult = -1;
	}

	const SearchFlow::ResultRow* bestOf(const std::vector<SearchFlow::ResultRow>& rows, const std::string& arm)
	{
		const SearchFlow::ResultRow* best = nullptr;
		for (const SearchFlow::ResultRow& row : rows)
			if (row._arm == arm && (!best || row._latencyMs < best->_latencyMs))
				best = &row;
		return best;
	}

	void drawSchemaTree(const std::string& path)
	{
		static std::string cachedPath;
		static SchemaConfig cachedSchema;
		static std::string cachedError;
		if (path != cachedPath)
		{
			cachedPath = path;
			cachedError.clear();
			try { cachedSchema = Config::loadSchemaConfig(path); }
			catch (const std::exception& exception) { cachedSchema = SchemaConfig(); cachedError = exception.what(); }
		}
		if (!cachedError.empty())
		{
			ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.45f, 1.0f), "%s", cachedError.c_str());
			return;
		}
		try
		{
			const SchemaConfig& schema = cachedSchema;
			size_t depth = 0;
			for (const SchemaLevelConfig& level : schema._levels)
			{
				const ImVec4 color = Gui::structureColor(level._typeName);
				ImGui::Indent(static_cast<float>(std::min<size_t>(depth, 6)) * 14.0f);
				ImGui::TextColored(color, "%s", level._typeName.c_str());
				ImGui::SameLine();
				ImGui::TextDisabled("levels %zu-%zu, leaf <= %zu%s%s", depth, depth + level._numLevels - 1, level._leafCapacity,
									level._axisPolicy.empty() ? "" : (", " + level._axisPolicy).c_str(),
									level._condition.empty() ? "" : ", conditional");
				ImGui::Unindent(static_cast<float>(std::min<size_t>(depth, 6)) * 14.0f);
				depth += level._numLevels;
			}
		}
		catch (const std::exception& exception)
		{
			ImGui::TextColored(ImVec4(1.0f, 0.5f, 0.45f, 1.0f), "%s", exception.what());
		}
	}

	void openInInspect(App& app, const std::string& schemaPath)
	{
		app._inspect._schemaPath = schemaPath;
		app._step = Step::Inspect;
	}

	void drawResultsStep(App& app)
	{
		if (!app._session._running)
		{
			std::lock_guard<std::mutex> lock(app._session._mutex);
			if (!app._session._results.empty() && app._loadedResultsDir.empty() && app._shownResults.empty())
			{
				app._shownResults = app._session._results;
				app._shownValidation = app._session._validation;
			}
		}

		sectionTitle("Results");
		if (ImGui::Button("Open a results folder..."))
		{
			const std::string folder = Gui::selectFolderDialog();
			if (!folder.empty())
				loadResultsFolder(app, folder);
		}
		Gui::helpMarker("Any folder holding ga.csv / singles.csv / heldout.csv, from the GUI or the CLI.");
		if (!app._loadedResultsDir.empty())
		{
			ImGui::SameLine();
			ImGui::TextDisabled("%s", app._loadedResultsDir.c_str());
		}
		if (app._shownResults.empty())
		{
			Gui::hint("No results yet: run a search (steps 1-4) or open a results folder.");
			return;
		}

		// The headline: does the searched schema beat the best single structure?
		const SearchFlow::ResultRow* searched = bestOf(app._shownResults, "Genetic search");
		const SearchFlow::ResultRow* single = bestOf(app._shownResults, "Tuned single structures");
		const auto verdict = [](const SearchFlow::ResultRow* a, const SearchFlow::ResultRow* b, const char* where) {
			if (!a || !b)
				return;
			const double speedup = b->_latencyMs / a->_latencyMs;
			ImGui::Text("%s: searched %s at %s vs best single %s at %s  ->  ", where, a->_schemaName.c_str(),
						Gui::formatMs(a->_latencyMs).c_str(), b->_schemaName.c_str(), Gui::formatMs(b->_latencyMs).c_str());
			ImGui::SameLine();
			ImGui::TextColored(speedup >= 1.0 ? ImVec4(0.55f, 0.83f, 0.72f, 1.0f) : ImVec4(1.0f, 0.62f, 0.45f, 1.0f),
							   speedup >= 1.0 ? "%.2fx faster" : "%.2fx slower", speedup >= 1.0 ? speedup : 1.0 / speedup);
		};
		ImGui::BeginChild("verdict", ImVec2(0.0f, app._shownValidation.empty() ? 50.0f : 110.0f), ImGuiChildFlags_Borders);
		verdict(searched, single, "Search queries");
		if (!app._shownValidation.empty())
		{
			// Held-out rows are keyed by schema name; pair the two arms' winners again there.
			const SearchFlow::ResultRow* searchedHeld = nullptr;
			const SearchFlow::ResultRow* singleHeld = nullptr;
			for (const SearchFlow::ResultRow& row : app._shownValidation)
			{
				if (searched && row._schemaName == searched->_schemaName) searchedHeld = &row;
				if (single && row._schemaName == single->_schemaName) singleHeld = &row;
			}
			verdict(searchedHeld, singleHeld, "Held-out queries");
			Gui::hint("Held-out queries were never seen by the search: this is the number to trust.");
		}
		ImGui::EndChild();

		// Latency vs build scatter, one colour per step.
		std::vector<Charts::Series> series;
		std::vector<std::string> arms;
		for (const SearchFlow::ResultRow& row : app._shownResults)
			if (std::find(arms.begin(), arms.end(), row._arm) == arms.end())
				arms.push_back(row._arm);
		std::vector<std::vector<const SearchFlow::ResultRow*>> rowsBySeries;
		for (size_t a = 0; a < arms.size(); ++a)
		{
			Charts::Series s;
			s._label = arms[a];
			s._line = false;
			s._color = ArmColors[a % 4];
			rowsBySeries.emplace_back();
			for (const SearchFlow::ResultRow& row : app._shownResults)
			{
				if (row._arm != arms[a])
					continue;
				s._x.push_back(row._buildMs / 1000.0);
				s._y.push_back(row._latencyMs * 1000.0);
				rowsBySeries.back().push_back(&row);
			}
			series.push_back(std::move(s));
		}
		Charts::PlotOptions options;
		options._xLabel = "build time (s)";
		options._yLabel = "mean query latency (us, log)";
		options._logY = true;
		options._height = 240.0f;
		Charts::plot("tradeoff", series, options, nullptr, [&](int si, int pi) {
			const SearchFlow::ResultRow* row = rowsBySeries[static_cast<size_t>(si)][static_cast<size_t>(pi)];
			return row->_schemaName + "\n" + Gui::formatMs(row->_latencyMs) + " per query, build " + Gui::formatMs(row->_buildMs);
		});

		// Full table.
		const ImGuiTableFlags flags = ImGuiTableFlags_Sortable | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersInnerH |
			ImGuiTableFlags_ScrollY | ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
		if (ImGui::BeginTable("results", 7, flags, ImVec2(0.0f, 260.0f)))
		{
			ImGui::TableSetupScrollFreeze(0, 1);
			ImGui::TableSetupColumn("Schema", ImGuiTableColumnFlags_WidthStretch);
			ImGui::TableSetupColumn("Step");
			ImGui::TableSetupColumn("Latency", ImGuiTableColumnFlags_DefaultSort);
			ImGui::TableSetupColumn("Build");
			ImGui::TableSetupColumn("Memory");
			ImGui::TableSetupColumn("Nodes/query");
			ImGui::TableSetupColumn("Points/query");
			ImGui::TableHeadersRow();
			if (ImGuiTableSortSpecs* sort = ImGui::TableGetSortSpecs(); sort && sort->SpecsDirty && sort->SpecsCount > 0)
			{
				const ImGuiTableColumnSortSpecs spec = sort->Specs[0];
				std::sort(app._shownResults.begin(), app._shownResults.end(), [spec](const SearchFlow::ResultRow& a, const SearchFlow::ResultRow& b) {
					double x = 0.0, y = 0.0;
					switch (spec.ColumnIndex)
					{
					case 0: return spec.SortDirection == ImGuiSortDirection_Ascending ? a._schemaName < b._schemaName : a._schemaName > b._schemaName;
					case 1: return spec.SortDirection == ImGuiSortDirection_Ascending ? a._arm < b._arm : a._arm > b._arm;
					case 2: x = a._latencyMs; y = b._latencyMs; break;
					case 3: x = a._buildMs; y = b._buildMs; break;
					case 4: x = a._memoryBytes; y = b._memoryBytes; break;
					case 5: x = a._visited; y = b._visited; break;
					default: x = a._tested; y = b._tested; break;
					}
					return spec.SortDirection == ImGuiSortDirection_Ascending ? x < y : x > y;
				});
				sort->SpecsDirty = false;
				app._selectedResult = -1;
			}
			for (size_t i = 0; i < app._shownResults.size(); ++i)
			{
				const SearchFlow::ResultRow& row = app._shownResults[i];
				ImGui::TableNextRow();
				ImGui::TableNextColumn();
				if (ImGui::Selectable((row._schemaName + "##" + std::to_string(i)).c_str(), app._selectedResult == static_cast<int>(i), ImGuiSelectableFlags_SpanAllColumns))
					app._selectedResult = static_cast<int>(i);
				ImGui::TableNextColumn(); ImGui::TextUnformatted(row._arm.c_str());
				ImGui::TableNextColumn(); ImGui::TextUnformatted(Gui::formatMs(row._latencyMs).c_str());
				ImGui::TableNextColumn(); ImGui::TextUnformatted(Gui::formatMs(row._buildMs).c_str());
				ImGui::TableNextColumn(); ImGui::TextUnformatted(Gui::formatBytes(row._memoryBytes).c_str());
				ImGui::TableNextColumn(); ImGui::Text("%.1f", row._visited);
				ImGui::TableNextColumn(); ImGui::Text("%.1f", row._tested);
			}
			ImGui::EndTable();
		}

		if (app._selectedResult >= 0 && app._selectedResult < static_cast<int>(app._shownResults.size()))
		{
			const SearchFlow::ResultRow& row = app._shownResults[static_cast<size_t>(app._selectedResult)];
			ImGui::Spacing();
			ImGui::Text("%s", row._schemaName.c_str());
			ImGui::SameLine();
			if (ImGui::Button("Inspect in 3D  >"))
				openInInspect(app, row._schemaPath);
			ImGui::SameLine();
			if (ImGui::Button("Copy schema path"))
				Gui::copyToClipboard(row._schemaPath);
			drawSchemaTree(row._schemaPath);
		}
	}

	// ------------------------------------------------------------ step 6: inspect

	void startInspectBuild(App& app)
	{
		Inspect& in = app._inspect;
		if (in._busy || in._schemaPath.empty() || !Gui::fileExists(app._settings._inputPath))
			return;
		if (in._worker.joinable())
			in._worker.join();
		in._busy = true;
		in._ready = false;
		const std::string cloudPath = app._settings._inputPath;
		const std::string schemaPath = in._schemaPath;
		std::shared_ptr<PointCloud> reuse = (in._cloudPath == cloudPath) ? in._cloud : nullptr;
		in._worker = std::thread([&in, cloudPath, schemaPath, reuse]() {
			try
			{
				std::shared_ptr<PointCloud> cloud = reuse;
				double loadMs = 0.0;
				if (!cloud)
				{
					{ std::lock_guard<std::mutex> lock(in._mutex); in._status = "Loading cloud..."; }
					const auto t0 = std::chrono::steady_clock::now();
					cloud = std::make_shared<PointCloud>(PointCloud::load(cloudPath));
					loadMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
				}
				{ std::lock_guard<std::mutex> lock(in._mutex); in._status = "Building index..."; }
				SchemaConfig schema = Config::loadSchemaConfig(schemaPath);
				auto index = std::make_shared<PointSpatialIndex>();
				const auto t1 = std::chrono::steady_clock::now();
				index->build(*cloud, schema);
				const double buildMs = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
				std::lock_guard<std::mutex> lock(in._mutex);
				in._cloud = cloud;
				in._cloudPath = cloudPath;
				in._index = index;
				in._schema = schema;
				if (loadMs > 0.0)
					in._loadMs = loadMs;
				in._buildMs = buildMs;
				in._stats = index->stats();
				in._status = "Ready";
				in._ready = true;
			}
			catch (const std::exception& exception)
			{
				std::lock_guard<std::mutex> lock(in._mutex);
				in._status = std::string("Failed: ") + exception.what();
			}
			in._busy = false;
		});
	}

	void frameCamera(Inspect& in)
	{
		const AABB bounds = in._cloud->bounds();
		in._camera._target = bounds.center();
		in._sceneRadius = std::max(glm::length(bounds.size()) * 0.5f, 1e-3f);
		in._camera._distance = in._sceneRadius * 2.2f;
		in._radius = in._sceneRadius * 0.01f;
		in._halfExtent = in._sceneRadius * 0.02f;
	}

	void pickRandomCenter(Inspect& in)
	{
		if (!in._cloud || in._cloud->empty())
			return;
		in._seed = in._seed * 6364136223846793005ull + 1442695040888963407ull;
		in._center = in._cloud->points()[(in._seed >> 33) % in._cloud->size()].position;
	}

	void drawInspectStep(App& app)
	{
		Inspect& in = app._inspect;
		if (in._ready.exchange(false))
		{
			// Fresh index: rebuild the scene and frame it.
			frameCamera(in);
			pickRandomCenter(in);
			in._sceneDirty = true;
			in._overlay = Viewer::QueryOverlay();
			in._overlayDirty = true;
		}

		const float controlsWidth = 340.0f;
		ImGui::BeginChild("inspect-controls", ImVec2(controlsWidth, 0.0f), ImGuiChildFlags_Borders);
		sectionTitle("Index");
		ImGui::TextWrapped("Cloud: %s", stemOf(app._settings._inputPath).c_str());
		ImGui::SetNextItemWidth(-80.0f);
		ImGui::InputText("##schema", &in._schemaPath);
		ImGui::SameLine();
		if (ImGui::Button("File..."))
		{
			const std::string picked = Gui::openFileDialog("Schemas\0*.json\0", in._schemaPath);
			if (!picked.empty())
				in._schemaPath = picked;
		}
		if (ImGui::BeginCombo("##presets", "or pick a shipped schema"))
		{
			for (const std::string& path : app._schemaChoices)
				if (ImGui::Selectable(stemOf(path).c_str()))
					in._schemaPath = path;
			ImGui::EndCombo();
		}
		ImGui::BeginDisabled(in._busy || in._schemaPath.empty());
		if (ImGui::Button(in._busy ? "Working..." : "Build and show", ImVec2(-1.0f, 30.0f)))
			startInspectBuild(app);
		ImGui::EndDisabled();
		{
			std::lock_guard<std::mutex> lock(in._mutex);
			if (!in._status.empty())
				ImGui::TextDisabled("%s", in._status.c_str());
		}

		if (in._index)
		{
			ImGui::Text("%s nodes, %s leaves, depth %zu", Gui::formatCount(static_cast<double>(in._stats._numNodes)).c_str(),
						Gui::formatCount(static_cast<double>(in._stats._numLeaves)).c_str(), in._stats._maxDepth);
			ImGui::Text("Build %s, memory %s", Gui::formatMs(in._buildMs).c_str(), Gui::formatBytes(static_cast<double>(in._index->memoryFootprintBytes())).c_str());
			drawSchemaTree(in._schemaPath);

			ImGui::Spacing();
			sectionTitle("Display");
			int pointMode = static_cast<int>(in._builder._pointColoring);
			if (ImGui::Combo("Points", &pointMode, "Height\0Owning structure\0"))
			{
				in._builder._pointColoring = static_cast<Viewer::PointColoring>(pointMode);
				in._sceneDirty = true;
			}
			int boxMode = static_cast<int>(in._builder._boxColoring);
			if (ImGui::Combo("Boxes", &boxMode, "Structure\0Depth\0"))
			{
				in._builder._boxColoring = static_cast<Viewer::BoxColoring>(boxMode);
				in._sceneDirty = true;
			}
			const int maxDepth = static_cast<int>(in._stats._maxDepth);
			in._sceneDirty |= ImGui::SliderInt("From depth", &in._builder._minDepth, 0, maxDepth);
			in._sceneDirty |= ImGui::SliderInt("To depth", &in._builder._maxDepth, 0, maxDepth);
			in._sceneDirty |= ImGui::Checkbox("Leaves only", &in._builder._leavesOnly);
			int maxBoxes = static_cast<int>(in._builder._maxBoxes);
			if (ImGui::SliderInt("Box budget", &maxBoxes, 0, 200000, "%d", ImGuiSliderFlags_Logarithmic))
			{
				in._builder._maxBoxes = static_cast<size_t>(maxBoxes);
				in._sceneDirty = true;
			}
			ImGui::SliderFloat("Point size", &in._pointSize, 1.0f, 6.0f);
			ImGui::TextDisabled("%zu boxes shown", in._boxesShown);

			// Legend: structures present, coloured as drawn.
			for (const std::string& name : in._index->structureBreakdownNames())
			{
				ImGui::ColorButton(("##" + name).c_str(), Gui::structureColor(name), ImGuiColorEditFlags_NoTooltip, ImVec2(12.0f, 12.0f));
				ImGui::SameLine();
				ImGui::TextUnformatted(name.c_str());
			}

			ImGui::Spacing();
			sectionTitle("Probe a query");
			int kind = static_cast<int>(in._queryKind);
			ImGui::RadioButton("Radius", &kind, 0); ImGui::SameLine();
			ImGui::RadioButton("kNN", &kind, 1); ImGui::SameLine();
			ImGui::RadioButton("Box", &kind, 2);
			in._queryKind = static_cast<Viewer::QueryKind>(kind);
			if (in._queryKind == Viewer::QueryKind::Radius)
				ImGui::SliderFloat("Radius", &in._radius, in._sceneRadius * 1e-4f, in._sceneRadius * 0.2f, "%.4f", ImGuiSliderFlags_Logarithmic);
			else if (in._queryKind == Viewer::QueryKind::Knn)
				ImGui::SliderInt("k", &in._k, 1, 256);
			else
				ImGui::SliderFloat("Half extent", &in._halfExtent, in._sceneRadius * 1e-4f, in._sceneRadius * 0.2f, "%.4f", ImGuiSliderFlags_Logarithmic);
			if (ImGui::Button("Random point"))
				pickRandomCenter(in);

			// Replay a query from the selected trace (world coordinates -> cloud frame).
			if (app._settings._useTrace && Gui::fileExists(app._settings._tracePath))
			{
				if (in._tracePath != app._settings._tracePath)
				{
					try { in._trace = Experiments::loadQueryTrace(app._settings._tracePath); }
					catch (...) { in._trace.clear(); }
					in._tracePath = app._settings._tracePath;
				}
				if (!in._trace.empty())
				{
					if (ImGui::SliderInt("Trace query #", &in._traceQuery, 0, static_cast<int>(in._trace.size()) - 1))
					{
						const Experiments::TraceQuery& q = in._trace[static_cast<size_t>(in._traceQuery)];
						in._center = in._cloud->toLocalPosition(q._center);
						if (q._kind == Experiments::TraceQuery::Kind::Knn) { in._queryKind = Viewer::QueryKind::Knn; in._k = static_cast<int>(q._k); }
						else if (q._kind == Experiments::TraceQuery::Kind::Radius) { in._queryKind = Viewer::QueryKind::Radius; in._radius = q._radius; }
						in._overlayDirty = true;
					}
				}
			}
			if (ImGui::Button("Run query", ImVec2(-1.0f, 28.0f)))
				in._overlayDirty = true;
			ImGui::BeginDisabled(!in._overlay._valid);
			if (ImGui::Button("Focus on query", ImVec2(-1.0f, 0.0f)))
			{
				// Frame the region the query touched: its radius, k-th distance or box.
				const float reach = in._overlay._kind == Viewer::QueryKind::Range
					? glm::length(in._overlay._box.size()) * 0.5f
					: (in._overlay._kind == Viewer::QueryKind::Knn ? static_cast<float>(in._overlay._kthDistance) : in._overlay._radius);
				in._camera._target = in._overlay._center;
				in._camera._distance = std::max(reach * 6.0f, in._sceneRadius * 1e-3f);
			}
			ImGui::EndDisabled();

			if (in._overlay._valid)
			{
				const PointSpatialIndex::QueryStats& st = in._overlay._stats;
				ImGui::Text("%zu returned, %zu nodes entered, %zu points tested", st._returnedPoints, st._visitedNodes, st._testedPoints);
				ImGui::Text("%s for this query (instrumented)", Gui::formatMs(st._elapsedMs).c_str());
				std::vector<Charts::Bar> bars;
				for (const auto& [name, count] : st._breakdown.namedCounts(st._breakdown._visitedByStructure))
					bars.push_back({ name, static_cast<double>(count), ImGui::ColorConvertFloat4ToU32(Gui::structureColor(name)) });
				if (!bars.empty())
				{
					ImGui::TextDisabled("Nodes entered, by structure");
					Charts::bars("entered", bars, "nodes");
				}
			}
		}
		ImGui::EndChild();

		ImGui::SameLine();
		ImGui::BeginChild("inspect-view", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders, ImGuiWindowFlags_NoScrollbar | ImGuiWindowFlags_NoScrollWithMouse);
		if (!in._index || !in._cloud)
		{
			Gui::hint("Pick a schema (from Results, a file or the presets) and press \"Build and show\". "
					  "The index is built on the cloud chosen in step 1.");
			ImGui::EndChild();
			return;
		}

		if (in._sceneDirty)
		{
			in._renderer.setPoints(in._builder.points(*in._cloud, *in._index));
			in._renderer.setBoxes(in._builder.boxes(*in._index, &in._boxesShown));
			in._sceneDirty = false;
		}
		if (in._overlayDirty)
		{
			in._overlay = Viewer::runQuery(*in._index, in._queryKind, in._center, in._radius, static_cast<size_t>(in._k), glm::vec3(in._halfExtent));
			std::vector<Viewer::ColoredVertex> points, lines;
			Viewer::overlayGeometry(*in._cloud, *in._index, in._overlay, points, lines);
			in._renderer.setHighlights(points, lines);
			in._overlayDirty = false;
		}

		const ImVec2 size = ImGui::GetContentRegionAvail();
		const unsigned int texture = in._renderer.render(static_cast<int>(size.x), static_cast<int>(size.y), in._camera, in._sceneRadius, in._pointSize);
		const ImVec2 origin = ImGui::GetCursorScreenPos();
		ImGui::Image(static_cast<ImTextureID>(static_cast<intptr_t>(texture)), size, ImVec2(0.0f, 1.0f), ImVec2(1.0f, 0.0f));
		if (ImGui::IsItemHovered())
		{
			ImGuiIO& io = ImGui::GetIO();
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Left))
			{
				in._camera._yaw -= io.MouseDelta.x * 0.008f;
				in._camera._pitch = std::clamp(in._camera._pitch + io.MouseDelta.y * 0.008f, -1.55f, 1.55f);
			}
			if (ImGui::IsMouseDragging(ImGuiMouseButton_Right))
			{
				const glm::mat4 view = in._camera.view();
				const glm::vec3 right(view[0][0], view[1][0], view[2][0]);
				const glm::vec3 up(view[0][1], view[1][1], view[2][1]);
				const float scale = in._camera._distance * 0.0015f;
				in._camera._target += (-io.MouseDelta.x * right + io.MouseDelta.y * up) * scale;
			}
			if (io.MouseWheel != 0.0f)
				in._camera._distance = std::max(in._sceneRadius * 1e-4f, in._camera._distance * std::pow(0.88f, io.MouseWheel));
		}
		ImGui::GetWindowDrawList()->AddText(ImVec2(origin.x + 10.0f, origin.y + 8.0f), IM_COL32(170, 178, 190, 255),
											"left drag: orbit   right drag: pan   wheel: zoom   |   red: returned points   magenta: entered leaves");
		ImGui::EndChild();
	}


	// ------------------------------------------------------------ capture mode
	//
	// MDS_GUI_CAPTURE=<dir> renders every step once into an offscreen framebuffer, writes
	// <dir>/step<N>.png and exits. It exists so the interface can be checked (and documented)
	// without a visible desktop: a locked or remote session has no screen to grab.
	// MDS_GUI_CAPTURE_SCHEMA=<schema.json> also builds that schema in the Inspect step and probes
	// one query; MDS_GUI_CAPTURE_RESULTS=<folder> loads a results folder into the Results step.

	uint32_t crc32(const uint8_t* data, size_t size)
	{
		uint32_t crc = 0xFFFFFFFFu;
		for (size_t i = 0; i < size; ++i)
		{
			crc ^= data[i];
			for (int k = 0; k < 8; ++k)
				crc = (crc >> 1) ^ (0xEDB88320u & (0u - (crc & 1u)));
		}
		return ~crc;
	}

	void appendBigEndian(std::vector<uint8_t>& out, uint32_t value)
	{
		for (int shift = 24; shift >= 0; shift -= 8)
			out.push_back(static_cast<uint8_t>(value >> shift));
	}

	// Minimal PNG: RGBA8, zlib "stored" (uncompressed) blocks. Large files, no dependency.
	bool writePng(const std::string& path, int width, int height, const std::vector<uint8_t>& rgbaTopDown)
	{
		std::vector<uint8_t> raw;
		raw.reserve(static_cast<size_t>(height) * (static_cast<size_t>(width) * 4 + 1));
		for (int y = 0; y < height; ++y)
		{
			raw.push_back(0);
			const auto row = rgbaTopDown.begin() + static_cast<ptrdiff_t>(y) * width * 4;
			raw.insert(raw.end(), row, row + static_cast<ptrdiff_t>(width) * 4);
		}
		std::vector<uint8_t> z = { 0x78, 0x01 };
		uint32_t a = 1, b = 0;
		for (const uint8_t byte : raw)
		{
			a = (a + byte) % 65521u;
			b = (b + a) % 65521u;
		}
		size_t offset = 0;
		do
		{
			const size_t block = std::min<size_t>(65535, raw.size() - offset);
			const bool last = offset + block >= raw.size();
			z.push_back(last ? 1 : 0);
			z.push_back(static_cast<uint8_t>(block & 0xFF));
			z.push_back(static_cast<uint8_t>(block >> 8));
			z.push_back(static_cast<uint8_t>(~block & 0xFF));
			z.push_back(static_cast<uint8_t>((~block >> 8) & 0xFF));
			z.insert(z.end(), raw.begin() + static_cast<ptrdiff_t>(offset), raw.begin() + static_cast<ptrdiff_t>(offset + block));
			offset += block;
		} while (offset < raw.size());
		appendBigEndian(z, (b << 16) | a);

		std::ofstream out(path, std::ios::binary);
		if (!out)
			return false;
		const uint8_t signature[8] = { 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
		out.write(reinterpret_cast<const char*>(signature), 8);
		const auto chunk = [&out](const char* type, const std::vector<uint8_t>& data) {
			std::vector<uint8_t> bytes;
			appendBigEndian(bytes, static_cast<uint32_t>(data.size()));
			std::vector<uint8_t> typed(type, type + 4);
			typed.insert(typed.end(), data.begin(), data.end());
			bytes.insert(bytes.end(), typed.begin(), typed.end());
			appendBigEndian(bytes, crc32(typed.data(), typed.size()));
			out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		};
		std::vector<uint8_t> header;
		appendBigEndian(header, static_cast<uint32_t>(width));
		appendBigEndian(header, static_cast<uint32_t>(height));
		header.insert(header.end(), { 8, 6, 0, 0, 0 });
		chunk("IHDR", header);
		chunk("IDAT", z);
		chunk("IEND", {});
		return static_cast<bool>(out);
	}

	struct Capture
	{
		std::string	_dir;
		std::string	_schema;
		bool		_runSearch = false;
		bool		_searchStarted = false;
		int			_step = 0;
		int			_frames = 0;
		double		_waitStart = 0.0;
		unsigned int	_fbo = 0, _texture = 0;
		int			_width = 0, _height = 0;

		bool active() const { return !_dir.empty(); }

		void target(int width, int height)
		{
			if (_fbo && width == _width && height == _height)
				return;
			if (!_fbo)
			{
				glGenFramebuffers(1, &_fbo);
				glGenTextures(1, &_texture);
			}
			glBindTexture(GL_TEXTURE_2D, _texture);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
			glBindFramebuffer(GL_FRAMEBUFFER, _fbo);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _texture, 0);
			glBindFramebuffer(GL_FRAMEBUFFER, 0);
			_width = width;
			_height = height;
		}

		~Capture()
		{
			if (_fbo) glDeleteFramebuffers(1, &_fbo);
			if (_texture) glDeleteTextures(1, &_texture);
		}
	};

	// ------------------------------------------------------------ shell

	bool stepDone(const App& app, Step step)
	{
		const SearchFlow::Settings& s = app._settings;
		switch (step)
		{
		case Step::Data: return Gui::fileExists(s._inputPath);
		case Step::Workload: return s._useTrace ? Gui::fileExists(s._tracePath) : Gui::fileExists(Gui::resolvePath(s._workloadPath));
		case Step::Search: return true;
		case Step::Run: return !app._session._running && !app._session._results.empty();
		case Step::Results: return !app._shownResults.empty();
		case Step::Inspect: return app._inspect._index != nullptr;
		}
		return false;
	}

	void drawInterface(App& app)
	{
		const ImGuiViewport* viewport = ImGui::GetMainViewport();
		ImGui::SetNextWindowPos(viewport->WorkPos);
		ImGui::SetNextWindowSize(viewport->WorkSize);
		ImGui::Begin("MultiDataStructure", nullptr, ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_NoMove |
					 ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoBringToFrontOnFocus);

		ImGui::TextUnformatted("MultiDataStructure");
		ImGui::SameLine();
		ImGui::TextDisabled("workload-tuned spatial indexes for point clouds");
		if (app._session._running)
		{
			ImGui::SameLine(ImGui::GetWindowWidth() - 330.0f);
			std::lock_guard<std::mutex> lock(app._session._mutex);
			ImGui::TextColored(ImVec4(0.95f, 0.78f, 0.32f, 1.0f), "running: %s (%.0f s)", app._session._status.c_str(), app._session._elapsedSeconds);
		}
		ImGui::Separator();

		ImGui::BeginChild("steps", ImVec2(190.0f, 0.0f), ImGuiChildFlags_Borders);
		for (int i = 0; i < 6; ++i)
		{
			const Step step = static_cast<Step>(i);
			const bool done = stepDone(app, step);
			ImGui::PushStyleColor(ImGuiCol_Text, done ? ImVec4(0.55f, 0.83f, 0.72f, 1.0f) : ImVec4(0.80f, 0.84f, 0.90f, 1.0f));
			if (ImGui::Selectable((std::string(done ? "* " : "  ") + StepNames[i]).c_str(), app._step == step, 0, ImVec2(0.0f, 30.0f)))
				app._step = step;
			ImGui::PopStyleColor();
		}
		ImGui::Spacing();
		Gui::hint("Steps 1-3 describe the run, 4 runs it, 5 compares the outcome, 6 shows any index in 3D. "
				  "Settings are remembered between sessions.");
		ImGui::EndChild();
		ImGui::SameLine();

		ImGui::BeginChild("main", ImVec2(0.0f, 0.0f), ImGuiChildFlags_Borders,
						  app._step == Step::Inspect ? ImGuiWindowFlags_NoScrollbar : 0);
		switch (app._step)
		{
		case Step::Data: drawDataStep(app); break;
		case Step::Workload: drawWorkloadStep(app); break;
		case Step::Search: drawSearchStep(app); break;
		case Step::Run: drawRunStep(app); break;
		case Step::Results: drawResultsStep(app); break;
		case Step::Inspect: drawInspectStep(app); break;
		}
		ImGui::EndChild();
		ImGui::End();
	}
}

int OptimizerGui::run(const AppConfig* config)
{
	if (!glfwInit())
		throw std::runtime_error("Unable to initialize GLFW.");

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	GLFWwindow* window = glfwCreateWindow(1500, 940, "MultiDataStructure", nullptr, nullptr);
	if (!window)
	{
		glfwTerminate();
		throw std::runtime_error("Unable to create the window.");
	}
	glfwMakeContextCurrent(window);
	glfwSwapInterval(1);
	glewExperimental = GL_TRUE;
	if (glewInit() != GLEW_OK)
	{
		glfwDestroyWindow(window);
		glfwTerminate();
		throw std::runtime_error("Unable to initialize GLEW.");
	}

	IMGUI_CHECKVERSION();
	ImGui::CreateContext();
	ImGui::GetIO().ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
	ImGui::GetIO().IniFilename = nullptr;
	Gui::applyTheme();
	ImGui_ImplGlfw_InitForOpenGL(window, true);
	ImGui_ImplOpenGL3_Init("#version 330");

	{
		App app;
		loadSettings(app._settings);
		if (config)
		{
			const auto& search = config->_schemaSearchOptions;
			if (!search._inputPaths.empty() && !search._inputPaths.front().empty())
				app._settings._inputPath = search._inputPaths.front();
			if (!search._inputTracePath.empty())
			{
				app._settings._useTrace = true;
				app._settings._tracePath = search._inputTracePath;
			}
		}
		app._cells = discoverCells();
		app._workloads = listJson("configs/workloads");
		app._schemaChoices = listJson("configs/schemas");
		for (const std::string& extra : listJson("configs/schemas/equivalence_set"))
			app._schemaChoices.push_back(extra);

		Capture capture;
		if (const char* dir = std::getenv("MDS_GUI_CAPTURE"))
		{
			capture._dir = dir;
			std::filesystem::create_directories(capture._dir);
			if (const char* schema = std::getenv("MDS_GUI_CAPTURE_SCHEMA"))
				capture._schema = schema;
			capture._runSearch = std::getenv("MDS_GUI_CAPTURE_RUN") != nullptr;
			if (const char* results = std::getenv("MDS_GUI_CAPTURE_RESULTS"))
				loadResultsFolder(app, results);
		}

		while (!glfwWindowShouldClose(window))
		{
			glfwPollEvents();
			app._session.poll();

			if (capture.active())
			{
				app._step = static_cast<Step>(capture._step);
				// MDS_GUI_CAPTURE_RUN: run the configured search for real (Quick budget) and capture
				// the Run and Results steps after it finishes - an end-to-end check of the GUI path.
				if (app._step == Step::Run && capture._runSearch && !capture._searchStarted)
				{
					app._settings._budget = SearchFlow::Budget::Quick;
					app._settings.applyBudget();
					app._session.start(app._settings);
					capture._searchStarted = true;
				}
				if (app._step == Step::Inspect && !capture._schema.empty() && !app._inspect._index && !app._inspect._busy)
				{
					app._inspect._schemaPath = capture._schema;
					startInspectBuild(app);
					capture._waitStart = ImGui::GetTime();
				}
			}

			ImGui_ImplOpenGL3_NewFrame();
			ImGui_ImplGlfw_NewFrame();
			ImGui::NewFrame();
			drawInterface(app);
			ImGui::Render();

			int width = 0, height = 0;
			glfwGetFramebufferSize(window, &width, &height);
			if (capture.active() && width > 0 && height > 0)
			{
				// Wait for the Inspect build (bounded), then let the layout settle a few frames.
				const bool waiting = (app._step == Step::Inspect && (app._inspect._busy || app._inspect._ready) &&
					ImGui::GetTime() - capture._waitStart < 300.0) || (app._step == Step::Run && app._session._running);
				if (!waiting && ++capture._frames >= 8)
				{
					capture.target(width, height);
					glBindFramebuffer(GL_FRAMEBUFFER, capture._fbo);
					glViewport(0, 0, width, height);
					glClearColor(0.055f, 0.064f, 0.075f, 1.0f);
					glClear(GL_COLOR_BUFFER_BIT);
					ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
					std::vector<uint8_t> pixels(static_cast<size_t>(width) * height * 4);
					glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
					glBindFramebuffer(GL_FRAMEBUFFER, 0);
					std::vector<uint8_t> flipped(pixels.size());
					for (int y = 0; y < height; ++y)
						std::memcpy(&flipped[static_cast<size_t>(y) * width * 4], &pixels[static_cast<size_t>(height - 1 - y) * width * 4], static_cast<size_t>(width) * 4);
					writePng(capture._dir + "/step" + std::to_string(capture._step + 1) + ".png", width, height, flipped);
					capture._frames = 0;
					if (++capture._step >= 6)
						glfwSetWindowShouldClose(window, GLFW_TRUE);
				}
			}

			glViewport(0, 0, width, height);
			glClearColor(0.055f, 0.064f, 0.075f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
			ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
			glfwSwapBuffers(window);
		}

		// Closing the window is allowed mid-run: the search is cancelled and joined first.
		saveSettings(app._settings);
		app._session.cancel();
	}

	ImGui_ImplOpenGL3_Shutdown();
	ImGui_ImplGlfw_Shutdown();
	ImGui::DestroyContext();
	glfwDestroyWindow(window);
	glfwTerminate();
	return 0;
}

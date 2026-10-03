#include "GuiCommon.h"

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <commdlg.h>
#include <shlobj.h>
#endif

namespace
{
	bool pathExists(const std::filesystem::path& value)
	{
		std::error_code error;
		return !value.empty() && std::filesystem::exists(value, error);
	}

	void addAncestors(std::vector<std::filesystem::path>& roots, std::filesystem::path start)
	{
		if (start.empty())
			return;
		std::error_code error;
		start = std::filesystem::absolute(start, error);
		if (error)
			return;
		if (start.has_extension())
			start = start.parent_path();
		for (;;)
		{
			const std::filesystem::path normalized = start.lexically_normal();
			if (std::find(roots.begin(), roots.end(), normalized) == roots.end())
				roots.push_back(normalized);
			if (!start.has_parent_path() || start == start.parent_path())
				break;
			start = start.parent_path();
		}
	}

	std::filesystem::path executablePath()
	{
#ifdef _WIN32
		char* programPath = nullptr;
		if (_get_pgmptr(&programPath) == 0 && programPath)
			return std::filesystem::path(programPath);
#endif
		return {};
	}

	std::vector<std::filesystem::path> searchRoots()
	{
		std::vector<std::filesystem::path> roots;
		addAncestors(roots, std::filesystem::current_path());
		addAncestors(roots, executablePath());
		return roots;
	}

	std::string normalizedName(std::string value)
	{
		value.erase(std::remove_if(value.begin(), value.end(), [](unsigned char c) {
			return std::isspace(c) || c == '_' || c == '-';
		}), value.end());
		std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
			return static_cast<char>(std::tolower(c));
		});
		return value;
	}

	bool oneOf(const std::string& value, std::initializer_list<const char*> names)
	{
		for (const char* name : names)
		{
			if (value == name)
				return true;
		}
		return false;
	}
}

std::filesystem::path Gui::projectRoot()
{
	for (const std::filesystem::path& root : searchRoots())
	{
		if (pathExists(root / "configs" / "schemas") && pathExists(root / "MultiDataStructure.sln"))
			return root;
	}
	return {};
}

std::string Gui::projectPath(const std::string& relative)
{
	const std::filesystem::path root = projectRoot();
	return root.empty() ? relative : (root / relative).lexically_normal().string();
}

std::string Gui::resolvePath(const std::string& configured)
{
	if (configured.empty())
		return configured;
	const std::filesystem::path path(configured);
	if (pathExists(path))
		return path.lexically_normal().string();
	if (path.is_absolute())
		return configured;
	for (const std::filesystem::path& root : searchRoots())
	{
		const std::filesystem::path candidate = (root / path).lexically_normal();
		if (pathExists(candidate))
			return candidate.string();
	}
	return configured;
}

bool Gui::fileExists(const std::string& path)
{
	std::error_code error;
	return !path.empty() && std::filesystem::is_regular_file(path, error);
}

std::filesystem::path Gui::executableDirectory()
{
	const std::filesystem::path exe = executablePath();
	return exe.empty() ? std::filesystem::current_path() : exe.parent_path();
}

std::string Gui::openFileDialog(const char* filter, const std::string& initialPath)
{
#ifdef _WIN32
	char buffer[MAX_PATH] = {};
	std::string initialDir;
	if (!initialPath.empty())
		initialDir = std::filesystem::path(initialPath).parent_path().string();
	OPENFILENAMEA dialog = {};
	dialog.lStructSize = sizeof(dialog);
	dialog.lpstrFilter = filter;
	dialog.lpstrFile = buffer;
	dialog.nMaxFile = MAX_PATH;
	dialog.lpstrInitialDir = initialDir.empty() ? nullptr : initialDir.c_str();
	dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
	if (GetOpenFileNameA(&dialog))
		return std::string(buffer);
#else
	(void)filter;
	(void)initialPath;
#endif
	return {};
}

std::string Gui::selectFolderDialog(const std::string& initialPath)
{
#ifdef _WIN32
	(void)initialPath;
	BROWSEINFOA info = {};
	info.lpszTitle = "Select output folder";
	info.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;
	PIDLIST_ABSOLUTE item = SHBrowseForFolderA(&info);
	if (!item)
		return {};
	char buffer[MAX_PATH] = {};
	const bool ok = SHGetPathFromIDListA(item, buffer) != FALSE;
	CoTaskMemFree(item);
	return ok ? std::string(buffer) : std::string();
#else
	(void)initialPath;
	return {};
#endif
}

void Gui::copyToClipboard(const std::string& text)
{
	ImGui::SetClipboardText(text.c_str());
}

size_t Gui::CsvTable::column(const std::string& name) const
{
	const auto found = std::find(_header.begin(), _header.end(), name);
	return found == _header.end() ? std::numeric_limits<size_t>::max() : static_cast<size_t>(found - _header.begin());
}

std::string Gui::CsvTable::value(size_t row, const std::string& name) const
{
	const size_t index = column(name);
	if (row >= _rows.size() || index >= _rows[row].size())
		return {};
	return _rows[row][index];
}

double Gui::CsvTable::number(size_t row, const std::string& name, double fallback) const
{
	const std::string text = value(row, name);
	if (text.empty())
		return fallback;
	try { return std::stod(text); }
	catch (...) { return fallback; }
}

std::vector<std::string> Gui::splitCsvLine(const std::string& line)
{
	std::vector<std::string> values;
	std::string current;
	bool quoted = false;
	for (size_t i = 0; i < line.size(); ++i)
	{
		const char c = line[i];
		if (quoted)
		{
			if (c == '"' && i + 1 < line.size() && line[i + 1] == '"')
			{
				current.push_back('"');
				++i;
			}
			else if (c == '"')
				quoted = false;
			else
				current.push_back(c);
			continue;
		}
		if (c == '"')
			quoted = true;
		else if (c == ',')
		{
			values.push_back(current);
			current.clear();
		}
		else if (c != '\r')
			current.push_back(c);
	}
	values.push_back(current);
	return values;
}

Gui::CsvTable Gui::readCsv(const std::string& path)
{
	CsvTable table;
	std::ifstream input(path);
	if (!input)
		return table;
	std::string line;
	if (!std::getline(input, line))
		return table;
	if (line.size() >= 3 && static_cast<unsigned char>(line[0]) == 0xEF)
		line = line.substr(3);
	table._header = splitCsvLine(line);
	while (std::getline(input, line))
	{
		if (!line.empty())
			table._rows.push_back(splitCsvLine(line));
	}
	return table;
}

std::string Gui::formatMs(double ms)
{
	char text[64];
	if (!std::isfinite(ms))
		return "-";
	if (ms < 0.001)
		std::snprintf(text, sizeof(text), "%.1f ns", ms * 1.0e6);
	else if (ms < 1.0)
		std::snprintf(text, sizeof(text), "%.2f us", ms * 1.0e3);
	else if (ms < 1000.0)
		std::snprintf(text, sizeof(text), "%.1f ms", ms);
	else
		std::snprintf(text, sizeof(text), "%.2f s", ms / 1000.0);
	return text;
}

std::string Gui::formatCount(double value)
{
	char text[64];
	if (value >= 1.0e9)
		std::snprintf(text, sizeof(text), "%.2fB", value / 1.0e9);
	else if (value >= 1.0e6)
		std::snprintf(text, sizeof(text), "%.2fM", value / 1.0e6);
	else if (value >= 1.0e3)
		std::snprintf(text, sizeof(text), "%.1fk", value / 1.0e3);
	else
		std::snprintf(text, sizeof(text), "%.0f", value);
	return text;
}

std::string Gui::formatBytes(double bytes)
{
	char text[64];
	if (bytes >= 1024.0 * 1024.0 * 1024.0)
		std::snprintf(text, sizeof(text), "%.2f GB", bytes / (1024.0 * 1024.0 * 1024.0));
	else if (bytes >= 1024.0 * 1024.0)
		std::snprintf(text, sizeof(text), "%.1f MB", bytes / (1024.0 * 1024.0));
	else
		std::snprintf(text, sizeof(text), "%.0f KB", bytes / 1024.0);
	return text;
}

void Gui::applyTheme()
{
	ImGui::StyleColorsDark();
	ImGuiStyle& style = ImGui::GetStyle();
	style.WindowRounding = 0.0f;
	style.ChildRounding = 6.0f;
	style.FrameRounding = 4.0f;
	style.PopupRounding = 6.0f;
	style.ScrollbarRounding = 4.0f;
	style.GrabRounding = 4.0f;
	style.WindowBorderSize = 0.0f;
	style.ChildBorderSize = 1.0f;
	style.ItemSpacing = ImVec2(10.0f, 8.0f);
	style.FramePadding = ImVec2(10.0f, 6.0f);
	style.CellPadding = ImVec2(8.0f, 4.0f);

	ImVec4* colors = style.Colors;
	colors[ImGuiCol_WindowBg] = ImVec4(0.055f, 0.064f, 0.075f, 1.0f);
	colors[ImGuiCol_ChildBg] = ImVec4(0.080f, 0.092f, 0.108f, 1.0f);
	colors[ImGuiCol_Border] = ImVec4(0.180f, 0.205f, 0.235f, 1.0f);
	colors[ImGuiCol_FrameBg] = ImVec4(0.120f, 0.135f, 0.158f, 1.0f);
	colors[ImGuiCol_FrameBgHovered] = ImVec4(0.170f, 0.195f, 0.225f, 1.0f);
	colors[ImGuiCol_FrameBgActive] = ImVec4(0.210f, 0.245f, 0.285f, 1.0f);
	colors[ImGuiCol_Header] = ImVec4(0.160f, 0.230f, 0.245f, 1.0f);
	colors[ImGuiCol_HeaderHovered] = ImVec4(0.200f, 0.300f, 0.320f, 1.0f);
	colors[ImGuiCol_HeaderActive] = ImVec4(0.245f, 0.370f, 0.385f, 1.0f);
	colors[ImGuiCol_Button] = ImVec4(0.170f, 0.315f, 0.330f, 1.0f);
	colors[ImGuiCol_ButtonHovered] = ImVec4(0.220f, 0.405f, 0.420f, 1.0f);
	colors[ImGuiCol_ButtonActive] = ImVec4(0.270f, 0.480f, 0.500f, 1.0f);
	colors[ImGuiCol_CheckMark] = ImVec4(0.550f, 0.830f, 0.720f, 1.0f);
	colors[ImGuiCol_SliderGrab] = ImVec4(0.550f, 0.830f, 0.720f, 1.0f);
	colors[ImGuiCol_SliderGrabActive] = ImVec4(0.680f, 0.920f, 0.820f, 1.0f);
	colors[ImGuiCol_PlotHistogram] = ImVec4(0.550f, 0.830f, 0.720f, 1.0f);
	colors[ImGuiCol_TableHeaderBg] = ImVec4(0.110f, 0.130f, 0.150f, 1.0f);
}

void Gui::helpMarker(const char* text)
{
	ImGui::SameLine();
	ImGui::TextDisabled("(?)");
	if (ImGui::IsItemHovered())
	{
		ImGui::BeginTooltip();
		ImGui::PushTextWrapPos(ImGui::GetFontSize() * 32.0f);
		ImGui::TextUnformatted(text);
		ImGui::PopTextWrapPos();
		ImGui::EndTooltip();
	}
}

void Gui::hint(const char* text)
{
	ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.62f, 0.68f, 0.78f, 1.0f));
	ImGui::TextWrapped("%s", text);
	ImGui::PopStyleColor();
}

ImVec4 Gui::structureColor(const std::string& typeName, float alpha)
{
	const glm::vec3 rgb = structureColorRgb(typeName);
	return ImVec4(rgb.r, rgb.g, rgb.b, alpha);
}

glm::vec3 Gui::structureColorRgb(const std::string& typeName)
{
	const std::string n = normalizedName(typeName);
	if (oneOf(n, { "quadtree", "quadtreenode", "qt" }))
		return { 0.38f, 0.86f, 0.58f };
	if (oneOf(n, { "kdtree", "kdtreenode", "kd" }))
		return { 0.95f, 0.78f, 0.32f };
	if (oneOf(n, { "bih", "binaryintervalhierarchy", "intervalhierarchy" }))
		return { 1.00f, 0.48f, 0.55f };
	if (oneOf(n, { "karrasoctree", "mortonoctree", "octreekarras", "octreemorton", "kot" }))
		return { 0.42f, 0.72f, 1.00f };
	if (oneOf(n, { "octree", "octreenode", "ot" }))
		return { 0.35f, 0.88f, 0.92f };
	if (oneOf(n, { "regulargrid", "uniformgrid", "grid", "grid3d", "rg" }))
		return { 1.00f, 0.58f, 0.28f };
	if (oneOf(n, { "hgrid", "hierarchicalgrid", "hierarchicalgrid3d", "hg" }))
		return { 0.78f, 0.64f, 1.00f };
	if (oneOf(n, { "bvh", "bvhnode" }))
		return { 0.55f, 0.70f, 1.00f };
	if (oneOf(n, { "lbvh", "linearbvh" }))
		return { 0.66f, 0.82f, 1.00f };
	return { 0.78f, 0.82f, 0.88f };
}

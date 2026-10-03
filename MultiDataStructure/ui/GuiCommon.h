#pragma once

#include "../stdafx.h"

// Small helpers shared by the optimizer GUI's panels: locating project files from wherever the
// executable was launched, CSV reading for the search's result files, a native file picker, and
// the visual language (theme, structure colours, help markers).
namespace Gui
{
	// Project root (the directory holding configs/), searched upward from the working directory
	// and the executable's directory; empty if none is found.
	std::filesystem::path projectRoot();
	// `relative` under the project root, or `relative` unchanged when there is no root.
	std::string projectPath(const std::string& relative);
	// An existing path as given, else the first match under a search root; else the input.
	std::string resolvePath(const std::string& configured);
	bool fileExists(const std::string& path);
	std::filesystem::path executableDirectory();

	// Native "open file" dialog (Windows); returns empty when cancelled or unsupported.
	// `filter` is a Windows filter string, e.g. "Point clouds\0*.las;*.ply\0All\0*.*\0".
	std::string openFileDialog(const char* filter, const std::string& initialPath = {});
	std::string selectFolderDialog(const std::string& initialPath = {});
	void copyToClipboard(const std::string& text);

	// Header-keyed CSV table: rows of strings, columns looked up by name.
	struct CsvTable
	{
		std::vector<std::string>				_header;
		std::vector<std::vector<std::string>>	_rows;

		size_t column(const std::string& name) const;
		std::string value(size_t row, const std::string& name) const;
		double number(size_t row, const std::string& name, double fallback = 0.0) const;
	};
	CsvTable readCsv(const std::string& path);
	std::vector<std::string> splitCsvLine(const std::string& line);

	std::string formatMs(double ms);
	std::string formatCount(double value);
	std::string formatBytes(double bytes);

	void applyTheme();
	void helpMarker(const char* text);
	// Short muted paragraph explaining what a panel is for.
	void hint(const char* text);
	ImVec4 structureColor(const std::string& typeName, float alpha = 1.0f);
	glm::vec3 structureColorRgb(const std::string& typeName);
}

#pragma once

#include "../stdafx.h"

// Minimal charts drawn with ImDrawList: enough for the optimizer's three views (convergence over
// evaluations, a latency/build scatter, per-structure bars) without pulling in a plotting library.
namespace Charts
{
	struct Series
	{
		std::string			_label;
		std::vector<double>	_x;
		std::vector<double>	_y;
		ImU32				_color = IM_COL32(140, 212, 184, 255);
		bool				_line = true;	// false = markers only
		bool				_step = false;	// draw as a staircase (running best)
	};

	struct PlotOptions
	{
		const char*	_xLabel = "";
		const char*	_yLabel = "";
		bool		_logY = false;
		float		_height = 220.0f;
	};

	// Returns the index of the point under the mouse in series `hoveredSeries`, or -1.
	int plot(const char* id, const std::vector<Series>& series, const PlotOptions& options,
			 int* hoveredSeries = nullptr,
			 const std::function<std::string(int series, int point)>& tooltip = {});

	struct Bar
	{
		std::string	_label;
		double		_value = 0.0;
		ImU32		_color = IM_COL32(140, 212, 184, 255);
	};

	void bars(const char* id, const std::vector<Bar>& bars, const char* valueLabel, float rowHeight = 22.0f);
}

#include "Charts.h"

namespace
{
	// "Nice" tick step (1, 2 or 5 times a power of ten) covering `range` in about `targetTicks`.
	double niceStep(double range, int targetTicks)
	{
		if (!(range > 0.0))
			return 1.0;
		const double raw = range / std::max(1, targetTicks);
		const double magnitude = std::pow(10.0, std::floor(std::log10(raw)));
		const double residual = raw / magnitude;
		const double nice = residual < 1.5 ? 1.0 : residual < 3.5 ? 2.0 : residual < 7.5 ? 5.0 : 10.0;
		return nice * magnitude;
	}

	std::string tickLabel(double value)
	{
		char text[32];
		const double magnitude = std::fabs(value);
		if (magnitude != 0.0 && (magnitude >= 1.0e5 || magnitude < 1.0e-3))
			std::snprintf(text, sizeof(text), "%.1e", value);
		else if (magnitude >= 100.0)
			std::snprintf(text, sizeof(text), "%.0f", value);
		else
			std::snprintf(text, sizeof(text), "%.3g", value);
		return text;
	}
}

int Charts::plot(const char* id, const std::vector<Series>& series, const PlotOptions& options,
				 int* hoveredSeries, const std::function<std::string(int, int)>& tooltip)
{
	if (hoveredSeries)
		*hoveredSeries = -1;

	ImGui::PushID(id);
	const ImVec2 origin = ImGui::GetCursorScreenPos();
	const float width = std::max(120.0f, ImGui::GetContentRegionAvail().x);
	const ImVec2 size(width, options._height);
	ImGui::InvisibleButton("##plot", size);
	const bool hovered = ImGui::IsItemHovered();
	ImDrawList* draw = ImGui::GetWindowDrawList();

	const float left = 58.0f, right = 12.0f, top = 10.0f, bottom = 34.0f;
	const ImVec2 p0(origin.x + left, origin.y + top);
	const ImVec2 p1(origin.x + size.x - right, origin.y + size.y - bottom);
	draw->AddRectFilled(origin, ImVec2(origin.x + size.x, origin.y + size.y), IM_COL32(20, 24, 28, 255), 6.0f);

	// Data range.
	double xMin = std::numeric_limits<double>::max(), xMax = -std::numeric_limits<double>::max();
	double yMin = xMin, yMax = xMax;
	const auto ty = [&](double y) { return options._logY ? std::log10(std::max(y, 1e-12)) : y; };
	for (const Series& s : series)
	{
		for (size_t i = 0; i < s._x.size() && i < s._y.size(); ++i)
		{
			if (!std::isfinite(s._x[i]) || !std::isfinite(s._y[i]))
				continue;
			xMin = std::min(xMin, s._x[i]);
			xMax = std::max(xMax, s._x[i]);
			yMin = std::min(yMin, ty(s._y[i]));
			yMax = std::max(yMax, ty(s._y[i]));
		}
	}
	if (xMin > xMax)
	{
		draw->AddText(ImVec2(p0.x + 8.0f, p0.y + 8.0f), IM_COL32(150, 160, 175, 255), "no data yet");
		ImGui::PopID();
		return -1;
	}
	if (xMax - xMin < 1e-12) { xMin -= 0.5; xMax += 0.5; }
	if (yMax - yMin < 1e-12) { yMin -= 0.5; yMax += 0.5; }
	const double yPad = (yMax - yMin) * 0.06;
	yMin -= yPad;
	yMax += yPad;

	const auto sx = [&](double x) { return p0.x + static_cast<float>((x - xMin) / (xMax - xMin)) * (p1.x - p0.x); };
	const auto sy = [&](double y) { return p1.y - static_cast<float>((ty(y) - yMin) / (yMax - yMin)) * (p1.y - p0.y); };

	// Grid and ticks.
	const ImU32 grid = IM_COL32(48, 56, 66, 255);
	const ImU32 label = IM_COL32(150, 160, 175, 255);
	const double xStep = niceStep(xMax - xMin, 6);
	for (double x = std::ceil(xMin / xStep) * xStep; x <= xMax + 1e-9; x += xStep)
	{
		draw->AddLine(ImVec2(sx(x), p0.y), ImVec2(sx(x), p1.y), grid);
		const std::string text = tickLabel(x);
		draw->AddText(ImVec2(sx(x) - ImGui::CalcTextSize(text.c_str()).x * 0.5f, p1.y + 4.0f), label, text.c_str());
	}
	const double yStep = niceStep(yMax - yMin, 4);
	for (double y = std::ceil(yMin / yStep) * yStep; y <= yMax + 1e-9; y += yStep)
	{
		const float py = p1.y - static_cast<float>((y - yMin) / (yMax - yMin)) * (p1.y - p0.y);
		draw->AddLine(ImVec2(p0.x, py), ImVec2(p1.x, py), grid);
		const std::string text = tickLabel(options._logY ? std::pow(10.0, y) : y);
		draw->AddText(ImVec2(p0.x - 6.0f - ImGui::CalcTextSize(text.c_str()).x, py - 7.0f), label, text.c_str());
	}
	draw->AddText(ImVec2((p0.x + p1.x) * 0.5f - ImGui::CalcTextSize(options._xLabel).x * 0.5f, p1.y + 18.0f), label, options._xLabel);
	draw->AddText(ImVec2(origin.x + 6.0f, origin.y + 2.0f), label, options._yLabel);

	// Series, clipped to the plot area.
	draw->PushClipRect(p0, p1, true);
	for (const Series& s : series)
	{
		const size_t n = std::min(s._x.size(), s._y.size());
		ImVec2 previous;
		bool hasPrevious = false;
		for (size_t i = 0; i < n; ++i)
		{
			if (!std::isfinite(s._x[i]) || !std::isfinite(s._y[i]))
				continue;
			const ImVec2 point(sx(s._x[i]), sy(s._y[i]));
			if (s._line && hasPrevious)
			{
				if (s._step)
				{
					draw->AddLine(previous, ImVec2(point.x, previous.y), s._color, 2.0f);
					draw->AddLine(ImVec2(point.x, previous.y), point, s._color, 2.0f);
				}
				else
					draw->AddLine(previous, point, s._color, 2.0f);
			}
			if (!s._line)
				draw->AddCircleFilled(point, 3.5f, s._color);
			previous = point;
			hasPrevious = true;
		}
	}
	draw->PopClipRect();

	// Legend, right-aligned so it never covers the axis label at the top left.
	float legendWidth = 0.0f;
	for (const Series& s : series)
		if (!s._label.empty())
			legendWidth += 24.0f + ImGui::CalcTextSize(s._label.c_str()).x;
	float legendX = p1.x - legendWidth;
	for (const Series& s : series)
	{
		if (s._label.empty())
			continue;
		draw->AddRectFilled(ImVec2(legendX, p0.y + 6.0f), ImVec2(legendX + 10.0f, p0.y + 16.0f), s._color, 2.0f);
		draw->AddText(ImVec2(legendX + 14.0f, p0.y + 3.0f), IM_COL32(205, 212, 222, 255), s._label.c_str());
		legendX += 24.0f + ImGui::CalcTextSize(s._label.c_str()).x;
	}

	// Nearest point under the mouse.
	int bestSeries = -1, bestPoint = -1;
	if (hovered)
	{
		const ImVec2 mouse = ImGui::GetIO().MousePos;
		float bestDistance = 12.0f * 12.0f;
		for (size_t si = 0; si < series.size(); ++si)
		{
			const Series& s = series[si];
			for (size_t i = 0; i < std::min(s._x.size(), s._y.size()); ++i)
			{
				if (!std::isfinite(s._x[i]) || !std::isfinite(s._y[i]))
					continue;
				const float dx = sx(s._x[i]) - mouse.x;
				const float dy = sy(s._y[i]) - mouse.y;
				if (dx * dx + dy * dy < bestDistance)
				{
					bestDistance = dx * dx + dy * dy;
					bestSeries = static_cast<int>(si);
					bestPoint = static_cast<int>(i);
				}
			}
		}
		if (bestSeries >= 0)
		{
			const Series& s = series[static_cast<size_t>(bestSeries)];
			draw->AddCircle(ImVec2(sx(s._x[static_cast<size_t>(bestPoint)]), sy(s._y[static_cast<size_t>(bestPoint)])), 6.0f, IM_COL32(255, 255, 255, 220), 0, 2.0f);
			if (tooltip)
			{
				const std::string text = tooltip(bestSeries, bestPoint);
				if (!text.empty())
					ImGui::SetTooltip("%s", text.c_str());
			}
		}
	}

	if (hoveredSeries)
		*hoveredSeries = bestSeries;
	ImGui::PopID();
	return bestPoint;
}

void Charts::bars(const char* id, const std::vector<Bar>& bars, const char* valueLabel, float rowHeight)
{
	ImGui::PushID(id);
	double maxValue = 0.0;
	float labelWidth = 0.0f;
	for (const Bar& bar : bars)
	{
		maxValue = std::max(maxValue, bar._value);
		labelWidth = std::max(labelWidth, ImGui::CalcTextSize(bar._label.c_str()).x);
	}
	labelWidth += 12.0f;
	const float width = std::max(160.0f, ImGui::GetContentRegionAvail().x);
	ImDrawList* draw = ImGui::GetWindowDrawList();
	for (const Bar& bar : bars)
	{
		const ImVec2 origin = ImGui::GetCursorScreenPos();
		ImGui::Dummy(ImVec2(width, rowHeight));
		draw->AddText(ImVec2(origin.x, origin.y + 3.0f), IM_COL32(205, 212, 222, 255), bar._label.c_str());
		const float barSpace = width - labelWidth - 90.0f;
		const float length = maxValue > 0.0 ? static_cast<float>(bar._value / maxValue) * barSpace : 0.0f;
		draw->AddRectFilled(ImVec2(origin.x + labelWidth, origin.y + 3.0f),
							ImVec2(origin.x + labelWidth + std::max(1.0f, length), origin.y + rowHeight - 3.0f), bar._color, 3.0f);
		char text[48];
		std::snprintf(text, sizeof(text), "%.3g %s", bar._value, valueLabel);
		draw->AddText(ImVec2(origin.x + labelWidth + length + 6.0f, origin.y + 3.0f), IM_COL32(170, 178, 190, 255), text);
	}
	ImGui::PopID();
}

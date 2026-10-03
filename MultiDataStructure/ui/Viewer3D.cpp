#include "Viewer3D.h"

#include "GuiCommon.h"

namespace
{
	const char* VertexShader = R"(#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
uniform mat4 mvp;
uniform float pointSize;
out vec3 vertexColor;
void main()
{
	gl_Position = mvp * vec4(position, 1.0);
	gl_PointSize = pointSize;
	vertexColor = color;
}
)";

	const char* FragmentShader = R"(#version 330 core
in vec3 vertexColor;
out vec4 fragColor;
void main()
{
	fragColor = vec4(vertexColor, 1.0);
}
)";

	unsigned int compile(GLenum type, const char* source)
	{
		const GLuint shader = glCreateShader(type);
		glShaderSource(shader, 1, &source, nullptr);
		glCompileShader(shader);
		GLint ok = 0;
		glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
		if (!ok)
		{
			char log[1024] = {};
			glGetShaderInfoLog(shader, sizeof(log), nullptr, log);
			throw std::runtime_error(std::string("viewer shader: ") + log);
		}
		return shader;
	}

	// Perceptually ordered ramp (viridis-like) for height colouring.
	glm::vec3 ramp(float t)
	{
		t = std::clamp(t, 0.0f, 1.0f);
		const glm::vec3 c0(0.267f, 0.005f, 0.329f), c1(0.229f, 0.322f, 0.546f), c2(0.128f, 0.567f, 0.551f),
			c3(0.369f, 0.789f, 0.383f), c4(0.993f, 0.906f, 0.144f);
		if (t < 0.25f) return glm::mix(c0, c1, t / 0.25f);
		if (t < 0.50f) return glm::mix(c1, c2, (t - 0.25f) / 0.25f);
		if (t < 0.75f) return glm::mix(c2, c3, (t - 0.50f) / 0.25f);
		return glm::mix(c3, c4, (t - 0.75f) / 0.25f);
	}

	glm::vec3 depthColor(size_t depth)
	{
		return ramp(std::fmod(static_cast<float>(depth) * 0.137f, 1.0f));
	}

	std::string structureName(const PointSpatialIndex& index, const PointSpatialIndex::NodeView& node)
	{
		const std::vector<std::string>& names = index.structureBreakdownNames();
		return node.structureIndex() < names.size() ? names[node.structureIndex()] : std::string();
	}

	float distanceSquaredToBox(const AABB& box, const glm::vec3& p)
	{
		const glm::vec3 lo = box.min(), hi = box.max();
		float d2 = 0.0f;
		for (int a = 0; a < 3; ++a)
		{
			if (p[a] < lo[a]) { const float d = lo[a] - p[a]; d2 += d * d; }
			else if (p[a] > hi[a]) { const float d = p[a] - hi[a]; d2 += d * d; }
		}
		return d2;
	}
}

// ---------------------------------------------------------------- camera

glm::vec3 Viewer::Camera::eye() const
{
	const glm::vec3 direction(std::cos(_pitch) * std::cos(_yaw), std::cos(_pitch) * std::sin(_yaw), std::sin(_pitch));
	return _target + direction * _distance;
}

glm::mat4 Viewer::Camera::view() const
{
	return glm::lookAt(eye(), _target, glm::vec3(0.0f, 0.0f, 1.0f));
}

glm::mat4 Viewer::Camera::projection(float aspect, float sceneRadius) const
{
	const float nearPlane = std::max(_distance * 0.001f, sceneRadius * 1e-5f);
	const float farPlane = _distance + sceneRadius * 4.0f;
	return glm::perspective(_fovY, std::max(aspect, 0.1f), nearPlane, farPlane);
}

// ---------------------------------------------------------------- renderer

Viewer::Renderer::~Renderer()
{
	if (!_program)
		return;
	release(_points);
	release(_boxes);
	release(_highlightPoints);
	release(_highlightLines);
	if (_framebuffer) glDeleteFramebuffers(1, &_framebuffer);
	if (_colorTexture) glDeleteTextures(1, &_colorTexture);
	if (_depthBuffer) glDeleteRenderbuffers(1, &_depthBuffer);
	glDeleteProgram(_program);
}

void Viewer::Renderer::initialize()
{
	if (_program)
		return;
	const GLuint vs = compile(GL_VERTEX_SHADER, VertexShader);
	const GLuint fs = compile(GL_FRAGMENT_SHADER, FragmentShader);
	_program = glCreateProgram();
	glAttachShader(_program, vs);
	glAttachShader(_program, fs);
	glLinkProgram(_program);
	glDeleteShader(vs);
	glDeleteShader(fs);
	_mvpLocation = glGetUniformLocation(_program, "mvp");
	_pointSizeLocation = glGetUniformLocation(_program, "pointSize");
}

void Viewer::Renderer::release(Buffer& buffer)
{
	if (buffer._vbo) glDeleteBuffers(1, &buffer._vbo);
	if (buffer._vao) glDeleteVertexArrays(1, &buffer._vao);
	buffer = Buffer();
}

void Viewer::Renderer::upload(Buffer& buffer, const std::vector<ColoredVertex>& vertices)
{
	if (!buffer._vao)
	{
		glGenVertexArrays(1, &buffer._vao);
		glGenBuffers(1, &buffer._vbo);
		glBindVertexArray(buffer._vao);
		glBindBuffer(GL_ARRAY_BUFFER, buffer._vbo);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(ColoredVertex), reinterpret_cast<void*>(0));
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ColoredVertex), reinterpret_cast<void*>(sizeof(glm::vec3)));
		glBindVertexArray(0);
	}
	glBindBuffer(GL_ARRAY_BUFFER, buffer._vbo);
	glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size() * sizeof(ColoredVertex)), vertices.data(), GL_STATIC_DRAW);
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	buffer._count = vertices.size();
}

void Viewer::Renderer::setPoints(const std::vector<ColoredVertex>& points) { initialize(); upload(_points, points); }
void Viewer::Renderer::setBoxes(const std::vector<ColoredVertex>& lines) { initialize(); upload(_boxes, lines); }

void Viewer::Renderer::setHighlights(const std::vector<ColoredVertex>& points, const std::vector<ColoredVertex>& lines)
{
	initialize();
	upload(_highlightPoints, points);
	upload(_highlightLines, lines);
}

void Viewer::Renderer::ensureTarget(int width, int height)
{
	if (_framebuffer && width == _targetWidth && height == _targetHeight)
		return;
	if (!_framebuffer)
	{
		glGenFramebuffers(1, &_framebuffer);
		glGenTextures(1, &_colorTexture);
		glGenRenderbuffers(1, &_depthBuffer);
	}
	glBindTexture(GL_TEXTURE_2D, _colorTexture);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, nullptr);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glBindRenderbuffer(GL_RENDERBUFFER, _depthBuffer);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT24, width, height);
	glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, _colorTexture, 0);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, _depthBuffer);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	_targetWidth = width;
	_targetHeight = height;
}

unsigned int Viewer::Renderer::render(int width, int height, const Camera& camera, float sceneRadius, float pointSize)
{
	initialize();
	width = std::max(width, 16);
	height = std::max(height, 16);
	ensureTarget(width, height);

	GLint previousViewport[4];
	glGetIntegerv(GL_VIEWPORT, previousViewport);
	glBindFramebuffer(GL_FRAMEBUFFER, _framebuffer);
	glViewport(0, 0, width, height);
	glClearColor(0.035f, 0.042f, 0.050f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_PROGRAM_POINT_SIZE);

	const glm::mat4 mvp = camera.projection(static_cast<float>(width) / static_cast<float>(height), sceneRadius) * camera.view();
	glUseProgram(_program);
	glUniformMatrix4fv(_mvpLocation, 1, GL_FALSE, &mvp[0][0]);

	const auto draw = [&](const Buffer& buffer, GLenum mode, float size) {
		if (!buffer._count)
			return;
		glUniform1f(_pointSizeLocation, size);
		glBindVertexArray(buffer._vao);
		glDrawArrays(mode, 0, static_cast<GLsizei>(buffer._count));
	};
	draw(_points, GL_POINTS, pointSize);
	draw(_boxes, GL_LINES, 1.0f);
	// Highlights are drawn without depth test so a query is visible through the cloud.
	glDisable(GL_DEPTH_TEST);
	draw(_highlightLines, GL_LINES, 1.0f);
	draw(_highlightPoints, GL_POINTS, pointSize + 3.0f);

	glBindVertexArray(0);
	glUseProgram(0);
	glDisable(GL_PROGRAM_POINT_SIZE);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glViewport(previousViewport[0], previousViewport[1], previousViewport[2], previousViewport[3]);
	return _colorTexture;
}

// ---------------------------------------------------------------- scene

void Viewer::appendBoxLines(std::vector<ColoredVertex>& lines, const AABB& box, const glm::vec3& color)
{
	const glm::vec3 lo = box.min(), hi = box.max();
	const glm::vec3 c[8] = {
		{ lo.x, lo.y, lo.z }, { hi.x, lo.y, lo.z }, { hi.x, hi.y, lo.z }, { lo.x, hi.y, lo.z },
		{ lo.x, lo.y, hi.z }, { hi.x, lo.y, hi.z }, { hi.x, hi.y, hi.z }, { lo.x, hi.y, hi.z } };
	static constexpr int Edges[12][2] = { {0,1},{1,2},{2,3},{3,0},{4,5},{5,6},{6,7},{7,4},{0,4},{1,5},{2,6},{3,7} };
	for (const auto& edge : Edges)
	{
		lines.push_back({ c[edge[0]], color });
		lines.push_back({ c[edge[1]], color });
	}
}

std::vector<Viewer::ColoredVertex> Viewer::SceneBuilder::points(const PointCloud& cloud, const PointSpatialIndex& index) const
{
	std::vector<ColoredVertex> out;
	const std::vector<uint32_t>& order = index.pointOrder();
	if (order.empty())
		return out;
	// Sampling in tree order keeps the subsample spatially even and lets each point inherit the
	// structure of the leaf it sits in, without a per-point lookup.
	const size_t stride = std::max<size_t>(1, order.size() / std::max<size_t>(1, _maxPoints));
	const AABB bounds = cloud.bounds();
	const float zMin = bounds.min().z;
	const float zRange = std::max(bounds.size().z, 1e-6f);
	out.reserve(order.size() / stride + 1);

	if (_pointColoring == PointColoring::Height)
	{
		for (size_t i = 0; i < order.size(); i += stride)
		{
			const glm::vec3 p = cloud.points()[order[i]].position;
			out.push_back({ p, ramp((p.z - zMin) / zRange) * 0.9f + 0.1f });
		}
		return out;
	}

	std::vector<PointSpatialIndex::NodeView> stack{ index.root() };
	while (!stack.empty())
	{
		const PointSpatialIndex::NodeView node = stack.back();
		stack.pop_back();
		if (!node.isLeaf())
		{
			for (size_t c = node.childCount(); c-- > 0;)
				stack.push_back(node.child(c));
			continue;
		}
		const glm::vec3 color = Gui::structureColorRgb(structureName(index, node)) * 0.85f;
		const size_t begin = node.pointOffset(), end = begin + node.pointCount();
		for (size_t i = begin + (stride - begin % stride) % stride; i < end; i += stride)
			out.push_back({ cloud.points()[order[i]].position, color });
	}
	return out;
}

std::vector<Viewer::ColoredVertex> Viewer::SceneBuilder::boxes(const PointSpatialIndex& index, size_t* emitted) const
{
	std::vector<ColoredVertex> lines;
	size_t count = 0;
	// Breadth-first, so a box budget shows the coarse levels completely rather than one deep path.
	std::deque<PointSpatialIndex::NodeView> queue;
	if (index.root())
		queue.push_back(index.root());
	while (!queue.empty() && count < _maxBoxes)
	{
		const PointSpatialIndex::NodeView node = queue.front();
		queue.pop_front();
		const int depth = static_cast<int>(node.depth());
		if (depth > _maxDepth)
			continue;
		const bool show = depth >= _minDepth && (!_leavesOnly || node.isLeaf()) && node.pointCount() > 0;
		if (show)
		{
			const glm::vec3 color = _boxColoring == BoxColoring::Depth
				? depthColor(node.depth())
				: Gui::structureColorRgb(structureName(index, node));
			appendBoxLines(lines, node.tightBounds(), color * 0.75f);
			++count;
		}
		for (size_t c = 0; c < node.childCount(); ++c)
			queue.push_back(node.child(c));
	}
	if (emitted)
		*emitted = count;
	return lines;
}

Viewer::QueryOverlay Viewer::runQuery(const PointSpatialIndex& index, QueryKind kind, const glm::vec3& center,
									   float radius, size_t k, const glm::vec3& halfExtent)
{
	QueryOverlay overlay;
	overlay._valid = true;
	overlay._kind = kind;
	overlay._center = center;
	overlay._radius = radius;
	overlay._k = k;
	overlay._box = AABB(center - halfExtent, center + halfExtent);

	PointSpatialIndex::QueryResult result;
	if (kind == QueryKind::Radius)
		result = index.radiusQuery(center, radius);
	else if (kind == QueryKind::Knn)
		result = index.knnQuery(center, k);
	else
		result = index.rangeQuery(overlay._box);
	overlay._returned = result._pointIndices;
	overlay._stats = result._stats;

	// Entered nodes, by the same prune the engine applies.
	float reachSquared = radius * radius;
	if (kind == QueryKind::Knn && !overlay._returned.empty() && index.cloud())
	{
		const glm::vec3 farthest = index.cloud()->points()[overlay._returned.back()].position;
		const glm::vec3 d = farthest - center;
		reachSquared = glm::dot(d, d);
		overlay._kthDistance = std::sqrt(static_cast<double>(reachSquared));
	}
	std::vector<PointSpatialIndex::NodeView> stack;
	if (index.root())
		stack.push_back(index.root());
	while (!stack.empty())
	{
		const PointSpatialIndex::NodeView node = stack.back();
		stack.pop_back();
		if (node.pointCount() == 0)
			continue;
		const AABB tight = node.tightBounds();
		const bool entered = kind == QueryKind::Range
			? tight.collides(overlay._box)
			: distanceSquaredToBox(tight, center) <= reachSquared;
		if (!entered)
			continue;
		overlay._enteredNodes.push_back(node.id());
		for (size_t c = 0; c < node.childCount(); ++c)
			stack.push_back(node.child(c));
	}
	return overlay;
}

void Viewer::overlayGeometry(const PointCloud& cloud, const PointSpatialIndex& index, const QueryOverlay& overlay,
							 std::vector<ColoredVertex>& points, std::vector<ColoredVertex>& lines)
{
	points.clear();
	lines.clear();
	if (!overlay._valid)
		return;

	const glm::vec3 returnedColor(1.0f, 0.35f, 0.30f);
	// Magenta: no structure uses it, so entered leaves never blend into the index's own boxes.
	const glm::vec3 nodeColor(1.0f, 0.30f, 0.95f);
	const glm::vec3 queryColor(1.0f, 1.0f, 1.0f);
	for (const size_t i : overlay._returned)
		points.push_back({ cloud.points()[i].position, returnedColor });
	points.push_back({ overlay._center, queryColor });

	// Entered leaves only: inner boxes would bury the picture at depth.
	std::vector<PointSpatialIndex::NodeView> stack;
	std::unordered_set<uint32_t> entered(overlay._enteredNodes.begin(), overlay._enteredNodes.end());
	if (index.root())
		stack.push_back(index.root());
	size_t drawn = 0;
	while (!stack.empty() && drawn < 20000)
	{
		const PointSpatialIndex::NodeView node = stack.back();
		stack.pop_back();
		if (!entered.count(node.id()))
			continue;
		if (node.isLeaf())
		{
			appendBoxLines(lines, node.tightBounds(), nodeColor);
			++drawn;
		}
		for (size_t c = 0; c < node.childCount(); ++c)
			stack.push_back(node.child(c));
	}

	if (overlay._kind == QueryKind::Range)
	{
		appendBoxLines(lines, overlay._box, queryColor);
		return;
	}
	// Three great circles for the query sphere (radius, or the k-th distance for kNN).
	const float r = overlay._kind == QueryKind::Knn ? static_cast<float>(overlay._kthDistance) : overlay._radius;
	if (r <= 0.0f)
		return;
	constexpr int Segments = 64;
	for (int plane = 0; plane < 3; ++plane)
	{
		for (int s = 0; s < Segments; ++s)
		{
			const float a0 = 6.2831853f * s / Segments, a1 = 6.2831853f * (s + 1) / Segments;
			glm::vec3 p0(0.0f), p1(0.0f);
			const int u = plane, v = (plane + 1) % 3;
			p0[u] = std::cos(a0) * r; p0[v] = std::sin(a0) * r;
			p1[u] = std::cos(a1) * r; p1[v] = std::sin(a1) * r;
			lines.push_back({ overlay._center + p0, queryColor });
			lines.push_back({ overlay._center + p1, queryColor });
		}
	}
}

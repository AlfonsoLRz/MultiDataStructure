#pragma once

#include "../stdafx.h"
#include "../workloads/points/PointSpatialIndex.h"

// The Inspect view: a built index drawn over its point cloud, and what one query does to it.
//
// Rendering is plain OpenGL 3.3 into an offscreen framebuffer that the GUI shows as an image, so
// it composes with the ImGui layout. Everything shown is read from the real index through
// PointSpatialIndex::NodeView - node boxes are the boxes the traversal tests, not an illustration.
namespace Viewer
{
	struct Camera
	{
		glm::vec3	_target = glm::vec3(0.0f);
		float		_distance = 10.0f;
		float		_yaw = 0.8f;
		float		_pitch = 0.5f;
		float		_fovY = 0.8f;

		glm::vec3 eye() const;
		glm::mat4 view() const;
		glm::mat4 projection(float aspect, float sceneRadius) const;
	};

	struct ColoredVertex
	{
		glm::vec3	_position;
		glm::vec3	_color;
	};

	class Renderer
	{
	public:
		~Renderer();
		bool initialized() const { return _program != 0; }
		void initialize();
		void setPoints(const std::vector<ColoredVertex>& points);
		void setBoxes(const std::vector<ColoredVertex>& lineVertices);
		void setHighlights(const std::vector<ColoredVertex>& points, const std::vector<ColoredVertex>& lineVertices);
		// Draws into the offscreen target (resized as needed) and returns the colour texture.
		unsigned int render(int width, int height, const Camera& camera, float sceneRadius, float pointSize);

	private:
		struct Buffer
		{
			unsigned int	_vao = 0;
			unsigned int	_vbo = 0;
			size_t			_count = 0;
		};
		void upload(Buffer& buffer, const std::vector<ColoredVertex>& vertices);
		void release(Buffer& buffer);
		void ensureTarget(int width, int height);

		unsigned int	_program = 0;
		int				_mvpLocation = -1;
		int				_pointSizeLocation = -1;
		Buffer			_points, _boxes, _highlightPoints, _highlightLines;
		unsigned int	_framebuffer = 0, _colorTexture = 0, _depthBuffer = 0;
		int				_targetWidth = 0, _targetHeight = 0;
	};

	enum class PointColoring { Height, Structure };
	enum class BoxColoring { Structure, Depth };
	enum class QueryKind { Radius, Knn, Range };

	struct QueryOverlay
	{
		bool					_valid = false;
		QueryKind				_kind = QueryKind::Radius;
		glm::vec3				_center = glm::vec3(0.0f);
		float					_radius = 0.0f;
		size_t					_k = 16;
		AABB					_box;
		std::vector<size_t>		_returned;
		std::vector<uint32_t>	_enteredNodes;
		PointSpatialIndex::QueryStats	_stats;
		double					_kthDistance = 0.0;
	};

	// Builds the vertex lists the renderer draws from an index. Keeps the scene radius for the
	// camera, the per-structure node counts for the legend, and caps what it emits so a
	// billion-point index stays interactive.
	struct SceneBuilder
	{
		size_t			_maxPoints = 2'000'000;
		size_t			_maxBoxes = 3000;	// breadth-first, so the default shows the coarse levels legibly
		int				_minDepth = 0;
		int				_maxDepth = 64;
		PointColoring	_pointColoring = PointColoring::Height;
		BoxColoring		_boxColoring = BoxColoring::Structure;
		bool			_leavesOnly = false;

		std::vector<ColoredVertex> points(const PointCloud& cloud, const PointSpatialIndex& index) const;
		std::vector<ColoredVertex> boxes(const PointSpatialIndex& index, size_t* emitted = nullptr) const;
	};

	// Runs one query against the index (instrumented) and records which nodes the traversal had
	// to enter. Radius and range report exactly the engine's entered set (bounds that pass the
	// same prune); for kNN it is the nodes whose bounds reach the final k-th distance, which every
	// exact kNN search must enter.
	QueryOverlay runQuery(const PointSpatialIndex& index, QueryKind kind, const glm::vec3& center,
						  float radius, size_t k, const glm::vec3& halfExtent);
	void overlayGeometry(const PointCloud& cloud, const PointSpatialIndex& index, const QueryOverlay& overlay,
						 std::vector<ColoredVertex>& points, std::vector<ColoredVertex>& lines);

	void appendBoxLines(std::vector<ColoredVertex>& lines, const AABB& box, const glm::vec3& color);
}

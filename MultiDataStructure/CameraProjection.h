#pragma once

class CameraProjection
{
protected:
	~CameraProjection() = default;
	static std::vector<std::shared_ptr<CameraProjection>> _cameraProjection;

public:
	enum Projection
	{
		PERSPECTIVE, ORTHOGRAPHIC
	};

	class CameraProperties
	{
	public:
		int			_cameraType;
		bool		_2d;

		glm::vec3	_eye, _lookAt, _up;
		float		_zNear, _zFar, _focusPoint;
		float		_aspect;
		float		_fovY, _fovX;
		glm::vec2	_bottomLeftCorner;
		uint16_t	_width, _height;
		glm::vec3	_n, _u, _v;
		glm::mat4	_viewMatrix, _projectionMatrix, _viewProjectionMatrix;

		float		computeAspect() const;
		void		computeAxes(glm::vec3& n, glm::vec3& u, glm::vec3& v) const;
		glm::vec2	computeBottomLeftCorner() const;
		float		computeFovY() const;

		void		computeProjectionMatrices(CameraProperties* camera);
		void		computeViewMatrices();
		void		computeViewMatrix();

		void		zoom(float speed);
	};

public:
	virtual glm::mat4	buildProjectionMatrix(CameraProperties* camera) = 0;
	virtual void		zoom(CameraProperties* camera, const float speed) = 0;
};

class PerspectiveProjection : public CameraProjection
{
public:
	glm::mat4	buildProjectionMatrix(CameraProperties* camera) override;
	void		zoom(CameraProperties* camera, const float speed) override;
};

class OrthographicProjection : public CameraProjection
{
public:
	glm::mat4	buildProjectionMatrix(CameraProperties* camera) override;
	void		zoom(CameraProperties* camera, const float speed) override;
};



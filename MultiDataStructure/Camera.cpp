#include "stdafx.h"
#include "Camera.h"

#include "CameraProjection.h"


// Public methods

Camera::Camera(uint16_t width, uint16_t height, bool is2D) : _backupCamera(nullptr)
{
	this->_properties._cameraType = CameraProjection::PERSPECTIVE;
	this->_properties._2d = is2D;

	this->_properties._eye = glm::vec3(0.0f, 0.0f, -15.0f);
	this->_properties._lookAt = glm::vec3(0.0f, 0.0f, 0.0f);
	this->_properties._up = glm::vec3(0.0f, 1.0f, 0.0f);

	this->_properties._zNear = 0.05f;
	this->_properties._zFar = 40.0f;
	this->_properties._focusPoint = 15.0f;

	this->_properties._width = width;
	this->_properties._height = height;
	this->_properties._aspect = this->_properties.computeAspect();

	this->_properties._bottomLeftCorner = glm::vec2(-2.0f * this->_properties._aspect, -2.0f);
	this->_properties._fovX = glm::radians(20.0f);
	this->_properties._fovY = this->_properties.computeFovY();

	this->_properties.computeAxes(this->_properties._n, this->_properties._u, this->_properties._v);
	this->_properties.computeViewMatrix();
	this->_properties.computeProjectionMatrices(&this->_properties);

	this->saveCamera();
}

Camera::Camera(const Camera& camera) : _backupCamera(nullptr)
{
	this->copyCameraAttributes(&camera);
}

Camera::~Camera()
{
	delete _backupCamera;
}

void Camera::reset()
{
	this->copyCameraAttributes(_backupCamera);
}

void Camera::track(const Model3D* model)
{
	AABB aabb = model->getAABB();

	this->setLookAt(aabb.center());
	this->setPosition(aabb.min() + glm::vec3(.0f, aabb.extent().y, 1.0f) - glm::vec3(aabb.extent().x, .0f, .0) * (1 + (1.0f / std::max(aabb.size().x, std::max(aabb.size().y, aabb.size().z))) * 4.0f));
}

void Camera::saveCamera()
{
	delete _backupCamera;
	_backupCamera = nullptr;

	_backupCamera = new Camera(*this);
}

void Camera::setBottomLeftCorner(const glm::vec2& bottomLeft)
{
	this->_properties._bottomLeftCorner = bottomLeft;
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void Camera::setCameraType(const CameraProjection::Projection projection)
{
	this->_properties._cameraType = projection;
	this->_properties.computeViewMatrices();
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void Camera::setFovX(const float fovX)
{
	this->_properties._fovX = fovX;
	this->_properties._fovY = this->_properties.computeFovY();
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void Camera::setFovY(const float fovY)
{
	this->_properties._fovY = fovY;
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void Camera::setLookAt(const glm::vec3& position)
{
	this->_properties._lookAt = position;
	this->_properties.computeAxes(this->_properties._n, this->_properties._u, this->_properties._v);
	this->_properties.computeViewMatrices();
}

void Camera::setPosition(const glm::vec3& position)
{
	this->_properties._eye = position;
	this->_properties.computeAxes(this->_properties._n, this->_properties._u, this->_properties._v);
	this->_properties.computeViewMatrices();
}

void Camera::setRaspect(const uint16_t width, const uint16_t height)
{
	this->_properties._width = width;
	this->_properties._height = height;
	this->_properties._aspect = this->_properties.computeAspect();
	this->_properties._bottomLeftCorner = glm::vec2(this->_properties._bottomLeftCorner.y * this->_properties._aspect, this->_properties._bottomLeftCorner.y);
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void Camera::setUp(const glm::vec3& up)
{
	this->_properties._up = up;
	this->_properties.computeViewMatrices();
}

void Camera::setZFar(const float zfar)
{
	this->_properties._zFar = zfar;
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void Camera::setZNear(const float znear)
{
	this->_properties._zNear = znear;
	this->_properties.computeProjectionMatrices(&this->_properties);
}

void Camera::updateMatrices()
{
	this->_properties.computeViewMatrix();
	this->_properties.computeProjectionMatrices(&_properties);
}

// [Movements] 

void Camera::boom(float speed)
{
	const glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), this->_properties._v * speed);			// Translation intersectsBoundingBox y axis

	this->_properties._eye = glm::vec3(translationMatrix * glm::vec4(this->_properties._eye, 1.0f));
	this->_properties._lookAt = glm::vec3(translationMatrix * glm::vec4(this->_properties._lookAt, 1.0f));

	this->_properties.computeViewMatrices();
}

void Camera::crane(float speed)
{
	boom(-speed);					// Implemented as another method to take advantage of nomenclature
}

void Camera::dolly(float speed)
{
	if (this->_properties._2d) return;

	const glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), -this->_properties._n * speed);			// Translation intersectsBoundingBox z axis
	this->_properties._eye = glm::vec3(translationMatrix * glm::vec4(this->_properties._eye, 1.0f));
	this->_properties._lookAt = glm::vec3(translationMatrix * glm::vec4(this->_properties._lookAt, 1.0f));

	this->_properties.computeViewMatrices();
}

void Camera::orbitXZ(float speed)
{
	if (this->_properties._2d) return;

	const glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), speed, this->_properties._u);					// We will pass over the scene, x or z axis could be used

	this->_properties._eye = glm::vec3(rotationMatrix * glm::vec4(this->_properties._eye - this->_properties._lookAt, 1.0f)) + this->_properties._lookAt;
	this->_properties._u = glm::vec3(rotationMatrix * glm::vec4(this->_properties._u, 0.0f));
	this->_properties._v = glm::vec3(rotationMatrix * glm::vec4(this->_properties._v, 0.0f));
	this->_properties._n = glm::vec3(rotationMatrix * glm::vec4(this->_properties._n, 0.0f));
	this->_properties._up = glm::normalize(glm::cross(this->_properties._n, this->_properties._u));						// Free rotation => we can look down or up

	this->_properties.computeViewMatrices();
}

void Camera::orbitY(float speed)
{
	if (this->_properties._2d) return;

	const glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), speed, glm::vec3(0.0, 1.0f, 0.0f));

	this->_properties._u = glm::vec3(rotationMatrix * glm::vec4(this->_properties._u, 0.0f));
	this->_properties._v = glm::vec3(rotationMatrix * glm::vec4(this->_properties._v, 0.0f));
	this->_properties._n = glm::vec3(rotationMatrix * glm::vec4(this->_properties._n, 0.0f));
	this->_properties._up = glm::normalize(glm::cross(this->_properties._n, this->_properties._u));								// This movement doesn't change UP, but it could occur as a result of previous operations
	this->_properties._eye = glm::vec3(rotationMatrix * glm::vec4(this->_properties._eye - this->_properties._lookAt, 1.0f)) + this->_properties._lookAt;

	this->_properties.computeViewMatrices();
}

void Camera::pan(float speed)
{
	if (this->_properties._2d) return;

	const glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), speed, glm::vec3(0.0f, 1.0f, 0.0f));

	// Up vector can change, not intersectsBoundingBox the original position tho. Example: orbit XZ (rotated camera) + pan
	this->_properties._u = glm::vec3(rotationMatrix * glm::vec4(this->_properties._u, 0.0f));
	this->_properties._v = glm::vec3(rotationMatrix * glm::vec4(this->_properties._v, 0.0f));
	this->_properties._n = glm::vec3(rotationMatrix * glm::vec4(this->_properties._n, 0.0f));
	this->_properties._up = glm::normalize(glm::cross(this->_properties._n, this->_properties._u));
	this->_properties._lookAt = glm::vec3(rotationMatrix * glm::vec4(this->_properties._lookAt - this->_properties._eye, 1.0f)) + this->_properties._eye;

	this->_properties.computeViewMatrices();
}

void Camera::tilt(float speed)
{
	if (this->_properties._2d) return;

	const glm::mat4 rotationMatrix = glm::rotate(glm::mat4(1.0f), speed, this->_properties._u);

	const glm::vec3 n = glm::vec3(rotationMatrix * glm::vec4(this->_properties._n, 0.0f));
	float alpha = glm::acos(glm::dot(n, glm::vec3(0.0f, 1.0f, 0.0f)));

	if (alpha < speed || alpha >(glm::pi<float>() - speed))
	{
		return;
	}

	this->_properties._v = glm::vec3(rotationMatrix * glm::vec4(this->_properties._v, 0.0f));
	this->_properties._n = n;
	this->_properties._up = glm::normalize(glm::cross(this->_properties._n, this->_properties._u));											// It could change because of the rotation
	this->_properties._lookAt = glm::vec3(rotationMatrix * glm::vec4(this->_properties._lookAt - this->_properties._eye, 1.0f)) + this->_properties._eye;

	this->_properties.computeViewMatrices();
}

void Camera::truck(float speed)
{
	const glm::mat4 translationMatrix = glm::translate(glm::mat4(1.0f), this->_properties._u * speed);				// Translation intersectsBoundingBox x axis

	this->_properties._eye = glm::vec3(translationMatrix * glm::vec4(this->_properties._eye, 1.0f));
	this->_properties._lookAt = glm::vec3(translationMatrix * glm::vec4(this->_properties._lookAt, 1.0f));

	this->_properties.computeViewMatrices();
}

void Camera::zoom(float speed)
{
	this->_properties.zoom(speed);
}

void Camera::buildRays(std::vector<RayGPU>& rays, const glm::uvec2& windowSize, const glm::uint numSamples) const
{
	float planeHeight = 2.0f * getFarClipPlane() * glm::tan(getFOV_Y() / 2.0f);
	float planeWidth = planeHeight * getAspect();
	glm::vec3 bottomLeft = glm::vec3(-planeWidth / 2.0f, -planeHeight / 2.0f, .0f);

	rays.resize(windowSize.x * windowSize.y * numSamples);

	for (int x = 0; x < windowSize.x; ++x)
	{
		for (glm::uint y = 0; y < windowSize.y; ++y)
		{
			for (glm::uint sample = 0; sample < numSamples; ++sample)
			{
				const int sqrtSamples = static_cast<int>(glm::sqrt(float(numSamples)));
				const glm::uint sampleX = sample % sqrtSamples, sampleY = sample / sqrtSamples;
				const float jitterX = static_cast<float>(sampleX) / static_cast<float>(sqrtSamples), jitterY = static_cast<float>(sampleY) / static_cast<float>(sqrtSamples);

				const float tx = (static_cast<float>(x) + jitterX) / static_cast<float>(windowSize.x);
				const float ty = (static_cast<float>(y) + jitterY) / static_cast<float>(windowSize.y);

				const glm::vec3 pointLocal = bottomLeft + glm::vec3(tx * planeWidth, ty * planeHeight, .0f);
				const glm::vec3 focusPoint = getPosition() + getRight() * pointLocal.x + getUp() * pointLocal.y + getForward() * getFarClipPlane();
				const glm::vec3 direction = glm::normalize(focusPoint - getPosition());

				rays[x + y * windowSize.x + sample * windowSize.x * windowSize.y] = RayGPU{ getPosition(), direction };
			}
		}
	}
}

/// [Private methods]

void Camera::copyCameraAttributes(const Camera* camera)
{
	this->_properties = camera->_properties;

	if (camera->_backupCamera)
	{
		Camera* backupCamera = this->_backupCamera;
		this->_backupCamera = new Camera(*camera->_backupCamera);

		delete backupCamera;
	}
}

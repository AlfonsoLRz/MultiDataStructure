#pragma once

#include "stdafx.h"

#include "CudaHelper.h"
#include <cuda_gl_interop.h>

struct CameraGPU
{
	glm::vec3	_bottomLeft;
	float		_planeWidth;

	glm::vec3	_right;
	float 		_planeHeight;

	glm::vec3	_up;
	float		_zfar;

	glm::vec3	_eye;
	float		_znear;

	glm::vec3	_forward;
	float		_focus;

	glm::mat4	_projectionMatrix;
};

struct VertexGPU
{
	glm::vec3	_position;
	float 		_padding1;

	glm::vec3	_normal;
	float		_padding2;

	glm::vec2	_textCoord;
	glm::vec2	_padding3;
};

struct MeshGPU
{
	glm::vec3	_diffuseColour; 
	glm::uint	_startIndex;

	glm::vec3	_emissionColor;
	float		_emissionStrength;

	glm::vec3   _specularColor;
	float		_metallic;

	glm::vec3	_max;
	glm::uint	_length;

	glm::vec3	_min;
	float		_smoothness;

	int			_textureIndex;
	float		_area;
	glm::uint	_padding1;
	glm::uint	_padding2;
};

struct HitInfo
{
	glm::vec3	_position;
	float		_t;

	glm::vec3	_normal;
	glm::uint 	_materialIndex;

	int			_hit;
	glm::uint	_triangleIndex;
	glm::uint	_padding1;
	glm::uint	_padding2;
};

struct RayGPU
{
	glm::vec3	_origin;
	glm::vec3	_direction;

	glm::vec3 getDirection() const { return _direction; }
	glm::vec3 getOrigin() const { return _origin; }

	void intersectsTriangle(const glm::vec3& a, const glm::vec3& b, const glm::vec3& c, HitInfo& hitInfo, float maxDistance) const
	{
		const glm::vec3 normal = glm::normalize(cross(b - a, c - a));
		const float t = -(dot(normal, _origin) + -glm::dot(normal, a)) / dot(normal, _direction);

		hitInfo._hit = 0;

		if (t > glm::epsilon<float>() && t < maxDistance)
		{
			const glm::vec3 p = _origin + t * _direction;
			const glm::vec3 v0 = c - a;
			const glm::vec3 v1 = b - a;
			const glm::vec3 v2 = p - a;
			const float dot00 = glm::dot(v0, v0);
			const float dot01 = glm::dot(v0, v1);
			const float dot02 = glm::dot(v0, v2);
			const float dot11 = glm::dot(v1, v1);
			const float dot12 = glm::dot(v1, v2);
			const float inverseDenominator = 1.0f / (dot00 * dot11 - dot01 * dot01);
			const float u = (dot11 * dot02 - dot01 * dot12) * inverseDenominator;
			const float v = (dot00 * dot12 - dot01 * dot02) * inverseDenominator;

			if (u >= 0.0f && v >= 0.0f && u + v <= 1.0f)
			{
				hitInfo._hit = 1;
				hitInfo._t = t;
				hitInfo._position = p;
				hitInfo._normal = normal;
			}
		}
	}
};

struct Node
{
	glm::vec3  _maxPoint;
	glm::uint  _prevIndex1;

	glm::vec3  _minPoint;
	glm::uint  _prevIndex2;

	glm::uint  _triangleIndex;
	glm::uint  _numTriangles;
	glm::uint  _meshIndex;

	float centroid(glm::uint axis) const
	{
		return (_minPoint[axis] + _maxPoint[axis]) * 0.5f;
	}

	glm::vec3 centroid() const
	{
		return (_minPoint + _maxPoint) * 0.5f;
	}

	float intersectsRay(const RayGPU& ray) const
	{
		glm::vec3 invDir = 1.0f / ray._direction;
		glm::vec3 t0 = (_minPoint - ray._origin) * invDir;
		glm::vec3 t1 = (_maxPoint - ray._origin) * invDir;
		glm::vec3 tMin = min(t0, t1);
		glm::vec3 tMax = max(t0, t1);

		float tNear = glm::max(glm::max(tMin.x, tMin.y), tMin.z);
		float tFar = glm::min(glm::min(tMax.x, tMax.y), tMax.z);

		return tFar >= tNear && tFar >= 0.0f ? tNear : FLT_MAX;
	}
};

struct FrameInfoGPU
{
	glm::vec4	_backgroundUniformColor;

	glm::uvec2	_windowSize;    
	float		_blurStrength;      
	float		_defocusAngle;       

	float		_gamma;
	float		_exposure;
	glm::uint	_numEmissiveTriangles;
	glm::uint	_numBounces;     

	cudaTextureObject_t _skybox;
	glm::uint	_useSkybox;
	glm::uint	_numPixels;

	glm::uint	_numSamples;
	float		_multiplyHDR;
	float		_additionHDR;
};

struct BufferInfoGPU
{
	VertexGPU*	_vertices;
	MeshGPU*	_meshes;
	glm::u32*	_indices;
	glm::u32*	_emissiveIndices;

	Node*	_bvhNodes;
	float*		_cdfIndicesBuffer;
	float*		_cdfLightIndicesBuffer;
	float*		_noiseBuffer;

	cudaTextureObject_t* _diffuseTexturesBuffer;
	glm::uint	_numTriangles;
	glm::uint	_numMeshes;

	glm::uint	_numBvhNodes;
	glm::uint	_noiseBufferSize;
};

struct TransientInfoGPU
{
	glm::vec3	_cameraPosition;
	glm::uint	_numBounces;

	glm::vec3	_laserPosition;
	glm::uint   _relayWall;

	glm::uint   _seed;
	float		_t0;
	float		_deltaT;
	glm::uint	_temporalResolution;

	float		_t1;
	glm::uint	_unwarpCamera;
};

struct TextureResources
{
	GLuint					_id, _pbo;
	cudaGraphicsResource*	_pboResource;
	GLuint					_width, _height;

	TextureResources(GLuint width, GLuint height) : _id(0), _pbo(0), _pboResource(nullptr), _width(width), _height(height) {}
	~TextureResources() { glDeleteTextures(1, &_id); glDeleteBuffers(1, &_pbo); }

	void init(unsigned int flags = cudaGraphicsMapFlagsNone)
	{
		glGenTextures(1, &_id);
		glBindTexture(GL_TEXTURE_2D, _id);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height, 0, GL_RGBA, GL_FLOAT, nullptr);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		glGenBuffers(1, &_pbo);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, _width * _height * sizeof(float) * 4, nullptr, GL_STATIC_DRAW);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glBindTexture(GL_TEXTURE_2D, 0);

		CudaHelper::checkError(cudaGraphicsGLRegisterBuffer(&_pboResource, _pbo, flags));
	}

	float4* mapCudaPointer()
	{
		float4* imagePtr;
		size_t numBytes;

		CudaHelper::checkError(cudaGraphicsMapResources(1, &_pboResource, 0));
		CudaHelper::checkError(cudaGraphicsResourceGetMappedPointer((void**)&imagePtr, &numBytes, _pboResource));

		return imagePtr;
	}

	void unmapCudaPointer()
	{
		cudaGraphicsUnmapResources(1, &_pboResource, 0);

		glBindTexture(GL_TEXTURE_2D, _id);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, _width, _height, GL_RGBA, GL_FLOAT, 0);
		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
	}

	void resize(GLuint width, GLuint height)
	{
		_width = width;
		_height = height;

		glBindTexture(GL_TEXTURE_2D, _id);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, _width, _height, 0, GL_RGBA, GL_FLOAT, nullptr);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, _pbo);
		glBufferData(GL_PIXEL_UNPACK_BUFFER, _width * _height * sizeof(float) * 4, nullptr, GL_STATIC_DRAW);

		glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
};
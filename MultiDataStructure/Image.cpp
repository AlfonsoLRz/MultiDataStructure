#include "stdafx.h"
#include "Image.h"

Image::Image()
{
}

Image::~Image()
{
	
}

void Image::fill(const float* image, uint16_t width, uint16_t height, uint16_t numSamples)
{
	_width = width;
	_height = height;
	_depth = 4;

	_image.resize(_width * _height * _depth, .0f);

	#pragma omp parallel for
	for (int x = 0; x < _width; ++x)
	{
		for (int y = 0; y < _height; ++y)
		{
			glm::uint numContributions = 0;

			for (int sample = 0; sample < numSamples; ++sample)
			{
				float val = image[sample * _width * _height + y * _width + x];
				if (val < FLT_MAX)
				{
					_image[y * _width * _depth + x * _depth] += val;
					++numContributions;
				}
			}

			if (numContributions > 0)
				_image[y * _width * _depth + x * _depth] /= numContributions;
			else
				_image[y * _width * _depth + x * _depth] = 0.0f;

			_image[y * _width * _depth + x * _depth + 2] = _image[y * _width * _depth + x * _depth + 1] = _image[y * _width * _depth + x * _depth];
			_image[y * _width * _depth + x * _depth + 3] = 1.0f;
		}
	}
}

void Image::fill(const glm::vec3* image, uint16_t width, uint16_t height)
{
	_width = width;
	_height = height;
	_depth = 3;

	_image.resize(_width * _height * _depth, .0f);

	#pragma omp parallel for
	for (int x = 0; x < _width; ++x)
	{
		for (int y = 0; y < _height; ++y)
		{
			const glm::vec3& pixel = (image[y * _width + x] + 1.0f) / 2.0f;
			_image[y * _width * _depth + x * _depth] = pixel.x;
			_image[y * _width * _depth + x * _depth + 1] = pixel.y;
			_image[y * _width * _depth + x * _depth + 2] = pixel.z;
		}
	}
}

void Image::flipVertically()
{
	flipVertically<float>(_image, _width, _height, _depth);
}

void Image::normalize()
{
	// Normalize by channels
	float max = 0.0f;
	for (int x = 0; x < _width; ++x)
		for (int y = 0; y < _height; ++y)
			for (int c = 0; c < glm::min(_depth, static_cast<uint16_t>(3)); ++c)
				max = std::max(max, _image[y * _width * _depth + x * _depth + c]);

	for (int x = 0; x < _width; ++x)
		for (int y = 0; y < _height; ++y)
			for (int c = 0; c < glm::min(_depth, static_cast<uint16_t>(3)); ++c)
				_image[y * _width * _depth + x * _depth + c] /= max;
}

void Image::save(const std::string& filename)
{
	std::vector<unsigned char> image(_width * _height * _depth);

	#pragma omp parallel for
	for (int x = 0; x < _width; ++x)
		for (int y = 0; y < _height; ++y)
			for (int c = 0; c < _depth; ++c)
				image[y * _width * _depth + x * _depth + c] = static_cast<unsigned char>(_image[y * _width * _depth + x * _depth + c] * 255);

	flipVertically<unsigned char>(image, _width, _height, _depth);

	SOIL_save_image(filename.c_str(), SOIL_SAVE_TYPE_PNG, _width, _height, _depth, image.data());
}


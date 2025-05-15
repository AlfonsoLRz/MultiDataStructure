#pragma once

#include "Image.h"

class Texture;

class Image
{
private:
	std::vector<float> _image;
	uint16_t _width, _height, _depth;

private:
	template<class T>
	void flipVertically(std::vector<T>& image, uint16_t width, uint16_t height, uint8_t depth);

public:
	Image();
	virtual ~Image();

	void fill(const float* image, uint16_t width, uint16_t height, uint16_t numSamples);
	void fill(const glm::vec3* image, uint16_t width, uint16_t height);
	void flipVertically();
	void normalize();
	void save(const std::string& filename);
};

template<class T>
inline void Image::flipVertically(std::vector<T>& image, uint16_t width, uint16_t height, uint8_t depth)
{
	int rowSize = width * depth;
	T* bits = image.data();
	T* tempBuffer = new T[rowSize];

	for (int i = 0; i < height / 2; ++i)
	{
		T* source = bits + i * rowSize;
		T* target = bits + (height - i - 1) * rowSize;

		memcpy(tempBuffer, source, rowSize);				
		memcpy(source, target, rowSize);
		memcpy(target, tempBuffer, rowSize);
	}

	delete[] tempBuffer;
}

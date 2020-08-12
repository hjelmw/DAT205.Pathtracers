#pragma once
#include <stb_image.h>
#include <string>
#include <glm/glm.hpp>

///////////////////////////////////////////////////////////////////////////
// Simple helper class for loading HDR images with STB image
///////////////////////////////////////////////////////////////////////////
struct HDRImage
{
	int width, height, components;
	float* data = nullptr;
	HDRImage(){};
	~HDRImage()
	{
		if(data != nullptr)
			stbi_image_free(data);
	};
	void load(const std::string& filename);
	glm::vec3 sample(float u, float v);
};
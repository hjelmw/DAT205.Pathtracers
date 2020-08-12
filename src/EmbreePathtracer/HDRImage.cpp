#include "HDRImage.h"
#include <iostream>

using namespace std;
using namespace glm;

void HDRImage::load(const string& filename)
{
	stbi_set_flip_vertically_on_load(false);
	data = stbi_loadf(filename.c_str(), &width, &height, &components, 3);
	stbi_set_flip_vertically_on_load(true);
	if(data == NULL)
	{
		std::cout << "Failed to load image: " << filename << ".\n";
		exit(1);
	}
};

vec3 HDRImage::sample(float u, float v)
{
	int x = int(u * width) % width;
	int y = int(v * height) % height;
	return vec3(data[(y * width + x) * 3 + 0], data[(y * width + x) * 3 + 1], data[(y * width + x) * 3 + 2]);
}
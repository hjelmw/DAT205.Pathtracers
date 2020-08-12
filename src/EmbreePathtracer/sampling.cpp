#include "sampling.h"
#include <random>
#include "labhelper.h"
#include <omp.h>
#include <iostream>
#include <glm/glm.hpp>

using namespace glm;

namespace pathtracer
{
///////////////////////////////////////////////////////////////////////////////
// Get a random float. Note that we need one "generator" per thread, or we
// would need to lock everytime someone called randf().
///////////////////////////////////////////////////////////////////////////////
std::mt19937 generators[24]; // Assuming no more than 24 cores
float randf()
{
	return float(generators[omp_get_thread_num()]() / double(generators[omp_get_thread_num()].max()));
}

///////////////////////////////////////////////////////////////////////////
// Generate uniform points on a disc
///////////////////////////////////////////////////////////////////////////
void concentricSampleDisk(float* dx, float* dy)
{
	float r, theta;
	float u1 = randf();
	float u2 = randf();
	// Map uniform random numbers to $[-1,1]^2$
	float sx = 2 * u1 - 1;
	float sy = 2 * u2 - 1;
	// Map square to $(r,\theta)$
	// Handle degeneracy at the origin
	if(sx == 0.0 && sy == 0.0)
	{
		*dx = 0.0;
		*dy = 0.0;
		return;
	}
	if(sx >= -sy)
	{
		if(sx > sy)
		{ // Handle first region of disk
			r = sx;
			if(sy > 0.0)
				theta = sy / r;
			else
				theta = 8.0f + sy / r;
		}
		else
		{ // Handle second region of disk
			r = sy;
			theta = 2.0f - sx / r;
		}
	}
	else
	{
		if(sx <= sy)
		{ // Handle third region of disk
			r = -sx;
			theta = 4.0f - sy / r;
		}
		else
		{ // Handle fourth region of disk
			r = -sy;
			theta = 6.0f + sx / r;
		}
	}
	theta *= float(M_PI) / 4.0f;
	*dx = r * cosf(theta);
	*dy = r * sinf(theta);
}

///////////////////////////////////////////////////////////////////////////
// Generate points with a cosine distribution on the hemisphere
///////////////////////////////////////////////////////////////////////////
glm::vec3 cosineSampleHemisphere()
{
	glm::vec3 ret;
	concentricSampleDisk(&ret.x, &ret.y);
	ret.z = sqrt(max(0.f, 1.f - ret.x * ret.x - ret.y * ret.y));
	return ret;
}

///////////////////////////////////////////////////////////////////////////
// Generate a vector that is perpendicular to another
///////////////////////////////////////////////////////////////////////////
glm::vec3 perpendicular(const glm::vec3& v)
{
	if(fabsf(v.x) < fabsf(v.y))
	{
		return glm::vec3(0.0f, -v.z, v.y);
	}
	return glm::vec3(-v.z, 0.0f, v.x);
}

///////////////////////////////////////////////////////////////////////////
// Check if wi and wo are on the same side of the plane defined by n
///////////////////////////////////////////////////////////////////////////
bool sameHemisphere(const vec3& i, const vec3& o, const vec3& n)
{
	return sign(dot(o, n)) == sign(dot(i, n));
}
} // namespace pathtracer
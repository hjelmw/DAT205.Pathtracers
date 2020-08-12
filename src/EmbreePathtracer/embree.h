#pragma once
#include <embree2/rtcore.h>
#include <embree2/rtcore_ray.h>
#include "Model.h"
#include <glm/glm.hpp>
#include <map>

namespace pathtracer
{
///////////////////////////////////////////////////////////////////////////
// Add a model to the embree scene
///////////////////////////////////////////////////////////////////////////
void addModel(const labhelper::Model* model, const glm::mat4& model_matrix);

///////////////////////////////////////////////////////////////////////////
// Build an acceleration structure for the scene
///////////////////////////////////////////////////////////////////////////
void buildBVH();

///////////////////////////////////////////////////////////////////////////
// This struct is what an embree Ray must look like. It contains the
// information about the ray to be shot and (after intersect() has been
// called) the geometry the ray hit.
///////////////////////////////////////////////////////////////////////////
struct RTCORE_ALIGN(16) Ray
{
	Ray(const glm::vec3& origin = glm::vec3(0.0f),
	    const glm::vec3& direction = glm::vec3(0.0f),
	    float near = 0.0f,
	    float far = FLT_MAX)
	    : o(origin), d(direction), tnear(near), tfar(far)
	{
		geomID = RTC_INVALID_GEOMETRY_ID;
		primID = RTC_INVALID_GEOMETRY_ID;
		instID = RTC_INVALID_GEOMETRY_ID;
	}
	// Ray data
	glm::vec3 o;
	float align0;
	glm::vec3 d;
	float align1;
	float tnear = 0.0f, tfar = FLT_MAX, time = 0.0f;
	uint32_t mask = 0xFFFFFFFF;
	// Hit Data
	glm::vec3 n;
	float align2;
	float u, v;
	uint32_t geomID = RTC_INVALID_GEOMETRY_ID;
	uint32_t primID = RTC_INVALID_GEOMETRY_ID;
	uint32_t instID = RTC_INVALID_GEOMETRY_ID;
};

///////////////////////////////////////////////////////////////////////////
// This struct describes an intersection, as extracted from the Embree
// ray.
///////////////////////////////////////////////////////////////////////////
struct Intersection
{
	glm::vec3 position;
	glm::vec3 geometry_normal;
	glm::vec3 shading_normal;
	glm::vec3 wo;
	const labhelper::Material* material;
};
Intersection getIntersection(const Ray& r);

///////////////////////////////////////////////////////////////////////////
// Test a ray against the scene and find the closest intersection
///////////////////////////////////////////////////////////////////////////
bool intersect(Ray& r);

///////////////////////////////////////////////////////////////////////////
// Test whether a ray is intersected by the scene (do not return an
// intersection).
///////////////////////////////////////////////////////////////////////////
bool occluded(Ray& r);
} // namespace pathtracer


#include "Pathtracer.h"
#include <memory>
#include <iostream>
#include <map>
#include <algorithm>

#include "material.h"
#include "embree.h"
#include "sampling.h"




using namespace std;
using namespace glm;

namespace pathtracer
{
	///////////////////////////////////////////////////////////////////////////////
	// Global variables
	///////////////////////////////////////////////////////////////////////////////
	Settings settings;
	Environment environment;
	Image rendered_image;
	PointLight point_light;

	///////////////////////////////////////////////////////////////////////////
	// Restart rendering of image
	///////////////////////////////////////////////////////////////////////////
	void restart()
	{
		// No need to clear image,
		rendered_image.number_of_samples = 0;
	}

	///////////////////////////////////////////////////////////////////////////
	// On window resize, window size is passed in, actual size of pathtraced
	// image may be smaller (if we're subsampling for speed)
	///////////////////////////////////////////////////////////////////////////
	void resize(int w, int h)
	{
		rendered_image.width = w / settings.subsampling;
		rendered_image.height = h / settings.subsampling;
		rendered_image.data.resize(rendered_image.width * rendered_image.height);
		restart();
	}

	///////////////////////////////////////////////////////////////////////////
	// Return the radiance from a certain direction wi from the environment
	// map.
	///////////////////////////////////////////////////////////////////////////
	vec3 Lenvironment(const vec3& wi)
	{
		const float theta = acos(std::max(-1.0f, std::min(1.0f, wi.y)));
		float phi = atan(wi.z, wi.x);
		if (phi < 0.0f)
			phi = phi + 2.0f * M_PI;
		vec2 lookup = vec2(phi / (2.0 * M_PI), theta / M_PI);
		return environment.multiplier * environment.map.sample(lookup.x, lookup.y);
		//return vec3(0.0f, 0.0f, 0.0f);
	}

	///////////////////////////////////////////////////////////////////////////
	// Calculate the radiance going from one point (r.hitPosition()) in one
	// direction (-r.d), through path tracing.
	///////////////////////////////////////////////////////////////////////////
	vec3 Li(Ray& primary_ray)
	{
		vec3 L = vec3(0.0f);
		vec3 path_throughput = vec3(1.0);
		Ray current_ray = primary_ray;
		Ray shadowRay;

		// TASK 5: Path tracer
		for (size_t i = 0; i < settings.max_bounces; i++)
		{
			// Get the intersection information from the ray
			Intersection hit = getIntersection(current_ray);

			// Create a Material tree
			Diffuse diffuse(hit.material->m_color);
			BlinnPhong dielectric(hit.material->m_shininess, hit.material->m_fresnel, &diffuse);
			BlinnPhongMetal metal(hit.material->m_color, hit.material->m_shininess,
				hit.material->m_fresnel);
			LinearBlend metal_blend(hit.material->m_metalness, &metal, &dielectric);
			LinearBlend reflectivity_blend(hit.material->m_reflectivity, &metal_blend, &diffuse);
			BRDF& mat = reflectivity_blend;

			// Direct illumination
			shadowRay = Ray();
			shadowRay.d = normalize(point_light.position - hit.position + EPSILON);
			shadowRay.o = hit.position + EPSILON;

			if (!occluded(shadowRay))
			{
				const float distance_to_light = length(point_light.position - hit.position);
				const float falloff_factor = 1.0f / (distance_to_light * distance_to_light);
				vec3 Li = point_light.intensity_multiplier * point_light.color * falloff_factor;
				vec3 wi = normalize(point_light.position - hit.position);

				L += mat.f(wi, hit.wo, hit.shading_normal) * Li * std::max(0.0f, dot(wi, hit.shading_normal));
			}

			// Add emitted radiance from intersection			
			L += path_throughput * hit.material->m_emission * hit.material->m_color;
			
			float pdf;
			vec3 wi;

			// Sample an incoming direction (and the brdf and pdf for that direction)
			auto brdf = mat.sample_wi(wi, hit.wo, hit.shading_normal, pdf);
			auto cosine_term = abs(dot(wi, hit.shading_normal));
			
			if (pdf < EPSILON) return L; // Without this check we get a stupid memory access exception
			path_throughput = path_throughput * (brdf * cosine_term) / pdf;
			

			// If path_throughput is zero there is no need to continue
			if(path_throughput == vec3(0.0f)) 
				return L;


			// Create next ray on path (existing instance can't be reused)
			// Point next ray in direction wi
			current_ray = Ray();
			current_ray.d = wi;
			current_ray.o = hit.position;
			current_ray.o += EPSILON * hit.shading_normal; 


			//return L;

			if (!intersect(current_ray))
				return L + path_throughput * Lenvironment(current_ray.d);
		}

		return vec3(0.0f);
	}

	///////////////////////////////////////////////////////////////////////////
	// Used to homogenize points transformed with projection matrices
	///////////////////////////////////////////////////////////////////////////
	inline static glm::vec3 homogenize(const glm::vec4& p)
	{
		return glm::vec3(p * (1.f / p.w));
	}

	///////////////////////////////////////////////////////////////////////////
	// Trace one path per pixel and accumulate the result in an image
	///////////////////////////////////////////////////////////////////////////
	void tracePaths(const glm::mat4& V, const glm::mat4& P)
	{
		// Stop here if we have as many samples as we want
		if ((int(rendered_image.number_of_samples) > settings.max_paths_per_pixel)
			&& (settings.max_paths_per_pixel != 0))
		{
			return;
		}
		vec3 camera_pos = vec3(glm::inverse(V) * vec4(0.0f, 0.0f, 0.0f, 1.0f));
		// Trace one path per pixel (the omp parallel stuf magically distributes the
		// pathtracing on all cores of your CPU).
		int num_rays = 0;
		vector<vec4> local_image(rendered_image.width * rendered_image.height, vec4(0.0f));

		#pragma omp parallel for
		for (int y = 0; y < rendered_image.height; y++)
		{
			for (int x = 0; x < rendered_image.width; x++)
			{
				vec3 color;
				Ray primaryRay;
				primaryRay.o = camera_pos;
				primaryRay.d
					= vec3(randf(), randf(), randf());	// TASK 1: Jittered Sampling - Random float for vec3 direction using randf()
				// Create a ray that starts in the camera position and points toward
				// the current pixel on a virtual screen.
				vec2 screenCoord = vec2(float(x) / float(rendered_image.width),
					float(y) / float(rendered_image.height));
				// Calculate direction
				vec4 viewCoord = vec4(screenCoord.x * 2.0f - 1.0f, screenCoord.y * 2.0f - 1.0f, 1.0f, 1.0f);
				vec3 p = homogenize(inverse(P * V) * viewCoord);
				primaryRay.d = normalize(p - camera_pos);
				// Intersect ray with scene
				if (intersect(primaryRay))
				{
					// If it hit something, evaluate the radiance from that point
					color = Li(primaryRay);
				}
				else
				{
					// Otherwise evaluate environment
					color = Lenvironment(primaryRay.d);
				}
				// Accumulate the obtained radiance to the pixels color
				float n = float(rendered_image.number_of_samples);
				rendered_image.data[y * rendered_image.width + x] = 
					rendered_image.data[y * rendered_image.width + x] * (n / (n + 1.0f)) 
				  + (1.0f / (n + 1.0f)) * color;
			}
		}
		rendered_image.number_of_samples += 1;
	}
}; // namespace pathtracer

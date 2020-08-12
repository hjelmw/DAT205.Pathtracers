/*
 * Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *  * Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *  * Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *  * Neither the name of NVIDIA CORPORATION nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
 * EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
 * PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
 * EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
 * PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
 * PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
 * OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <optixu/optixu_math_namespace.h>
#include <optixu/optixu_matrix_namespace.h>

#include <optix.h>
#include <optix_math.h>

#include "helpers.h"
#include "common.h"



 //////////////////////////////// //////////////////////////////// 
 //
 //			Ray gen, closest hit, any hit (shadow) programs
 //				Also see brdf_helper.cu for helper functions
 // 
 //////////////////////////////// //////////////////////////////// 

#include "brdf_helper.cu"

//////////////////////////////// //////////////////////////////// 



using namespace optix;


// Variables from 
rtDeclareVariable(float3, front_hit_point, attribute front_hit_point, );
rtDeclareVariable(float3, geometric_normal, attribute geometric_normal, );
rtDeclareVariable(float3, shading_normal, attribute shading_normal, );


// Ray tracing variables and structs for storing results
rtDeclareVariable(optix::Ray, ray, rtCurrentRay, );
rtDeclareVariable(PerRayData_pathtrace, current_prd, rtPayload, );
rtDeclareVariable(PerRayData_pathtrace_shadow, current_prd_shadow, rtPayload, );
rtDeclareVariable(float, t_hit, rtIntersectionDistance, );
rtDeclareVariable(float, scene_epsilon, , );


// BVH declaration
rtDeclareVariable(rtObject, top_object, , );
rtDeclareVariable(rtObject, top_shadower, , );


// Le scene lights
rtBuffer<BasicLight> lights;
rtTextureSampler<float4, 2> envmap;


// Pinhole camera variables
rtDeclareVariable(float3, eye, , );
rtDeclareVariable(float3, U, , );
rtDeclareVariable(float3, V, , );
rtDeclareVariable(float3, W, , );
rtDeclareVariable(Matrix3x3, normal_matrix, , );
rtDeclareVariable(float3, bad_color, , );
rtDeclareVariable(uint2, launch_dim, rtLaunchDim, );
rtDeclareVariable(uint2, launch_index, rtLaunchIndex, );


// Buffers for storing intermediary and final output pixel value
rtBuffer<float4, 2>              output_buffer;
rtBuffer<float4, 2>              accum_buffer;
rtBuffer<float4, 2>              input_albedo_buffer;
rtBuffer<float4, 2>              input_normal_buffer;

// Ray gen variables
rtDeclareVariable(unsigned int, max_depth, , );
rtDeclareVariable(unsigned int, sqrt_num_samples, , );
rtDeclareVariable(unsigned int, frame_number, , );



RT_PROGRAM void trace_paths()
{
	//
	// Per thread variables
	//
	const size_t2 screen = output_buffer.size();
	unsigned int seed = tea<16>(screen.x * launch_index.y + launch_index.x, frame_number + 1);

	const float jitter_x = rnd(seed);
	const float jitter_y = rnd(seed);
	const float2 jitter = make_float2(jitter_x, jitter_y);
	const float2 d = (make_float2(launch_index) + jitter) / make_float2(launch_dim) * 2.f - 1.f;

	float3 ray_origin = eye;
	float3 ray_direction = normalize(d.x * U + d.y * V + W);


	//
	// Initialze per-ray data
	//
	PerRayData_pathtrace prd;
	prd.result = make_float3(0.f);
	prd.attenuation = make_float3(1.0f);
	prd.done = false;
	prd.next_done = false;
	prd.seed = seed;
	prd.depth = 0;


	//
	// Create initial ray that we fire from the camera
	//
	Ray ray = make_Ray(ray_origin, ray_direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);
	rtTrace(top_object, ray, prd);


	//
	// Store resulting color
	//
	prd.result = prd.radiance;


	//
	// Update our result
	//
	float3 result = prd.result;
	float3 albedo = prd.albedo;
	float3 normal_eyespace = (length(prd.normal) > 0.f) ? normalize(normal_matrix * prd.normal) : make_float3(0., 0., 1.);
	float3 normal = normal_eyespace;
	seed = prd.seed;


	//
	// Update the output buffer
	//
	float3 pixel_color = result;
	float3 pixel_albedo = albedo;
	float3 pixel_normal = normal;
	if (frame_number > 1)
	{
		float a = 1.0f / (float)frame_number;
		float3 old_color = make_float3(output_buffer[launch_index]);
		float3 old_albedo = make_float3(input_albedo_buffer[launch_index]);
		float3 old_normal = make_float3(input_normal_buffer[launch_index]);
		output_buffer[launch_index] = make_float4(lerp(old_color, pixel_color, a));
		input_albedo_buffer[launch_index] = make_float4(lerp(old_albedo, pixel_albedo, a), 1.0f);


		// this is not strictly a correct accumulation of normals, but it will do for this sample
		float3 accum_normal = lerp(old_normal, pixel_normal, a);
		input_normal_buffer[launch_index] = make_float4((length(accum_normal) > 0.f) ? normalize(accum_normal) : pixel_normal, 1.0f);

	}
	else
	{
		output_buffer[launch_index] = make_float4(pixel_color);
		input_albedo_buffer[launch_index] = make_float4(pixel_albedo, 1.0f);
		input_normal_buffer[launch_index] = make_float4(pixel_normal, 1.0f);
	}

}



//-----------------------------------------------------------------------------
//
//  Closest hit program
//
//-----------------------------------------------------------------------------

RT_PROGRAM void closest_hit_li()
{

	//
	// Initialize variables
	//
	float3 result = make_float3(0.0f);
	float3 path_throughput = make_float3(1.0f);

	float3 world_shading_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, shading_normal));
	float3 world_geometric_normal = normalize(rtTransformNormal(RT_OBJECT_TO_WORLD, geometric_normal));
	float3 ffnormal = faceforward(world_shading_normal, -ray.direction, world_geometric_normal);

	float3 hit_point = ray.origin + t_hit * ray.direction;

	
	// The albedo buffer should contain an approximation of the overall surface albedo (i.e. a single
	// color value approximating the ratio of irradiance reflected to the irradiance received over the
	// hemisphere). This can be approximated for very simple materials by using the diffuse color of
	// the first hit.

	if (current_prd.depth == 0)
	{
		current_prd.albedo = Kd;
		current_prd.normal = ffnormal;
	}


	//
	// Next event estimation (compute direct lighting).
	//
	unsigned int num_lights = lights.size();


	//
	// Light constants (currently we only have one light)
	//
	BasicLight light = lights[0];
	float3 light_pos = light.position;
	const float  Ldist = length(light_pos - hit_point);
	const float3 L = normalize(light_pos - hit_point);


	//
	// Store resulting shadow attenuation in this struct
	// 
	PerRayData_pathtrace_shadow shadow_prd;
	shadow_prd.inShadow = false;


	//
	// Shoot a shadow ray
	//
	Ray shadow_ray = make_Ray(hit_point, L, SHADOW_RAY_TYPE, scene_epsilon, Ldist - scene_epsilon);
	rtTrace(top_shadower, shadow_ray, shadow_prd);


	//
	// Direct lighting contribution
	//
	if (!shadow_prd.inShadow)
	{
		const float distance_to_light = length(light_pos - hit_point);
		const float falloff_factor = 1.0f / (distance_to_light * distance_to_light);
		float3 Li = 2500.0f * light.color * falloff_factor;
		float3 wi = normalize(light_pos - hit_point);

		// Add direct light contribution
		//
		// Evaluate f using
		// 1. diffuse_f
		// 2. blinnphong_f
		// 3. linearblend_f

		result += linearblend_f(wi, -ray.direction, ffnormal) * Li * max(0.0f, dot(wi, ffnormal));
	}


	//
	// Add emissive lighting (e.g ship cannons)
	//
	result += path_throughput * Ke * Kd;

	float pdf;
	float3 wi;


	// Evaluate brdf, pdf and sample new wi for redirecting next ray
	//
	// Select from
	// 1. diffuse_samplewi
	// 2. blinnphong_samplewi
	// 3. linearblend_samplewi

	float3 brdf = linearblend_samplewi(current_prd.seed, wi, -ray.direction, ffnormal, pdf);
	float cosine_term = abs(dot(wi, ffnormal));


	//
	// If pdf becomes too small we reuturn to avoid NaN values
	//
	if (pdf < scene_epsilon)
	{
		current_prd.radiance = result;
		current_prd.done = true;
		return;
	}


	//
	// Math
	//
	path_throughput = path_throughput * (brdf * cosine_term) / pdf;
	

	//
	// If attenuation becomes too small we need to break
	//
	if (path_throughput.x == 0 &&
		path_throughput.y == 0 &&
		path_throughput.z == 0 )
	{
		current_prd.radiance = result;
		current_prd.done = true;
		return;
	}


	//
	// Redirect our next ray according to what we sampled, this gives pretty reflections
	//
	current_prd.origin = hit_point;
	current_prd.direction = wi;


	//
	// Avoid pesky recursion max depth issues
	//
	if (current_prd.depth < max_depth)
	{

		//
		// Create reflection ray
		//
		Ray reflection_ray = make_Ray(current_prd.origin, current_prd.direction, RADIANCE_RAY_TYPE, scene_epsilon, RT_DEFAULT_MAX);

		//
		// Store reflection result in separate struct
		//
		PerRayData_pathtrace reflection_prd;


		//
		// Before recursing we need to pass it the currently accumulated color (and traversal depth)
		//
		reflection_prd.attenuation = path_throughput;
		reflection_prd.radiance = result;
		reflection_prd.depth = current_prd.depth + 1;


		//
		// Shoot the reflection ray (recursion time)
		//
		rtTrace(top_object, reflection_ray, reflection_prd);


		//
		// Add resulting color
		//
		result  += reflection_prd.radiance;

	}

	//
	// We are done
	//
	current_prd.radiance = result;
}



//-----------------------------------------------------------------------------
//
//  Exception program
//
//-----------------------------------------------------------------------------

// Set pixel to solid color upon failure
RT_PROGRAM void exception()
{
	// Something went wrong? Pink pixel
	output_buffer[launch_index] = make_float4(255, 0, 255, 255);
}



//-----------------------------------------------------------------------------
//
//  Environment Map Miss program
//
//-----------------------------------------------------------------------------
// !! Environment map is defined in brdf_helper.cu !!

RT_PROGRAM void envmap_miss()
{
	const float theta = acos(max(-1.0f, min(1.0f, -ray.direction.y)));
	float phi = atan2f(ray.direction.z, ray.direction.x);
	if (phi < 0.0f)
		phi = phi + 2.0f * M_PIf;
	float2 lookup = make_float2(phi / (2.0 * M_PIf), theta / M_PIf);

	current_prd.radiance = make_float3(tex2D(envmap, lookup.x, lookup.y)) * current_prd.attenuation;

	//current_prd.radiance = make_float3(0.0f, 0.0f, 0.0);
	//reflection_prd.radiance = make_float3(0.4f, 0.0f, 0.0f);

	current_prd.done = true;
	//reflection_prd.done = true;

	if (current_prd.depth == 0)
	{
		current_prd.albedo = make_float3(0, 0, 0);
		current_prd.normal = make_float3(0, 0, 0);
	}
}


RT_PROGRAM void miss()
{
	current_prd.done = true;
	current_prd.radiance = make_float3(0.0f);
	current_prd.attenuation = make_float3(0.0f);
}


//-----------------------------------------------------------------------------
//
//  Shadow any-hit
//
//-----------------------------------------------------------------------------

RT_PROGRAM void shadow()
{
	current_prd_shadow.inShadow = true;
	//current_prd.attenuation = make_float3(0.0f);
	//current_prd.radiance = make_float3(0.0f);
	rtTerminateRay();
}

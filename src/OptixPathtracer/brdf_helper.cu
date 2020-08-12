#include <optixu/optixu_math_namespace.h>
#include <optix.h>
#include <optix_math.h>

#include "common.h"
#include "random.h"
#include "helpers.h"





//////////////////////////////// //////////////////////////////// 
//
//			Helper functions for brdf, pdf and wi evaluation
//
//						**********************
//								TODO
//
//						Fix linear blending!!
//
//						**********************
// 
//////////////////////////////// //////////////////////////////// 


//////////////////////////////// //////////////////////////////// 
//
//						Material parameters
//
//////////////////////////////// //////////////////////////////// 


rtDeclareVariable(float3, Kd, , );          // Diffuse
rtDeclareVariable(float3, Ks, , );			// Specular
rtDeclareVariable(float3, Kr, , );			// Reflective
rtDeclareVariable(float3, Ka, , );			// Ambient
rtDeclareVariable(float, phong_exp, , );	// Phong lol
rtDeclareVariable(float, Pm, , );			// Metalness
rtDeclareVariable(float, Pr, , );			// Shininess
rtDeclareVariable(float, Ps, , );			// Fresnel
rtDeclareVariable(float3, Tf, , );			// Transparency
rtDeclareVariable(float3, Ke, , );			// Emissive
rtDeclareVariable(int, Kd_mapped, , );	    // Has textures



//////////////////////////////// //////////////////////////////// 
//
//						BRDF Evaluation
// 
//////////////////////////////// //////////////////////////////// 


// Simple diffuse brdf
RT_CALLABLE_PROGRAM float3 diffuse_f(float3 wi, float3 wo, float3 n)
{

	if (dot(wi, n) <= 0.0f)
		return make_float3(0.0f);
	else if (!(signbit(dot(wi, n)) == signbit(dot(wo, n))))
		return make_float3(0.0f);
	else
		return (1.0f / M_PIf) * Kd;

}


// Simple blinn phong brdf
RT_CALLABLE_PROGRAM float3 blinnphong_f(float3 wi, float3 wo, float3 n)
{
	//return make_float3(1.0f);

	float3 blinnphong_reflection_brdf;
	float3 blinnphong_refraction_brdf;
	float3 blinnphong_brdf;

	if (dot(n, wi) <= 0.0f)
		return make_float3(0.0f);

	float3 wh = normalize(wi + wo);

	float WhWi = abs(dot(wh, wi));
	float NWh = dot(n, wh);
	float NWi = dot(n, wi);
	float WoWh = dot(wo, wh);
	float NWo = dot(n, wo);

	float F = Ps + ((1.0f - Ps) * powf(1.0f - WhWi, 5.0f));

	if (Pr <= 0.0f)
		return make_float3(0.0f);

	float s = powf(NWh, Pr);
	float a2 = Pr + 2.0f;
	float D = (a2 / (2.0f * M_PIf)) * s;

	if (WoWh <= 0.0f)
		return make_float3(0.0f);

	float m1 = 2 * (NWh * NWo / WoWh);
	float m2 = 2 * (NWh * NWi / WoWh);
	float G = min(1.0f, min(m1, m2));

	if (NWo <= 0 || NWi <= 0)
		return make_float3(0.0f);
		//blinnphong_reflection_brdf = make_float3(0.0f);

	// Blinn Phong Reflection BRDF
	blinnphong_reflection_brdf = make_float3(F * D * G / (4 * NWo * NWi));

	//return blinnphong_reflection_brdf;

	float3 diffuse_brdf = diffuse_f(wo, wi, n);

	blinnphong_refraction_brdf = make_float3(1.0f - Ps) * diffuse_brdf;

	// Complete Blinn Phong BRDF
	blinnphong_brdf = blinnphong_reflection_brdf + blinnphong_refraction_brdf;
	return blinnphong_brdf;
}

// blinn phong with added calculations for metalness material parameter
RT_CALLABLE_PROGRAM float3 blinnphongmetal_f(float3 wi, float3 wo, float3 n)
{
	float3 blinnphongmetal_brdf = make_float3(0.0f);

	float3 blinnphong_reflection_brdf;
	float3 blinnphong_refraction_brdf;
	float3 blinnphong_brdf;

	if (dot(n, wi) <= 0.0f)
		return make_float3(0.0f);

	float3 wh = normalize(wi + wo);


	float WhWi = abs(dot(wh, wi));
	float NWh = dot(n, wh);
	float NWi = dot(n, wi);
	float WoWh = dot(wo, wh);
	float NWo = dot(n, wo);

	float F = Ps + ((1.0f - Ps) * powf(1.0f - WhWi, 5.0f));

	float s = powf(NWh, Pr);
	float a2 = Pr + 2.0f;
	float D = (a2 / (2.0f * M_PIf)) * s;

	if (WoWh <= 0.0f)
		return make_float3(0.0f);

	float m1 = 2 * (NWh * NWo / WoWh);
	float m2 = 2 * (NWh * NWi / WoWh);
	float G = min(1.0f, min(m1, m2));

	if (NWo <= 0 || NWi <= 0)
		return make_float3(0.0f);

	blinnphongmetal_brdf = make_float3(F * D * G / (4 * NWo * NWi));

	return blinnphongmetal_brdf * Kd;
}

// linear blend brdf between reflective and metalness blinn phong
RT_CALLABLE_PROGRAM float3 linearblend_f(float3 wi, float3 wo, float3 n)
{

	float3 metal_blend = Pm * blinnphongmetal_f(wi, wo, n) + (1.0f -Pm) * blinnphong_f(wi, wo, n);
	float3 reflectivity_blend = Ks * metal_blend + (1.0f - Ks) * diffuse_f(wi, wo, n);

	//return metal_blend;
	return reflectivity_blend;

}



//////////////////////////////// //////////////////////////////// 
//
//						Importance sampling
// 
//////////////////////////////// //////////////////////////////// 


// Samples a new wi direction for pathtracing and returns a simple diffuse brdf
RT_CALLABLE_PROGRAM float3 diffuse_samplewi(unsigned int seed, float3& wi, const float3& wo, const float3& n, float& p)
{
	float3 tangent;
	float3 bitangent;

	create_onb(n, tangent, bitangent);

	float z1 = rnd(seed);
	float z2 = rnd(seed);
	float3 sample;
	optix::cosine_sample_hemisphere(z1, z2, sample);

	wi = normalize(sample.x * tangent + sample.y * bitangent + sample.z * n);
	if (dot(wi, n) <= 0.0f)
		p = 0.0f;
	else
		p = max(0.0f, dot(n, wi)) / M_PIf;
	return diffuse_f(wi, wo, n);
}

// Samples a new wi direction for pathtracing and returns a simple blinnphong brdf
RT_CALLABLE_PROGRAM float3 blinnphong_samplewi(unsigned int seed, float3& wi, const float3& wo, const float3& n, float& p)
{
	float pdf;
	float bdf_val;
	float3 tangent;
	float3 bitangent;

	// Importance sample
	create_onb(n, tangent, bitangent);
	float2 sample = make_float2(rnd(seed), rnd(seed));

	float3 wh = sample_phong_lobe(sample, Pr, tangent, bitangent, n);

	if (dot(wo, n) <= 0.0f)
		return make_float3(0.0f);

	if (rnd(seed) < 0.5f)
	{
		wi = normalize(-wo + 2 * dot(wh, wo) * wh); // reflect wo around wh
		//wi = reflect(-wo, wh);
		float p_wh = (Pr + 1) * pow(dot(n, wh), Pr) / (2 * M_PIf);
		float p_wi = p_wh / (4 * dot(wo, wh));

		p = p_wi;
		p = p * 0.5f;

		// Reflection BRDF
		float3 blinnphong_reflection_brdf;

		if (dot(n, wi) <= 0.0f)
			return make_float3(0.0f);

		float3 wh = normalize(wi + wo);

		float WhWi = abs(dot(wh, wi));
		float NWh = dot(n, wh);
		float NWi = dot(n, wi);
		float WoWh = dot(wo, wh);
		float NWo = dot(n, wo);

		float F = Ps + ((1.0f - Ps) * powf(1.0f - WhWi, 5.0f));

		float s = powf(NWh, Pr);
		float a2 = Pr + 2.0f;
		float D = (a2 / (2.0f * M_PIf)) * s;

		if (WoWh == 0.0f)
			return make_float3(0.0f);

		float m1 = 2 * (NWh * NWo / WoWh);
		float m2 = 2 * (NWh * NWi / WoWh);
		float G = min(1.0f, min(m1, m2));

		if (NWo == 0 || NWi == 0)
			return make_float3(0.0f);
		//blinnphong_reflection_brdf = make_float3(0.0f);

		// Blinn Phong Reflection BRDF
		blinnphong_reflection_brdf = make_float3(F * D * G / (4 * NWo * NWi));
		
		return blinnphong_reflection_brdf;
	}
	else
	{
		float3 brdf = diffuse_samplewi(seed, wi, wo, n, p);
		p = p * 0.5f;
		float F = Ps + (1.0f - Ps) * pow(1.0f - abs(dot(wh, wi)), 5.0f);
		return (1 - F) * brdf;
	}
}

// Samples a new wi direction for pathtracing and returns a simple blinnphong brdf
RT_CALLABLE_PROGRAM float3 blinnphongmetal_samplewi(unsigned int seed, float3& wi, const float3& wo, const float3& n, float& p)
{
	float pdf;
	float bdf_val;
	float3 tangent;
	float3 bitangent;

	// Importance sample
	create_onb(n, tangent, bitangent);
	float2 sample = make_float2(rnd(seed), rnd(seed));

	float3 wh = sample_phong_lobe(sample, Pr, tangent, bitangent, n);

	if (dot(wo, n) <= 0.0f)
		return make_float3(0.0f);

	if (rnd(seed) < 0.5f)
	{
		wi = normalize(-wo + 2 * dot(wh, wo) * wh); // reflect wo around wh
		float p_wh = (Pr + 1) * pow(dot(n, wh), Pr) / (2 * M_PIf);
		float p_wi = p_wh / (4 * dot(wo, wh));

		p = p_wi;
		p = p * 0.5f;

		// Reflection BRDF
		float3 blinnphong_reflection_brdf;

		if (dot(n, wi) <= 0.0f)
			return make_float3(0.0f);

		float3 wh = normalize(wi + wo);

		float WhWi = abs(dot(wh, wi));
		float NWh = dot(n, wh);
		float NWi = dot(n, wi);
		float WoWh = dot(wo, wh);
		float NWo = dot(n, wo);

		float F = Ps + ((1.0f - Ps) * powf(1.0f - WhWi, 5.0f));

		if (Pr <= 0.0f)
			return make_float3(0.0f);

		float s = powf(NWh, Pr);
		float a2 = Pr + 2.0f;
		float D = (a2 / (2.0f * M_PIf)) * s;

		if (WoWh <= 0.0f)
			return make_float3(0.0f);

		float m1 = 2 * (NWh * NWo / WoWh);
		float m2 = 2 * (NWh * NWi / WoWh);
		float G = min(1.0f, min(m1, m2));

		if (NWo <= 0 || NWi <= 0)
			return make_float3(0.0f);
		//blinnphong_reflection_brdf = make_float3(0.0f);

		// Blinn Phong Reflection BRDF
		blinnphong_reflection_brdf = make_float3(F * D * G / (4 * NWo * NWi));

		return blinnphong_reflection_brdf * Kd;
	}
	else
	{
		p = p * 0.5f;
		return make_float3(0.0f);
	}
}



// Samples a new wi direction for pathtracing and returns a brdf linearly blended between reflectivity and metalness
RT_CALLABLE_PROGRAM float3 linearblend_samplewi(unsigned int seed, float3& wi, const float3& wo, const float3& n, float& p)
{
	p = 0.0f;

	// Reflectivity
	// Ks - average reflectivity
	if (rnd(seed) < (Ks.x + Ks.y + Ks.z)/3)
	{
		// Metalness
		if (rnd(seed) < Pm)
		{
			return blinnphongmetal_samplewi(seed, wi, wo, n, p);
		}
		else
		{
			return blinnphong_samplewi(seed, wi, wo, n, p);
		}
	}
	else
	{
		return diffuse_samplewi(seed, wi, wo, n, p);
	}
}
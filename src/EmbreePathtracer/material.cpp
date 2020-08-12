#include "material.h"
#include "sampling.h"

namespace pathtracer
{
	///////////////////////////////////////////////////////////////////////////
	// A Lambertian (diffuse) material
	///////////////////////////////////////////////////////////////////////////
	vec3 Diffuse::f(const vec3& wi, const vec3& wo, const vec3& n)
	{
		if (dot(wi, n) <= 0.0f)
			return vec3(0.0f);
		if (!sameHemisphere(wi, wo, n))
			return vec3(0.0f);
		return (1.0f / M_PI) * color;
	}

	vec3 Diffuse::sample_wi(vec3& wi, const vec3& wo, const vec3& n, float& p)
	{
		vec3 tangent = normalize(perpendicular(n));
		vec3 bitangent = normalize(cross(tangent, n));
		vec3 sample = cosineSampleHemisphere();
		wi = normalize(sample.x * tangent + sample.y * bitangent + sample.z * n);
		if (dot(wi, n) <= 0.0f)
			p = 0.0f;
		else
			p = max(0.0f, dot(n, wi)) / M_PI;
		return f(wi, wo, n);
	}

	///////////////////////////////////////////////////////////////////////////
	// A Blinn Phong Dielectric Microfacet BRFD
	///////////////////////////////////////////////////////////////////////////
	vec3 BlinnPhong::refraction_brdf(const vec3& wi, const vec3& wo, const vec3& n)
	{
		if (refraction_layer != NULL)
		{
			vec3 wh = normalize(wi + wo);
			float WhWi = abs(dot(wh, wi));
			float F = R0 + ((1.0 - R0) * pow(1.0 - WhWi, 5.0));

			return vec3((1.0f - F)) * refraction_layer->f(wi, wo, n);
		}

		return vec3(0.0f);
	}

	// TASK 3: Blinn Phong BRDF
	vec3 BlinnPhong::reflection_brdf(const vec3& wi, const vec3& wo, const vec3& n)
	{

		if (dot(n, wi) <= 0.0f)
			return vec3(0.0f);

		vec3 wh = normalize(wi + wo);

		float WhWi = abs(dot(wh, wi));
		float NWh = dot(n, wh);
		float NWi = dot(n, wi);
		float WoWh = dot(wo, wh);
		float NWo = dot(n, wo);

		float F = R0 + ((1.0 - R0) * pow(1.0 - WhWi, 5.0));

		//return vec3(F);

		float s = pow(NWh, shininess);
		float a2 = shininess + 2;
		float D = (a2 / (2 * M_PI)) * s;

		//return vec3(D);

		float m1 = 2 * (NWh * NWo / WoWh);
		float m2 = 2 * (NWh * NWi / WoWh);
		float G = min(1.0f, min(m1, m2));

		//return vec3(G);

		if (NWo <= 0 || NWi <= 0)
			return vec3(0);

		float brdf = F * D * G / (4 * NWo * NWi);

		return vec3(brdf);
	}

	vec3 BlinnPhong::f(const vec3& wi, const vec3& wo, const vec3& n)
	{
		return reflection_brdf(wi, wo, n) + refraction_brdf(wi, wo, n);
	}

	vec3 BlinnPhong::sample_wi(vec3& wi, const vec3& wo, const vec3& n, float& p)
	{
		vec3 tangent = normalize(perpendicular(n));
		vec3 bitangent = normalize(cross(tangent, n));
		float phi = 2.0f * M_PI * randf();
		float cos_theta = pow(randf(), 1.0f / (shininess + 1));
		float sin_theta = sqrt(max(0.0f, 1.0f - cos_theta * cos_theta));
		vec3 wh = normalize(sin_theta * cos(phi) * tangent +
			sin_theta * sin(phi) * bitangent +
			cos_theta * n);
		if (dot(wo, n) <= 0.0f) return vec3(0.0f);

		if (randf() < 0.5)
		{
			wi = normalize(-wo + 2 * dot(wh, wo) * wh); // reflect wo around wh
			float p_wh = (shininess + 1) * pow(dot(n, wh), shininess) / (2 * M_PI);
			float p_wi = p_wh / (4 * dot(wo, wh));
			
			p = p_wi;
			p = p * 0.5f;

			return reflection_brdf(wi, wo, n);
		}
		else 
		{
			if (refraction_layer == NULL)
				return vec3(0.0f);
			vec3 brdf = refraction_layer->sample_wi(wi, wo, n, p);
			p = p * 0.5f;
			float F = R0 + (1.0f - R0) * pow(1.0f - abs(dot(wh, wi)), 5.0f);
			return (1 - F) * brdf;
		}
	}
	///////////////////////////////////////////////////////////////////////////
	// A Blinn Phong Metal Microfacet BRFD (extends the BlinnPhong class)
	///////////////////////////////////////////////////////////////////////////
	vec3 BlinnPhongMetal::refraction_brdf(const vec3& wi, const vec3& wo, const vec3& n)
	{
		return vec3(0.0f);
	}
	vec3 BlinnPhongMetal::reflection_brdf(const vec3& wi, const vec3& wo, const vec3& n)
	{
		return BlinnPhong::reflection_brdf(wi, wo, n) * color;
	};

	///////////////////////////////////////////////////////////////////////////
	// A Linear Blend between two BRDFs
	///////////////////////////////////////////////////////////////////////////
	vec3 LinearBlend::f(const vec3& wi, const vec3& wo, const vec3& n)
	{
		return w * bsdf0->f(wi, wo, n) + (1 - w) * bsdf1->f(wi, wo, n);
	}

	vec3 LinearBlend::sample_wi(vec3& wi, const vec3& wo, const vec3& n, float& p)
	{
		p = 0.0f;
		if (randf() < w)
		{
			return bsdf0->sample_wi(wi, wo, n, p);
		}
		else
		{
			return bsdf1->sample_wi(wi, wo, n, p);
		}

	}

	///////////////////////////////////////////////////////////////////////////
	// A perfect specular refraction.
	///////////////////////////////////////////////////////////////////////////
} // namespace pathtracer
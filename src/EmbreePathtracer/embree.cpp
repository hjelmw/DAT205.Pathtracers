#include "embree.h"
#include <iostream>
#include <map>


using namespace std;
using namespace glm;

namespace pathtracer
{
///////////////////////////////////////////////////////////////////////////
// Global variables
///////////////////////////////////////////////////////////////////////////
RTCDevice embree_device;
RTCScene embree_scene;

///////////////////////////////////////////////////////////////////////////
// Build an acceleration structure for the scene
///////////////////////////////////////////////////////////////////////////
void buildBVH()
{
	cout << "Embree building BVH..." << flush;
	rtcCommit(embree_scene);
	cout << "done.\n";
}

///////////////////////////////////////////////////////////////////////////
// Called when there is an embree error
///////////////////////////////////////////////////////////////////////////
void embreeErrorHandler(const RTCError code, const char* str)
{
	cout << "Embree ERROR: " << str << endl;
	exit(1);
}

///////////////////////////////////////////////////////////////////////////
// Used to map an Embree geometry ID to our scene Meshes and Materials
///////////////////////////////////////////////////////////////////////////
map<uint32_t, const labhelper::Model*> map_geom_ID_to_model;
map<uint32_t, const labhelper::Mesh*> map_geom_ID_to_mesh;

///////////////////////////////////////////////////////////////////////////
// Add a model to the embree scene
///////////////////////////////////////////////////////////////////////////
void addModel(const labhelper::Model* model, const mat4& model_matrix)
{
	///////////////////////////////////////////////////////////////////////
	// Lazy initialize embree on first use
	///////////////////////////////////////////////////////////////////////
	cout << "Initializing embree..." << flush;
	static bool embree_is_initialized = false;
	if(!embree_is_initialized)
	{
		embree_is_initialized = true;
		embree_device = rtcNewDevice();
		rtcDeviceSetErrorFunction(embree_device, embreeErrorHandler);
		embree_scene = rtcDeviceNewScene(embree_device, RTC_SCENE_STATIC, RTC_INTERSECT1);
	}
	cout << "done.\n";

	///////////////////////////////////////////////////////////////////////
	// Transform and add each mesh in the model as a geometry in embree,
	// and create mappings so that we can connect an embree geom_ID to a
	// Material.
	///////////////////////////////////////////////////////////////////////
	cout << "Adding " << model->m_name << " to embree scene..." << flush;
	for(auto& mesh : model->m_meshes)
	{
		uint32_t geom_ID = rtcNewTriangleMesh(embree_scene, RTC_GEOMETRY_STATIC,
		                                      mesh.m_number_of_vertices / 3, mesh.m_number_of_vertices);
		map_geom_ID_to_mesh[geom_ID] = &mesh;
		map_geom_ID_to_model[geom_ID] = model;
		// Transform and commit vertices
		vec4* embree_vertices = (vec4*)rtcMapBuffer(embree_scene, geom_ID, RTC_VERTEX_BUFFER);
		for(uint32_t i = 0; i < mesh.m_number_of_vertices; i++)
		{
			embree_vertices[i] = model_matrix * vec4(model->m_positions[mesh.m_start_index + i], 1.0f);
		}
		rtcUnmapBuffer(embree_scene, geom_ID, RTC_VERTEX_BUFFER);
		// Commit triangle indices
		int* embree_tri_idxs = (int*)rtcMapBuffer(embree_scene, geom_ID, RTC_INDEX_BUFFER);
		for(uint32_t i = 0; i < mesh.m_number_of_vertices; i++)
		{
			embree_tri_idxs[i] = i;
		}
		rtcUnmapBuffer(embree_scene, geom_ID, RTC_INDEX_BUFFER);
	}
	cout << "done.\n";
}

///////////////////////////////////////////////////////////////////////////
// Extract an intersection from an embree ray.
///////////////////////////////////////////////////////////////////////////
Intersection getIntersection(const Ray& r)
{
	const labhelper::Model* model = map_geom_ID_to_model[r.geomID];
	const labhelper::Mesh* mesh = map_geom_ID_to_mesh[r.geomID];
	Intersection i;
	i.material = &(model->m_materials[mesh->m_material_idx]);
	vec3 n0 = model->m_normals[((mesh->m_start_index / 3) + r.primID) * 3 + 0];
	vec3 n1 = model->m_normals[((mesh->m_start_index / 3) + r.primID) * 3 + 1];
	vec3 n2 = model->m_normals[((mesh->m_start_index / 3) + r.primID) * 3 + 2];
	float w = 1.0f - (r.u + r.v);
	i.shading_normal = normalize(w * n0 + r.u * n1 + r.v * n2);
	i.geometry_normal = -normalize(r.n);
	i.position = r.o + r.tfar * r.d;
	i.wo = normalize(-r.d);
	return i;
}

///////////////////////////////////////////////////////////////////////////
// Test a ray against the scene and find the closest intersection
///////////////////////////////////////////////////////////////////////////
bool intersect(Ray& r)
{
	rtcIntersect(embree_scene, *((RTCRay*)&r));
	return r.geomID != RTC_INVALID_GEOMETRY_ID;
}

///////////////////////////////////////////////////////////////////////////
// Test whether a ray is intersected by the scene (do not return an
// intersection).
///////////////////////////////////////////////////////////////////////////
bool occluded(Ray& r)
{
	rtcOccluded(embree_scene, *((RTCRay*)&r));
	return r.geomID != RTC_INVALID_GEOMETRY_ID;
}
} // namespace pathtracer
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

 //-----------------------------------------------------------------------------
 //
 // optixMeshViewer: simple interactive mesh viewer 
 //
 //-----------------------------------------------------------------------------


# include <GL/glew.h>
# if defined( _WIN32 )
#   include <GL/wglew.h>
#   include <GL/freeglut.h>
# else
#   include <GL/glut.h>
# endif


#include <optixu/optixpp_namespace.h>
#include <optixu/optixu_math_stream_namespace.h>

#include <sutil.h>
#include "common.h"
#include <Arcball.h>
#include <OptiXMesh.h>

#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdint.h>
#include <OptiXMesh.cpp>

using namespace optix;

const char* const SAMPLE_NAME = "pathtracerGPU";

//------------------------------------------------------------------------------
//
// Globals
//
//------------------------------------------------------------------------------

// Optix globals
optix::Context        context;
uint32_t       width = 1366;
uint32_t       height = 768;
bool           use_pbo = true;
bool           use_tri_api = true;
bool           ignore_mats = false;
bool		   postprocessing_needs_init = true;
optix::Aabb    aabb;


// Pathtracing variables
int            frame_number = 1;
int            samples_per_pixel = 1;
int            max_depth = 8;
Program        pgram_intersection = 0;
Program        pgram_bounding_box = 0;


// Camera state
float3         camera_up;
float3         camera_lookat;
float3         camera_eye;
Matrix4x4      camera_rotate;
sutil::Arcball arcball;
bool		   camera_changed = true;


// Mouse state
int2           mouse_prev_pos;
int            mouse_button;


// Optix AI Denoising variables
CommandList commandListWithDenoiser;
CommandList commandListWithoutDenoiser;

PostprocessingStage tonemapStage;
PostprocessingStage denoiserStage;
Buffer denoisedBuffer;
Buffer emptyBuffer;
Buffer trainingDataBuffer;

// number of frames that show the original image before switching on denoising
int denoise_frame_number = 0;

// Defines the amount of the original image that is blended with the denoised result
// ranging from 0.0 to 1.0
float denoiseBlend = 0.0f;

// Defines which buffer to show.
// 0 - denoised 1 - original, 2 - tonemapped, 3 - albedo, 4 - normal
int showBuffer = 0;

// The denoiser mode.
// 0 - RGB only, 1 - RGB + albedo, 2 - RGB + albedo + normals
int denoiseMode = 1;

// The path to the training data file set with -t or empty
std::string training_file;

// The path to the second training data file set with -t2 or empty
std::string training_file_2;

// Toggles between using custom training data (if set) or the built in training data.
bool showDenoiseBuffer = false;

// Toggles the custom data between the one specified with -t1 and -t2, if available.
bool useFirstTrainingDataPath = true;

// Text string for indicating currently used buffer
std::string bufferInfo;



//------------------------------------------------------------------------------
//
// Forward declarations 
//
//------------------------------------------------------------------------------

struct UsageReportLogger;

Buffer getOutputBuffer();
void destroyContext();
void registerExitHandler();
void createContext(int usage_report_level, UsageReportLogger* logger);
void loadMeshes(const std::string& filename);
void setupCamera();
void setupLights();
void updateCamera();
void glutInitialize(int* argc, char** argv);
void glutRun();

void glutDisplay();
void glutKeyboardPress(unsigned char k, int x, int y);
void glutMousePress(int button, int state, int x, int y);
void glutMouseMotion(int x, int y);
void glutResize(int w, int h);
void setupPostProcessing();


//------------------------------------------------------------------------------
//
//  Helper functions
//
//
//	Want a buffer?
//
//------------------------------------------------------------------------------
Buffer getOutputBuffer()
{
	return context["output_buffer"]->getBuffer();
}
Buffer getInputBuffer()
{
	return context["input_buffer"]->getBuffer();
}
Buffer getTonemappedBuffer()
{
	return context["tonemapped_buffer"]->getBuffer();
}
Buffer getAlbedoBuffer()
{
	return context["input_albedo_buffer"]->getBuffer();
}
Buffer getNormalBuffer()
{
	return context["input_normal_buffer"]->getBuffer();
}



void destroyContext()
{
	if (context)
	{
		context->destroy();
		context = 0;
	}
}


struct UsageReportLogger
{
	void log(int lvl, const char* tag, const char* msg)
	{
		std::cout << "[" << lvl << "][" << std::left << std::setw(12) << tag << "] " << msg;
	}
};

// Static callback
void usageReportCallback(int lvl, const char* tag, const char* msg, void* cbdata)
{
	// Route messages to a C++ object (the "logger"), as a real app might do.
	// We could have printed them directly in this simple case.

	UsageReportLogger* logger = reinterpret_cast<UsageReportLogger*>(cbdata);
	logger->log(lvl, tag, msg);
}

void registerExitHandler()
{
	// register shutdown handler
#ifdef _WIN32
	glutCloseFunc(destroyContext);  // this function is freeglut-only
#else
	atexit(destroyContext);
#endif
}



//------------------------------------------------------------------------------
//
// Create Optix context and setup some global variables
//
//------------------------------------------------------------------------------
void createContext(int usage_report_level, UsageReportLogger* logger)
{
	// Set up context
	context = Context::create();
	context->setRayTypeCount(2);
	context->setEntryPointCount(1);
	context->setStackSize(4640);
	context->setMaxTraceDepth(31);

	if (usage_report_level > 0)
	{
		context->setUsageReportCallback(usageReportCallback, usage_report_level, logger);
	}

	context["frame"]->setUint(0u);
	context["scene_epsilon"]->setFloat(1.e-4f);

	// Output buffer
	Buffer output_buffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
	context["output_buffer"]->set(output_buffer);

	// Accumulation buffer
	Buffer accum_buffer = context->createBuffer(RT_BUFFER_INPUT_OUTPUT | RT_BUFFER_GPU_LOCAL,
		RT_FORMAT_FLOAT4, width, height);
	context["accum_buffer"]->set(accum_buffer);

	Buffer tonemappedBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
	context["tonemapped_buffer"]->set(tonemappedBuffer);

	Buffer albedoBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
	context["input_albedo_buffer"]->set(albedoBuffer);

	// The normal buffer use float4 for performance reasons, the fourth channel will be ignored.
	Buffer normalBuffer = sutil::createInputOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
	context["input_normal_buffer"]->set(normalBuffer);

	denoisedBuffer = sutil::createOutputBuffer(context, RT_FORMAT_FLOAT4, width, height, use_pbo);
	emptyBuffer = context->createBuffer(RT_BUFFER_OUTPUT, RT_FORMAT_FLOAT4, 0, 0);
	trainingDataBuffer = context->createBuffer(RT_BUFFER_INPUT, RT_FORMAT_UNSIGNED_BYTE, 0);

	// Ray generation program
	//const char* ptx = sutil::getPtxString(SAMPLE_NAME, "pinhole_camera.cu");
	const char* ptx = sutil::getPtxString(SAMPLE_NAME, "pathtracer.cu");
	Program ray_gen_program = context->createProgramFromPTXString(ptx, "trace_paths");
	context->setRayGenerationProgram(0, ray_gen_program);

	// Exception program
	Program exception_program = context->createProgramFromPTXString(ptx, "exception");
	context->setExceptionProgram(0, exception_program);
	// Miss program
	//context->setMissProgram(0, context->createProgramFromPTXString(ptx, "miss"));
	
	// Miss program
	const float3 default_color = make_float3(1000.0f, 0.0f, 0.0f);
	const std::string texpath = std::string(sutil::samplesDir()) + "/scenes/envmaps/001.hdr";
	context["envmap"]->setTextureSampler(sutil::loadTexture(context, texpath, default_color));
	context->setMissProgram(0, context->createProgramFromPTXString(ptx, "envmap_miss"));
	
	// Note: high max depth for reflection and refraction through glass
	context["sqrt_num_samples"]->setUint(samples_per_pixel);

	// Trace settings
	context["max_depth"]->setUint(max_depth);

	context["bad_color"]->setFloat(0.0f, 0.0f, 0.0f); // Super magenta to make sure it doesn't get averaged out in the progressive rendering.
}



//------------------------------------------------------------------------------
//
// Load vector of meshes and put them in world space positions
//
//------------------------------------------------------------------------------
void loadMeshes(Context ctx, std::vector<std::string> filenames, std::vector<float3> positions)
{
	std::cerr << "Creating geometry ... \n";

	// Specify BVH structure
	Acceleration accel = ctx->createAcceleration("Trbvh");

	// BVH
	GeometryGroup m_top_object;
	m_top_object = ctx->createGeometryGroup();
	m_top_object->setAcceleration(accel);


	// Setup closest and any hit programs for our meshes
	const char* ptx = sutil::getPtxString(SAMPLE_NAME, "pathtracer.cu");
	const char* ptx2 = sutil::getPtxString(SAMPLE_NAME, "triangle_mesh.cu");
	Program closest_hit = context->createProgramFromPTXString(ptx, "closest_hit_li");
	Program any_hit = context->createProgramFromPTXString(ptx, "shadow");

	// Intersection and out of bounds program for our meshes
	Program intersection = context->createProgramFromPTXString(ptx2, "mesh_intersect");
	Program bounds = context->createProgramFromPTXString(ptx2, "mesh_bounds");


	for (int i = 0; i < filenames.size(); ++i)
	{
		std::cerr << "Loading mesh: " + filenames[i] + "... ";

		OptiXMesh omesh;
		omesh.context = ctx;

		// True uses default triangle ray intersection test
		omesh.use_tri_api = false;


		// Change default programs
		omesh.closest_hit = closest_hit;
		omesh.any_hit = any_hit;
		omesh.bounds = bounds;
		omesh.intersection = intersection;


		// Optix loads our mesh
		loadMesh(filenames[i], omesh, Matrix4x4::translate(positions[i]));


		// Add to BVH
		aabb.include(omesh.bbox_min, omesh.bbox_max);
		m_top_object->addChild(omesh.geom_instance);
		std::cerr << "done \n";
	}


	// Set top BVH node
	ctx["top_object"]->set(m_top_object);
	ctx["top_shadower"]->set(m_top_object);

	std::cerr << "Geometry loaded" << std::endl;
}



//------------------------------------------------------------------------------
//
// Initilize camera position and view direction
//
//------------------------------------------------------------------------------
void setupCamera()
{
	const float max_dim = fmaxf(aabb.extent(0), aabb.extent(1)); // max of x, y components

	camera_eye = aabb.center() + make_float3(-75.0f, 30.0f, max_dim * 0.7);
	camera_lookat = aabb.center();
	camera_up = make_float3(0.0f, 1.0f, 0.0f);

	camera_rotate = Matrix4x4::identity();
}



//------------------------------------------------------------------------------
//
// Setup our point light
//
//------------------------------------------------------------------------------
void setupLights()
{
	BasicLight lights[] = {
		{ make_float3(10.0f, 40.0f, 10.0f), make_float3(1.0f, 1.0f, 1.0f), 1 }
	};


	// Light buffer
	Buffer light_buffer = context->createBuffer(RT_BUFFER_INPUT);
	light_buffer->setFormat(RT_FORMAT_USER);
	light_buffer->setElementSize(sizeof(BasicLight));
	light_buffer->setSize(sizeof(lights) / sizeof(lights[0]));
	memcpy(light_buffer->map(), lights, sizeof(lights));
	light_buffer->unmap();


	context["lights"]->set(light_buffer);
}



//------------------------------------------------------------------------------
//
// Upate camera view matrix
//
//------------------------------------------------------------------------------
void updateCamera()
{
	const float vfov = 35.0f;
	const float aspect_ratio = static_cast<float>(width) /
		static_cast<float>(height);

	float3 camera_u, camera_v, camera_w;
	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		camera_u, camera_v, camera_w, true);

	const Matrix4x4 frame = Matrix4x4::fromBasis(
		normalize(camera_u),
		normalize(camera_v),
		normalize(-camera_w),
		camera_lookat);
	const Matrix4x4 frame_inv = frame.inverse();
	// Apply camera rotation twice to match old SDK behavior
	const Matrix4x4 trans = frame * camera_rotate * camera_rotate * frame_inv;

	camera_eye = make_float3(trans * make_float4(camera_eye, 1.0f));
	camera_lookat = make_float3(trans * make_float4(camera_lookat, 1.0f));
	camera_up = make_float3(trans * make_float4(camera_up, 0.0f));

	sutil::calculateCameraVariables(
		camera_eye, camera_lookat, camera_up, vfov, aspect_ratio,
		camera_u, camera_v, camera_w, true);

	camera_rotate = Matrix4x4::identity();

	if (camera_changed) // reset accumulation
	{
		showDenoiseBuffer = false;
		frame_number = 1;
	}
	camera_changed = false;

	context["frame_number"]->setUint(frame_number++);
	context["eye"]->setFloat(camera_eye);
	context["U"]->setFloat(camera_u);
	context["V"]->setFloat(camera_v);
	context["W"]->setFloat(camera_w);

	const Matrix4x4 current_frame_inv = Matrix4x4::fromBasis(
		normalize(camera_u),
		normalize(camera_v),
		normalize(-camera_w),
		camera_lookat).inverse();
	Matrix3x3 normal_matrix = make_matrix3x3(current_frame_inv);

	context["normal_matrix"]->setMatrix3x3fv(false, normal_matrix.getData());
}


void glutInitialize(int* argc, char** argv)
{
	glutInit(argc, argv);
	glutInitDisplayMode(GLUT_RGB | GLUT_ALPHA | GLUT_DEPTH | GLUT_DOUBLE);
	glutInitWindowSize(width, height);
	glutInitWindowPosition(100, 100);
	glutCreateWindow(SAMPLE_NAME);
	glutHideWindow();
}


//------------------------------------------------------------------------------
//
// Finalize setup and enter main async render loop
//
//------------------------------------------------------------------------------
void glutRun()
{
	// Initialize GL state                                                            
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	glOrtho(0, 1, 0, 1, -1, 1);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	glViewport(0, 0, width, height);

	glutShowWindow();
	glutReshapeWindow(width, height);

	// register glut callbacks
	glutDisplayFunc(glutDisplay);
	glutIdleFunc(glutDisplay);
	glutReshapeFunc(glutResize);
	glutKeyboardFunc(glutKeyboardPress);
	glutMouseFunc(glutMousePress);
	glutMotionFunc(glutMouseMotion);

	registerExitHandler();

	std::cerr << "Starting... ";
	glutMainLoop();
}


void setupPostprocessing()
{

	if (!tonemapStage)
	{
		// create stages only once: they will be reused in several command lists without being re-created
		tonemapStage = context->createBuiltinPostProcessingStage("TonemapperSimple");
		denoiserStage = context->createBuiltinPostProcessingStage("DLDenoiser");
		if (trainingDataBuffer)
		{
			Variable trainingBuff = denoiserStage->declareVariable("training_data_buffer");
			trainingBuff->set(trainingDataBuffer);
		}

		tonemapStage->declareVariable("input_buffer")->set(getOutputBuffer());
		tonemapStage->declareVariable("output_buffer")->set(getTonemappedBuffer());
		tonemapStage->declareVariable("exposure")->setFloat(2.2f);
		tonemapStage->declareVariable("gamma")->setFloat(1.1f);
		tonemapStage->declareVariable("hdr")->setFloat(1);

		denoiserStage->declareVariable("input_buffer")->set(getTonemappedBuffer());
		denoiserStage->declareVariable("output_buffer")->set(denoisedBuffer);
		denoiserStage->declareVariable("hdr")->setUint(0);

		denoiserStage->declareVariable("blend")->setFloat(denoiseBlend);
		denoiserStage->declareVariable("input_albedo_buffer");
		denoiserStage->declareVariable("input_normal_buffer");
	}

	if (commandListWithDenoiser)
	{
		commandListWithDenoiser->destroy();
		commandListWithoutDenoiser->destroy();
	}

	// Create two command lists with two postprocessing topologies we want:
	// One with the denoiser stage, one without. Note that both share the same
	// tonemap stage.

	commandListWithDenoiser = context->createCommandList();
	commandListWithDenoiser->appendLaunch(0, width, height);
	commandListWithDenoiser->appendPostprocessingStage(tonemapStage, width, height);
	commandListWithDenoiser->appendPostprocessingStage(denoiserStage, width, height);
	commandListWithDenoiser->finalize();

	commandListWithoutDenoiser = context->createCommandList();
	commandListWithoutDenoiser->appendLaunch(0, width, height);
	commandListWithoutDenoiser->appendPostprocessingStage(tonemapStage, width, height);
	commandListWithoutDenoiser->finalize();

	postprocessing_needs_init = false;
}



//------------------------------------------------------------------------------
//
//  GLUT callbacks
//
//------------------------------------------------------------------------------
void glutDisplay()
{
	updateCamera();


	if (postprocessing_needs_init)
	{
		setupPostprocessing();
	}


	Variable(denoiserStage->queryVariable("blend"))->setFloat(denoiseBlend);


	if (!showDenoiseBuffer)
	{
		commandListWithoutDenoiser->execute();
		// gamma correction already applied by tone mapper, avoid doing it twice
		sutil::displayBufferGL(getOutputBuffer(), BUFFER_PIXEL_FORMAT_DEFAULT, true);

		denoise_frame_number = frame_number;
		sutil::displayText("Accumulating frames...", 10, 55);
		sutil::displayText("Press F to denoise", 10, 40);
	}
	else
	{
		commandListWithDenoiser->execute();
		sutil::displayBufferGL(denoisedBuffer, BUFFER_PIXEL_FORMAT_DEFAULT, true);

		char str[64];
		sprintf(str, "Denoising at frame #%d", denoise_frame_number);
		sutil::displayText(str, 10, 55);
		sutil::displayText("Press F to toggle back to accumulation buffer", 10, 40);
	}

	static unsigned frame_count = 0;
	sutil::displayFps(frame_count++);
	char str[64];
	sprintf(str, "Frame   #%d", frame_number);
	sutil::displayText(str, 10, 25);
	sutil::displayText("Hello Optix!", 140, 10);



	glutSwapBuffers();
}



//------------------------------------------------------------------------------
//
// Keyboard input
//
//------------------------------------------------------------------------------
void glutKeyboardPress(unsigned char k, int x, int y)
{

	switch (k)
	{
	case('q'):
	case(27): // ESC
	{
		destroyContext();
		exit(0);
	}
	case('s'):
	{
		const std::string outputImage = std::string(SAMPLE_NAME) + ".ppm";
		std::cerr << "Saving current frame to '" << outputImage << "'\n";
		sutil::displayBufferPPM(outputImage.c_str(), getOutputBuffer());
		break;
	}
	case('f'):
		showDenoiseBuffer = !showDenoiseBuffer;
	}
}



void glutMousePress(int button, int state, int x, int y)
{
	if (state == GLUT_DOWN)
	{
		mouse_button = button;
		mouse_prev_pos = make_int2(x, y);
	}
	else
	{
		// nothing
	}
}



//------------------------------------------------------------------------------
//
// Moving the mouse
//
//------------------------------------------------------------------------------
void glutMouseMotion(int x, int y)
{
	if (mouse_button == GLUT_RIGHT_BUTTON)
	{
		const float dx = static_cast<float>(x - mouse_prev_pos.x) /
			static_cast<float>(width);
		const float dy = static_cast<float>(y - mouse_prev_pos.y) /
			static_cast<float>(height);
		const float dmax = fabsf(dx) > fabs(dy) ? dx : dy;
		const float scale = fminf(dmax, 0.9f);
		camera_eye = camera_eye + (camera_lookat - camera_eye) * scale;
		camera_changed = true;
	}
	else if (mouse_button == GLUT_LEFT_BUTTON)
	{
		const float2 from = { static_cast<float>(mouse_prev_pos.x),
							  static_cast<float>(mouse_prev_pos.y) };
		const float2 to = { static_cast<float>(x),
							  static_cast<float>(y) };

		const float2 a = { from.x / width, from.y / height };
		const float2 b = { to.x / width, to.y / height };

		camera_rotate = arcball.rotate(b, a);
		camera_changed = true;

	}

	mouse_prev_pos = make_int2(x, y);
}



//------------------------------------------------------------------------------
//
// Resize window
//
//------------------------------------------------------------------------------
void glutResize(int w, int h)
{
	if (w == (int)width && h == (int)height) return;
	camera_changed = true;
	width = w;
	height = h;
	sutil::ensureMinimumSize(width, height);

	sutil::resizeBuffer(context["accum_buffer"]->getBuffer(), width, height);
	sutil::resizeBuffer(getOutputBuffer(), width, height);
	sutil::resizeBuffer(getTonemappedBuffer(), width, height);
	sutil::resizeBuffer(getAlbedoBuffer(), width, height);
	sutil::resizeBuffer(getNormalBuffer(), width, height);
	sutil::resizeBuffer(denoisedBuffer, width, height);


	glViewport(0, 0, width, height);

	postprocessing_needs_init = true;

	glutPostRedisplay();

}



//------------------------------------------------------------------------------
//
// Main
//
//------------------------------------------------------------------------------
int main(int argc, char** argv)
{
	std::string out_file;


	// Load out models
	std::vector<std::string> mesh_filenames =
	{
		std::string(sutil::samplesDir()) + "/scenes/NewShip.obj",
		std::string(sutil::samplesDir()) + "/scenes/LandingPad.obj",
	};

	std::vector<float3> mesh_positions =
	{
		make_float3(0.0f, 10.0f, 0.0f),
		make_float3(0.0f, 0.0f, 0.0f),
	};

	int usage_report_level = 0;
	try
	{
		glutInitialize(&argc, argv);
		glewInit();

		UsageReportLogger logger;
		createContext(usage_report_level, &logger);

		//loadMesh(mesh_filenames[0], make_float3( 0.0f, 10.0f, 0.0f ) );

		loadMeshes(context, mesh_filenames, mesh_positions);

		setupCamera();
		setupLights();

		context->validate();

		if (out_file.empty())
		{
			glutRun();
		}
		else
		{
			updateCamera();
			context->launch(0, width, height);
			sutil::displayBufferPPM(out_file.c_str(), getOutputBuffer());
			destroyContext();
		}
		return 0;
	}
	SUTIL_CATCH(context->get())
}


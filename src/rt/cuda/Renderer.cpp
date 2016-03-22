/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include "cuda/Renderer.hpp"
#include "cuda/RendererKernels.hpp"
#include "gui/Window.hpp"
#include "io/File.hpp"
#include "bvh/HLBVH/HLBVHBuilder.hpp"

#define SBVH

#ifndef SBVH
#include "persistentds/CudaPersistentBVHBuilder.hpp"
#else
#include "persistentds/CudaPersistentSBVHBuilder.hpp"
#endif
#include "persistentds/CudaPersistentKdtreeBuilder.hpp"

#include "3d/Light.hpp"

#include "..\tools\BvhInspector.h"

//#define CPU

using namespace FW;

//------------------------------------------------------------------------

Renderer::Renderer()
:   m_raygen            (1 << 20),

    m_window            (NULL),
    m_enableRandom      (Environment::GetSingleton()->GetBool("Raygen.random")),

    m_mesh              (NULL),
    m_scene             (NULL),

    m_sampleImage		(NULL),
    m_image             (NULL),
    m_cameraFar         (0.0f),

    m_newBatch          (true),
    m_batchRays         (NULL),
    m_batchStart        (0),

    m_accelStruct		(NULL),

    m_vis				(NULL),
    m_showVis			(false),
	m_vplsGenerated		(false),
	m_meshChanged		(false)
{

	string ds;
	Environment::GetSingleton()->GetStringValue("Renderer.dataStructure", ds);

    // Setup data structure.
	if(ds == "BVH")
		m_cudaTracer = new CudaBVHTracer();
	else if(ds == "KDTree")
		m_cudaTracer = new CudaKDTreeTracer();
	else
		fail("Incorrect data structure type!");

	m_cudaTracer->setScene(NULL);

    m_compiler.setSourceFile("src/rt/cuda/RendererKernels.cu");
    m_compiler.addOptions("-use_fast_math");
    m_compiler.include("src/rt");
    m_compiler.include("src/framework");
    m_cachePath = "bvhcache";

    m_platform = Platform("GPU");
    m_platform.setLeafPreferences(16, 1);

	Environment::GetSingleton()->GetIntValue("VPL.primaryLights", m_lightCount);
	Environment::GetSingleton()->GetIntValue("VPL.maxLightBounces", m_lightBounces);
	
}

//------------------------------------------------------------------------

Renderer::~Renderer(void)
{
    setMesh(NULL);

	delete m_sampleImage;
    delete m_image;
	delete m_cudaTracer;
	delete m_accelStruct;

	cudaDeviceReset();
}

//------------------------------------------------------------------------

void Renderer::setMesh(MeshBase* mesh)
{
    // Same mesh => done.

    if (mesh == m_mesh)
        return;

	m_meshChanged = true;

    // Deinit scene and BVH.

    delete m_scene;
    m_scene = NULL;
	
    invalidateBVH();

    // Create scene.

    m_mesh = mesh;
    if (mesh)
	{
        m_scene = new Scene(*mesh);
		m_cudaTracer->setScene(m_scene);
	}
}

//------------------------------------------------------------------------

void Renderer::setParams(const Params& params)
{
    m_params = params;

	m_cudaTracer->setKernel(params.kernelName);
}

//------------------------------------------------------------------------

CudaAS* Renderer::getCudaBVH(GLContext* gl, const CameraControls& camera)
{
    // BVH is already valid => done.

    BVHLayout layout = m_cudaTracer->getDesiredBVHLayout();
    if (!m_mesh || (m_accelStruct && m_accelStruct->getLayout() == layout))
        return m_accelStruct;

    // Deinit.

    delete m_accelStruct;
    m_accelStruct = NULL;

    // Setup build parameters.

    BVH::Stats stats;
    m_buildParams.stats = &stats;

	string ds;
	Environment::GetSingleton()->GetStringValue("Renderer.dataStructure", ds);

	string builder;
	Environment::GetSingleton()->GetStringValue("Renderer.builder", builder);

    // Determine cache file name.
	
    String cacheFileName = FW::sprintf("%s/%08x_%s.dat", m_cachePath.getPtr(), hashBits(
        m_scene->hash(),
        m_platform.computeHash(),
        m_buildParams.computeHash(),
        layout,
		hash<String>(String(ds.c_str()))), builder.c_str());

    // Cache file exists => import.

    if (!hasError() && Environment::GetSingleton()->GetBool("Renderer.cacheDataStructure"))
    {
        File file(cacheFileName, File::Read);
        if (!hasError())
        {
			m_accelStruct = new CudaBVH(file);
			return m_accelStruct;
        }
        clearError();
    }

    // Display status.

    printf("\nBuilding BVH...\nThis will take a while.\n");
    if (m_window)
        m_window->showModalMessage("Building BVH...");

	// If we use HLBVH, use HLBVH

	if (builder == "HLBVH")
	{
		HLBVHParams params;
		params.hlbvh = true;
		params.hlbvhBits = 4;
		params.leafSize = 8;
		params.epsilon = 0.001f;
		m_accelStruct = new HLBVHBuilder(m_scene, m_platform, params);
	}

	// For OcclusionBVH - first build SAHBVH and then build OcclusionBVH
	if (builder == "OcclusionBVH")
	{
		Environment::GetSingleton()->SetString("Renderer.builder", "SAHBVH");
		
		// Build BVH.
		BVH bvh(m_scene, m_platform, m_buildParams);
		stats.print();
		CudaBVH cudaBvh(bvh, layout);
		failIfError();

		cudaBvh.setTraceParams(&m_platform, m_scene);
		m_cudaTracer->setBVH(&cudaBvh);
		
		// Initialize the visibility buffer
		m_triangleVisibility.resize(m_scene->getNumTriangles() * sizeof(S32));
		m_triangleVisibility.clear(0);
		
		// Generate primary rays.
		const Vec2i& size = gl->getViewSize();
		U32 randomSeed = (m_enableRandom) ? m_random.getU32() : 0;
		m_raygen.primary(m_primaryRays,
			camera.getPosition(),
			invert(gl->xformFitToView(-1.0f, 2.0f) * camera.getWorldToClip()),
			size.x, size.y,
			camera.getFar(), randomSeed);

		// Secondary rays enabled => trace primary rays.
		if (m_params.rayType != RayType_Primary && m_params.rayType != RayType_Textured && m_params.rayType != RayType_PathTracing)
		{
			m_cudaTracer->traceBatch(m_primaryRays);
			getVisibleTriangles(&m_primaryRays);
		}

		m_cameraFar     = camera.getFar();
		m_newBatch      = true;
		m_batchRays		= &m_primaryRays;
		m_batchStart    = 0;
		while (nextBatch())
		{
			traceBatch();
			getVisibleTriangles(m_batchRays);
		}
		m_buildParams.visibility = &m_triangleVisibility;

		Environment::GetSingleton()->SetString("Renderer.builder", builder.c_str());
	}

    // Build BVH.
	if(m_accelStruct == NULL)
	{
		if (builder == "PersistentBVH")
		{
#ifndef SBVH
			m_accelStruct = new CudaPersistentBVHBuilder(*m_scene, FLT_EPSILON);
			((CudaPersistentBVHBuilder*)m_accelStruct)->resetBuffers(true);
			float time = ((CudaPersistentBVHBuilder*)m_accelStruct)->build();
			((CudaPersistentBVHBuilder*)m_accelStruct)->resetBuffers(false);

			printf("Build time: %f\n", time);
			BvhInspector ins((CudaBVH*)m_accelStruct);
			float sah = 0.f;
			Vec3f lo; Vec3f hi;
			m_scene->getBBox(lo, hi);
			ins.computeSubtreeProbabilities(0, AABB(lo, hi), m_platform, 1.0, sah);
			printf("SAH = %f\n", sah);
			BVH::Stats stats;
			ins.inspect(stats);
			printf("INodes: %i LNodes: %i, max depth: %i\n", stats.numInnerNodes, stats.numLeafNodes, stats.maxDepth);
			fflush(stdout);
#endif
		}
		else if(builder == "PersistentSBVH")
		{
#ifdef SBVH
			m_accelStruct = new CudaPersistentSBVHBuilder(*m_scene, FLT_EPSILON);
			((CudaPersistentSBVHBuilder*)m_accelStruct)->resetBuffers(true);
			float time = ((CudaPersistentSBVHBuilder*)m_accelStruct)->build();

			FW::U32 allocNum;
			FW::F32 allocSum;
			FW::F32 allocSquare;

			((CudaPersistentSBVHBuilder*)m_accelStruct)->getAllocStats(allocNum, allocSum, allocSquare);

			printf("Build time: %f\n", time);
			BvhInspector ins((CudaBVH*)m_accelStruct);
			float sah = 0.f;
			Vec3f lo; Vec3f hi;
			m_scene->getBBox(lo, hi);
			ins.computeSubtreeProbabilities(0, AABB(lo, hi), m_platform, 1.0, sah);
			printf("SAH = %f\n", sah);
			BVH::Stats stats;
			ins.inspect(stats);
			printf("INodes: %i LNodes: %i, max depth: %i\n", stats.numInnerNodes, stats.numLeafNodes, stats.maxDepth);
			printf("Alloc stats: num: %u size: %f (%f MB)\n", allocNum, allocSum, allocSum / (1024.f * 1024.f));
			fflush(stdout);

			((CudaPersistentSBVHBuilder*)m_accelStruct)->resetBuffers(false);
#endif
		}
		else
		{
			BVH bvh(m_scene, m_platform, m_buildParams);
			stats.print();
			m_accelStruct = new CudaBVH(bvh, layout);

			//printf("Build time: %f\n", time);
			BvhInspector ins((CudaBVH*)m_accelStruct);
			float sah = 0.f;
			Vec3f lo; Vec3f hi;
			m_scene->getBBox(lo, hi);
			ins.computeSubtreeProbabilities(0, AABB(lo, hi), m_platform, 1.0, sah);
			printf("SAH = %f\n", sah);
			BVH::Stats stats;
			ins.inspect(stats);
			printf("INodes: %i LNodes: %i, max depth: %i\n", stats.numInnerNodes, stats.numLeafNodes, stats.maxDepth);
			fflush(stdout);

			failIfError();
		}
	}

    // Write to cache.

    if (!hasError())
    {
		/*
        CreateDirectory(m_cachePath.getPtr(), NULL);
        File file(cacheFileName, File::Create);
        m_accelStruct->serialize(file);
        clearError();
		*/
    }

    // Display status.

    printf("Done.\n\n");
    return m_accelStruct;
}

//------------------------------------------------------------------------

CudaAS*	Renderer::getCudaKDTree(void)
{
	BVHLayout layout = m_cudaTracer->getDesiredBVHLayout();
	if (!m_mesh || (m_accelStruct && m_accelStruct->getLayout() == layout))
		return m_accelStruct;

	delete m_accelStruct;
	m_accelStruct = NULL;

	string ds;
	Environment::GetSingleton()->GetStringValue("Renderer.dataStructure", ds);

	string builder;
	Environment::GetSingleton()->GetStringValue("Renderer.builder", builder);

	String cacheFileName = sprintf("%s/%08x_%s.dat", m_cachePath.getPtr(), hashBits(
		m_scene->hash(),
		m_platform.computeHash(),
		m_buildParams.computeHash(),
		layout,
		hash<String>(String(ds.c_str()))), builder.c_str());

	if(!hasError() && Environment::GetSingleton()->GetBool("Renderer.cacheDataStructure"))
	{
		File file(cacheFileName, File::Read);
		if (!hasError())
		{
			m_accelStruct = new CudaKDTree(file);
			return m_accelStruct;
		}
		clearError();
	}

	printf("\nBuilding k-d tree...\nThis will take a while.\n");
    if (m_window)
        m_window->showModalMessage("Building k-d tree...");

	// Setup build parameters.

	if (builder == "PersistentKDTree")
	{
		FW::U32 allocNum = 0;
		FW::F32 allocSum = 0.f;
		FW::F32 allocSquare = 0.f;


		m_accelStruct = new CudaPersistentKDTreeBuilder(*m_scene, FLT_EPSILON);
		((CudaPersistentKDTreeBuilder*)m_accelStruct)->resetBuffers(true);
		float time = ((CudaPersistentKDTreeBuilder*)m_accelStruct)->build();
		printf("Build time: %f\n", time);
		((CudaPersistentKDTreeBuilder*)m_accelStruct)->getAllocStats(allocNum, allocSum, allocSquare);
		((CudaPersistentKDTreeBuilder*)m_accelStruct)->resetBuffers(false);
		printf("Alloc stats: num: %u size: %f (%f MB)\n", allocNum, allocSum, allocSum / (1024.f * 1024.f));
		fflush(stdout);

	}
	else
	{
		KDTree::BuildParams params;
		params.enablePrints = m_buildParams.enablePrints;
		KDTree::Stats stats;
		params.stats = &stats;

		KDTree kdtree(m_scene, m_platform, params);
		stats.print();
		m_accelStruct = new CudaKDTree(kdtree);
		failIfError();
	}

    // Write to cache.

    if (!hasError())
    {
        CreateDirectory(m_cachePath.getPtr(), NULL);
        File file(cacheFileName, File::Create);
        m_accelStruct->serialize(file);
        clearError();
    }

	// Display status.

    printf("Done.\n\n");
    return m_accelStruct;
}

//------------------------------------------------------------------------

F32 Renderer::renderFrame(GLContext* gl, const CameraControls& camera)
{
    F32 launchTime = 0.0f;
    beginFrame(gl, camera);
    while (nextBatch())
    {
        launchTime += traceBatch();
        updateResult();
    }
    displayResult(gl);
	CameraControls& c = const_cast<CameraControls&>(camera);
	if(m_showVis && m_vis != NULL) // If visualization is enabled
		m_vis->draw(gl, c);

    return launchTime;
}

//------------------------------------------------------------------------

void Renderer::beginFrame(GLContext* gl, const CameraControls& camera)
{
    FW_ASSERT(gl && m_mesh);

	string ds;
	Environment::GetSingleton()->GetStringValue("Renderer.dataStructure", ds);

    // Setup BVH.
	if(ds == "BVH")
		m_cudaTracer->setBVH(getCudaBVH(gl, camera));
	else if(ds == "KDTree")
		m_cudaTracer->setBVH(getCudaKDTree());
	else
		fail("Incorrect data structure type!");

    // Setup result image.

    const Vec2i& size = gl->getViewSize();
    if (!m_image || m_image->getSize() != size)
    {
		delete m_sampleImage;
		m_sampleImage = new Image(size, ImageFormat::RGBA_Vec4f);
		m_sampleImage->getBuffer().setHints(Buffer::Hint_CudaGL);
		m_sampleImage->clear();
        delete m_image;
        m_image = new Image(size, ImageFormat::ABGR_8888);
        m_image->getBuffer().setHints(Buffer::Hint_CudaGL);
        m_image->clear();

		m_pixels.resize(size.x * size.y * sizeof(Vec4f));
		
    }

	if(m_params.rayType == RayType_VPL) {
		m_image->clear();
		m_pixels.clear();

		// Generate VPLs if they are not or the mesh changed
		if(!m_vplsGenerated || m_meshChanged) {

			m_lights.clear();

			Random rand(156);

			m_vplsGenerated = true;
			m_raygen.primaryVPL(m_lights, m_vplBuffer, m_scene, m_lightCount, m_lightBounces, 10000.0f, rand.getU32());

			for (int i = 0; i < m_lightCount; i++) {
				Light* lightBuffer = (Light*)m_lights.getPtr();
			}

			m_cudaTracer->traceBatch(m_vplBuffer);

			for(int i = 0; i <= m_lightBounces; i++) {
				m_raygen.reflectedVPL(m_lights, m_vplBuffer, m_lightCount, i, m_scene, 10000.0f);
				if(i < m_lightBounces) {
					m_cudaTracer->traceBatch(m_vplBuffer);
				}
			}


		}
	}


    // Generate primary rays.

    U32 randomSeed = (m_enableRandom) ? m_random.getU32() : 0;
    m_raygen.primary(m_primaryRays,
        camera.getPosition(),
        invert(gl->xformFitToView(-1.0f, 2.0f) * camera.getWorldToClip()),
        size.x, size.y,
        camera.getFar(), randomSeed);

    // Secondary rays enabled => trace primary rays.

	if (m_params.rayType != RayType_Primary && m_params.rayType != RayType_Textured && m_params.rayType != RayType_PathTracing)
	{
#ifndef CPU
		m_cudaTracer->traceBatch(m_primaryRays);
#else
		m_accelStruct->trace(m_primaryRays, m_triangleVisibility);
#endif
	}

    // Initialize state.

    m_cameraFar     = camera.getFar();
    m_newBatch      = true;
    m_batchRays     = NULL;
    m_batchStart    = 0;
	m_currentLight	= 0;
}

//------------------------------------------------------------------------

bool Renderer::nextBatch(void)
{
    FW_ASSERT(m_scene);

    // Clean up the previous batch.

    if (m_batchRays)
        m_batchStart += m_batchRays->getSize();
    m_batchRays = NULL;

	Light* lights = (Light*)m_lights.getPtr();

    // Generate new batch.

    U32 randomSeed = (m_enableRandom) ? m_random.getU32() : 0;
    switch (m_params.rayType)
    {
    case RayType_Primary:
	case RayType_Textured:
	case RayType_PathTracing:
        if (!m_newBatch)
            return false;
        m_newBatch = false;
        m_batchRays = &m_primaryRays;
        break;

    case RayType_AO:
        if (!m_raygen.ao(m_secondaryRays, m_primaryRays, *m_scene, m_params.numSamples, m_params.aoRadius, m_newBatch, randomSeed))
            return false;
        m_batchRays = &m_secondaryRays;
        break;

    case RayType_Diffuse:
        if (!m_raygen.ao(m_secondaryRays, m_primaryRays, *m_scene, m_params.numSamples, m_cameraFar, m_newBatch, randomSeed))
            return false;
        m_secondaryRays.setNeedClosestHit(true);
        m_batchRays = &m_secondaryRays;
        break;

	case RayType_VPL:
		if(!m_raygen.shadow(m_shadowRays, m_primaryRays, 1, lights[m_currentLight].position , 0, m_newBatch)) {
			if(m_currentLight == (m_lightCount * (m_lightBounces + 2)) - 1){
				return false;
			} else {
				m_newBatch = true;
				m_batchStart = 0;
				m_currentLight++;
				return nextBatch();
			}
		} else {
			m_batchRays = &m_shadowRays;
		}
		break;
    default:
        FW_ASSERT(false);
        return false;
    }

    // Sort rays.

    if (m_params.sortSecondary && (m_params.rayType != RayType_Primary || m_params.rayType != RayType_Textured || m_params.rayType != RayType_PathTracing))
        m_batchRays->mortonSort();
    return true;
}

//------------------------------------------------------------------------

F32 Renderer::traceBatch(void)
{
    FW_ASSERT(m_batchRays);
#ifndef CPU
	return m_cudaTracer->traceBatch(*m_batchRays);
#else
	Timer timer(true);
	m_accelStruct->trace(*m_batchRays, m_triangleVisibility);
	((CudaBVH*)m_accelStruct)->trace(*m_batchRays, m_triangleVisibility, false);
	return timer.getElapsed();
#endif
}

//------------------------------------------------------------------------

void Renderer::updateResult(void)
{
    FW_ASSERT(m_scene && m_image && m_batchRays);
	
	m_sampleCount += 1.0f;

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Setup input struct.

	module->setTexRef("t_textures", *m_scene->getTextureAtlas()->getAtlasTexture().getImage(), true, true, true, false);

	if(m_params.rayType == RayType_VPL) {
		VPLReconstructInput& in    = *(VPLReconstructInput*)module->getGlobal("c_VPLReconstructInput").getMutablePtr();
		in.currentLight			= m_currentLight;
		in.lights				= m_lights.getCudaPtr();
		in.lightCount			= m_lightCount * (m_lightBounces + 2);
		in.firstPrimary			= m_batchStart;
		in.numPrimary           = m_primaryRays.getSize();
		in.primarySlotToID      = m_primaryRays.getSlotToIDBuffer().getCudaPtr();
		in.primaryResults       = m_primaryRays.getResultBuffer().getCudaPtr();
		in.primaryRays			= m_primaryRays.getRayBuffer().getCudaPtr();
		in.pixels				= m_pixels.getMutableCudaPtr();
		in.shadowResults		= m_shadowRays.getResultBuffer().getCudaPtr();
		in.shadowIdToSlot		= m_shadowRays.getIDToSlotBuffer().getCudaPtr();

		in.texCoords			= m_scene->getVtxTexCoordBuffer().getCudaPtr();
		in.normals				= m_scene->getVtxNormalBuffer().getCudaPtr();
		in.triVertIndex			= m_scene->getTriVtxIndexBuffer().getCudaPtr();
		in.vertices				= m_scene->getVtxPosBuffer().getCudaPtr();
		in.atlasInfo			= m_scene->getTextureAtlasInfo().getCudaPtr();
		in.matId				= m_scene->getMaterialIds().getCudaPtr();
		in.matInfo				= m_scene->getMaterialInfo().getCudaPtr();
		in.triMaterialColor		= m_scene->getTriMaterialColorBuffer().getCudaPtr();

		module->getKernel("vplReconstructKernel").launch(m_shadowRays.getSize());
		if(m_currentLight == (m_lightCount * (m_lightBounces + 2)) - 1) {
			module->getKernel("vplNormalizeKernel").setParams(CudaKernel::Param(m_pixels.getCudaPtr()), CudaKernel::Param(m_image->getBuffer().getMutableCudaPtr()), CudaKernel::Param(m_lightCount * (m_lightBounces + 2))).launch(in.numPrimary);
		}

	} else {

		ReconstructInput& in    = *(ReconstructInput*)module->getGlobal("c_ReconstructInput").getMutablePtr();
		in.numRaysPerPrimary    = (m_params.rayType == RayType_Primary || m_params.rayType == RayType_Textured || m_params.rayType == RayType_PathTracing) ? 1 : m_params.numSamples;
		in.firstPrimary         = m_batchStart / in.numRaysPerPrimary;
		in.numPrimary           = m_batchRays->getSize() / in.numRaysPerPrimary;
		in.isPrimary            = (m_params.rayType == RayType_Primary);
		in.isAO                 = (m_params.rayType == RayType_AO);
		in.isDiffuse            = (m_params.rayType == RayType_Diffuse);
		in.isTextured           = (m_params.rayType == RayType_Textured);
		in.isPathTraced         = (m_params.rayType == RayType_PathTracing);
		in.primarySlotToID      = m_primaryRays.getSlotToIDBuffer().getCudaPtr();
		in.primaryResults       = m_primaryRays.getResultBuffer().getCudaPtr();
		in.batchIDToSlot        = m_batchRays->getIDToSlotBuffer().getCudaPtr();
		in.batchResults         = m_batchRays->getResultBuffer().getCudaPtr();
		in.triMaterialColor     = m_scene->getTriMaterialColorBuffer().getCudaPtr();
		in.triShadedColor       = m_scene->getTriShadedColorBuffer().getCudaPtr();
		in.pixels               = m_image->getBuffer().getMutableCudaPtr();

		in.texCoords			= m_scene->getVtxTexCoordBuffer().getCudaPtr();
		in.normals				= m_scene->getVtxNormalBuffer().getCudaPtr();
		in.triVertIndex			= m_scene->getTriVtxIndexBuffer().getCudaPtr();
		in.atlasInfo			= m_scene->getTextureAtlasInfo().getCudaPtr();
		in.matId				= m_scene->getMaterialIds().getCudaPtr();
		in.matInfo				= m_scene->getMaterialInfo().getCudaPtr();
		in.outputColor			= m_sampleImage->getBuffer().getMutableCudaPtr();
		in.samplesCount			= m_sampleCount;

		

		// Launch.

		module->getKernel("reconstructKernel").launch(in.numPrimary);
	}
}

//------------------------------------------------------------------------

void Renderer::displayResult(GLContext* gl)
{
    FW_ASSERT(gl);
    Mat4f oldXform = gl->setVGXform(Mat4f());
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_DEPTH_TEST);
    gl->drawImage(*m_image, Vec2f(0.0f), 0.5f, false);
    gl->setVGXform(oldXform);
    glPopAttrib();
}

//------------------------------------------------------------------------

int Renderer::getTotalNumRays(void)
{
    // Casting primary rays => no degenerates.

    if (m_params.rayType == RayType_Primary || m_params.rayType == RayType_Textured || m_params.rayType == RayType_PathTracing) {
        return m_primaryRays.getSize();
	} else if(m_params.rayType == RayType_VPL) {
		return m_primaryRays.getSize() * (m_lights.getSize() / sizeof(Light)) + m_primaryRays.getSize();
	}


    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Set input and output.

    CountHitsInput& in = *(CountHitsInput*)module->getGlobal("c_CountHitsInput").getMutablePtr();
    in.numRays = m_primaryRays.getSize();
    in.rayResults = m_primaryRays.getResultBuffer().getCudaPtr();
    in.raysPerThread = 32;
    module->getGlobal("g_CountHitsOutput").clear();

    // Count primary ray hits.

    module->getKernel("countHitsKernel").launch(
        (in.numRays - 1) / in.raysPerThread + 1,
        Vec2i(CountHits_BlockWidth, CountHits_BlockHeight));

    int numHits = *(S32*)module->getGlobal("g_CountHitsOutput").getPtr();

    // numSecondary = secondaryPerPrimary * primaryHits

    return numHits * m_params.numSamples;
}

//------------------------------------------------------------------------

F32 Renderer::calcNodeSAHCostKdtree(const Platform& platform, Buffer* nodes, Buffer* tris, S32 n, AABB bbox, S32 depth, S32& maxDepth, S32& sumDepth, S32& numNodes, S32& numLeaves, F32& nodeArea, F32 &weightedLeafArea, F32& test)
{
	numNodes++;
	if(depth > maxDepth)
		maxDepth = depth;
	sumDepth += depth;

	F32 pa = bbox.area();

	if((n & KDTREE_MASK) == KDTREE_EMPTYLEAF) // Empty leaf
	{
		return calcLeafSAHCostNum(platform, 0, numLeaves)*pa;
	}

	if(n >= 0) // Inner node
	{
		Vec4i cell = ((Vec4i*)nodes)[n];
		unsigned int type = cell.w & KDTREE_MASK;
		float split = *(float*)&cell.z;

		AABB bboxLeft = bbox;
		AABB bboxRight = bbox;
		
		switch((type >> KDTREE_DIMPOS))
		{
		case 0:
			bboxLeft.max().x = split;
			bboxRight.min().x = split;
			break;

		case 1:
			bboxLeft.max().y = split;
			bboxRight.min().y = split;
			break;

		case 2:
			bboxLeft.max().z = split;
			bboxRight.min().z = split;
			break;
		}

		S32 nl = cell.x;
		S32 nr = cell.y;

		nodeArea += pa;		

		weightedLeafArea += calcNodeSAHCostKdtree(platform, nodes, tris, nl, bboxLeft, depth+1, maxDepth, sumDepth, numNodes, numLeaves, nodeArea, weightedLeafArea, test);
		weightedLeafArea += calcNodeSAHCostKdtree(platform, nodes, tris, nr, bboxRight, depth+1, maxDepth, sumDepth, numNodes, numLeaves, nodeArea, weightedLeafArea, test);

		if(depth == 0)
		{
			F32 pa = bbox.area();
			return platform.getNodeCost(1) * nodeArea/pa + weightedLeafArea/pa;
		}

		return 0.f;
	}
	else // Leaf
	{
		return calcLeafSAHCostCompact(platform, tris, ~n, numLeaves)*pa;
	}
}

//------------------------------------------------------------------------

F32 Renderer::calcLeafSAHCostCompact(const Platform& platform, Buffer* triIdx, S32 n, S32& numLeaves)
{
	numLeaves++;

	int* idx = (int*)triIdx + n;

	S32 cnt = 0;
	while(idx[cnt] != 0x80000000)
		cnt++;

	return platform.getTriangleCost(cnt);
}

//------------------------------------------------------------------------

F32 Renderer::calcLeafSAHCostNum(const Platform& platform, S32 n, S32& numLeaves)
{
	numLeaves++;
	return platform.getTriangleCost(0);
}

//------------------------------------------------------------------------

void Renderer::startBVHVis(void)
{
	if(m_window == NULL)
		return;

	if(m_vis != NULL)
		endBVHVis();

	// Trace the primary rays so that we have correct rays to visualize
	// TODO: Enable secondary rays visualization too?
	RayStats stats;
	//m_as->setTraceParams(&m_platform, m_scene);
	//setTracerBVH(m_bvh);
	//traceBatch(m_primaryRays, &stats);
	//m_newBatch = true;
	//m_batchRays = NULL;
	//m_batchStart = 0;
	//m_sampleIndex = 0;
	//m_secondaryIndex = 0;

	//m_buildParams.visibility = &getVisibleTriangles(m_scene->getNumTriangles(), true); // Get visibility buffer
	//if(m_params.rayType != RayType_Primary)
	//{
	//	traceBatch();
	//}
	//while (nextBatch())
	//{
	//	traceBatch();
	//}

	string ds;
	Environment::GetSingleton()->GetStringValue("Renderer.dataStructure", ds);

	string builder;
	Environment::GetSingleton()->GetStringValue("Renderer.builder", builder);

    // Setup visualization.
	Buffer* visib = NULL;
	if (builder == "OcclusionBVH")
		visib = &m_triangleVisibility;

	if(ds == "BVH")
		m_vis = new VisualizationBVH((CudaBVH*)getCudaBVH(), m_scene, &m_primaryRays, visib);
	else if(ds == "KDTree")
		m_vis = new VisualizationKDTree((CudaKDTree*)getCudaKDTree(), m_scene, &m_primaryRays, visib);

	m_vis->setVisible(true);
	m_window->addListener(m_vis);

	m_showVis = true;
}

//------------------------------------------------------------------------

void Renderer::endBVHVis(void)
{
	if(m_window != NULL)
		m_window->removeListener(m_vis);
	delete m_vis;
	m_vis = NULL;

	m_showVis = false;
}

//------------------------------------------------------------------------

void Renderer::getVisibleTriangles(RayBuffer* rayBuffer)
{
	// Compile kernel.
    CudaModule* module = m_compiler.compile();

	CudaKernel kernel = module->getKernel("getVisibility");

	// Set parameters.

	kernel.setParams(
		rayBuffer->getResultBuffer().getCudaPtr(),   // results
		rayBuffer->getSize(),                      // number of rays
		m_triangleVisibility.getMutableCudaPtr()); // visibility

    // Get information about triangles hit by rays.

    kernel.launch(rayBuffer->getSize(), Vec2i(CountHits_BlockWidth, CountHits_BlockHeight));
}

//------------------------------------------------------------------------
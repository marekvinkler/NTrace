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

#include "cuda/CudaBVHTracer.hpp"
#include "gui/Window.hpp"
#include "io/File.hpp"

using namespace FW;

//------------------------------------------------------------------------

CudaBVHTracer::CudaBVHTracer()
:   m_bvh(NULL)
{
    CudaModule::staticInit();
    m_compiler.addOptions("-use_fast_math");
	m_otrace = false;
}

//------------------------------------------------------------------------

CudaBVHTracer::~CudaBVHTracer(void)
{
}

//------------------------------------------------------------------------

void CudaBVHTracer::setKernel(const String& kernelName)
{
    // Not changed => done.

    if (m_kernelName == kernelName)
        return;
    m_kernelName = kernelName;

    // Compile kernel.

    CudaModule* module = compileKernel();

    // Initialize config with default values.
    {
        KernelConfig& c         = *(KernelConfig*)module->getGlobal("g_config").getMutablePtr();
        c.bvhLayout             = BVHLayout_Max;
        c.blockWidth            = 0;
        c.blockHeight           = 0;
        c.usePersistentThreads  = 0;
    }

    // Query config.

    module->getKernel("queryConfig").launch(1, 1);
    m_kernelConfig = *(const KernelConfig*)module->getGlobal("g_config").getPtr();

	// Detect whether this is one of the otrace kernels (otrace kernels compute the final shading
	// color inside the kernel
	if(strstr(kernelName.getPtr(), "otracer"))
	{
		m_otrace = true;
	}
}

//------------------------------------------------------------------------

F32 CudaBVHTracer::traceBatch(RayBuffer& rays)
{
    // No rays => done.

    int numRays = rays.getSize();
    if (!numRays)
        return 0.0f;

    // Check BVH consistency.

    if (!m_bvh)
        fail("CudaBVHTracer: No BVH!");
    if (m_bvh->getLayout() != getDesiredBVHLayout())
        fail("CudaBVHTracer: Incorrect BVH layout!");


	if(!m_otrace)
	{
		// Get BVH buffers.

		CUdeviceptr nodePtr     = m_bvh->getNodeBuffer().getCudaPtr();
		CUdeviceptr triPtr      = m_bvh->getTriWoopBuffer().getCudaPtr();
		Buffer&     indexBuf    = m_bvh->getTriIndexBuffer();
		Vec2i       nodeOfsA    = m_bvh->getNodeSubArray(0);
		Vec2i       nodeOfsB    = m_bvh->getNodeSubArray(1);
		Vec2i       nodeOfsC    = m_bvh->getNodeSubArray(2);
		Vec2i       nodeOfsD    = m_bvh->getNodeSubArray(3);
		Vec2i       triOfsA     = m_bvh->getTriWoopSubArray(0);
		Vec2i       triOfsB     = m_bvh->getTriWoopSubArray(1);
		Vec2i       triOfsC     = m_bvh->getTriWoopSubArray(2);

		// Compile kernel.

		CudaModule* module = compileKernel();
		CudaKernel kernel = module->getKernel("trace_bvh");

		// Set parameters.

		kernel.setParams(
			numRays,                                    // numRays
			(rays.getNeedClosestHit()) ? 0 : 1,         // anyHit
			rays.getRayBuffer().getCudaPtr(),           // rays
			rays.getResultBuffer().getMutableCudaPtr(), // results
			m_triangleVisibility.getMutableCudaPtr(),	// visibility
			nodePtr + nodeOfsA.x,                       // nodesA
			nodePtr + nodeOfsB.x,                       // nodesB
			nodePtr + nodeOfsC.x,                       // nodesC
			nodePtr + nodeOfsD.x,                       // nodesD
			triPtr + triOfsA.x,                         // trisA
			triPtr + triOfsB.x,                         // trisB
			triPtr + triOfsC.x,                         // trisC
			indexBuf.getCudaPtr());                     // triIndices

		// Set texture references.

		module->setTexRef("t_rays", rays.getRayBuffer(), CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_nodesA", nodePtr + nodeOfsA.x, nodeOfsA.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_nodesB", nodePtr + nodeOfsB.x, nodeOfsB.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_nodesC", nodePtr + nodeOfsC.x, nodeOfsC.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_nodesD", nodePtr + nodeOfsD.x, nodeOfsD.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_trisA", triPtr + triOfsA.x, triOfsA.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_trisB", triPtr + triOfsB.x, triOfsB.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_trisC", triPtr + triOfsC.x, triOfsC.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_triIndices", indexBuf, CU_AD_FORMAT_SIGNED_INT32, 1);

		// Determine block and grid sizes.

		int desiredWarps = (numRays + 31) / 32;
		if (m_kernelConfig.usePersistentThreads != 0)
		{
			*(S32*)module->getGlobal("g_warpCounter").getMutablePtr() = 0;
			desiredWarps = 720; // Tesla: 30 SMs * 24 warps, Fermi: 15 SMs * 48 warps
		}

		Vec2i blockSize(m_kernelConfig.blockWidth, m_kernelConfig.blockHeight);
		int blockWarps = (blockSize.x * blockSize.y + 31) / 32;
		int numBlocks = (desiredWarps + blockWarps - 1) / blockWarps;

		// Launch.

		return kernel.launchTimed(numBlocks * blockSize.x * blockSize.y, blockSize);
	}
	else
	{
		// Get BVH buffers.
		CUdeviceptr nodePtr     = m_bvh->getNodeBuffer().getCudaPtr();
		CUdeviceptr triPtr      = m_bvh->getTriWoopBuffer().getCudaPtr();
		Buffer&     indexBuf    = m_bvh->getTriIndexBuffer();
		Vec2i       nodeOfsA    = m_bvh->getNodeSubArray(0);
		Vec2i       nodeOfsB    = m_bvh->getNodeSubArray(1);
		Vec2i       nodeOfsC    = m_bvh->getNodeSubArray(2);
		Vec2i       nodeOfsD    = m_bvh->getNodeSubArray(3);
		Vec2i       triOfsA     = m_bvh->getTriWoopSubArray(0);
		Vec2i       triOfsB     = m_bvh->getTriWoopSubArray(1);
		Vec2i       triOfsC     = m_bvh->getTriWoopSubArray(2);

		// Compile kernel.
		CudaModule* module = compileKernel();
		CudaKernel kernel = module->getKernel("otrace_kernel");

		// Setup input struct.
		OtraceInput& in			= *(OtraceInput*)module->getGlobal("c_OtraceInput").getMutablePtr();
		in.numRays				= numRays;
		in.anyHit				= (rays.getNeedClosestHit()) ? 0 : 1;
		in.rays					= rays.getRayBuffer().getMutableCudaPtr();
		in.results				= rays.getResultBuffer().getMutableCudaPtr();
		in.nodesA				= nodePtr + nodeOfsA.x;
		in.nodesB				= nodePtr + nodeOfsB.x;
		in.nodesC				= nodePtr + nodeOfsC.x;
		in.nodesD				= nodePtr + nodeOfsD.x;
		in.trisA				= triPtr + triOfsA.x;
		in.trisB				= triPtr + triOfsB.x;
		in.trisC				= triPtr + triOfsC.x;
		in.triIndices			= indexBuf.getCudaPtr();
		if(m_scene)
		{
			in.texCoords			= m_scene->getVtxTexCoordBuffer().getCudaPtr();
			in.normals				= m_scene->getVtxNormalBuffer().getCudaPtr();
			in.triVertIndex			= m_scene->getTriVtxIndexBuffer().getCudaPtr();
			in.atlasInfo			= m_scene->getTextureAtlasInfo().getCudaPtr();
			in.matId				= m_scene->getMaterialIds().getCudaPtr();
			in.matInfo				= m_scene->getMaterialInfo().getCudaPtr();
			in.matColor				= m_scene->getTriMaterialColorBuffer().getCudaPtr();
			in.emissiveNum			= m_scene->getNumEmissive();
			in.emissive				= m_scene->getEmissiveTris().getCudaPtr();
			in.trisCount			= m_scene->getNumTriangles();
			in.tris					= m_scene->getTriVtxIndexBuffer().getCudaPtr();
			in.vertsCount			= m_scene->getNumVertices();
			in.verts				= m_scene->getVtxPosBuffer().getCudaPtr();
			in.randomSeed			= rand();
		}

		// Set texture references.

		module->setTexRef("t_textureAtlas", *m_scene->getTextureAtlas()->getAtlasTexture().getImage(), true, true, true, false);
		module->setTexRef("t_rays", rays.getRayBuffer(), CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_nodesA", nodePtr + nodeOfsA.x, nodeOfsA.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_nodesB", nodePtr + nodeOfsB.x, nodeOfsB.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_nodesC", nodePtr + nodeOfsC.x, nodeOfsC.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_nodesD", nodePtr + nodeOfsD.x, nodeOfsD.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_trisA", triPtr + triOfsA.x, triOfsA.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_trisB", triPtr + triOfsB.x, triOfsB.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_trisC", triPtr + triOfsC.x, triOfsC.y, CU_AD_FORMAT_FLOAT, 4);
		module->setTexRef("t_triIndices", indexBuf, CU_AD_FORMAT_SIGNED_INT32, 1);

		// Determine block and grid sizes.

		int desiredWarps = (numRays + 31) / 32;
		if (m_kernelConfig.usePersistentThreads != 0)
		{
			*(S32*)module->getGlobal("g_warpCounter").getMutablePtr() = 0;
			desiredWarps = 720; // Tesla: 30 SMs * 24 warps, Fermi: 15 SMs * 48 warps
		}

		Vec2i blockSize(m_kernelConfig.blockWidth, m_kernelConfig.blockHeight);
		int blockWarps = (blockSize.x * blockSize.y + 31) / 32;
		int numBlocks = (desiredWarps + blockWarps - 1) / blockWarps;

		// Launch.
		return kernel.launchTimed(numBlocks * blockSize.x * blockSize.y, blockSize);
	}
}

//------------------------------------------------------------------------

CudaModule* CudaBVHTracer::compileKernel(void)
{
    m_compiler.setSourceFile(sprintf("src/rt/kernels/%s.cu", m_kernelName.getPtr()));
    m_compiler.clearDefines();
    CudaModule* module = m_compiler.compile();
    return module;
}

//------------------------------------------------------------------------

void CudaBVHTracer::setVisibility()
{
	if(m_kernelName.getLength() == 0)
		return;

	// Compile kernel.
    CudaModule* module = compileKernel();

	//CUdeviceptr visib = m_triangleVisibility.getMutableCudaPtr();
	//*(CUdeviceptr*)module->getGlobal("g_Visib").getMutablePtr() = visib;      // triVisibility
}

//------------------------------------------------------------------------
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

#include "CudaKDTreeTracer.hpp"
#include "gui/Window.hpp"
#include "io/File.hpp"

using namespace FW;

//------------------------------------------------------------------------

CudaKDTreeTracer::CudaKDTreeTracer(void)
:	m_kdtree(NULL)
{
	CudaModule::staticInit();
	m_compiler.addOptions("-use_fast_math");
}

//------------------------------------------------------------------------

CudaKDTreeTracer::~CudaKDTreeTracer(void)
{
}

//------------------------------------------------------------------------

void CudaKDTreeTracer::setKernel(const String& kernelName)
{
    //if (m_kernelName == kernelName)
    //    return;
    //m_kernelName = kernelName;

	m_kernelName = String("fermi_kdtree_while_while_leafRef");

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
}

//------------------------------------------------------------------------

F32 CudaKDTreeTracer::traceBatch(RayBuffer& rays)
{
	// No rays => done.

    int numRays = rays.getSize();
    if (!numRays)
        return 0.0f;

	CUdeviceptr nodePtr     = m_kdtree->getNodeBuffer().getCudaPtr();
	Vec2i       nodeOfsA    = Vec2i(0, (S32)m_kdtree->getNodeBuffer().getSize());

	CUdeviceptr triPtr      = m_kdtree->getTriWoopBuffer().getCudaPtr();
	Vec2i       triOfsA     = Vec2i(0, (S32)m_kdtree->getTriWoopBuffer().getSize());
	Buffer&     indexBuf    = m_kdtree->getTriIndexBuffer();
	
	CudaModule* module = compileKernel();
	CudaKernel kernel = module->getKernel("trace_kdtree");

	CudaKernel::Param bmin(m_bbox.min().getPtr(), 3);
	CudaKernel::Param bmax(m_bbox.max().getPtr(), 3);

	// Set parameters.
	
	kernel.setParams(
	rays.getSize(),
	rays.getNeedClosestHit() == false,
	bmin,
	bmax,
	(m_bbox.max() + m_bbox.min()).length() * 0.000001f,
	rays.getRayBuffer().getCudaPtr(),           // rays
    rays.getResultBuffer().getMutableCudaPtr(), // results
	nodePtr + nodeOfsA.x,
	nodePtr + nodeOfsA.x,
	nodePtr + nodeOfsA.x,
	nodePtr + nodeOfsA.x,
	triPtr + triOfsA.x,
	triPtr + triOfsA.x,
	triPtr + triOfsA.x,
	indexBuf.getCudaPtr()
	);

	// Set texture references.
	module->setTexRef("t_rays", rays.getRayBuffer(), CU_AD_FORMAT_FLOAT, 4);
	//m_module->setTexRef("t_nodesI", nodePtr + nodeOfsA.x, nodeOfsA.y, CU_AD_FORMAT_FLOAT, 4);
	//m_module->setTexRef("t_trisA", triPtr + triOfsA.x, triOfsA.y, CU_AD_FORMAT_FLOAT, 4);
	//m_module->setTexRef("t_triIndices", indexBuf, CU_AD_FORMAT_SIGNED_INT32, 1);

	// Determine block and grid sizes.
	int desiredWarps = (rays.getSize() + 31) / 32;
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

//------------------------------------------------------------------------

CudaModule* CudaKDTreeTracer::compileKernel(void)
{
    m_compiler.setSourceFile(sprintf("src/rt/kernels/%s.cu", m_kernelName.getPtr()));
    m_compiler.clearDefines();
    CudaModule* module = m_compiler.compile();
    return module;
}

//------------------------------------------------------------------------

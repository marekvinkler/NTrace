
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

	//KernelInput& in = *((KernelInput*)module->getGlobal("c_in").getMutablePtr());
	// Start the timer
	m_timer.unstart();
	m_timer.start();

	CUdeviceptr nodePtr     = m_kdtree->getNodeBuffer().getCudaPtr();
	Vec2i       nodeOfsA    = Vec2i(0, (S32)m_kdtree->getNodeBuffer().getSize());

	CUdeviceptr triPtr      = m_kdtree->getTriWoopBuffer().getCudaPtr();
	Vec2i       triOfsA     = Vec2i(0, (S32)m_kdtree->getTriWoopBuffer().getSize());
	Buffer&     indexBuf    = m_kdtree->getTriIndexBuffer();
	
	CudaModule* module = compileKernel();
	CudaKernel kernel = module->getKernel("trace_kdtree");

	CudaKernel::Param bmin(m_bbox.min().getPtr(), 3);
	CudaKernel::Param bmax(m_bbox.max().getPtr(), 3);

	// Set input.
	// The new (this?) version has it via parameters, not const memory
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
	//return  module->launchKernelTimed(kernel, blockSize, gridSize);
}

//------------------------------------------------------------------------

CudaModule* CudaKDTreeTracer::compileKernel(void)
{
    m_compiler.setSourceFile(FW::sprintf("src/rt/kernels/%s.cu", m_kernelName.getPtr()));
    m_compiler.clearDefines();
    CudaModule* module = m_compiler.compile();
    return module;
}

//------------------------------------------------------------------------
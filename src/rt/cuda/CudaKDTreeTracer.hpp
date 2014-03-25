#pragma once
#include "gpu/CudaCompiler.hpp"
#include "cuda/CudaKDTree.hpp"
#include "ray/RayBuffer.hpp"
#include "base/Timer.hpp"
#include "cuda/CudaVirtualTracer.hpp"

namespace FW
{
//------------------------------------------------------------------------

class CudaKDTreeTracer : public CudaVirtualTracer
{
public:
                        CudaKDTreeTracer        (void);
						~CudaKDTreeTracer       (void);

    void                setMessageWindow        (Window* window)			{ m_compiler.setMessageWindow(window); }
    void                setKernel               (const String& kernelName);
    BVHLayout           getDesiredBVHLayout     (void) const				{ return (BVHLayout)m_kernelConfig.bvhLayout; }
	void                setBVH					(CudaAS* kdtree)       { m_kdtree = (CudaKDTree*)kdtree; m_bbox = m_kdtree->getBBox(); }

    F32                 traceBatch              (RayBuffer& rays); // returns launch time in seconds

private:
    CudaModule*         compileKernel           (void);

private:
                        CudaKDTreeTracer        (const CudaKDTreeTracer&); // forbidden
    CudaKDTreeTracer&   operator=               (const CudaKDTreeTracer&); // forbidden

private:
    CudaCompiler        m_compiler;
    String              m_kernelName;
    KernelConfig        m_kernelConfig;
    CudaKDTree*         m_kdtree;
	Timer				m_timer;
	AABB				m_bbox;
};

//------------------------------------------------------------------------
}

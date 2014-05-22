
#pragma once
#include "gpu/CudaCompiler.hpp"
#include "cuda/CudaBVH.hpp"
#include "ray/RayBuffer.hpp"

namespace FW
{
//------------------------------------------------------------------------

class CudaVirtualTracer
{
public:
	virtual             ~CudaVirtualTracer     (void) {}

    virtual void        setMessageWindow        (Window* window) = 0;
    virtual void        setKernel               (const String& kernelName) = 0;
    virtual	BVHLayout	getDesiredBVHLayout     (void) const = 0;
    virtual void		setBVH                  (CudaAS* as) = 0;
	void				setScene				(Scene* scene) { m_scene = scene; }

    virtual	F32			traceBatch              (RayBuffer& rays) = 0; // returns launch time in seconds

	Scene* m_scene;
};

}

#pragma once
#include "Renderer.hpp"

namespace FW
{
class CudaRenderer : public Renderer
{
public:
	                    CudaRenderer        ();
	virtual             ~CudaRenderer       (void);

	
    virtual void        setParams           (const Params& params);
    virtual void        setMessageWindow    (Window* window)        { m_window = window; m_compiler.setMessageWindow(window); m_tracer.setMessageWindow(window); }
	virtual BVHLayout	getLayout           (void)                  { return m_tracer.getDesiredBVHLayout(); }
	
    virtual bool        nextBatch           (void);
    virtual F32         traceBatch          (RayStats* stats = NULL); // returns launch time
    virtual void        updateResult        (void); // for current batch

    virtual int         getTotalNumRays     (void); // for selected ray type, excluding degenerates
	
private:
                        CudaRenderer        (const CudaRenderer&); // forbidden
    CudaRenderer&       operator=           (const CudaRenderer&); // forbidden
	
private:
    CudaCompiler        m_compiler;
    CudaBVHTracer       m_tracer;
    CudaBVH*            m_bvh;
};
}
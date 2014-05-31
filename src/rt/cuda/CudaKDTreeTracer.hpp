#pragma once
#include "gpu/CudaCompiler.hpp"
#include "cuda/CudaKDTree.hpp"
#include "ray/RayBuffer.hpp"
#include "base/Timer.hpp"
#include "cuda/CudaVirtualTracer.hpp"

/**
 * \file
 * \brief Definitions for the Cuda KDTree Tracer.
 */

namespace FW
{
//------------------------------------------------------------------------

/**
 * \brief Cuda tracer for the k-d tree acceleration structure. Performs kd-tree traversal on the GPU.
 */
class CudaKDTreeTracer : public CudaVirtualTracer
{
public:
	/**
	* \brief Constructor.
	*/
                        CudaKDTreeTracer        (void);

	/**
	* \brief Destructor.
	*/
						~CudaKDTreeTracer       (void);

	/**
	 * \brief Sets message window for the CUDA compiler.
	 * \details Used to print info about kernel compilation to main window (not the console).
	 * \param[in] window Desired window.
	 */
    void                setMessageWindow        (Window* window)			{ m_compiler.setMessageWindow(window); }

	/**
	 * \brief Sets kernel that should perform the actual traversation of the k-d tree on the gpu.
	 * \param[in] kernelName Name of the kernel.
	 */
    void                setKernel               (const String& kernelName);
    BVHLayout           getDesiredBVHLayout     (void) const				{ return (BVHLayout)m_kernelConfig.bvhLayout; }

	/**
	 * \brief Sets k-d tree acceleration structure that will be traversed.
	 * \param kdtree K-d tree to traverse.
	 */
	void                setBVH					(CudaAS* kdtree)       { m_kdtree = (CudaKDTree*)kdtree; m_bbox = m_kdtree->getBBox(); }

	/**
	 * \brief Traces given batch of rays.
	 * \param[in,out] rays Rays to be cast to the k-d tree.
	 * \return Launch time in seconds.
	 */
    F32                 traceBatch              (RayBuffer& rays); // returns launch time in seconds

private:
	/**
	 * \brief Compiles CUDA kernel.
	 * \return Compiled kernel.
	 */
    CudaModule*         compileKernel           (void);

private:
                        CudaKDTreeTracer        (const CudaKDTreeTracer&); // forbidden
    CudaKDTreeTracer&   operator=               (const CudaKDTreeTracer&); // forbidden

private:
    CudaCompiler        m_compiler;				//!< CUDA compiler.
    String              m_kernelName;			//!< Name of the traversal kernel.
    KernelConfig        m_kernelConfig;			//!< Configuration of the kernel.
    CudaKDTree*         m_kdtree;				//!< K-d tree being traversed.
	Timer				m_timer;				//!< Timer.
	AABB				m_bbox;					//!< Bounding box of the whole scene.
};

//------------------------------------------------------------------------
}

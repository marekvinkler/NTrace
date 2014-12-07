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

#pragma once
#include "gpu/CudaCompiler.hpp"
#include "cuda/CudaBVH.hpp"
#include "ray/RayBuffer.hpp"
#include "cuda/CudaVirtualTracer.hpp"

/**
 * \file
 * \brief Definitions for the BVH CUDA Tracer.
 */

namespace FW
{
//------------------------------------------------------------------------

/**
 * \brief CUDA tracer for the BVH acceleration structure. Performs BVH traversal on the GPU.
 */
class CudaBVHTracer : public CudaVirtualTracer
{
public:
	/**
	 * \brief Constructor.
	 */
                        CudaBVHTracer           (void);

	/**
	 * \brief Destructor.
	 */
                        ~CudaBVHTracer          (void);

	/**
	 * \brief Sets message window for the CUDA compiler.
	 * \details Used to print info about kernel compilation to main window (not the console).
	 * \param[in] window Desired window.
	 */
    void                setMessageWindow        (Window* window)        { m_compiler.setMessageWindow(window); }

	/**
	 * \brief Sets kernel that should perform the actual traversal of the k-d tree on the gpu.
	 * \param[in] kernelName Name of the kernel.
	 */
    void                setKernel               (const String& kernelName);

	/**
	 * \return Kernel's desired layout of the BVH buffers.
	 */
    BVHLayout           getDesiredBVHLayout     (void) const            { return (BVHLayout)m_kernelConfig.bvhLayout; }

	/**
	 * \brief Sets BVH acceleration structure that will be traversed.
	 * \param[in] bvh BVH to traverse.
	 */
    void                setBVH                  (CudaAS* bvh)          { m_bvh = (CudaBVH*)bvh; }

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
                        CudaBVHTracer           (const CudaBVHTracer&); // forbidden
    CudaBVHTracer&         operator=            (const CudaBVHTracer&); // forbidden

private:
    CudaCompiler        m_compiler;				//!< CUDA compiler.
    String              m_kernelName;			//!< Name of the traversal kernel.
    KernelConfig        m_kernelConfig;			//!< Configuration of the kernel.
    CudaBVH*            m_bvh;					//!< BVH being traversed.
	bool				m_otrace;				//!< Flag whether to use otracer kernels.
};

//------------------------------------------------------------------------
}

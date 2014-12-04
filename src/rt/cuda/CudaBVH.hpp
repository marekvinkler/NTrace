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

/**
 * \file
 * \brief Declarations for the Cuda version of the BVH.
 */

#pragma once
#include "gpu/Buffer.hpp"
#include "io/Stream.hpp"
#include "bvh/BVH.hpp"
#include "CudaAS.hpp"
#include "kernels/CudaTracerKernels.hpp"

namespace FW
{
//------------------------------------------------------------------------
// Nodes / BVHLayout_Compact
//      nodes[innerOfs + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
//      nodes[innerOfs + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//      nodes[innerOfs + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[innerOfs + 48] = Vec4i(c0.innerOfs or ~c0.triOfs, c1.innerOfs or ~c1.triOfs, bitFlag, 0)
//
// TriWoop / BVHLayout_Compact
//      triWoop[triOfs*16 + 0 ] = Vec4f(woopZ)
//      triWoop[triOfs*16 + 16] = Vec4f(woopU)
//      triWoop[triOfs*16 + 32] = Vec4f(woopV)
//      triWoop[endOfs*16 + 0 ] = Vec4f(-0.0f, -0.0f, -0.0f, -0.0f)
//
// TriIndex / BVHLayout_Compact
//      triIndex[triOfs*4] = origIdx
//
//------------------------------------------------------------------------
//
// Nodes / BVHLayout_AOS_AOS, BVHLayout_AOS_SOA
//      nodes[node*64  + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
//      nodes[node*64  + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//      nodes[node*64  + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[inner*64 + 48] = Vec4f(c0.inner or ~c0.leaf, c1.inner or ~c1.leaf, 0, 0)
//      nodes[leaf*64  + 48] = Vec4i(triStart, triEnd, bitFlag, 0)
//
// Nodes / BVHLayout_SOA_AOS, BVHLayout_SOA_SOA
//      nodes[node*16  + size*0/4] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
//      nodes[node*16  + size*1/4] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//      nodes[node*16  + size*2/4] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[inner*16 + size*3/4] = Vec4f(c0.inner or ~c0.leaf, c1.inner or ~c1.leaf, 0, 0)
//      nodes[leaf*16  + size*3/4] = Vec4i(triStart, triEnd, bitFlag, 0)
//
// TriWoop / BVHLayout_AOS_AOS, BVHLayout_SOA_AOS
//      triWoop[tri*64 + 0 ] = Vec4f(woopZ)
//      triWoop[tri*64 + 16] = Vec4f(woopU)
//      triWoop[tri*64 + 32] = Vec4f(woopV)
//
// TriWoop / BVHLayout_AOS_SOA, BVHLayout_SOA_SOA
//      triWoop[tri*16 + size*0/4] = Vec4f(woopZ)
//      triWoop[tri*16 + size*1/4] = Vec4f(woopU)
//      triWoop[tri*16 + size*2/4] = Vec4f(woopV)
//
// TriIndex / BVHLayout_AOS_AOS, BVHLayout_AOS_SOA, BVHLayout_SOA_AOS, BVHLayout_SOA_SOA
//      triIndex[tri*4] = origIdx
//------------------------------------------------------------------------

//#define VISIBLE_CUDA_TESTED
#define VISIBLE_RAY_HITS

/**
 * \brief Cuda BVH class.
 * \details Graphic card friendly version of the BVH acceleration structure.
 */
class CudaBVH : public CudaAS
{
public:
    enum
    {
        Align = 4096
    };

public:
	/**
	 * \brief Constructor.
	 * \param[in] bvh Existing BVH that will be converted.
	 * \param[in] layout Layout of buffers.
	 */
    explicit    CudaBVH             (const BVH& bvh, BVHLayout layout);

	/**
	 * \brief Constructor
	 * \param[in] layout Layout of buffers.
	 */
	explicit    CudaBVH             (BVHLayout layout) : m_layout(layout) { ; }

	/**
	 * \brief Copy constructor.
	 * \param[in] other Existing Cuda BVH to be copied.
	 */
                CudaBVH             (CudaBVH& other)        { operator=(other); }

	/**
	 * \brief Constructor. Reads Cuda BVH from a file.
	 * \param[in] in Input stream to read the Cuda BVH from.
	 */
    explicit    CudaBVH             (InputStream& in);

	/**
	 * \brief Destructor.
	 */
                ~CudaBVH            (void);

	/**
	 * \return Layout of buffers.
	 */
    BVHLayout   getLayout           (void) const            { return m_layout; }

	/**
	 * \return Node buffer.
	 */
    Buffer&     getNodeBuffer       (void)                  { return m_nodes; }

	/**
	 * \return Woop triangle buffer.
	 */
    Buffer&     getTriWoopBuffer    (void)                  { return m_triWoop; }

	/**
	 * \return Triangle index buffer.
	 */
    Buffer&     getTriIndexBuffer   (void)                  { return m_triIndex; }

	/**
	 * \brief Returns node subarray.
	 * \details AOS: idx ignored, returns entire buffer; SOA: 0 <= idx < 4, returns one subarray.
	 * \return Node subarray.
	 */
    Vec2i       getNodeSubArray     (int idx) const; // (ofs, size)

	/**
	 * \brief Returns woop triangle subarray.
	 * \details AOS: idx ignored, returns entire buffer; SOA: 0 <= idx < 4, returns one subarray.
	 * \return Woop triangle subarray.
	 */
    Vec2i       getTriWoopSubArray  (int idx) const; // (ofs, size)

	/**
	 * \brief Assignment operator.
	 * \param[in] other Cuda BVH to assign.
	 * \return Result of the assignment.
	 */
    CudaBVH&    operator=           (CudaBVH& other);

	/**
	 * \brief Writes Cuda BVH to the output stream.
	 * \param[in] Target to write to.
	 */
    void        serialize           (OutputStream& out);

	void        setTraceParams      (Platform* platform, Scene* scene) { FW_ASSERT(platform && scene); m_platform = platform; m_scene = scene; }
	void        findVisibleTriangles(RayBuffer& rays, S32* references, S32 offset);
	void        trace               (RayBuffer& rays, Buffer& visibility, bool twoTrees, RayStats* stats = NULL);
	//void        trace               (RayBuffer& rays, CudaBVH& emptyBVH, RayStats* stats = NULL);
	void        trace               (RayBuffer& rays, Buffer& visibility, Array<AABB>& emptyBVH, RayStats* stats = NULL);

	bool        isLeaf              (S32 node)              {return node < 0;}
	void        getNode             (S32 node, SplitInfo *splitInfo, AABB &child0, AABB &child1, S32 &child0Addr, S32 &child1Addr);
	void        getTriangleIndices  (S32 node, Array<S32>& indices);

	Scene*		getScene			()						{return m_scene;}

private:
    void        createNodeBasic     (const BVH& bvh);
    void        createTriWoopBasic  (const BVH& bvh);
    void        createTriIndexBasic (const BVH& bvh);
    void        createCompact       (const BVH& bvh, int nodeOffsetSizeDiv);

    void        woopifyTri          (const BVH& bvh, int idx);
	
	template <BVHLayout LAYOUT>
	void        trace               (S32 node, Ray& ray, RayResult& result);
	template <BVHLayout LAYOUT>
	//void        trace               (S32 node, Ray& ray, RayResult& result, CudaBVH& emptyBVH);
	void        trace               (S32 node, Ray& ray, RayResult& result, Array<AABB>& emptyBVH);
	template <BVHLayout LAYOUT>
	bool        intersectTriangles  (S32 node, Ray& ray, RayResult& result);
	template <BVHLayout LAYOUT>
	void        getNodeTemplate     (S32 node, SplitInfo *splitInfo, AABB &child0, AABB &child1, S32 &child0Addr, S32 &child1Addr);
	bool        updateHit           (Ray& ray, RayResult& result, float t, S32 index);

private:
    BVHLayout   m_layout;
    Buffer      m_nodes;
    Buffer      m_triWoop;
    Buffer      m_triIndex;
    Vec4f       m_woop[3];
	
	Platform*   m_platform;
	Scene*      m_scene;

	bool        m_needClosestHit;
	S32*        m_references;
	S32         m_offset;

public:
	RayStats*   m_stats;
};

//------------------------------------------------------------------------
}

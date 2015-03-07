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

#define DFS 0
#define BFS 1
#define RND 2

#define METHOD RND

#include "base/Random.hpp"
#include <queue>

#include "cuda/CudaBVH.hpp"
#include "base/Sort.hpp"

#define MASK_TRACE_EMPTY

using namespace FW;

// Explicit declarations of specializations: needed so it does not matter when they are first used
template <>
bool CudaBVH::intersectTriangles<BVHLayout_AOS_AOS>(S32 node, Ray& ray, RayResult& result);
template <>
bool CudaBVH::intersectTriangles<BVHLayout_Compact>(S32 node, Ray& ray, RayResult& result);
template <>
bool CudaBVH::intersectTriangles<BVHLayout_CPU>(S32 node, Ray& ray, RayResult& result);
//template <>
//void CudaBVH::getNodeTemplate<BVHLayout_AOS_AOS>(S32 node, SplitInfo *splitInfo, AABB &child0, AABB &child1, S32 &child0Addr, S32 &child1Addr);
template <>
void CudaBVH::getNodeTemplate<BVHLayout_Compact>(S32 node, SplitInfo *splitInfo, AABB &child0, AABB &child1, S32 &child0Addr, S32 &child1Addr);
//template <>
//void CudaBVH::getNodeTemplate<BVHLayout_CPU>(S32 node, SplitInfo *splitInfo, AABB &child0, AABB &child1, S32 &child0Addr, S32 &child1Addr);

//------------------------------------------------------------------------

CudaBVH::CudaBVH(const BVH& bvh, BVHLayout layout)
:   m_layout    (layout)
{
    FW_ASSERT(layout >= 0 && layout < BVHLayout_Max);

    if (layout == BVHLayout_Compact)
    {
#if METHOD == DFS
		printf("DFS~~~~~~~\n");
        createCompact(bvh,1);
		return;
#elif METHOD == RND
		printf("RND~~~~~~~~\n");
		createCompact(bvh, 1);
		Shuffle();
		return;
#else
		printf("BFS~~~~~~~~~\n");
		createCompactBFS(bvh);
		return;
#endif
    }
	else
	{
		FW_ASSERT(false);
	}

    if (layout == BVHLayout_Compact2)
    {
        createCompact(bvh,16);
#ifdef SHUFFLE
		Shuffle();
#endif
        return;
    }

    createNodeBasic(bvh);
	if (layout != BVHLayout_CPU)
		createTriWoopBasic(bvh);
    createTriIndexBasic(bvh);

#ifdef SHUFFLE
	Shuffle();
#endif
}

//------------------------------------------------------------------------

CudaBVH::CudaBVH(InputStream& in)
{
    in >> (S32&)m_layout >> m_nodes >> m_triWoop >> m_triIndex;
}

//------------------------------------------------------------------------

CudaBVH::~CudaBVH(void)
{
}

//------------------------------------------------------------------------

void CudaBVH::serialize(OutputStream& out)
{
	// Transfer data to CPU before serialization
	m_nodes.setOwner(Buffer::CPU, false);
	m_triWoop.setOwner(Buffer::CPU, false);
	m_triIndex.setOwner(Buffer::CPU, false);
    out << (S32)m_layout << m_nodes << m_triWoop << m_triIndex;
}

//------------------------------------------------------------------------

void CudaBVH::findVisibleTriangles(RayBuffer& rays, S32* references, S32 offset)
{
#if defined(VISIBLE_CUDA_TESTED) // Copy info data from the GPU
	for(int i = 0; i < m_triFlags.getSize()>>2; i++)
	{
		S32 flag = *(S32*)m_triFlags.getPtr(i * 4);
		S32 *ref = references + i*offset;
		(*ref) = flag>0 ? 1 : 0;
	}
#elif defined(VISIBLE_RAY_HITS) // Get visibility info from the hit triangles
	for(S32 i=0;i<rays.getSize();i++)
		{
			const RayResult& result = rays.getResultForSlot(i);

			// Increment the triangle hit count
			if(result.hit())
			{
				S32 *ref = references + result.id*offset;
				if((*ref) == 0)
					(*ref)++;
			}
		}
#else // Compute the visibility info by tracing the rays
	m_needClosestHit = rays.getNeedClosestHit();
	m_stats = NULL;
	m_references = references;
	m_offset = offset;

    for(S32 i=0;i<rays.getSize();i++)
    {
        Ray ray = rays.getRayForSlot(i);    // takes a local copy
        RayResult& result = rays.getMutableResultForSlot(i);

        result.clear();

#ifdef VISIBLE_HIDDEN
		m_rayHidden = 0;
#endif

        switch(m_layout)
		{
		case BVHLayout_AOS_AOS:
			trace<BVHLayout_AOS_AOS>(0, ray, result);
			break;
		case BVHLayout_Compact:
			trace<BVHLayout_Compact>(0, ray, result);
			break;
		case BVHLayout_CPU:
			trace<BVHLayout_CPU>(0, ray, result);
			break;
		default:
			FW_ASSERT(0);
		}

		//if(result.hit())         // Updates the ray so that it cannot traverse further than the closest hit (Only for OSAH build)
		//{
		//	rays.getMutableRayForSlot(i).tmax = result.t; // Works only for benchmark, in interactive this code is not run for each frame!
		//	                                              // However, this is almost correct behaviour as for OSAH and interactive the BVH is not updated when camera moves
		//}

#if !defined(VISIBLE_TOUCHED) && !defined(VISIBLE_TOUCHED_TESTED) && !defined(VISIBLE_HIDDEN)
		// Set hit triangle as visible
		if(result.hit())
		{
			S32 *ref = references + result.id*offset;
			if((*ref) == 0)
				(*ref)++;
		}
#endif
#ifdef VISIBLE_HIDDEN
		if(m_references)
		{
			S32 *ref = m_references + result.id*(*m_offset);
			(*ref) = m_rayHidden == 0 ? 0 : m_rayHidden-1;
		}
#endif
    }

	//rays.getRayBuffer().setDirtyExcept(Buffer::CPU); // We have updated the CPU buffer, mark the other as dirty - they will be updated later
#endif
}

//------------------------------------------------------------------------

void CudaBVH::trace(RayBuffer& rays, Buffer& visibility, bool twoTrees, RayStats* stats)
{
	m_needClosestHit = rays.getNeedClosestHit();
	m_stats = stats;
	m_references = NULL;

	S32* visib = (S32*)visibility.getMutablePtr();

	/*Ray ray = rays.getRayForSlot(0);    // takes a local copy
	ray.direction = Vec3f(0.f, -1.f, 0.f);
    RayResult& result = rays.getMutableResultForSlot(0);
	result.clear();
	trace<BVHLayout_Compact>(0, ray, result);*/

	if(twoTrees)
	{
		for(S32 i=0;i<rays.getSize();i++)
		{
			Ray ray = rays.getRayForSlot(i);    // takes a local copy
			RayResult& result = rays.getMutableResultForSlot(i);

			result.clear();
			result.t = ray.tmax;

			if(stats)
			{
				stats->platform = *m_platform;
				stats->numRays++;
			}

			switch(m_layout)
			{
			case BVHLayout_AOS_AOS:
				trace<BVHLayout_AOS_AOS>(1, ray, result);
				trace<BVHLayout_AOS_AOS>(2, ray, result);
				break;
			case BVHLayout_Compact:
				trace<BVHLayout_Compact>(64, ray, result);
				trace<BVHLayout_Compact>(128, ray, result);
				break;
			case BVHLayout_CPU:
				trace<BVHLayout_CPU>(1, ray, result);
				trace<BVHLayout_CPU>(2, ray, result);
				break;
			default:
				FW_ASSERT("Unspported BVH layout\n");
			}

			// Set visibility
			if(visibility.getSize() > 0 && result.hit())
				visib[result.id] = 1;
		}
	}
	else
	{
		for(S32 i=0;i<rays.getSize();i++)
		{
			Ray ray = rays.getRayForSlot(i);    // takes a local copy
			RayResult& result = rays.getMutableResultForSlot(i);

			result.clear();
			result.t = ray.tmax;

			if(stats)
			{
				stats->platform = *m_platform;
				stats->numRays++;
			}

			switch(m_layout)
			{
			case BVHLayout_AOS_AOS:
				trace<BVHLayout_AOS_AOS>(0, ray, result);
				break;
			case BVHLayout_Compact:
				trace<BVHLayout_Compact>(0, ray, result);
				break;
			case BVHLayout_CPU:
				trace<BVHLayout_CPU>(0, ray, result);
				break;
			default:
				FW_ASSERT("Unspported BVH layout\n");
			}

			// Set visibility
			if(visibility.getSize() > 0 && result.hit())
				visib[result.id] = 1;
		}
	}
}


//------------------------------------------------------------------------

//void CudaBVH::trace(RayBuffer& rays, CudaBVH& emptyBVH, RayStats* stats)
void CudaBVH::trace(RayBuffer& rays, Buffer& visibility, Array<AABB>& emptyBVH, RayStats* stats)
{
	m_needClosestHit = rays.getNeedClosestHit();
	m_stats = stats;
	m_references = NULL;

	S32* visib = (S32*)visibility.getMutablePtr();

	/*Ray ray = rays.getRayForSlot(0);    // takes a local copy
	ray.direction = Vec3f(0.f, -1.f, 0.f);
    RayResult& result = rays.getMutableResultForSlot(0);
	result.clear();
	trace<BVHLayout_Compact>(0, ray, result, emptyBVH);*/

	for(S32 i=0;i<rays.getSize();i++)
	{
		Ray ray = rays.getRayForSlot(i);    // takes a local copy
		RayResult& result = rays.getMutableResultForSlot(i);

		result.clear();

		if(stats)
		{
			stats->platform = *m_platform;
			stats->numRays++;
		}

		switch(m_layout)
		{
		case BVHLayout_AOS_AOS:
			trace<BVHLayout_AOS_AOS>(0, ray, result, emptyBVH);
			break;
		case BVHLayout_Compact:
			trace<BVHLayout_Compact>(0, ray, result, emptyBVH);
			break;
		case BVHLayout_CPU:
			trace<BVHLayout_CPU>(0, ray, result, emptyBVH);
			break;
		default:
			FW_ASSERT("Unspported BVH layout\n");
		}

		// Set visibility
		if(result.hit())
			visib[result.id] = 1;
	}
}

//------------------------------------------------------------------------

void CudaBVH::getNode(S32 node, SplitInfo *splitInfo, AABB &child0, AABB &child1, S32 &child0Addr, S32 &child1Addr)
{
	switch(m_layout)
	{
	case BVHLayout_Compact:
		getNodeTemplate<BVHLayout_Compact>(node, splitInfo, child0, child1, child0Addr, child1Addr);
		break;
	case BVHLayout_AOS_AOS:
	case BVHLayout_CPU:
		getNodeTemplate<BVHLayout_CPU>(node, splitInfo, child0, child1, child0Addr, child1Addr);
		break;
	default:
		FW_ASSERT("Unspported BVH layout\n");
	}
}

//------------------------------------------------------------------------

void CudaBVH::getTriangleIndices(S32 node, Array<S32>& indices)
{
	Buffer &nodes = getNodeBuffer();
	Buffer &tris = getTriIndexBuffer();
	Buffer &woop = getTriWoopBuffer();

	switch(m_layout)
	{
	case BVHLayout_Compact:
		{
			for(int triAddr = (-node-1); ; triAddr += 3)
			{
				U32 guard = floatToBits(*(F32*)woop.getMutablePtr(triAddr * 16 + 0));
				if(guard == 0x80000000)
					break;

				indices.add(*(S32*)tris.getMutablePtr(triAddr*4));
			}
		}
		break;
	case BVHLayout_AOS_AOS:
	case BVHLayout_CPU:
		{
			node = (-node-1);

			int lo = *(S32*)nodes.getMutablePtr(node * 64 + 48);
			int hi = *(S32*)nodes.getMutablePtr(node * 64 + 52);

			for(int i=lo; i<hi; i++)
				indices.add(*(S32*)tris.getMutablePtr(i*4));
		}
		break;
	default:
		FW_ASSERT("Unspported BVH layout\n");
	}
}

//------------------------------------------------------------------------

Vec2i CudaBVH::getNodeSubArray(int idx) const
{
    FW_ASSERT(idx >= 0 && idx < 4);
    S32 size = (S32)m_nodes.getSize();

    if (m_layout == BVHLayout_SOA_AOS || m_layout == BVHLayout_SOA_SOA)
        return Vec2i((size >> 2) * idx, (size >> 2));
    return Vec2i(0, size);
}

//------------------------------------------------------------------------

Vec2i CudaBVH::getTriWoopSubArray(int idx) const
{
    FW_ASSERT(idx >= 0 && idx < 4);
    S32 size = (S32)m_triWoop.getSize();

    if (m_layout == BVHLayout_AOS_SOA || m_layout == BVHLayout_SOA_SOA)
        return Vec2i((size >> 2) * idx, (size >> 2));
    return Vec2i(0, size);
}

//------------------------------------------------------------------------

CudaBVH& CudaBVH::operator=(CudaBVH& other)
{
    if (&other != this)
    {
        m_layout    = other.m_layout;
        m_nodes     = other.m_nodes;
        m_triWoop   = other.m_triWoop;
        m_triIndex  = other.m_triIndex;
    }
    return *this;
}

//------------------------------------------------------------------------

void CudaBVH::createNodeBasic(const BVH& bvh)
{
    struct StackEntry
    {
        const BVHNode*  node;
        S32             idx;

        StackEntry(const BVHNode* n = NULL, int i = 0) : node(n), idx(i) {}
        int encodeIdx(void) const { return (node->isLeaf()) ? ~idx : idx; }
    };

    const BVHNode* root = bvh.getRoot();
    m_nodes.resizeDiscard((root->getSubtreeSize(BVH_STAT_NODE_COUNT) * 64 + Align - 1) & -Align);

    int nextNodeIdx = 0;
    Array<StackEntry> stack(StackEntry(root, nextNodeIdx++));
    while (stack.getSize())
    {
        StackEntry e = stack.removeLast();
        const AABB* b0;
        const AABB* b1;
        int c0;
        int c1;
        SplitInfo splitInfo;

        // Leaf?

        if (e.node->isLeaf())
        {
            const LeafNode* leaf = reinterpret_cast<const LeafNode*>(e.node);
            b0 = &leaf->m_bounds;
            b1 = &leaf->m_bounds;
            c0 = leaf->m_lo;
            c1 = leaf->m_hi;
        }

        // Internal node?

        else
        {
            StackEntry e0 = stack.add(StackEntry(e.node->getChildNode(0), nextNodeIdx++));
            StackEntry e1 = stack.add(StackEntry(e.node->getChildNode(1), nextNodeIdx++));
            b0 = &e0.node->m_bounds;
            b1 = &e1.node->m_bounds;
            c0 = e0.encodeIdx();
            c1 = e1.encodeIdx();
            const InnerNode *eN = reinterpret_cast<const InnerNode*>(e.node);
            splitInfo = eN->getSplitInfo();
        }

        // Write entry.

        Vec4i data[] =
        {
            Vec4i(floatToBits(b0->min().x), floatToBits(b0->max().x), floatToBits(b0->min().y), floatToBits(b0->max().y)),
            Vec4i(floatToBits(b1->min().x), floatToBits(b1->max().x), floatToBits(b1->min().y), floatToBits(b1->max().y)),
            Vec4i(floatToBits(b0->min().z), floatToBits(b0->max().z), floatToBits(b1->min().z), floatToBits(b1->max().z)),
            //Vec4i(c0, c1, 0, 0)
            Vec4i(c0, c1, splitInfo.getBitCode(), 0)
        };

        switch (m_layout)
        {
        case BVHLayout_AOS_AOS:
        case BVHLayout_AOS_SOA:
        case BVHLayout_CPU:
            memcpy(m_nodes.getMutablePtr(e.idx * 64), data, 64);
            break;

        case BVHLayout_SOA_AOS:
        case BVHLayout_SOA_SOA:
            for (int i = 0; i < 4; i++)
                memcpy(m_nodes.getMutablePtr(e.idx * 16 + (m_nodes.getSize() >> 2) * i), &data[i], 16);
            break;

        default:
            FW_ASSERT(false);
            break;
        }
    }
}

//------------------------------------------------------------------------

void CudaBVH::createTriWoopBasic(const BVH& bvh)
{
    const Array<S32>& tidx = bvh.getTriIndices();
    m_triWoop.resizeDiscard((tidx.getSize() * 64 + Align - 1) & -Align);

    for (int i = 0; i < tidx.getSize(); i++)
    {
        woopifyTri(bvh, i);

        switch (m_layout)
        {
        case BVHLayout_AOS_AOS:
        case BVHLayout_SOA_AOS:
            memcpy(m_triWoop.getMutablePtr(i * 64), m_woop, 48);
            break;

        case BVHLayout_AOS_SOA:
        case BVHLayout_SOA_SOA:
            for (int j = 0; j < 3; j++)
                memcpy(m_triWoop.getMutablePtr(i * 16 + (m_triWoop.getSize() >> 2) * j), &m_woop[j], 16);
            break;

        default:
            FW_ASSERT(false);
            break;
        }
    }
}

//------------------------------------------------------------------------

void CudaBVH::createTriIndexBasic(const BVH& bvh)
{
    const Array<S32>& tidx = bvh.getTriIndices();
    m_triIndex.resizeDiscard(tidx.getSize() * 4);

    for (int i = 0; i < tidx.getSize(); i++)
        *(S32*)m_triIndex.getMutablePtr(i * 4) = tidx[i];
}

//------------------------------------------------------------------------

void CudaBVH::createCompact(const BVH& bvh, int nodeOffsetSizeDiv)
{
    struct StackEntry
    {
        const BVHNode*  node;
        S32             idx;

        StackEntry(const BVHNode* n = NULL, int i = 0) : node(n), idx(i) {}
    };

    // Construct data.

    Array<Vec4i> nodeData(NULL, 4);
    Array<Vec4i> triWoopData;
    Array<S32> triIndexData;
    Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0));

    while (stack.getSize())
    {
        StackEntry e = stack.removeLast();
		//printf("%i %f | ", e.idx/4, e.node->getArea());
        FW_ASSERT(e.node->getNumChildNodes() == 2);
        const AABB* cbox[2];
        int cidx[2];

        // Process children.

        for (int i = 0; i < 2; i++)
        {
            // Inner node => push to stack.

            const BVHNode* child = e.node->getChildNode(i);
            cbox[i] = &child->m_bounds;
            if (!child->isLeaf())
            {
                cidx[i] = nodeData.getNumBytes() / nodeOffsetSizeDiv;
                stack.add(StackEntry(child, nodeData.getSize()));
                nodeData.add(NULL, 4);
                continue;
            }

            // Leaf => append triangles.

            const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child);
            cidx[i] = ~triWoopData.getSize();
            for (int j = leaf->m_lo; j < leaf->m_hi; j++)
            {
                woopifyTri(bvh, j);
                if (m_woop[0].x == 0.0f)
                    m_woop[0].x = 0.0f;
                triWoopData.add((Vec4i*)m_woop, 3);
                triIndexData.add(bvh.getTriIndices()[j]);
                triIndexData.add(0);
                triIndexData.add(0);
            }

            // Terminator.

            triWoopData.add(0x80000000);
            triIndexData.add(0);
        }

        const InnerNode *eN = reinterpret_cast<const InnerNode*>(e.node);
        const SplitInfo &splitInfo = eN->getSplitInfo();

        // Write entry.

        Vec4i* dst = nodeData.getPtr(e.idx);
        dst[0] = Vec4i(floatToBits(cbox[0]->min().x), floatToBits(cbox[0]->max().x), floatToBits(cbox[0]->min().y), floatToBits(cbox[0]->max().y));
        dst[1] = Vec4i(floatToBits(cbox[1]->min().x), floatToBits(cbox[1]->max().x), floatToBits(cbox[1]->min().y), floatToBits(cbox[1]->max().y));
        dst[2] = Vec4i(floatToBits(cbox[0]->min().z), floatToBits(cbox[0]->max().z), floatToBits(cbox[1]->min().z), floatToBits(cbox[1]->max().z));
        //dst[3] = Vec4i(cidx[0], cidx[1], 0, 0);
        dst[3] = Vec4i(cidx[0], cidx[1], splitInfo.getBitCode(), 0);
    }

    // Write to buffers.

    m_nodes.resizeDiscard(nodeData.getNumBytes());
    m_nodes.set(nodeData.getPtr(), nodeData.getNumBytes());

    m_triWoop.resizeDiscard(triWoopData.getNumBytes());
    m_triWoop.set(triWoopData.getPtr(), triWoopData.getNumBytes());

    m_triIndex.resizeDiscard(triIndexData.getNumBytes());
    m_triIndex.set(triIndexData.getPtr(), triIndexData.getNumBytes());
}

//------------------------------------------------------------------------

void CudaBVH::woopifyTri(const BVH& bvh, int idx)
{
    const Vec3i* triVtxIndex = (const Vec3i*)bvh.getScene()->getTriVtxIndexBuffer().getPtr();
    const Vec3f* vtxPos = (const Vec3f*)bvh.getScene()->getVtxPosBuffer().getPtr();
    const Vec3i& inds = triVtxIndex[bvh.getTriIndices()[idx]];
    const Vec3f& v0 = vtxPos[inds.x];
    const Vec3f& v1 = vtxPos[inds.y];
    const Vec3f& v2 = vtxPos[inds.z];

    Mat4f mtx;
    mtx.setCol(0, Vec4f(v0 - v2, 0.0f));
    mtx.setCol(1, Vec4f(v1 - v2, 0.0f));
    mtx.setCol(2, Vec4f(cross(v0 - v2, v1 - v2), 0.0f));
    mtx.setCol(3, Vec4f(v2, 1.0f));
    mtx = invert(mtx);

    m_woop[0] = Vec4f(mtx(2,0), mtx(2,1), mtx(2,2), -mtx(2,3));
    m_woop[1] = mtx.getRow(0);
    m_woop[2] = mtx.getRow(1);
}

//------------------------------------------------------------------------

S32 _max(S32 a, F32 b)
{
	return a > b ? a : b;
}

//------------------------------------------------------------------------

template <BVHLayout LAYOUT>
void CudaBVH::trace(S32 node, Ray& ray, RayResult& result)
{
	S32 stack[100];
	//F32 tStack[100];
	int stackIndex = 1;	

	while(stackIndex > 0)
	{
		for(;;)
		{
			if(node < 0)
			{
				bool end = intersectTriangles<LAYOUT>(node, ray, result);
				if(end)
					return;

				break;
			}
			else
			{
				const int TMIN = 0;
				const int TMAX = 1;

				AABB child0, child1;
				S32 child0Addr, child1Addr;

				getNodeTemplate<LAYOUT>(node, NULL, child0, child1, child0Addr, child1Addr);

				Vec2f tspan0 = Intersect::RayBox(child0, ray);
				Vec2f tspan1 = Intersect::RayBox(child1, ray);
#ifdef VISIBLE_HIDDEN
				bool intersect0, intersect1;
				if(m_references)
				{
					intersect0 = (tspan0[TMIN]<=tspan0[TMAX]) && (tspan0[TMAX]>=ray.tmin);
					intersect1 = (tspan1[TMIN]<=tspan1[TMAX]) && (tspan1[TMAX]>=ray.tmin);
				}
				else
				{
					intersect0 = (tspan0[TMIN]<=tspan0[TMAX]) && (tspan0[TMAX]>=ray.tmin) && (tspan0[TMIN]<=ray.tmax);
					intersect1 = (tspan1[TMIN]<=tspan1[TMAX]) && (tspan1[TMAX]>=ray.tmin) && (tspan1[TMIN]<=ray.tmax);
				}
#else
				bool intersect0 = (tspan0[TMIN]<=tspan0[TMAX]) && (tspan0[TMAX]>=ray.tmin) && (tspan0[TMIN]<=ray.tmax);
				bool intersect1 = (tspan1[TMIN]<=tspan1[TMAX]) && (tspan1[TMAX]>=ray.tmin) && (tspan1[TMIN]<=ray.tmax);
#endif

				if(m_stats)
				{
					m_stats->numNodeTests += m_platform->roundToNodeBatchSize( 2 );
					result.padB += m_platform->roundToNodeBatchSize( 2 );

					if(intersect0 && intersect1)
						result.padA = _max(result.padA, max(tspan0[TMIN], tspan1[TMIN]));
					else if(intersect0)
						result.padA = _max(result.padA, tspan0[TMIN]);
					else if(intersect1)
						result.padA = _max(result.padA, tspan1[TMIN]);
				}

				if(intersect0 && intersect1)
				{
					if(tspan0[TMIN] > tspan1[TMIN])
					{
						swap(tspan0,tspan1);
						swap(child0Addr,child1Addr);
					}
					node = child0Addr;
					//tStack[stackIndex] = tspan1[TMIN];
					stack[stackIndex++] = child1Addr;
				}
				else if(intersect0)
					node = child0Addr;
				else if(intersect1)
					node = child1Addr;
				else
					break;
			}
		}
		//do
		//{
			stackIndex--;
			node = stack[stackIndex];
		//} while(tStack[stackIndex] > ray.tmax);
	}
}

//------------------------------------------------------------------------

//int eCompare(void* data, int idxA, int idxB)
bool eCompare(void* data, int idxA, int idxB)
{
    const Vec2f* ptr = (const Vec2f*)data;
    const F32& ma = ptr[idxA].x;
    const F32& mb = ptr[idxB].x;
    return (ma < mb) ? false : (ma > mb) ? true : false;
}

//------------------------------------------------------------------------

void eSwap(void* data, int idxA, int idxB)
{
    Vec2f* ptr = (Vec2f*)data;
    FW::swap(ptr[idxA], ptr[idxB]);
}


//------------------------------------------------------------------------

template <BVHLayout LAYOUT>
//void CudaBVH::trace(S32 node, Ray& ray, RayResult& result, CudaBVH& emptyBVH)
void CudaBVH::trace(S32 node, Ray& ray, RayResult& result, Array<AABB>& emptyBVH)
{
	S32 stack[100];
	//Vec2f inter[100];
	Vec2f empty[12];
	int stackIndex = 1;
	int emptyIndex = 1;

	// Find empty intervals
	int hitCount = 1;
	for(int i = 0; i < emptyBVH.getSize(); i++)
	{
		Vec2f tspan = Intersect::RayBox(emptyBVH[i], ray);
		if(tspan.x<tspan.y && tspan.y>ray.tmin) // box hit
		{
			empty[hitCount] = tspan;
			hitCount++;
		}
	}
	//if(m_stats)
	//{
	//	m_stats->numNodeTests += m_platform->roundToNodeBatchSize( emptyBVH.getSize() );
	//	result.padB += m_platform->roundToNodeBatchSize( emptyBVH.getSize() );
	//}

	empty[0] = Vec2f((hitCount == 1) ? FW_F32_MAX : -FW_F32_MAX, ray.tmin); // If no empty box is hit, disable empty skips

	// Sort the empty intervals
	//sort(1, hitCount, empty, eCompare, eSwap);
	sort(empty, 1, hitCount, eCompare, eSwap);

	// Compact the empty intervals
	for(int i = 1; i < hitCount;)
	{
		if(empty[i].x < empty[i-1].y) // Overlapping
		{
			empty[i-1].y = max(empty[i-1].y, empty[i].y);
			hitCount--;
			memmove(&empty[i], &empty[i+1], (hitCount-i)*sizeof(Vec2f));
		}
		else
		{
			i++;
		}
	}

	empty[hitCount] = Vec2f(FW_F32_MAX, ray.tmin); // Set the end of the array
	ray.tmin = empty[0].y;

	// Root skip
	/*Vec2f tspan = Intersect::RayBox(emptyBVH[0], ray);
	if(tspan.x <= ray.tmin)
		ray.tmin = max(ray.tmin, tspan.y);*/

	// Trace the empty BVH
	/*Vec2f tspan0;
	while(stackIndex > 0)
	{
		for(;;)
		{
			if(node < 0)
			{
				if(emptyIndex != 0 && tspan0.x < empty[emptyIndex-1].y)
				{
					empty[emptyIndex-1].y = max(tspan0.y, empty[emptyIndex-1].y);
				}
				else
				{
					empty[emptyIndex] = tspan0;
					//empty[emptyIndex] = Vec2f(FW_F32_MAX, FW_F32_MAX);
					emptyIndex++;
				}

				break;
			}
			else
			{
				const int TMIN = 0;
				const int TMAX = 1;

				AABB child0, child1;
				S32 child0Addr, child1Addr;

				emptyBVH.getNodeTemplate<LAYOUT>(node, NULL, child0, child1, child0Addr, child1Addr);

				      tspan0 = Intersect::RayBox(child0, ray);
				Vec2f tspan1 = Intersect::RayBox(child1, ray);

				bool intersect0 = (tspan0[TMIN]<=tspan0[TMAX]) && (tspan0[TMAX]>=ray.tmin);
				bool intersect1 = (tspan1[TMIN]<=tspan1[TMAX]) && (tspan1[TMAX]>=ray.tmin);

				if(m_stats)
				{
					m_stats->numNodeTests += m_platform->roundToNodeBatchSize( 2 );
					result.padB += m_platform->roundToNodeBatchSize( 2 );
				}

				if(intersect0 && intersect1)
				{
					if(tspan0[TMIN] > tspan1[TMIN])
					{
						swap(tspan0,tspan1);
						swap(child0Addr,child1Addr);
					}
					node = child0Addr;
					stack[stackIndex] = child1Addr;
					inter[stackIndex] = tspan1;
					stackIndex++;
				}
				else if(intersect0)
				{
					node = child0Addr;
				}
				else if(intersect1)
				{
					node = child1Addr;
					tspan0 = tspan1;
				}
				else
					break;
			}
		}
		
		stackIndex--;
		node = stack[stackIndex];
		tspan0 = inter[stackIndex];
	}

	stackIndex = 1;
	node = 0;   // Start from the root.
	empty[emptyIndex] = Vec2f(FW_F32_MAX, FW_F32_MAX); // Set the end of the array
	emptyIndex = 0; // Rewind to the beginning*/
	S32 eidx[100];
	eidx[0] = 0;

	// Trace the BVH
	while(stackIndex > 0)
	{
		for(;;)
		{
			if(node < 0)
			{
				bool end = intersectTriangles<LAYOUT>(node, ray, result);
				if(end)
					return;

				break;
			}
			else
			{
				const int TMIN = 0;
				const int TMAX = 1;

				AABB child0, child1;
				S32 child0Addr, child1Addr;

				getNodeTemplate<LAYOUT>(node, NULL, child0, child1, child0Addr, child1Addr);

				Vec2f tspan0 = Intersect::RayBox(child0, ray);
				Vec2f tspan1 = Intersect::RayBox(child1, ray);

				bool intersect0 = (tspan0[TMIN]<=tspan0[TMAX]) && (tspan0[TMAX]>=ray.tmin) && (tspan0[TMIN]<=ray.tmax);
				bool intersect1 = (tspan1[TMIN]<=tspan1[TMAX]) && (tspan1[TMAX]>=ray.tmin) && (tspan1[TMIN]<=ray.tmax);

				if(m_stats)
				{
					m_stats->numNodeTests += m_platform->roundToNodeBatchSize( 2 );
					result.padB += m_platform->roundToNodeBatchSize( 2 );

					if(intersect0 && intersect1)
						result.padA = _max(result.padA, max(tspan0[TMIN], tspan1[TMIN]));
					else if(intersect0)
						result.padA = _max(result.padA, tspan0[TMIN]);
	 				else if(intersect1)
						result.padA = _max(result.padA, tspan1[TMIN]);
				}

				if(intersect0 && intersect1)
				{
					if(tspan0[TMIN] > tspan1[TMIN])
					{
						swap(tspan0,tspan1);
						swap(child0Addr,child1Addr);
					}
					node = child0Addr;
					stack[stackIndex] = child1Addr;
					if(tspan1[TMIN] > empty[emptyIndex].x)
						eidx[stackIndex] = emptyIndex+1;
					else
						eidx[stackIndex] = emptyIndex;
					stackIndex++;
				}
				else if(intersect0)
				{
					node = child0Addr;
				}
				else if(intersect1)
				{
					node = child1Addr;
					tspan0 = tspan1;
				}
				else
					break;

				if(tspan0[TMIN] > empty[emptyIndex].x) // Pop node from empty stack and update tmin
				{
					ray.tmin = max(ray.tmin, empty[emptyIndex].y); // Set tmin to the empty box's cmax
					emptyIndex++;
				}
			}
		}
		
		stackIndex--;
		node = stack[stackIndex];
		emptyIndex = eidx[stackIndex];
		ray.tmin = empty[max(emptyIndex-1, 0)].y;
		//ray.tmin = 0.0f;
		//emptyIndex = 0;
	}
}

//------------------------------------------------------------------------

template <>
bool CudaBVH::intersectTriangles<BVHLayout_AOS_AOS>(S32 node, Ray& ray, RayResult& result)
{
	Buffer &nodes = getNodeBuffer();
	Buffer &woop = getTriWoopBuffer();
	Buffer &tris = getTriIndexBuffer();

	node = (-node-1);

	int lo, hi;
	lo = *(S32*)nodes.getMutablePtr(node * 64 + 48);
	hi = *(S32*)nodes.getMutablePtr(node * 64 + 52);
	if(m_stats)
	{
		m_stats->numTriangleTests += m_platform->roundToTriangleBatchSize( hi-lo );
		result.padB += m_platform->roundToTriangleBatchSize( hi-lo ) << 16;
	}

#ifndef MASK_TRACE_EMPTY
	if(lo == hi) // empty leaf
	{
		AABB bound;
		bound.min() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 0)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 8)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 32)));
		bound.max() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 4)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 12)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 36)));

		Vec2f tspan = Intersect::RayBox(bound, ray);
		ray.tmin = tspan[1];
	}
#endif

	for(int i=lo; i<hi; i++)
	{
		S32 index = *(S32*)tris.getMutablePtr(i*4);

		const Vec4f& zpleq = *(Vec4f*)woop.getMutablePtr(i * 64 + 0);
		const Vec4f& upleq = *(Vec4f*)woop.getMutablePtr(i * 64 + 16);
		const Vec4f& vpleq = *(Vec4f*)woop.getMutablePtr(i * 64 + 32);
		Vec3f bary = Intersect::RayTriangleWoop(zpleq,upleq,vpleq, ray);
		float t = bary[2];

		bool end = updateHit(ray, result, t, index);
		if(end)
			return true;
	}

	return false;
}

//------------------------------------------------------------------------

template <>
bool CudaBVH::intersectTriangles<BVHLayout_Compact>(S32 node, Ray& ray, RayResult& result)
{
	Buffer &woop = getTriWoopBuffer();
	Buffer &tris = getTriIndexBuffer();

	for(int triAddr = (-node-1); ; triAddr += 3)
	{
		U32 guard = floatToBits(*(F32*)woop.getMutablePtr(triAddr * 16 + 0));
		if(guard == 0x80000000)
			break;
#ifndef MASK_TRACE_EMPTY
		else if(floatToBits(*(F32*)woop.getMutablePtr(triAddr * 16 + 12)) == 0x80000000) // empty leaf
		{
			AABB bound;
			bound.min() = *(Vec3f*)woop.getMutablePtr(triAddr * 16 + 0);
			bound.max() = *(Vec3f*)woop.getMutablePtr(triAddr * 16 + 16);

			Vec2f tspan = Intersect::RayBox(bound, ray);
			ray.tmin = tspan[1];
			break;
		}
#endif

		if(m_stats)
		{
			m_stats->numTriangleTests ++;
			result.padB += 1 << 16;
		}

		S32 index = *(S32*)tris.getMutablePtr(triAddr*4);
		const Vec4f& zpleq = *(Vec4f*)woop.getMutablePtr(triAddr * 16 + 0);
		const Vec4f& upleq = *(Vec4f*)woop.getMutablePtr(triAddr * 16 + 16);
		const Vec4f& vpleq = *(Vec4f*)woop.getMutablePtr(triAddr * 16 + 32);
		Vec3f bary = Intersect::RayTriangleWoop(zpleq,upleq,vpleq, ray);
		float t = bary[2];

		bool end = updateHit(ray, result, t, index);
		if(end)
			return true;
	}

	return false;
}

//------------------------------------------------------------------------

template <>
bool CudaBVH::intersectTriangles<BVHLayout_CPU>(S32 node, Ray& ray, RayResult& result)
{
	Buffer &nodes = getNodeBuffer();
	Buffer &tris = getTriIndexBuffer();

	node = (-node-1);

	int lo, hi;
	lo = *(S32*)nodes.getMutablePtr(node * 64 + 48);
	hi = *(S32*)nodes.getMutablePtr(node * 64 + 52);
	if(m_stats)
	{
		m_stats->numTriangleTests += m_platform->roundToTriangleBatchSize( hi-lo );
		result.padB += m_platform->roundToTriangleBatchSize( hi-lo ) << 16;
	}

#ifndef MASK_TRACE_EMPTY
	if(lo == hi) // empty leaf
	{
		AABB bound;
		bound.min() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 0)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 8)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 32)));
		bound.max() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 4)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 12)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 36)));

		Vec2f tspan = Intersect::RayBox(bound, ray);
		ray.tmin = tspan[1];
	}
#endif

	for(int i=lo; i<hi; i++)
	{
		S32 index = *(S32*)tris.getMutablePtr(i*4);
		//const Vec3i& ind = m_scene->getTriangle(index).vertices;
		//const Vec3f& v0 = m_scene->getVertex(ind.x);
		//const Vec3f& v1 = m_scene->getVertex(ind.y);
		//const Vec3f& v2 = m_scene->getVertex(ind.z);
		const Vec3i& ind = ((Vec3i*)m_scene->getTriVtxIndexBuffer().getPtr())[index];
		const Vec3f& v0 = ((Vec3f*)m_scene->getVtxPosBuffer().getPtr())[ind.x];
		const Vec3f& v1 = ((Vec3f*)m_scene->getVtxPosBuffer().getPtr())[ind.y];
		const Vec3f& v2 = ((Vec3f*)m_scene->getVtxPosBuffer().getPtr())[ind.z];
		Vec3f bary = Intersect::RayTriangle(v0,v1,v2, ray);
		float t = bary[2];

		bool end = updateHit(ray, result, t, index);
		if(end)
			return true;
	}

	return false;
}

//------------------------------------------------------------------------

bool CudaBVH::updateHit(Ray& ray, RayResult& result, float t, S32 index)
{
#ifdef VISIBLE_TOUCHED
	// Set close triangle as visible
	if(m_references)
	{
		S32 *ref = m_references + index*(m_offset);
		if((*ref) == 0)
			(*ref)++;
	}
#endif

#ifdef VISIBLE_HIDDEN
	if(t < FW_F32_MAX)
		m_rayHidden++;
#endif

	if(t>ray.tmin && t<ray.tmax)
	{
#ifdef VISIBLE_TOUCHED_TESTED
		// Set close triangle as visible
		if(m_references)
		{
			S32 *ref = m_references + index*(m_offset);
			if((*ref) == 0)
				(*ref)++;
		}
#endif
#ifdef VISIBLE_HIDDEN
		if(!m_references)
			ray.tmax    = t;
#else
		ray.tmax    = t;
#endif
		result.t    = t;
		result.id   = index;

		if(!m_needClosestHit)
			return true;
	}

	return false;
}

//------------------------------------------------------------------------

template <BVHLayout LAYOUT>
void CudaBVH::getNodeTemplate(S32 node, SplitInfo *splitInfo, AABB &child0, AABB &child1, S32 &child0Addr, S32 &child1Addr)
{
	Buffer &nodes = getNodeBuffer();

	child0.min() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 0)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 8)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 32)));
	child0.max() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 4)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 12)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 36)));
	
	child1.min() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 16)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 24)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 40)));
	child1.max() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 20)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 28)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 44)));

	child0Addr = *(S32*)nodes.getMutablePtr(node * 64 + 48);
	child1Addr = *(S32*)nodes.getMutablePtr(node * 64 + 52);

	if(splitInfo != NULL)
		*splitInfo = SplitInfo(*(unsigned long*)nodes.getMutablePtr(node * 64 + 56));
}

//------------------------------------------------------------------------

template <>
void CudaBVH::getNodeTemplate<BVHLayout_Compact>(S32 node, SplitInfo *splitInfo, AABB &child0, AABB &child1, S32 &child0Addr, S32 &child1Addr)
{
	Buffer &nodes = getNodeBuffer();

	child0.min() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node + 0)), bitsToFloat(*(U32*)nodes.getMutablePtr(node + 8)), bitsToFloat(*(U32*)nodes.getMutablePtr(node + 32)));
	child0.max() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node + 4)), bitsToFloat(*(U32*)nodes.getMutablePtr(node + 12)), bitsToFloat(*(U32*)nodes.getMutablePtr(node + 36)));
	
	child1.min() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node + 16)), bitsToFloat(*(U32*)nodes.getMutablePtr(node + 24)), bitsToFloat(*(U32*)nodes.getMutablePtr(node + 40)));
	child1.max() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node + 20)), bitsToFloat(*(U32*)nodes.getMutablePtr(node + 28)), bitsToFloat(*(U32*)nodes.getMutablePtr(node + 44)));

	child0Addr = *(S32*)nodes.getMutablePtr(node + 48);
	child1Addr = *(S32*)nodes.getMutablePtr(node + 52);

	if(splitInfo != NULL)
		*splitInfo = SplitInfo(*(unsigned long*)nodes.getMutablePtr(node + 56));
}

/*template <>
void CudaBVH::getNodeTemplate<BVHLayout_CPU>(S32 node, SplitInfo *splitInfo, AABB &child0, AABB &child1, S32 &child0Addr, S32 &child1Addr)
{
	Buffer &nodes = getNodeBuffer();

	child0.min() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 0)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 8)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 32)));
	child0.max() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 4)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 12)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 36)));
	
	child1.min() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 16)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 24)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 40)));
	child1.max() = Vec3f(bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 20)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 28)), bitsToFloat(*(U32*)nodes.getMutablePtr(node * 64 + 44)));

	child0Addr = *(S32*)nodes.getMutablePtr(node * 64 + 48);
	child1Addr = *(S32*)nodes.getMutablePtr(node * 64 + 52);

	if(splitInfo != NULL)
		*splitInfo = SplitInfo(*(unsigned long*)nodes.getMutablePtr(node * 64 + 56));
}*/

//------------------------------------------------------------------------

void CudaBVH::Shuffle(S32 numOfSwaps)
{
	struct ParentInfo
	{
		S32 parentIdx;
		bool leftChild;

		ParentInfo(S32 idx, bool left) : parentIdx(idx), leftChild(left) {}
		ParentInfo() : parentIdx(-1) {}
	};


	const int numOfNodes = m_nodes.getSize() / 64;

	Array<ParentInfo> parents;
	parents.reset(numOfNodes);

	int nodeOffsetSizeDiv = (m_layout == BVHLayout_Compact ? 1 : 16);
	numOfSwaps = (numOfSwaps < 0 ? numOfNodes : numOfSwaps);

	switch (m_layout)
	{
	case BVHLayout_AOS_AOS:
	case BVHLayout_AOS_SOA:
	case BVHLayout_CPU:
	case BVHLayout_SOA_AOS:
	case BVHLayout_SOA_SOA:
		FW_ASSERT(false);
		break;
	case BVHLayout_Compact:
		{
			S32* ptr = (S32*)m_nodes.getPtr();
			for (int i = 0; i < m_nodes.getSize() / 64; i++)
			{
				S32 ptr1 = ptr[i*16+12] / 64;
				S32 ptr2 = ptr[i*16+13] / 64;

				if(ptr1 > 0)
					parents[ptr1] = ParentInfo(i, true);
				if(ptr2 > 0)
					parents[ptr2] = ParentInfo(i, false);
			}

			Random rnd;
			Vec4i* vecPtr = (Vec4i*)m_nodes.getPtr();
			numOfSwaps = numOfNodes;
			for (int i = 0; i < numOfSwaps; i++)
			{
				S32 idx1 = rnd.getU32(1, numOfNodes-1);
				S32 idx2 = rnd.getU32(1, numOfNodes-1);
				if(idx1 == idx2)
					continue;

				S32 chIdx1l = ptr[idx1 * 16 + 12] / 64;
				S32 chIdx1r = ptr[idx1 * 16 + 13] / 64;
				S32 chIdx2l = ptr[idx2 * 16 + 12] / 64;
				S32 chIdx2r = ptr[idx2 * 16 + 13] / 64;
			
				//printf("idx1: %u, idx2: %u | ", idx1, idx2);
				
				//swap nodes
				swap(vecPtr[idx1*4+0], vecPtr[idx2*4+0]);
				swap(vecPtr[idx1*4+1], vecPtr[idx2*4+1]);
				swap(vecPtr[idx1*4+2], vecPtr[idx2*4+2]);
				swap(vecPtr[idx1*4+3], vecPtr[idx2*4+3]);

				//swap parent info
				swap(parents[idx1], parents[idx2]);

				
				//fix parent info for children
				/*
				for (int i = 0; i < parents.getSize(); i++)
				{
					if(parents[i].parentIdx == idx1)
						int x = 1 + i;//parents[i].parentIdx = idx2;
					else if(parents[i].parentIdx == idx2)
						int x = i + i;//parents[i].parentIdx = idx1;
				}
				*/

				//fix parent info for children
				if(chIdx1l > 0 && parents[chIdx1l].parentIdx == idx1)
					parents[chIdx1l].parentIdx = idx2;
				if(chIdx1r > 0 && parents[chIdx1r].parentIdx == idx1)
					parents[chIdx1r].parentIdx = idx2;
				if(chIdx2l > 0 && parents[chIdx2l].parentIdx == idx2)
					parents[chIdx2l].parentIdx = idx1;
				if(chIdx2r > 0 && parents[chIdx2r].parentIdx == idx2)
					parents[chIdx2r].parentIdx = idx1;
					
				//??
				if(parents[idx1].parentIdx == idx1)
					parents[idx1].parentIdx = idx2;
				if(parents[idx2].parentIdx == idx2)
					parents[idx2].parentIdx = idx1;
				
				//fix child info for swapped nodes
				int offset;
				offset = (parents[idx1].leftChild ? 0 : 1);	
				ptr[parents[idx1].parentIdx*16+12+offset] = idx1*64;

				offset = (parents[idx2].leftChild ? 0 : 1);	
				ptr[parents[idx2].parentIdx*16+12+offset] = idx2*64;
			}

			break;
		}
	case BVHLayout_Compact2:
		FW_ASSERT(false);
		break;
	default:
		break;
	}
}

//------------------------------------------------------------------------

void CudaBVH::createCompactBFS (const BVH& bvh)
{
	    struct StackEntry
    {
        const BVHNode*  node;
        S32             idx;

        StackEntry(const BVHNode* n = NULL, int i = 0) : node(n), idx(i) {}
    };

    // Construct data.

    Array<Vec4i> nodeData(NULL, 4);
    Array<Vec4i> triWoopData;
    Array<S32> triIndexData;
	std::queue<StackEntry> queue;
	queue.push(StackEntry(bvh.getRoot(), 0));
    //Array<StackEntry> stack(StackEntry(bvh.getRoot(), 0));

	while (!queue.empty())
    {
		StackEntry e = queue.front();
		queue.pop();
		//printf("%i %f | ", e.idx/4, e.node->getArea());
        FW_ASSERT(e.node->getNumChildNodes() == 2);
        const AABB* cbox[2];
        int cidx[2];

        // Process children.

        for (int i = 0; i < 2; i++)
        {
            // Inner node => push to stack.

            const BVHNode* child = e.node->getChildNode(i);
            cbox[i] = &child->m_bounds;
            if (!child->isLeaf())
            {
                cidx[i] = nodeData.getNumBytes() / 1;
				queue.push(StackEntry(child, nodeData.getSize()));
                nodeData.add(NULL, 4);
                continue;
            }

            // Leaf => append triangles.

            const LeafNode* leaf = reinterpret_cast<const LeafNode*>(child);
            cidx[i] = ~triWoopData.getSize();
            for (int j = leaf->m_lo; j < leaf->m_hi; j++)
            {
                woopifyTri(bvh, j);
                if (m_woop[0].x == 0.0f)
                    m_woop[0].x = 0.0f;
                triWoopData.add((Vec4i*)m_woop, 3);
                triIndexData.add(bvh.getTriIndices()[j]);
                triIndexData.add(0);
                triIndexData.add(0);
            }

            // Terminator.

            triWoopData.add(0x80000000);
            triIndexData.add(0);
        }

        const InnerNode *eN = reinterpret_cast<const InnerNode*>(e.node);
        const SplitInfo &splitInfo = eN->getSplitInfo();

        // Write entry.

        Vec4i* dst = nodeData.getPtr(e.idx);
        dst[0] = Vec4i(floatToBits(cbox[0]->min().x), floatToBits(cbox[0]->max().x), floatToBits(cbox[0]->min().y), floatToBits(cbox[0]->max().y));
        dst[1] = Vec4i(floatToBits(cbox[1]->min().x), floatToBits(cbox[1]->max().x), floatToBits(cbox[1]->min().y), floatToBits(cbox[1]->max().y));
        dst[2] = Vec4i(floatToBits(cbox[0]->min().z), floatToBits(cbox[0]->max().z), floatToBits(cbox[1]->min().z), floatToBits(cbox[1]->max().z));
        //dst[3] = Vec4i(cidx[0], cidx[1], 0, 0);
        dst[3] = Vec4i(cidx[0], cidx[1], splitInfo.getBitCode(), 0);
    }

    // Write to buffers.

    m_nodes.resizeDiscard(nodeData.getNumBytes());
    m_nodes.set(nodeData.getPtr(), nodeData.getNumBytes());

    m_triWoop.resizeDiscard(triWoopData.getNumBytes());
    m_triWoop.set(triWoopData.getPtr(), triWoopData.getNumBytes());

    m_triIndex.resizeDiscard(triIndexData.getNumBytes());
    m_triIndex.set(triIndexData.getPtr(), triIndexData.getNumBytes());

}


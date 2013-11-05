/*
 *  Copyright 2009-2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once
#include "bvh/BVH.hpp"
#include "cuda/CudaBVH.hpp"
#include "base/Timer.hpp"

//#define BVH_EPSILON 0.00001f
#define BVH_EPSILON 0.001f // PowerPlant
//#define BVH_EPSILON 0.01f

namespace FW
{
//------------------------------------------------------------------------

class SAHBVHBuilder
{
protected:
    enum
    {
        MaxDepth        = 64,
    };

    struct Reference
    {
        S32                 triIdx;
        AABB                bounds;

        Reference(void) : triIdx(-1) {}
    };

    struct NodeSpec
    {
        S32                 numRef;
        AABB                bounds;

        NodeSpec(void) : numRef(0) {}
    };

    struct ObjectSplit
    {
        F32                 sah;
        S32                 sortDim;
        S32                 numLeft;
        AABB                leftBounds;
        AABB                rightBounds;

        ObjectSplit(void) : sah(FW_F32_MAX), sortDim(0), numLeft(0) {}
    };

public:
                            SAHBVHBuilder				(BVH& bvh, const BVH::BuildParams& params);
    virtual                 ~SAHBVHBuilder			    (void);

    virtual BVHNode*        run							(void);

protected:
    //static int              sortCompare					(void* data, int idxA, int idxB);
	static bool             sortCompare					(void* data, int idxA, int idxB);
    static void             sortSwap					(void* data, int idxA, int idxB);

    BVHNode*                buildNode					(const NodeSpec& spec, int level, F32 progressStart, F32 progressEnd);
	BVHNode*                buildNode					(const NodeSpec& spec, int start, int end, int level, F32 progressStart, F32 progressEnd);
    BVHNode*                createLeaf					(const NodeSpec& spec);
	BVHNode*                createLeaf					(const NodeSpec& spec, int start, int end);

    ObjectSplit             findObjectSplit				(const NodeSpec& spec, F32 nodeSAH);
	ObjectSplit				findObjectSplit				(int start, int end, F32 nodeSAH);
    void                    performObjectSplit			(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split);
	void                    performObjectSplit			(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, int start, int end, const ObjectSplit& split);

private:
                            SAHBVHBuilder				(const SAHBVHBuilder&); // forbidden
    SAHBVHBuilder&          operator=					(const SAHBVHBuilder&); // forbidden

protected:
    BVH&                    m_bvh;
    const Platform&         m_platform;
    const BVH::BuildParams& m_params;

    Array<Reference>        m_refStack;
    Array<AABB>             m_rightBounds;
    S32                     m_sortDim;

    Timer                   m_progressTimer;
    S32                     m_numDuplicates;
};

//------------------------------------------------------------------------
}

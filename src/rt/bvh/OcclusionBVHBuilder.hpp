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
#include "SplitBVHBuilder.hpp"

namespace FW
{
//------------------------------------------------------------------------

class OcclusionBVHBuilder : public SplitBVHBuilder
{
protected:
	enum
    {
		MaxVisibleDepth = 48,
    };

    struct NodeSpecOcl : public NodeSpec
    {
		S32					numVisible;
		//AABB                boundsVisible;

        NodeSpecOcl(void) : NodeSpec(), numVisible(0) {}
    };

    struct ObjectSplitOcl : public ObjectSplit
    {
		S32					leftVisible;
		S32					rightVisible;
		//AABB                leftVisibleBounds;
        //AABB                rightVisibleBounds;
		bool                osahTested;
		bool                osahChosen;

        ObjectSplitOcl(void) : ObjectSplit(), leftVisible(0), rightVisible(0), osahTested(false), osahChosen(false) {}
    };

    struct SpatialSplitOcl : public SpatialSplit
    {
		//F32                 leftArea;
        //F32                 rightArea;
		S32					leftNum;
		S32					rightNum;
		S32					leftVisible;
		S32					rightVisible;
		bool                osahChosen;

        SpatialSplitOcl(void) : SpatialSplit(), leftNum(0), rightNum(0), leftVisible(0), rightVisible(0), osahChosen(false) {}
    };

    struct SpatialBinOcl : public SpatialBin
    {
		S32                 enterVisible;
		S32                 exitVisible;
    };

public:
                            OcclusionBVHBuilder			(BVH& bvh, const BVH::BuildParams& params);
    virtual                 ~OcclusionBVHBuilder		(void);

    virtual BVHNode*        run							(void);

protected:
	BVHNode*				buildNode					(const NodeSpecOcl& spec, int start, int end, int level, F32 progressStart, F32 progressEnd);

	ObjectSplitOcl			findObjectSplit				(const NodeSpecOcl& spec, int start, int end, F32 nodeSAH);
	ObjectSplitOcl			findObjectOccludeSplit		(const NodeSpecOcl& spec, int start, int end, F32 nodeSAH);
	void                    performObjectSplit			(NodeSpecOcl& left, NodeSpecOcl& right, const NodeSpecOcl& spec, int start, int end, const ObjectSplitOcl& split);

	SpatialSplitOcl         findSpatialSplit			(const NodeSpecOcl& spec, int start, int end, F32 nodeSAH);
	SpatialSplitOcl         findSpatialOccludeSplit		(const NodeSpecOcl& spec, int start, int end, F32 nodeSAH);
	//SpatialSplit            findVisibleSplit		 (const NodeSpecOcl& spec, int start, int end, F32 nodeSAH, int level);
	void                    performSpatialOccludeSplit	(NodeSpecOcl& left, NodeSpecOcl& right, int& start, int& end, const SpatialSplitOcl& split);

private:
                            OcclusionBVHBuilder			(const OcclusionBVHBuilder&); // forbidden
    OcclusionBVHBuilder&    operator=					(const OcclusionBVHBuilder&); // forbidden

protected:
	//Array<AABB>             m_rightVisibleBounds;
	SpatialBinOcl           m_bins[3][NumSpatialBins];
	Array<S32>              m_visibility;
	S32                     m_MaxVisibleDepth;
};

//------------------------------------------------------------------------
}

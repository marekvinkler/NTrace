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
                            OcclusionBVHBuilder			(BVH& bvh, const BVH::BuildParams& params, const Vec3f& cameraPosition);
    virtual                 ~OcclusionBVHBuilder		(void);

    virtual BVHNode*        run							(void);

protected:
	static bool				sortCompare					(void* data, int idxA, int idxB);
    static void             sortSwap					(void* data, int idxA, int idxB);

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
	Vec3f					m_cameraPos; // position of the camera
	Array<S32>              m_visibility;
	S32                     m_MaxVisibleDepth;
};

//------------------------------------------------------------------------
}

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

/**
 * \file
 * \brief Declarations for SAHBVHBuilder.
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

/**
 * \brief Class that builds a BVH using SAH.
 */
class SAHBVHBuilder
{
protected:

	/**
	 * \brief Maximum depth of the BVH.
	 */
    enum
    {
        MaxDepth        = 64,
    };

	/**
	 * \brief Structure holding triangle's index together with its bounding box.
	 */
    struct Reference
    {
        S32                 triIdx;				//!< Index of the triangle.
        AABB                bounds;				//!< Bounding box of the triangle.

		/**
		 * \brief Constructor.
		 */
        Reference(void) : triIdx(-1) {}
    };

	/**
	 * \brief Structure holding specifications of a BVH's node.
	 */
    struct NodeSpec
    {
        S32                 numRef;				//!< Number of the node's references saved in a node stack.
        AABB                bounds;				//!< Bounding box of the node.

		/**
		 * \brief Constructor.
		 */
        NodeSpec(void) : numRef(0) {}
    };

	/**
	 * \brief Structure holding info about a split of the BVH node.
	 */
    struct ObjectSplit
    {
        F32                 sah;				//!< SAH cost of the split.
        S32                 sortDim;			//!< Dimension in which triangles are sorted.
        S32                 numLeft;			//!< Number of triangles in left child node.
        AABB                leftBounds;			//!< AABB of a left child node.
        AABB                rightBounds;		//!< AABB of a right child node.

		/**
		 * \brief Constructor.
		 */
        ObjectSplit(void) : sah(FW_F32_MAX), sortDim(0), numLeft(0) {}
    };

public:
	/**
	 * \brief Constructor.
	 * \param[out] bvh BVH to be built.
	 * \param[in] params Build parameters.
	 */
                            SAHBVHBuilder				(BVH& bvh, const BVH::BuildParams& params);

	/**
	 * \brief Destructor.
	 */
    virtual                 ~SAHBVHBuilder			    (void);

	/**
	 * \brief Performs the actual build.
	 * \return Root node of the built BVH.
	 */
    virtual BVHNode*        run							(void);

protected:
    //static int              sortCompare					(void* data, int idxA, int idxB);

	/**
	 * \brief Sort comparator. Sorts references according to their position in descending order. For details see framework/base.Sort.hpp.
	 * \return True if idxB should go before idxA.
	 */
	static bool             sortCompare					(void* data, int idxA, int idxB);

	/**
	 * \brief Sort swap function. Swaps two references placed in the reference stack. For details see framework/base.Sort.hpp.
	 */
    static void             sortSwap					(void* data, int idxA, int idxB);

	/**
	 * \brief Builds a BVH node. The built node may be an inner node as well as a leaf node.
	 * \param[in] spec Specifications of the node.
	 * \param[in] level Level of the node in the tree.
	 * \param[in] proggressStart Percentage of already built subtree.
	 * \param[in] progressEnd Percentage of already built subtree including node's subtree.
	 * \return Built node.
	 */
    BVHNode*                buildNode					(const NodeSpec& spec, int level, F32 progressStart, F32 progressEnd);

	/**
	 * \brief Builds a node. The built node may be an inner node as well as leaf node. Processes specific part of the reference stack.
	 * \param[in] spec Specifications of the node.
	 * \param[in] start Processes reference stack section starting from this position.
	 * \param[in] end Processes reference stack section ending in this position.
	 * \param[in] level Level of the node in the tree.
	 * \param[in] proggressStart Percentage of already built subtree.
	 * \param[in] progressEnd Percentage of already built subtree including node's subtree.
	 * \return Built node.
	 */
	BVHNode*                buildNode					(const NodeSpec& spec, int start, int end, int level, F32 progressStart, F32 progressEnd);

	/**
	 * \brief Builds a leaf node.
	 * \param[in] spec Specification of the node.
	 * \return Built leaf node.
	 */
    BVHNode*                createLeaf					(const NodeSpec& spec);

	/**
	 * \brief Builds a leaf node. Processes specific part of the reference stack.
	 * \param[in] spec Specifications of the node.
	 * \param[in] start Processes reference stack section starting from this position.
	 * \param[in] end Processes reference stack section ending in this position.
	 * \return Built leaf node.
	 */
	BVHNode*                createLeaf					(const NodeSpec& spec, int start, int end);

	/** 
	 * \brief Finds the best object split of the node.
	 * \param[in] spec Specifications of the node.
	 * \param[in] nodeSAH Cost of the split without the cost of the triangles.
	 * \return Found split.
	 */
    ObjectSplit             findObjectSplit				(const NodeSpec& spec, F32 nodeSAH);

	/** 
	 * \brief Finds the best object split of the node.
	 * \param[in] start Processes reference stack section starting from this position.
	 * \param[in] end Processes reference stack section ending in this position.
	 * \param[in] nodeSAH Cost of the split without the cost of the triangles.
	 * \return Found split.
	 */
	ObjectSplit				findObjectSplit				(int start, int end, F32 nodeSAH);

	/** 
	 * \brief Performs the split operation.
	 * \param[out] left Left child node specification.
	 * \param[out] right Right child node specification.
	 * \param[in] spec Specification of the node being split.
	 * \param[in] split Split information.
	 */
    void                    performObjectSplit			(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split);

	/**
	 * \brief Performs the split operation.
	 * \param[out] left Left child node specification.
	 * \param[out] right Right child node specification.
	 * \param[in] start Processes reference stack section starting from this position.
	 * \param[in] end Processes reference stack section ending in this position.
	 * \param[in] split Split information.
	 */
	void                    performObjectSplit			(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, int start, int end, const ObjectSplit& split);

private:
                            SAHBVHBuilder				(const SAHBVHBuilder&); // forbidden
    SAHBVHBuilder&          operator=					(const SAHBVHBuilder&); // forbidden

protected:	
    BVH&                    m_bvh;						//!< BVH being built.
    const Platform&         m_platform;					//!< Platform settings.
    const BVH::BuildParams& m_params;					//!< Build parameters.

    Array<Reference>        m_refStack;					//!< Reference stack.
    Array<AABB>             m_rightBounds;				//!< Bounding boxes of all the possible right children.
    S32                     m_sortDim;					//!< Sort dimension. Used by the sort method.

    Timer                   m_progressTimer;			//!< Progress timer.
    S32                     m_numDuplicates;			//!< Number of duplicated references.
};

//------------------------------------------------------------------------
}

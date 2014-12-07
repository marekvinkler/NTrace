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
 * \brief Declarations for SAHBVHBuilder.
 */

#pragma once
#include "bvh/BVH.hpp"
#include "base/Timer.hpp"

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
        MaxDepth        = 64					//!< Maximum depth of the BVH tree.
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
    BVHNode*                buildNode					(NodeSpec& spec, int level, F32 progressStart, F32 progressEnd);

	/**
	 * \brief Builds a leaf node.
	 * \param[in] spec Specification of the node.
	 * \return Built leaf node.
	 */
    BVHNode*                createLeaf					(const NodeSpec& spec);

	/** 
	 * \brief Finds the best object split of the node.
	 * \param[in] spec Specifications of the node.
	 * \param[in] nodeSAH Cost of the split without the cost of the triangles.
	 * \return Found split.
	 */
    ObjectSplit             findObjectSplit				(const NodeSpec& spec, F32 nodeSAH);

	/** 
	 * \brief Performs the split operation.
	 * \param[out] left Left child node specification.
	 * \param[out] right Right child node specification.
	 * \param[in] spec Specification of the node being split.
	 * \param[in] split Split information.
	 */
    void                    performObjectSplit			(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split);

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

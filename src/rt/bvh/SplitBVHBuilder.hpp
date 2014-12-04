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
 * \brief Declarations for SplitBVHBuilder.
 */

#pragma once
#include "bvh/BVH.hpp"
#include "base/Timer.hpp"
#include "SAHBVHBuilder.hpp"

namespace FW
{
//------------------------------------------------------------------------

/**
 * \brief Class performing SBVH build.
 */
class SplitBVHBuilder : public SAHBVHBuilder
{
protected:

	/**
	 * \brief Several tree build constrains.
	 */
    enum
    {
        MaxSpatialDepth = 48,				//!< Maximum depth of the BVH where spatial split will still be used.
        NumSpatialBins  = 128,				//!< Number of spatial bins per node in each axis.
    };

	/**
	 * \brief Structure holding info about spatial split of a BVH node.
	 */
    struct SpatialSplit
    {
        F32                 sah;			//!< SAH cost of the split.
        S32                 dim;			//!< Dimension of the split.
        F32                 pos;			//!< Position of the split.

		/**
		 * \brief Constructor.
		 */
        SpatialSplit(void) : sah(FW_F32_MAX), dim(0), pos(0.0f) {}
    };

	/**
	 * \brief Structure holding info about a spatial bin.
	 */
    struct SpatialBin
    {
        AABB                bounds;				//!< Bounding box of the bin.
        S32                 enter;				//!< Number of triangles entering the bin.
        S32                 exit;				//!< Number of triangles leaving the bin.
    };

public:

	/**
	 * \brief Constructor.
	 * \param[out] bvh Emtpy BVH to be built.
	 * \param[in] params Build parameters.
	 */
                            SplitBVHBuilder     (BVH& bvh, const BVH::BuildParams& params);

	/**
	 * \brief Destructor.
	 */
     virtual                ~SplitBVHBuilder    (void);

	/**
	 * \brief Performs the actual build.
	 * \return Root node of the built BVH.
	 */
    virtual BVHNode*        run                 (void);

protected:

	/**
	 * \brief Builds a BVH node. The built node may be an inner node as well as a leaf node.
	 * \param[in] spec Specifications of the node.
	 * \param[in] level Level of the node in the tree.
	 * \param[in] proggressStart Percentage of already built subtree.
	 * \param[in] progressEnd Percentage of already built subtree including node's subtree.
	 * \return Built node.
	 */
    BVHNode*                buildNode           (NodeSpec spec, int level, F32 progressStart, F32 progressEnd);

	/** 
	 * \brief Finds the best spatial split of the node.
	 * \param[in] spec Specifications of the node.
	 * \param[in] nodeSAH Cost of the split without the cost of the triangles.
	 * \return Found split.
	 */
    SpatialSplit            findSpatialSplit    (const NodeSpec& spec, F32 nodeSAH);

	/** 
	 * \brief Performs the spatial split operation.
	 * \param[out] left Left child node specification.
	 * \param[out] right Right child node specification.
	 * \param[in] spec Specification of the node being split.
	 * \param[in] split Spatial split information.
	 */
    void                    performSpatialSplit (NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const SpatialSplit& split);

	/**
	 * \brief Splits the triangle's bounding box.
	 * \param[out] left Left reference.
	 * \param[out] right Right reference.
	 * \param[in] ref Reference being split.
	 * \param[in] dim Axis of the split.
	 * \param[in] pos Position of the split.
	 */
	void                    splitReference      (Reference& left, Reference& right, const Reference& ref, int dim, F32 pos);

private:
                            SplitBVHBuilder     (const SplitBVHBuilder&); // forbidden
    SplitBVHBuilder&        operator=           (const SplitBVHBuilder&); // forbidden

protected:
    F32                     m_minOverlap;				//!< Minimum overlap of the left and right AABB of the object split needed to make spatial split worth finding.
    SpatialBin              m_bins[3][NumSpatialBins];	//!< Spatial bins.
};

//------------------------------------------------------------------------
}

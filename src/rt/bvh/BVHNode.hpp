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
 * \brief Declarations for a BVH node.
 */

#pragma once
#include "base/Array.hpp"
#include "bvh/Platform.hpp"
#include "Util.hpp"

namespace FW
{

// TODO: remove m_probability. Node needed after all?

/**
 * \brief Available BVH stats.
 */
enum BVH_STAT
{
	BVH_STAT_MAX_DEPTH,
    BVH_STAT_NODE_COUNT,
    BVH_STAT_INNER_COUNT,
    BVH_STAT_LEAF_COUNT,
    BVH_STAT_TRIANGLE_COUNT,
    BVH_STAT_CHILDNODE_COUNT,
	BVH_STAT_OSAH_TESTED,
	BVH_STAT_OSAH_CHOSEN
};

/**
 * \brief Class holding information about a split of a BVH node.
 */
class SplitInfo
{
public:

	/**
	 * \brief Available split types.
	 */
	enum SplitType               { SAH, SBVH, OSAH };

	/**
	 * \brief Available split axes.
	 */
	enum SplitAxis               { SPLIT_X, SPLIT_Y, SPLIT_Z };

public:
	/**
	 * \brief Constructor.
	 */
	SplitInfo(): m_code(0) {}

	/**
	 * \brief Constructor.
	 * \param[in] axis Split axis.
	 * \param[in] splitType Type of the split.
	 * \param[in] osahTested Flag whether the split has been tested for a OSAH split.
	 */
	SplitInfo(S32 axis, SplitType splitType, bool osahTested): m_code(0) { m_code |= (osahTested << 31); m_code |= splitType << 2; m_code |= axis; }

	/**
	* \brief Constructor
	* \param[in] bitCode SplitInfo encoded in long int form.
	*/
	SplitInfo(unsigned long bitCode): m_code(bitCode) {}

	/**
	 * \return Whether the split has been tested for OSAH.
	 */
	bool           getOSAHTested() const           { return (m_code & 0x80000000) > 0; }

	/**
	 * \return Whether the split has been chosen for OSAH.
	 */
	bool           getOSAHChosen() const           { return getType() == OSAH; }

	/**
	 * \return Type of the split.
	 */
	SplitType      getType() const                 { return (SplitType)((m_code & 0xC) >> 2); }

	/**
	 * \return Name of the split type.
	 */
	String         getTypeName() const             { return getOSAHTested() ? String(m_typeNames[getType()]) +  "+" : String(m_typeNames[getType()]); }

	/**
	 * \return Split axis.
	 */
	SplitAxis      getAxis() const                 { return (SplitAxis)(m_code & 0x3); }

	/**
	 * \return Name of the split axis.
	 */
	String         getAxisName() const             { return m_axisNames[getAxis()]; }

	/**
	 * \return SplitInfo encoded in an unsigned long integer.
	 */
	unsigned long  getBitCode() const              { return m_code; }

private:
	unsigned long m_code;					//!< Split info data.
	static char* m_axisNames[3];			//!< Axis names.
	static char* m_typeNames[3];			//!< Split type names.
};

/**
 * \brief BVH virtual node. Parent class of both a leaf node and an inner node.
 */
class BVHNode
{
public:

	/** 
	 * \brief Constructor.
	 */
    BVHNode() : m_probability(1.f),m_parentProbability(1.f),m_treelet(-1),m_index(-1) {}
	virtual ~BVHNode() {}

	/**
	 * \return Whether the node is a leaf node.
	 */
    virtual bool        isLeaf() const = 0;

	/**
	 * \return Number of the node's child nodes.
	 */
    virtual S32         getNumChildNodes() const = 0;

	/**
	 * \brief Returns one of the node's child nodes.
	 * \param[in] i Index of the child node.
	 * \return Selected child node.
	 */
    virtual BVHNode*    getChildNode(S32 i) const   = 0;

	/**
	 * \return Number of node's triangles.
	 */
    virtual S32         getNumTriangles() const { return 0; }

	/**
	 * \return Surface area of the node.
	 */
    float       getArea() const     { return m_bounds.area(); }

    AABB        m_bounds;				//!< Bounding box of the node.

    // These are somewhat experimental, for some specific test and may be invalid...
    float       m_probability;          //!< Probability of coming here (widebvh uses this).
    float       m_parentProbability;    //!< Probability of coming to parent (widebvh uses this).

    int         m_treelet;              //!< For queuing tests (qmachine uses this).
    int         m_index;                //!< in linearized tree (qmachine uses this).

    // Subtree functions

	/**
	 * \brief Calculates various information about the node's subtree.
	 * \param[in] stat Desired information.
	 * \return Calculated information.
	 */
    int     getSubtreeSize(BVH_STAT stat=BVH_STAT_NODE_COUNT) const;

	/**
	 * \brief Calculates node's subtree probabilities and also sah price.
	 * \param[in] p Platform settings.
	 * \param[in] parentProbability Parent node probability.
	 * \param[out] sah Calculated SAH cost.
	 */
    void    computeSubtreeProbabilities(const Platform& p, float parentProbability, float& sah);

	/**
	 * \brief Calculates subtree SAH cost. Requires calculated probabilities.
	 * \param[in] p Platform settings.
	 * \return SAH cost of the node's subtree..
	 */
    float   computeSubtreeSAHCost(const Platform& p) const;     // NOTE: assumes valid probabilities

	/**
	 * \brief Deletes node's subtree.
	 */
    void    deleteSubtree();

	/**
	 * \brief Assigns node's sbutree indices in depth first order.
	 * \param[in] index Index to be assigned to this node.
	 * \pararm[in] includeLeafNodes Flag whether to assign indices to leaf nodes as well.
	 */
    void    assignIndicesDepthFirst  (S32 index=0, bool includeLeafNodes=true);

	/**
	 * \brief Assigns node's subtree indices in breadth first order.
	 * \param[in] index Index to be assigned to the this node.
	 * \param[in] includeLeafNodes Flag whether to assign indices to leaf nodes as well.
	 */
    void    assignIndicesBreadthFirst(S32 index=0, bool includeLeafNodes=true);
};

/**
 * \brief BVH inner node.
 */
class InnerNode : public BVHNode
{
public:

	/**
	 * \brief Constructor.
	 * \param[in] bounds Node's bounding box.
	 * \param[in] child0 Left child node.
	 * \param[in] child1 Right child node.
	 */
    InnerNode(const AABB& bounds, BVHNode* child0, BVHNode* child1) { m_bounds=bounds; m_children[0] = child0; m_children[1] = child1; };
    
	/**
	 * \brief Constructor.
	 * \param[in] bounds Node's bounding box.
	 * \param[in] child0 Left child node.
	 * \param[in] child1 Right child node.
	 * \param[in] axis Axis of the node's split.
	 * \param[in] splitType Type of the node's split.
	 * \param[in] osahTested Flag whether the split was tested for OSAH.
	 */
	InnerNode(const AABB& bounds, BVHNode* child0, BVHNode* child1, S32 axis, SplitInfo::SplitType splitType, bool osahTested): m_splitInfo(axis, splitType, osahTested) { m_bounds=bounds; m_children[0] = child0; m_children[1] = child1; };
	virtual ~InnerNode() {}

	/**
	 * \return Whether the node is a leaf node (always false).
	 */
    bool        isLeaf() const                  { return false; }

	/**
	 * \return Number of the node's child nodes.
	 */
    S32         getNumChildNodes() const        { return 2; }

	/**
	 * \brief Returns one of the node's child nodes.
	 * \param[in] i Index of the child node.
	 * \return Node's selected child node.
	 */
    BVHNode*    getChildNode(S32 i) const       { FW_ASSERT(i>=0 && i<2); return m_children[i]; }

	/**
	 * \brief Returns the split info.
	 * \return Information about the node's split.
	 */
	const SplitInfo& getSplitInfo() const       { return m_splitInfo; }

    BVHNode*    m_children[2];					//!< Child nodes.
    SplitInfo   m_splitInfo;					//!< Split info.
};

/**
 * \brief BVH leaf node.
 */
class LeafNode : public BVHNode
{
public:

	/**
	 * \brief Constructor.
	 * \param[in] bounds Node's bounding box.
	 * \param[in] lo Lower index to the BVH's triangle index array.
	 * \param[in] hi Higher index to the BVH's triangle index array.
	 */
    LeafNode(const AABB& bounds,int lo,int hi)  { m_bounds=bounds; m_lo=lo; m_hi=hi; }

	/**
	 * \brief Copy constructor.
	 * \param[in] s Leaf node to copy.
	 */
    LeafNode(const LeafNode& s)                 { *this = s; }
	virtual ~LeafNode() {}

	/**
	 * \return Whether the node is a leaf node (always true).
	 */
    bool        isLeaf() const                  { return true; }

	/**
	 * \return Number of the node's child nodes.
	 */
    S32         getNumChildNodes() const        { return 0; }

	/**
	 * \brief Returns one of the node's child nodes.
	 * \param[in] i Index of the child node.
	 * \return Node's selected child node.
	 */
    BVHNode*    getChildNode(S32) const         { return NULL; }

	/**
	 * \return Number of node's triangles.
	 */
    S32         getNumTriangles() const         { return m_hi-m_lo; }
    S32         m_lo;							//!< Lower index to the BVH's triangle index array.
    S32         m_hi;							//!< Higher index to the BVH's triangle index array.
};

} //

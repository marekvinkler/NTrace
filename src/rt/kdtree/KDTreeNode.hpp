/*
*  Copyright (c) 2013, Radek Stibora
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of the <organization> nor the
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
#include "bvh/Platform.hpp"
#include "Util.hpp"

namespace FW
{

/**
 * \brief Available statistics. Used in getSubtreeSize.
 */
enum KDTREE_STAT
{
    KDTREE_STAT_NODE_COUNT,
    KDTREE_STAT_INNER_COUNT,
    KDTREE_STAT_LEAF_COUNT,
    KDTREE_STAT_TRIANGLE_COUNT,
    KDTREE_STAT_CHILDNODE_COUNT,
	KDTREE_STAT_EMPTYLEAF_COUNT
};

/**
 * \brief K-d tree virtual parent node class.
 */
class KDTreeNode
{
public:
	/**
	 * \brief Returns whether the node is a leaf node.
	 * \return True if the node is a leaf node, false if it is an inner node.
	 */
	virtual bool		isLeaf() const = 0;

	/**
	 * \brief Returns number of the node's child nodes.
	 * \return 2 if the node is an inner node, 0 if it is a leaf node.
	 */
	virtual S32			getNumChildNodes() const = 0;

	/**
	 * \brief Returns node's child node (left or right).
	 * \param[in] i Which child to get. 0 = left, 1 = right.
	 * \return Node's child node (left or right) or NULL if the node has not that child.
	 */
	virtual KDTreeNode* getChildNode(S32 i) const = 0;

	/**
	 * \brief Returns number of triangles this node references. Only leaf nodes will return non-zero values.
	 * \return Number of triangles this node references. 
	 */
	virtual S32			getNumTriangles() const { return 0; }

	/**
	 * \brief Deletes node's subtree.
	 */
	void				deleteSubtree();

	/**
	 * \brief Computes given statistics about node's subtree.
	 * \param[in] stat Desired statistics.
	 */
	int					getSubtreeSize(KDTREE_STAT stat = KDTREE_STAT_NODE_COUNT) const;
};

/**
 * \brief K-d tree's inner node class.
 */
class KDTInnerNode : public KDTreeNode
{
public:
	/**
	 * \brief Constructor.
	 * \param[in] split Split position.
	 * \param[in] axis Split axis.
	 * \param[in] child0 Left child.
	 * \param[in] child1 Right child.
	 */
	KDTInnerNode(F32 split, S32 axis, KDTreeNode* child0, KDTreeNode* child1) 
		{ m_pos = split, m_axis = axis, m_children[0] = child0; m_children[1] = child1; }

	/**
	 * \brief Returns whether the node is a leaf node.
	 * \return Always false.
	 */
	bool			isLeaf() const				{ return false; }

	/**
	 * \brief Returns number of the node's child nodes.
	 * \return Always 2.
	 */
	S32				getNumChildNodes() const	{ return 2; }

	/**
	 * \brief Returns node's child node (left or right).
	 * \param[in] i Which child to gte. 0 = left, 1 = right.
	 * \return Node's child node (left or right) or NULL if the node has not that child.
	 */
	KDTreeNode*		getChildNode(S32 i) const	{ FW_ASSERT(i>=0 && i<2); return m_children[i]; }

	KDTreeNode*		m_children[2];	//!< Node's child nodes.	
	F32				m_pos;		//!< Split position.
	S32				m_axis;			//!< Split dimension.
};

/**
 * \brief K-d tree's leaf node class.
 */
class KDTLeafNode : public KDTreeNode
{
public:
	/**
	 * \brief Constructor.
	 * \param[in] lo Lower index to the tree's triangle references array.
	 * \param[in] hi Higher index to the tree's triangle references array.
	 */
	KDTLeafNode(int lo, int hi)					{ m_lo = lo; m_hi=hi; }

	/**
	 * \brief Returns whether the node is a leaf node.
	 * \return Always true.
	 */
	bool		isLeaf() const					{ return true; }

	/**
	 * \brief Returns number of the node's child nodes.
	 * \return Always 0.
	 */
	S32         getNumChildNodes() const        { return 0; }

	/**
	 * \brief Returns node's child node (left or right).
	 * \param[in] i Which child to get. 0 = left, 1 = right.
	 * \return Always NULL.
	 */
	KDTreeNode*	getChildNode(S32) const         { return NULL; }

	/**
	 * \brief Returns number of triangles this node references.
	 * \return Number of triangles this node references. 
	 */
	S32			getNumTriangles() const			{ return m_hi - m_lo; }

	S32			m_lo;			//!< Lower index to the tree's triangle references array.
	S32			m_hi;			//!< Higher index to the tree's triangle references array.
};


}
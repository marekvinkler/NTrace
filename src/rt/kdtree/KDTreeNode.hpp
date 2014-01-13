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

enum KDTREE_STAT
{
    KDTREE_STAT_NODE_COUNT,
    KDTREE_STAT_INNER_COUNT,
    KDTREE_STAT_LEAF_COUNT,
    KDTREE_STAT_TRIANGLE_COUNT,
    KDTREE_STAT_CHILDNODE_COUNT,
	KDTREE_STAT_EMPTYLEAF_COUNT
};

class KDTreeNode
{
public:
	KDTreeNode() {}
	virtual bool		isLeaf() const = 0;
	virtual S32			getNumChildNodes() const = 0;
	virtual KDTreeNode* getChildNode(S32 i) const = 0;
	virtual S32			getNumTriangles() const { return 0; }

	void				deleteSubtree();
	int					getSubtreeSize(KDTREE_STAT stat = KDTREE_STAT_NODE_COUNT) const;
};


class KDTInnerNode : public KDTreeNode
{
public:
	KDTInnerNode(F32 split, S32 axis, KDTreeNode* child0, KDTreeNode* child1) 
		{ m_split = split, m_axis = axis, m_children[0] = child0; m_children[1] = child1; }

	bool			isLeaf() const				{ return false; }
	S32				getNumChildNodes() const	{ return 2; }
	KDTreeNode*		getChildNode(S32 i) const	{ FW_ASSERT(i>=0 && i<2); return m_children[i]; }
	
	KDTreeNode*		m_children[2];
	
	F32				m_split;
	S32				m_axis;
};


class KDTLeafNode : public KDTreeNode
{
public:
	KDTLeafNode(int lo, int hi)					{ m_lo = lo; m_hi=hi; }

	bool		isLeaf() const					{ return true; }
	S32         getNumChildNodes() const        { return 0; }
	KDTreeNode*	getChildNode(S32) const         { return NULL; }

	S32			getNumTriangles() const			{ return m_hi - m_lo; }
	S32			m_lo;
	S32			m_hi;
};


}
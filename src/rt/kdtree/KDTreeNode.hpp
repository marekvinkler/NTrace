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
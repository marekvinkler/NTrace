#include "KDTreeNode.hpp"

namespace FW
{


void KDTreeNode::deleteSubtree()
{
    for(int i=0;i<getNumChildNodes();i++)
        getChildNode(i)->deleteSubtree();

    delete this;
}


int KDTreeNode::getSubtreeSize(KDTREE_STAT stat) const
{
	int cnt;
    switch(stat)
    {
        default: FW_ASSERT(0);  // unknown mode
        case KDTREE_STAT_NODE_COUNT:      cnt = 1; break;
        case KDTREE_STAT_LEAF_COUNT:      cnt = isLeaf() ? 1 : 0; break;
        case KDTREE_STAT_INNER_COUNT:     cnt = isLeaf() ? 0 : 1; break;
        case KDTREE_STAT_TRIANGLE_COUNT:  cnt = isLeaf() ? reinterpret_cast<const KDTLeafNode*>(this)->getNumTriangles() : 0; break;
        case KDTREE_STAT_CHILDNODE_COUNT: cnt = getNumChildNodes(); break;
		case KDTREE_STAT_EMPTYLEAF_COUNT: cnt = isLeaf() ? ((reinterpret_cast<const KDTLeafNode*>(this)->getNumTriangles() == 0) ? 1 : 0) : 0; break;
    }

    if(!isLeaf())
    {
        for(int i=0;i<getNumChildNodes();i++)
            cnt += getChildNode(i)->getSubtreeSize(stat);
    }

    return cnt;
}


}
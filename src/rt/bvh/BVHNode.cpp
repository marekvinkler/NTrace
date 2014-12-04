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

#include "bvh/BVHNode.hpp"

namespace FW
{

char* SplitInfo::m_axisNames[3] = { "x", "y", "z" };
char* SplitInfo::m_typeNames[3] = { "SAH", "SBVH", "OSAH" };

int BVHNode::getSubtreeSize(BVH_STAT stat) const
{
    int cnt;
    switch(stat)
    {
        default: FW_ASSERT(0);  // unknown mode
		case BVH_STAT_MAX_DEPTH:
        case BVH_STAT_NODE_COUNT:      cnt = 1; break;
        case BVH_STAT_LEAF_COUNT:      cnt = isLeaf() ? 1 : 0; break;
        case BVH_STAT_INNER_COUNT:     cnt = isLeaf() ? 0 : 1; break;
        case BVH_STAT_TRIANGLE_COUNT:  cnt = isLeaf() ? reinterpret_cast<const LeafNode*>(this)->getNumTriangles() : 0; break;
        case BVH_STAT_CHILDNODE_COUNT: cnt = getNumChildNodes(); break;
		case BVH_STAT_OSAH_TESTED:  cnt = isLeaf() ? 0 : reinterpret_cast<const InnerNode*>(this)->getSplitInfo().getOSAHTested(); break;
		case BVH_STAT_OSAH_CHOSEN:  cnt = isLeaf() ? 0 : reinterpret_cast<const InnerNode*>(this)->getSplitInfo().getOSAHChosen(); break;
    }

    if(!isLeaf())
    {
		if(stat != BVH_STAT_MAX_DEPTH)
		{
			for(int i=0;i<getNumChildNodes();i++)
				cnt += getChildNode(i)->getSubtreeSize(stat);
		}
		else
		{
			for(int i=0;i<getNumChildNodes();i++)
				cnt = max(cnt, getChildNode(i)->getSubtreeSize(stat)+1);
		}
    }

    return cnt;
}


void BVHNode::deleteSubtree()
{
    for(int i=0;i<getNumChildNodes();i++)
        getChildNode(i)->deleteSubtree();

    delete this;
}


void BVHNode::computeSubtreeProbabilities(const Platform& p,float probability, float& sah)
{
    sah += probability * p.getCost(this->getNumChildNodes(),this->getNumTriangles());

    m_probability = probability;

    for(int i=0;i<getNumChildNodes();i++)
    {
        BVHNode* child = getChildNode(i);
        child->m_parentProbability = probability;
        float childProbability = 0.0f;
        if (probability > 0.0f)
            childProbability = probability * child->m_bounds.area()/this->m_bounds.area();
        child->computeSubtreeProbabilities(p, childProbability, sah );
    }
}


// TODO: requires valid probabilities...
float BVHNode::computeSubtreeSAHCost(const Platform& p) const
{
    float SAH = m_probability * p.getCost( getNumChildNodes(),getNumTriangles());

    for(int i=0;i<getNumChildNodes();i++)
        SAH += getChildNode(i)->computeSubtreeSAHCost(p);

    return SAH;
}

//-------------------------------------------------------------

void assignIndicesDepthFirstRecursive( BVHNode* node, S32& index, bool includeLeafNodes )
{
    if(node->isLeaf() && !includeLeafNodes)
        return;

    node->m_index = index++;
    for(int i=0;i<node->getNumChildNodes();i++)
        assignIndicesDepthFirstRecursive(node->getChildNode(i), index, includeLeafNodes);
}

void BVHNode::assignIndicesDepthFirst( S32 index, bool includeLeafNodes )
{
    assignIndicesDepthFirstRecursive( this, index, includeLeafNodes );
}

//-------------------------------------------------------------

void BVHNode::assignIndicesBreadthFirst( S32 index, bool includeLeafNodes )
{
    Array<BVHNode*> nodes;
    nodes.add(this);
    S32 head=0;

    while(head < nodes.getSize())
    {
        // pop
        BVHNode* node = nodes[head++];

        // discard
        if(node->isLeaf() && !includeLeafNodes)
            continue;

        // assign
        node->m_index = index++;

        // push children
        for(int i=0;i<node->getNumChildNodes();i++)
            nodes.add(node->getChildNode(i));
    }
}


} //

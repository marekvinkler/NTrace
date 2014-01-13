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

#include "KDTree.hpp"
#include "NaiveKDTreeBuilder.hpp"
#include "FastKDTreeBuilder.hpp"

namespace FW
{

KDTree::KDTree(Scene* scene, const Platform& platform, const BuildParams& params) : AccelerationStructure(scene, platform)
{
	S32 numDuplicates = 0;

	if (params.builder == SpatialMedian || params.builder == ObjectMedian)
	{
		NaiveKDTreeBuilder builder(*this, params);
		m_root = builder.run();
		numDuplicates = builder.getNumDuplicates();
	}
	else if (params.builder == SAH)
	{
		FastKDTreeBuilder builder(*this, params);
		m_root = builder.run();
		numDuplicates = builder.getNumDuplicates();
	}
	else
		FW_ASSERT(0);

	if(params.stats)
    {
        params.stats->numLeafNodes      = m_root->getSubtreeSize(KDTREE_STAT_LEAF_COUNT);
        params.stats->numInnerNodes     = m_root->getSubtreeSize(KDTREE_STAT_INNER_COUNT);
        params.stats->numTris           = m_root->getSubtreeSize(KDTREE_STAT_TRIANGLE_COUNT);
        params.stats->numChildNodes     = m_root->getSubtreeSize(KDTREE_STAT_CHILDNODE_COUNT);
		params.stats->numEmptyLeaves	= m_root->getSubtreeSize(KDTREE_STAT_EMPTYLEAF_COUNT);
		params.stats->percentDuplicates	= (float)numDuplicates / m_scene->getNumTriangles() * 100;
    }
}

}
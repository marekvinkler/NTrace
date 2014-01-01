
#include "KDTree.hpp"
#include "NaiveKDTreeBuilder.hpp"
#include "FastKDTreeBuilder.hpp"

namespace FW
{

KDTree::KDTree(Scene* scene, const Platform& platform, const BuildParams& params) : m_scene(scene), m_platform(platform)
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
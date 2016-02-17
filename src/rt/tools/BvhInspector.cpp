#include "BvhInspector.h"

void BvhInspector::inspect(BVH::Stats& stats)
{
	inspectRecursive(0, 1.f, 0, stats);
}

void BvhInspector::inspectRecursive(S32 node, float probability, int depth, BVH::Stats& stats)
{
	if(node < 0)
	{
		FW::Array<int> tris;
		m_bvh->getTriangleIndices(node, tris);
		stats.SAHCost += probability * getCost(0,tris.getSize());
		stats.numLeafNodes += 1;
		stats.numTris += tris.getSize();
		if(stats.maxDepth < depth)
			stats.maxDepth = depth;
	}
	else
	{
		SplitInfo splitInfo;
		AABB leftBox, rightBox;
		S32 leftAddr, rightAddr;

		m_bvh->getNode(node, &splitInfo, leftBox, rightBox, leftAddr, rightAddr);

		stats.SAHCost += probability * getCost(2, 0);
		stats.numInnerNodes += 1;

		float probabilityLeft = leftBox.area() / (leftBox + rightBox).area();
		float probabilityRight = rightBox.area() / (leftBox + rightBox).area();

		inspectRecursive(leftAddr, probabilityLeft, depth+1, stats);
		inspectRecursive(rightAddr, probabilityRight, depth+1, stats);
	}
}

float BvhInspector::getCost(int numChildNodes, int numTriangles)
{
	return numChildNodes + numTriangles;
}
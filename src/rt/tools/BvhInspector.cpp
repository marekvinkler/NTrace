#include "BvhInspector.h"

void BvhInspector::inspect(BVH::Stats& stats)
{
	inspectRecursive(0, 1.f, 0, stats);
}

void BvhInspector::computeSubtreeProbabilities(S32 node, const AABB& box, const Platform& p,float probability, float& sah)
{
	S32 numChildNodes = 0;
	S32  numTriangles = 0;

	if(node < 0)
	{

		FW::Array<int> tris;
		m_bvh->getTriangleIndices(node, tris);
		numTriangles = tris.getSize();
	}
	else
	{
		numChildNodes = 2;

		SplitInfo splitInfo;
		AABB leftBox, rightBox;
		S32 leftAddr, rightAddr;

		m_bvh->getNode(node, &splitInfo, leftBox, rightBox, leftAddr, rightAddr);

		F32 leftProb = probability * leftBox.area()/box.area();
		computeSubtreeProbabilities(leftAddr, leftBox, p, leftProb, sah);

		F32 rightProb = probability * rightBox.area()/box.area();
		computeSubtreeProbabilities(rightAddr, rightBox, p, rightProb, sah);
	}

    sah += probability * p.getCost(numChildNodes,numTriangles);
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
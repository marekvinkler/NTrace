#include "..\cuda\CudaBVH.hpp"
#include "..\bvh\BVH.hpp"

using namespace FW;

class BvhInspector
{
public:
	BvhInspector(CudaBVH* bvh) : m_bvh(bvh) {}
	void BvhInspector::computeSubtreeProbabilities(S32 node, const AABB& box, const Platform& p,float probability, float& sah);
	void inspect(BVH::Stats& stats);

private:
	CudaBVH * m_bvh;
	void inspectRecursive(S32 node, float probability, int depth, BVH::Stats& stats);
	float getCost(int numChildNodes, int numTriangles);
};
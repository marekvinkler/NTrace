#include "..\cuda\CudaBVH.hpp"
#include "..\bvh\BVH.hpp"

using namespace FW;

class BvhInspector
{
public:
	BvhInspector(CudaBVH* bvh) : m_bvh(bvh) {}
	void inspect(BVH::Stats& stats);

private:
	CudaBVH * m_bvh;
	void inspectRecursive(S32 node, float probability, int depth, BVH::Stats& stats);
	float getCost(int numChildNodes, int numTriangles);
};
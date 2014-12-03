
#include "InterpolationKernels.hpp"

using namespace FW;

//------------------------------------------------------------------------

extern "C" __global__ void interpolateVertices(void)
{
	const VertInterpolationInput& in = c_VertInterpolationInput;
   
	const Vec3f*		positionsA		= (const Vec3f*) in.posA;
	const Vec3f*		positionsB		= (const Vec3f*) in.posB;
	const Vec3f*		normalsA		= (const Vec3f*) in.normalA;
	const Vec3f*		normalsB		= (const Vec3f*) in.normalB;
	const float			weight			= (const float) in.weight;
	const int			vertCount		= (const int) in.vertCount;
	Vec3f*				positionsI		= (Vec3f*) in.posIntrp;
	Vec3f*				normalsI		= (Vec3f*) in.normalIntrp;

	const int taskIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (taskIdx >= in.vertCount)
        return;

	positionsI[taskIdx] = positionsA[taskIdx] + weight * (positionsB[taskIdx] - positionsA[taskIdx]);

	normalsI[taskIdx] = ((1.f - weight) * normalsA[taskIdx] + weight * normalsB[taskIdx]);
	normalsI[taskIdx].normalize();
}

extern "C" __global__ void interpolateTriangles(void)
{
	const TriInterpolationInput& in = c_TriInterpolationInput;

	const Vec3i*		triIdx			= (const Vec3i*) in.triIdx;
	const Vec3f*		vertPos			= (const Vec3f*) in.vertPos;
	const int			triCount		= (const int) in.triCount;
	Vec3f*				triNormal		= (Vec3f*) in.triNormal;

	const int taskIdx = threadIdx.x + blockDim.x * blockIdx.x;
	if(taskIdx >= in.triCount)
		return;

	triNormal[taskIdx] = normalize(cross(vertPos[triIdx[taskIdx].y] - vertPos[triIdx[taskIdx].x], vertPos[triIdx[taskIdx].z] - vertPos[triIdx[taskIdx].y]));
}
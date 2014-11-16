
#include "InterpolationKernels.hpp"
#include "../../framework/base/Math.hpp"

using namespace FW;

//------------------------------------------------------------------------

extern "C" __global__ void interpolateVertices(void)
{
	const InterpolationInput& in = c_InterpolationInput;
   
	const Vec3f*		verticesA		= (const Vec3f*) in.verticesA;
	const Vec3f*		verticesB		= (const Vec3f*) in.verticesB;
	const float			weight			= (const float) in.weight;
	const int			vertCount		= (const int) in.vertCount;
 	Vec3f*				verticesI		= (Vec3f*) in.verticesIntrp;

	const int taskIdx = threadIdx.x + blockDim.x * blockIdx.x;
    if (taskIdx >= in.vertCount)
        return;

	verticesI[taskIdx] = verticesA[taskIdx] + weight * (verticesB[taskIdx] - verticesA[taskIdx]);
}

#pragma once
#include "../Util.hpp"
#include "../../framework/base/DLLImports.hpp"

namespace FW
{

struct InterpolationInput
{
	float			weight;
	int				vertCount;
	CUdeviceptr		verticesA;
	CUdeviceptr		verticesB;
	CUdeviceptr		verticesIntrp;
};

//------------------------------------------------------------------------

#if FW_CUDA
extern "C"
{
	__constant__ InterpolationInput c_InterpolationInput;

}
#endif

};
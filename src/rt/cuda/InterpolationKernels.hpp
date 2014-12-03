
#pragma once
#include "Util.hpp"
#include "base/Math.hpp"
#include "base/DLLImports.hpp"

namespace FW
{

struct VertInterpolationInput
{
	float			weight;
	int				vertCount;
	CUdeviceptr		posA;
	CUdeviceptr		posB;
	CUdeviceptr		posIntrp;
	CUdeviceptr		normalA;
	CUdeviceptr		normalB;
	CUdeviceptr		normalIntrp;
};

struct TriInterpolationInput
{
	int				triCount;
	CUdeviceptr		triIdx;
	CUdeviceptr		vertPos;
	CUdeviceptr		triNormal;
};

//------------------------------------------------------------------------

#if FW_CUDA
extern "C"
{
	__constant__ TriInterpolationInput c_TriInterpolationInput;
	__constant__ VertInterpolationInput c_VertInterpolationInput;

}
#endif

};
#pragma once

//#include "base/Defs.hpp"
//#include "../base/Buffer.hpp"
//#include "gpu/Buffer.hpp"
//#include "../../framework/base/Defs.hpp"

#include <cuda.h>

//namespace FW {

float radixSortCuda(CUdeviceptr keys, CUdeviceptr values, int n);

float createClusters(CUdeviceptr values, int n, int d, CUdeviceptr out, int &out_cnt);

//};
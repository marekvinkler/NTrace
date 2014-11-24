/*
 *  Copyright 2009-2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#ifdef __CUDACC__
#include <stdio.h>

typedef unsigned int uint;
typedef unsigned long long int ullint;
#endif

// Enum for various locking states
enum AllocatorLockType {AllocatorLockType_Free = 0, AllocatorLockType_Set};

// A structure holding information about dynamic memory heap
struct AllocInfo
{
	unsigned int heapSize;
	unsigned int payload;
	double maxFrag;
	double chunkRatio;
};

//------------------------------------------------------------------------
// Debugging tests
//------------------------------------------------------------------------

#define CHECK_OUT_OF_MEMORY // For unknown reason cannot be used together with WRITE_ALL_TEST

//------------------------------------------------------------------------
// CircularMalloc
//------------------------------------------------------------------------

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
#define CIRCULAR_MALLOC_HEADER_SIZE (2*sizeof(unsigned int))
#define CIRCULAR_MALLOC_NEXT_OFS sizeof(unsigned int)
#else
#define CIRCULAR_MALLOC_HEADER_SIZE (4*sizeof(unsigned int))
#define CIRCULAR_MALLOC_PREV_OFS sizeof(unsigned int)
#define CIRCULAR_MALLOC_NEXT_OFS (2*sizeof(unsigned int))
#endif

//------------------------------------------------------------------------
// ScatterAlloc
//------------------------------------------------------------------------

#ifdef __CUDACC__
//set the template arguments using HEAPARGS
// pagesize ... byter per page
// accessblocks ... number of superblocks
// regionsize ... number of regions for meta data structur
// wastefactor ... how much memory can be wasted per alloc (multiplicative factor)
// use_coalescing ... combine memory requests of within each warp
// resetfreedpages ... allow pages to be reused with a different size
#define HEAPARGS SCATTER_ALLOC_PAGESIZE, SCATTER_ALLOC_ACCESSBLOCKS, SCATTER_ALLOC_REGIONSIZE, SCATTER_ALLOC_WASTEFACTOR, SCATTER_ALLOC_COALESCING, SCATTER_ALLOC_RESETPAGES
//include the scatter alloc heap
#include "ScatterAlloc/heap_impl.cuh"
#include "ScatterAlloc/utils.h"

template __global__ void GPUTools::initHeap<HEAPARGS>(DeviceHeap<HEAPARGS>* heap, void* heapmem, uint memsize);

//------------------------------------------------------------------------
// FDGMalloc
//------------------------------------------------------------------------

//------------------------------------------------------------------------

// Heap data
__device__ char* g_heapBase; // The base pointer to the heap
__device__ uint g_heapOffset; // Current location in the heap
__device__ uint* g_heapMultiOffset; // Current location in the heap for each multiprocessor
__device__ uint g_numSM; // Number of SMs on the device
__device__ uint g_heapLock; // Lock for updating the heap

__constant__ AllocInfo c_alloc;

__device__ __forceinline__ void* mallocCudaMalloc(uint allocSize);
__device__ __forceinline__ void freeCudaMalloc(void* ptr);

__device__ __forceinline__ void* mallocAtomicMalloc(uint allocSize);

__device__ __forceinline__ void* mallocAtomicMallocCircular(uint allocSize);

__device__ __forceinline__ void* mallocCircularMalloc(uint allocSize);
__device__ __forceinline__ void freeCircularMalloc(void* ptr);

__device__ __forceinline__ void* mallocCircularMallocFused(uint allocSize);
__device__ __forceinline__ void freeCircularMallocFused(void* ptr);

__device__ __forceinline__ void* mallocScatterAlloc(uint allocSize);
__device__ __forceinline__ void freeScatterAlloc(void* ptr);

extern "C" __global__ void CircularMallocPrepare1(uint numChunks);
extern "C" __global__ void CircularMallocPrepare2(uint numChunks);
extern "C" __global__ void CircularMallocPrepare3(uint numChunks, uint rootChunk);

extern "C" __global__ void CircularMallocFusedPrepare1(uint numChunks);
extern "C" __global__ void CircularMallocFusedPrepare2(uint numChunks);
extern "C" __global__ void CircularMallocFusedPrepare3(uint numChunks, uint rootChunk);

extern "C" __global__ void CircularMultiMallocPrepare1(uint numChunks);
extern "C" __global__ void CircularMultiMallocPrepare2(uint numChunks);
extern "C" __global__ void CircularMultiMallocPrepare3(uint numChunks, uint rootChunk);

extern "C" __global__ void CircularMultiMallocFusedPrepare1(uint numChunks);
extern "C" __global__ void CircularMultiMallocFusedPrepare2(uint numChunks);
extern "C" __global__ void CircularMultiMallocFusedPrepare3(uint numChunks, uint rootChunk);

#endif

//------------------------------------------------------------------------
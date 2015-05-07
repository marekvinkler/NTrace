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

/*
    Warp wide utility functions.

    "Massively Parallel Hierarchical Scene Sorting with Applications in Rendering",
    Marek Vinkler, Michal Hapala, Jiri Bittner and Vlastimil Havran,
    Computer Graphics Forum 2012
*/

#pragma once
#include "CudaTracerDefines.h"


//------------------------------------------------------------------------
// Inline PTX loading functions with caching.
// For details refer to the PTX ISA manual for "Cache Operators"
//------------------------------------------------------------------------

// Loads data cached at all levels - default
template<typename T>
__device__ __forceinline__ T loadCA(const T* address);

// Loads data cached at L2 and above
template<typename T>
__device__ __forceinline__ T loadCG(const T* address);

// Loads data cached at all levels, with evict-first policy
template<typename T>
__device__ __forceinline__ T loadCS(const T* address);

// Loads data similar to loadCS, on local addresses discards L1 cache following the load
template<typename T>
__device__ __forceinline__ T loadLU(const T* address);

// Loads data volatilely, again from memory
template<typename T>
__device__ __forceinline__ T loadCV(const T* address);


//------------------------------------------------------------------------
// Inline PTX saving functions with caching.
// For details refer to the PTX ISA manual for "Cache Operators"
//------------------------------------------------------------------------

// Saves data cached at all levels - default
template<typename T>
__device__ __forceinline__ void saveWB(const T* address, T val);

// Saves data cached at L2 and above
template<typename T>
__device__ __forceinline__ void saveCG(const T* address, T val);

// Saves data cached at all levels, with evict-first policy
template<typename T>
__device__ __forceinline__ void saveCS(const T* address, T val);

// Saves data volatilely, write-through
template<typename T>
__device__ __forceinline__ void saveWT(const T* address, T val);


//------------------------------------------------------------------------
// Inline PTX volatile struct loading functions.
// For details refer to the PTX ISA manual for "Cache Operators"
//------------------------------------------------------------------------

__device__ __forceinline__ float4 loadfloat4V(const volatile float4* addr);

//------------------------------------------------------------------------
// Functions for reading and saving memory in a safe way.
//------------------------------------------------------------------------

// Read data from global memory
template<typename T>
__device__ __forceinline__ T getMemory(T* ptr);

// Save data to global memory
template<typename T>
__device__ __forceinline__ void setMemory(T* ptr, T value);


//------------------------------------------------------------------------
// Functional objects.
//------------------------------------------------------------------------


// Function for summing
template<typename T>
__device__ __forceinline__ T plus(T lhs, T rhs);


//------------------------------------------------------------------------
// Reduction.
//------------------------------------------------------------------------


// Warp sized reduction with butterfly pattern
template<typename T>
__device__ __forceinline__ void reduceWarp(int tid, volatile T* data, T(*op)(T,T));

// Warp sized reduction with tree pattern
template<typename T>
__device__ __forceinline__ void reduceWarpTree(int tid, volatile T* data, T(*op)(T,T));

// Warp sized reduction on items with different owners
template<typename T>
__device__ __forceinline__ void reduceWarpDiv(int tid, volatile T* data, volatile T* owner, T(*op)(T,T));

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
// Warp sized reduction with butterfly pattern using warp shuffles
template<typename T>
__device__ __forceinline__ void reduceWarp(T& value, T(*op)(T,T));
#endif

// Warp sized inclusive scan choosing between shared memory and shuffle based on a macro
template<typename T>
__device__ __forceinline__ void reduceWarp(int tid, volatile T* data, T& value, T(*op)(T,T));

// Warp sized segmented reduction with tree pattern
template<typename T>
__device__ void segReduceWarp(int tid, volatile T* data, volatile int* owner, T(*op)(T,T));

// Warp sized segmented reduction with tree pattern on three elements at once
template<typename T>
__device__ void segReduceWarp(int tid, volatile T* x, volatile T* y, volatile T* z, volatile int* owner, T(*op)(T,T));


//------------------------------------------------------------------------
// Prefix scans.
//------------------------------------------------------------------------

// Warp sized inclusive prefix scan
template<typename T>
__device__ __forceinline__ void scanWarp(int tid, volatile T* data, T(*op)(T,T));

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
// Warp sized inclusive prefix scan using warp shuffles
template<typename T>
__device__ __forceinline__ void scanWarp(int tid, T& value, T(*op)(T,T));
#endif

// Warp sized inclusive prefix scan
template<typename T>
__device__ __forceinline__ void scanWarp(int tid, volatile T* data, T& value, T(*op)(T,T));


//------------------------------------------------------------------------
// Warp position computation.
//------------------------------------------------------------------------

// Compute threads position inside warp based on a condition
__device__ __forceinline__ int threadPosWarp(int tid, volatile int* data, bool predicate, int& count);


//------------------------------------------------------------------------
// Sorting.
//------------------------------------------------------------------------


// Warp sized odd-even sort
template<typename T>
__device__ void transposition_sort(volatile T* keys, const unsigned int i, const unsigned int end);

// Warp sized odd-even sort of key-value pairs
template<typename T>
__device__ void transposition_sort_values(volatile T* keys, volatile T* values, const unsigned int i, const unsigned int end);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
// Warp sized odd-even sort for using warp shuffles
template<typename T>
__device__ void sortWarp(int tid, T& key, const unsigned int end);

// Warp sized odd-even sort of key-value pairs for using warp shuffles
template<typename T, typename S>
__device__ void sortWarp(int tid, T& key, S& value, const unsigned int end);

// Warp sized segmented odd-even sort of key-value pairs using warp shuffles
template<typename T>
__device__ void sortWarpSegmented(int tid, T& key, const int segmentID, const unsigned int end);

// Warp sized segmented odd-even sort of key-value pairs using warp shuffles
template<typename T, typename S>
__device__ void sortWarpSegmented(int tid, T& key, S& value, const int segmentID, const unsigned int end);
#endif


//------------------------------------------------------------------------
// Hashing.
//------------------------------------------------------------------------


// Jenkins mix hash
__device__ __forceinline__ void jenkinsMix(unsigned int& a, unsigned int& b, unsigned int& c);


//------------------------------------------------------------------------
// Utility.
//------------------------------------------------------------------------


// Alignment to multiply of S
template<typename T, int  S>
__device__ __forceinline__ T align(T a);

//------------------------------------------------------------------------
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
#include "warp_common.cuh"


//------------------------------------------------------------------------

// Loads data cached at all levels - default
template<>
__device__ __forceinline__ int loadCA<int>(const int* address)
{
	int val;
	asm("{\n\t"
		"ld.ca.s32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ unsigned int loadCA<unsigned int>(const unsigned int* address)
{
	unsigned int val;
	asm("{\n\t"
		"ld.ca.u32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint2 loadCA<uint2>(const uint2* address)
{
	uint2 val;
	asm("{\n\t"
		"ld.ca.v2.u32\t{%0, %1}, [%2];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint4 loadCA<uint4>(const uint4* address)
{
	uint4 val;
	asm("{\n\t"
		"ld.ca.v4.u32\t{%0, %1, %2, %3}, [%4];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ float loadCA<float>(const float* address)
{
	float val;
	asm("{\n\t"
		"ld.ca.f32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x64
		: "=f"(val) : "r"(address));
#else // x64
		: "=f"(val) : "l"(address));
#endif

	return val;
}

//------------------------------------------------------------------------

// Loads data cached at L2 and above
template<>
__device__ __forceinline__ int loadCG<int>(const int* address)
{
	int val;
	asm("{\n\t"
		"ld.cg.s32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ unsigned int loadCG<unsigned int>(const unsigned int* address)
{
	unsigned int val;
	asm("{\n\t"
		"ld.cg.u32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint2 loadCG<uint2>(const uint2* address)
{
	uint2 val;
	asm("{\n\t"
		"ld.cg.v2.u32\t{%0, %1}, [%2];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint4 loadCG<uint4>(const uint4* address)
{
	uint4 val;
	asm("{\n\t"
		"ld.cg.v4.u32\t{%0, %1, %2, %3}, [%4];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(address));
#endif

	return val;
}

//------------------------------------------------------------------------

// Loads data cached at all levels, with evict-first policy
template<>
__device__ __forceinline__ int loadCS<int>(const int* address)
{
	int val;
	asm("{\n\t"
		"ld.cs.s32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ unsigned int loadCS<unsigned int>(const unsigned int* address)
{
	unsigned int val;
	asm("{\n\t"
		"ld.cs.u32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint2 loadCS<uint2>(const uint2* address)
{
	uint2 val;
	asm("{\n\t"
		"ld.cs.v2.u32\t{%0, %1}, [%2];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint4 loadCS<uint4>(const uint4* address)
{
	uint4 val;
	asm("{\n\t"
		"ld.cs.v4.u32\t{%0, %1, %2, %3}, [%4];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ float loadCS<float>(const float* address)
{
	float val;
	asm("{\n\t"
		"ld.cs.f32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=f"(val) : "r"(address));
#else // x64
		: "=f"(val) : "l"(address));
#endif

	return val;
}

//------------------------------------------------------------------------

// Loads data similar to loadCS, on local addresses discards L1 cache following the load
template<>
__device__ __forceinline__ int loadLU<int>(const int* address)
{
	int val;
	asm("{\n\t"
		"ld.lu.s32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ unsigned int loadLU<unsigned int>(const unsigned int* address)
{
	unsigned int val;
	asm("{\n\t"
		"ld.lu.u32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint2 loadLU<uint2>(const uint2* address)
{
	uint2 val;
	asm("{\n\t"
		"ld.lu.v2.u32\t{%0, %1}, [%2];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint4 loadLU<uint4>(const uint4* address)
{
	uint4 val;
	asm("{\n\t"
		"ld.lu.v4.u32\t{%0, %1, %2, %3}, [%4];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ float loadLU<float>(const float* address)
{
	float val;
	asm("{\n\t"
		"ld.lu.f32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=f"(val) : "r"(address));
#else // x64
		: "=f"(val) : "l"(address));
#endif

	return val;
}

//------------------------------------------------------------------------

// Loads data volatilely, again from memory
template<>
__device__ __forceinline__ int loadCV<int>(const int* address)
{
	int val;
	asm("{\n\t"
		"ld.cv.s32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ unsigned int loadCV<unsigned int>(const unsigned int* address)
{
	unsigned int val;
	asm("{\n\t"
		"ld.cv.u32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val) : "r"(address));
#else // x64
		: "=r"(val) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint2 loadCV<uint2>(const uint2* address)
{
	uint2 val;
	asm("{\n\t"
		"ld.cv.v2.u32\t{%0, %1}, [%2];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ uint4 loadCV<uint4>(const uint4* address)
{
	uint4 val;
	asm("{\n\t"
		"ld.cv.v4.u32\t{%0, %1, %2, %3}, [%4];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "r"(address));
#else // x64
		: "=r"(val.x), "=r"(val.y), "=r"(val.z), "=r"(val.w) : "l"(address));
#endif

	return val;
}

template<>
__device__ __forceinline__ float loadCV<float>(const float* address)
{
	float val;
	asm("{\n\t"
		"ld.cv.f32\t%0, [%1];\n\t"
		"}"
#ifndef _M_X64 // x86
		: "=f"(val) : "r"(address));
#else // x64
		: "=f"(val) : "l"(address));
#endif

	return val;
}

//------------------------------------------------------------------------

// Saves data cached at all levels - default
template<>
__device__ __forceinline__ void saveWB<int>(const int* address, int val)
{
	asm("{\n\t"
		"st.wb.s32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveWB<unsigned int>(const unsigned int* address, unsigned int val)
{
	asm("{\n\t"
		"st.wb.u32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveWB<uint2>(const uint2* address, uint2 val)
{
	asm("{\n\t"
		"st.wb.v2.u32\t[%0], {%1, %2};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y));
#endif
}

template<>
__device__ __forceinline__ void saveWB<uint4>(const uint4* address, uint4 val)
{
	asm("{\n\t"
		"st.wb.v4.u32\t[%0], {%1, %2, %3, %4};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

template<>
__device__ __forceinline__ void saveWB<float>(const float* address, float val)
{
	asm("{\n\t"
		"st.wb.f32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "f"(val));
#else // x64
		:: "l"(address), "f"(val));
#endif
}

//------------------------------------------------------------------------

// Saves data cached at L2 and above
template<>
__device__ __forceinline__ void saveCG<int>(const int* address, int val)
{
	asm("{\n\t"
		"st.cg.s32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveCG<unsigned int>(const unsigned int* address, unsigned int val)
{
	asm("{\n\t"
		"st.cg.u32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveCG<uint2>(const uint2* address, uint2 val)
{
	asm("{\n\t"
		"st.cg.v2.u32\t[%0], {%1, %2};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y));
#endif
}

template<>
__device__ __forceinline__ void saveCG<uint4>(const uint4* address, uint4 val)
{
	asm("{\n\t"
		"st.cg.v4.u32\t[%0], {%1, %2, %3, %4};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

template<>
__device__ __forceinline__ void saveCG<float>(const float* address, float val)
{
	asm("{\n\t"
		"ld.st.f32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "f"(val));
#else // x64
		:: "l"(address), "f"(val));
#endif
}

//------------------------------------------------------------------------

// Saves data cached at all levels, with evict-first policy
template<>
__device__ __forceinline__ void saveCS<int>(const int* address, int val)
{
	asm("{\n\t"
		"st.cs.s32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveCS<unsigned int>(const unsigned int* address, unsigned int val)
{
	asm("{\n\t"
		"st.cs.u32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveCS<uint2>(const uint2* address, uint2 val)
{
	asm("{\n\t"
		"st.cs.v2.u32\t[%0], {%1, %2};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y));
#endif
}

template<>
__device__ __forceinline__ void saveCS<uint4>(const uint4* address, uint4 val)
{
	asm("{\n\t"
		"st.cs.v4.u32\t[%0], {%1, %2, %3, %4};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

template<>
__device__ __forceinline__ void saveCS<float>(const float* address, float val)
{
	asm("{\n\t"
		"st.cs.f32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "f"(val));
#else // x64
		:: "l"(address), "f"(val));
#endif
}

//------------------------------------------------------------------------

// Saves data volatilely, write-through
template<>
__device__ __forceinline__ void saveWT<int>(const int* address, int val)
{
	asm("{\n\t"
		"st.wt.s32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveWT<unsigned int>(const unsigned int* address, unsigned int val)
{
	asm("{\n\t"
		"st.wt.u32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val));
#else // x64
		:: "l"(address), "r"(val));
#endif
}

template<>
__device__ __forceinline__ void saveWT<uint2>(const uint2* address, uint2 val)
{
	asm("{\n\t"
		"st.wt.v2.u32\t[%0], {%1, %2};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y));
#endif
}

template<>
__device__ __forceinline__ void saveWT<uint4>(const uint4* address, uint4 val)
{
	asm("{\n\t"
		"st.wt.v4.u32\t[%0], {%1, %2, %3, %4};\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#else // x64
		:: "l"(address), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w));
#endif
}

template<>
__device__ __forceinline__ void saveWT<float>(const float* address, float val)
{
	asm("{\n\t"
		"st.wt.f32\t[%0], %1;\n\t"
		"}"
#ifndef _M_X64 // x86
		:: "r"(address), "f"(val));
#else // x64
		:: "l"(address), "f"(val));
#endif
}

//------------------------------------------------------------------------

__device__ __forceinline__ float4 loadfloat4V(const volatile float4* addr)
{
	float4 ret;
	asm("{\n\t"
		"ld.volatile.v4.f32\t{%0, %1, %2, %3}, [%4];\n\t"
		"}"
		: "=f"(ret.x), "=f"(ret.y), "=f"(ret.z), "=f"(ret.w)
#ifndef _M_X64 // x86
		: "r"(addr));
#else // x64
		: "l"(addr));
#endif

	return ret;
}

//------------------------------------------------------------------------

__device__ __forceinline__ unsigned int getMemory(unsigned int* ptr)
{
#if (MEM_ACCESS_TYPE == 0)
	return *ptr;
#elif (MEM_ACCESS_TYPE == 1)
	return loadCG<uint>(ptr);
#elif (MEM_ACCESS_TYPE == 2)
	return atomicCAS(ptr, 0xFFFFFFFF, 0); // Use atomic CAS with almost impossible value to get a value
#endif
}

__device__ __forceinline__ uint2 getMemory(uint2* ptr)
{
#if (MEM_ACCESS_TYPE == 0)
	return *ptr;
#elif (MEM_ACCESS_TYPE == 1)
	return loadCG<uint2>(ptr);
#elif (MEM_ACCESS_TYPE == 2)
	unsigned long long int ret = atomicCAS((unsigned long long int*)ptr, 0xFFFFFFFFFFFFFFFF, 0); // Use atomic CAS with almost impossible value to get a value
	return make_uint2(ret & 0xFFFFFFFF, ret >> 32);
#endif
}

__device__ __forceinline__ uint4 getMemory(uint4* ptr)
{
#if (MEM_ACCESS_TYPE == 0)
	return *ptr;
#elif (MEM_ACCESS_TYPE == 1)
	return loadCG<uint4>(ptr);
#elif (MEM_ACCESS_TYPE == 2)
	unsigned long long int ret1 = atomicCAS((unsigned long long int*)ptr, 0xFFFFFFFFFFFFFFFF, 0); // Use atomic CAS with almost impossible value to get a value
	unsigned long long int ret2 = atomicCAS((unsigned long long int*)((char*)ptr+sizeof(unsigned long long int)), 0xFFFFFFFFFFFFFFFF, 0); // Use atomic CAS with almost impossible value to get a value
	return make_uint4(ret1 & 0xFFFFFFFF, ret1 >> 32, ret2 & 0xFFFFFFFF, ret2 >> 32);
#endif
}

//------------------------------------------------------------------------

__device__ __forceinline__ void setMemory(unsigned int* ptr, unsigned int value)
{
#if (MEM_ACCESS_TYPE == 0)
	*ptr = value;
#elif (MEM_ACCESS_TYPE == 1)
	saveCG<uint>(ptr, value);
#elif (MEM_ACCESS_TYPE == 2)
	atomicExch(ptr, value);
#endif
}

__device__ __forceinline__ void setMemory(uint2* ptr, uint2 value)
{
#if (MEM_ACCESS_TYPE == 0)
	*ptr = value;
#elif (MEM_ACCESS_TYPE == 1)
	saveCG<uint2>(ptr, value);
#elif (MEM_ACCESS_TYPE == 2)
	unsigned long long int val;
	val = ((unsigned long long int)value.y << 32) | (unsigned long long int)value.x;
	atomicExch((unsigned long long int*)ptr, val);
#endif
}

__device__ __forceinline__ void setMemory(uint4* ptr, uint4 value)
{
#if (MEM_ACCESS_TYPE == 0)
	*ptr = value;
#elif (MEM_ACCESS_TYPE == 1)
	saveCG<uint4>(ptr, value);
#elif (MEM_ACCESS_TYPE == 2)
	unsigned long long int val;
	val = ((unsigned long long int)value.y << 32) | (unsigned long long int)value.x;
	atomicExch((unsigned long long int*)ptr, val);
	val = ((unsigned long long int)value.w << 32) | (unsigned long long int)value.z;
	atomicExch((unsigned long long int*)((char*)ptr+sizeof(unsigned long long int)), val);
#endif
}

//------------------------------------------------------------------------

// Function for summing
template<typename T>
__device__ __forceinline__ T plus(T lhs, T rhs)
{
	return lhs + rhs;
}

//------------------------------------------------------------------------

// Warp sized reduction with butterfly pattern
template<typename T>
__device__ __forceinline__ void reduceWarp(int tid, volatile T* data, T(*op)(T,T))
{
	data[tid] = op(data[tid], data[tid ^ 1]);
	data[tid] = op(data[tid], data[tid ^ 2]);
	data[tid] = op(data[tid], data[tid ^ 4]);
	data[tid] = op(data[tid], data[tid ^ 8]);
	data[tid] = op(data[tid], data[tid ^ 16]);
}

/*template<class T>
class Min
{
public:
	__device__ __forceinline__ T operator() (T a, T b) const { return min(a, b); }
};

template<class T, class O>
__device__ __forceinline__ void reduceWarp(int tid, volatile T* data, O op)
{
	data[tid] = op(data[tid], data[tid ^ 1]);
	data[tid] = op(data[tid], data[tid ^ 2]);
	data[tid] = op(data[tid], data[tid ^ 4]);
	data[tid] = op(data[tid], data[tid ^ 8]);
	data[tid] = op(data[tid], data[tid ^ 16]);
}*/

//------------------------------------------------------------------------

// Warp sized reduction with tree pattern
template<typename T>
__device__ __forceinline__ void reduceWarpTree(int tid, volatile T* data, T(*op)(T,T))
{
	if((tid &  1) == 1)
		data[tid] = op(data[tid], data[tid -  1]);
	if((tid &  3) == 3)
		data[tid] = op(data[tid], data[tid -  2]);
	if((tid &  7) == 7)
		data[tid] = op(data[tid], data[tid -  4]);
	if((tid & 15) == 15)
		data[tid] = op(data[tid], data[tid -  8]);
	if((tid & 31) == 31)
		data[tid] = op(data[tid], data[tid - 16]);
}

//------------------------------------------------------------------------

// Warp sized reduction on items with different owners
template<typename T>
__device__ __forceinline__ void reduceWarpDiv(int tid, volatile T* data, volatile T* owner, T(*op)(T,T))
{
	if(owner[tid] != owner[tid ^ 1])
		data[tid] = op(data[tid], data[tid ^ 1]);
	if(owner[tid] != owner[tid ^ 2])
		data[tid] = op(data[tid], data[tid ^ 2]);
	if(owner[tid] != owner[tid ^ 4])
		data[tid] = op(data[tid], data[tid ^ 4]);
	if(owner[tid] != owner[tid ^ 8])
		data[tid] = op(data[tid], data[tid ^ 8]);
	if(owner[tid] != owner[tid ^ 16])
		data[tid] = op(data[tid], data[tid ^ 16]);
}

//------------------------------------------------------------------------

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
// Warp sized reduction with butterfly pattern using warp shuffles
template<typename T>
__device__ __forceinline__ void reduceWarp(T& value, T(*op)(T,T))
{
	// Use XOR mode to perform butterfly reduction
	value = op(value, __shfl_xor(value, 1));
	value = op(value, __shfl_xor(value, 2));
	value = op(value, __shfl_xor(value, 4));
	value = op(value, __shfl_xor(value, 8));
	value = op(value, __shfl_xor(value, 16));
}
//#endif

//------------------------------------------------------------------------

// Warp sized reduction with butterfly pattern choosing between shared memory and shuffle based on a macro
template<typename T>
__device__ __forceinline__ void reduceWarp(int tid, volatile T* data, T& value, T(*op)(T,T))
{
#ifndef SHUFFLE_RED_SCAN
	reduceWarp<T>(tid, data, op);
#else
	reduceWarp<T>(value, op);
#endif
}

//------------------------------------------------------------------------

// Warp sized segmented reduction with tree pattern
template<typename T>
__device__ void segReduceWarp(int tid, volatile T* data, volatile int* owner, T(*op)(T,T))
{
	int owner0 = owner[tid];
	int owner1;

	if((tid & 1) == 0)
	{
		owner1 = owner[tid + 1];
		if(owner0 == owner1)
		{
			data[tid] = op(data[tid], data[tid + 1]);
		}
		else
		{
			data[owner1] = op(data[owner1], data[tid + 1]);
		}
	}

	if((tid & 3) == 0)
	{
		owner1 = owner[tid + 2];
		if(owner0 == owner[tid + 2])
		{
			data[tid] = op(data[tid], data[tid + 2]);
		}
		else
		{
			data[owner1] = op(data[owner1], data[tid + 2]);
		}
	}

	if((tid & 7) == 0)
	{
		owner1 = owner[tid + 4];
		if(owner0 == owner[tid + 4])
		{
			data[tid] = op(data[tid], data[tid + 4]);
		}
		else
		{
			data[owner1] = op(data[owner1], data[tid + 4]);
		}
	}

	if((tid & 15) == 0)
	{
		owner1 = owner[tid + 8];
		if(owner0 == owner[tid + 8])
		{
			data[tid] = op(data[tid], data[tid + 8]);
		}
		else
		{
			data[owner1] = op(data[owner1], data[tid + 8]);
		}
	}

	if((tid & 31) == 0)
	{
		owner1 = owner[tid + 16];
		if(owner0 == owner[tid + 16])
		{
			data[tid] = op(data[tid], data[tid + 16]);
		}
		else
		{
			data[owner1] = op(data[owner1], data[tid + 16]);
		}
	}
}

//------------------------------------------------------------------------

// Warp sized segmented reduction with tree pattern on three elements at once
template<typename T>
__device__ void segReduceWarp(int tid, volatile T* x, volatile T* y, volatile T* z, volatile int* owner, T(*op)(T,T))
{
	int owner0 = owner[tid];
	int owner1;

	if((tid & 1) == 0)
	{
		owner1 = owner[tid + 1];
		if(owner0 == owner1)
		{
			x[tid] = op(x[tid], x[tid + 1]);
			y[tid] = op(y[tid], y[tid + 1]);
			z[tid] = op(z[tid], z[tid + 1]);
		}
		else
		{
			x[owner1] = op(x[owner1], x[tid + 1]);
			y[owner1] = op(y[owner1], y[tid + 1]);
			z[owner1] = op(z[owner1], z[tid + 1]);
		}
	}

	if((tid & 3) == 0)
	{
		owner1 = owner[tid + 2];
		if(owner0 == owner[tid + 2])
		{
			x[tid] = op(x[tid], x[tid + 2]);
			y[tid] = op(y[tid], y[tid + 2]);
			z[tid] = op(z[tid], z[tid + 2]);
		}
		else
		{
			x[owner1] = op(x[owner1], x[tid + 2]);
			y[owner1] = op(y[owner1], y[tid + 2]);
			z[owner1] = op(z[owner1], z[tid + 2]);
		}
	}

	if((tid & 7) == 0)
	{
		owner1 = owner[tid + 4];
		if(owner0 == owner[tid + 4])
		{
			x[tid] = op(x[tid], x[tid + 4]);
			y[tid] = op(y[tid], y[tid + 4]);
			z[tid] = op(z[tid], z[tid + 4]);
		}
		else
		{
			x[owner1] = op(x[owner1], x[tid + 4]);
			y[owner1] = op(y[owner1], y[tid + 4]);
			z[owner1] = op(z[owner1], z[tid + 4]);
		}
	}

	if((tid & 15) == 0)
	{
		owner1 = owner[tid + 8];
		if(owner0 == owner[tid + 8])
		{
			x[tid] = op(x[tid], x[tid + 8]);
			y[tid] = op(y[tid], y[tid + 8]);
			z[tid] = op(z[tid], z[tid + 8]);
		}
		else
		{
			x[owner1] = op(x[owner1], x[tid + 8]);
			y[owner1] = op(y[owner1], y[tid + 8]);
			z[owner1] = op(z[owner1], z[tid + 8]);
		}
	}

	if((tid & 31) == 0)
	{
		owner1 = owner[tid + 16];
		if(owner0 == owner[tid + 16])
		{
			x[tid] = op(x[tid], x[tid + 16]);
			y[tid] = op(y[tid], y[tid + 16]);
			z[tid] = op(z[tid], z[tid + 16]);
		}
		else
		{
			x[owner1] = op(x[owner1], x[tid + 16]);
			y[owner1] = op(y[owner1], y[tid + 16]);
			z[owner1] = op(z[owner1], z[tid + 16]);
		}
	}
}

//------------------------------------------------------------------------

// Warp sized inclusive prefix scan
template<typename T>
__device__ __forceinline__ void scanWarp(int tid, volatile T* data, T(*op)(T,T))
{
	if(tid >=  1)
		data[tid] = op(data[tid], data[tid -  1]);
	if(tid >=  2)
		data[tid] = op(data[tid], data[tid -  2]);
	if(tid >=  4)
		data[tid] = op(data[tid], data[tid -  4]);
	if(tid >=  8)
		data[tid] = op(data[tid], data[tid -  8]);
	if(tid >= 16)
		data[tid] = op(data[tid], data[tid - 16]);
}

//------------------------------------------------------------------------

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
// Warp sized inclusive prefix scan using warp shuffles
template<typename T>
__device__ __forceinline__ void scanWarp(int tid, T& value, T(*op)(T,T))
{
	int n;
	n = __shfl_up(value, 1);
	if(tid >=  1)
		value = op(value, n);
	n = __shfl_up(value, 2);
	if(tid >=  2)
		value = op(value, n);
	n = __shfl_up(value, 4);
	if(tid >=  4)
		value = op(value, n);
	n = __shfl_up(value, 8);
	if(tid >=  8)
		value = op(value, n);
	n = __shfl_up(value, 16);
	if(tid >= 16)
		value = op(value, n);
}
//#endif

//------------------------------------------------------------------------

// Warp sized inclusive scan choosing between shared memory and shuffle based on a macro
template<typename T>
__device__ __forceinline__ void scanWarp(int tid, volatile T* data, T& value, T(*op)(T,T))
{
#ifndef SHUFFLE_RED_SCAN
	scanWarp<T>(tid, data, op);
#else
	scanWarp<T>(tid, value, op);
#endif
}

//------------------------------------------------------------------------

// Compute threads position inside warp based on a condition
__device__ __forceinline__ int threadPosWarp(int tid, volatile int* data, bool predicate, int& count)
{
#if 1
	data[tid] = 0;
	if(predicate)
		data[tid] = 1;

	scanWarp<int>(tid, data, plus);
	int exclusiveScan = (data[tid] - 1);

	count = data[WARP_SIZE-1];
#else
	unsigned int mask = __ballot(predicate);
	int exclusiveScan = __popc(mask & ((1 << tid) - 1));
	count = __popc(mask);
#endif

	return exclusiveScan;
}

//------------------------------------------------------------------------

// Conditional swap operation for sorting keys
template<typename T>
__device__ void conditional_swap(volatile T* keys, const unsigned int i, const unsigned int end, bool pred)
{
	if(pred && i+1<end)
	{
		T xi = keys[i];
		T xj = keys[i+1];

		// swap if xj sorts before xi
		if(xj < xi)
		{
			keys[i]     = xj;
			keys[i+1]   = xi;
		}
	}
}

//------------------------------------------------------------------------

// Conditional swap operation for sorting keys
template<typename T>
__device__ void conditional_swap_values(volatile T* keys, volatile T* values, const unsigned int i, const unsigned int end, bool pred)
{
	if(pred && i+1<end)
	{
		T xi = keys[i];
		T xj = keys[i+1];

		// swap if xj sorts before xi
		if(xj < xi)
		{
			T yi = values[i];
			T yj = values[i+1];

			keys[i]     = xj;
			keys[i+1]   = xi;
			values[i]   = yj;
			values[i+1] = yi;

		}
	}
}

//------------------------------------------------------------------------

// Warp sized odd-even sort
template<typename T>
__device__ void transposition_sort(volatile T* keys, const unsigned int i, const unsigned int end)
{
	const bool is_odd = i&0x1;

	for(unsigned int round=WARP_SIZE/2; round>0; --round)
	{
		// ODDS
		conditional_swap(keys, i, end, is_odd);

		// EVENS
		conditional_swap(keys, i, end, !is_odd);
	}
}

//------------------------------------------------------------------------

// Warp sized odd-even sort
template<typename T>
__device__ void transposition_sort_values(volatile T* keys, volatile T* values, const unsigned int i, const unsigned int end)
{
	const bool is_odd = i&0x1;

	for(unsigned int round=WARP_SIZE/2; round>0; --round)
	{
		// ODDS
		conditional_swap_values(keys, values, i, end, is_odd);

		// EVENS
		conditional_swap_values(keys, values, i, end, !is_odd);
	}
}

//#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)

//------------------------------------------------------------------------

// Conditional swap operation for sorting keys
template<typename T>
__device__ bool conditional_swap(const int tid, T& key, const unsigned int end, const int dir)
{
	if((dir > 0 && tid+1 < end) || (dir < 0 && tid > 0 && tid < end)) // Out of range access for tid == 0 is NOT handled by _shfl!!!
	{
		T xi = key;
		T xj = __shfl(key, tid+dir);

		// Swap if xj sorts before xi
		if((dir > 0 && xj < xi) || (dir < 0 && xi < xj)) // Both lanes must be active for the swap
		{
			key     = xj; // Already swaped
			return false;
		}
	}

	return true;
}

//------------------------------------------------------------------------

// Conditional swap operation for sorting key-value pairs
template<typename T, typename S>
__device__ bool conditional_swap_values(const int tid, T& key, S& value, const unsigned int end, const int dir)
{
	if((dir > 0 && tid+1 < end) || (dir < 0 && tid > 0 && tid < end)) // Out of range access for tid == 0 is NOT handled by _shfl!!!
	{
		T xi = key;
		T xj = __shfl(key, tid+dir);

		// Swap if xj sorts before xi
		if((dir > 0 && xj < xi) || (dir < 0 && xi < xj)) // Both lanes must be active for the swap
		{
			key     = xj; // Already swaped
			value   = __shfl(value, tid+dir);
			return false;
		}
	}

	return true;
}

//------------------------------------------------------------------------

// Conditional swap operation for sorting key-value pairs
template<typename T>
__device__ bool segmented_swap(const int tid, T& key, const int segmentID, const unsigned int end, const int dir)
{
	int sID = __shfl(segmentID, tid+dir);
	if((dir > 0 && tid+1 < end && segmentID == sID) || (dir < 0 && tid > 0 && tid < end && segmentID == sID)) // Make sure swapping takes place intra-segment
	{
		T xi = key;
		T xj = __shfl(key, tid+dir);

		// Swap if xj sorts before xi
		if((dir > 0 && xj < xi) || (dir < 0 && xi < xj)) // Both lanes must be active for the swap
		{
			key     = xj; // Already swaped
			return false;
		}
	}

	return true;
}

//------------------------------------------------------------------------

// Conditional swap operation for sorting key-value pairs
template<typename T, typename S>
__device__ bool segmented_swap_values(const int tid, T& key, S& value, const int segmentID, const unsigned int end, const int dir)
{
	int sID = __shfl(segmentID, tid+dir);
	if((dir > 0 && tid+1 < end && segmentID == sID) || (dir < 0 && tid > 0 && tid < end && segmentID == sID)) // Make sure swapping takes place intra-segment
	{
		T xi = key;
		T xj = __shfl(key, tid+dir);

		// Swap if xj sorts before xi
		if((dir > 0 && xj < xi) || (dir < 0 && xi < xj)) // Both lanes must be active for the swap
		{
			key     = xj; // Already swaped
			value   = __shfl(value, tid+dir);
			return false;
		}
	}

	return true;
}

//------------------------------------------------------------------------

// Warp sized odd-even sort using warp shuffles
template<typename T>
__device__ void sortWarp(int tid, T& key, const unsigned int end)
{
	const int is_odd = (tid&0x1)*2 - 1;
	bool sorted = false;

	for(unsigned int round=WARP_SIZE/2; round>0; --round)
	{
		// ODDS
		sorted = __all(conditional_swap_values(tid, key, end, is_odd));

		// EVENS
		sorted = __all(conditional_swap_values(tid, key, end, is_odd*(-1))) && sorted;
	}
}

//------------------------------------------------------------------------

// Warp sized odd-even sort of key-value pairs using warp shuffles
template<typename T, typename S>
__device__ void sortWarp(int tid, T& key, S& value, const unsigned int end)
{
	const int is_odd = (tid&0x1)*2 - 1;
	bool sorted = false;

	for(unsigned int round=WARP_SIZE/2; round>0; --round)
	{
		// ODDS
		sorted = __all(conditional_swap_values(tid, key, value, end, is_odd));

		// EVENS
		sorted = __all(conditional_swap_values(tid, key, value, end, is_odd*(-1))) && sorted;
	}
}

//------------------------------------------------------------------------

// Warp sized odd-even sort using warp shuffles
template<typename T>
__device__ void sortWarpSegmented(int tid, T& key, const int segmentID, const unsigned int end)
{
	const int is_odd = (tid&0x1)*2 - 1;
	bool sorted = false;

	for(unsigned int round=WARP_SIZE/2; round>0; --round)
	{
		// ODDS
		sorted = __all(segmented_swap(tid, key, segmentID, end, is_odd));

		// EVENS
		sorted = __all(segmented_swap(tid, key, segmentID, end, is_odd*(-1))) && sorted;
	}
}

//------------------------------------------------------------------------

// Warp sized odd-even sort of key-value pairs using warp shuffles
template<typename T, typename S>
__device__ void sortWarpSegmented(int tid, T& key, S& value, const int segmentID, const unsigned int end)
{
	const int is_odd = (tid&0x1)*2 - 1;
	bool sorted = false;

	for(unsigned int round=WARP_SIZE/2; !sorted && round>0; --round)
	{
		// ODDS
		sorted = __all(segmented_swap_values(tid, key, value, segmentID, end, is_odd));

		// EVENS
		sorted = __all(segmented_swap_values(tid, key, value, segmentID, end, is_odd*(-1))) && sorted;
	}
}
//#endif

//------------------------------------------------------------------------

// Jenkins mix hash
__device__ __forceinline__ void jenkinsMix(unsigned int& a, unsigned int& b, unsigned int& c)
{
    a -= b; a -= c; a ^= (c>>13);
    b -= c; b -= a; b ^= (a<<8);
    c -= a; c -= b; c ^= (b>>13);
    a -= b; a -= c; a ^= (c>>12);
    b -= c; b -= a; b ^= (a<<16);
    c -= a; c -= b; c ^= (b>>5);
    a -= b; a -= c; a ^= (c>>3);
    b -= c; b -= a; b ^= (a<<10);
    c -= a; c -= b; c ^= (b>>15);
}

//------------------------------------------------------------------------

// Alignment to multiply of S
template<typename T, int  S>
__device__ __forceinline__ T align(T a)
{
	 return (a+S-1) & ~(S-1);
}

//------------------------------------------------------------------------
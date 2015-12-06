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
    Common functionality used for each framework specialization.

    "Massively Parallel Hierarchical Scene Sorting with Applications in Rendering",
    Marek Vinkler, Michal Hapala, Jiri Bittner and Vlastimil Havran,
    Computer Graphics Forum 2012
*/

#pragma once
#include "allocators.cu"

//-----------------------------------------------------------------------

// Allocate memory for the root node
__global__ void allocFreeableMemory(int numTris, int numRays)
{
	// Save the base pointer (hopefully) to the heap

#if (MALLOC_TYPE == CUDA_MALLOC)
	//g_heapBase = (char*)mallocCudaMalloc(1024 * 1024 * 5);
	g_heapBase = (char*)mallocCudaMalloc(numTris*sizeof(int));
#elif (MALLOC_TYPE == CIRCULAR_MALLOC)
	mallocCircularMalloc(numTris*sizeof(int));
#elif (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED)
	mallocCircularMallocFused(numTris*sizeof(int));
#elif (MALLOC_TYPE == SCATTER_ALLOC)
	g_heapBase = (char*)mallocScatterAlloc(numTris*sizeof(int));
#elif (MALLOC_TYPE == HALLOC)
	g_heapBase2 = 0;
	void *heap[32];
	for(int i = 0; i < 32; i++)
	{
		heap[i] = malloc(1<<i);
		g_heapBase2 = (char*)max((unsigned long long)g_heapBase2, (unsigned long long)heap[i]);
		//printf("%d : %p\n", i, g_heapBase2);
	}
	for(int i = 0; i < 32; i++)
	{
		free(heap[i]);
	}
	g_heapBase = (char*)mallocHalloc(numTris*sizeof(int));
#elif (MALLOC_TYPE == FDG_MALLOC)
	g_heapBase = (char*)mallocFDGMalloc(warp, numTris*sizeof(int));
#endif

#ifdef CHECK_OUT_OF_MEMORY
	if(g_heapBase == NULL)
		printf("Out of memory!\n");
#endif
}

//------------------------------------------------------------------------

// Deallocate all memory
__global__ void deallocFreeableMemory()
{
	free((void*)g_heapBase);
}

//------------------------------------------------------------------------

// Copy data for the root node from CPU allocated to GPU allocated device space.
__global__ void MemCpyIndex(CUdeviceptr src, int ofs, int size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if(tid < size)
		((int*)(g_heapBase+ofs))[tid] = ((int*)src)[tid];
}

//------------------------------------------------------------------------
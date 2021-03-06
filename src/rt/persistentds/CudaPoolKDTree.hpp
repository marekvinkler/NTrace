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
    GF100-optimized variant of the "Speculative while-while"
    kernel used in:

    "Understanding the Efficiency of Ray Traversal on GPUs",
    Timo Aila and Samuli Laine,
    Proc. High-Performance Graphics 2009
*/

#pragma once
#include "CudaPool.hpp"
#include "rt_common.cuh"

//------------------------------------------------------------------------
// Optimizations macros
//------------------------------------------------------------------------

#define WOOP_TRIANGLES // Save triangles in woop representation
#define COMPACT_LAYOUT // Save triangle pointers immediately in parent
//#define DUPLICATE_REFERENCES

#define ALIGN 16
//#define COALESCE_WARP
//#define NO_FREE // Do not free dynamic memory, for testing purposes only

#define CUDA_MALLOC 0 // Default CUDA allocator
#define ATOMIC_MALLOC 1 // Allocate memory by just atomically adding an offset
#define ATOMIC_MALLOC_CIRCULAR 2 // Circular variant of atomic malloc
#define CIRCULAR_MALLOC 3 // Allocator that used a linked list in global heap
#define CIRCULAR_MALLOC_FUSED 4 // Allocator that used a linked list in global heap and fuses the lock and next pointer
#define CIRCULAR_MULTI_MALLOC 5 // Allocator that used a linked list in global heap and multiple heap offsets
#define CIRCULAR_MULTI_MALLOC_FUSED 6 // Allocator that used a linked list in global heap, multiple heap offsets and fuses the lock and next pointer
#define SCATTER_ALLOC 7 // Use ScatterAlloc for allocations
#define FDG_MALLOC 8 // Use FDG for allocations
#define HALLOC 9 // Use Halloc for allocations

// NOTICE: Due to the unknown base of CudaMalloc CUDA_MALLOC, FDG_MALLOC and HALLOC allocators may be unstable
#define MALLOC_TYPE CIRCULAR_MALLOC

//------------------------------------------------------------------------
// AtomicMalloc
//------------------------------------------------------------------------

#if (MALLOC_TYPE == ATOMIC_MALLOC) || (MALLOC_TYPE == ATOMIC_MALLOC_CIRCULAR)
#define NO_FREE // Memory cannot be dealocated in this simple strategy
#endif

//------------------------------------------------------------------------
// CircularMalloc
//------------------------------------------------------------------------

#define CIRCULAR_MALLOC_CHECK_DEADLOCK
//#define CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK // Use a single global lock for the heap
//#define CIRCULAR_MALLOC_DOUBLY_LINKED
#define CIRCULAR_MALLOC_PRELOCK
#define CIRCULAR_MALLOC_CONNECT_CHUNKS
// How to write data into global memory
// 0 - direct memory access
// 1 - through inline PTX caching qualifier
// 2 - through atomic operation
#define CIRCULAR_MALLOC_MEM_ACCESS_TYPE 0
#define MEM_ACCESS_TYPE CIRCULAR_MALLOC_MEM_ACCESS_TYPE // Macro in warp_common.cu
#define ALIGN 16
#define CIRCULAR_MALLOC_WAIT_COUNT 1000000
//#define CIRCULAR_MALLOC_WITH_SCATTER_ALLOC // Use ScatterAlloc for small nodes
//#define CIRCULAR_MALLOC_SWITCH_SIZE 1024

//------------------------------------------------------------------------
// ScatterAlloc
//------------------------------------------------------------------------

#define SCATTER_ALLOC_PAGESIZE 16384
#define SCATTER_ALLOC_ACCESSBLOCKS 8
#define SCATTER_ALLOC_REGIONSIZE 16
#define SCATTER_ALLOC_WASTEFACTOR 2
#define SCATTER_ALLOC_COALESCING 0
#define SCATTER_ALLOC_RESETPAGES 1

// Include the allocators modified by the above macros
#include "Allocators.hpp"

//------------------------------------------------------------------------
// Kd-tree data structures
//------------------------------------------------------------------------

struct KernelInputKdtree
{
	int             numTris;        // Total number of tris.
	CUdeviceptr     tris;
	CUdeviceptr     trisIndex;      // Triangle index remapping table.

	CUdeviceptr		trisOut;
	CUdeviceptr		trisIndexOut;
	
	CUdeviceptr		debug;
};

//------------------------------------------------------------------------

// BEWARE THAT STRUCT ALIGNMENT MAY INCREASE THE DATA SIZE AND BREAK 1 INSTRUCTION LOAD/STORE
// CURRENT ORDER ENSURES THAT splitPlane AND bbox ARE ALIGNED TO 16B
struct __align__(128) TaskKdtree
{
	int       unfinished;        // Counts the number of unfinished sub tasks
    int       type;              // Type of work to be done
	int       dynamicMemoryLeft; // Chunk of dynamic memory given to the left child as offset from g_heapBase
	int       dynamicMemoryRight;// Chunk of dynamic memory given to the right child as offset from g_heapBase
	
	int       nodeIdx;           // Address of this node
    int       parentIdx;         // Where to write node data
	int       taskID;            // Index among the subtasks of the parent
	int       axis;              // Splitting axis set in round-robin fashion
							     
	int       triStart;          // Start of the triangle interval, inclusive
	int       triEnd;            // End of the triangle interval, exclusive
	int       triLeft;           // End of the triangle left interval
	int       triRight;          // Start of the triangle right interval, exclusive
							     
	CudaAABB  bbox;              // The box corresponding to the space occupied by triangles, needed for median splitting, may be changed in future
	int       step;              // Counts the progress in the multi-pass work
	
	// This block of tasks must be in this order!!! Required for simultaneous copy by multiple threads
	int       lock;              // Lock for setting the best plane + holds information about synchronization skipping
	float4    splitPlane;        // The plane we chose to split this task's rays and triangles
	float     bestCost;          // Cost of the best split plane
	int       bestOrder;         // Best traversal order of the split plane + holds information about the number of ray tasks for RAYTRI_PARALLEL
	
	int       depth;             // Depth of the current node, for subdivision ending
	int       dynamicMemory;     // Chunk of dynamic memory given to this node

	int       pivot;		     // For test kernel holds the value of the pivot
	int       subFailureCounter; // Increments each time task fails to subdivide properly
	int       origSize;			 // Size the task was created with
	int       terminatedBy;

	// 128B boundary

	// Data order is to maintain structure alignment
	CudaAABB  bboxLeft;          // Left childs bounding box
	long long int clockStart;
	
	CudaAABB  bboxMiddle;        // Middle childs bounding box
	long long int clockEnd;

	CudaAABB  bboxRight;         // Right childs bounding box

	int       parent;            // Index of the parent task
	int       cached;            // Flag stating whether the item has been cached and thus has to be uncached

	int       sync;              // Number of global memory synchronizations
	int       rayPackets;        // Index into the LUT for child generation

	int       nextRay;           // Current ray index in global buffer.
	int       rayCount;          // Number of rays in the local pool.
	int       popCount;          // Number of subtasks poped from gmem
	int       popStart;          // Starting subtask poped from gmem
	int       popSubtask;        // Current subtask
	int       popTaskIdx;        // Index of the current subtask
};
#define TASK_GLOBAL_KDTREE 25 // Marks end position of task data loaded from global memory

//------------------------------------------------------------------------

// A work queue divided into two arrays with same indexing and other auxiliary global data
struct TaskStackKdtree : public TaskStackBase
{
	TaskKdtree   *tasks;               // Holds task data
	int          nodeTop;              // Top of the node array
	int          triTop;               // Top of the triangle array
	unsigned int numSortedTris;        // Number of inner nodes emited
	unsigned int numNodes;             // Number of inner nodes emited
	unsigned int numLeaves;            // Number of leaves emited
	unsigned int numEmptyLeaves;       // Number of leaves emited
	unsigned int numAllocations;       // Number of allocations (same as deallocations)
	float        allocSum;             // Sum of allocation sizes
	float        allocSumSquare;        // Sum of squared allocation sizes
	int          warpCounter;          // Work counter for persistent threads.
};

//------------------------------------------------------------------------

// A structure holding statistics for each split
struct SplitDataTri
{
	int    tf;                 // Number of triangles in front of plane
	int    tb;                 // Number of triangles behind the plane
};

// A structure holding statistics for each splited task
struct SplitInfoTri
{
	SplitDataTri splits[PLANE_COUNT]; // Split info for each tested plane (WARP_SIZE planes)
};

#ifdef __CUDACC__
//------------------------------------------------------------------------
// Common data items.
//------------------------------------------------------------------------

// Heap data
__device__ char* g_heapBase2; // The base pointer to a second the heap (for combination of allocation strategies)

//------------------------------------------------------------------------
// Root memory allocations.
//------------------------------------------------------------------------


// Allocate memory for the root node
extern "C" __global__ void allocFreeableMemory(int numTris, int numRays);

// Deallocate memory for the root node
extern "C" __global__ void deallocFreeableMemory();

// Copy data for the root node from CPU allocated to GPU allocated device space.
extern "C" __global__ void MemCpyIndex(CUdeviceptr src, int ofs, int size);

#include "alloc_common.cu"

//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------

__constant__ KernelInputKdtree c_kdtree_in;   // Input of build() for Kd-tree builder.

extern "C" __global__ void build(void);        // Launched for each batch of rays.

//------------------------------------------------------------------------
// Task queue globals.
//------------------------------------------------------------------------

__device__ TaskStackKdtree g_taskStackKdtree;   // Task queue variable
__device__ SplitInfoTri *g_splitStack;   // Split stack head
__device__ CudaKdtreeNode *g_kdtree;   // Split stack head

//------------------------------------------------------------------------
#endif
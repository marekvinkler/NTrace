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
#include "CudaPool.hpp"
#include "rt_common.cuh"

struct KernelInputBVH
{
	int             numTris;        // Total number of tris.
	CUdeviceptr     tris;
	CUdeviceptr     trisIndex;      // Triangle index remapping table.
	CUdeviceptr     trisBox;        // Triangle bounding boxes

	CUdeviceptr     ppsTrisBuf;
	CUdeviceptr     ppsTrisIndex;
	CUdeviceptr		sortTris;

	CUdeviceptr		trisOut;
	CUdeviceptr		trisIndexOut;
	
	CUdeviceptr		debug;
};

//------------------------------------------------------------------------
// Task stack types
//------------------------------------------------------------------------

// BEWARE THAT STRUCT ALIGNMENT MAY INCREASE THE DATA SIZE AND BREAK 1 INSTRUCTION LOAD/STORE
// CURRENT ORDER ENSURES THAT splitPlane AND bbox ARE ALIGNED TO 16B
struct __align__(128) TaskBVH
{
#ifndef MALLOC_SCRATCHPAD
    int       unfinished;        // Counts the number of unfinished sub tasks
    int       type;              // Type of work to be done
	int       depend1;           // Index to the first task dependent on this one, -1 is pointer to TaskStack::unfinished, -2 is empty link
	int       depend2;           // Index to the second task dependent on this one, -1 is pointer to TaskStack::unfinished, -2 is empty link
	
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
	int       triIdxCtr;         // Indicator which item to use for reading the triangle indices, odd value means trisIndex even value means sortTris
	int       origSize;			 // Size the task was created with
	int       terminatedBy;
#else
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
#endif

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
#define TASK_GLOBAL 25 // Marks end position of task data loaded from global memory

// A work queue divided into two arrays with same indexing and other auxiliary global data
struct TaskStackBVH : public TaskStackBase
{
	TaskBVH      *tasks;               // Holds task data
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

// Structure for holding information needed for one child
struct ChildData
{
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
	CudaAABB bbox;   // Bounding box of triangles falling inside this bin
#else
	CudaAABBInt bbox;// Bounding box of triangles falling inside this bin
#endif
	int cnt;         // Number of triangles falling inside this bin
	int padding;
};

// A structure for holding a single reducable split plane info
struct SplitRed
{
	ChildData children[2];
};

/*// A structure for holding a single summable bin
struct BinSum
{
	CudaAABB bboxCur;   // Bounding box of triangles falling inside this bin
	int cntCur;         // Number of triangles falling inside this bin
	int cntLeft;        // Number of triangles to the left, inclusive.

	CudaAABB bboxLeft;  // Bounding box of triangles to the left, inclusive.
	int cntRight;       // Number of triangles to the right, inclusive.
	int padding0;

	CudaAABB bboxRight; // Bounding box of triangles to the right, inclusive.
};*/

// Array of SplitRed structures for each warp
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
struct SplitArray
{
	SplitRed splits[NUM_WARPS][PLANE_COUNT]; // Number of warps time number of planes
};
#else
struct SplitArray
{
	SplitRed splits[PLANE_COUNT]; // Number of warps time number of planes
};
#endif

//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------

#ifdef __CUDACC__
extern "C"
{
texture<float4, 1> t_trisAOut;
texture<int,  1>   t_triIndicesOut;
__constant__ KernelInputBVH c_bvh_in;   // Input of trace() for BVH builder.

__global__ void build(void);        // Launched for each batch of rays.

//------------------------------------------------------------------------
// Task queue globals.
//------------------------------------------------------------------------

__device__ TaskStackBVH g_taskStackBVH;   // Task queue variable

//#if SPLIT_TYPE == 3
__device__ SplitInfoTri *g_splitStack;   // Split stack head
//#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
__device__ SplitArray *g_redSplits;   // Split stack head
//#endif

__device__ CudaBVHNode *g_bvh;   // Split stack head
#ifndef INTERLEAVED_LAYOUT
__device__ CudaKdtreeNode *g_kdtree;   // Split stack head
#else
__device__ char *g_kdtree;   // Split stack head
#endif
}
#endif
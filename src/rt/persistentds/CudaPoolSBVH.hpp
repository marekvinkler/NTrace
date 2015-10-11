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

//------------------------------------------------------------------------
// BVH data structures
//------------------------------------------------------------------------

struct KernelInputSBVH
{
	int             numTris;        // Total number of tris.
	CUdeviceptr     tris;			// Vertices
	CUdeviceptr     refs;			// References.

	CUdeviceptr     ppsTrisBuf;
	CUdeviceptr     ppsTrisIndex;
	CUdeviceptr		sortRefs;

	CUdeviceptr		trisOut;
	CUdeviceptr		trisIndexOut;
	
	CUdeviceptr		debug;
};

//------------------------------------------------------------------------

// BEWARE THAT STRUCT ALIGNMENT MAY INCREASE THE DATA SIZE AND BREAK 1 INSTRUCTION LOAD/STORE
// CURRENT ORDER ENSURES THAT splitPlane AND bbox ARE ALIGNED TO 16B
struct __align__(128) TaskBVH
{
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
#define TASK_GLOBAL_BVH 25 // Marks end position of task data loaded from global memory

//------------------------------------------------------------------------

// A work queue divided into two arrays with same indexing and other auxiliary global data
struct TaskStackBVH : public TaskStackBase
{
	TaskBVH      *tasks;               // Holds task data
	int          nodeTop;              // Top of the node array
	int          triTop;               // Top of the triangle array
	unsigned int numSortedTris;        // Number of inner nodes emited
	unsigned int numNodes;             // Number of inner nodes emited
	unsigned int numLeaves;            // Number of leaves emited
	int          warpCounter;          // Work counter for persistent threads.
};

//------------------------------------------------------------------------

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

// Array of SplitRed structures for each warp
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
struct SplitArray
{
	SplitRed splits[NUM_WARPS][PLANE_COUNT]; // Number of warps time number of planes
};
#else
struct SplitArray
{
	SplitRed splits[PLANE_COUNT]; // Number of planes is enough when using atomics
	SplitRed spatialSplits[PLANE_COUNT]; // Same here :)
};
#endif

#ifdef __CUDACC__
//------------------------------------------------------------------------
// Globals.
//------------------------------------------------------------------------

__constant__ KernelInputSBVH c_bvh_in;   // Input of build() for BVH builder.

extern "C" __global__ void build(void);        // Launched for each batch of rays.

//------------------------------------------------------------------------
// Task queue globals.
//------------------------------------------------------------------------

__device__ TaskStackBVH g_taskStackBVH;   // Task queue variable
__device__ SplitArray *g_redSplits;   // Split stack head
__device__ CudaBVHNode *g_bvh;   // Split stack head
#endif

//------------------------------------------------------------------------
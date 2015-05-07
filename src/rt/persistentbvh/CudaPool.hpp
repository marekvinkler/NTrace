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
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "CudaTracerDefines.h"
#ifdef __CUDACC__
#include <assert.h>
#include <stdio.h>
#include "../../helper_math.h"
#include <math_constants.h>
#endif

//------------------------------------------------------------------------
// Generic macros
//------------------------------------------------------------------------

#define WARP_SIZE 32
#define STR_VALUE(arg) #arg
#define STR(arg) STR_VALUE(arg)

//------------------------------------------------------------------------
// Debugging macros
//------------------------------------------------------------------------

//#define KEEP_ALL_TASKS // Each task is saved to a different location so that they are all visible afterwards
//#define DEBUG_INFO // Enables gathering of statistics for each task
#define BVH_COUNT_NODES // Enables gathering of emited number of nodes
//#define ONE_WARP_RUN
//#define DEBUG_PPS
#define TRAVERSAL_TEST
//#define LEAF_HISTOGRAM
//#define ONDEMAND_FULL_BUILD // Build entire data structure in the ondemand kernel - for overhead timing purpouses

#define PARALLELISM_TEST -1 // -1 Means off
							// 0 Means subtract before potentional add
						    // 1 Means potentional add before subtract

//#define DIVERGENCE_TEST // Test whether all threads in a warp run in a lockstep

//#define SNAPSHOT_POOL 10000
//#define SNAPSHOT_WARP 3000

//#define COUNT_STEPS_LEFT
//#define COUNT_STEPS_RIGHT
//#define COUNT_STEPS_CACHE
//#define COUNT_STEPS_DEQUEUE
//#define COUNT_STEPS_ALLOC

//#define CUTOFF_DEPTH // Ends computation at level CUTOFF_DEPTH

#define WAIT_COUNT 100000 // How long should a warp wait before it assumes it is deadlocked

//------------------------------------------------------------------------
// Optimizations macros
//------------------------------------------------------------------------

//#define CLIP_INTERSECT // Clip before intersections
#define WOOP_TRIANGLES // Save triangles in woop representation
#define COMPACT_LAYOUT // Save triangle pointers immediately in parent
//#define INTERLEAVED_LAYOUT // Save nodes, triangles and references in a single array
#define DUPLICATE_REFERENCES

//#define MALLOC_SCRATCHPAD // Allocate and dealocate memory for auxilliary arrays with malloc
#ifdef MALLOC_SCRATCHPAD
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

#else
#define MALLOC_TYPE -1
#endif // MALLOC_SCRATCHPAD

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

#define ACTIVE_MAX (32*WARP_SIZE-1)
#define EMPTY_MAX (32*WARP_SIZE-1)

//------------------------------------------------------------------------
// Safety conditions
//------------------------------------------------------------------------

#if SCAN_TYPE == 2 && AABB_TYPE < 3
#error Conflicting use of c_bvh_in.sortTris
#endif

//------------------------------------------------------------------------
// Auxiliary types
//------------------------------------------------------------------------

// Enum for threshold values in header array
enum TaskHeader {TaskHeader_Empty = 0xA0000000, TaskHeader_Waiting = 0xC0000000, TaskHeader_Dependent = 0xE0000000, TaskHeader_Locked = -1, TaskHeader_Active = 0};
// Enum for various locking states
enum LockType {LockType_Free = 0, LockType_Set, LockType_None, LockType_Subtask};
// Enum for types of dependence
enum DependType {DependType_None = -2, DependType_Root};

// A work queue divided into two arrays with same indexing and other auxiliary global data
struct TaskStackBase
{
	int          *header;              // Holds state of each task
	int          top;                  // Top of the stack, points to the last filled element
	int          bottom;               // Bottom of the stack, points to the lowest empty element
	int          active[ACTIVE_MAX+1]; // Locations of the last used active elements
	unsigned int activeTop;            // Top of active stack
	unsigned int activeBottom;         // Bottom of active stack
	int          empty[EMPTY_MAX+1];   // Locations of the last opened elements
	unsigned int emptyTop;             // Top of empty stack
	unsigned int emptyBottom;          // Bottom of empty stack
	int          unfinished;           // Number of tasks that need to be finished before all work is done - NOT THE SAME AS THE NUMBER OF TASKS IN THE ARRAY
	int          sizePool;             // Size of the pool - FOR TESTING PURPOUSES ONLY
	int          sizeNodes;            // Size of the node memory - FOR TESTING PURPOUSES ONLY
	int          sizeTris;             // Size of the triangle memory - FOR TESTING PURPOUSES ONLY
	unsigned int leafHist[32];         // Histogram of leaf sizes
	int          launchFlag;
};

#ifdef SNAPSHOT_POOL
// A structure holding information about the state of the pool
struct PoolInfo
{
	int pool;
	int tasks;
	int active;
	int chunks;
	float depth;
	long long int clockStart;
	long long int clockEnd;
};
#endif

//------------------------------------------------------------------------

#ifdef SNAPSHOT_WARP
// A structure holding information about tasks processed by a warp
struct WarpInfo
{
	int reads;
	int rays;
	int tris;
	int type;
	int chunks;
	int popCount;
	int depth;
	int stackTop;
	long long int clockSearch;
	long long int clockDequeue;
	long long int clockFinished;
};
#endif

//------------------------------------------------------------------------

#ifdef __CUDACC__
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 200)
#undef assert
#define assert(arg)
#define printf(f, ...) ((void)(f, __VA_ARGS__),0)
#endif

//------------------------------------------------------------------------

#ifdef DIVERGENCE_TEST
#if 0 // Ends on first divergence, cannot cause unknown error later
	#define ASSERT_DIVERGENCE(S, T) \
	{ \
		assert(__ballot(1) == 0xFFFFFFFF); \
	}
#else // Writes out all encountered divergences
	#define ASSERT_DIVERGENCE(S, T) \
	{ \
		unsigned int ballot = __ballot(1); \
		if(ballot != 0xFFFFFFFF) \
		{\
			printf("Divergence %s tid %d, %u!\n", S, T, ballot); \
		}\
	}
#endif
#else
	#define ASSERT_DIVERGENCE(S, T) ((void)0)
#endif

//------------------------------------------------------------------------

// Number of step counts
__shared__ volatile int maxSteps[NUM_WARPS_PER_BLOCK];
__shared__ volatile int sumSteps[NUM_WARPS_PER_BLOCK];
__shared__ volatile int numSteps[NUM_WARPS_PER_BLOCK];
__shared__ volatile int numReads[NUM_WARPS_PER_BLOCK];
__shared__ volatile int numRestarts[NUM_WARPS_PER_BLOCK];

#ifdef SNAPSHOT_WARP
#undef COUNT_STEPS_LEFT
#undef COUNT_STEPS_RIGHT
#undef COUNT_STEPS_CACHE
#undef COUNT_STEPS_DEQUEUE
#define COUNT_STEPS_DEQUEUE
#endif

#ifdef DEBUG_PPS
#undef PHASE_TEST
#define PHASE_TEST
#endif


//------------------------------------------------------------------------
// Common data items.
//------------------------------------------------------------------------


#if PARALLELISM_TEST >= 0
__device__ int g_numActive;
#endif

#ifdef SNAPSHOT_POOL
__device__ PoolInfo *g_snapshots;
#endif

#ifdef SNAPSHOT_WARP
__device__ WarpInfo *g_snapshots;
#endif

// Heap data
#ifdef MALLOC_SCRATCHPAD
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
#endif


//------------------------------------------------------------------------
// Functions for computing number of warps needed for various tasks.
//------------------------------------------------------------------------


// Return the number of warp sized subtasks but at least 1 task
__device__ __forceinline__ int taskWarpSubtasks(int threads);

// Return the number of warp sized subtasks
__device__ __forceinline__ int taskWarpSubtasksZero(int threads);

// Computes the number of warps needed for the naive PPS
__device__ __forceinline__ int taskNextPhaseCountPPS(int step, int start, int end);

// Computes the number of levels in the tree defined by interval [start, end) and ceils the value to the multiple of LOG_WARP_SIZE
__device__ __forceinline__ int taskTopTreeLevel(int start, int end);

// Computes the number of warps needed for a specific level of a hierarchical algorithm
__device__ __forceinline__ int taskNextPhaseCountTree(int level, int start, int end);

// Computes the number of warps needed for the PPS up phase
__device__ __forceinline__ int taskNextPhaseCountPPSUp(int step, int start, int end);

// Computes the number of warps needed for the PPS down phase
__device__ __forceinline__ int taskNextPhaseCountPPSDown(int step, int start, int end);

// Computes the number of warps needed for the reduction level by level
__device__ __forceinline__ int taskNextPhaseCountReduce(int step, int start, int end);

// Computes the number of warps needed for the reduction multiplied by a constant
__device__ __forceinline__ int taskNextPhaseCountReduceMultiplied(int step, int start, int end, int multiply);

// Computes the number of warps needed for the reduction
__device__ __forceinline__ int taskNextPhaseCountReduceBlock(int step, int start, int end);

//------------------------------------------------------------------------
// Support for floating point minimums and maximums.
//------------------------------------------------------------------------


// Converts float to an orderable int
__device__ __forceinline__ int floatToOrderedInt(float floatVal);

// Converts an orderable int back to a float
__device__ __forceinline__ float orderedIntToFloat(int intVal);

// Atomic minimum on floats
__device__ __forceinline__ float atomicMinFloat(int* address, float value);

// Atomic maximum on floats
__device__ __forceinline__ float atomicMaxFloat(int* address, float value);


//------------------------------------------------------------------------
// Task enqueue and dequeue support.
//------------------------------------------------------------------------


// Adds a task into a global task queue in place of an empty task with position lower or equal of taskIdx and returns its position in taskIdx
__device__ void taskEnqueueRight(int tid, int *head, volatile int* sharedData, int taskStatus, int& beg, int end);

// Adds a task into a global task queue in place of an empty task with position lower or equal of taskIdx and returns its position in taskIdx
__device__ void taskEnqueueLeft(int tid, int *head, volatile int* sharedData, int taskStatus, int& beg, int* unfinished, int size);

// Adds a task into a global task queue in place of an empty task
__device__ void taskEnqueueCache(int tid, TaskStackBase* task, volatile int* s_sharedData, int& status, int& pos, int& beg, int& top);

// Checks whether there is more work chunks to process
__device__ __forceinline__ int reduceSubtask(int &subtask, const int& count, const int& popCount);

// Caches an item in the active cache
__device__ __forceinline__ void taskCacheActive(int taskIdx, int* activeCache, unsigned int* activeCacheTop);

// Uncaches an item in the active cache
__device__ __forceinline__ void taskUncacheActive(int tid, int taskIdx, int* activeCache, unsigned int* activeCacheTop);

// Caches an item in the empty cache
__device__ __forceinline__ void taskCacheEmpty(int taskIdx, int* emptyCache, unsigned int* emptyCacheTop);


//------------------------------------------------------------------------
// Prefix scan and sorting.
//------------------------------------------------------------------------


// Naive inclusive prefix scan
template<typename T>
__device__ void pps(int tid, int subtask, int step, T* dataSrc, T* dataBuff, volatile T* red, unsigned int start, unsigned int end, int test);

// Inclusive scan by Harris: up-sweep phase
template<typename T>
__device__ void scanUp(int tid, int subtask, int step, T* data, T* cls, volatile T* red, int start, int end, T test, T(*op)(T,T), T identity);

// Inclusive scan by Harris: down-sweep phase
template<typename T>
__device__ void scanDown(int tid, int subtask, int step, T* data, T* cls, volatile T* red, int start, int end, T test, T(*op)(T,T), T identity);

// Comments and variables are with regard to -1,0 to the left, 1 to the right
__device__ void sort(int tid, int subtask, int step, int* dataPPSSrc, int* dataPPSBuf, int* dataSort, int* dataIndex, int start, int end, int mid, int test, bool swapIndex);

#endif
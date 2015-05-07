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
    BVH building specialization of the framework.

    "Massively Parallel Hierarchical Scene Sorting with Applications in Rendering",
    Marek Vinkler, Michal Hapala, Jiri Bittner and Vlastimil Havran,
    Computer Graphics Forum 2012
*/

#include "rt_common.cu"
#include "CudaBuilderKernels.hpp"
#include "allocators.cu"

//------------------------------------------------------------------------
// Shared variables.
//------------------------------------------------------------------------

__shared__ volatile TaskBVH s_task[NUM_WARPS_PER_BLOCK]; // Memory holding information about the currently processed task
__shared__ volatile TaskBVH s_newTask[NUM_WARPS_PER_BLOCK]; // Memory for the new task to be created in
__shared__ volatile int s_sharedData[NUM_WARPS_PER_BLOCK][WARP_SIZE]; // Shared memory for inside warp use
__shared__ volatile int s_owner[NUM_WARPS_PER_BLOCK][WARP_SIZE]; // Another shared pool, not necessary for Kdtree


//------------------------------------------------------------------------
// Function Headers.
//------------------------------------------------------------------------

// Loading and saving functions
__device__ __forceinline__ void taskLoadFirstFromGMEM(int tid, int taskIdx, volatile TaskBVH& task);
__device__ __forceinline__ void taskLoadSecondFromGMEM(int tid, int taskIdx, volatile TaskBVH& task);
__device__ __forceinline__ void taskSaveFirstToGMEM(int tid, int taskIdx, const volatile TaskBVH& task);
__device__ __forceinline__ void taskSaveSecondToGMEM(int tid, int taskIdx, const volatile TaskBVH& task);

//------------------------------------------------------------------------

// Copies first cache line of the Task taskIdx to task
__device__ __forceinline__ void taskLoadFirstFromGMEM(int tid, int taskIdx, volatile TaskBVH& task)
{
	ASSERT_DIVERGENCE("taskLoadFirstFromGMEM top", tid);
	volatile int* taskAddr = (volatile int*)(&task);
	TaskBVH* g_task = &g_taskStackBVH.tasks[taskIdx];
	taskAddr[tid] = ((int*)g_task)[tid]; // Every thread copies one word of task data
	ASSERT_DIVERGENCE("taskLoadFirstFromGMEM bottom", tid);

#ifdef DEBUG_INFO
	taskLoadSecondFromGMEM(tid, taskIdx, task); // Save the debug info statistics as well
#endif
}

// Copies second cache line of the Task taskIdx to task
__device__ __forceinline__ void taskLoadSecondFromGMEM(int tid, int taskIdx, volatile TaskBVH& task)
{
	ASSERT_DIVERGENCE("taskLoadSecondFromGMEM top", tid);
	volatile int* taskAddr = (volatile int*)(&task);
	TaskBVH* g_task = &g_taskStackBVH.tasks[taskIdx];
	int offset = 128/sizeof(int); // 128B offset
	if(tid < TASK_GLOBAL_KDTREE) // Prevent overwriting local data saved in task
		taskAddr[tid+offset] = ((int*)g_task)[tid+offset]; // Every thread copies one word of task data
	ASSERT_DIVERGENCE("taskLoadSecondFromGMEM bottom", tid);
}

//------------------------------------------------------------------------

// Copies first cache line of the task to Task taskIdx
__device__ __forceinline__ void taskSaveFirstToGMEM(int tid, int taskIdx, const volatile TaskBVH& task)
{
	ASSERT_DIVERGENCE("taskSaveFirstToGMEM top", tid);
	// Copy the data to global memory
	int* taskAddr = (int*)(&g_taskStackBVH.tasks[taskIdx]);
	taskAddr[tid] = ((const volatile int*)&task)[tid]; // Every thread copies one word of data of its task
	ASSERT_DIVERGENCE("taskSaveFirstToGMEM bottom", tid);

#ifdef DEBUG_INFO
	taskSaveSecondToGMEM(tid, taskIdx, task); // Save the debug info statistics as well
#endif
}

// Copies second cache line of the task to Task taskIdx
__device__ __forceinline__ void taskSaveSecondToGMEM(int tid, int taskIdx, const volatile TaskBVH& task)
{
	ASSERT_DIVERGENCE("taskSaveSecondToGMEM top", tid);
	// Copy the data to global memory
	int* taskAddr = (int*)(&g_taskStackBVH.tasks[taskIdx]);
	int offset = 128/sizeof(int); // 128B offset
	taskAddr[tid+offset] = ((const volatile int*)&task)[tid+offset]; // Every thread copies one word of data of its task
	ASSERT_DIVERGENCE("taskSaveSecondToGMEM bottom", tid);
}

//------------------------------------------------------------------------

__device__ __forceinline__ int taskPopCount(int status)
{
	//float nTris = c_bvh_in.numTris;

	//return 1;
	//return (int)((float)nTris/((float)WARP_SIZE*NUM_WARPS*c_env.granularity))+1;
	return c_env.popCount;
}

//------------------------------------------------------------------------

__device__ __forceinline__ int* getPPSTrisPtr(int dynamicMemory, int tris)
{
#ifndef MALLOC_SCRATCHPAD
	return (int*)c_bvh_in.ppsTrisBuf;
#else
	return ((int*)(g_heapBase + dynamicMemory))+tris;
	//return ((int*)(g_heapBase + dynamicMemory))+c_bvh_in.numTris;
#endif
}

//------------------------------------------------------------------------

__device__ __forceinline__ int* getPPSTrisIdxPtr(int dynamicMemory, int tris)
{
#ifndef MALLOC_SCRATCHPAD
	return (int*)c_bvh_in.ppsTrisIndex;
#else
	return ((int*)(g_heapBase + dynamicMemory))+2*tris;
	//return ((int*)(g_heapBase + dynamicMemory))+2*c_bvh_in.numTris;
#endif
}

//------------------------------------------------------------------------

__device__ __forceinline__ int* getSortTrisPtr(int dynamicMemory, int tris)
{
#ifndef MALLOC_SCRATCHPAD
	return (int*)c_bvh_in.sortTris;
#else
	return ((int*)(g_heapBase + dynamicMemory))+3*tris;
	//return ((int*)(g_heapBase + dynamicMemory))+3*c_bvh_in.numTris;
#endif
}

//------------------------------------------------------------------------

__device__ __forceinline__ int* getTriIdxPtr(int dynamicMemory, int tris)
{
#ifndef MALLOC_SCRATCHPAD
#if SCAN_TYPE < 2
	return (int*)c_bvh_in.trisIndex;
#else
	if((triIdxCtr % 2) == 0)
		return (int*)c_bvh_in.trisIndex;
	else
		return (int*)c_bvh_in.sortTris;
#endif
#else
#if SCAN_TYPE < 2
	return (int*)(g_heapBase + dynamicMemory);
#else
	/*if((triIdxCtr % 2) == 0)
		return (int*)(g_heapBase + dynamicMemory);
	else
		return ((int*)(g_heapBase + dynamicMemory))+3*tris;
		//return ((int*)(g_heapBase + dynamicMemory))+3*c_bvh_in.numTris;*/

#if ((MALLOC_TYPE == CMALLOC) || (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC_FUSED)) && defined(CIRCULAR_MALLOC_WITH_SCATTER_ALLOC)
	if(dynamicMemory <= 0)
		return (int*)(g_heapBase2 - dynamicMemory);
	else
		return (int*)(g_heapBase + dynamicMemory);

#elif (MALLOC_TYPE == HALLOC)
	if(dynamicMemory < 0)
		return (int*)(g_heapBase + dynamicMemory);
	else
		return (int*)(g_heapBase2 + dynamicMemory);

#else
	return (int*)(g_heapBase + dynamicMemory);
#endif // MALLOC_TYPE
#endif
#endif
}

//------------------------------------------------------------------------

__device__ __forceinline__ int allocBuffers(int tris)
{
	// Allocators do align memory automatically 
#if SCAN_TYPE < 2
	uint allocSize = 4*tris*sizeof(int);
#else
	uint allocSize = tris*sizeof(int);
#endif

#if (MALLOC_TYPE == CUDA_MALLOC)
	void* alloc = mallocCudaMalloc(allocSize);
#elif (MALLOC_TYPE == ATOMIC_MALLOC)
	void* alloc = mallocAtomicMalloc(allocSize);
#elif (MALLOC_TYPE == ATOMIC_MALLOC_CIRCULAR)
	void* alloc = mallocAtomicMallocCircular(allocSize);
#elif (MALLOC_TYPE == CIRCULAR_MALLOC)
	void* alloc = mallocCircularMalloc(allocSize);
#elif (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED)
	void* alloc = mallocCircularMallocFused(allocSize);
#elif (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC)
	void* alloc = mallocCircularMultiMalloc(allocSize);
#elif (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC_FUSED)
	void* alloc = mallocCircularMultiMallocFused(allocSize);
#elif (MALLOC_TYPE == SCATTER_ALLOC)
	void* alloc = mallocScatterAlloc(allocSize);
#elif (MALLOC_TYPE == HALLOC)
	void* alloc = mallocHalloc(allocSize);
#endif

#ifdef CHECK_OUT_OF_MEMORY
	if(alloc == NULL)
	{
		printf("Out of memory!\n");
		g_taskStackBVH.unfinished = 1;
	}
	//if((CUdeviceptr)alloc < g_heapBase)
	//	printf("Incorrect base ptr!\n");
#endif

#ifdef BVH_COUNT_NODES
		atomicAdd(&g_taskStackBVH.numAllocations, 1);
		atomicAdd(&g_taskStackBVH.allocSum, allocSize);
		atomicAdd(&g_taskStackBVH.allocSumSquare, allocSize*allocSize);
#endif

#if (MALLOC_TYPE == HALLOC)
	if(allocSize <= MAX_BLOCK_SZ) // Specific to Halloc
		return ((char*)alloc) - g_heapBase2;
	else
		return ((char*)alloc) - g_heapBase;
#else
	return ((char*)alloc) - g_heapBase;
#endif
}

//------------------------------------------------------------------------

__device__ __forceinline__ void freeBuffers(int dynamicMemory, int size)
{
#ifndef NO_FREE
#if (MALLOC_TYPE == CUDA_MALLOC)
	freeCudaMalloc((void*)(g_heapBase+dynamicMemory));
#elif (MALLOC_TYPE == CIRCULAR_MALLOC) || (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC_FUSED)
#ifdef CIRCULAR_MALLOC_WITH_SCATTER_ALLOC
	if(dynamicMemory <= 0)
	{
		freeScatterAlloc((void*)(g_heapBase2-dynamicMemory));
		return;
	}
#endif

#if (MALLOC_TYPE == CIRCULAR_MALLOC)
	freeCircularMalloc((void*)(g_heapBase+dynamicMemory));
#elif (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED)
	freeCircularMallocFused((void*)(g_heapBase+dynamicMemory));
#elif (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC)
	freeCircularMultiMalloc((void*)(g_heapBase+dynamicMemory));
#elif (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC_FUSED)
	freeCircularMultiMallocFused((void*)(g_heapBase+dynamicMemory));
#endif
#elif (MALLOC_TYPE == SCATTER_ALLOC)
	freeScatterAlloc((void*)(g_heapBase+dynamicMemory));
#elif (MALLOC_TYPE == HALLOC)
	if(dynamicMemory < 0)
		freeHalloc((void*)(g_heapBase + dynamicMemory));
	else
		freeHalloc((void*)(g_heapBase2 + dynamicMemory));
#endif
#endif // NO_FREE
}

//------------------------------------------------------------------------

__device__ __forceinline__ void allocChildren(volatile int& leftMem, volatile int& rightMem, int cntLeft, int cntRight)
{
#if (MALLOC_TYPE == ATOMIC_MALLOC) || (MALLOC_TYPE == ATOMIC_MALLOC_CIRCULAR) // Memory can be allocated as a single chunk because there is no pointer related free
	if(cntLeft == 0 || cntRight == 0) // One leaf is empty, use parent memory with no copy
		leftMem = s_task[threadIdx.y].dynamicMemory;
	else
		leftMem = allocBuffers(cntLeft+cntRight);
#if SCAN_TYPE < 2
	int allocSize = 4*cntLeft*sizeof(int);
#else
	int allocSize = cntLeft*sizeof(int);
#endif
	rightMem = leftMem+allocSize;
#elif (MALLOC_TYPE == CIRCULAR_MALLOC) || (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC_FUSED)
	int leftSize = cntLeft*sizeof(int); // Requested memory for left child
	int rightSize = cntRight*sizeof(int); // Requested memory for right child

	if(cntLeft != 0)
	{
#ifdef CIRCULAR_MALLOC_WITH_SCATTER_ALLOC
		if(leftSize <= CIRCULAR_MALLOC_SWITCH_SIZE)
		{
			void* alloc = mallocScatterAlloc(leftSize);
#ifdef CHECK_OUT_OF_MEMORY
			if(alloc == NULL)
			{
				printf("Out of memory!\n");
				g_taskStackBVH.unfinished = 1;
			}
#endif
			leftMem = -(((char*)alloc) - g_heapBase2);
		}
		else
#endif
			if(cntRight == 0) // One leaf is empty, use parent memory with no copy
				leftMem = s_task[threadIdx.y].dynamicMemory;
			else
				leftMem = allocBuffers(cntLeft);
	}
	
	if(cntRight != 0)
	{
#ifdef CIRCULAR_MALLOC_WITH_SCATTER_ALLOC
		if(rightSize <= CIRCULAR_MALLOC_SWITCH_SIZE)
		{
			void* alloc = mallocScatterAlloc(rightSize);
#ifdef CHECK_OUT_OF_MEMORY
			if(alloc == NULL)
			{
				printf("Out of memory!\n");
				g_taskStackBVH.unfinished = 1;
			}
#endif
			rightMem = -(((char*)alloc) - g_heapBase2);
		}
		else
#endif
			if(cntLeft == 0) // One leaf is empty, use parent memory with no copy
				rightMem = s_task[threadIdx.y].dynamicMemory;
			else
				rightMem = allocBuffers(cntRight);
	}
#else
	if(cntLeft != 0)
	{
		if(cntRight == 0) // One leaf is empty, use parent memory with no copy
			leftMem = s_task[threadIdx.y].dynamicMemory;
		else
			leftMem = allocBuffers(cntLeft);
	}
	if(cntRight != 0)
	{
		if(cntLeft == 0) // One leaf is empty, use parent memory with no copy
			rightMem = s_task[threadIdx.y].dynamicMemory;
		else
			rightMem = allocBuffers(cntRight);
	}
#endif
}

//------------------------------------------------------------------------
__device__ __forceinline__ void backcopy(int tid, int triIdxCtr, int triStart, int triEnd)
{
	int* curMem = getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart);
#ifndef MALLOC_SCRATCHPAD
	if(curMem != (int*)c_bvh_in.trisIndex)
#else
#endif
	{
		int* inIdx = curMem + triStart + tid;
		int* outIdx = (int*)c_bvh_in.trisIndex + triStart + tid;

		for(int i = triStart + tid; i < triEnd; i += WARP_SIZE)
		{
			*outIdx = *inIdx;
			inIdx += WARP_SIZE;
			outIdx += WARP_SIZE;
		}
	}
}

//------------------------------------------------------------------------

// Prepares the processing of the next phase
__device__ __forceinline__ void taskPrepareNext(int tid, int taskIdx, int phase)
{
#ifdef PHASE_TEST
	//int endAfter = TaskType_Sort_PPS1;
	//int endAfter = TaskType_Sort_PPS1_Up;
	//int endAfter = TaskType_Sort_PPS1_Down;
	//int endAfter = TaskType_Sort_SORT1;
	//int endAfter = TaskType_AABB_Min;
	//int endAfter = TaskType_AABB;
	//int endAfter = TaskType_InitMemory;
	//int endAfter = TaskType_BinTriangles;
	//int endAfter = TaskType_ReduceBins;
	int endAfter = TaskType_Max;

	if(phase == endAfter && s_task[threadIdx.y].type != phase)
	{
#ifdef KEEP_ALL_TASKS
		taskSaveFirstToGMEM(tid, taskIdx, s_task[threadIdx.y]);
#endif
		g_taskStackBVH.unfinished = 1;
	}
#endif

	int popCount = taskPopCount(s_task[threadIdx.y].unfinished);
	if(s_task[threadIdx.y].unfinished <= popCount*POP_MULTIPLIER) // No need to share through global memory
	{
#ifdef PHASE_TEST
		if(phase == endAfter && s_task[threadIdx.y].type != phase)
		{
			s_task[threadIdx.y].lock = LockType_Free;
		}
		else
#endif
		{
			s_task[threadIdx.y].origSize = s_task[threadIdx.y].unfinished;
			s_task[threadIdx.y].lock = LockType_None;
			s_task[threadIdx.y].popCount = s_task[threadIdx.y].unfinished;
		}
	}
	else
	{
#ifdef DEBUG_INFO
		if(tid == 1)
			s_task[threadIdx.y].sync++;
#endif

		s_task[threadIdx.y].lock = LockType_Free;
		s_task[threadIdx.y].origSize = s_task[threadIdx.y].unfinished;

		// Copy the data to global memory
		taskSaveFirstToGMEM(tid, taskIdx, s_task[threadIdx.y]);
		__threadfence(); // Make sure task is updated in global memory before we unlock it

#ifdef PHASE_TEST
		if(phase == endAfter && s_task[threadIdx.y].type != phase)
		{
			g_taskStackBVH.header[taskIdx] = TaskHeader_Locked;
		}
		else
#endif
		{
			// Take some work for this warp
			s_task[threadIdx.y].lock = LockType_Subtask;
			s_task[threadIdx.y].popCount = popCount;
			s_task[threadIdx.y].unfinished -= popCount;

			g_taskStackBVH.header[taskIdx] = s_task[threadIdx.y].unfinished; // Restart this task
		}
	}
}

//------------------------------------------------------------------------

// Checks whether the task is finished
__device__ __forceinline__ bool taskCheckFinished(int tid, int taskIdx, int countDown)
{
	if(tid == 0)
	{
		if(s_task[threadIdx.y].lock == LockType_None)
			s_sharedData[threadIdx.y][0] = countDown;
		else
			s_sharedData[threadIdx.y][0] = atomicSub(&g_taskStackBVH.tasks[taskIdx].unfinished, countDown); // Lower the number of unfinished tasks
	}

	ASSERT_DIVERGENCE("taskCheckFinished top", tid);

	return s_sharedData[threadIdx.y][0] == countDown; // Finished is the value before Dec, thus == countDown means last. We have finished the task and are responsible for cleaning up
}

//------------------------------------------------------------------------

// Decides what type of task should be created
__device__ bool taskTerminationCriteria(int tris, int trisLeft, int trisRight, volatile CudaAABB& bbox, volatile float4& splitPlane, int& termCrit, bool& leftLeaf, bool& rightLeaf)
{
	//if(s_task[threadIdx.y].depth < 2) // Unknown error if we try to split an empty task
	//	return false;

	float areaLeft, areaRight, areaParent;
	int dim = getPlaneDimension(*(float4*)&splitPlane);
	switch(dim)
	{
	case 0:
		areaAABBX(bbox, splitPlane.w, areaLeft, areaRight);
		break;
	case 1:
		areaAABBY(bbox, splitPlane.w, areaLeft, areaRight);
		break;
	case 2:
		areaAABBZ(bbox, splitPlane.w, areaLeft, areaRight);
		break;
	}
	areaParent = areaAABB(bbox);
	
	float leafCost, leftCost, rightCost;
	// Evaluate if the termination criteria are met
	leafCost = c_env.optCi * (float)tris;
	leftCost = areaLeft*(float)trisLeft;
	rightCost = areaRight*(float)trisRight;
	float subdivisionCost = c_env.optCt + c_env.optCi*(leftCost + rightCost)/areaParent;
/*#if defined(SAH_TERMINATION) && SPLIT_TYPE != 0
	if(leafCost < subdivisionCost)
	{
		termCrit = TerminatedBy_Cost;
		return true; // Trivial computation
	}
#endif*/

	//if(threadIdx.x == 0 && (float)(trisLeft+trisRight)/(float)tris > 1.5f)
	//	printf("Tris %d Left %d Right %d Duplicates (%d)(%f)\n", tris, trisLeft, trisRight, (trisLeft+trisRight)-tris, (float)(trisLeft+trisRight)/(float)tris);

	//if(s_task[threadIdx.y].dynamicMemoryLeft == 0 && s_task[threadIdx.y].dynamicMemoryRight == 0)
	//	return true;

	leftLeaf = trisLeft <= c_env.triLimit || s_task[threadIdx.y].depth > (c_env.optMaxDepth-2);
	rightLeaf = trisRight <= c_env.triLimit || s_task[threadIdx.y].depth > (c_env.optMaxDepth-2);
	
	float ratioWork = subdivisionCost/leafCost;
	if(ratioWork > c_env.failRq)
	{
		s_task[threadIdx.y].subFailureCounter++;

		if(s_task[threadIdx.y].subFailureCounter > c_env.failureCount)
		{
			termCrit = TerminatedBy_FailureCounter;
			return true; // Trivial computation
		}
	}

	return false; // Continue subdivision
}

#ifndef INTERLEAVED_LAYOUT
//------------------------------------------------------------------------

#ifndef COMPACT_LAYOUT
// Create node in NoCompactNoWoop layout
__device__ __forceinline__ void taskSaveNodeParent(int tid, int triStart, int triEnd, int numLeft, int numRight, int parentIdx, int nodeIdx, int taskID)
{
	if(tid == 0)
	{
		//printf("Left %d, right %d, depth %d\n", numLeft, numRight, newTask->depth);
		s_sharedData[threadIdx.y][3] = atomicAdd(&g_taskStackBVH.triTop, (triEnd-triStart)*3); // Atomically acquire leaf space
	}

	int triOfs = s_sharedData[threadIdx.y][3];
	int triNum = createKdtreeLeafWoop(tid, triOfs, (float4*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, triStart, triEnd, (float4*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart));

	if(tid == 0)
	{
		g_kdtree[nodeIdx] = make_int4(triNum, triOfs, parentIdx, 0); // Sets the leaf child pointers
	}
}

#else

// Create node in NoCompactNoWoop layout
__device__ __forceinline__ void taskSaveNodeParent(int tid, int triStart, int triEnd, int numLeft, int numRight, int parentIdx, int nodeIdx, int taskID)
{
	if(tid == 0)
	{
		//printf("Left %d, right %d, depth %d\n", numLeft, numRight, newTask->depth);
#ifdef DUPLICATE_REFERENCES
		s_sharedData[threadIdx.y][3] = atomicAdd(&g_taskStackBVH.triTop, (triEnd-triStart)*3+1); // Atomically acquire leaf space, +1 is for the triangle sentinel
#else
		s_sharedData[threadIdx.y][3] = atomicAdd(&g_taskStackBVH.triTop, (triEnd-triStart)+1); // Atomically acquire leaf space, +1 is for the triangle sentinel
#endif
	}

	int triOfs = s_sharedData[threadIdx.y][3];
#ifndef WOOP_TRIANGLES
	int triIdx = createLeaf(tid, triOfs, (float*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, triStart, triEnd, (float*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart)); 
#else
#ifdef DUPLICATE_REFERENCES
	int triIdx = createLeafWoop(tid, triOfs, (float4*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, triStart, triEnd, (float4*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart));
#else
	int triIdx = createLeafReference(tid, triOfs, (int*)c_bvh_in.trisIndexOut, triStart, triEnd, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart));
#endif
#endif

	if(tid == 0)
	{
		taskUpdateParentPtr(g_kdtree, parentIdx, taskID, triIdx); // Mark this node as leaf in the hierarchy
	}
}
#endif

//------------------------------------------------------------------------
#ifndef COMPACT_LAYOUT

// Create left child in NoCompactNoWoop layout
__device__ __forceinline__ void taskSaveNodeLeft(int tid, int leftOfs, int childLeft, int numLeft, int nodeIdx, volatile TaskBVH* newTask)
{
	int triNum = createKdtreeLeafWoop(tid, leftOfs, (float4*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, 0, numLeft, (float4*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemoryLeft, numLeft));
	g_kdtree[childLeft] = make_int4(triNum, leftOfs, nodeIdx, 0); // Sets the leaf child pointers
	s_sharedData[threadIdx.y][0] = triNum;
}

#else

// Create left child in NoCompactNoWoop layout
__device__ __forceinline__ void taskSaveNodeLeft(int tid, int leftOfs, int& childLeft, int numLeft, int nodeIdx, volatile TaskBVH* newTask)
{
	if(numLeft == 0)
	{
		childLeft = KDTREE_EMPTYLEAF;
		s_sharedData[threadIdx.y][0] = childLeft;
	}
	else
	{
#ifndef WOOP_TRIANGLES
		int triIdx = createLeaf(tid, leftOfs, (float*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, 0, numLeft, (float*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, numLeft));
#else
#ifdef DUPLICATE_REFERENCES
		int triIdx = createLeafWoop(tid, leftOfs, (float4*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, 0, numLeft, (float4*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemoryLeft, numLeft));
#else
		int triIdx = createLeafReference(tid, leftOfs, (int*)c_bvh_in.trisIndexOut, 0, numLeft, getTriIdxPtr(s_task[threadIdx.y].dynamicMemoryLeft, numLeft));
#endif
#endif

		childLeft = triIdx;
		s_sharedData[threadIdx.y][0] = childLeft;
	}
}
#endif

//------------------------------------------------------------------------
#ifndef COMPACT_LAYOUT

// Create right child in NoCompactNoWoop layout
__device__ __forceinline__ void taskSaveNodeRight(int tid, int rightOfs, int childRight, int numRight, int nodeIdx, volatile TaskBVH* newTask)
{
	int triNum = createKdtreeLeafWoop(tid, rightOfs, (float4*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, 0, numRight, (float4*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemoryRight, numRight));
	g_kdtree[childRight] = make_int4(triNum, rightOfs, nodeIdx, 0); // Sets the leaf child pointers
	s_sharedData[threadIdx.y][1] = triNum;
}

#else

// Create right child in NoCompactNoWoop layout
__device__ __forceinline__ void taskSaveNodeRight(int tid, int rightOfs, int& childRight, int numRight, int nodeIdx, volatile TaskBVH* newTask)
{
	if(numRight == 0)
	{
		childRight = KDTREE_EMPTYLEAF;
		s_sharedData[threadIdx.y][1] = childRight;
	}
	else
	{
		// OPTIMIZE: Write out both children in one call?
#ifndef WOOP_TRIANGLES
		int triIdx = childRight = createLeaf(tid, rightOfs, (float*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, 0, numRight, (float*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemoryRight, numRight));
#else
#ifdef DUPLICATE_REFERENCES
		int triIdx = createLeafWoop(tid, rightOfs, (float4*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, 0, numRight, (float4*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemoryRight, numRight));
#else
		int triIdx = createLeafReference(tid, rightOfs, (int*)c_bvh_in.trisIndexOut, 0, numRight, getTriIdxPtr(s_task[threadIdx.y].dynamicMemoryRight, numRight));
#endif
#endif

		childRight = triIdx;
		s_sharedData[threadIdx.y][1] = childRight;
	}
}

#endif

//------------------------------------------------------------------------

// Allocate memory for children
__device__ __forceinline__ void taskAllocNodeTris(int tid, bool leftLeaf, bool rightLeaf, int numLeft, int numRight, int& childLeft, int& childRight, int& leftOfs, int& rightOfs)
{
#ifndef COMPACT_LAYOUT
	int innerNodes = 2;
	int leafChunks = ((leftLeaf && numLeft != 0) ? (numLeft*3) : 0) + ((rightLeaf && numRight != 0) ? (numRight*3) : 0); // Atomically acquire leaf space, +1 is for the triangle sentinel
#else
	int innerNodes = (int)(!leftLeaf) + (int)(!rightLeaf);
#ifdef DUPLICATE_REFERENCES
	int leafChunks = ((leftLeaf && numLeft != 0) ? (numLeft*3+1) : 0) + ((rightLeaf && numRight != 0) ? (numRight*3+1) : 0); // Atomically acquire leaf space, +1 is for the triangle sentinel
#else
	int leafChunks = ((leftLeaf && numLeft != 0) ? (numLeft+1) : 0) + ((rightLeaf && numRight != 0) ? (numRight+1) : 0); // Atomically acquire leaf space, +1 is for the triangle sentinel
#endif
#endif
	if(tid == 0)
	{
		if(innerNodes > 0)
			s_sharedData[threadIdx.y][2] = atomicAdd(&g_taskStackBVH.nodeTop, innerNodes); // Inner node -> create new subtasks in the final array
		if(leafChunks > 0)
			s_sharedData[threadIdx.y][3] = atomicAdd(&g_taskStackBVH.triTop, leafChunks); // Atomically acquire leaf space

		// Check if there is enough memory to write the nodes and triangles
		if((innerNodes > 0 && s_sharedData[threadIdx.y][2]+innerNodes >= g_taskStackBVH.sizeNodes) || (leafChunks > 0 && s_sharedData[threadIdx.y][3]+leafChunks >= g_taskStackBVH.sizeTris))
		{
			g_taskStackBVH.unfinished = 1;
		}
	}
	int childrenIdx = s_sharedData[threadIdx.y][2];
	int triOfs = s_sharedData[threadIdx.y][3];

#ifndef COMPACT_LAYOUT
	childLeft = childrenIdx+0;
	childRight = childrenIdx+1;
	leftOfs = triOfs;
	rightOfs = triOfs+((leftLeaf && numLeft != 0) ? (numLeft*3) : 0);
#else
	childLeft = childrenIdx+0;
	childRight = childrenIdx+(leftLeaf ? 0 : 1);
	leftOfs = triOfs;
#ifdef DUPLICATE_REFERENCES
	rightOfs = triOfs+((leftLeaf && numLeft != 0) ? (numLeft*3+1) : 0);
#else
	rightOfs = triOfs+((leftLeaf && numLeft != 0) ? (numLeft+1) : 0);
#endif
#endif
}

#endif

//------------------------------------------------------------------------

// Decides what type of task should be created
__device__ bool taskDecideType(int tid, volatile TaskBVH* newTask)
{
	// Update this task in the final array
	int termCrit;
	int triStart = newTask->triStart;
	int triLeft = newTask->triLeft;
	int triRight = newTask->triRight;
	int triEnd = newTask->triEnd;
	int numLeft = triLeft;
	int numRight = triRight;
	bool leftLeaf, rightLeaf;
	int parentIdx = newTask->parentIdx;
	int nodeIdx = newTask->nodeIdx;
	int taskID = newTask->taskID;

	ASSERT_DIVERGENCE("taskDecideType top", tid);
	
	if(taskTerminationCriteria(triEnd-triStart, numLeft, numRight, newTask->bbox, newTask->splitPlane, termCrit, leftLeaf, rightLeaf))
	{
#ifndef INTERLEAVED_LAYOUT
		taskSaveNodeParent(tid, triStart, triEnd, numLeft, numRight, parentIdx, nodeIdx, taskID);
#else
		if(tid == 0)
		{
			//printf("Left %d, right %d, depth %d\n", numLeft, numRight, newTask->depth);
			int triMem = align<int, 16>((triEnd-triStart)*(3*sizeof(float4)+sizeof(int)));
			s_sharedData[threadIdx.y][3] = atomicAdd(&g_taskStackBVH.nodeTop, triMem); // Atomically acquire leaf space in bytes for woop values and indices
		}
		int triOfs = s_sharedData[threadIdx.y][3];
		int triNum = createKdtreeInterleavedLeafWoop(tid, triOfs, g_kdtree, triStart, triEnd, (float4*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart));
		*(int4*)(&g_kdtree[nodeIdx]) = make_int4(triNum, triOfs/sizeof(float4), parentIdx, 0); // Sets the leaf child pointers
#endif

		// Mark children as leaves -> correct update of unfinished counter
		s_sharedData[threadIdx.y][0] = KDTREE_LEAF;
		s_sharedData[threadIdx.y][1] = KDTREE_LEAF;
		s_sharedData[threadIdx.y][4] = 1;
		s_sharedData[threadIdx.y][5] = 1;

		if(tid == 0)
		{
#ifdef BVH_COUNT_NODES
			// Beware this node has a preallocated space and is thus also counted as an inner node
			atomicAdd(&g_taskStackBVH.numNodes, 1);
			atomicAdd(&g_taskStackBVH.numLeaves, 1);
			atomicAdd(&g_taskStackBVH.numSortedTris, triEnd - triStart);
			//printf("Split Leaf (%d, %d)\n", triStart, triEnd);
#endif
#ifdef LEAF_HISTOGRAM
			atomicAdd(&g_taskStackBVH.leafHist[numLeft+numRight], 1); // Update histogram
#endif
		}

		ASSERT_DIVERGENCE("taskDecideType leaf", tid);

#ifdef DEBUG_INFO
		newTask->terminatedBy = termCrit;
#endif

		return true;
	}
	else
	{
		ASSERT_DIVERGENCE("taskDecideType node1", tid);

		s_sharedData[threadIdx.y][4] = numLeft;
		s_sharedData[threadIdx.y][5] = numRight;
		
#ifndef INTERLEAVED_LAYOUT
		//volatile CudaBVHNode* node0 = ((CudaBVHNode*)&s_newTask[threadIdx.y])+0;
		//volatile CudaBVHNode* node1 = ((CudaBVHNode*)&s_newTask[threadIdx.y])+1;
		
		int childLeft, childRight, leftOfs, rightOfs;
		taskAllocNodeTris(tid, leftLeaf, rightLeaf, numLeft, numRight, childLeft, childRight, leftOfs, rightOfs);
#else
		const int tMem = 3*sizeof(float4)+sizeof(int);
		const int nodeMem = 2*sizeof(CudaKdtreeNode);
		int leftLeafMem = (leftLeaf) ? align<int, 16>(numLeft*tMem) : 0;
		int rightLeafMem = (rightLeaf) ? align<int, 16>(numRight*tMem) : 0;
		int leafMem = leftLeafMem + rightLeafMem; // Atomically acquire leaf space
		if(tid == 0)
		{
			s_sharedData[threadIdx.y][2] = atomicAdd(&g_taskStackBVH.nodeTop, nodeMem+leafMem); // Inner node and triangles -> create new subtasks in the final array
		}
		int childrenIdx = s_sharedData[threadIdx.y][2];
		int triOfs = childrenIdx + nodeMem;

		int childLeft = childrenIdx+0*sizeof(CudaKdtreeNode);
		int childRight = childrenIdx+1*sizeof(CudaKdtreeNode);

		int leftOfs = triOfs;
		int rightOfs = triOfs+leftLeafMem;
#endif

		ASSERT_DIVERGENCE("taskDecideType node2", tid);

		// Check if children should be leaves
		if(leftLeaf)
		{
#ifndef INTERLEAVED_LAYOUT
			taskSaveNodeLeft(tid, leftOfs, childLeft, numLeft, nodeIdx, newTask);
#else
			int triNum = createKdtreeInterleavedLeafWoop(tid, leftOfs, g_kdtree, 0, triLeft, (float4*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemoryLeft, numLeft));
			*(int4*)(&g_kdtree[childLeft]) = make_int4(triNum, leftOfs/sizeof(float4), nodeIdx, 0); // Sets the leaf child pointers
			s_sharedData[threadIdx.y][0] = triNum;
#endif

			if(tid == 0)
			{
				if(numLeft != 0)
					freeBuffers(s_task[threadIdx.y].dynamicMemoryLeft, numLeft*sizeof(int));
#ifdef BVH_COUNT_NODES
				atomicAdd(&g_taskStackBVH.numLeaves, 1);
				if(numLeft == 0)
					atomicAdd(&g_taskStackBVH.numEmptyLeaves, 1);
				//printf("Force Leaf left (%d, %d) (%d)(%d)\n", triStart, triEnd, numLeft, numRight);
#endif
#ifdef LEAF_HISTOGRAM
				atomicAdd(&g_taskStackBVH.leafHist[numLeft], 1); // Update histogram
#endif
			}
		}
		else
		{
			s_sharedData[threadIdx.y][0] = childLeft;
		}

		ASSERT_DIVERGENCE("taskDecideType node3", tid);

		if(rightLeaf)
		{
#ifndef INTERLEAVED_LAYOUT
			taskSaveNodeRight(tid, rightOfs, childRight, numRight, nodeIdx, newTask);
#else
			int triNum = createKdtreeInterleavedLeafWoop(tid, rightOfs, g_kdtree, 0, triRight, (float4*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemoryRight, numRight));
			*(int4*)(&g_kdtree[childRight]) = make_int4(triNum, rightOfs/sizeof(float4), nodeIdx, 0); // Sets the leaf child pointers
			s_sharedData[threadIdx.y][1] = triNum;
#endif

			if(tid == 0)
			{
				if(numRight != 0)
					freeBuffers(s_task[threadIdx.y].dynamicMemoryRight, numRight*sizeof(int));
#ifdef BVH_COUNT_NODES
				atomicAdd(&g_taskStackBVH.numLeaves, 1);
				if(numRight == 0)
					atomicAdd(&g_taskStackBVH.numEmptyLeaves, 1);
				//printf("Force Leaf right (%d, %d) (%d)(%d)\n", triStart, triEnd, numLeft, numRight);
#endif
#ifdef LEAF_HISTOGRAM
				atomicAdd(&g_taskStackBVH.leafHist[numRight], 1); // Update histogram
#endif
			}
		}
		else
		{
			s_sharedData[threadIdx.y][1] = childRight;
		}

		ASSERT_DIVERGENCE("taskDecideType nodeBottom", tid);

		//if(tid == 0)
		//	printf("Constructed node %d\n", nodeIdx);
#ifndef INTERLEAVED_LAYOUT
		int dim = getPlaneDimension(*(float4*)&newTask->splitPlane);
#ifndef COMPACT_LAYOUT
		int flag = childLeft | (dim << 30);
		g_kdtree[nodeIdx] = make_int4(flag, __float_as_int(newTask->splitPlane.w), parentIdx, 0); // Sets the leaf child pointers
#else
		int flag = (dim << KDTREE_DIMPOS);
		g_kdtree[nodeIdx] = make_int4(childLeft, childRight, __float_as_int(newTask->splitPlane.w), flag); // Sets the leaf child pointers
#endif
#else
		int dim = getPlaneDimension(*(float4*)&newTask->splitPlane);
		int flag = (childLeft/sizeof(CudaKdtreeNode)) | (dim << 30);
		*(int4*)(&g_kdtree[nodeIdx]) = make_int4(flag, __float_as_int(newTask->splitPlane.w), parentIdx, 0); // Sets the leaf child pointers
#endif

#ifdef BVH_COUNT_NODES
		if(tid == 0)
		{
			atomicAdd(&g_taskStackBVH.numNodes, 1);
			atomicAdd(&g_taskStackBVH.numSortedTris, triEnd - triStart);
			//	printf("Node (%d, %d)\n", triStart, triEnd);
		}
#endif

		ASSERT_DIVERGENCE("taskDecideType nodeBottom", tid);

		return leftLeaf && rightLeaf;
	}
}

//------------------------------------------------------------------------

// Creates child task
__device__ void taskChildTask(volatile TaskBVH* newTask)
{
	newTask->lock = LockType_Free;
#if SPLIT_TYPE == 0 || SPLIT_TYPE == 4
	newTask->axis = (s_task[threadIdx.y].axis+1)%3;
#endif

#if SPLIT_TYPE == 0
	// Precompute the splitting plane
	int axis = newTask->axis;
	//int axis = taskAxis(newTask->splitPlane, newTask->bbox, s_sharedData[threadIdx.y][WARP_SIZE-1], s_task[threadIdx.y].axis);
	splitMedian(threadIdx.x, axis, newTask->splitPlane, newTask->bbox);

	newTask->unfinished = taskWarpSubtasks(newTask->triEnd-newTask->triStart);
	newTask->type = taskChooseScanType(newTask->unfinished);

	if(tid == 0)
	{
		allocChildren(newTask->dynamicMemoryLeft, newTask->dynamicMemoryRight, newTask->triEnd-newTask->triStart, newTask->triEnd-newTask->triStart);
	}
#elif SPLIT_TYPE == 1
	int tris = newTask->triEnd-newTask->triStart;
	newTask->type = TaskType_Split;
	newTask->bestCost   = CUDART_INF_F;
#if 0 // SQRT candidates
	int evaluatedCandidates = getNumberOfSamples(tris);
	int evaluatedCandidates = 1;
	newTask->unfinished = taskWarpSubtasks(c_env.optPlaneSelectionOverhead * tris/evaluatedCandidates); // Number of warp sized subtasks
#elif 0 // Fixed candidates
	newTask->unfinished = 1024; // Number of warp sized subtasks
#else // All candidates
	newTask->unfinished = tris*6; // Number of warp sized subtasks
#endif
	if(newTask->unfinished == 1)
		newTask->lock = LockType_None; // Set flag to skip update
#elif SPLIT_TYPE == 2
	newTask->type = TaskType_Split;
	newTask->unfinished = 1;
#elif SPLIT_TYPE == 3
	int evaluatedTris = taskWarpSubtasks(getNumberOfSamples(newTask->triEnd-newTask->triStart));
	newTask->unfinished = PLANE_COUNT*evaluatedTris; // Each WARP_SIZE rays and tris add their result to one plane
	//if(newTask->unfinished > PLANE_COUNT*2)
	//{
		newTask->type = TaskType_SplitParallel;
	/*}
	else
	{
		newTask->unfinished = 1;
	}*/
#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
	newTask->type         = TaskType_InitMemory;
	newTask->unfinished   = taskWarpSubtasks(sizeof(SplitArray)/sizeof(int));
#else
	newTask->type = TaskType_BinTriangles;
	newTask->unfinished = (taskWarpSubtasks(newTask->triEnd-newTask->triStart)+BIN_MULTIPLIER-1)/BIN_MULTIPLIER; // Each thread bins one triangle
#endif // BINNING_TYPE == 0 || BINNING_TYPE == 1

#if defined(OBJECT_SAH)/* && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)*/
	// Compute the rest with object SAH
	if(newTask->triEnd-newTask->triStart <= WARP_SIZE)
	{
		newTask->type = TaskType_BuildObjectSAH;
		newTask->unfinished = 1;
	}
#endif

#endif // SPLIT_TYPE == 0

	newTask->step = 0;
	newTask->depth = s_task[threadIdx.y].depth+1;
	newTask->subFailureCounter = s_task[threadIdx.y].subFailureCounter;
	newTask->origSize = newTask->unfinished;
#if SCAN_TYPE == 2 || SCAN_TYPE == 3
	newTask->triLeft = 0;
	newTask->triRight = 0;
#endif

#ifdef DEBUG_INFO
	newTask->terminatedBy = TerminatedBy_None;
	newTask->sync = 0;

	newTask->clockStart = 0;
	newTask->clockEnd = 0;
#endif
}

//------------------------------------------------------------------------

// Prepares a new task based on its ID
__device__ void taskCreateSubtask(int tid, volatile TaskBVH* newTask, int subIdx)
{
	ASSERT_DIVERGENCE("taskCreateSubtask", tid);

	volatile float* bbox = (volatile float*)&(newTask->bbox);
	volatile const float* srcBox = (volatile const float*)&s_sharedData[threadIdx.y][10];

	//if(tid == 4)
	{
		switch(subIdx) // Choose which subtask we are creating
		{
		case 1:
			newTask->triStart   = 0;
			newTask->triEnd     = s_task[threadIdx.y].triRight;
			newTask->dynamicMemory = s_task[threadIdx.y].dynamicMemoryRight;
			//srcBox = (volatile const float*)&(s_task[threadIdx.y].bboxRight);
			srcBox = (volatile const float*)&(s_task[threadIdx.y].bbox);
			break;

		case 0:
			newTask->triStart   = 0;
			newTask->triEnd     = s_task[threadIdx.y].triLeft;
			newTask->dynamicMemory = s_task[threadIdx.y].dynamicMemoryLeft;
			//srcBox = (volatile const float*)&(s_task[threadIdx.y].bboxLeft);
			srcBox = (volatile const float*)&(s_task[threadIdx.y].bbox);
			break;
		}
	}

	// Copy CudaAABB from corresponding task
	if(tid < sizeof(CudaAABB)/sizeof(float))
	{
		bbox[tid] = srcBox[tid];
	}

	int dim = getPlaneDimension(*(float4*)&s_task[threadIdx.y].splitPlane);
	switch(subIdx)
	{
	case 1:
		*(((volatile float*)&newTask->bbox.m_mn.x)+dim) = fmaxf(*(((volatile float*)&newTask->bbox.m_mn.x)+dim), s_task[threadIdx.y].splitPlane.w);
		break;

	case 0:
		*(((volatile float*)&newTask->bbox.m_mx.x)+dim) = fminf(*(((volatile float*)&newTask->bbox.m_mx.x)+dim), s_task[threadIdx.y].splitPlane.w);
		break;
	}

	//if(tid == 30)
	{
		taskChildTask(newTask);
	}

	newTask->taskID = subIdx;
}

//------------------------------------------------------------------------

#if ENQUEUE_TYPE != 3

// Adds subtasks of a task into a global task queue
__device__ void taskEnqueueSubtasks(int tid, int taskIdx)
{
	ASSERT_DIVERGENCE("taskEnqueueSubtasks top", tid);

	int *stackTop = &g_taskStackBVH.top;
#if ENQUEUE_TYPE == 0
	int beg = *stackTop;
	bool goRight = true;
#elif ENQUEUE_TYPE == 1
	int beg = g_taskStackBVH.bottom;
#elif ENQUEUE_TYPE == 2
	int beg = *stackTop;
#endif

//#pragma unroll 2 // OPTIMIZE: Is this beneficial?
	for(int i = 0; i < 2; i++)
	{
		if(isKdLeaf(s_sharedData[threadIdx.y][i])) // Skip leaf
			continue;
		taskCreateSubtask(tid, &s_newTask[threadIdx.y], i); // Fill newTask with valid task for ID=i

		s_newTask[threadIdx.y].parentIdx = s_task[threadIdx.y].nodeIdx;
		s_newTask[threadIdx.y].nodeIdx = s_sharedData[threadIdx.y][i];
		int newStatus = s_newTask[threadIdx.y].unfinished;

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle0", tid);

#ifdef DEBUG_INFO
		s_newTask[threadIdx.y].parent = taskIdx;
#endif

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT) || defined(COUNT_STEPS_CACHE)
		numReads[threadIdx.y] = 0;
#endif

#if ENQUEUE_TYPE == 0
		if(goRight)
		{
			taskEnqueueRight(tid, g_taskStackBVH.header, s_sharedData[threadIdx.y], newStatus, beg, 0); // Go right of beg and fill empty tasks

			if(beg < 0) // Not added when going right
			{
				goRight = false;
				beg = *stackTop;
			}
		}

		// Cannot be else, both paths may need to be taken for same i
		if(!goRight)
		{
			taskEnqueueLeft(tid, g_taskStackBVH.header, s_sharedData[threadIdx.y], newStatus, beg, &g_taskStackBVH.unfinished, g_taskStackBVH.sizePool); // Go left of beg and fill empty tasks
			if(beg == -1)
			{
				atomicMax(&g_taskStackBVH.top, g_taskStackBVH.sizePool);
				return;
			}
		}
#else
		taskEnqueueLeft(tid, g_taskStackBVH.header, s_sharedData[threadIdx.y], newStatus, beg, &g_taskStackBVH.unfinished, g_taskStackBVH.sizePool); // Go left of beg and fill empty tasks
		if(beg == -1)
		{
			atomicMax(&g_taskStackBVH.top, g_taskStackBVH.sizePool);
			return;
		}
#endif

		// All threads
		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle1", tid);

		/*if(tid == 0)
		{
			printf("warpId %d\n", blockDim.y*blockIdx.x + threadIdx.y);
			printf("task taskIdx: %d\n", beg);
			printf("Header: %d\n", newStatus);
			printf("Unfinished: %d\n", s_newTask[threadIdx.y].unfinished);
			printf("Type: %d\n", s_newTask[threadIdx.y].type);
			printf("TriStart: %d\n", s_newTask[threadIdx.y].triStart);
			printf("TriEnd: %d\n", s_newTask[threadIdx.y].triEnd);
			printf("Box: (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", s_newTask[threadIdx.y].bbox.m_mn.x, s_newTask[threadIdx.y].bbox.m_mn.y, s_newTask[threadIdx.y].bbox.m_mn.z,
				s_newTask[threadIdx.y].bbox.m_mx.x, s_newTask[threadIdx.y].bbox.m_mx.y, s_newTask[threadIdx.y].bbox.m_mx.z);
			printf("\n");
		}*/

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT)
		maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], numReads[threadIdx.y]);
		sumSteps[threadIdx.y] += numReads[threadIdx.y];
		numSteps[threadIdx.y]++;
#endif

		if(tid == 24)
		{
			atomicMax(&g_taskStackBVH.top, beg); // Update the stack top to be a larger position than all nonempty tasks
#if ENQUEUE_TYPE == 1
			atomicMax(&g_taskStackBVH.bottom, beg);  // Update the stack bottom
#endif
		}

#ifdef DIVERGENCE_TEST
		if(beg >= 0 && beg < g_taskStackBVH.sizePool) // TESTING ONLY - WRITE WILL CAUSE "UNKNOWN ERROR" IF WARP DIVERGES
			taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
		else
			printf("task adding on invalid index: %d, Tid %d\n", beg, tid);
#else
		taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
#endif

#if SPLIT_TYPE >= 3
#if PLANE_COUT > WARP_SIZE // Clear cannot be processed by a single warp
		assert(PLANE_COUT < WARP_SIZE);
#endif
		// Clear the SplitStack for the next use
		int *split = (int*)&g_splitStack[beg];
		int numElems = (sizeof(SplitInfoTri)/sizeof(int));
#pragma unroll
		for(int j = tid; j < numElems; j += WARP_SIZE) // Zero 1 SplitData = PLANE_COUNT*2 ints
		{
			split[tid] = 0; // Each thread clears 1 int-sized variable
			split += WARP_SIZE; // Each thread move to the next clear task
		}
#endif
/*#if SPLIT_TYPE == 3
#if PLANE_COUT > WARP_SIZE // Clear cannot be processed by a single warp
		assert(PLANE_COUT < WARP_SIZE);
#endif
		// Clear the SplitStack for the next use
		if(s_newTask[threadIdx.y].type == TaskType_SplitParallel && tid < PLANE_COUNT)
		{
			int *split = (int*)&g_splitStackBVH[beg];
#pragma unroll
			for(int j = 0; j < sizeof(SplitDataBVH)/sizeof(int); j++) // Zero 1 SplitData = PLANE_COUNT*2 ints
			{
				split[tid] = 0; // Each thread clears 1 int-sized variable
				split += PLANE_COUNT; // Each thread move to the next clear task
			}
		}
#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE != 0 && BINNING_TYPE != 1
		// Copy empty task from the end of the array
		if(s_newTask[threadIdx.y].type == TaskType_BinTriangles)
		{
			int *orig = (int*)&g_redSplits[g_taskStackBVH.sizePool];
			int *split = (int*)&g_redSplits[beg];
			int numElems = (sizeof(SplitArray)/sizeof(int));
			for(int j = tid; j < numElems; j+=WARP_SIZE)
			{
				split[tid] = orig[tid]; // Each thread copies 1 int-sized variable
				orig += WARP_SIZE; // Each thread move to the next clear task
				split += WARP_SIZE; // Each thread move to the next clear task
			}
		}
#endif
#endif*/

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle2", tid);

		
#if PARALLELISM_TEST >= 0
		if(tid == 0)
		{
#ifdef CUTOFF_DEPTH
			int active;
			if(s_newTask[threadIdx.y].depth > c_env.optCutOffDepth)
				active = atomicAdd(&g_numActive, 0);
			else
				active = atomicAdd(&g_numActive, 1)+1;
#else
			int active = atomicAdd(&g_numActive, 1);
#endif
			int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
			//if(active > ACTIVE_MAX)
			//	printf("Warp %d too much [%d] subtasks\n", warpIdx, active);
#ifdef CUTOFF_DEPTH
			if(/*beg == 124 || */(active == 0 && i == 1))
#else
			if(active == 0)
#endif
			{
				//printf("Warp %d no active tasks before adding task with %d subtasks\n", warpIdx, newStatus);
				g_taskStackBVH.unfinished = 1;
			}
		}
#endif
		__threadfence(); // Make sure task is copied to the global memory before we unlock it

#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
		if(tid == 24)
		{
			taskCacheActive(beg, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
		}
#endif

		// Unlock the task - set the task status
#ifdef CUTOFF_DEPTH
		if(s_newTask[threadIdx.y].depth > c_env.optCutOffDepth)
			g_taskStackBVH.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks
		else
			g_taskStackBVH.header[beg] = newStatus; // This operation is atomic anyway
#else
		g_taskStackBVH.header[beg] = newStatus; // This operation is atomic anyway
		//g_taskStackBVH.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks
#endif

#if ENQUEUE_TYPE == 0
		if(goRight)
			beg--;
		else
			beg++;
#else
		beg++;
#endif

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle3", tid);
	}

	//if(s_task[threadIdx.y].depth > c_env.optCutOffDepth)
	//g_taskStackBVH.unfinished = 1; // Finish computation after first task

	ASSERT_DIVERGENCE("taskEnqueueSubtasks aftercycle", tid);
}

//------------------------------------------------------------------------

#else // ENQUEUE_TYPE == 3
// Adds subtasks of a task into a global task queue
__device__ void taskEnqueueSubtasksCache(int tid, int taskIdx)
{
	ASSERT_DIVERGENCE("taskEnqueueSubtasksParallel top", tid);

	int *stackTop = &g_taskStackBVH.top;
	unsigned int *emptyTop = &g_taskStackBVH.emptyTop;
	unsigned int *emptyBottom = &g_taskStackBVH.emptyBottom;
	int pos = *emptyBottom;
	int top = *emptyTop;
	int beg = -1;
	int mid = -1;
	int status;

//#pragma unroll 2 // OPTIMIZE: Is this beneficial?
	for(int i = 0; i < 2; i++)
	{
		if(isKdLeaf(s_sharedData[threadIdx.y][i])) // Skip leaf
			continue;
		taskCreateSubtask(tid, &s_newTask[threadIdx.y], i); // Fill newTask with valid task for ID=i

		s_newTask[threadIdx.y].parentIdx = s_task[threadIdx.y].nodeIdx;
		s_newTask[threadIdx.y].nodeIdx = s_sharedData[threadIdx.y][i];
		int newStatus = s_newTask[threadIdx.y].unfinished;

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle0", tid);

#ifdef DEBUG_INFO
		s_newTask[threadIdx.y].parent = taskIdx;
#endif

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT) || defined(COUNT_STEPS_CACHE)
		numReads[threadIdx.y] = 0;
#endif

		status = TaskHeader_Locked;
		while(mid == -1 && pos != top && status != TaskHeader_Empty)
		{
#ifdef COUNT_STEPS_CACHE
			numReads[threadIdx.y]++;
#endif
			taskEnqueueCache(tid, &g_taskStackBVH, s_sharedData[threadIdx.y], status, pos, beg, top);
			ASSERT_DIVERGENCE("taskEnqueueSubtasks cache", tid);

			// Test that we have not read all cached items
			pos++;
			if(pos > EMPTY_MAX)
				pos = 0;

			if(pos >= top)
			{
				int tmp = *emptyTop;
				if(top != tmp)
					top = tmp;
				else
				{
					break;
				}
			}
		}

		/*while(mid == -1 && (pos != top && taskEnqueueCache(tid, status, pos, beg, top)) && status != TaskHeader_Empty)
			;*/

		if(status != TaskHeader_Empty)
		{
			if(mid == -1)
			{
				//beg = g_taskStackBVH.bottom;
				//beg = g_taskStackBVH.top;
				beg = max(*stackTop - WARP_SIZE, 0);
				mid = 0;
			}
#ifdef COUNT_STEPS_LEFT
				numReads[threadIdx.y]++;
#endif
			taskEnqueueLeft(tid, g_taskStackBVH.header, s_sharedData[threadIdx.y], newStatus, beg, &g_taskStackBVH.unfinished, g_taskStackBVH.sizePool); // Go left of beg and fill empty tasks
			if(beg == -1)
			{
				atomicMax(&g_taskStackBVH.top, g_taskStackBVH.sizePool);
				return;
			}
		}

		// All threads
		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle1", tid);

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT) || defined(COUNT_STEPS_CACHE)
		maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], numReads[threadIdx.y]);
		sumSteps[threadIdx.y] += numReads[threadIdx.y];
		numSteps[threadIdx.y]++;
#endif

		if(tid == 24)
		{
			atomicMax(&g_taskStackBVH.top, beg); // Update the stack top to be a larger position than all nonempty tasks
		}

#ifdef DIVERGENCE_TEST
		if(beg >= 0 && beg < g_taskStackBVH.sizePool) // TESTING ONLY - WRITE WILL CAUSE "UNKNOWN ERROR" IF WARP DIVERGES
			taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
		else
			printf("task adding on invalid index: %d, Tid %d\n", beg, tid);
#else
		taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
#endif

#if SPLIT_TYPE >= 3
#if PLANE_COUT > WARP_SIZE // Clear cannot be processed by a single warp
		assert(PLANE_COUT < WARP_SIZE);
#endif
		// Clear the SplitStack for the next use
		int *split = (int*)&g_splitStack[beg];
		int numElems = (sizeof(SplitInfoTri)/sizeof(int));
#pragma unroll
		for(int j = tid; j < numElems; j += WARP_SIZE) // Zero 1 SplitData = PLANE_COUNT*2 ints
		{
			split[tid] = 0; // Each thread clears 1 int-sized variable
			split += WARP_SIZE; // Each thread move to the next clear task
		}
#endif
/*#if SPLIT_TYPE == 3
#if PLANE_COUT > WARP_SIZE // Clear cannot be processed by a single warp
		assert(PLANE_COUT < WARP_SIZE);
#endif
		// Clear the SplitStack for the next use
		if(s_newTask[threadIdx.y].type == TaskType_SplitParallel && tid < PLANE_COUNT)
		{
			int *split = (int*)&g_splitStackBVH[beg];
#pragma unroll
			for(int j = 0; j < sizeof(SplitDataBVH)/sizeof(int); j++) // Zero 1 SplitData = PLANE_COUNT*2 ints
			{
				split[tid] = 0; // Each thread clears 1 int-sized variable
				split += PLANE_COUNT; // Each thread move to the next clear task
			}
		}
#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE != 0 && BINNING_TYPE != 1
		// Copy empty task from the end of the array
		if(s_newTask[threadIdx.y].type == TaskType_BinTriangles)
		{
			int *orig = (int*)&g_redSplits[g_taskStackBVH.sizePool];
			int *split = (int*)&g_redSplits[beg];
			int numElems = (sizeof(SplitArray)/sizeof(int));
			for(int j = tid; j < numElems; j+=WARP_SIZE)
			{
				split[tid] = orig[tid]; // Each thread copies 1 int-sized variable
				orig += WARP_SIZE; // Each thread move to the next clear task
				split += WARP_SIZE; // Each thread move to the next clear task
			}
		}
#endif
#endif*/

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle2", tid);

#if PARALLELISM_TEST >= 0
		if(tid == 0)
		{
#ifdef CUTOFF_DEPTH
			int active;
			if(s_newTask[threadIdx.y].depth > c_env.optCutOffDepth)
				active = atomicAdd(&g_numActive, 0);
			else
				active = atomicAdd(&g_numActive, 1)+1;
#else
			int active = atomicAdd(&g_numActive, 1);
#endif
			int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
			//if(active > ACTIVE_MAX)
			//	printf("Warp %d too much [%d] subtasks\n", warpIdx, active);
#ifdef CUTOFF_DEPTH
			if(/*beg == 124 || */(active == 0 && i == 1))
#else
			if(active == 0)
#endif
			{
				//printf("Warp %d no active tasks before adding task with %d subtasks\n", warpIdx, newStatus);
				g_taskStackBVH.unfinished = 1;
			}
		}
#endif
		__threadfence(); // Make sure task is copied to the global memory before we unlock it

#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
		if(tid == 24)
		{
			taskCacheActive(beg, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
		}
#endif

		// Unlock the task - set the task status
#ifdef CUTOFF_DEPTH
		if(s_newTask[threadIdx.y].depth > c_env.optCutOffDepth)
			g_taskStackBVH.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks
		else
			g_taskStackBVH.header[beg] = newStatus; // This operation is atomic anyway
#else
		g_taskStackBVH.header[beg] = newStatus; // This operation is atomic anyway
		//g_taskStackBVH.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks
#endif

		beg++; // Move for next item

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle3", tid);
	}

	ASSERT_DIVERGENCE("taskEnqueueSubtasks aftercycle", tid);


	//ASSERT_DIVERGENCE("taskEnqueueSubtasks bottom", tid); // Tid 24 diverges here but it converges at the end of this function
}

#endif // ENQUEUE_TYPE == 3

//------------------------------------------------------------------------

#if DEQUEUE_TYPE <= 3

__device__ __noinline__ bool taskDequeue(int tid, int& subtask, int& taskIdx, int &popCount)
{
	ASSERT_DIVERGENCE("taskDequeue", tid);

#if PARALLELISM_TEST >= 0
	int* active = &g_numActive;
#endif

	if(tid == 13) // Only thread 0 acquires the work
	{
/*#if PARALLELISM_TEST >= 0
		int *active = &g_numActive;
		unsigned int waitCounter = 0;
		clock_t clockStart = clock();
		while(*active == 0)
			waitCounter++;
		clock_t clockEnd = clock();
		if(waitCounter != 0)
		{
			int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
			printf("Warp %d waited %u iterations taking %u\n", warpIdx, waitCounter, clockEnd-clockStart);
		}
#endif*/

		// Initiate variables
		int* header = g_taskStackBVH.header;
		int *unfinished = &g_taskStackBVH.unfinished;
		int *stackTop = &g_taskStackBVH.top;

		int status = TaskHeader_Active;
		int counter = 0; // TESTING ONLY: Allows to undeadlock a failed run!
#ifdef COUNT_STEPS_DEQUEUE
		int readCounter = 1;
#endif

#ifdef SNAPSHOT_WARP
	long long int clock = clock64();
	*(long long int*)&(s_sharedData[threadIdx.y][4]) = clock;
#endif

#if DEQUEUE_TYPE == 0
		int beg = *stackTop;
#elif DEQUEUE_TYPE == 1
		int beg = 0;
		if((blockDim.y*blockIdx.x + threadIdx.y) & 0) // Even warpID
		{
			beg = *stackTop;
		}
#elif DEQUEUE_TYPE == 2
		int warpIdx = (blockDim.y*blockIdx.x + threadIdx.y);
		int limit = 4;
		int step = (warpIdx % 5)+1;
		int top = *stackTop;
		int beg = warpIdx % (top + 1);
		//int beg = top - (warpIdx % (top + 1));
		//int beg = (int)((float)warpIdx*((float)(top + 1) / (float)NUM_WARPS));
#elif DEQUEUE_TYPE == 3
		unsigned int *activeTop = &g_taskStackBVH.activeTop;
		int* act = g_taskStackBVH.active;
		int pos = (*activeTop)-1;
		if(pos < 0)
			pos = ACTIVE_MAX;
		int top = pos;
		int beg = -1;
#endif

#if DEQUEUE_TYPE == 3
		while((beg == -1 || pos != top) && *unfinished < 0 && status <= TaskHeader_Active)
		{
			beg = act[pos];
			status = header[beg];

#ifdef COUNT_STEPS_DEQUEUE
			readCounter++;
#endif

			if(status > TaskHeader_Active)
			{
				//popCount = taskPopCount(status);
				popCount = taskPopCount(*((int*)&g_taskStackBVH.tasks[beg].origSize));
				// Try acquire the current task - decrease it
				status = atomicSub(&g_taskStackBVH.header[beg], popCount); // Try to update and return the current value
			}

			/*if(status <= TaskHeader_Active) // Immediate exit
				break;*/

			counter++;
			pos--;

			if(pos < 0)
				pos = ACTIVE_MAX;
		}

		if(status <= TaskHeader_Active)
			beg = *stackTop;
#endif

		/*if(status <= TaskHeader_Active) // Immediate exit
			counter = WAIT_COUNT;*/

		while(counter < WAIT_COUNT && *unfinished < 0 && status <= TaskHeader_Active) // while g_taskStackBVH is not empty and we have not found ourselves a task
		{
			// Find first active task, end if we have reached start of the array
#if DEQUEUE_TYPE == 0 || DEQUEUE_TYPE == 2 || DEQUEUE_TYPE == 3
			while(beg >= 0 && (status = header[beg]) <= TaskHeader_Active)
			{
#if DEQUEUE_TYPE == 2
				if(beg > limit)
					beg -= step;
				else
#endif
					beg--;
#ifdef COUNT_STEPS_DEQUEUE
				//if(*active == 0)
				//if(status == TaskHeader_Empty)
				readCounter++;
#endif
			}

#ifdef COUNT_STEPS_DEQUEUE
			/*if(readCounter > 600) // Test long waiting
			{
				*unfinished = 1;
				int warpIdx = (blockDim.y*blockIdx.x + threadIdx.y);
				printf("Warp %d ended on task: %d\n", warpIdx, beg);
				break;
			}*/
#endif

			if(beg < 0) // We have found no active task
			{
#if DEQUEUE_TYPE == 0 || DEQUEUE_TYPE == 2 || DEQUEUE_TYPE == 3
				// Try again from a new beginning
				beg = *stackTop;
#endif
				counter++;

				/*// Sleep - works but increases time if used always
				clock_t start = clock();
				clock_t end = start + 1000;
				while(start < end)
				{
				//g_taskStackBVH.tasks[0].padding1 = 0; // DOES NOT SEEM TO BE BETTER
				//__threadfence_system();
				start = clock();
				}*/
				continue;
			}
#elif DEQUEUE_TYPE == 1
			if((blockDim.y*blockIdx.x + threadIdx.y) & 0) // Even warpID
			{
				while(beg >= 0 && (status = header[beg]) <= TaskHeader_Active)
				{
					beg--;
	#ifdef COUNT_STEPS_DEQUEUE
					readCounter++;
	#endif
				}

				if(beg < 0) // We have found no active task
				{
					beg = *stackTop; // Try again from a new beginning
					counter++;
					continue;
				}
			}
			else
			{
				while(beg <= (*stackTop) && (status = header[beg]) <= TaskHeader_Active)
				{
					beg++;
	#ifdef COUNT_STEPS_DEQUEUE
					readCounter++;
	#endif
				}

				if(beg > (*stackTop)) // We have found no active task
				{
					beg = 0; // Try again from a new beginning
					counter++;
					continue;
				}
			}
#endif

			//popCount = taskPopCount(status);
			popCount = taskPopCount(*((int*)&g_taskStackBVH.tasks[beg].origSize));
			// Try acquire the current task - decrease it
			status = atomicSub(&g_taskStackBVH.header[beg], popCount); // Try to update and return the current value

			//int warpIdx = (blockDim.y*blockIdx.x + threadIdx.y);
			//if(status < TaskHeader_Active)
				//printf("W %d U %d pC %d\n", warpIdx, status, popCount);

			if(status <= TaskHeader_Active) // We have not succeeded
			{
#ifdef COUNT_STEPS_DEQUEUE
				readCounter++;
#endif
				// Move to next task
#if DEQUEUE_TYPE == 0 || DEQUEUE_TYPE == 2 || DEQUEUE_TYPE == 3
#if DEQUEUE_TYPE == 2
				if(beg > limit)
					beg -= step;
				else
#endif
					beg--;
#elif DEQUEUE_TYPE == 1
				if((blockDim.y*blockIdx.x + threadIdx.y) & 0) // Even warpID
					beg--;
				else
					beg++;
#endif
				// OPTIMIZE: we shall move beg--; as the first statement in the outer while and start with g_taskStackBVH.top+1.
			}
		}

		// Distribute information to all threads through shared memory
		if(counter >= WAIT_COUNT || status <= TaskHeader_Active || *unfinished == 0) // g_taskStackBVH is empty, no more work to do
		{
			s_sharedData[threadIdx.y][0] = -1; // no task, end
		}
		else
		{
			// status now holds the information about what tasks this warp has to do but in a reversed order and offseted by 1
			// (status starts at the number of subtasks and ends at 1)
			s_sharedData[threadIdx.y][0] = status - 1;
			s_sharedData[threadIdx.y][1] = beg;
			s_sharedData[threadIdx.y][2] = popCount;
		}

#ifdef COUNT_STEPS_DEQUEUE
		//if(s_task[threadIdx.y].depth < 2)
		{
			maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], readCounter);
			sumSteps[threadIdx.y] += readCounter;
			numSteps[threadIdx.y]++;
			numRestarts[threadIdx.y] += counter;
		}

		//atomicExch(&g_taskStackBVH.active, beg); // Update the last active position

		if(counter >= WAIT_COUNT || status <= TaskHeader_Active || *unfinished == 0)
			numSteps[threadIdx.y] -= 1;

		if(counter >= WAIT_COUNT)
		{
			maxSteps[threadIdx.y] = -maxSteps[threadIdx.y];
			sumSteps[threadIdx.y] = -sumSteps[threadIdx.y];
			numSteps[threadIdx.y] = -numSteps[threadIdx.y];
			numRestarts[threadIdx.y] = -numRestarts[threadIdx.y];
		}
#endif
	}

	// All threads
	ASSERT_DIVERGENCE("taskDequeue bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1) // Exit, no more work
	{
		return false;
	}
	else
	{
		subtask = s_sharedData[threadIdx.y][0];
		taskIdx = s_sharedData[threadIdx.y][1];
		popCount = s_sharedData[threadIdx.y][2];
		return true;
	}
}

#else // DEQUEUE_TYPE > 3

__device__ __noinline__ bool taskDequeue(int tid)
{
	ASSERT_DIVERGENCE("taskDequeue", tid);

	// Initiate variables
	int* header = g_taskStackBVH.header;
	int* unfinished = &g_taskStackBVH.unfinished;
	int* stackTop = &g_taskStackBVH.top;
	volatile int* red = (volatile int*)&s_newTask[threadIdx.y];

	int status = TaskHeader_Active;
	int counter = 0; // TESTING ONLY: Allows to undeadlock a failed run!
#ifdef COUNT_STEPS_DEQUEUE
	int readCounter = 1;
#endif

#ifdef SNAPSHOT_WARP
	long long int clock = clock64();
	*(long long int*)&(s_sharedData[threadIdx.y][4]) = clock;
#endif

	int warpIdx = (blockDim.y*blockIdx.x + threadIdx.y);

#if DEQUEUE_TYPE == 4
	//int beg = taskWarpSubtasks(*stackTop) * WARP_SIZE - tid;
	int topChunk = taskWarpSubtasks(*stackTop);
	int beg = (warpIdx % topChunk) * WARP_SIZE - tid;

#elif DEQUEUE_TYPE == 5
	//unsigned int *activeTop = &g_taskStackBVH.activeTop;
	int* cache = g_taskStackBVH.active;
	s_task[threadIdx.y].popSubtask = status;
	//bool cached;

	//int dir = 0;
	//int pos = (*activeTop)-1;
	//if(pos < 0)
	//{
	//	dir = 1;
	//	pos = ACTIVE_MAX;
	//}
	int pos = ACTIVE_MAX;
	//int top = pos;
	int beg = -1;

	//while(((dir == 0 && pos <= top) || (dir == 1 && pos > top)) && status <= TaskHeader_Active)
	//while(pos >= 0 && status <= TaskHeader_Active)
	//do
	while(counter < (ACTIVE_MAX+1)/WARP_SIZE && status <= TaskHeader_Active)
	{
		int item = counter*WARP_SIZE + tid;
		status = TaskHeader_Active; // Reset state of all threads
		//int item = pos - tid;
		//if(item >= 0)
		if(/**stackTop > 4096 &&*/ item <= pos)
		{
			beg = cache[item];
			/*status = header[beg];

			if(__all(status <= TaskHeader_Active))
			{
				numRestarts[threadIdx.y]++;
				break;
			}

			s_sharedData[threadIdx.y][tid] = status;
			status = TaskHeader_Active;*/
			//s_owner[threadIdx.y][tid] = beg;

#if 0
			transposition_sort<int>(s_owner[threadIdx.y], tid, WARP_SIZE); // Sort the positions to form groups
			//transposition_sort_values<int>(s_owner[threadIdx.y], s_sharedData[threadIdx.y], tid, WARP_SIZE); // Sort the positions to form groups

			if(tid == 0 || s_owner[threadIdx.y][tid-1] != s_owner[threadIdx.y][tid]) // First in a group
			{
				beg = s_owner[threadIdx.y][tid];
			
#elif 0 
			for(int i = tid+1; i < WARP_SIZE; i++)
			{
				if(s_owner[threadIdx.y][i] == beg) // Duplicate
				{
					s_owner[threadIdx.y][i] = -1;
					break; // Further duplicates will be dealt with by the thread we have just eliminated
				}
			}

			if(s_owner[threadIdx.y][tid] == beg) // First in a group
			{
#else
			if(beg >= 0)
			{
#endif
				status = header[beg];
				//status = s_sharedData[threadIdx.y][tid];
			}
		}

#ifdef COUNT_STEPS_DEQUEUE
		readCounter++;
#endif

		// Initialize memory for reduction
		/*red[tid] = 0;
		//owner[tid] = -1;
		if(status > TaskHeader_Active)
		{
			red[tid] = status;
			//red[tid] = *((int*)&g_taskStackBVH.tasks[beg].origSize);
			//owner[tid] = beg;
		}*/

		// Reduce work that does not come from the same task (do not count it more than 1 time)
		//reduceWarpDiv(tid, red, owner, plus);
		//if(__any(status > TaskHeader_Active))
		//	reduceWarp<int>(tid, red, plus);

		//int popCount = taskPopCount(status);
		//int popCount = taskPopCount(*((int*)&g_taskStackBVH.tasks[beg].origSize));
		//int popCount = max((red[0] / NUM_WARPS) + 1, taskPopCount(status));
		int popCount = max((status / NUM_WARPS) + 1, taskPopCount(status));

		/*red[tid] = 0;
		if(status > TaskHeader_Active)
		{
			red[tid] = 1;
		}
		scanWarp<int>(tid, red, plus);

		int xPos = (warpIdx % red[WARP_SIZE-1]) + 1; // Position of the 1 to take work from*/

		//if(tid == 0 && red[WARP_SIZE-1] > 1)
		//	printf("Active: %d\n", red[WARP_SIZE-1]);

		if(status > TaskHeader_Active)
		{
			// Choose some with active task
			red[0] = tid;

			if(red[0] == tid)
			//if(red[tid] == xPos)
			{
				// Try acquire the current task - decrease it
				s_task[threadIdx.y].popSubtask = atomicSub(&g_taskStackBVH.header[beg], popCount); // Try to update and return the current value
				s_task[threadIdx.y].popCount = popCount;
				s_task[threadIdx.y].popTaskIdx = beg;
				//s_sharedData[threadIdx.y][1] = beg;
				//status = TaskHeader_Active;
			}
		}

		counter++;

		//if(__all(status <= TaskHeader_Active))
		//	pos -= WARP_SIZE;

		status = s_task[threadIdx.y].popSubtask;

		/*if(tid == 0 && status < TaskHeader_Dependent)
		{
			atomicAdd(&g_taskStackBVH.header[beg], s_task[threadIdx.y].popCount); // Revert if we have accidentaly changed waiting task
		}*/

		//if(pos < 0)
		//{
		//	dir ^= 1; // Flip dir
		//	pos = ACTIVE_MAX;
		//}
	} //while(false);

	//status = s_sharedData[threadIdx.y][0];

	if(status <= TaskHeader_Active)
	{
		int topChunk = taskWarpSubtasks(*stackTop);
		beg = (warpIdx % topChunk) * WARP_SIZE - tid;
	}
	//else
	//	cached = true;
	ASSERT_DIVERGENCE("taskDequeue cache", tid);
#endif

	//---------------------------- REVERT TO POOL SEARCH ----------------------------//

	while(counter < WAIT_COUNT && status <= TaskHeader_Active && *unfinished < 0) // while g_taskStack is not empty and we have not found ourselves a task
	{
		// Find first active task, end if we have reached start of the array
		while(__any(beg >= 0) && __all(beg < 0 || (status = header[beg]) <= TaskHeader_Active) /*&& restartPos != tryCounter*/)
		{
			beg -= WARP_SIZE;
#ifdef COUNT_STEPS_DEQUEUE
			readCounter++;
#endif
		}

		// OPTIMIZE: How to use the threads with beg < 0 in a clever way?
		if(__all(status <= TaskHeader_Active)) // We have found no active task
		{
			beg = taskWarpSubtasks(*stackTop) * WARP_SIZE - tid;
			counter++;
			continue;
		}

		// Initialize memory for reduction
		/*red[tid] = 0;
		if(status > TaskHeader_Active)
		{
			red[tid] = *((int*)&g_taskStackBVH.tasks[beg].origSize);
			//red[tid] = status;
		}

		reduceWarp<int>(tid, red, plus);

		int popCount = max((red[tid] / NUM_WARPS) + 1, taskPopCount(status));*/
		int popCount = max((status / NUM_WARPS) + 1, taskPopCount(status));

		// Choose the right position
		/*int warpIdx = (blockDim.y*blockIdx.x + threadIdx.y);
		int tidPos = warpIdx % WARP_SIZE; // Position of the tid to take work from

		red[tid] = -1;
		if(status > TaskHeader_Active)
		{
			red[tid] = tid;
		}

		if(red[tid] < 0 || (red[tid ^ 1] > 0 && abs(red[tid] - tidPos) > abs(red[tid ^ 1] - tidPos)))
			red[tid] = red[tid ^ 1];
		if(red[tid] < 0 || (red[tid ^ 2] > 0 && abs(red[tid] - tidPos) > abs(red[tid ^ 2] - tidPos)))
			red[tid] = red[tid ^ 2];
		if(red[tid] < 0 || (red[tid ^ 4] > 0 && abs(red[tid] - tidPos) > abs(red[tid ^ 4] - tidPos)))
			red[tid] = red[tid ^ 4];
		if(red[tid] < 0 || (red[tid ^ 8] > 0 && abs(red[tid] - tidPos) > abs(red[tid ^ 8] - tidPos)))
			red[tid] = red[tid ^ 8];
		if(red[tid] < 0 || (red[tid ^ 16] > 0 && abs(red[tid] - tidPos) > abs(red[tid ^ 16] - tidPos)))
			red[tid] = red[tid ^ 16];*/

		/*red[tid] = 0;
		if(status > TaskHeader_Active)
		{
			red[tid] = 1;
		}
		scanWarp<int>(tid, red, plus);

		int xPos = (warpIdx % red[WARP_SIZE-1]) + 1; // Position of the 1 to take work from*/
		
		if(status > TaskHeader_Active)
		{
			// Choose some with active task
			red[0] = tid;

			if(red[0] == tid)
			//if(red[tid] == xPos)
			{
				// Try acquire the current task - decrease it
				// status now holds the information about what tasks this warp has to do but in a reversed order and offseted by 1
				// (status starts at the number of subtasks and ends at 1)
				s_task[threadIdx.y].popSubtask = atomicSub(&g_taskStackBVH.header[beg], popCount); // Try to update and return the current value
				s_task[threadIdx.y].popCount = popCount;
				s_task[threadIdx.y].popTaskIdx = beg;

#ifdef INTERSECT_TEST
				if(s_sharedData[threadIdx.y][0] == TaskHeader_Empty)
					printf("Latency error!\n");
#endif
			}
		}
		status = s_task[threadIdx.y].popSubtask;

		if(status <= TaskHeader_Active) // We have not succeeded
		{
#ifdef COUNT_STEPS_DEQUEUE
			readCounter++;
#endif
			// Move to next task
			beg -= WARP_SIZE;
			// OPTIMIZE: we shall move beg -= WARP_SIZE; as the first statement in the outer while and start with g_taskStack.top+1.
		}
	}

#ifdef SNAPSHOT_WARP
	s_sharedData[threadIdx.y][3] = readCounter;
#endif

#ifdef COUNT_STEPS_DEQUEUE
	maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], readCounter);
	sumSteps[threadIdx.y] += readCounter;
	numSteps[threadIdx.y]++;
	numRestarts[threadIdx.y] += counter;

	if(counter >= WAIT_COUNT || status <= TaskHeader_Active)
		numSteps[threadIdx.y] -= 1;

	if(counter >= WAIT_COUNT)
	{
		maxSteps[threadIdx.y] = -maxSteps[threadIdx.y];
		sumSteps[threadIdx.y] = -sumSteps[threadIdx.y];
		numSteps[threadIdx.y] = -numSteps[threadIdx.y];
		numRestarts[threadIdx.y] = -numRestarts[threadIdx.y];
	}
#endif

	// All threads
	ASSERT_DIVERGENCE("taskDequeue bottom", tid);

	// Distribute information to all threads through shared memory
	if(counter >= WAIT_COUNT || status <= TaskHeader_Active) // g_taskStack is empty, no more work to do
	{
		return false;
	}
	else
	{
		s_task[threadIdx.y].popSubtask--;
		return true;
	}
}

#endif

//------------------------------------------------------------------------

// Finishes a sort task
__device__ void taskFinishSort(int tid, int taskIdx)
{
	// Free the memory
	//freeBuffers(s_task[threadIdx.y].dynamicMemory, (s_task[threadIdx.y].triEnd-s_task[threadIdx.y].triStart)*sizeof(int));

	// We should update the dependencies before we start adding new tasks because these subtasks may be finished before this is done

#ifndef KEEP_ALL_TASKS
	atomicCAS(&g_taskStackBVH.top, taskIdx, max(taskIdx-1, 0)); // Try decreasing the stack top
#endif

#if PARALLELISM_TEST == 0
	atomicSub(&g_numActive, 1);
#endif

	int fullLeft = isKdLeaf(s_sharedData[threadIdx.y][0]) ? 0 : 1;
	int fullRight = isKdLeaf(s_sharedData[threadIdx.y][1]) ? 0 : 1;
	int numSubtasks = fullLeft+fullRight;
#ifdef CUTOFF_DEPTH
	if(s_task[threadIdx.y].depth == c_env.optCutOffDepth)
		numSubtasks = 0;
#endif

	// Update taskStackBVH.unfinished
	atomicSub(&g_taskStackBVH.unfinished, numSubtasks-1);

	// Free the memory
	if((s_sharedData[threadIdx.y][4] > 0 && s_sharedData[threadIdx.y][5] > 0)) // There is no empty leaf, free the no longer used parent memory
		freeBuffers(s_task[threadIdx.y].dynamicMemory, (s_task[threadIdx.y].triEnd-s_task[threadIdx.y].triStart)*sizeof(int));

#ifndef KEEP_ALL_TASKS
	g_taskStackBVH.header[taskIdx] = TaskHeader_Empty; // Empty this task

#if ENQUEUE_TYPE == 1
	atomicMin(&g_taskStackBVH.bottom, taskIdx); // Update the stack bottom
#elif ENQUEUE_TYPE == 3
	taskCacheEmpty(taskIdx, g_taskStackBVH.empty, &g_taskStackBVH.emptyTop);
#endif
#endif
}

//------------------------------------------------------------------------

__device__ void taskFinishTask(int tid, int taskIdx)
{
#ifdef DEBUG_INFO
		if(tid == 1)
			s_task[threadIdx.y].sync++;
#endif

#ifndef KEEP_ALL_TASKS
#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
		taskUncacheActive(tid, taskIdx, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
#endif
#endif

		s_task[threadIdx.y].lock = LockType_Free;

		//g_taskStackBVH.unfinished = 1;
		//return; // Measure time without enqueue

		ASSERT_DIVERGENCE("taskFinishTask top", tid);

		bool leaf = taskDecideType(tid, &s_task[threadIdx.y]);

		if(tid == 0)
		{
			taskFinishSort(tid, taskIdx);
		}

		if(!leaf) // Subdivide
		{
			ASSERT_DIVERGENCE("taskFinishTask node", tid);

			// Enqueue the new tasks
#if ENQUEUE_TYPE != 3
			taskEnqueueSubtasks(tid, taskIdx);
#else
			taskEnqueueSubtasksCache(tid, taskIdx);
#endif
		}

#ifdef DEBUG_INFO
		taskSaveFirstToGMEM(tid, taskIdx, s_task[threadIdx.y]); // Make sure results are visible in global memory
#endif

#if PARALLELISM_TEST == 1
		if(tid == 0)
			atomicSub(&g_numActive, 1);
#endif
}

//------------------------------------------------------------------------

// Update best plane in global memory
// MERGE: Get rid of this function or merge it?
__device__ void taskUpdateBestPlane(int tid, int taskIdx)
{
	ASSERT_DIVERGENCE("taskUpdateBestPlane top", tid);

	float bestCost = s_task[threadIdx.y].bestCost;
	volatile float* g_bestCost = &g_taskStackBVH.tasks[taskIdx].bestCost;
	
	//if(tid == 11)
	{
		s_sharedData[threadIdx.y][0] = 0;

#ifdef UPDATEDEADLOCK_TEST
		int lockCounter = 0;
#endif

		// Atomicaly update the best plane in global memory
#ifdef UPDATEDEADLOCK_TEST
		while(s_sharedData[threadIdx.y][0] == 0 && bestCost < *g_bestCost && lockCounter < 1000)
#else
		while(s_sharedData[threadIdx.y][0] == 0 && bestCost < *g_bestCost)
#endif
		{
			if(tid == 11 && atomicCAS(&g_taskStackBVH.tasks[taskIdx].lock, LockType_Free, LockType_Set) == LockType_Free)
				s_sharedData[threadIdx.y][0] = 1;
				//break;
			ASSERT_DIVERGENCE("taskUpdateBestPlane while", tid);
#ifdef UPDATEDEADLOCK_TEST
			lockCounter++;
#endif
		}

#ifdef UPDATEDEADLOCK_TEST
		assert(lockCounter < 1000);
#endif

		// Update the best cost
		if(bestCost < *g_bestCost)
			s_sharedData[threadIdx.y][0] = -1;
		else if(s_sharedData[threadIdx.y][0] == 1)
			g_taskStackBVH.tasks[taskIdx].lock = LockType_Free;
	}

	ASSERT_DIVERGENCE("taskUpdateBestPlane mid", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		if(tid < (sizeof(float4) + sizeof(float) + sizeof(int)) / sizeof(float)) // Copy splitPlane
		{
			float* split = (float*)&g_taskStackBVH.tasks[taskIdx].splitPlane;
			float* shared = (float*)&s_task[threadIdx.y].splitPlane;
			split[tid] = shared[tid];
		}

		g_taskStackBVH.tasks[taskIdx].triLeft = s_task[threadIdx.y].triLeft;
		g_taskStackBVH.tasks[taskIdx].triRight = s_task[threadIdx.y].triRight;

		__threadfence();

		//if(tid == 5)
			g_taskStackBVH.tasks[taskIdx].lock = LockType_Free;
	}

	ASSERT_DIVERGENCE("taskUpdateBestPlane bottom", tid);
}

//------------------------------------------------------------------------

#if SPLIT_TYPE != 0

// Finishes a split task
__device__ void taskFinishSplit(int tid, int taskIdx, int countDown)
{
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	ASSERT_DIVERGENCE("taskFinishSplit top", tid);

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		// Advance the automaton in shared memory
		s_task[threadIdx.y].unfinished = taskWarpSubtasks(triEnd - triStart); // Number of warp sized subtasks
		s_task[threadIdx.y].type = taskChooseScanType(s_task[threadIdx.y].unfinished);
		s_task[threadIdx.y].step = 0;
		s_sharedData[threadIdx.y][0] = -1;

		int triLeft, triRight;

#if SPLIT_TYPE == 1
		if(s_task[threadIdx.y].lock != LockType_None)
		{
			// Reload splitPlane from gmem, some other warp may have improved the cost
			// OPTIMIZE: Save everything but the split to gmem
			if(tid < (sizeof(float4)) / sizeof(float)) // Copy splitPlane
			{
				float* split = (float*)&g_taskStackBVH.tasks[taskIdx].splitPlane;
				float* shared = (float*)&s_task[threadIdx.y].splitPlane;
				shared[tid] = split[tid];
			}

			triLeft = g_taskStackBVH.tasks[taskIdx].triLeft;
			triRight = g_taskStackBVH.tasks[taskIdx].triRight;
		}
		else
		{
			triLeft = s_task[threadIdx.y].triLeft;
			triRight = s_task[threadIdx.y].triRight;
		}
#else
		triLeft = s_task[threadIdx.y].triLeft;
		triRight = s_task[threadIdx.y].triRight;
#endif

		//if(tid == 0)
		//	printf("%d\n", (triLeft+triRight)-(triEnd-triStart));

		// Allocate memory for children
		// OPTIMIZE: Only allocate if they are large enough for and inner node?
		if(tid == 0)
		{
			/*if(s_task[threadIdx.y].depth < 4)
			{
				printf("NodeIdx %d, ParentIdx %d. Plane: (%.2f, %.2f, %.2f, %.2f)\n", s_task[threadIdx.y].nodeIdx, s_task[threadIdx.y].parentIdx, s_task[threadIdx.y].splitPlane.x, s_task[threadIdx.y].splitPlane.y, s_task[threadIdx.y].splitPlane.z, s_task[threadIdx.y].splitPlane.w);
				printf("NodeIdx %d, ParentIdx %d. Box: (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", s_task[threadIdx.y].nodeIdx, s_task[threadIdx.y].parentIdx, s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
					s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
				printf("BoxLeft: (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", bboxLeft.m_mn.x, bboxLeft.m_mn.y, bboxLeft.m_mn.z,
				bboxLeft.m_mx.x, bboxLeft.m_mx.y, bboxLeft.m_mx.z);
				printf("BoxRight: (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", bboxRight.m_mn.x, bboxRight.m_mn.y, bboxRight.m_mn.z,
				bboxRight.m_mx.x, bboxRight.m_mx.y, bboxRight.m_mx.z);
			}*/

			allocChildren(s_task[threadIdx.y].dynamicMemoryLeft, s_task[threadIdx.y].dynamicMemoryRight, triLeft, triRight);
		}

#if SCAN_TYPE == 2 || SCAN_TYPE == 3
		s_task[threadIdx.y].triLeft = 0;
		s_task[threadIdx.y].triRight = 0;
#endif

#ifdef SPLIT_TEST
		// Test that split is inside the bounding box
		if(tid == 12)
		{
			if(s_task[threadIdx.y].splitPlane.x != 0.f && (s_task[threadIdx.y].splitPlane.x*(-s_task[threadIdx.y].splitPlane.w) < s_task[threadIdx.y].bbox.m_mn.x || s_task[threadIdx.y].splitPlane.x*(-s_task[threadIdx.y].splitPlane.w) > s_task[threadIdx.y].bbox.m_mx.x))
			{
				printf("X-split outside! (%f, %f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)\n",
					s_task[threadIdx.y].splitPlane.x, s_task[threadIdx.y].splitPlane.y, s_task[threadIdx.y].splitPlane.z, s_task[threadIdx.y].splitPlane.w,
					s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
					s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
			}

			if(s_task[threadIdx.y].splitPlane.y != 0.f && (s_task[threadIdx.y].splitPlane.y*(-s_task[threadIdx.y].splitPlane.w) < s_task[threadIdx.y].bbox.m_mn.y || s_task[threadIdx.y].splitPlane.y*(-s_task[threadIdx.y].splitPlane.w) > s_task[threadIdx.y].bbox.m_mx.y))
			{
				printf("Y-split outside! (%f, %f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)\n",
					s_task[threadIdx.y].splitPlane.x, s_task[threadIdx.y].splitPlane.y, s_task[threadIdx.y].splitPlane.z, s_task[threadIdx.y].splitPlane.w,
					s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
					s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
			}

			if(s_task[threadIdx.y].splitPlane.z != 0.f && (s_task[threadIdx.y].splitPlane.z*(-s_task[threadIdx.y].splitPlane.w) < s_task[threadIdx.y].bbox.m_mn.z || s_task[threadIdx.y].splitPlane.z*(-s_task[threadIdx.y].splitPlane.w) > s_task[threadIdx.y].bbox.m_mx.z))
			{
				printf("Z-split outside! (%f, %f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)\n",
					s_task[threadIdx.y].splitPlane.x, s_task[threadIdx.y].splitPlane.y, s_task[threadIdx.y].splitPlane.z, s_task[threadIdx.y].splitPlane.w,
					s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
					s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
			}
		}
#endif

		taskPrepareNext(tid, taskIdx, TaskType_Split);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}

	ASSERT_DIVERGENCE("taskFinishSplit bottom", tid);
}

#endif

//------------------------------------------------------------------------

#if SPLIT_TYPE == 3
// Finishes a split task
__device__ void taskFinishSplitParallel(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSplitParallel top", tid);

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		// Compute best cost from data in global memory
		volatile float* red = (volatile float*)&s_sharedData[threadIdx.y][0];
		red[tid] = CUDART_INF_F;
		int planePos = tid;
#if PLANE_COUT > WARP_SIZE // Cannot be processed by a single warp
		assert(PLANE_COUT < WARP_SIZE);
#endif

		if(planePos < PLANE_COUNT)
		{
			float sumWeights = c_env.optAxisAlignedWeight + c_env.optTriangleBasedWeight;
			// Assign the planes to different methods
			int numAxisAlignedPlanes = c_env.optAxisAlignedWeight/sumWeights*PLANE_COUNT;
			int numTriangleBasedPlanes = c_env.optTriangleBasedWeight/sumWeights*PLANE_COUNT;

			// Recompute the split plane. OPTIMIZE: Save the split plane to global memory?
			float areaLeft, areaRight;
			float4 plane;
			int triStart = s_task[threadIdx.y].triStart;
			int triEnd = s_task[threadIdx.y].triEnd;
			volatile CudaAABB &bbox = s_task[threadIdx.y].bbox;
			findPlane(planePos, triStart, triEnd, bbox, areaLeft, areaRight, numAxisAlignedPlanes, plane);
			
			volatile SplitDataBVH *split = &(g_splitStackBVH[taskIdx].splits[planePos]);

			// Compute the plane cost
			float leftCost = areaLeft/areaAABB(bbox)*(float)split->tb;
			float rightCost = areaRight/areaAABB(bbox)*(float)split->tf;
			float cost = c_env.optCt + c_env.optCi*(leftCost + rightCost);

#ifdef SPLIT_TEST
			assert(cost > 0.f);
#endif

			// Reduce the best cost within the warp
			red[tid] = cost;
			reduceWarp(tid, red, min);
			//reduceWarp<float, Min<float>>(tid, red, Min<float>());

			// Return the best plane for this warp
			if(__ffs(__ballot(red[tid] == cost)) == tid+1) // First thread with such condition, OPTIMIZE: Can be also computed by overwrite and test: better?
			{
				s_task[threadIdx.y].splitPlane.x = plane.x;
				s_task[threadIdx.y].splitPlane.y = plane.y;
				s_task[threadIdx.y].splitPlane.z = plane.z;
				s_task[threadIdx.y].splitPlane.w = plane.w;
				s_task[threadIdx.y].bestCost = cost;
			}
		}


		// Advance the automaton in shared memory
		s_task[threadIdx.y].unfinished = taskWarpSubtasks(s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triStart); // Number of warp sized subtasks
		s_task[threadIdx.y].type = taskChooseScanType(s_task[threadIdx.y].unfinished);
		s_task[threadIdx.y].step = 0;

		s_sharedData[threadIdx.y][0] = -1;
	}

	ASSERT_DIVERGENCE("taskFinishSplitParallel mid2", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
#ifdef SPLIT_TEST
		// Test that split is inside the bounding box
		if(tid == 12)
		{
			if(s_task[threadIdx.y].splitPlane.x != 0.f && (s_task[threadIdx.y].splitPlane.x*(-s_task[threadIdx.y].splitPlane.w) < s_task[threadIdx.y].bbox.m_mn.x || s_task[threadIdx.y].splitPlane.x*(-s_task[threadIdx.y].splitPlane.w) > s_task[threadIdx.y].bbox.m_mx.x))
			{
				printf("X-split outside! (%f, %f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)\n",
					s_task[threadIdx.y].splitPlane.x, s_task[threadIdx.y].splitPlane.y, s_task[threadIdx.y].splitPlane.z, s_task[threadIdx.y].splitPlane.w,
					s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
					s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
			}

			if(s_task[threadIdx.y].splitPlane.y != 0.f && (s_task[threadIdx.y].splitPlane.y*(-s_task[threadIdx.y].splitPlane.w) < s_task[threadIdx.y].bbox.m_mn.y || s_task[threadIdx.y].splitPlane.y*(-s_task[threadIdx.y].splitPlane.w) > s_task[threadIdx.y].bbox.m_mx.y))
			{
				printf("Y-split outside! (%f, %f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)\n",
					s_task[threadIdx.y].splitPlane.x, s_task[threadIdx.y].splitPlane.y, s_task[threadIdx.y].splitPlane.z, s_task[threadIdx.y].splitPlane.w,
					s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
					s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
			}

			if(s_task[threadIdx.y].splitPlane.z != 0.f && (s_task[threadIdx.y].splitPlane.z*(-s_task[threadIdx.y].splitPlane.w) < s_task[threadIdx.y].bbox.m_mn.z || s_task[threadIdx.y].splitPlane.z*(-s_task[threadIdx.y].splitPlane.w) > s_task[threadIdx.y].bbox.m_mx.z))
			{
				printf("Z-split outside! (%f, %f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)\n",
					s_task[threadIdx.y].splitPlane.x, s_task[threadIdx.y].splitPlane.y, s_task[threadIdx.y].splitPlane.z, s_task[threadIdx.y].splitPlane.w,
					s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
					s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
			}
		}
#endif

		taskPrepareNext(tid, taskIdx, TaskType_SplitParallel);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}

	ASSERT_DIVERGENCE("taskFinishSplitParallel bottom", tid);
}

#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
// Initilizes the split info
__device__ void taskFinishInit(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishInit top", tid);

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		// Advance the automaton in shared memory
		int triStart = s_task[threadIdx.y].triStart;
		int triEnd = s_task[threadIdx.y].triEnd;

		s_task[threadIdx.y].type = TaskType_BinTriangles;
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
		s_task[threadIdx.y].unfinished = taskWarpSubtasksZero(triEnd - triStart);
		//s_task[threadIdx.y].unfinished = taskWarpSubtasksZero(triEnd - triStart)*WARP_SIZE;
#else
		s_task[threadIdx.y].unfinished = taskWarpSubtasksZero(triEnd - triStart)*WARP_SIZE;
#endif
		s_task[threadIdx.y].step = 0;
		s_sharedData[threadIdx.y][0] = -1;
	}

	ASSERT_DIVERGENCE("taskFinishInit mid", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_InitMemory);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}

	ASSERT_DIVERGENCE("taskFinishInit bottom", tid);
}

//------------------------------------------------------------------------

// Finishes a split task
__device__ void taskFinishBinning(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishBinning top", tid);

	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		// Advance the automaton in shared memory
		s_task[threadIdx.y].unfinished = taskWarpSubtasksZero(triEnd - triStart);
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
		s_task[threadIdx.y].type = TaskType_ReduceBins;
		s_task[threadIdx.y].unfinished = taskWarpSubtasks(NUM_WARPS)*WARP_SIZE*2; // ChildData are reduced in parallel
		s_task[threadIdx.y].step = 0;
#else
		s_task[threadIdx.y].type = taskChooseScanType(s_task[threadIdx.y].unfinished); 
		s_task[threadIdx.y].step = 0;
#endif
		s_sharedData[threadIdx.y][0] = -1;
	}

	ASSERT_DIVERGENCE("taskFinishBinning mid", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		// Choose best plane
#if BINNING_TYPE == 2
		// Compute best plane
		//SplitArray *splitArray = &(g_redSplits[taskIdx]);
		// Reduced left and right data
		//ChildData* left = (ChildData*)&splitArray->splits[tid].children[0];
		//ChildData* right = (ChildData*)&splitArray->splits[tid].children[1];
		SplitDataTri split1 = g_splitStack[taskIdx].splits[tid];
#if PLANE_COUNT > 32
		SplitDataTri split2 = g_splitStack[taskIdx].splits[tid+WARP_SIZE];
#endif

		volatile float* red = (volatile float*)&s_sharedData[threadIdx.y][0];
		red[tid] = CUDART_INF_F;
		
		if(tid < PLANE_COUNT)
		{
			float4 plane1;
#if PLANE_COUNT > 32
			float4 plane2;
#endif
			volatile CudaAABB& bbox = s_task[threadIdx.y].bbox;
#if SPLIT_TYPE == 4
			int axis = s_task[threadIdx.y].axis;
			//int axis = taskAxis(s_task[threadIdx.y].splitPlane, s_task[threadIdx.y].bbox, s_sharedData[threadIdx.y][0], s_task[threadIdx.y].axis);
			findPlaneRobin(tid, bbox, axis, plane);
#elif SPLIT_TYPE == 5
			findPlaneAABB(tid, bbox, plane1);
#if PLANE_COUNT > 32
			findPlaneAABB(tid+WARP_SIZE, bbox, plane2);
#endif
#elif SPLIT_TYPE == 6
			if(triEnd - triStart < c_env.childLimit)
				findPlaneTriAA(tid, c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart, s_task[threadIdx.y].triIdxCtr), triStart, triEnd, plane);
			else
				findPlaneAABB(tid, bbox, plane);
#endif

			float areaLeft, areaRight;
			int dim = getPlaneDimension(plane1);
			switch(dim)
			{
			case 0:
				areaAABBX(bbox, plane1.w, areaLeft, areaRight);
				break;
			case 1:
				areaAABBY(bbox, plane1.w, areaLeft, areaRight);
				break;
			case 2:
				areaAABBZ(bbox, plane1.w, areaLeft, areaRight);
				break;
			}

			// Compute cost
			/*CudaAABB bboxLeft;
			bboxLeft.m_mn.x = orderedIntToFloat(left->bbox.m_mn.x);
			bboxLeft.m_mn.y = orderedIntToFloat(left->bbox.m_mn.y);
			bboxLeft.m_mn.z = orderedIntToFloat(left->bbox.m_mn.z);

			bboxLeft.m_mx.x = orderedIntToFloat(left->bbox.m_mx.x);
			bboxLeft.m_mx.y = orderedIntToFloat(left->bbox.m_mx.y);
			bboxLeft.m_mx.z = orderedIntToFloat(left->bbox.m_mx.z);*/

			/*CudaAABB bboxLeft = *(CudaAABB*)&bbox;
			int dim = getPlaneDimension(plane);
			switch(dim)
			{
			case 0: bboxLeft.m_mx.x = fminf(bboxLeft.m_mx.x, plane.w); break;
			case 1: bboxLeft.m_mx.y = fminf(bboxLeft.m_mx.y, plane.w); break;
			case 2: bboxLeft.m_mx.z = fminf(bboxLeft.m_mx.z, plane.w); break;
			}
			//*(((float*)&bboxLeft.m_mx.x)-(int)plane.y-(int)plane.z*2) = fminf(*(((float*)&bboxLeft.m_mx.x)-(int)plane.y-(int)plane.z*2), plane.w);*/

#ifdef BBOX_TEST
			if(bboxLeft.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x)
				printf("Min left x cost bound error %f should be %f!\n", bboxLeft.m_mn.x, s_task[threadIdx.y].bbox.m_mn.x);
			if(bboxLeft.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y)
				printf("Min left y cost bound error %f should be %f!\n", bboxLeft.m_mn.y, s_task[threadIdx.y].bbox.m_mn.y);
			if(bboxLeft.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z)
				printf("Min left z cost bound error %f should be %f!\n", bboxLeft.m_mn.z, s_task[threadIdx.y].bbox.m_mn.z);

			if(bboxLeft.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x)
				printf("Max left x cost bound error %f should be %f!\n", bboxLeft.m_mx.x, s_task[threadIdx.y].bbox.m_mx.x);
			if(bboxLeft.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y)
				printf("Max left y cost bound error %f should be %f!\n", bboxLeft.m_mx.y, s_task[threadIdx.y].bbox.m_mx.y);
			if(bboxLeft.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
				printf("Max left z cost bound error %f should be %f!\n", bboxLeft.m_mx.z, s_task[threadIdx.y].bbox.m_mx.z);
#endif

			//float leftCnt = (float)left->cnt;
			float leftCnt = (float)split1.tf;
			float leftCost = areaLeft*leftCnt;
			//float leftCost = areaAABB(bboxLeft)*leftCnt;
			/*CudaAABB bboxRight;
			bboxRight.m_mn.x = orderedIntToFloat(right->bbox.m_mn.x);
			bboxRight.m_mn.y = orderedIntToFloat(right->bbox.m_mn.y);
			bboxRight.m_mn.z = orderedIntToFloat(right->bbox.m_mn.z);

			bboxRight.m_mx.x = orderedIntToFloat(right->bbox.m_mx.x);
			bboxRight.m_mx.y = orderedIntToFloat(right->bbox.m_mx.y);
			bboxRight.m_mx.z = orderedIntToFloat(right->bbox.m_mx.z);*/

			/*CudaAABB bboxRight = *(CudaAABB*)&bbox;
			// Clip by candidate plane
			// OPTIMIZE: Move to taskFinishBinning
			switch(dim)
			{
			case 0: bboxRight.m_mn.x = fmaxf(bboxRight.m_mn.x, plane.w); break;
			case 1: bboxRight.m_mn.y = fmaxf(bboxRight.m_mn.y, plane.w); break;
			case 2: bboxRight.m_mn.z = fmaxf(bboxRight.m_mn.z, plane.w); break;
			}
			//*(((float*)&bboxRight.m_mn.x)-(int)plane.y-(int)plane.z*2) = fmaxf(*(((float*)&bboxRight.m_mn.x)-(int)plane.y-(int)plane.z*2), plane.w);*/

#ifdef BBOX_TEST
			if(bboxRight.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x)
				printf("Min right x cost bound error %f should be %f!\n", bboxRight.m_mn.x, s_task[threadIdx.y].bbox.m_mn.x);
			if(bboxRight.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y)
				printf("Min right y cost bound error %f should be %f!\n", bboxRight.m_mn.y, s_task[threadIdx.y].bbox.m_mn.y);
			if(bboxRight.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z)
				printf("Min right z cost bound error %f should be %f!\n", bboxRight.m_mn.z, s_task[threadIdx.y].bbox.m_mn.z);

			if(bboxRight.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x)
				printf("Max right x cost bound error %f should be %f!\n", bboxRight.m_mx.x, s_task[threadIdx.y].bbox.m_mx.x);
			if(bboxRight.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y)
				printf("Max right y cost bound error %f should be %f!\n", bboxRight.m_mx.y, s_task[threadIdx.y].bbox.m_mx.y);
			if(bboxRight.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
				printf("Max right z cost bound error %f should be %f!\n", bboxRight.m_mx.z, s_task[threadIdx.y].bbox.m_mx.z);
#endif

			//float rightCnt = (float)right->cnt;
			float rightCnt = (float)split1.tb;
			float rightCost = areaRight*rightCnt;
			//float rightCost = areaAABB(bboxRight)*rightCnt;

			float cost1 = leftCost + rightCost;

#if PLANE_COUNT > 32
			dim = getPlaneDimension(plane2);
			switch(dim)
			{
			case 0:
				areaAABBX(bbox, plane2.w, areaLeft, areaRight);
				break;
			case 1:
				areaAABBY(bbox, plane2.w, areaLeft, areaRight);
				break;
			case 2:
				areaAABBZ(bbox, plane2.w, areaLeft, areaRight);
				break;
			}
			leftCnt = (float)split2.tf;
			leftCost = areaLeft*leftCnt;
			rightCnt = (float)split2.tb;
			rightCost = areaRight*rightCnt;

			float cost2 = leftCost + rightCost;
			float cost = min(cost1, cost2);
#else
			float cost = cost1;
#endif

			// Reduce the best cost within the warp
			red[tid] = cost;
			reduceWarp(tid, red, min);

			s_owner[threadIdx.y][0] = -1; // Mark as no split
			// Return the best plane for this warp
			if(__ffs(__ballot(red[tid] == cost)) == tid+1) // First thread with such condition, OPTIMIZE: Can be also computed by overwrite and test: better?
			{
#if PLANE_COUNT > 32
				if(cost == cost1)
				{
					s_task[threadIdx.y].splitPlane.x = plane1.x;
					s_task[threadIdx.y].splitPlane.y = plane1.y;
					s_task[threadIdx.y].splitPlane.z = plane1.z;
					s_task[threadIdx.y].splitPlane.w = plane1.w;
					leftCnt = (float)split1.tf;
					rightCnt = (float)split1.tb;
				}
				else
				{
					s_task[threadIdx.y].splitPlane.x = plane2.x;
					s_task[threadIdx.y].splitPlane.y = plane2.y;
					s_task[threadIdx.y].splitPlane.z = plane2.z;
					s_task[threadIdx.y].splitPlane.w = plane2.w;
					leftCnt = (float)split2.tf;
					rightCnt = (float)split2.tb;
				}
#else
				s_task[threadIdx.y].splitPlane.x = plane1.x;
				s_task[threadIdx.y].splitPlane.y = plane1.y;
				s_task[threadIdx.y].splitPlane.z = plane1.z;
				s_task[threadIdx.y].splitPlane.w = plane1.w;
				leftCnt = (float)split1.tf;
				rightCnt = (float)split1.tb;
#endif
				s_task[threadIdx.y].bestCost = cost;
				//s_owner[threadIdx.y][0] = tid;

				/*if(s_task[threadIdx.y].depth < 4)
				{
					printf("NodeIdx %d, ParentIdx %d. Plane: (%.2f, %.2f, %.2f, %.2f)\n", s_task[threadIdx.y].nodeIdx, s_task[threadIdx.y].parentIdx, plane.x, plane.y, plane.z, plane.w);
					printf("NodeIdx %d, ParentIdx %d. Box: (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", s_task[threadIdx.y].nodeIdx, s_task[threadIdx.y].parentIdx, s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
						s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
					printf("BoxLeft: (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", bboxLeft.m_mn.x, bboxLeft.m_mn.y, bboxLeft.m_mn.z,
						bboxLeft.m_mx.x, bboxLeft.m_mx.y, bboxLeft.m_mx.z);
					printf("BoxRight: (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", bboxRight.m_mn.x, bboxRight.m_mn.y, bboxRight.m_mn.z,
						bboxRight.m_mx.x, bboxRight.m_mx.y, bboxRight.m_mx.z);
				}*/
				
				// Allocate memory for children
				int termCrit;
				bool leftLeaf, rightLeaf;
				bool leaf = taskTerminationCriteria(triEnd-triStart, leftCnt, rightCnt, s_task[threadIdx.y].bbox, s_task[threadIdx.y].splitPlane, termCrit, leftLeaf, rightLeaf);
				// OPTIMIZE: Only allocate if they are large enough for and inner node?
				if(!leaf)
					allocChildren(s_task[threadIdx.y].dynamicMemoryLeft, s_task[threadIdx.y].dynamicMemoryRight, leftCnt, rightCnt);

				if(leftCnt == 0 || rightCnt == 0 || leaf) // One leaf is empty, use parent memory with no copy (skip partition)
				{
					s_task[threadIdx.y].triLeft = leftCnt;
					s_task[threadIdx.y].triRight = rightCnt;
					s_owner[threadIdx.y][0] = -100;
				}

				// Set child bounding boxes
				/*CudaAABB& t_bboxLeft = g_taskStackBVH.tasks[taskIdx].bboxLeft;
				t_bboxLeft.m_mn.x = bboxLeft.m_mn.x;
				t_bboxLeft.m_mn.y = bboxLeft.m_mn.y;
				t_bboxLeft.m_mn.z = bboxLeft.m_mn.z;

				t_bboxLeft.m_mx.x = bboxLeft.m_mx.x;
				t_bboxLeft.m_mx.y = bboxLeft.m_mx.y;
				t_bboxLeft.m_mx.z = bboxLeft.m_mx.z;

				CudaAABB& t_bboxRight = g_taskStackBVH.tasks[taskIdx].bboxRight;
				t_bboxRight.m_mn.x = bboxRight.m_mn.x;
				t_bboxRight.m_mn.y = bboxRight.m_mn.y;
				t_bboxRight.m_mn.z = bboxRight.m_mn.z;

				t_bboxRight.m_mx.x = bboxRight.m_mx.x;
				t_bboxRight.m_mx.y = bboxRight.m_mx.y;
				t_bboxRight.m_mx.z = bboxRight.m_mx.z;*/
			}
		}

		if(s_owner[threadIdx.y][0] == -1)
		{
			taskPrepareNext(tid, taskIdx, TaskType_BinTriangles);
		}
		else if(s_owner[threadIdx.y][0] == -100) // Finish the whole sort
		{
#ifndef DEBUG_INFO
			// Ensure we save valid child box data
			//taskLoadSecondFromGMEM(tid, taskIdx, s_task[threadIdx.y]); // Load bbox data
#endif

			taskFinishTask(tid, taskIdx);
		}

		/*if(s_owner[threadIdx.y][0] == -1) // No split found, do object median
		{
			s_task[threadIdx.y].dynamicMemoryLeft = 0;
			s_task[threadIdx.y].dynamicMemoryRight = 0;
#ifndef DEBUG_INFO
			// Ensure we save valid child box data
			//taskLoadSecondFromGMEM(tid, taskIdx, s_task[threadIdx.y]); // Load bbox data
#endif

			taskFinishTask(tid, taskIdx);
		}
		else
		{
			// Copy boxes
			left = (ChildData*)&splitArray->splits[s_owner[threadIdx.y][0]].children[0];
			right = (ChildData*)&splitArray->splits[s_owner[threadIdx.y][0]].children[1];

//#ifdef SPLIT_TEST
			//if(tid == 0)
			//{
			//	if(left->cnt == triEnd - triStart)
			//	{
			//		printf("Failed node left in task %d (%d x %d), %d, %d!\n", taskIdx, left->cnt, triEnd - triStart, right->cnt, s_task[threadIdx.y].depth);
			//	}
			//	if(right->cnt == triEnd - triStart)
			//	{
			//		printf("Failed node right in task %d (%d x %d), %d, %d!\n", taskIdx, right->cnt, triEnd - triStart, left->cnt, s_task[threadIdx.y].depth);
			//	}
			//}
//#endif

			float* t_bboxLeft = (float*)&(g_taskStackBVH.tasks[taskIdx].bboxLeft);
			const float* g_bboxLeft = (const float*)&(left->bbox);

			float* t_bboxRight = (float*)&(g_taskStackBVH.tasks[taskIdx].bboxRight);
			const float* g_bboxRight = (const float*)&(right->bbox);
			// Copy CudaAABB from corresponding task
			if(tid < sizeof(CudaAABB)/sizeof(float))
			{
				float cor;
				if(tid < sizeof(float3)/sizeof(float))
					cor = -c_env.epsilon;
				else
					cor = c_env.epsilon;

#if BINNING_TYPE == 0 || BINNING_TYPE == 1
				t_bboxLeft[tid] = g_bboxLeft[tid];
				t_bboxRight[tid] = g_bboxRight[tid];
#else
				t_bboxLeft[tid] = orderedIntToFloat(*(int*)&g_bboxLeft[tid])+cor;
				t_bboxRight[tid] = orderedIntToFloat(*(int*)&g_bboxRight[tid])+cor;
#endif

#ifdef DEBUG_INFO
				volatile float* s_bboxLeft = (volatile float*)&(s_task[threadIdx.y].bboxLeft);
				volatile float* s_bboxRight = (volatile float*)&(s_task[threadIdx.y].bboxRight);

#if BINNING_TYPE == 0 || BINNING_TYPE == 1
				s_bboxLeft[tid] = g_bboxLeft[tid];
				s_bboxRight[tid] = g_bboxRight[tid];
#else
				s_bboxLeft[tid] = orderedIntToFloat(*(int*)&g_bboxLeft[tid]);
				s_bboxRight[tid] = orderedIntToFloat(*(int*)&g_bboxRight[tid]);
#endif
#endif
			}
			taskPrepareNext(tid, taskIdx, TaskType_BinTriangles);
		}*/
//#else
		//taskPrepareNext(tid, taskIdx, TaskType_BinTriangles);	
#endif
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}

	ASSERT_DIVERGENCE("taskFinishBinning bottom", tid);
}

//------------------------------------------------------------------------

#if BINNING_TYPE == 0 || BINNING_TYPE == 1

__device__ void taskFinishReduce(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishReduce top", tid);

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished = taskNextPhaseCountReduceMultiplied(step, 0, NUM_WARPS, WARP_SIZE*2); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step++;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to PPS1
		{
			// Compute best plane
			SplitArray *splitArray = &(g_redSplits[taskIdx]);
			// Reduced left and right data
			ChildData* left = (ChildData*)&splitArray->splits[0][tid].children[0];
			ChildData* right = (ChildData*)&splitArray->splits[0][tid].children[1];

#ifdef SPLIT_TEST
			if(left->cnt+right->cnt != s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triStart)
			{
				printf("Failed reduction in task %d!\n", taskIdx);
				g_taskStackBVH.unfinished = 1;
			}
#endif

			// Compute cost
			float leftCost = areaAABB(left->bbox)*(float)left->cnt;
			float rightCost = areaAABB(right->bbox)*(float)right->cnt;
			float cost = leftCost + rightCost;

			// Reduce the best cost within the warp
			volatile float* red = (volatile float*)&s_sharedData[threadIdx.y][0];
			red[tid] = cost;
			reduceWarp(tid, red, min);

			float4 plane;
			volatile CudaAABB& bbox = s_task[threadIdx.y].bbox;
#if SPLIT_TYPE == 4
			int axis = s_task[threadIdx.y].axis;
			//int axis = taskAxis(s_task[threadIdx.y].splitPlane, s_task[threadIdx.y].bbox, s_sharedData[threadIdx.y][0], s_task[threadIdx.y].axis);
			findPlaneRobin(tid, bbox, axis, plane);
#elif SPLIT_TYPE == 5
			findPlaneAABB(tid, bbox, plane);
#elif SPLIT_TYPE == 6
			if(s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triStart < c_env.childLimit)
				findPlaneTriAA(tid, c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, s_task[threadIdx.y].triEnd-s_task[threadIdx.y].triStart, s_task[threadIdx.y].triIdxCtr), s_task[threadIdx.y].triStart, s_task[threadIdx.y].triEnd, plane);
			else
				findPlaneAABB(tid, bbox, plane);
#endif

			// Return the best plane for this warp
			if(__ffs(__ballot(red[tid] == cost)) == tid+1) // First thread with such condition, OPTIMIZE: Can be also computed by overwrite and test: better?
			{
				s_task[threadIdx.y].splitPlane.x = plane.x;
				s_task[threadIdx.y].splitPlane.y = plane.y;
				s_task[threadIdx.y].splitPlane.z = plane.z;
				s_task[threadIdx.y].splitPlane.w = plane.w;
				s_task[threadIdx.y].bestCost = cost;
				s_sharedData[threadIdx.y][0] = tid;

				// Allocate memory for children
				allocChildren(s_task[threadIdx.y].dynamicMemoryLeft, s_task[threadIdx.y].dynamicMemoryRight, leftCnt, rightCnt);

/*#ifdef SPLIT_TEST
				printf("Chosen split for task %d: (%f, %f, %f, %f)\n", taskIdx, 
					s_task[threadIdx.y].splitPlane.x, s_task[threadIdx.y].splitPlane.y, s_task[threadIdx.y].splitPlane.z, s_task[threadIdx.y].splitPlane.w);
#endif*/
			}

			// Copy boxes
			left = (ChildData*)&splitArray->splits[0][s_sharedData[threadIdx.y][0]].children[0];
			right = (ChildData*)&splitArray->splits[0][s_sharedData[threadIdx.y][0]].children[1];

			volatile float* bboxLeft = (volatile float*)&(g_taskStackBVH.tasks[taskIdx].bboxLeft);
			const float* g_bboxLeft = (const float*)&(left->bbox);

			volatile float* bboxRight = (volatile float*)&(g_taskStackBVH.tasks[taskIdx].bboxRight);
			const float* g_bboxRight = (const float*)&(right->bbox);
			// Copy CudaAABB from corresponding task
			if(tid < sizeof(CudaAABB)/sizeof(float))
			{
				bboxLeft[tid] = g_bboxLeft[tid];
				bboxRight[tid] = g_bboxRight[tid];

/*#ifdef SPLIT_TEST
				if(tid == 0)
				{
					printf("Left (%f, %f, %f) - (%f, %f, %f)\n",
					left->bbox.m_mn.x, left->bbox.m_mn.y, left->bbox.m_mn.z,
					left->bbox.m_mx.x, left->bbox.m_mx.y, left->bbox.m_mx.z);
					printf("Right (%f, %f, %f) - (%f, %f, %f)\n",
					right->bbox.m_mn.x, right->bbox.m_mn.y, right->bbox.m_mn.z,
					right->bbox.m_mx.x, right->bbox.m_mx.y, right->bbox.m_mx.z);
				}
#endif*/

#ifdef DEBUG_INFO
				volatile float* s_bboxLeft = (volatile float*)&(s_task[threadIdx.y].bboxLeft);
				volatile float* s_bboxRight = (volatile float*)&(s_task[threadIdx.y].bboxRight);

				s_bboxLeft[tid] = g_bboxLeft[tid];
				s_bboxRight[tid] = g_bboxRight[tid];
#endif
			}

			// Advance
			int triStart = s_task[threadIdx.y].triStart;
			int triEnd = s_task[threadIdx.y].triEnd;

			s_task[threadIdx.y].unfinished = taskWarpSubtasksZero(triEnd - triStart);
			s_task[threadIdx.y].type = taskChooseScanType(s_task[threadIdx.y].unfinished);
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
	}

	ASSERT_DIVERGENCE("taskFinishReduce bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_ReduceBins);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

#endif

#endif

//------------------------------------------------------------------------

__device__ void taskFinishSortPPS1(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortPPS1 top", tid);

	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished = taskNextPhaseCountPPS(step, triStart, triEnd); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step++;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to SORT1
		{
			int triRight = triEnd - getPPSTrisPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart)[triEnd - 1]; // Must be volatile
#ifdef RAYTRI_TEST
			if(triRight < triStart || triRight > triEnd)
			{
				printf("PPS1 error triStart %d, triRight %d, triEnd %d!\n", triStart, triRight, triEnd);
				//triRight = triStart;
			}
#endif

			unfinished = 0; // OPTIMIZE: Should not be needed
			if(triRight != triStart && triRight != triEnd)
			{
				s_task[threadIdx.y].type = TaskType_Sort_SORT1;
				unfinished = taskWarpSubtasksZero(triEnd - triRight);
			}

			if(unfinished == 0) // Nothing to sort -> Move to AABB_Min
			{
				triRight = triStart + (triEnd - triStart) / 2; // Force split on unsubdivided task
				s_task[threadIdx.y].type = taskChooseAABBType();
				unfinished = taskWarpSubtasksZero(triEnd - triStart);
			}

			s_task[threadIdx.y].triRight = triRight;
			s_task[threadIdx.y].unfinished = unfinished;
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
	}

	ASSERT_DIVERGENCE("taskFinishSortPPS1 bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_PPS1);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

__device__ void taskFinishSortPPSUp(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortPPSUp top", tid);

	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished = taskNextPhaseCountPPSUp(step, triStart, triEnd); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step += LOG_WARP_SIZE;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to PPS1_Down
		{
			getPPSTrisPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart)[triEnd - 1] = 0; // Set the last element to 0 as required by Harris scan

			s_task[threadIdx.y].type = TaskType_Sort_PPS1_Down;
			// Make the level multiple of LOG_WARP_SIZE to end at step 0
			int level = taskTopTreeLevel(triStart, triEnd);
			s_task[threadIdx.y].unfinished = 1; // Top levels for tris
			s_task[threadIdx.y].step = level;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
	}

	ASSERT_DIVERGENCE("taskFinishSortPPSUp bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_PPS1_Up);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

__device__ void taskFinishSortPPSDown(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortPPSDown top", tid);

	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished = taskNextPhaseCountPPSDown(step, triStart, triEnd); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step -= LOG_WARP_SIZE;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to SORT1
		{
			int triRight = triEnd - getPPSTrisPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart)[triEnd - 1]; // Must be volatile
#ifdef RAYTRI_TEST
			if(triRight < triStart || triRight > triEnd)
			{
				printf("PPS1 error triStart %d, triRight %d, triEnd %d!\n", triStart, triRight, triEnd);
			}
#endif

			unfinished = 0; // OPTIMIZE: Should not be needed
			if(triRight != triStart && triRight != triEnd)
			{
				s_task[threadIdx.y].type = TaskType_Sort_SORT1;
				unfinished = taskWarpSubtasksZero(triEnd - triRight);
			}

			if(unfinished == 0) // Nothing to sort -> Move to AABB_Min
			{
				triRight = triStart + (triEnd - triStart) / 2; // Force split on unsubdivided task
				s_task[threadIdx.y].type = taskChooseAABBType();
				unfinished = taskWarpSubtasksZero(triEnd - triStart);
			}

			s_task[threadIdx.y].triRight = triRight;
			s_task[threadIdx.y].unfinished = unfinished;
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
	}

	ASSERT_DIVERGENCE("taskFinishSortPPSDown bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_PPS1_Down);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

__device__ void taskFinishSortSORT1(int tid, int taskIdx, unsigned countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortSORT1 top", tid);

	int triStart = s_task[threadIdx.y].triStart;
	int triRight = s_task[threadIdx.y].triRight;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished;

		if(step == 0) // Move to next phase of the current task
		{
			unfinished = taskWarpSubtasksZero(triRight - triStart);

			s_task[threadIdx.y].unfinished = unfinished;
			step++;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to AABB
		{
#ifdef RAYTRI_TEST
			if(s_task[threadIdx.y].type != TaskType_RayTriTestSORT1)
			{
				unfinished = taskWarpSubtasks(triEnd - triStart);
				s_task[threadIdx.y].unfinished = unfinished;
				s_task[threadIdx.y].type = TaskType_RayTriTestSORT1;
				s_task[threadIdx.y].step = 0;
				s_sharedData[threadIdx.y][0] = -1;
			}
			else
#endif

			{
				s_sharedData[threadIdx.y][0] = -100; // Make the other threads join for the finish
			}
		}
	}

	//ASSERT_DIVERGENCE("taskFinishSortSORT1 bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_SORT1);
	}
	else if(s_sharedData[threadIdx.y][0] == -100) // Finish the whole sort
	{
#ifndef DEBUG_INFO
		// Ensure we save valid child box data
		//taskLoadSecondFromGMEM(tid, taskIdx, s_task[threadIdx.y]); // Load bbox data
#endif

		taskFinishTask(tid, taskIdx);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

__device__ void taskFinishPartition(int tid, int taskIdx, unsigned countDown)
{
	ASSERT_DIVERGENCE("taskFinishPartition top", tid);

	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		// Load correct split positions from global memory
		int triLeft = g_taskStackBVH.tasks[taskIdx].triLeft;
		int triRight = g_taskStackBVH.tasks[taskIdx].triRight;

		s_task[threadIdx.y].triLeft = triLeft;
		s_task[threadIdx.y].triRight = triRight;

#ifdef RAYTRI_TEST
		if(tid == 0 && triLeft != triRight)
			printf("Incorrect partition on interval %d - %d (%d x %d)!\n", triStart, triEnd, triLeft, triRight);

		if(s_task[threadIdx.y].type != TaskType_RayTriTestSORT1)
		{
			s_task[threadIdx.y].unfinished = taskWarpSubtasks(triEnd - triStart);
			s_task[threadIdx.y].type = TaskType_RayTriTestSORT1;
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1;
		}
		else
#endif

		{
			s_sharedData[threadIdx.y][0] = -100; // Make the other threads join for the finish
		}
	}

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_SORT1);
	}
	else if(s_sharedData[threadIdx.y][0] == -100) // Finish the whole sort
	{
#ifndef DEBUG_INFO
		// Ensure we save valid child box data
		//taskLoadSecondFromGMEM(tid, taskIdx, s_task[threadIdx.y]); // Load bbox data
#endif

		taskFinishTask(tid, taskIdx);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

// Compute number of triangles on both sides of the plane
__device__ void trianglesPlanePosition(int triStart, int triEnd, const float4& plane, int& tb, int& tf)
{
	ASSERT_DIVERGENCE("trianglesPlanePosition", threadIdx.x);

	//int tris = triEnd - triStart;
	//int desiredSamples = getNumberOfSamples(tris);

	tb = 0;
	tf = 0;

	//int step = tris / desiredSamples;
	//if(step < 1)
	//	step = 1;
	int step = 1;

	for(int triPos = triStart; triPos < triEnd; triPos += step)
	{
		int* inIdx = getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart);
		int triIdx = inIdx[triPos]*3;

		// Fetch triangle
		float3 v0, v1, v2;
		taskFetchTri(c_bvh_in.tris, triIdx, v0, v1, v2);

#if (TRIANGLE_CLIPPING != 5 && TRIANGLE_CLIPPING != 6)
		int pos = getPlanePosition(plane, v0, v1, v2);
#if TRIANGLE_CLIPPING == 2 || TRIANGLE_CLIPPING == 3
		if(pos == 0)
		{
			pos = getPlanePositionClipped(plane, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
		}
#endif
#else
		int pos = getTriChildOverlap(plane, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
		/*int pos = getPlanePosition(plane, v0, v1, v2);
		if(pos == 0)
		{
			pos = getPlanePositionClipped(plane, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
		}*/
#endif
		
		if(pos <= 0)
			tb++;
		if (pos >= 0)
			tf++;
	}
}

//------------------------------------------------------------------------

// Compute cost of several splitting strategies and choose the best one
__device__ void splitCost(int tid, int subtask, int triStart, int triEnd, const volatile CudaAABB& bbox,
	volatile float4& bestPlane, volatile float& bestCost, volatile int& bestOrder)
{
	ASSERT_DIVERGENCE("splitCost top", tid);

	// Each thread computes its plane
	int planePos = subtask*WARP_SIZE + tid;

	// Number of evaluated candidates per plane
#if SPLIT_TYPE == 1
	int tris = triEnd - triStart;

#if 0 // SQRT candidates
	int evaluatedCandidates = getNumberOfSamples(tris);
	int evaluatedCandidates = 1;
	int numPlanes = taskWarpSubtasks(c_env.optPlaneSelectionOverhead * tris/evaluatedCandidates)*WARP_SIZE;
#elif 0 // Fixed candidates
	int numPlanes = 32768;
#else // All candidates
	int numPlanes = tris*6;
#endif
#else
	int numPlanes = WARP_SIZE;
#endif

	float areaLeft, areaRight;
	float4 plane;
#if SPLIT_TYPE != 1
	findPlaneAABB(planePos, bbox, areaLeft, areaRight, plane, numPlanes);
#else
#if 0
	findPlaneAABB(planePos, bbox, areaLeft, areaRight, plane, numPlanes);
#else
	findPlaneTriAABB(planePos, (float4*)c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart), triStart, bbox, areaLeft, areaRight, plane, numPlanes);
#endif
#endif

	// Count the number of rays and triangles on both sides of the plane
	int tb, tf;
	trianglesPlanePosition(triStart, triEnd, plane, tb, tf);

	// Compute the plane cost
	float leftCost = areaLeft*(float)tb;
	float rightCost = areaRight*(float)tf;
	float cost = c_env.optCt + c_env.optCi*(leftCost + rightCost)/areaAABB(bbox);

	// Reduce the best cost within the warp
	volatile float* red = (volatile float*)&s_sharedData[threadIdx.y][0];
	red[tid] = cost;
	reduceWarp(tid, red, min);
	//reduceWarp<float, Min<float>>(tid, red, Min<float>());
	
	// Return the best plane for this warp
	if(__ffs(__ballot(red[tid] == cost)) == tid+1) // First thread with such condition, OPTIMIZE: Can be also computed by overwrite and test: better?
	{
		bestPlane.x = plane.x;
		bestPlane.y = plane.y;
		bestPlane.z = plane.z;
		bestPlane.w = plane.w;
		bestCost = cost;

		// Allocate memory for children
		// OPTIMIZE: Only allocate if they are large enough for and inner node?
		s_task[threadIdx.y].triLeft = tb;
		s_task[threadIdx.y].triRight = tf;
	}

#ifdef SPLIT_TEST
	if(!__any(red[tid] == cost) || red[tid] < 0.f || cost < red[tid])
		printf("Errorneous cost reduction tid %d, %cost %f, best cost %f!\n", tid, cost, red[tid]);

	//if(tid == 0 && s_task[threadIdx.y].splitPlane.x > 0.f && s_task[threadIdx.y].splitPlane.y > 0.f && s_task[threadIdx.y].splitPlane.z > 0.f && s_task[threadIdx.y].splitPlane.w > 0.f)
	//	printf("In Error plane (%f, %f, %f, %f)!\n", s_task[threadIdx.y].splitPlane.x, s_task[threadIdx.y].splitPlane.y, s_task[threadIdx.y].splitPlane.z, s_task[threadIdx.y].splitPlane.w);
#endif

	ASSERT_DIVERGENCE("splitCost bottom", tid);
}

//------------------------------------------------------------------------

#if SPLIT_TYPE == 3

// Compute all data needed for each splitting plane in parallel
__device__ void splitCostParallel(int tid, int subtask, int taskIdx, int triStart, int triEnd, const volatile CudaAABB& bbox)
{
	ASSERT_DIVERGENCE("splitCostParallel top", tid);

	// Each thread computes its ray-plane or tri-plane task
	int warptask = subtask*WARP_SIZE;
	int task = warptask + tid;

	int tris = triEnd - triStart;

	// Number of evaluated candidates per plane
	int tritasks = taskWarpSubtasks(getNumberOfSamples(tris));
	int evaluatedTris = min(tritasks*WARP_SIZE, tris); // Choose either WARP_SIZE multiple of sampled tris or all tris

	float sumWeights = c_env.optAxisAlignedWeight + c_env.optTriangleBasedWeight;
	// Assign the planes to different methods
	int numAxisAlignedPlanes = c_env.optAxisAlignedWeight/sumWeights*PLANE_COUNT;
	int numTriangleBasedPlanes = c_env.optTriangleBasedWeight/sumWeights*PLANE_COUNT;
	
	float areaLeft, areaRight;
	float4 plane;
	volatile int* red = (volatile int*)&s_sharedData[threadIdx.y][0];
	red[tid] = 0;

	int planePos = subtask % PLANE_COUNT; // Succesive subtasks do far away planes -> hopefully less lock contention in atomic operations
	findPlane(planePos, triStart, triEnd, bbox, areaLeft, areaRight, numAxisAlignedPlanes, plane);

	// Count the number of triangles on both sides of the plane
	int tb, tf;
	tb = 0;
	tf = 0;

	task = (subtask/PLANE_COUNT)*WARP_SIZE + tid;
	//int step = tris / evaluatedTris;
	//int triPos = triStart + task*step;
	float step = (float)tris / (float)evaluatedTris;
	int triPos = triStart + (int)((float)task*step);

	if(triPos < triEnd)
	{
		volatile int* inIdx = getTriIdxPtr(s_task[threadIdx.y].pivot);
		int triIdx = inIdx[triPos]*3;

		// Fetch triangle
		float3 v0, v1, v2;
		taskFetchTri(triIdx, v0, v1, v2);

		CudaAABB dummy;
		int pos = getPlaneCentroidPosition(plane, v0, v1, v2, dummy);

		if(pos < 0)
			tb++;
		else
			tf++;

		// Update the split info
		SplitDataBVH *split = &(g_splitStackBVH[taskIdx].splits[planePos]);

		// Reduce the numbers within the warp
		red[tid] = tf;
		reduceWarp(tid, red, plus);
		if(tid == 0)
			atomicAdd(&split->tf, red[tid]);

		red[tid] = tb;
		reduceWarp(tid, red, plus);
		if(tid == 0)
			atomicAdd(&split->tb, red[tid]);
	}

	ASSERT_DIVERGENCE("splitCostParallel bottom", tid);
}

//------------------------------------------------------------------------

#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1

// Compute all data needed for each bin in parallel planes and sequentially over trianlges
__device__ void binTriangles(int tid, int subtask, int taskIdx, int triStart, int triEnd, const volatile CudaAABB& bbox, int axis)
{
	// Update the bins
	int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
	SplitRed *split = &(g_redSplits[taskIdx].splits[warpIdx][tid]);

	float4 plane;
#if SPLIT_TYPE == 4
	findPlaneRobin(tid, bbox, axis, plane);
#elif SPLIT_TYPE == 5
	findPlaneAABB(tid, bbox, plane);
#elif SPLIT_TYPE == 6
	if(triEnd - triStart < c_env.childLimit)
		findPlaneTriAA(tid, c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart, s_task[threadIdx.y].triIdxCtr), triStart, triEnd, plane);
	else
		findPlaneAABB(tid, bbox, plane);
#endif

	//CudaAABB tbox;
	//int pos;

	int triPos = triStart + subtask*WARP_SIZE;
	for(int i = 0; i < WARP_SIZE && triPos < triEnd; i++, triPos++)
	{
		int* inIdx = getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart, s_task[threadIdx.y].triIdxCtr);
		int triIdx = inIdx[triPos]*3;

		// Fetch triangle
		float3 v0, v1, v2;
		taskFetchTri(triIdx, v0, v1, v2);

		CudaAABB tbox;
#if (TRIANGLE_CLIPPING != 5 && TRIANGLE_CLIPPING != 6)
		int pos = getPlanePosition(plane, v0, v1, v2);
		//pos = getPlanePosition(plane, v0, v1, v2);
#if TRIANGLE_CLIPPING == 2 || TRIANGLE_CLIPPING == 3
		if(pos == 0)
		{
			pos = getPlanePositionClipped(plane, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
		}
#endif
#else
		int pos = getTriChildOverlap(plane, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
		/*int pos = getPlanePosition(plane, v0, v1, v2);
		if(pos == 0)
		{
			pos = getPlanePositionClipped(plane, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
		}*/
#endif
		getAABB(v0, v1, v2, tbox);

		// Write immediately - LOT OF MEMORY TRANSACTIONS (SLOW) but little memory usage

		// Update the bounding boxes and the counts
		if(pos <= 0)
		{
			split->children[0].bbox.m_mn = fminf(split->children[0].bbox.m_mn, tbox.m_mn-c_env.epsilon);
			split->children[0].bbox.m_mx = fmaxf(split->children[0].bbox.m_mx, tbox.m_mx+c_env.epsilon);
			split->children[0].cnt++;
		}
		if(pos >= 0)
		{
			split->children[1].bbox.m_mn = fminf(split->children[1].bbox.m_mn, tbox.m_mn-c_env.epsilon);
			split->children[1].bbox.m_mx = fmaxf(split->children[1].bbox.m_mx, tbox.m_mx+c_env.epsilon);
			split->children[1].cnt++;
		}
	}

	// Reduce in thread, write in the end - FAST BUT WITH EXTENSIVE MEMORY USAGE
	// Can be further accelerated by using less splitting planes and parallelizing the for cycle

	// Update the bounding boxes and the counts
	/*if(pos <= 0)
	{
		split->children[0].bbox.m_mn = fminf(split->children[0].bbox.m_mn, tbox.m_mn-c_env.epsilon);
		split->children[0].bbox.m_mx = fmaxf(split->children[0].bbox.m_mx, tbox.m_mx+c_env.epsilon);
		split->children[0].cnt++;
	}
	if(pos >= 0)
	{
		split->children[1].bbox.m_mn = fminf(split->children[1].bbox.m_mn, tbox.m_mn-c_env.epsilon);
		split->children[1].bbox.m_mx = fmaxf(split->children[1].bbox.m_mx, tbox.m_mx+c_env.epsilon);
		split->children[1].cnt++;
	}*/
}

//------------------------------------------------------------------------

// Compute all data needed for each bin in parallel over triangles and planes
__device__ void binTrianglesParallel(int tid, int subtask, int taskIdx, int triStart, int triEnd, const volatile CudaAABB& bbox, int axis)
{	
	// Compute binning data
	CudaAABB tbox;
	int pos = -2; // Mark inactive threads
	//__shared__ volatile SplitRed out[NUM_WARPS_PER_BLOCK];
	float3 v0, v1, v2;

	int triPos = triStart + subtask*WARP_SIZE + tid;
	if(triPos < triEnd)
	{
		// Fetch triangle
		int* inIdx = getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart, s_task[threadIdx.y].triIdxCtr);
		int triIdx = inIdx[triPos]*3;
		taskFetchTri(triIdx, v0, v1, v2);
	}

	volatile float* red = (volatile float*)&s_sharedData[threadIdx.y][0];
	volatile int* redI = (volatile int*)&s_sharedData[threadIdx.y][0];
	int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID

	for(int planePos = 0; planePos < WARP_SIZE; planePos++)
	{
		// Update the bins
		SplitRed *split = (SplitRed*)&(g_redSplits[taskIdx].splits[warpIdx][planePos]);

		float4 plane;
#if SPLIT_TYPE == 4
		findPlaneRobin(planePos, bbox, axis, plane);
#elif SPLIT_TYPE == 5
		findPlaneAABB(planePos, bbox, plane);
#elif SPLIT_TYPE == 6
		if(triEnd - triStart < c_env.childLimit)
			findPlaneTriAA(tid, c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart, s_task[threadIdx.y].triIdxCtr), triStart, triEnd, plane);
		else
			findPlaneAABB(tid, bbox, plane);
#endif

		if(triPos < triEnd)
		{
#if (TRIANGLE_CLIPPING != 5 && TRIANGLE_CLIPPING != 6)
			pos = getPlanePosition(plane, v0, v1, v2);
#if TRIANGLE_CLIPPING == 2 || TRIANGLE_CLIPPING == 3
			if(pos == 0)
			{
				pos = getPlanePositionClipped(plane, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
			}
#endif
#else
			pos = getTriChildOverlap(plane, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
			/*pos = getPlanePosition(plane, v0, v1, v2);
			if(pos == 0)
			{
				pos = getPlanePositionClipped(plane, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
			}*/
#endif
			getAABB(v0, v1, v2, tbox);
		}

		// Warpwide update of the left child
		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		// OPTIMIZE: Segmented scan on left and right child simultaneously?
		// Reduce min
		if(pos != -2 && pos <= 0)
			red[tid] = tbox.m_mn.x - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[0].bbox.m_mn.x)
			split->children[0].bbox.m_mn.x = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.x = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos != -2 && pos <= 0)
			red[tid] = tbox.m_mn.y - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[0].bbox.m_mn.y)
			split->children[0].bbox.m_mn.y = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.y = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos != -2 && pos <= 0)
			red[tid] = tbox.m_mn.z - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[0].bbox.m_mn.z)
			split->children[0].bbox.m_mn.z = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.z = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		// Reduce max
		ifpos != -2 && pos <= 0)
			red[tid] = tbox.m_mx.x + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[0].bbox.m_mx.x)
			split->children[0].bbox.m_mx.x = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.x = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data
		
		if(pos != -2 && pos <= 0)
			red[tid] = tbox.m_mx.y + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[0].bbox.m_mx.y)
			split->children[0].bbox.m_mx.y = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.y = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos != -2 && pos <= 0)
			red[tid] = tbox.m_mx.z + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[0].bbox.m_mx.z)
			split->children[0].bbox.m_mx.z = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.z = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		redI[tid] = 0; // Save identities so that we do not work with uninitialized data
		
		// Reduce cnt
		if(pos != -2 && pos <= 0)
			redI[tid] = 1;
		reduceWarp(tid, &redI[0], plus);
		if(redI[0] != 0)
			split->children[0].cnt = split->children[0].cnt + redI[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].cnt = redI[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them
		

		// Warpwide update of the right child
		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		// OPTIMIZE: Segmented scan on left and right child simultaneously?
		// Reduce min
		if(pos >= 0)
			red[tid] = tbox.m_mn.x - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[1].bbox.m_mn.x)
			split->children[1].bbox.m_mn.x = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.x = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos >= 0)
			red[tid] = tbox.m_mn.y - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[1].bbox.m_mn.y)
			split->children[1].bbox.m_mn.y = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.y = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos >= 0)
			red[tid] = tbox.m_mn.z - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[1].bbox.m_mn.z)
			split->children[1].bbox.m_mn.z = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.z = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		// Reduce max
		if(pos >= 0)
			red[tid] = tbox.m_mx.x + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[1].bbox.m_mx.x)
			split->children[1].bbox.m_mx.x = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.x = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos >= 0)
			red[tid] = tbox.m_mx.y + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[1].bbox.m_mx.y)
			split->children[1].bbox.m_mx.y = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.y = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos >= 0)
			red[tid] = tbox.m_mx.z + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[1].bbox.m_mx.z)
			split->children[1].bbox.m_mx.z = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.z = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		redI[tid] = 0; // Save identities so that we do not work with uninitialized data

		// Reduce cnt
		if(pos >= 0)
			redI[tid] = 1;
		reduceWarp(tid, &redI[0], plus);
		if(redI[0] != 0)
			split->children[1].cnt = split->children[1].cnt + redI[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].cnt = redI[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		// Copy to GMEM
		//if(tid < sizeof(SplitRed) / sizeof(int))
		//{
		//	int* g_split = (int*)split;
		//	volatile int* s_split = (volatile int*)&out;
		//	g_split[tid] = s_split[tid];
		//}
	}
}

//------------------------------------------------------------------------

#elif BINNING_TYPE == 2

// Compute all data needed for each bin in parallel planes and sequentially over trianlges
__device__ void binTrianglesAtomic(int tid, int subtaskFirst, int subtaskLast, int taskIdx, int triStart, int triEnd, const volatile CudaAABB& bbox, int axis)
{
	if(tid < PLANE_COUNT)
	{
		float4 plane1;
#if PLANE_COUNT > 32
		float4 plane2;
#endif
#if SPLIT_TYPE == 4
		findPlaneRobin(tid, bbox, axis, plane);
#elif SPLIT_TYPE == 5
		findPlaneAABB(tid, bbox, plane1);
#if PLANE_COUNT > 32
		findPlaneAABB(tid+WARP_SIZE, bbox, plane2);
#endif
#elif SPLIT_TYPE == 6
		if(triEnd - triStart < c_env.childLimit)
			findPlaneTriAA(tid, c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart, s_task[threadIdx.y].triIdxCtr), triStart, triEnd, plane);
		else
			findPlaneAABB(tid, bbox, plane);
#endif

#if PLANE_COUNT > 32
		int cntLeft1, cntRight1, cntLeft2, cntRight2;
		cntLeft1 = cntRight1 = cntLeft2 = cntRight2 = 0;
#else
		int cntLeft1, cntRight1;
		cntLeft1 = cntRight1 = 0;
#endif

		//CudaAABB bboxLeft, bboxRight;
		// Initialize boxes
		/*bboxLeft.m_mn.x = bboxLeft.m_mn.y = bboxLeft.m_mn.z = CUDART_INF_F;
		bboxRight.m_mn.x = bboxRight.m_mn.y = bboxRight.m_mn.z = CUDART_INF_F;
		bboxLeft.m_mx.x = bboxLeft.m_mx.y = bboxLeft.m_mx.z = -CUDART_INF_F;
		bboxRight.m_mx.x = bboxRight.m_mx.y = bboxRight.m_mx.z = -CUDART_INF_F;*/

		int* inIdx = getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart);
		int triPos = triStart + subtaskFirst*WARP_SIZE*BIN_MULTIPLIER;
		int triLast = min(triStart + subtaskLast*WARP_SIZE*BIN_MULTIPLIER, triEnd);
		for(;triPos < triLast; triPos++)
		{
			// OPTIMIZE: Can the data be loaded into shared memory at once by all the threads?
			int triIdx = inIdx[triPos]*3;

			// Fetch triangle
			float3 v0, v1, v2;
			taskFetchTri(c_bvh_in.tris, triIdx, v0, v1, v2);

			//CudaAABB tbox;
#if (TRIANGLE_CLIPPING != 5 && TRIANGLE_CLIPPING != 6)
			int pos = getPlanePosition(plane1, v0, v1, v2);
#if TRIANGLE_CLIPPING == 2 || TRIANGLE_CLIPPING == 3
			if(pos == 0)
			{
				pos = getPlanePositionClipped(plane, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
			}
#endif
#else
			int pos = getTriChildOverlap(plane1, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
			/*int pos = getPlanePosition(plane1, v0, v1, v2);
			if(pos == 0)
			{
				pos = getTriChildOverlap(plane1, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
			}*/
#endif
			//getAABB(v0, v1, v2, tbox);

			if(pos <= 0)
			{
				//bboxLeft.m_mn = fminf(bboxLeft.m_mn, tbox.m_mn/*-c_env.epsilon*/);
				//bboxLeft.m_mx = fmaxf(bboxLeft.m_mx, tbox.m_mx/*+c_env.epsilon*/);

				cntLeft1++;
			}
			if(pos >= 0)
			{
				//bboxRight.m_mn = fminf(bboxRight.m_mn, tbox.m_mn/*-c_env.epsilon*/);
				//bboxRight.m_mx = fmaxf(bboxRight.m_mx, tbox.m_mx/*+c_env.epsilon*/);

				cntRight1++;
			}

#if PLANE_COUNT > 32
#if (TRIANGLE_CLIPPING != 5 && TRIANGLE_CLIPPING != 6)
			pos = getPlanePosition(plane2, v0, v1, v2);
#if TRIANGLE_CLIPPING == 2 || TRIANGLE_CLIPPING == 3
			if(pos == 0)
			{
				pos = getPlanePositionClipped(plane2, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
			}
#endif
#else
			pos = getTriChildOverlap(plane2, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
			/*pos = getPlanePosition(plane2, v0, v1, v2);
			if(pos == 0)
			{
				pos = getTriChildOverlap(plane2, v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
			}*/
#endif

			if(pos <= 0)
				cntLeft2++;
			if(pos >= 0)
				cntRight2++;
#endif
		}

		// OPTIMIZE: Can possibly be further accelerated by using less splitting planes and parallelizing the for cycle

		// Update the bins
		//SplitRed *split = &(g_redSplits[taskIdx].splits[tid]);
		SplitDataTri *split = &(g_splitStack[taskIdx].splits[tid]);

		if(cntLeft1 != 0)
		{
#ifdef BBOX_TEST
			if(tid == 0)
			{
				if(bboxLeft.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x)
					printf("Min x binning bound error %f should be %f!\n", bboxLeft.m_mn.x, s_task[threadIdx.y].bbox.m_mn.x);
				if(bboxLeft.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y)
					printf("Min y binning bound error %f should be %f!\n", bboxLeft.m_mn.y, s_task[threadIdx.y].bbox.m_mn.y);
				if(bboxLeft.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z)
					printf("Min z binning bound error %f should be %f!\n", bboxLeft.m_mn.z, s_task[threadIdx.y].bbox.m_mn.z);

				if(bboxLeft.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x)
					printf("Max x binning bound error %f should be %f!\n", bboxLeft.m_mx.x, s_task[threadIdx.y].bbox.m_mx.x);
				if(bboxLeft.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y)
					printf("Max y binning bound error %f should be %f!\n", bboxLeft.m_mx.y, s_task[threadIdx.y].bbox.m_mx.y);
				if(bboxLeft.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
					printf("Max z binning bound error %f should be %f!\n", bboxLeft.m_mx.z, s_task[threadIdx.y].bbox.m_mx.z);
			}
#endif

			if(s_task[threadIdx.y].lock != LockType_None) // Multiple threads cooperate on this task
			{
				/*atomicMinFloat(&split->children[0].bbox.m_mn.x, bboxLeft.m_mn.x);
				atomicMinFloat(&split->children[0].bbox.m_mn.y, bboxLeft.m_mn.y);
				atomicMinFloat(&split->children[0].bbox.m_mn.z, bboxLeft.m_mn.z);

				atomicMaxFloat(&split->children[0].bbox.m_mx.x, bboxLeft.m_mx.x);
				atomicMaxFloat(&split->children[0].bbox.m_mx.y, bboxLeft.m_mx.y);
				atomicMaxFloat(&split->children[0].bbox.m_mx.z, bboxLeft.m_mx.z);*/

				//atomicAdd(&split->children[0].cnt, cntLeft);
				atomicAdd(&split->tf, cntLeft1);
			}
			else
			{
				/*split->children[0].bbox.m_mn.x = floatToOrderedInt(bboxLeft.m_mn.x);
				split->children[0].bbox.m_mn.y = floatToOrderedInt(bboxLeft.m_mn.y);
				split->children[0].bbox.m_mn.z = floatToOrderedInt(bboxLeft.m_mn.z);

				split->children[0].bbox.m_mx.x = floatToOrderedInt(bboxLeft.m_mx.x);
				split->children[0].bbox.m_mx.y = floatToOrderedInt(bboxLeft.m_mx.y);
				split->children[0].bbox.m_mx.z = floatToOrderedInt(bboxLeft.m_mx.z);*/

				//split->children[0].cnt = cntLeft;
				split->tf = cntLeft1;
			}
		}

		if(cntRight1 != 0)
		{
#ifdef BBOX_TEST
			if(tid == 0)
			{
				if(bboxRight.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x)
					printf("Min x binning bound error %f should be %f!\n", bboxRight.m_mn.x, s_task[threadIdx.y].bbox.m_mn.x);
				if(bboxRight.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y)
					printf("Min y binning bound error %f should be %f!\n", bboxRight.m_mn.y, s_task[threadIdx.y].bbox.m_mn.y);
				if(bboxRight.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z)
					printf("Min z binning bound error %f should be %f!\n", bboxRight.m_mn.z, s_task[threadIdx.y].bbox.m_mn.z);

				if(bboxRight.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x)
					printf("Max x binning bound error %f should be %f!\n", bboxRight.m_mx.x, s_task[threadIdx.y].bbox.m_mx.x);
				if(bboxRight.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y)
					printf("Max y binning bound error %f should be %f!\n", bboxRight.m_mx.y, s_task[threadIdx.y].bbox.m_mx.y);
				if(bboxRight.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
					printf("Max z binning bound error %f should be %f!\n", bboxRight.m_mx.z, s_task[threadIdx.y].bbox.m_mx.z);
			}
#endif

			if(s_task[threadIdx.y].lock != LockType_None) // Multiple threads cooperate on this task
			{
				/*atomicMinFloat(&split->children[1].bbox.m_mn.x, bboxRight.m_mn.x);
				atomicMinFloat(&split->children[1].bbox.m_mn.y, bboxRight.m_mn.y);
				atomicMinFloat(&split->children[1].bbox.m_mn.z, bboxRight.m_mn.z);

				atomicMaxFloat(&split->children[1].bbox.m_mx.x, bboxRight.m_mx.x);
				atomicMaxFloat(&split->children[1].bbox.m_mx.y, bboxRight.m_mx.y);
				atomicMaxFloat(&split->children[1].bbox.m_mx.z, bboxRight.m_mx.z);*/

				//atomicAdd(&split->children[1].cnt, cntRight);
				atomicAdd(&split->tb, cntRight1);
			}
			else
			{
				/*split->children[1].bbox.m_mn.x = floatToOrderedInt(bboxRight.m_mn.x);
				split->children[1].bbox.m_mn.y = floatToOrderedInt(bboxRight.m_mn.y);
				split->children[1].bbox.m_mn.z = floatToOrderedInt(bboxRight.m_mn.z);

				split->children[1].bbox.m_mx.x = floatToOrderedInt(bboxRight.m_mx.x);
				split->children[1].bbox.m_mx.y = floatToOrderedInt(bboxRight.m_mx.y);
				split->children[1].bbox.m_mx.z = floatToOrderedInt(bboxRight.m_mx.z);*/

				//split->children[1].cnt = cntRight;
				split->tb = cntRight1;
			}
		}

#if PLANE_COUNT > 32
		split = &(g_splitStack[taskIdx].splits[tid+WARP_SIZE]);

		if(cntLeft2 != 0)
		{
			if(s_task[threadIdx.y].lock != LockType_None) // Multiple threads cooperate on this task
			{
				atomicAdd(&split->tf, cntLeft2);
			}
			else
			{
				split->tf = cntLeft2;
			}
		}

		if(cntRight2 != 0)
		{
			if(s_task[threadIdx.y].lock != LockType_None) // Multiple threads cooperate on this task
			{
				atomicAdd(&split->tb, cntRight2);
			}
			else
			{
				split->tb = cntRight2;
			}
		}
#endif
	}
}
#endif // BINNING_TYPE == 2

#endif // SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6

//------------------------------------------------------------------------

__device__ int classifyTri(int tid, int subtask, int start, int end, volatile const float4& splitPlane)
{
	ASSERT_DIVERGENCE("classifyTri", tid);

	int tripos = start + subtask*WARP_SIZE + tid;
	if(tripos < end)
	{
		int triidx = getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, end-start)[tripos]*3;

		// Fetch triangle
		float3 v0, v1, v2;
		taskFetchTri(c_bvh_in.tris, triidx, v0, v1, v2);

		// Test that all triangles are inside the bounding box
#ifdef BBOX_TEST
		if(v0.x < s_task[threadIdx.y].bbox.m_mn.x || v0.y < s_task[threadIdx.y].bbox.m_mn.y || v0.z < s_task[threadIdx.y].bbox.m_mn.z
			|| v0.x > s_task[threadIdx.y].bbox.m_mx.x || v0.y > s_task[threadIdx.y].bbox.m_mx.y || v0.z > s_task[threadIdx.y].bbox.m_mx.z)
		{
			printf("Tri %d v0 outside bbox!\n", triidx);
		}
		if(v1.x < s_task[threadIdx.y].bbox.m_mn.x || v1.y < s_task[threadIdx.y].bbox.m_mn.y || v1.z < s_task[threadIdx.y].bbox.m_mn.z
			|| v1.x > s_task[threadIdx.y].bbox.m_mx.x || v1.y > s_task[threadIdx.y].bbox.m_mx.y || v1.z > s_task[threadIdx.y].bbox.m_mx.z)
		{
			printf("Tri %d v1 outside bbox!\n", triidx);
		}
		if(v2.x < s_task[threadIdx.y].bbox.m_mn.x || v2.y < s_task[threadIdx.y].bbox.m_mn.y || v2.z < s_task[threadIdx.y].bbox.m_mn.z
			|| v2.x > s_task[threadIdx.y].bbox.m_mx.x || v2.y > s_task[threadIdx.y].bbox.m_mx.y || v2.z > s_task[threadIdx.y].bbox.m_mx.z)
		{
			printf("Tri %d v2 outside bbox!\n", triidx);
		}
#endif

#if (TRIANGLE_CLIPPING != 4 && TRIANGLE_CLIPPING != 6)
		int pos = getPlanePosition(*((float4*)&splitPlane), v0, v1, v2);
#if TRIANGLE_CLIPPING == 1 || TRIANGLE_CLIPPING == 3
		if(pos == 0)
		{
			pos = getPlanePositionClipped(*((float4*)&splitPlane), v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
		}
#endif
#else
		int pos = getTriChildOverlap(*((float4*)&splitPlane), v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
		/*int pos = getPlanePosition(*((float4*)&splitPlane), v0, v1, v2);
		if(pos == 0)
		{
			pos = getPlanePositionClipped(*((float4*)&splitPlane), v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
		}*/
#endif

		/*float4 plane; // Without typecast, possible data copy
		plane.x = splitPlane.x;
		plane.y = splitPlane.y;
		plane.z = splitPlane.z;
		plane.w = splitPlane.w;
		int pos = getPlanePosition(plane, v0, v1, v2);*/

#if SCAN_TYPE == 3
		// Write to auxiliary array
		getPPSTrisIdxPtr(s_task[threadIdx.y].dynamicMemory, end-start)[tripos] = pos;
#endif

		return pos;
	}
	return 2;
}

//------------------------------------------------------------------------

__device__ __noinline__ void computeSplit()
{
	int subtasksDone;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int step = s_task[threadIdx.y].step;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do {
		// Run the split task
		splitCost(threadIdx.x, popSubtask, triStart, triEnd,
			s_task[threadIdx.y].bbox, s_task[threadIdx.y].splitPlane, s_task[threadIdx.y].bestCost, s_task[threadIdx.y].bestOrder);
#if SPLIT_TYPE == 1
		if(s_task[threadIdx.y].lock != LockType_None)
			taskUpdateBestPlane(threadIdx.x, popTaskIdx); // OPTIMIZE: Do not copy through shared memory
#endif

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);
	taskFinishSplit(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

#if SPLIT_TYPE == 3
__device__ __noinline__ void computeSplitParallel()
{
	int subtasksDone;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int step = s_task[threadIdx.y].step;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do {
		splitCostParallel(threadIdx.x, popSubtask, popTaskIdx, triStart, triEnd, s_task[threadIdx.y].bbox);
		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);
	taskFinishSplitParallel(threadIdx.x, popTaskIdx, subtasksDone);
}
#endif

//------------------------------------------------------------------------
#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
__device__ __noinline__ void computeBins()
{
	int subtasksDone;
	int axis = s_task[threadIdx.y].axis;
	//int axis = taskAxis(s_task[threadIdx.y].splitPlane, s_task[threadIdx.y].bbox, s_sharedData[threadIdx.y][0], s_task[threadIdx.y].axis);
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

#if BINNING_TYPE < 2
	do
	{
#if BINNING_TYPE == 0
		binTriangles(threadIdx.x, popSubtask, popTaskIdx, triStart, triEnd, s_task[threadIdx.y].bbox, axis);
#elif BINNING_TYPE == 1
		binTrianglesParallel(threadIdx.x, popSubtask, popTaskIdx, triStart, triEnd, s_task[threadIdx.y].bbox, axis);
#endif
		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);
#elif BINNING_TYPE == 2
		subtasksDone = min(popStart+1, popCount);
		binTrianglesAtomic(threadIdx.x, popStart-(subtasksDone-1), popStart+1, popTaskIdx, triStart, triEnd, s_task[threadIdx.y].bbox, axis);
#endif

	//#if BINNING_TYPE == 0 || BINNING_TYPE == 1
	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	//#endif
	taskFinishBinning(threadIdx.x, popTaskIdx, subtasksDone);
}
#endif

//------------------------------------------------------------------------

__device__ __noinline__ void computePPS()
{
	int subtasksDone;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int step = s_task[threadIdx.y].step;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		if(step == 0) // Classify all triangles against the best plane
			classifyTri(threadIdx.x, popSubtask, triStart, triEnd, s_task[threadIdx.y].splitPlane);

		pps<int>(threadIdx.x, popSubtask, step, getPPSTrisIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart), getPPSTrisPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart), s_sharedData[threadIdx.y], triStart, triEnd, 1);
		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortPPS1(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computePPSUp()
{
	int subtasksDone;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int step = s_task[threadIdx.y].step;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		if(step == 0) // Classify all triangles against the best plane
			classifyTri(threadIdx.x, popSubtask, triStart, triEnd, s_task[threadIdx.y].splitPlane);

		scanUp<int>(threadIdx.x, popSubtask, step, getPPSTrisPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart), getPPSTrisIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart), s_sharedData[threadIdx.y], triStart, triEnd, 1, plus, 0);
		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortPPSUp(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computePPSDown()
{
	int subtasksDone;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int step = s_task[threadIdx.y].step;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		scanDown<int>(threadIdx.x, popSubtask, step, getPPSTrisPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart), getPPSTrisIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart), s_sharedData[threadIdx.y], triStart, triEnd, 1, plus, 0);
		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortPPSDown(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computeSort()
{
	int subtasksDone;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int triRight = s_task[threadIdx.y].triRight;
	int step = s_task[threadIdx.y].step;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		sort(threadIdx.x, popSubtask, step, getPPSTrisIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart), getPPSTrisPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart), getSortTrisPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart), (int*)c_bvh_in.trisIndex, triStart, triEnd, triRight, 1, false);
		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortSORT1(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computePartition()
{
	int subtasksDone;
	int tid = threadIdx.x;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;
	bool singleWarp = s_task[threadIdx.y].lock == LockType_None;

	// Set the swap arrays
	int* inIdx = getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart);
	int* outIdxLeft = getTriIdxPtr(s_task[threadIdx.y].dynamicMemoryLeft, 0);
	int* outIdxRight = getTriIdxPtr(s_task[threadIdx.y].dynamicMemoryRight, 0);

	s_owner[threadIdx.y][0] = 0;
	s_owner[threadIdx.y][1] = 0;

#if SCAN_TYPE == 2
	do
	{
		// Classify the triangles
		int pos = -2; // Outside of the interval
		int triPos = triStart + popSubtask*WARP_SIZE + tid;
		int triSum = min(triEnd - (triStart + popSubtask*WARP_SIZE), WARP_SIZE); // Triangles to process
		int triIdx = -1;
		if(triPos < triEnd)
		{
			triIdx = inIdx[triPos];

			// Fetch triangle
			float3 v0, v1, v2;
			taskFetchTri(c_bvh_in.tris, triIdx*3, v0, v1, v2);

			// Test that all triangles are inside the bounding box
#ifdef BBOX_TEST
			if(v0.x < s_task[threadIdx.y].bbox.m_mn.x || v0.y < s_task[threadIdx.y].bbox.m_mn.y || v0.z < s_task[threadIdx.y].bbox.m_mn.z
				|| v0.x > s_task[threadIdx.y].bbox.m_mx.x || v0.y > s_task[threadIdx.y].bbox.m_mx.y || v0.z > s_task[threadIdx.y].bbox.m_mx.z)
			{
				printf("Tri %d v0 outside bbox!\n", triIdx);
			}
			if(v1.x < s_task[threadIdx.y].bbox.m_mn.x || v1.y < s_task[threadIdx.y].bbox.m_mn.y || v1.z < s_task[threadIdx.y].bbox.m_mn.z
				|| v1.x > s_task[threadIdx.y].bbox.m_mx.x || v1.y > s_task[threadIdx.y].bbox.m_mx.y || v1.z > s_task[threadIdx.y].bbox.m_mx.z)
			{
				printf("Tri %d v1 outside bbox!\n", triIdx);
			}
			if(v2.x < s_task[threadIdx.y].bbox.m_mn.x || v2.y < s_task[threadIdx.y].bbox.m_mn.y || v2.z < s_task[threadIdx.y].bbox.m_mn.z
				|| v2.x > s_task[threadIdx.y].bbox.m_mx.x || v2.y > s_task[threadIdx.y].bbox.m_mx.y || v2.z > s_task[threadIdx.y].bbox.m_mx.z)
			{
				printf("Tri %d v2 outside bbox!\n", triIdx);
			}
#endif

#if (TRIANGLE_CLIPPING != 4 && TRIANGLE_CLIPPING != 6)
			pos = getPlanePosition(*((float4*)&s_task[threadIdx.y].splitPlane), v0, v1, v2);
#if TRIANGLE_CLIPPING == 1 || TRIANGLE_CLIPPING == 3
			if(pos == 0)
			{
				pos = getPlanePositionClipped(*((float4*)&s_task[threadIdx.y].splitPlane), v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
			}
#endif
#else
			/*//int posRef = getPlanePosition(*((float4*)&s_task[threadIdx.y].splitPlane), v0, v1, v2);*/
			pos = getTriChildOverlap(*((float4*)&s_task[threadIdx.y].splitPlane), v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));

			/*if(pos == 0 && posRef != 0)
			{
				posRef = getPlanePosition(*((float4*)&s_task[threadIdx.y].splitPlane), v0, v1, v2);
				pos = getTriChildOverlap(*((float4*)&s_task[threadIdx.y].splitPlane), v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
				//pos = posRef;
				printf("Ref %d exact %d\n", posRef, pos);
			}*/
			/*pos = getPlanePosition(*((float4*)&s_task[threadIdx.y].splitPlane), v0, v1, v2);
			if(pos == 0)
			{
				pos = getTriChildOverlap(*((float4*)&s_task[threadIdx.y].splitPlane), v0, v1, v2, *((CudaAABB*)&s_task[threadIdx.y].bbox));
			}*/
#endif
		}

		// Partition the triangles to the left and right children intervals

		// Scan the number of triangles to the left of the splitting plane
		int triCnt;
		int exclusiveScan = threadPosWarp(tid, s_sharedData[threadIdx.y], pos > -2 && pos <= 0, triCnt);
		
		if(!singleWarp && tid == 0 && triCnt > 0)
			s_owner[threadIdx.y][0] = atomicAdd(&g_taskStackBVH.tasks[popTaskIdx].triLeft, triCnt); // Add the number of triangles to the left of the plane to the global counter

		//if(pos > -2 && pos <= 0)
		//	printf("Left %d\n", s_owner[threadIdx.y][0] + exclusiveScan);

		// Find the output position for each thread as the sum of the output position and the exclusive scanned value
		if(pos > -2 && pos <= 0)
			outIdxLeft[s_owner[threadIdx.y][0] + exclusiveScan] = triIdx;
		s_owner[threadIdx.y][0] += triCnt; // Move the position by the number of written nodes

		// Compute the number of triangles to the right of the splitting plane
		int inverseExclusiveScan = threadPosWarp(tid, s_sharedData[threadIdx.y], pos >= 0, triCnt);

		if(!singleWarp && tid == 0 && triCnt > 0)
			s_owner[threadIdx.y][1] = atomicAdd(&g_taskStackBVH.tasks[popTaskIdx].triRight, triCnt); // Add the number of triangles to the right of the plane to the global counter

		//if(pos >= 0)
		//	printf("Right %d\n", s_owner[threadIdx.y][1] + inverseExclusiveScan);

		// Find the output position for each thread as the output position minus the triangle count plus the scanned value
		if(pos >= 0)
			outIdxRight[s_owner[threadIdx.y][1] + inverseExclusiveScan] = triIdx;
		s_owner[threadIdx.y][1] += triCnt; // Move the position by the number of written nodes


		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	// Write out the final positions
	if(singleWarp)
	{
		g_taskStackBVH.tasks[popTaskIdx].triLeft = s_owner[threadIdx.y][0];
		g_taskStackBVH.tasks[popTaskIdx].triRight = s_owner[threadIdx.y][1];
	}

#elif SCAN_TYPE == 3
#ifdef MALLOC_SCRATCHPAD
#error Code not updated!
#endif
	int cntLeft = 0;
	int cntRight = 0;
	int* inPos = getPPSTrisIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart);

	do
	{
		// Classify the triangles
		int pos = classifyTri(threadIdx.x, popSubtask, triStart, triEnd, s_task[threadIdx.y].splitPlane);

		// Update the counters
		if(pos <= 0)
			cntLeft++;
		if(pos >= 0)
			cntRight++;

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	// Reduce the counters
	s_sharedData[threadIdx.y][tid] = cntLeft;
	reduceWarp(tid, s_sharedData[threadIdx.y], plus);
	int triCnt = s_sharedData[threadIdx.y][WARP_SIZE-1];

	if(tid == 0 && triCnt > 0)
		s_owner[threadIdx.y][0] = atomicAdd(&g_taskStackBVH.tasks[popTaskIdx].triLeft, triCnt); // Add the number of triangles to the left of the plane to the global counter
	int warpAtomicLeft = s_owner[threadIdx.y][0];

	s_sharedData[threadIdx.y][tid] = cntRight;
	reduceWarp(tid, s_sharedData[threadIdx.y], plus);
	triCnt = s_sharedData[threadIdx.y][WARP_SIZE-1];

	if(tid == 0 && triCnt > 0)
		s_owner[threadIdx.y][0] = atomicSub(&g_taskStackBVH.tasks[popTaskIdx].triRight, triCnt); // Add the number of triangles to the right of the plane to the global counter
	int warpAtomicRight = s_owner[threadIdx.y][0];

	popSubtask = popStart; // Sweep the interval once more
	do
	{

		int pos = -2; // Outside of the interval
		int triPos = triStart + popSubtask*WARP_SIZE + tid;
		int triSum = min(triEnd - (triStart + popSubtask*WARP_SIZE), WARP_SIZE); // Triangles to process
		int triIdx = -1;
		if(triPos < triEnd)
		{
			// Load the triangle index and position from the global memory
			triIdx = inIdx[triPos];
			pos = inPos[triPos];
		}

		// Partition the triangles to the left and right children intervals

		// Scan the number of triangles to the left of the splitting plane
		s_sharedData[threadIdx.y][tid] = 0;
		if(pos > -2 && pos <= 0)
			s_sharedData[threadIdx.y][tid] = 1;

		scanWarp<int>(tid, s_sharedData[threadIdx.y], plus);
		int exclusiveScan = (s_sharedData[threadIdx.y][tid] - 1);
		int triCnt = s_sharedData[threadIdx.y][WARP_SIZE-1];

		// Find the output position for each thread as the sum of the output position and the exclusive scanned value
		if(pos > -2 && pos <= 0)
			outIdx[warpAtomicLeft + exclusiveScan] = triIdx;
		warpAtomicLeft += triCnt; // Move the position by the number of written nodes

		// Scan the number of triangles to the right of the splitting plane
		s_sharedData[threadIdx.y][tid] = 0;
		if(pos >= 0)
			s_sharedData[threadIdx.y][tid] = 1;

		scanWarp<int>(tid, s_sharedData[threadIdx.y], plus);
		int inverseExclusiveScan = (s_sharedData[threadIdx.y][tid] - 1);
		triCnt = s_sharedData[threadIdx.y][WARP_SIZE-1];

		// Find the output position for each thread as the output position minus the triangle count plus the scanned value
		if(pos >= 0)
			outIdx[warpAtomicRight - triCnt + inverseExclusiveScan] = triIdx;
		warpAtomicRight -= triCnt; // Move the position by the number of written nodes

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);
#endif

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishPartition(tid, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1

// Local + global reduction on bins
// OPTIMIZE: It could be more efficient to pass op as template parameter instead of function pointer?
__device__ void reduceBins(int tid, int subtask, int taskIdx, int step)
{
	// Get split
	SplitArray *splitArray = &(g_redSplits[taskIdx]);

	if(step == 0) // Do local reduction, step 0
	{
		int warpBlock = subtask / (WARP_SIZE*2); // Different warps reduce different SplitRed in the array
		int warpIdx = warpBlock * WARP_SIZE + tid;
		int split = subtask % WARP_SIZE; // Different warps reduce different splits in the array
		int child = (subtask/WARP_SIZE) % 2; // Different warps reduce different ChildData in the array
		ASSERT_DIVERGENCE("reduceBins step0 top", tid);

		ChildData* data = (ChildData*)&splitArray->splits[warpIdx][split].children[child];
		ChildData* out = (ChildData*)&splitArray->splits[warpBlock * WARP_SIZE][split].children[child];

		volatile float* red = (volatile float*)&s_sharedData[threadIdx.y][0];
		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(warpIdx < NUM_WARPS)
		{
			// Reduce min
			red[tid] = data->bbox.m_mn.x;
			reduceWarp(tid, &red[0], min);
			out->bbox.m_mn.x = red[tid]; // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

#ifdef BBOX_TEST
			if(red[tid] < s_task[threadIdx.y].bbox.m_mn.x)
			{
				printf("Min x step0 bound error task %d!\n", taskIdx);
				g_taskStackBVH.unfinished = 1;
			}
#endif

			red[tid] = data->bbox.m_mn.y;
			reduceWarp(tid, &red[0], min);
			out->bbox.m_mn.y = red[tid]; // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

#ifdef BBOX_TEST
			if(red[tid] < s_task[threadIdx.y].bbox.m_mn.y)
			{
				printf("Min y step0 bound error task %d!\n", taskIdx);
				g_taskStackBVH.unfinished = 1;
			}
#endif

			red[tid] = data->bbox.m_mn.z;
			reduceWarp(tid, &red[0], min);
			out->bbox.m_mn.z = red[tid]; // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

#ifdef BBOX_TEST
			if(red[tid] < s_task[threadIdx.y].bbox.m_mn.z)
			{
				printf("Min z step0 bound error task %d!\n", taskIdx);
				g_taskStackBVH.unfinished = 1;
			}
#endif
		}

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(warpIdx < NUM_WARPS)
		{
			// Reduce max
			red[tid] = data->bbox.m_mx.x;
			reduceWarp(tid, &red[0], max);
			out->bbox.m_mx.x = red[tid]; // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

#ifdef BBOX_TEST
			if(red[tid] > s_task[threadIdx.y].bbox.m_mx.x)
			{
				printf("Max x step0 bound error task %d!\n", taskIdx);
				g_taskStackBVH.unfinished = 1;
			}
#endif

			red[tid] = data->bbox.m_mx.y;
			reduceWarp(tid, &red[0], max);
			out->bbox.m_mx.y = red[tid]; // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

#ifdef BBOX_TEST
			if(red[tid] > s_task[threadIdx.y].bbox.m_mx.y)
			{
				printf("Max y step0 bound error task %d!\n", taskIdx);
				g_taskStackBVH.unfinished = 1;
			}
#endif

			red[tid] = data->bbox.m_mx.z;
			reduceWarp(tid, &red[0], max);
			out->bbox.m_mx.z = red[tid]; // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

#ifdef BBOX_TEST
			if(red[tid] > s_task[threadIdx.y].bbox.m_mx.z)
			{
				printf("Max z step0 bound error task %d!\n", taskIdx);
				g_taskStackBVH.unfinished = 1;
			}
#endif
		}

		volatile int* redI = (volatile int*)&s_sharedData[threadIdx.y][0];
		redI[tid] = 0; // Save identities so that we do not work with uninitialized data

		if(warpIdx < NUM_WARPS)
		{
			// Reduce cnt
			redI[tid] = data->cnt;
			reduceWarp(tid, &redI[0], plus);
			out->cnt = redI[tid]; // Copy results to gmem

#ifdef BBOX_TEST
			if(redI[tid] > s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triStart)
			{
				printf("Count step0 bound error task %d!\n", taskIdx);
				g_taskStackBVH.unfinished = 1;
			}
#endif
		}
	}
	else // Do global reduction, step 1 to n
	{
		ASSERT_DIVERGENCE("reduceBins step1-n top", tid);

		int i          = subtask / 2;
		int blockSize  = (1 << (step+LOG_WARP_SIZE));
		int halfBlock  = blockSize >> 1;
		int blockStart = blockSize * i;
		int posThis = blockStart;
		int posNext = blockStart + halfBlock;

		int split = tid; // Different warps reduce different splits in the array
		int child = subtask % 2; // Different warps reduce different ChildData in the array

		ChildData* data = (ChildData*)&splitArray->splits[posNext][split].children[child];
		ChildData* out = (ChildData*)&splitArray->splits[posThis][split].children[child];

		if(posNext < NUM_WARPS)
		{
			// Reduce min
			out->bbox.m_mn.x = fminf(out->bbox.m_mn.x, data->bbox.m_mn.x); // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

			out->bbox.m_mn.y = fminf(out->bbox.m_mn.y, data->bbox.m_mn.y); // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

			out->bbox.m_mn.z = fminf(out->bbox.m_mn.z, data->bbox.m_mn.z); // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

			// Reduce max
			out->bbox.m_mx.x = fmaxf(out->bbox.m_mx.x, data->bbox.m_mx.x); // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

			out->bbox.m_mx.y = fmaxf(out->bbox.m_mx.y, data->bbox.m_mx.y); // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

			out->bbox.m_mx.z = fmaxf(out->bbox.m_mx.z, data->bbox.m_mx.z); // Copy results to gmem
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

			// Reduce cnt
			out->cnt = out->cnt + data->cnt; // Copy results to gmem
		}

		//__threadfence(); // Optimize: Maybe not needed here
	}
}

#endif
#endif

//------------------------------------------------------------------------

#if defined(OBJECT_SAH)/* && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)*/
// Build the rest of the tree with object splits instead of spatial ones
// Expects less than WARP_SIZE triangles
__device__ __noinline__ void computeObjectSplitTree()
{
	int tid = threadIdx.x;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int end = triEnd-triStart;

	ASSERT_DIVERGENCE("computeObjectSplitTree top", tid);

	//if(tid == 0 && end <= 1)
	//	printf("Node: %d-%d\n", triStart, triEnd);

	// Load one triangle per thread
	int triIdx;
	float3 v0, v1, v2;
	if(tid < end)
	{
		int* inIdx = getTriIdxPtr(s_task[threadIdx.y].dynamicMemory, triEnd-triStart, s_task[threadIdx.y].triIdxCtr);
		triIdx = inIdx[triStart + tid];
		taskFetchTri(c_bvh_in.tris, triIdx*3, v0, v1, v2);
	}

	CudaAABB tbox;
	float3 centroid = getCentroid(v0, v1, v2, tbox);
	tbox.m_mn -= c_env.epsilon;
	tbox.m_mx += c_env.epsilon;

	// Init subtree root
	int parentIdx = s_task[threadIdx.y].parentIdx;
	int nodeIdx = s_task[threadIdx.y].nodeIdx;
	int taskID = s_task[threadIdx.y].taskID;
	int bestAxis = 0;
	int segmentStart = 0;
	int segmentEnd = end;
	int triangleThread = tid; // Index to the thread holding the triangle data
	CudaAABB bboxLeft, bboxRight;
	bool isLeaf = (end <= c_env.triLimit) ? true : false; // Stop immediately if input is just 1 triangle

#ifndef COMPACT_LAYOUT
	if(end <= c_env.triLimit) // Write the single leaf
	{
		volatile CudaBVHNode* node = (CudaBVHNode*)&s_newTask[threadIdx.y];

		*((float4*)&node->c0xy) = make_float4(s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mx.y);
		*((float4*)&node->c1xy) = make_float4(s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mx.y);
		*((float4*)&node->c01z) = make_float4(s_task[threadIdx.y].bbox.m_mn.z, s_task[threadIdx.y].bbox.m_mx.z, s_task[threadIdx.y].bbox.m_mn.z, s_task[threadIdx.y].bbox.m_mx.z);
		*((int4*)&node->children) = make_int4(triStart, triEnd, parentIdx, 0);
		taskSaveNodeToGMEM(g_kdtree, tid, nodeIdx, *node);

#if SCAN_TYPE == 2 || SCAN_TYPE == 3
		// Back-copy triangles to the correct array
		backcopy(tid, s_task[threadIdx.y].triIdxCtr, triStart, triEnd);
#endif

		if(tid == 0)
			taskUpdateParentPtr(g_kdtree, parentIdx, taskID, ~nodeIdx); // Mark this node as leaf in the hierarchy	

		return;
	}
#endif

	if(tid >= end) // Prepare out-of-range threads
	{
		parentIdx = -1;
		nodeIdx = -(WARP_SIZE+1);
		segmentStart = end;
		segmentEnd = end;
		isLeaf = true;
	}

	ASSERT_DIVERGENCE("computeObjectSplitTree init", tid);

	// For all levels
	// IMPORTANT: all threads must participate in the __shfl instruction, otherwise undefined data will be loaded
	while(__any(!isLeaf)) // While there is a segment that is not yet a leaf
	{
		float bestCost = CUDART_INF_F;

		// Sort in every axis
		for(int axis = 0; axis < 3; axis++)
		{
			// Initialize bounding boxes
			bboxLeft.m_mn.x = bboxLeft.m_mn.y = bboxLeft.m_mn.z = CUDART_INF_F;
			bboxRight.m_mn.x = bboxRight.m_mn.y = bboxRight.m_mn.z = CUDART_INF_F;
			bboxLeft.m_mx.x = bboxLeft.m_mx.y = bboxLeft.m_mx.z = -CUDART_INF_F;
			bboxRight.m_mx.x = bboxRight.m_mx.y = bboxRight.m_mx.z = -CUDART_INF_F;

			float pos = (axis == 0) ? centroid.x : ((axis == 1) ? centroid.y : centroid.z);
			pos = __shfl(pos, triangleThread); // Load the position from the thread holding the data
			sortWarpSegmented(tid, pos, triangleThread, segmentStart, end); // Sort pair (pos, triangleThread) in registers

			// Update split cost
			for(int i = segmentStart; __any(i < segmentEnd); i++) // All threads must be active for the shuffle instruction!!!
			{
				ASSERT_DIVERGENCE("computeObjectSplitTree bestSplitShuffle", tid);

				int trianglePos = __shfl(triangleThread, i); // Find thread holding data for triangle i
				CudaAABB boxI;
				// All threads broadcast triangle box from the thread holding the triangle
				// OPTIMIZE: Grow by individual points?
				boxI.m_mn.x = __shfl(tbox.m_mn.x, trianglePos); boxI.m_mn.y = __shfl(tbox.m_mn.y, trianglePos);  boxI.m_mn.z = __shfl(tbox.m_mn.z, trianglePos);
				boxI.m_mx.x = __shfl(tbox.m_mx.x, trianglePos); boxI.m_mx.y = __shfl(tbox.m_mx.y, trianglePos);  boxI.m_mx.z = __shfl(tbox.m_mx.z, trianglePos);

				// Update the counters
				// Do not let threads that already swept their interval update their boxes
				// HACK: Some other form of this preventing condition may be optimized and break the instruction order
				if(i < segmentEnd && (i < tid || tid == segmentEnd-1))
					bboxLeft.grow(boxI);
				else if(i < segmentEnd)
					bboxRight.grow(boxI);
			}

			// Compute the cost for position "tid" in axis "axis"
			float cost = areaAABB(bboxLeft)*(tid-segmentStart) + areaAABB(bboxRight)*(segmentEnd-tid);
			if(tid < end && cost < bestCost) // Do not compute cost for threads with invalid data - will break reduced cost
			{
				bestCost = cost;
				bestAxis = axis;
			}
		}

		ASSERT_DIVERGENCE("computeObjectSplitTree bestSplit", tid);

		// Compute best split for each node
		s_owner[threadIdx.y][tid] = segmentStart;
		volatile float* costs = (volatile float*)s_sharedData[threadIdx.y];
		costs[tid] = bestCost;
		segReduceWarp<float>(tid, costs, s_owner[threadIdx.y], min); // Find minimum cost

		// Broadcast the data
		if(costs[segmentStart] == bestCost) // This thread is the winner
			s_sharedData[threadIdx.y][segmentStart] = tid;

		int bestPos = s_sharedData[threadIdx.y][segmentStart];
		bestAxis = __shfl(bestAxis, bestPos);

		// Resort the array
		// OPTIMIZE: MOVE THIS CLOSER TO WRITING OF THE BOUNDING BOXES

		// Initialize bounding boxes
		bboxLeft.m_mn.x = bboxLeft.m_mn.y = bboxLeft.m_mn.z = CUDART_INF_F;
		bboxRight.m_mn.x = bboxRight.m_mn.y = bboxRight.m_mn.z = CUDART_INF_F;
		bboxLeft.m_mx.x = bboxLeft.m_mx.y = bboxLeft.m_mx.z = -CUDART_INF_F;
		bboxRight.m_mx.x = bboxRight.m_mx.y = bboxRight.m_mx.z = -CUDART_INF_F;

		// Tell each thread the axis we wish to get position from
		/*s_sharedData[threadIdx.y][triangleThread] = bestAxis;
		int desiredAxis = s_sharedData[threadIdx.y][tid];
		float pos = (desiredAxis == 0) ? centroid.x : ((desiredAxis == 1) ? centroid.y : centroid.z);
		pos = __shfl(pos, triangleThread); // Load the position from the thread holding the data*/
		float posX = __shfl(centroid.x, triangleThread); float posY = __shfl(centroid.y, triangleThread); float posZ = __shfl(centroid.z, triangleThread);
		float pos = (bestAxis == 0) ? posX : ((bestAxis == 1) ? posY : posZ);
		sortWarpSegmented(tid, pos, triangleThread, segmentStart, end); // Sort pair (pos, trianglePos) in registers

		// Find the left and right bounding boxes
		for(int i = segmentStart; __any(i < segmentEnd); i++) // All threads must be active for the shuffle instruction!!!
		{
			ASSERT_DIVERGENCE("computeObjectSplitTree bboxesShuffle", tid);
			int trianglePos = __shfl(triangleThread, i); // Find thread holding data for triangle i
			CudaAABB boxI;
			// All threads broadcast triangle box from the thread holding the triangle
			// OPTIMIZE: Grow by individual points?
			boxI.m_mn.x = __shfl(tbox.m_mn.x, trianglePos); boxI.m_mn.y = __shfl(tbox.m_mn.y, trianglePos); boxI.m_mn.z = __shfl(tbox.m_mn.z, trianglePos);
			boxI.m_mx.x = __shfl(tbox.m_mx.x, trianglePos); boxI.m_mx.y = __shfl(tbox.m_mx.y, trianglePos); boxI.m_mx.z = __shfl(tbox.m_mx.z, trianglePos);

			// Update the counters
			// Do not let threads that already swept their interval update their boxes
			// HACK: Some other form of this preventing condition may be optimized and break the instruction order
			if(i < segmentEnd && i < bestPos)
				bboxLeft.grow(boxI);
			else if(i < segmentEnd)
				bboxRight.grow(boxI);
		}

		ASSERT_DIVERGENCE("computeObjectSplitTree bboxes", tid);

		// Perform the split and write out the data

#ifdef SAH_TERMINATION
		float leftCount, rightCount;
		leftCount = bestPos-segmentStart;
		rightCount = segmentEnd-bestPos;

		if(!isLeaf && leftCount+rightCount <= c_env.triMaxLimit)
		{
			float leafCost, leftCost, rightCost;
			CudaAABB bbox = bboxLeft;
			bbox.grow(bboxRight);
			// Evaluate if the termination criteria are met
			leafCost = c_env.optCi * (leftCount+rightCount);
			leftCost = areaAABB(bboxLeft)/areaAABB(bbox)*leftCount;
			rightCost = areaAABB(bboxRight)/areaAABB(bbox)*rightCount;
			float subdivisionCost = c_env.optCt + c_env.optCi*(leftCost + rightCost);

			if(leafCost < subdivisionCost)
			{
#ifndef COMPACT_LAYOUT
				if(tid == segmentStart)
				{
					g_kdtree[nodeIdx].c0xy = make_float4(bbox.m_mn.x, bbox.m_mx.x, bbox.m_mn.y, bbox.m_mx.y);
					g_kdtree[nodeIdx].c1xy = make_float4(bbox.m_mn.x, bbox.m_mx.x, bbox.m_mn.y, bbox.m_mx.y);
					g_kdtree[nodeIdx].c01z = make_float4(bbox.m_mn.z, bbox.m_mx.z, bbox.m_mn.z, bbox.m_mx.z);
					g_kdtree[nodeIdx].children = make_int4(triStart+segmentStart, triStart+segmentEnd, parentIdx, 0); // Sets the leaf child pointers

					taskUpdateParentPtr(g_kdtree, parentIdx, taskID, ~nodeIdx); // Mark this node as finished in the hierarchy
				}
#endif

				isLeaf = true;
				nodeIdx = -(segmentStart+1);
#ifdef BVH_COUNT_NODES
				// Beware the node count must still be updated
				if(tid == segmentStart)
					atomicAdd(&g_taskStackBVH.numNodes, 1);
#endif
			}
		}
#endif

		// Scan the array to compute the number (and position) of new nodes to write out
		int childrenCount = 0;
#ifndef COMPACT_LAYOUT
		if(!isLeaf && tid == segmentStart) // Not a leaf already
			childrenCount = 2;
#else
		if(!isLeaf && tid == segmentStart)
			childrenCount = ((bestPos-segmentStart <= c_env.triLimit) ? 0 : 1) + ((segmentEnd-bestPos <= c_env.triLimit) ? 0 : 1); // Count the number of segments in the next iteration
#endif

		int nodePosition = childrenCount;
		scanWarp<int>(tid, nodePosition, plus);
		int nodeCount = __shfl(nodePosition, end-1); // Read out the scaned sum from the last thread

		int nodeOffset = 0;
		if(tid == 0 && nodeCount != 0)
		{
			// Inner node -> create new subtasks in the final array
			nodeOffset = atomicAdd(&g_taskStackBVH.nodeTop, nodeCount);
		}
		// Compute the positions of new children
		childrenCount = __shfl(childrenCount, segmentStart); // Broadcast children count from the segment start
		nodePosition -= childrenCount; // Make exclusive scan from inclusive scan
		nodePosition = __shfl(nodeOffset, 0) + nodePosition; // Compute global position

		ASSERT_DIVERGENCE("computeObjectSplitTree nodeOffset", tid);

		// Add the child nodes if this segment is not already a leaf
		if(!isLeaf)
		{

#ifndef COMPACT_LAYOUT
			int childLeft = nodePosition+0;
			int childRight = nodePosition+1;
#else
			int childLeft = nodePosition+0;
			int childRight = nodePosition+((childrenCount < 2) ? 0 : 1);
#endif

#ifndef COMPACT_LAYOUT
			// Check if children should be leaves
			if(tid == segmentStart && bestPos-segmentStart <= c_env.triLimit) // Only the first thread of the segment may write the leafs because it may be the only thread in the segment
			{
				// Leaf -> same bounding boxes
				g_kdtree[childLeft].c0xy = make_float4(bboxLeft.m_mn.x, bboxLeft.m_mx.x, bboxLeft.m_mn.y, bboxLeft.m_mx.y);
				g_kdtree[childLeft].c1xy = make_float4(bboxLeft.m_mn.x, bboxLeft.m_mx.x, bboxLeft.m_mn.y, bboxLeft.m_mx.y);
				g_kdtree[childLeft].c01z = make_float4(bboxLeft.m_mn.z, bboxLeft.m_mx.z, bboxLeft.m_mn.z, bboxLeft.m_mx.z);
				g_kdtree[childLeft].children = make_int4(triStart+segmentStart, triStart+bestPos, parentIdx, 0);

				childLeft = ~childLeft;
			}

			if(tid == segmentStart && segmentEnd-bestPos <= c_env.triLimit) // Only the first thread of the segment may write the leafs because it may be the only thread in the segment
			{
				// Leaf -> same bounding boxes
				g_kdtree[childRight].c0xy = make_float4(bboxRight.m_mn.x, bboxRight.m_mx.x, bboxRight.m_mn.y, bboxRight.m_mx.y);
				g_kdtree[childRight].c1xy = make_float4(bboxRight.m_mn.x, bboxRight.m_mx.x, bboxRight.m_mn.y, bboxRight.m_mx.y);
				g_kdtree[childRight].c01z = make_float4(bboxRight.m_mn.z, bboxRight.m_mx.z, bboxRight.m_mn.z, bboxRight.m_mx.z);
				g_kdtree[childRight].children = make_int4(triStart+bestPos, triStart+segmentEnd, parentIdx, 0);

				childRight = ~childRight;
			}
#endif

			if(tid == segmentStart)
			{
#ifdef BBOX_TEST
				//printf("Segment %d - %d (%d)\n", segmentStart, segmentEnd, end);
				if(bboxLeft.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x)
					printf("Min left x objectSAH bound error %f should be %f!\n", bboxLeft.m_mn.x, s_task[threadIdx.y].bbox.m_mn.x);
				if(bboxLeft.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y)
					printf("Min left y objectSAH bound error %f should be %f!\n", bboxLeft.m_mn.y, s_task[threadIdx.y].bbox.m_mn.y);
				if(bboxLeft.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z)
					printf("Min left z objectSAH bound error %f should be %f!\n", bboxLeft.m_mn.z, s_task[threadIdx.y].bbox.m_mn.z);

				if(bboxLeft.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x)
					printf("Max left x objectSAH bound error %f should be %f!\n", bboxLeft.m_mx.x, s_task[threadIdx.y].bbox.m_mx.x);
				if(bboxLeft.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y)
					printf("Max left y objectSAH bound error %f should be %f!\n", bboxLeft.m_mx.y, s_task[threadIdx.y].bbox.m_mx.y);
				if(bboxLeft.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
					printf("Max left z objectSAH bound error %f should be %f!\n", bboxLeft.m_mx.z, s_task[threadIdx.y].bbox.m_mx.z);

				if(bboxRight.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x)
					printf("Min right x objectSAH bound error %f should be %f!\n", bboxRight.m_mn.x, s_task[threadIdx.y].bbox.m_mn.x);
				if(bboxRight.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y)
					printf("Min right y objectSAH bound error %f should be %f!\n", bboxRight.m_mn.y, s_task[threadIdx.y].bbox.m_mn.y);
				if(bboxRight.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z)
					printf("Min right z objectSAH bound error %f should be %f!\n", bboxRight.m_mn.z, s_task[threadIdx.y].bbox.m_mn.z);

				if(bboxRight.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x)
					printf("Max right x objectSAH bound error %f should be %f!\n", bboxRight.m_mx.x, s_task[threadIdx.y].bbox.m_mx.x);
				if(bboxRight.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y)
					printf("Max right y objectSAH bound error %f should be %f!\n", bboxRight.m_mx.y, s_task[threadIdx.y].bbox.m_mx.y);
				if(bboxRight.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
					printf("Max right z objectSAH bound error %f should be %f!\n", bboxRight.m_mx.z, s_task[threadIdx.y].bbox.m_mx.z);
#endif

				//printf("Parent %d, node %d -> left %d right %d!\n", parentIdx, nodeIdx, childLeft, childRight);
				/*printf("Split %d: left %d ( %.2f , %.2f , %.2f ) - ( %.2f , %.2f , %.2f ) X right %d ( %.2f , %.2f , %.2f ) - ( %.2f , %.2f , %.2f )\n\n",
					segmentEnd-segmentStart, bestPos-segmentStart,
					bboxLeft.m_mn.x, bboxLeft.m_mn.y, bboxLeft.m_mn.z, bboxLeft.m_mx.x, bboxLeft.m_mx.y, bboxLeft.m_mx.z,
					segmentEnd-bestPos,
					bboxRight.m_mn.x, bboxRight.m_mn.y, bboxRight.m_mn.z, bboxRight.m_mx.x, bboxRight.m_mx.y, bboxRight.m_mx.z);*/

				g_kdtree[nodeIdx].c0xy = make_float4(bboxLeft.m_mn.x, bboxLeft.m_mx.x, bboxLeft.m_mn.y, bboxLeft.m_mx.y);
				g_kdtree[nodeIdx].c1xy = make_float4(bboxRight.m_mn.x, bboxRight.m_mx.x, bboxRight.m_mn.y, bboxRight.m_mx.y);
				g_kdtree[nodeIdx].c01z = make_float4(bboxLeft.m_mn.z, bboxLeft.m_mx.z, bboxRight.m_mn.z, bboxRight.m_mx.z);
#ifndef COMPACT_LAYOUT
				g_kdtree[nodeIdx].children = make_int4(childLeft, childRight, parentIdx, 0); // Sets the leaf child pointers
#else
				g_kdtree[nodeIdx].children = make_int4(childLeft*64, childRight*64, parentIdx, 0); // Sets the leaf child pointers
#endif

				//if(nodeIdx >= 0 && nodeIdx <= 50)
				//	printf("SN %d, l %d r %d\n", nodeIdx, childLeft*64, childRight*64);
			}

			// Prepare for next iteration

			parentIdx = nodeIdx;
			if(tid < bestPos)
			{
				nodeIdx = childLeft;
				segmentEnd = bestPos;
				taskID = 0;
			}
			else
			{
				nodeIdx = childRight;
				segmentStart = bestPos;
				taskID = 1;
			}

			isLeaf = (segmentEnd-segmentStart <= c_env.triLimit);
			if(isLeaf)
				nodeIdx = -(segmentStart+1); // Prevent two segments sharing the same nodeIdx
		}
	}

	ASSERT_DIVERGENCE("computeObjectSplitTree triangleOffset", tid);


	// Write out the triangles
#ifndef COMPACT_LAYOUT
	// Everything is already set, just write out the sorted triangle indices
	if(tid < end)
	{
		int sortedTriIdx = __shfl(triIdx, triangleThread); // Load the index from the thread holding the data
		int* outIdx = (int*)c_bvh_in.trisIndex;
		outIdx[triStart + tid] = sortedTriIdx;
	}
#else
	// Scan the array to compute the number (and position) of triangles to write out
	int triCount = 0;
	if(tid < end)
		triCount = 3 + ((tid == segmentEnd-1) ? 1 : 0); // Count the number of vertices + the number of segments

	int triPosition = triCount;
	scanWarp<int>(tid, triPosition, plus);
	int bufferSize = __shfl(triPosition, end-1); // Read out the scaned sum from the last thread
	triPosition -= triCount; // Make exclusive scan from inclusive scan

	int triOffset;
	if(tid == 0)
	{
		// Leaf -> create new triangles in the final array
		triOffset = atomicAdd(&g_taskStackBVH.triTop, bufferSize);

#ifdef BVH_COUNT_NODES
		int cntSegments = bufferSize-3*end; // Subtract size of triangle data to get the number of segments (leaves)
		atomicAdd(&g_taskStackBVH.numNodes, cntSegments-1);
		atomicAdd(&g_taskStackBVH.numSortedTris, __log2f(cntSegments)*(s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triStart));
		atomicAdd(&g_taskStackBVH.numLeaves, cntSegments);
#endif
	}
	triPosition = __shfl(triOffset, 0) + triPosition;

	// Instead of shuffling triangle data find positions where to write the triangles
	s_sharedData[threadIdx.y][triangleThread] = triPosition;
	int writePos = s_sharedData[threadIdx.y][tid];
	float4* outTri = ((float4*)c_bvh_in.trisOut) + writePos; // Memory for the first triangle data
	int* outIdx = ((int*)c_bvh_in.trisIndexOut) + writePos; // Memory for the first triangle index

	ASSERT_DIVERGENCE("computeObjectSplitTree triangleWrite", tid);

#ifndef WOOP_TRIANGLES
	// Write out the triangle
	if(tid < end)
	{
		outTri[0] = make_float4(v0);
		outTri[1] = make_float4(v1);
		outTri[2] = make_float4(v2);

		*outIdx = triIdx;
	}
#else
	// Compute Woop representation
	float4 o0, o1, o2;
	calcWoop(v0, v1, v2, o0, o1, o2);

	// Write out the triangle
	if(tid < end)
	{
		outTri[0] = o0;
		outTri[1] = o1;
		outTri[2] = o2;

		*outIdx = triIdx;
	}
#endif // WOOP_TRIANGLES

	// Write the sentinels
	outTri = ((float4*)c_bvh_in.trisOut) + triPosition + 3; // Memory for the sentinel
	outIdx = ((int*)c_bvh_in.trisIndexOut) + triPosition + 3; // Memory for the sentinel

	if(tid < end && tid == segmentEnd-1)
	{
#ifdef WOOP_TRIANGLES
		outTri[0].x = __int_as_float(0x80000000);
#else
		outTri[0] = make_float4(__int_as_float(0x80000000));
#endif
		outIdx[0] = 0;
	}

	// Update the parents with the starts of the triangle list
	if(tid < end && tid == segmentStart)
	{
		//printf("Segment %d: Parent %d <- node %d (%d)!\n", segmentStart, parentIdx, triPosition, taskID);
		taskUpdateParentPtr(g_kdtree, parentIdx, taskID, ~triPosition); // Mark this node as leaf in the hierarchy
#ifdef LEAF_HISTOGRAM
		atomicAdd(&g_taskStackBVH.leafHist[segmentEnd-segmentStart], 1); // Update histogram
#endif
	}
#endif // COMPACT_LAYOUT

	ASSERT_DIVERGENCE("computeObjectSplitTree bottom", tid);


	__threadfence(); // Probably not needed since the data are read-only
	taskFinishObjectSplitTree(tid, s_task[threadIdx.y].popTaskIdx);
}
#endif

//------------------------------------------------------------------------

#ifdef SNAPSHOT_POOL
// Constantly take snapshots of the pool
__device__ void snapshot(int tid, int* header, volatile TaskBVH* tasks, int *unfinished, int *stackTop, volatile int* img, volatile int* red)
{
	int counter = 0; // TESTING ONLY: Allows to undeadlock a failed run!
	int beg = 0;
	int item = 0;
	PoolInfo info;

	// Run until finished
	while(*unfinished < 0 && counter < SNAPSHOT_POOL)
	{
		beg = *stackTop;
		item = beg - tid;
		info.pool = 0;
		info.tasks = 0;
		info.active = 0;
		info.chunks = 0;
		info.depth = 0;
		info.clockStart = clock64();

		// Take a snapshot
#if 0
		while(__any(item >= 0))
		{
			info.pool += min(beg+1, WARP_SIZE);
			// Take a snapshot of the pool
			img[tid] = TaskHeader_Empty;
			img[tid+WARP_SIZE] = 0;
			if(item >= 0)
			{
				img[tid] = header[item];
				img[tid+WARP_SIZE] = tasks[item].depth;
			}

			// Reduce number of nonempty tasks
			red[tid] = (img[tid] != TaskHeader_Empty) ? 1 : 0;
			reduceWarp(tid, red, plus);
			info.tasks += red[tid];

			// Reduce number of active tasks
			red[tid] = (img[tid] > TaskHeader_Active) ? 1 : 0;
			reduceWarp(tid, red, plus);
			info.active += red[tid];

			// Reduce number of work chunks
			red[tid] = (img[tid] > TaskHeader_Active) ? img[tid] : 0;
			reduceWarp(tid, red, plus);
			info.chunks += red[tid];

			// Reduce depths
			red[tid] = (img[tid] > TaskHeader_Active) ? img[tid+WARP_SIZE] : 0;
			reduceWarp(tid, red, plus);
			info.depth += red[tid];

			beg -= WARP_SIZE;
			item = beg - tid;
		}
#else
		info.pool = beg+1;

		while(__any(item >= 0))
		{
			// Take a snapshot of the pool
			img[tid] = TaskHeader_Empty;
			img[tid+WARP_SIZE] = 0;
			if(item >= 0)
			{
				img[tid] = header[item];
				img[tid+WARP_SIZE] = tasks[item].depth;

				if(img[tid] != TaskHeader_Empty)
				{
					info.tasks++;
				}
				if(img[tid] > TaskHeader_Active)
				{
					info.active++;
					info.chunks += img[tid];
					info.depth += img[tid+WARP_SIZE];
				}
			}

			beg -= WARP_SIZE;
			item = beg - tid;
		}

		// Reduce number of nonempty tasks
		red[tid] = info.tasks;
		reduceWarp(tid, red, plus);
		info.tasks = red[tid];

		// Reduce number of active tasks
		red[tid] = info.active;
		reduceWarp(tid, red, plus);
		info.active = red[tid];

		// Reduce number of work chunks
		red[tid] = info.chunks;
		reduceWarp(tid, red, plus);
		info.chunks = red[tid];

		// Reduce depths
		red[tid] = info.depth;
		reduceWarp(tid, red, plus);
		info.depth = red[tid];
#endif

		info.clockEnd = clock64();
		// Compute average depth
		if(info.active > 0)
			info.depth /= (float)info.active;
		else
			info.depth = -1; // Mark no depth info

		// Save the snapshot
		int *g_ptr = (int*)&g_snapshots[counter];
		int *ptr = (int*)&info;
		if(tid < sizeof(PoolInfo)/sizeof(int))
			g_ptr[tid] = ptr[tid];

		// Sleep
		/*clock_t start = clock();
		clock_t end = start + 10;
		while(start < end)
		{
			start = clock();
		}*/

		// Next shot
		counter++;
	}
}
#endif

//------------------------------------------------------------------------

// Main function of the programme - may be the kernel function
extern "C" __global__ void __launch_bounds__(NUM_THREADS, NUM_BLOCKS_PER_SM) build(void)
{
	volatile int* taskAddr = (volatile int*)(&s_task[threadIdx.y]);
	int tid = threadIdx.x;

#ifdef SNAPSHOT_POOL
	int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
	if(warpIdx == 0)
	{
		snapshot(tid, g_taskStackBVH.header, g_taskStackBVH.tasks, &g_taskStackBVH.unfinished, &g_taskStackBVH.top, (volatile int*)&s_task[threadIdx.y], (volatile int*)s_sharedData[threadIdx.y]);
		return;
	}
#endif

/*#if PARALLELISM_TEST >= 0
	int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
	if(warpIdx == 0 && tid == 0)
		printf("Output start!\n");
#endif*/

	s_task[threadIdx.y].lock = LockType_Free; // Prepare task

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT) || defined(COUNT_STEPS_DEQUEUE) || defined(COUNT_STEPS_ALLOC)
	//if(tid == 0)
	{
		maxSteps[threadIdx.y] = 0;
		sumSteps[threadIdx.y] = 0;
		numSteps[threadIdx.y] = 0;
		numRestarts[threadIdx.y] = 0;
	}
#endif

	while(s_task[threadIdx.y].lock != LockType_Free || taskDequeue(tid)) // Main loop of the programme, while there is some task to do
	{
#ifdef SNAPSHOT_WARP
		WarpInfo info;
		s_sharedData[threadIdx.y][3] = 0;
		*(long long int*)&(s_sharedData[threadIdx.y][4]) = 0;
#endif
		if(s_task[threadIdx.y].lock == LockType_Free)
		{
		// Copy first cache line of task to shared memory
#ifdef DIVERGENCE_TEST
			if(s_task[threadIdx.y].popTaskIdx >= 0 && s_task[threadIdx.y].popTaskIdx < g_taskStackBVH.sizePool)
			{
				//taskLoadFirstFromGMEM(tid, s_task[threadIdx.y].popTaskIdx, &s_task[threadIdx.y]);
				TaskBVH* g_task = &g_taskStackBVH.tasks[s_task[threadIdx.y].popTaskIdx];
				taskAddr[tid] = ((int*)g_task)[tid]; // Every thread copies one word of task data
#ifdef DEBUG_INFO
				int offset = 128/sizeof(int); // 128B offset
				taskAddr[tid+offset] = ((int*)g_task)[tid+offset]; // Every thread copies one word of task data
#endif
				
				s_task[threadIdx.y].popStart = s_task[threadIdx.y].popSubtask;
				// If we have poped all of the task's work we do not have to update unfinished atomicaly
				if(s_task[threadIdx.y].popSubtask == s_task[threadIdx.y].origSize-1 && s_task[threadIdx.y].popCount >= s_task[threadIdx.y].origSize)
					s_task[threadIdx.y].lock = LockType_None;
			}
			else
			{
				printf("Fetched task %d out of range!\n", s_task[threadIdx.y].popTaskIdx);
				g_taskStackBVH.unfinished = 1;
			}
#else
			//taskLoadFirstFromGMEM(tid, s_task[threadIdx.y].popTaskIdx, &s_task[threadIdx.y]);
			TaskBVH* g_task = &g_taskStackBVH.tasks[s_task[threadIdx.y].popTaskIdx];
			taskAddr[tid] = ((int*)g_task)[tid]; // Every thread copies one word of task data
#ifdef DEBUG_INFO
			int offset = 128/sizeof(int); // 128B offset
			if(tid < TASK_GLOBAL) // Prevent overwriting local data saved in task
				taskAddr[tid+offset] = ((int*)g_task)[tid+offset]; // Every thread copies one word of task data
#endif
			s_task[threadIdx.y].popStart = s_task[threadIdx.y].popSubtask;
			// If we have poped all of the task's work we do not have to update unfinished atomicaly
			if(s_task[threadIdx.y].popSubtask == s_task[threadIdx.y].origSize-1 && s_task[threadIdx.y].popCount >= s_task[threadIdx.y].origSize)
				s_task[threadIdx.y].lock = LockType_None;
#endif

#ifdef SNAPSHOT_WARP
			// Write out information about this dequeue
			info.reads = s_sharedData[threadIdx.y][3];
			info.tris = s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triStart;
			info.type = s_task[threadIdx.y].type;
			info.chunks = s_task[threadIdx.y].origSize;
			info.popCount = s_task[threadIdx.y].popCount;
			info.depth = s_task[threadIdx.y].depth;
			info.idx = s_task[threadIdx.y].nodeIdx;
			info.stackTop = *((int*)&g_taskStackBVH.top);
			info.clockSearch = *(long long int*)&(s_sharedData[threadIdx.y][4]);
			info.clockDequeue = clock64();
#endif
		}

		//if(s_task[threadIdx.y].popTaskIdx > 7 || (s_task[threadIdx.y].popTaskIdx == 7 && s_task[threadIdx.y].type == 9 && s_task[threadIdx.y].step > 0))
		//if(s_task[threadIdx.y].popTaskIdx > 6 || s_task[threadIdx.y].popTaskIdx == 6 && s_task[threadIdx.y].type > 9)
		//if(s_task[threadIdx.y].popTaskIdx > 7 || (s_task[threadIdx.y].popTaskIdx == 7 && s_task[threadIdx.y].type == 10))
		/*int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
		if(warpIdx == 0)
		{
			if(tid == 0)
			{
				printf("warpId %d\n", blockDim.y*blockIdx.x + threadIdx.y);
				printf("task s_task[threadIdx.y].popTaskIdx: %d\n", s_task[threadIdx.y].popTaskIdx);
				printf("Global unfinished: %d\n", g_taskStackBVH.warpCounter);
				//printf("Header: %d\n", s_task[threadIdx.y].popSubtask);
				printf("Unfinished: %d\n", s_task[threadIdx.y].unfinished);
				printf("Type: %d\n", s_task[threadIdx.y].type);
				printf("TriStart: %d\n", s_task[threadIdx.y].triStart);
				printf("TriEnd: %d\n", s_task[threadIdx.y].triEnd);
				printf("Depend1: %d\n", s_task[threadIdx.y].depend1);
				printf("Depend2: %d\n", s_task[threadIdx.y].depend2);
				printf("Box: (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
					s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
				printf("\n");
			}
		}
		return;*/

		ASSERT_DIVERGENCE("taskProcessWorkUntilDone top", tid);

		switch(s_task[threadIdx.y].type) // Decide what to do
		{
#if SPLIT_TYPE == 1 || SPLIT_TYPE == 2 || SPLIT_TYPE == 3
		case TaskType_Split:
			computeSplit();
			break;
#endif

#if SPLIT_TYPE == 3
		case TaskType_SplitParallel:
			computeSplitParallel();
			break;
#endif

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
		case TaskType_InitMemory:
			do {
				// Initialize the task by copying it from the end of the array
				int *orig = ((int*)&g_redSplits[g_taskStackBVH.sizePool])+s_task[threadIdx.y].popSubtask*WARP_SIZE;
				int *split = ((int*)&g_redSplits[s_task[threadIdx.y].popTaskIdx])+s_task[threadIdx.y].popSubtask*WARP_SIZE;
				split[tid] = orig[tid]; // Each thread copies 1 int-sized variable

				subtasksDone = taskReduceSubtask(s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popStart, s_task[threadIdx.y].popCount);
			} while(subtasksDone == -1);

			__threadfence(); // Probably needed so that next iteration does not read uninitialized data
			taskFinishInit(tid, s_task[threadIdx.y].popTaskIdx, subtasksDone);
			break;
#endif

		case TaskType_BinTriangles:
			computeBins();
			break;

#if BINNING_TYPE == 0 || BINNING_TYPE == 1
			case TaskType_ReduceBins:
			do {
				reduceBins(tid, s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popTaskIdx, s_task[threadIdx.y].step);
				subtasksDone = taskReduceSubtask(s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popStart, s_task[threadIdx.y].popCount);
			} while(subtasksDone == -1);

			__threadfence(); // Probably needed so that next iteration does not read uninitialized data
			taskFinishReduce(tid, s_task[threadIdx.y].popTaskIdx, subtasksDone);
			break;
#endif
#endif

		// --------------------------------------------------

#if SCAN_TYPE < 2
		case TaskType_Sort_PPS1:
			computePPS();
			break;

#if SCAN_TYPE == 1
		case TaskType_Sort_PPS1_Up:
			computePPSUp();
			break;

		case TaskType_Sort_PPS1_Down:
			computePPSDown();
			break;
#endif

		case TaskType_Sort_SORT1:
			computeSort();
			break;
#elif SCAN_TYPE == 2 || SCAN_TYPE == 3
		case TaskType_Sort_SORT1:
			computePartition();
			break;
#else
#error Unknown SCAN_TYPE!
#endif

		// --------------------------------------------------

#ifdef RAYTRI_TEST
		case TaskType_RayTriTestSORT1:
			do {
				int triidx = s_task[threadIdx.y].triStart + s_task[threadIdx.y].popSubtask*WARP_SIZE + tid;
				int* ppsTrisIndex = getPPSTrisIdxPtr(s_task[threadIdx.y].dynamicMemory, s_task[threadIdx.y].triEnd-s_task[threadIdx.y].triStart);
				
				if(triidx < s_task[threadIdx.y].triRight && ((int*)c_bvh_in.ppsTrisIndex)[triidx] > 0)
					printf("Tri error should be -1/0 is %d! Start %d, Left %d, Right %d, End %d\n", ppsTrisIndex[triidx], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triLeft, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd);

				if(triidx >= s_task[threadIdx.y].triRight && triidx < s_task[threadIdx.y].triEnd && ppsTrisIndex[triidx] < 1)
					printf("Tri error should be 1 is %d! Start %d, Left %d, Right %d, End %d\n", ppsTrisIndex[triidx], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triLeft, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd);
				subtasksDone = taskReduceSubtask(s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popStart, s_task[threadIdx.y].popCount);
			} while(subtasksDone == -1);
			
			taskFinishSortSORT1(tid, s_task[threadIdx.y].popTaskIdx, subtasksDone);
			break;
#endif
			
		// --------------------------------------------------

#if defined(OBJECT_SAH)/* && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)*/
		case TaskType_BuildObjectSAH:
			// Build the subtree with object SAH
			computeObjectSplitTree();
			break;
#endif

		// --------------------------------------------------
		}

#ifdef SNAPSHOT_WARP
		if(numSteps[threadIdx.y] - 1 < SNAPSHOT_WARP)
		{
			info.clockFinished = clock64();

			// Save the snapshot
			int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
			int *g_ptr = (int*)&g_snapshots[warpIdx*SNAPSHOT_WARP + numSteps[threadIdx.y] - 1];
			int *ptr = (int*)&info;
			if(tid < sizeof(WarpInfo)/sizeof(int))
				g_ptr[tid] = ptr[tid];
			numSteps[threadIdx.y]++;
		}
#endif

		// Last finished had work for only one warp
		if(s_task[threadIdx.y].lock == LockType_None || s_task[threadIdx.y].lock == LockType_Subtask)
		{
			// Convert to the multiple subtask solution
			//s_task[threadIdx.y].popSubtask = s_task[threadIdx.y].unfinished-1;
			s_task[threadIdx.y].popSubtask = s_task[threadIdx.y].origSize-1;
			s_task[threadIdx.y].popStart = s_task[threadIdx.y].popSubtask;
		}

		ASSERT_DIVERGENCE("taskProcessWorkUntilDone bottom", tid);
	}

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT) || defined(COUNT_STEPS_DEQUEUE) || defined(COUNT_STEPS_ALLOC)
	// Write out work statistics
	if(tid == 0)
	{
		int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
		((float4*)c_bvh_in.debug)[warpIdx] = make_float4(numSteps[threadIdx.y]*1.0f, sumSteps[threadIdx.y]*1.0f/numSteps[threadIdx.y], maxSteps[threadIdx.y]*1.0f, numRestarts[threadIdx.y]*1.0f);
	}
#endif
}

//------------------------------------------------------------------------
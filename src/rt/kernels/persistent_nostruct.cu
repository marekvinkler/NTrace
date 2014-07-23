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
    Divide-and-conquer ray tracing specialization of the framework.

    "Massively Parallel Hierarchical Scene Sorting with Applications in Rendering",
    Marek Vinkler, Michal Hapala, Jiri Bittner and Vlastimil Havran,
    Computer Graphics Forum 2012
*/

#include "rt_common.cu"
#include "CudaNoStructKernels.hpp"

//------------------------------------------------------------------------
// Shared variables.
//------------------------------------------------------------------------

__shared__ volatile Task s_task[NUM_WARPS_PER_BLOCK]; // Memory holding information about the currently processed task
__shared__ volatile Task s_newTask[NUM_WARPS_PER_BLOCK]; // Memory for the new task to be created in
__shared__ volatile int s_sharedData[NUM_WARPS_PER_BLOCK][WARP_SIZE]; // Shared memory for inside warp use
__shared__ volatile int s_owner[NUM_WARPS_PER_BLOCK][WARP_SIZE]; // Another shared pool

//------------------------------------------------------------------------
// Dependency table.
//------------------------------------------------------------------------

struct Dependency
{
	int count;
	int lastBlock;
	int subtaskPriority[7];
	int dependencyTable[7*2];
	int dependencyStatus[7];
};

#define DEPENDENCY_LUT

#ifndef DEPENDENCY_LUT
// First corrdinate is rays, second is triangles
__constant__ Dependency c_dependency[4] = {
	{2, 2, {7, 1}, {0,1, 0,1}, {0, 0}},
	{4, 1, {6, 2, 7, 1}, {0,1, 2,6, 3,6, 3,6}, {-2, -3, 0, 0}},
	{4, 1, {5, 3, 7, 1}, {0,1, 2,6, 3,6, 3,6}, {-2, -3, 0, 0}},
	{7, 2, {6, 5, 3, 2, 7, 4, 1}, {0,1, 0,1, 2,3, 2,3, 4,5, 4,5, 4,5}, {-3, -3, -4, -4, 0, 0, 0}}
};
#else
// LUT for subtask generation with 2^6 slots (indexed by Left-Start, Right-Left and End-Right bits for tris and rays in this order)
// 3 LSB bits are for rays, the next 3 bits are for tris
// If multiple block divisions are possible we choose to have more active tasks earlier in the computation
__constant__ Dependency c_dependency[64] = {
	{0, 0, {0}, {0}, {0}},																			// 000 000
	{0, 0, {0}, {0}, {0}},																			// 000 001
	{0, 0, {0}, {0}, {0}},																			// 000 010
	{0, 0, {0}, {0}, {0}},																			// 000 011
	{0, 0, {0}, {0}, {0}},																			// 000 100
	{0, 0, {0}, {0}, {0}},																			// 000 101
	{0, 0, {0}, {0}, {0}},																			// 000 110
	{0, 0, {0}, {0}, {0}},																			// 000 111
	
	{0, 0, {0}, {0}, {0}},																			// 001 000
	{1, 1, {7}, {0,1}, {0}},																		// 001 001
	{1, 1, {5}, {0,1}, {0}},																		// 001 010
	{2, 1, {5, 7}, {0,1, 2,9}, {-2, 0}},															// 001 011
	{0, 0, {0}, {0}, {0}},																			// 001 100
	{1, 1, {7}, {0,1}, {0}},																		// 001 101
	{1, 1, {5}, {0,1}, {0}},																		// 001 110
	{2, 1, {5, 7}, {0,1, 2,9}, {-2, 0}},															// 001 111
	
	{0, 0, {0}, {0}, {0}},																			// 010 000
	{1, 1, {6}, {0,1}, {0}},																		// 010 001
	{1, 1, {4}, {0,1}, {0}},																		// 010 010
	{2, 1, {6, 4}, {0,1, 2,9}, {-2, 0}},															// 010 011
	{1, 1, {2}, {0,1}, {0}},																		// 010 100
	{2, 1, {6, 2}, {0,1, 2,9}, {-2, 0}},															// 010 101
	{2, 1, {2, 4}, {0,1, 2,9}, {-2, 0}},															// 010 110
	{3, 1, {6, 2, 4}, {0,1, 2,9, 3,9}, {-2, -2, 0}},												// 010 111

	{0, 0, {0}, {0}, {0}},																			// 011 000
	{2, 1, {6, 7}, {0,1, 2,9}, {-2, 0}},															// 011 001
	{2, 1, {5, 4}, {0,1, 2,9}, {-2, 0}},															// 011 010
	{4, 2, {6, 5, 7, 4}, {0,1, 0,1, 2,3, 2,3}, {-3, -3, 0, 0}},										// 011 011
	{1, 1, {2}, {0,1}, {0}},																		// 011 100
	{3, 1, {6, 2, 7}, {0,1, 2,9, 2,9}, {-3, 0, 0}},													// 011 101
	{3, 1, {4, 5, 2}, {0,1, 2,9, 2,9}, {-3, 0, 0}},													// 011 110
	{5, 2, {2, 5, 6, 7, 4}, {0,1, 0,1, 2,9, 3,4, 3,4}, {-2, -3, -3, 0, 0}},							// 011 111

	{0, 0, {0}, {0}, {0}},																			// 100 000
	{0, 0, {0}, {0}, {0}},																			// 100 001
	{1, 1, {3}, {0,1}, {0}},																		// 100 010
	{1, 1, {3}, {0,1}, {0}},																		// 100 011
	{1, 1, {1}, {0,1}, {0}},																		// 100 100
	{1, 1, {1}, {0,1}, {0}},																		// 100 101
	{2, 1, {3, 1}, {0,1, 2,9}, {-2, 0}},															// 100 110
	{2, 1, {3, 1}, {0,1, 2,9}, {-2, 0}},															// 100 111

	{0, 0, {0}, {0}, {0}},																			// 101 000
	{1, 1, {7}, {0,1}, {0}},																		// 101 001
	{2, 1, {5, 3}, {0,1, 2,9}, {-2, 0}},															// 101 010
	{3, 1, {5, 3, 7}, {0,1, 2,9, 2,9}, {-3, 0, 0}},													// 101 011
	{1, 1, {1}, {0,1}, {0}},																		// 101 100
	{2, 2, {7, 1}, {0,1, 0,1}, {0, 0}},																// 101 101
	{3, 1, {3, 5, 1}, {0,1, 2,9, 2,9}, {-3, 0, 0}},													// 101 110
	{4, 2, {3, 7, 5, 1}, {0,1, 0,1, 2,3, 2,9}, {-3, -2, 0, 0}},										// 101 111

	{0, 0, {0}, {0}, {0}},																			// 110 000
	{1, 1, {6}, {0,1}, {0}},																		// 110 001
	{2, 1, {3, 4}, {0,1, 2,9}, {-2, 0}},															// 110 010
	{3, 1, {4, 6, 3}, {0,1, 2,9, 2,9}, {-3, 0, 0}},													// 110 011
	{2, 1, {2, 1}, {0,1, 2,9}, {-2, 0}},															// 110 100
	{3, 1, {2, 6, 1}, {0,1, 2,9, 2,9}, {-3, 0, 0}},													// 110 101
	{4, 2, {3, 2, 4, 1}, {0,1, 0,1, 2,3, 2,3}, {-3, -3, 0, 0}},										// 110 110
	{5, 2, {6, 3, 2, 4, 1}, {0,1, 0,1, 2,9, 3,4, 3,4}, {-2, -3, -3, 0, 0}},							// 110 111

	{0, 0, {0}, {0}, {0}},																			// 111 000
	{2, 1, {6, 7}, {0,1, 2,9}, {-2, 0}},															// 111 001
	{3, 1, {5, 3, 4}, {0,1, 2,9, 3,9}, {-2, -2, 0}},												// 111 010
	{5, 2, {3, 6, 5, 7, 4}, {0,1, 0,1, 2,9, 3,4, 3,4}, {-2, -3, -3, 0, 0}},							// 111 011
	{2, 1, {2, 1}, {0,1, 2,9}, {-2, 0}},															// 111 100
	{4, 2, {2, 7, 6, 1}, {0,1, 0,1, 2,3, 2,9}, {-3, -2, 0, 0}},										// 111 101
	{5, 2, {5, 2, 3, 4, 1}, {0,1, 0,1, 2,9, 3,4, 3,4}, {-2, -3, -3, 0, 0}},							// 111 110
	//{7, 2, {6, 5, 3, 2, 7, 4, 1}, {0,1, 0,1, 3,9, 2,9, 2,3, 4,5, 4,5}, {-3, -3, -3, -3, 0, 0, 0}},	// 111 111
	{7, 2, {6, 5, 7, 3, 2, 4, 1}, {0,1, 0,1, 2,3, 3,9, 2,9, 5,6, 5,6}, {-3, -3, 0, -3, -3, 0, 0}},	// 111 111
};
#endif

//------------------------------------------------------------------------
// Function Headers.
//------------------------------------------------------------------------
__device__ void splitCost(int tid, int subtask, int rayStart, int rayEnd, int triStart, int triEnd, const volatile CudaAABB& bbox,
	volatile float4& bestPlane, volatile float& bestCost, volatile int& bestOrder);

// Loading and saving functions
__device__ __forceinline__ void taskLoadFirstFromGMEM(int tid, int taskIdx, volatile Task& task);
__device__ __forceinline__ void taskLoadSecondFromGMEM(int tid, int taskIdx, volatile Task& task);
__device__ __forceinline__ void taskSaveFirstToGMEM(int tid, int taskIdx, const volatile Task& task);
__device__ __forceinline__ void taskSaveSecondToGMEM(int tid, int taskIdx, const volatile Task& task);

//------------------------------------------------------------------------

// Copies first cache line of the Task taskIdx to task
__device__ __forceinline__ void taskLoadFirstFromGMEM(int tid, int taskIdx, volatile Task& task)
{
	ASSERT_DIVERGENCE("taskLoadFirstFromGMEM top", tid);
	volatile int* taskAddr = (volatile int*)(&task);
	Task* g_task = &g_taskStack.tasks[taskIdx];
	taskAddr[tid] = ((int*)g_task)[tid]; // Every thread copies one word of task data
	ASSERT_DIVERGENCE("taskLoadFirstFromGMEM bottom", tid);

#ifdef DEBUG_INFO
	taskLoadSecondFromGMEM(tid, taskIdx, task); // Save the debug info statistics as well
#endif
}

// Copies second cache line of the Task taskIdx to task
__device__ __forceinline__ void taskLoadSecondFromGMEM(int tid, int taskIdx, volatile Task& task)
{
	ASSERT_DIVERGENCE("taskLoadSecondFromGMEM top", tid);
	volatile int* taskAddr = (volatile int*)(&task);
	Task* g_task = &g_taskStack.tasks[taskIdx];
	int offset = 128/sizeof(int); // 128B offset
	if(tid < TASK_GLOBAL) // Prevent overwriting local data saved in task
		taskAddr[tid+offset] = ((int*)g_task)[tid+offset]; // Every thread copies one word of task data
	ASSERT_DIVERGENCE("taskLoadSecondFromGMEM bottom", tid);
}

//------------------------------------------------------------------------

// Copies first cache line of the task to Task taskIdx
__device__ __forceinline__ void taskSaveFirstToGMEM(int tid, int taskIdx, const volatile Task& task)
{
	ASSERT_DIVERGENCE("taskSaveFirstToGMEM top", tid);
	// Copy the data to global memory
	int* taskAddr = (int*)(&g_taskStack.tasks[taskIdx]);
	taskAddr[tid] = ((const volatile int*)&task)[tid]; // Every thread copies one word of data of its task
	ASSERT_DIVERGENCE("taskSaveFirstToGMEM bottom", tid);

#ifdef DEBUG_INFO
	taskSaveSecondToGMEM(tid, taskIdx, task); // Save the debug info statistics as well
#endif
}

// Copies second cache line of the task to Task taskIdx
__device__ __forceinline__ void taskSaveSecondToGMEM(int tid, int taskIdx, const volatile Task& task)
{
	ASSERT_DIVERGENCE("taskSaveSecondToGMEM top", tid);
	// Copy the data to global memory
	int* taskAddr = (int*)(&g_taskStack.tasks[taskIdx]);
	int offset = 128/sizeof(int); // 128B offset
	taskAddr[tid+offset] = ((const volatile int*)&task)[tid+offset]; // Every thread copies one word of data of its task
	ASSERT_DIVERGENCE("taskSaveSecondToGMEM bottom", tid);
}

//------------------------------------------------------------------------

__device__ __forceinline__ int taskPopCount(int status)
{
	//return 1;
	//return max((status / NUM_WARPS), 1);
	//return (status / NUM_WARPS) + 1;
	return c_env.popCount;
}

//------------------------------------------------------------------------

// Computes index into the dependency table coding which tasks to add
__device__ __forceinline__ unsigned int taskComputeSubtasksIndex()
{
	unsigned int index = 0;

#ifndef DEPENDENCY_LUT
	unsigned int rayMiddle = (s_task[threadIdx.y].rayRight - s_task[threadIdx.y].rayLeft) > 0;
	unsigned int triMiddle = (s_task[threadIdx.y].triRight - s_task[threadIdx.y].triLeft) > 0;
	index = 2*rayMiddle + triMiddle;
#else
	unsigned int triLeft = (s_task[threadIdx.y].triLeft - s_task[threadIdx.y].triStart) > 0;
	index |= triLeft << 5;
#ifdef LUTINDEX_TEST
	if(triLeft != 0 && triLeft << 5 != 32)
		printf("triLeft error value %u\n", triLeft);
#endif
	unsigned int triMiddle = (s_task[threadIdx.y].triRight - s_task[threadIdx.y].triLeft) > 0;
	index |= triMiddle << 4;
#ifdef LUTINDEX_TEST
	if(triMiddle != 0 && triMiddle << 4 != 16)
		printf("triMiddle error value %u\n", triMiddle);
#endif
	unsigned int triRight = (s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triRight) > 0;
	index |= triRight << 3;
#ifdef LUTINDEX_TEST
	if(triRight != 0 && triRight << 3 != 8)
		printf("triRight error value %u\n", triRight);
#endif

	unsigned int rayLeft = (s_task[threadIdx.y].rayLeft - s_task[threadIdx.y].rayStart) > 0;
	index |= rayLeft << 2;
#ifdef LUTINDEX_TEST
	if(rayLeft != 0 && rayLeft << 2 != 4)
		printf("rayLeft error value %u\n", rayLeft);
#endif
	unsigned int rayMiddle = (s_task[threadIdx.y].rayRight - s_task[threadIdx.y].rayLeft) > 0;
	index |= rayMiddle << 1;
#ifdef LUTINDEX_TEST
	if(rayMiddle != 0 && rayMiddle << 1 != 2)
		printf("rayMiddle error value %u\n", rayMiddle);
#endif
	unsigned int rayRight = (s_task[threadIdx.y].rayEnd - s_task[threadIdx.y].rayRight) > 0;
	index |= rayRight;
#ifdef LUTINDEX_TEST
	if(rayRight != 0 && rayRight != 1)
		printf("rayRight error value %u\n", rayRight);

	if(index >= 64)
		printf("Invalid index %u!\n", index);
#endif
#endif

	return index;
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
	//int endAfter = TaskType_Sort_PPS2;
	//int endAfter = TaskType_Sort_PPS2_Up;
	//int endAfter = TaskType_Sort_PPS2_Down;
	//int endAfter = TaskType_Sort_SORT2;
	//int endAfter = TaskType_AABB_Min;
	//int endAfter = TaskType_AABB;
	//int endAfter = TaskType_Max;

#ifdef DEBUG_PPS
	int endAfter = TaskType_Sort_PPS1_Down;
#endif

	if(phase == endAfter && s_task[threadIdx.y].type != phase)
	{
#ifdef KEEP_ALL_TASKS
		taskSaveFirstToGMEM(tid, taskIdx, s_task[threadIdx.y]);
#endif
		g_taskStack.unfinished = 1;
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
			g_taskStack.header[taskIdx] = TaskHeader_Locked;
		}
		else
#endif
		{
			// Take some work for this warp
			s_task[threadIdx.y].lock = LockType_Subtask;
			s_task[threadIdx.y].popCount = popCount;
			s_task[threadIdx.y].unfinished -= popCount;

			g_taskStack.header[taskIdx] = s_task[threadIdx.y].unfinished; // Restart this task
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
			s_sharedData[threadIdx.y][0] = atomicSub(&g_taskStack.tasks[taskIdx].unfinished, countDown); // Lower the number of unfinished tasks
	}

	ASSERT_DIVERGENCE("taskCheckFinished top", tid);

	return s_sharedData[threadIdx.y][0] == countDown; // Finished is the value before Dec, thus == countDown means last. We have finished the task and are responsible for cleaning up
}

//------------------------------------------------------------------------

// Decides what type of task should be created
__device__ bool taskTerminationCriteria(int rays, int tris, int& termCrit)
{
	//if(s_task[threadIdx.y].depth < 2) // Unknown error if we try to split an empty task
	//	return false;
	if(s_task[threadIdx.y].depth > (c_env.optMaxDepth-1))
	{
		termCrit = TerminatedBy_Depth;
		return true;
	}

	if(s_task[threadIdx.y].subFailureCounter > c_env.failureCount)
	{
		termCrit = TerminatedBy_FailureCounter;
		return true; // Trivial computation
	}

	float total = (float)rays * (float)tris; // The work to be done

	if(total >= 32768) // Prevent huge leafs
	//if(tris >= 1024) // Prevent huge leafs
	{
		//termCrit = TerminatedBy_TotalLimit;
		termCrit = TerminatedBy_None;
		return false;
	}

	if(rays <= c_env.rayLimit || tris <= c_env.triLimit)
	{
		termCrit = TerminatedBy_OverheadLimit;
		return true;
	}

	// Evaluate if the termination cirteria are met
	float intersectionCost = c_env.optCi * total;
	//float subdivisionCost = c_env.optCt * (rays + tris);
	float subdivisionCost = c_env.optCtr * rays + c_env.optCtt * tris;

	if(intersectionCost < subdivisionCost)
	{
		termCrit = TerminatedBy_Cost;
		return true; // Trivial computation
	}

	termCrit = TerminatedBy_None;
	return false; // Continue subdivision
}

//------------------------------------------------------------------------

// Decides what type of task should be created
__device__ void taskDecideType(volatile Task* newTask)
{
	int termCrit;
	int rays = newTask->rayEnd - newTask->rayStart;
	int tris = newTask->triEnd - newTask->triStart;

	newTask->lock = LockType_Free;
	newTask->axis = (s_task[threadIdx.y].axis+1)%3;

	if(taskTerminationCriteria(rays, tris, termCrit))
	{
#ifndef CLIP_INTERSECT
		newTask->type = TaskType_Intersect;
#else
		newTask->type = TaskType_ClipPPS;
#endif

#if ISECT_TYPE == 0
		newTask->unfinished = taskWarpSubtasks(rays); // Number of warp sized subtasks, parallelism by rays
#elif ISECT_TYPE == 1
		newTask->unfinished = taskWarpSubtasks(rays*tris); // Number of warp sized subtasks, parallelism by intersection pairs
#endif

#ifdef DEBUG_INFO
		newTask->clippedRays = 0;
#endif
	}
	else
	{
#if SPLIT_TYPE == 0
		// Precompute the splitting plane
		int axis = newTask->axis;
		//int axis = taskAxis(newTask->splitPlane, newTask->bbox, s_sharedData[threadIdx.y][WARP_SIZE-1], s_task[threadIdx.y].axis);
		splitMedian(threadIdx.x, axis, newTask->splitPlane, newTask->bbox);

		int wRays = taskWarpSubtasks(rays);
		int wTris = taskWarpSubtasks(tris);
		newTask->bestOrder = wRays;
		newTask->unfinished = wRays + wTris;
		newTask->type = taskChooseScanType1();
#elif SPLIT_TYPE == 1
#error No longer supported!
		newTask->type = TaskType_Split;
		newTask->bestCost = CUDART_INF_F;
		int evaluatedCandidates = getNumberOfSamples(rays) + getNumberOfSamples(tris);
		newTask->unfinished = taskWarpSubtasks(c_env.optPlaneSelectionOverhead * (rays + tris)/evaluatedCandidates); // Number of warp sized subtasks
		if(newTask->unfinished == 1)
			newTask->lock = LockType_None; // Set flag to skip update
#elif SPLIT_TYPE == 2
		newTask->type = TaskType_Split;
		newTask->unfinished = 1;
#elif SPLIT_TYPE == 3
		int evaluatedRays = taskWarpSubtasks(getNumberOfSamples(rays));
		int evaluatedTris = taskWarpSubtasks(getNumberOfSamples(tris));
		newTask->unfinished = PLANE_COUNT*(evaluatedRays+evaluatedTris); // Each WARP_SIZE rays and tris add their result to one plane
		
		// Do split immediately or add task
		int popCount = taskPopCount(newTask->unfinished);
		if(newTask->unfinished <= popCount*POP_MULTIPLIER) // No need to share through global memory
		{
			int axis = newTask->axis;
			//int axis = taskAxis(newTask->splitPlane, newTask->bbox, s_sharedInt[threadIdx.y], s_task[threadIdx.y].axis);
			//splitCost(threadIdx.x, 0, newTask->rayStart, newTask->rayEnd, newTask->triStart, newTask->triEnd,
			//	newTask->bbox, newTask->splitPlane, newTask->bestCost, newTask->bestOrder);
			splitMedian(threadIdx.x, axis, newTask->splitPlane, newTask->bbox);

			int wRays = taskWarpSubtasks(rays);
			int wTris = taskWarpSubtasks(tris);
			newTask->bestOrder = wRays;
			newTask->unfinished = wRays + wTris;
			newTask->type = taskChooseScanType1();
			
			//newTask->unfinished = 1;
		}
		else
		{
			newTask->type = TaskType_SplitParallel;
		}
#else
#error Unsupported SPLIT_TYPE!
#endif // SPLIT_TYPE == 0
	}
	newTask->step = 0;
	newTask->depth = s_task[threadIdx.y].depth+1;
	newTask->subFailureCounter = s_task[threadIdx.y].subFailureCounter;
	newTask->origSize = newTask->unfinished;
#if SCAN_TYPE == 2 || SCAN_TYPE == 3
	newTask->rayLeft = newTask->rayStart;
	newTask->rayRight = newTask->rayEnd;
	newTask->triLeft = newTask->triStart;
	newTask->triRight = newTask->triEnd;
#endif

#ifdef DEBUG_INFO
	newTask->terminatedBy = termCrit;
	newTask->sync = 0;

	newTask->clockStart = 0;
	newTask->clockEnd = 0;
#endif
}

//------------------------------------------------------------------------

// Prepares a new task based on its ID
__device__ void taskCreateSubtask(int tid, volatile Task* newTask, int subIdx)
{
	ASSERT_DIVERGENCE("taskCreateSubtask", tid);

	volatile float* bbox = (volatile float*)&(newTask->bbox);
	volatile const float* srcBox = (volatile const float*)&s_sharedData[threadIdx.y][10];

	//if(tid == 4)
	{
		switch(subIdx) // Choose which subtask we are creating
		{
		case 6:
			newTask->rayStart   = s_task[threadIdx.y].rayRight;
			newTask->rayEnd     = s_task[threadIdx.y].rayEnd;
			newTask->triStart   = s_task[threadIdx.y].triLeft;
			newTask->triEnd     = s_task[threadIdx.y].triRight;
			srcBox = (volatile const float*)&(s_task[threadIdx.y].bboxMiddle);
			break;

		case 5:
			newTask->rayStart   = s_task[threadIdx.y].rayLeft;
			newTask->rayEnd     = s_task[threadIdx.y].rayRight;
			newTask->triStart   = s_task[threadIdx.y].triRight;
			newTask->triEnd     = s_task[threadIdx.y].triEnd;
			srcBox = (volatile const float*)&(s_task[threadIdx.y].bboxRight);
			break;

		case 3:
			newTask->rayStart   = s_task[threadIdx.y].rayLeft;
			newTask->rayEnd     = s_task[threadIdx.y].rayRight;
			newTask->triStart   = s_task[threadIdx.y].triStart;
			newTask->triEnd     = s_task[threadIdx.y].triLeft;
			srcBox = (volatile const float*)&(s_task[threadIdx.y].bboxLeft);
			break;

		case 2:
			newTask->rayStart   = s_task[threadIdx.y].rayStart;
			newTask->rayEnd     = s_task[threadIdx.y].rayLeft;
			newTask->triStart   = s_task[threadIdx.y].triLeft;
			newTask->triEnd     = s_task[threadIdx.y].triRight;
			srcBox = (volatile const float*)&(s_task[threadIdx.y].bboxMiddle);
			break;

		case 7:
			newTask->rayStart   = s_task[threadIdx.y].rayRight;
			newTask->rayEnd     = s_task[threadIdx.y].rayEnd;
			newTask->triStart   = s_task[threadIdx.y].triRight;
			newTask->triEnd     = s_task[threadIdx.y].triEnd;
			srcBox = (volatile const float*)&(s_task[threadIdx.y].bboxRight);
			break;

		case 4:
			newTask->rayStart   = s_task[threadIdx.y].rayLeft;
			newTask->rayEnd     = s_task[threadIdx.y].rayRight;
			newTask->triStart   = s_task[threadIdx.y].triLeft;
			newTask->triEnd     = s_task[threadIdx.y].triRight;
			srcBox = (volatile const float*)&(s_task[threadIdx.y].bboxMiddle);
			break;

		case 1:
			newTask->rayStart   = s_task[threadIdx.y].rayStart;
			newTask->rayEnd     = s_task[threadIdx.y].rayLeft;
			newTask->triStart   = s_task[threadIdx.y].triStart;
			newTask->triEnd     = s_task[threadIdx.y].triLeft;
			srcBox = (volatile const float*)&(s_task[threadIdx.y].bboxLeft);
			break;
		}
	}

	// Copy CudaAABB from corresponding task
	if(tid < sizeof(CudaAABB)/sizeof(float))
	{
		bbox[tid] = srcBox[tid];
	}

	taskDecideType(newTask);
}

//------------------------------------------------------------------------

#if ENQUEUE_TYPE != 3

// Adds subtasks of a task into a global task queue
__device__ void taskEnqueueSubtasks(int tid, int taskIdx, unsigned int index)
{
	ASSERT_DIVERGENCE("taskEnqueueSubtasks top", tid);

	int *stackTop = &g_taskStack.top;
#if ENQUEUE_TYPE == 0
	int beg = *stackTop;
	bool goRight = true;
#elif ENQUEUE_TYPE == 1
	int beg = g_taskStack.bottom;
#elif ENQUEUE_TYPE == 2
	int beg = *stackTop;
#elif ENQUEUE_TYPE == 3
#error Unsupported ENQUEUE_TYPE!
#endif

	s_sharedData[threadIdx.y][0] = s_task[threadIdx.y].depend1;
	s_sharedData[threadIdx.y][1] = s_task[threadIdx.y].depend2;
	s_sharedData[threadIdx.y][9] = DependType_None;

	Dependency& table = c_dependency[index];
//#pragma unroll 7 // OPTIMIZE: Is this beneficial?
	for(int i = 0; i < table.count; i++)
	{
		taskCreateSubtask(tid, &s_newTask[threadIdx.y], table.subtaskPriority[i]); // Fill newTask with valid task for ID=subtaskPriority[i]
		int newStatus = (table.dependencyStatus[i] == 0) ? s_newTask[threadIdx.y].unfinished : table.dependencyStatus[i]; // If task is not waiting (== 0) set status to the number of subtasks, else use the waiting count

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle0", tid);

		// Add task to work queue
		s_newTask[threadIdx.y].depend1 = s_sharedData[threadIdx.y][table.dependencyTable[i*2+0]];
		s_newTask[threadIdx.y].depend2 = s_sharedData[threadIdx.y][table.dependencyTable[i*2+1]];

#ifdef DEBUG_INFO
		s_newTask[threadIdx.y].parent = taskIdx;
		s_newTask[threadIdx.y].taskID = table.subtaskPriority[i];
#endif

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT) || defined(COUNT_STEPS_CACHE)
		numReads[threadIdx.y] = 0;
#endif

		/*if(s_newTask[threadIdx.y].type == TaskType_Intersect)
		{
			goRight = false;
			beg = *stackTop;
		}
		else
		{
			goRight = true;
		}*/

#if ENQUEUE_TYPE == 0
		if(goRight)
		{
			taskEnqueueRight(tid, g_taskStack.header, s_sharedData[threadIdx.y], newStatus, beg, 0); // Go right of beg and fill empty tasks

			if(beg < 0) // Not added when going right
			{
				goRight = false;
				beg = *stackTop;
			}
		}

		// Cannot be else, both paths may need to be taken for same i
		if(!goRight)
		{
			taskEnqueueLeft(tid, g_taskStack.header, s_sharedData[threadIdx.y], newStatus, beg, &g_taskStack.unfinished, g_taskStack.sizePool); // Go left of beg and fill empty tasks
			if(beg == -1)
				return;
		}
#else
		taskEnqueueLeft(tid, g_taskStack.header, s_sharedData[threadIdx.y], newStatus, beg, &g_taskStack.unfinished, g_taskStack.sizePool); // Go left of beg and fill empty tasks
		if(beg == -1)
			return;
#endif

		// All threads
		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle1", tid);

		/*if(tid == 0)
		{
			printf("warpId %d\n", blockDim.y*blockIdx.x + threadIdx.y);
			printf("Task taskIdx: %d\n", beg);
			printf("Header: %d\n", newStatus);
			printf("Unfinished: %d\n", s_newTask[threadIdx.y].unfinished);
			printf("Type: %d\n", s_newTask[threadIdx.y].type);
			printf("RayStart: %d\n", s_newTask[threadIdx.y].rayStart);
			printf("RayEnd: %d\n", s_newTask[threadIdx.y].rayEnd);
			printf("TriStart: %d\n", s_newTask[threadIdx.y].triStart);
			printf("TriEnd: %d\n", s_newTask[threadIdx.y].triEnd);
			printf("Depend1: %d\n", s_newTask[threadIdx.y].depend1);
			printf("Depend2: %d\n", s_newTask[threadIdx.y].depend2);
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
			atomicMax(&g_taskStack.top, beg); // Update the stack top to be a larger position than all nonempty tasks
#if ENQUEUE_TYPE == 1
			atomicMax(&g_taskStack.bottom, beg);  // Update the stack bottom
#endif
		}

#ifdef DIVERGENCE_TEST
		if(beg >= 0 && beg < g_taskStack.sizePool) // TESTING ONLY - WRITE WILL CAUSE "UNKNOWN ERROR" IF WARP DIVERGES
			taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
		else
			printf("Task adding on invalid index: %d, Tid %d\n", beg, tid);
#else
		taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
#endif

#if SPLIT_TYPE == 3
#if PLANE_COUT > WARP_SIZE // Clear cannot be processed by a single warp
		assert(PLANE_COUT < WARP_SIZE);
#endif
		// Clear the SplitStack for the next use
		if(s_newTask[threadIdx.y].type == TaskType_SplitParallel && tid < PLANE_COUNT)
		{
			int *split = (int*)&g_splitStack[beg];
#pragma unroll
			for(int j = 0; j < sizeof(SplitData)/sizeof(int); j++) // Zero 1 SplitData = PLANE_COUNT*5 ints
			{
				split[tid] = 0; // Each thread clears 1 int-sized variable
				split += PLANE_COUNT; // Each thread move to the next clear task
			}
		}
#endif

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle2", tid);

		if(newStatus > TaskHeader_Active) // No need to wait if the task is inactive anyway
		{
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
				if(/*beg == 124 || */(active == 0 && i == table.count-1))
#else
				if(active == 0)
#endif
				{
					//printf("Warp %d no active tasks before adding task with %d subtasks\n", warpIdx, newStatus);
					g_taskStack.unfinished = 1;
				}
			}
#endif
			__threadfence(); // Make sure task is copied to the global memory before we unlock it

#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
			if(tid == 24)
				taskCacheActive(beg, g_taskStack.active, &g_taskStack.activeTop);
#endif
		}

		// Unlock the task - set the task status
#ifdef CUTOFF_DEPTH
		if(s_newTask[threadIdx.y].depth > c_env.optCutOffDepth)
			g_taskStack.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks
		else
			g_taskStack.header[beg] = newStatus; // This operation is atomic anyway
#else
		g_taskStack.header[beg] = newStatus; // This operation is atomic anyway
		//g_taskStack.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks
#endif

		s_sharedData[threadIdx.y][i+2] = beg; // Save the index for later use as dependency, +2 moves it after parent dependencies

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
	//g_taskStack.unfinished = 1; // Finish computation after first task

	ASSERT_DIVERGENCE("taskEnqueueSubtasks aftercycle", tid);
}

#else // ENQUEUE_TYPE == 3

//------------------------------------------------------------------------

// Adds subtasks of a task into a global task queue
__device__ void taskEnqueueSubtasksCache(int tid, int taskIdx, unsigned int index)
{
	ASSERT_DIVERGENCE("taskEnqueueSubtasksParallel top", tid);

	int *stackTop = &g_taskStack.top;
	unsigned int *emptyTop = &g_taskStack.emptyTop;
	unsigned int *emptyBottom = &g_taskStack.emptyBottom;
	int pos = *emptyBottom;
	int top = *emptyTop;
	int beg = -1;
	int mid = -1;
	int status;

	s_sharedData[threadIdx.y][0] = s_task[threadIdx.y].depend1;
	s_sharedData[threadIdx.y][1] = s_task[threadIdx.y].depend2;
	s_sharedData[threadIdx.y][9] = DependType_None;

	Dependency& table = c_dependency[index];
//#pragma unroll 7 // OPTIMIZE: Is this beneficial?
	for(int i = 0; i < table.count; i++)
	{
		taskCreateSubtask(tid, &s_newTask[threadIdx.y], table.subtaskPriority[i]); // Fill newTask with valid task for ID=subtaskPriority[i]
		int newStatus = (table.dependencyStatus[i] == 0) ? s_newTask[threadIdx.y].unfinished : table.dependencyStatus[i]; // If task is not waiting (== 0) set status to the number of subtasks, else use the waiting count

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle0", tid);

		// Add task to work queue
		s_newTask[threadIdx.y].depend1 = s_sharedData[threadIdx.y][table.dependencyTable[i*2+0]];
		s_newTask[threadIdx.y].depend2 = s_sharedData[threadIdx.y][table.dependencyTable[i*2+1]];

#ifdef DEBUG_INFO
		s_newTask[threadIdx.y].parent = taskIdx;
		s_newTask[threadIdx.y].taskID = table.subtaskPriority[i];
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
			taskEnqueueCache(tid, &g_taskStack, s_sharedData[threadIdx.y], status, pos, beg, top);
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

		if(status != TaskHeader_Empty)
		{
			if(mid == -1)
			{
				//beg = g_taskStack.bottom;
				beg = max(*stackTop - WARP_SIZE, 0);
				mid = 0;
			}

#ifdef COUNT_STEPS_LEFT
				numReads[threadIdx.y]++;
#endif
			taskEnqueueLeft(tid, g_taskStack.header, s_sharedData[threadIdx.y], newStatus, beg, &g_taskStack.unfinished, g_taskStack.sizePool); // Go left of beg and fill empty tasks
			if(beg == -1)
				return;
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
			atomicMax(&g_taskStack.top, beg); // Update the stack top to be a larger position than all nonempty tasks
		}

#ifdef DIVERGENCE_TEST
		if(beg >= 0 && beg < g_taskStack.sizePool) // TESTING ONLY - WRITE WILL CAUSE "UNKNOWN ERROR" IF WARP DIVERGES
			taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
		else
			printf("Task adding on invalid index: %d, Tid %d\n", beg, tid);
#else
		taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
#endif

#if SPLIT_TYPE == 3
#if PLANE_COUT > WARP_SIZE // Clear cannot be processed by a single warp
		assert(PLANE_COUT < WARP_SIZE);
#endif
		// Clear the SplitStack for the next use
		if(s_newTask[threadIdx.y].type == TaskType_SplitParallel && tid < PLANE_COUNT)
		{
			int *split = (int*)&g_splitStack[beg];
#pragma unroll
			for(int j = 0; j < 5; j++) // Zero 1 SplitData = PLANE_COUNT*5 ints
			{
				split[tid] = 0; // Each thread clears 1 int-sized variable
				split += PLANE_COUNT; // Each thread move to the next clear task
			}
		}
#endif

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle2", tid);

		if(newStatus > TaskHeader_Active) // No need to wait if the task is inactive anyway
		{
#if PARALLELISM_TEST >= 0
			if(tid == 0)
			{
				int active = atomicAdd(&g_numActive, 1);
				int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
				if(active == 0)
					printf("Warp %d no active tasks before adding task with %d subtasks\n", warpIdx, newStatus);
			}
#endif
			__threadfence(); // Make sure task is copied to the global memory before we unlock it

#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
			if(tid == 24)
				taskCacheActive(beg, g_taskStack.active, &g_taskStack.activeTop);
#endif
		}

		// Unlock the task - set the task status
		g_taskStack.header[beg] = newStatus; // This operation is atomic anyway
		//g_taskStack.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks

		s_sharedData[threadIdx.y][i+2] = beg; // Save the index for later use as dependency, +2 moves it after parent dependencies
		beg++; // Move for next item

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle3", tid);
	}

	ASSERT_DIVERGENCE("taskEnqueueSubtasks aftercycle", tid);
}

#endif // ENQUEUE_TYPE == 3

//------------------------------------------------------------------------

#if DEQUEUE_TYPE <= 3

__device__ __noinline__ bool taskDequeue(int tid)
{
	ASSERT_DIVERGENCE("taskDequeue", tid);

#if PARALLELISM_TEST >= 0
	int* active = &g_numActive;
#endif

	if(tid == 13) // Only thread 0 acquires the work
	{
		// Initiate variables
		int* header = g_taskStack.header;
		int *unfinished = &g_taskStack.unfinished;
		int *stackTop = &g_taskStack.top;

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
		int beg = warpIdx % (*stackTop + 2);
#elif DEQUEUE_TYPE == 3
		unsigned int *activeTop = &g_taskStack.activeTop;
		int* act = g_taskStack.active;
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
				popCount = taskPopCount(*((int*)&g_taskStack.tasks[beg].origSize));
				// Try acquire the current task - decrease it
				status = atomicSub(&g_taskStack.header[beg], popCount); // Try to update and return the current value
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

		while(counter < WAIT_COUNT && *unfinished < 0 && status <= TaskHeader_Active) // while g_taskStack is not empty and we have not found ourselves a task
		{
			// Find first active task, end if we have reached start of the array
#if DEQUEUE_TYPE == 0 || DEQUEUE_TYPE == 2 || DEQUEUE_TYPE == 3
			while(beg >= 0 && (status = header[beg]) <= TaskHeader_Active)
			{
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
				*unfinished = 0;
				int warpIdx = (blockDim.y*blockIdx.x + threadIdx.y);
				printf("Warp %d ended on task: %d\n", warpIdx, beg);
				break;
			}*/
#endif

			if(beg < 0) // We have found no active task
			{
#if DEQUEUE_TYPE == 0 || DEQUEUE_TYPE == 2 || DEQUEUE_TYPE == 3
				beg = *stackTop; // Try again from a new beginning
#endif
				counter++;

				/*// Sleep - works but increases time if used always
				clock_t start = clock();
				clock_t end = start + 1000;
				while(start < end)
				{
				//g_taskStack.tasks[0].padding1 = 0; // DOES NOT SEEM TO BE BETTER
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
			popCount = taskPopCount(*((int*)&g_taskStack.tasks[beg].origSize));
			// Try acquire the current task - decrease it
			status = atomicSub(&g_taskStack.header[beg], popCount); // Try to update and return the current value

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
				beg--;
#elif DEQUEUE_TYPE == 1
				if((blockDim.y*blockIdx.x + threadIdx.y) & 0) // Even warpID
					beg--;
				else
					beg++;
#endif
				// OPTIMIZE: we shall move beg--; as the first statement in the outer while and start with g_taskStack.top+1.
			}
		}

#ifdef SNAPSHOT_WARP
		s_sharedData[threadIdx.y][3] = readCounter;
#endif

		// Distribute information to all threads through shared memory
		if(counter >= WAIT_COUNT || status <= TaskHeader_Active || *unfinished == 0) // g_taskStack is empty, no more work to do
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
		maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], readCounter);
		sumSteps[threadIdx.y] += readCounter;
		numSteps[threadIdx.y]++;
		numRestarts[threadIdx.y] += counter;

		//atomicExch(&g_taskStack.active, beg); // Update the last active position

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
	int* header = g_taskStack.header;
	int* unfinished = &g_taskStack.unfinished;
	int* stackTop = &g_taskStack.top;
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
	int beg = (topChunk - (warpIdx % topChunk)) * WARP_SIZE - tid;

#elif DEQUEUE_TYPE == 5
	//unsigned int *activeTop = &g_taskStack.activeTop;
	int* cache = g_taskStack.active;
	s_task[threadIdx.y].popSubtask = status;

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

	//while(((dir == 0 && pos <= top) || (dir == 1 && pos > top)) && *unfinished < 0 && status <= TaskHeader_Active)
	//while(pos >= 0 && *unfinished < 0 && status <= TaskHeader_Active)
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
			red[tid] = *((int*)&g_taskStack.tasks[beg].origSize);
			//owner[tid] = beg;
		}*/

		// Reduce work that does not come from the same task (do not count it more than 1 time)
		//reduceWarpDiv(tid, red, owner, plus);
		//if(__any(status > TaskHeader_Active))
		//	reduceWarp(tid, red, plus);
		//popCount = taskPopCount(status);
		//popCount = taskPopCount(*((int*)&g_taskStack.tasks[beg].origSize));
		//popCount = max((red[0] / NUM_WARPS) + 1, taskPopCount(status));
		int popCount = max((status / NUM_WARPS) + 1, taskPopCount(status));

		if(status > TaskHeader_Active)
		{
			// Choose some with active task
			red[0] = tid;

			if(red[0] == tid)
			{
				// Try acquire the current task - decrease it
				// BUG: Potential bug if task is finished (state set to empty) before other warps notice it
				s_task[threadIdx.y].popSubtask = atomicSub(&g_taskStack.header[beg], popCount); // Try to update and return the current value
				s_task[threadIdx.y].popCount = popCount;
				s_task[threadIdx.y].popTaskIdx = beg;
				//status = TaskHeader_Active;
			}
		}

		counter++;

		//if(__all(status <= TaskHeader_Active))
		//	pos -= WARP_SIZE;

		status = s_task[threadIdx.y].popSubtask;
		//if(pos < 0)
		//{
		//	dir ^= 1; // Flip dir
		//	pos = ACTIVE_MAX;
		//}
	}

	if(status <= TaskHeader_Active)
	{
		int topChunk = taskWarpSubtasks(*stackTop);
		beg = (topChunk - (warpIdx % topChunk)) * WARP_SIZE - tid;
	}
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
		red[tid] = 0;
		if(status > TaskHeader_Active)
		{
			red[tid] = *((int*)&g_taskStack.tasks[beg].origSize);
			//red[tid] = status;
		}

		reduceWarp<int>(tid, red, plus);

		//int popCount = taskPopCount(status);
		//int popCount = taskPopCount(*((int*)&g_taskStack.tasks[beg].origSize));
		int popCount = max((red[tid] / NUM_WARPS) + 1, taskPopCount(status));

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

		red[tid] = 0;
		if(status > TaskHeader_Active)
		{
			red[tid] = 1;
		}
		scanWarp<int>(tid, red, plus);

		int xPos = (warpIdx % red[WARP_SIZE-1]) + 1; // Position of the 1 to take work from
		
		if(status > TaskHeader_Active)
		{
			// Choose some with active task
			//red[0] = tid;

			//if(red[0] == tid)
			if(red[tid] == xPos)
			{
				// Try acquire the current task - decrease it
				// status now holds the information about what tasks this warp has to do but in a reversed order and offseted by 1
				// (status starts at the number of subtasks and ends at 1)
				s_task[threadIdx.y].popSubtask = atomicSub(&g_taskStack.header[beg], popCount); // Try to update and return the current value
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

	if(counter >= WAIT_COUNT || status <= TaskHeader_Active /*|| *unfinished == 0*/)
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
	if(counter >= WAIT_COUNT || status <= TaskHeader_Active /*|| *unfinished == 0*/) // g_taskStack is empty, no more work to do
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

// Update best plane in global memory
// MERGE: Get rid of this function or merge it?
__device__ void taskUpdateBestPlane(int tid, int taskIdx)
{
	ASSERT_DIVERGENCE("taskUpdateBestPlane top", tid);

	float bestCost = s_task[threadIdx.y].bestCost;
	float* g_bestCost = &g_taskStack.tasks[taskIdx].bestCost;
	
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
			if(tid == 11 && atomicCAS(&g_taskStack.tasks[taskIdx].lock, LockType_Free, LockType_Set) == LockType_Free)
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
			g_taskStack.tasks[taskIdx].lock = LockType_Free;
	}

	ASSERT_DIVERGENCE("taskUpdateBestPlane mid", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		if(tid < (sizeof(float4) + sizeof(float) + sizeof(int)) / sizeof(float)) // Copy splitPlane, bestCost and bestOrder
		{
			float* split = (float*)&g_taskStack.tasks[taskIdx].splitPlane;
			volatile float* shared = (volatile float*)&s_task[threadIdx.y].splitPlane;
			split[tid] = shared[tid];
			__threadfence();
		}

		//if(tid == 5)
			g_taskStack.tasks[taskIdx].lock = LockType_Free;
	}

	ASSERT_DIVERGENCE("taskUpdateBestPlane bottom", tid);
}

#ifdef CLIP_INTERSECT

//------------------------------------------------------------------------

// Computes the prefix sum for clipping

__device__ void taskFinishClipPPS(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishClipPPS top", tid);

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished = taskNextPhaseCountPPS(step, rayStart, rayEnd); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step++;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to ClipSORT
		{
			int rayActive = rayEnd - ((int*)c_ns_in.ppsRaysBuf)[rayEnd - 1]; // Seems that must be volatile

			unfinished = 0; // OPTIMIZE: Should not be needed
			s_task[threadIdx.y].type = TaskType_ClipSORT;
			if(rayActive != rayStart && rayActive != rayEnd)
			{
				unfinished += taskWarpSubtasksZero(rayEnd - rayStart);
			}

			if(unfinished == 0) // Nothing to sort -> Move to Intersect
			{
				s_task[threadIdx.y].type = TaskType_Intersect;

				unfinished = taskWarpSubtasks(rayActive - rayStart);
				//s_task[threadIdx.y].rayEnd = rayActive;
			}

			s_task[threadIdx.y].rayActive = rayActive;
			s_task[threadIdx.y].unfinished = unfinished;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
	}

	ASSERT_DIVERGENCE("taskFinishClipPPS bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_ClipPPS);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

__device__ void taskFinishClipSORT(int tid, int taskIdx, unsigned countDown)
{
	ASSERT_DIVERGENCE("taskFinishClipSORT top", tid);

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayActive = s_task[threadIdx.y].rayActive;
	int rayEnd = s_task[threadIdx.y].rayEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		// Move to PPS2
		s_task[threadIdx.y].type = TaskType_Intersect;
		//s_task[threadIdx.y].rayEnd = rayActive;
		s_task[threadIdx.y].unfinished = taskWarpSubtasks(rayActive - rayStart);
		s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
	}

	ASSERT_DIVERGENCE("taskFinishClipSORT bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_ClipSORT);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

#endif

//------------------------------------------------------------------------

// Finishes a split task
__device__ void taskFinishSplit(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSplit top", tid);

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		// Advance the automaton in shared memory
		int rays = taskWarpSubtasks(s_task[threadIdx.y].rayEnd - s_task[threadIdx.y].rayStart);
		int tris = taskWarpSubtasks(s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triStart);
		s_task[threadIdx.y].bestOrder = rays;
		s_task[threadIdx.y].unfinished = rays + tris; // Number of warp sized subtasks
		s_task[threadIdx.y].type = taskChooseScanType1();
		s_task[threadIdx.y].step = 0;
		s_sharedData[threadIdx.y][0] = -1;
	}

	ASSERT_DIVERGENCE("taskFinishSplit mid", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
#if SPLIT_TYPE == 1
		if(s_task[threadIdx.y].lock != LockType_None)
		// Reload splitPlane from gmem, some other warp may have improved the cost
		// OPTIMIZE: Save everything but the split to gmem
		if(tid < (sizeof(float4)) / sizeof(float)) // Copy splitPlane
		{
			float* split = (float*)&g_taskStack.tasks[taskIdx].splitPlane;
			volatile float* shared = (volatile float*)&s_task[threadIdx.y].splitPlane;
			shared[tid] = split[tid];
		}
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
			float sumWeights = c_env.optAxisAlignedWeight + c_env.optTriangleBasedWeight + c_env.optRayBasedWeight;
			// Assign the planes to different methods
			int numAxisAlignedPlanes = c_env.optAxisAlignedWeight/sumWeights*PLANE_COUNT;
			int numTriangleBasedPlanes = c_env.optTriangleBasedWeight/sumWeights*PLANE_COUNT;
			//int numRayBasedPlanes = c_env.optRayBasedWeight/sumWeights*PLANE_COUNT;

			// Recompute the split plane. OPTIMIZE: Save the split plane to global memory?
			float4 plane;
			int rayStart = s_task[threadIdx.y].rayStart;
			int rayEnd = s_task[threadIdx.y].rayEnd;
			int triStart = s_task[threadIdx.y].triStart;
			int triEnd = s_task[threadIdx.y].triEnd;
			volatile CudaAABB &bbox = s_task[threadIdx.y].bbox;
			findPlane(planePos, c_ns_in.rays, c_ns_in.raysIndex, rayStart, rayEnd, c_ns_in.tris, c_ns_in.trisIndex, triStart, triEnd, bbox, numAxisAlignedPlanes, numTriangleBasedPlanes, plane);
			
			SplitData *split = &(g_splitStack[taskIdx].splits[planePos]);

			// Compute the plane cost
			float obj_disbalance = 0.5f * fabs((float)split->tb-split->tf);
			//float cost = split->rb*split->tb + split->rf*split->tf + obj_disbalance;
			float cost;
			if(split->rb + split->rf == 0)
				cost = split->tb + split->tf + obj_disbalance;
			else
				cost = split->rb*split->tb + split->rf*split->tf + obj_disbalance;


#ifdef SPLIT_TEST
			assert(cost > 0.f);
#endif
			//if(cost <= 0.f)
			//	cost = CUDART_INF_F;

			// Reduce the best cost within the warp
			red[tid] = cost;
			reduceWarp(tid, red, min);
			//reduceWarp<float, Min<float>>(tid, red, Min<float>());

			//if(red[tid] != CUDART_INF_F)
			//{
				// Return the best plane for this warp
				if(__ffs(__ballot(red[tid] == cost)) == tid+1) // First thread with such condition, OPTIMIZE: Can be also computed by overwrite and test: better?
				{
					if(split->order < 0)
						plane = -plane;

					s_task[threadIdx.y].splitPlane.x = plane.x;
					s_task[threadIdx.y].splitPlane.y = plane.y;
					s_task[threadIdx.y].splitPlane.z = plane.z;
					s_task[threadIdx.y].splitPlane.w = plane.w;
					s_task[threadIdx.y].bestCost = cost;
					s_task[threadIdx.y].bestOrder = split->order;
				}
			/*}
			else
			{
				volatile float* tPln = ((volatile float*)&s_task[threadIdx.y].splitPlane)+tid;
				volatile float* tMin = ((volatile float*)&bbox.m_mn)+tid;
				volatile float* tMax = ((volatile float*)&bbox.m_mx)+tid;

				if(tid < 3)
				{
					*tPln = *tMax - *tMin;
					float dMax = max3(s_task[threadIdx.y].splitPlane.x, s_task[threadIdx.y].splitPlane.y, s_task[threadIdx.y].splitPlane.z);
					if(__ffs(__ballot(dMax == *tPln)) == tid+1) // First thread with such condition
					{
						s_task[threadIdx.y].splitPlane.w = -(*tMin + *tMax) / 2.0f;
						*tPln = 1;
					}
					else
					{
						*tPln = 0;
					}
				}
			}*/
		}


		// Advance the automaton in shared memory
		int rays = taskWarpSubtasks(s_task[threadIdx.y].rayEnd - s_task[threadIdx.y].rayStart);
		int tris = taskWarpSubtasks(s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triStart);
		s_task[threadIdx.y].bestOrder = rays;
		s_task[threadIdx.y].unfinished = rays + tris; // Number of warp sized subtasks
		s_task[threadIdx.y].type = taskChooseScanType1();
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
#endif

//------------------------------------------------------------------------

// Finishes a sort task
__device__ void taskFinishSort(int tid, int taskIdx, unsigned int index)
{
	// We should update the dependencies before we start adding new tasks because these subtasks may be finished before this is done

#ifndef KEEP_ALL_TASKS
	atomicCAS(&g_taskStack.top, taskIdx, max(taskIdx-1, 0)); // Try decreasing the stack top
#endif

#if PARALLELISM_TEST == 0
	atomicSub(&g_numActive, 1);
#endif

	int numSubtasks = c_dependency[index].lastBlock; // OPTIMIZE: skip queue management if c_dependency[index].count == 1?
#ifdef CUTOFF_DEPTH
	if(s_task[threadIdx.y].depth == c_env.optCutOffDepth)
		numSubtasks = 0;
#endif
	int status = 0;

	int depend1 = s_task[threadIdx.y].depend1;
	int depend2 = s_task[threadIdx.y].depend2;

	// Decrease the waiting counters (increase the number of tasks the task is waiting on) for tasks dependent on this one
	// Update task depend1
	if(depend1 == DependType_Root) // Pointer to unfinished, OPTIMIZE: stacks bottom most entry is sentinel?
	{
		atomicSub(&g_taskStack.unfinished, numSubtasks-1);
	}
	else if(depend1 != DependType_None) // Valid link
	{
		status = atomicSub(&g_taskStack.header[depend1], numSubtasks-1); // -1 is for the task that has just ended
		if(numSubtasks == 0 && status == TaskHeader_Locked-1) // We have unlocked the last dependency
		{
			Task *dependTask = &g_taskStack.tasks[depend1]; // Must be volatile
#if PARALLELISM_TEST >= 0
			int active = atomicAdd(&g_numActive, 1);
			int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
#ifndef CUTOFF_DEPTH
			//if(active > ACTIVE_MAX)
			//	printf("Warp %d too much [%d] subtasks\n", warpIdx, active);
			if(active == 0)
			{
				printf("Warp %d no active tasks before sort opening task with %d subtasks\n", warpIdx, dependTask->unfinished);
				//printf("Distance from stackTop: %d\n", g_taskStack.top - depend1);
				g_taskStack.unfinished = 1;
			}
#endif
#endif
			g_taskStack.header[depend1] = dependTask->unfinished; // Unlock the task

#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
			taskCacheActive(depend1, g_taskStack.active, &g_taskStack.activeTop);
#endif
		}
	}

	// Update task depend1
	if(depend2 == DependType_Root) // Pointer to unfinished, OPTIMIZE: stacks bottom most entry is sentinel?
	{
		atomicSub(&g_taskStack.unfinished, numSubtasks-1);
	}
	else if(depend2 != DependType_None) // Valid link
	{
		status = atomicSub(&g_taskStack.header[depend2], numSubtasks-1); // -1 is for the task that has just ended
		if(numSubtasks == 0 && status == TaskHeader_Locked-1) // We have unlocked the last dependency
		{
			Task *dependTask = &g_taskStack.tasks[depend2]; // Must be volatile
#if PARALLELISM_TEST >= 0
			int active = atomicAdd(&g_numActive, 1);
			int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
#ifndef CUTOFF_DEPTH
			//if(active > ACTIVE_MAX)
			//	printf("Warp %d too much [%d] subtasks\n", warpIdx, active);
			if(active == 0)
			{
				printf("Warp %d no active tasks before sort opening task with %d subtasks\n", warpIdx, dependTask->unfinished);
				//printf("Distance from stackTop: %d\n", g_taskStack.top - depend2);
				g_taskStack.unfinished = 1;
			}
#endif
#endif
			g_taskStack.header[depend2] = dependTask->unfinished; // Unlock the task

#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
			taskCacheActive(depend2, g_taskStack.active, &g_taskStack.activeTop);
#endif
		}
	}

#ifndef KEEP_ALL_TASKS
	g_taskStack.header[taskIdx] = TaskHeader_Empty; // Empty this task

#if ENQUEUE_TYPE == 1
	atomicMin(&g_taskStack.bottom, taskIdx); // Update the stack bottom
#elif ENQUEUE_TYPE == 3
	taskCacheEmpty(taskIdx, g_taskStack.empty, &g_taskStack.emptyTop);
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
		taskUncacheActive(tid, taskIdx, g_taskStack.active, &g_taskStack.activeTop);
#endif
#endif

		s_task[threadIdx.y].lock = LockType_Free;

		// Compute index to the dependency table
		unsigned int index = taskComputeSubtasksIndex();

		ASSERT_DIVERGENCE("taskFinishAABB sort", tid);

		if(tid == 15)
		{
			taskFinishSort(tid, taskIdx, index);
		}

		float triStart = s_task[threadIdx.y].triStart;
		float triLeft = s_task[threadIdx.y].triLeft;
		float triRight = s_task[threadIdx.y].triRight;
		float triEnd = s_task[threadIdx.y].triEnd;

		float rayStart = s_task[threadIdx.y].rayStart;
		float rayLeft = s_task[threadIdx.y].rayLeft;
		float rayRight = s_task[threadIdx.y].rayRight;
		float rayEnd = s_task[threadIdx.y].rayEnd;

		float totalWork = (triEnd - triStart)*(rayEnd - rayStart);
		float newWork = totalWork;
		newWork -= (triEnd - triRight)*(rayLeft - rayStart);
		newWork -= (triLeft - triStart)*(rayEnd - rayRight);
		// How much work is computed now and before subdivision
		float ratioWork = newWork/totalWork;

		if(ratioWork > 0.99f)
		{
			int sfCnt = s_task[threadIdx.y].subFailureCounter+1;
			s_task[threadIdx.y].subFailureCounter = sfCnt;
		}

		ASSERT_DIVERGENCE("taskFinishAABB enqueue", tid);

		// Enqueue the new tasks
#if ENQUEUE_TYPE != 3
		taskEnqueueSubtasks(tid, taskIdx, index);
#else
		taskEnqueueSubtasksCache(tid, taskIdx, index);
#endif

#ifdef DEBUG_INFO
		// Restore ray interval
		int temp = s_task[threadIdx.y].rayEnd;
		s_task[threadIdx.y].rayEnd = s_task[threadIdx.y].rayActive;
		s_task[threadIdx.y].rayActive = temp;

		s_task[threadIdx.y].subtaskIdx = index;
		taskSaveFirstToGMEM(tid, taskIdx, s_task[threadIdx.y]); // Make sure results are visible in global memory
#endif

#if PARALLELISM_TEST == 1
		if(tid == 0)
			atomicSub(&g_numActive, 1);
#endif
}

//------------------------------------------------------------------------

__device__ void taskFinishSortPPS1(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortPPS1 top", tid);

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished = taskNextPhaseCountPPS(step, rayStart, rayEnd); // Returns 0 when no new phase is needed
		s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
		unfinished += taskNextPhaseCountPPS(step, triStart, triEnd); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step++;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to SORT1
		{
			int rayRight = rayEnd - ((int*)c_ns_in.ppsRaysBuf)[rayEnd - 1]; // Seems that must be volatile
			int triRight = triEnd - ((int*)c_ns_in.ppsTrisBuf)[triEnd - 1]; // Seems that must be volatile

			unfinished = 0; // OPTIMIZE: Should not be needed
			s_task[threadIdx.y].type = TaskType_Sort_SORT1;
			if(rayRight != rayStart && rayRight != rayEnd)
			{
				unfinished += taskWarpSubtasksZero(rayEnd - rayRight); // OPTIMIZE: Choose smaller from rayRight - rayStart and rayEnd - rayRight?
			}
			s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
			if(triRight != triStart && triRight != triEnd)
			{
				unfinished += taskWarpSubtasksZero(triEnd - triRight);
			}

			if(unfinished == 0) // Nothing to sort -> Move to PPS2
			{
				s_task[threadIdx.y].type = TaskType_Sort_PPS2;

				unfinished = taskWarpSubtasksZero(rayRight - rayStart);
				s_task[threadIdx.y].rayDivisor = unfinished;
				unfinished += taskWarpSubtasksZero(rayEnd - rayRight);
				s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
				unfinished += taskWarpSubtasksZero(triRight - triStart);
			}

			s_task[threadIdx.y].rayRight = rayRight;
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

__device__ void taskFinishSortPPS1Up(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortPPS1Up top", tid);

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished = taskNextPhaseCountPPSUp(step, rayStart, rayEnd); // Returns 0 when no new phase is needed
		s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
		unfinished += taskNextPhaseCountPPSUp(step, triStart, triEnd); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step += LOG_WARP_SIZE;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to PPS1_Down
		{
			// Set the last element to 0 as required by Harris scan
			((int*)c_ns_in.ppsRaysBuf)[rayEnd - 1] = 0;
			((int*)c_ns_in.ppsTrisBuf)[triEnd - 1] = 0;

			s_task[threadIdx.y].type = TaskType_Sort_PPS1_Down;
			// Make the level multiple of LOG_WARP_SIZE to end at step 0
			int rayLevel = taskTopTreeLevel(rayStart, rayEnd);
			int triLevel = taskTopTreeLevel(triStart, triEnd);
			int level = max(rayLevel, triLevel);
			unfinished = taskNextPhaseCountPPSDown(level+LOG_WARP_SIZE, rayStart, rayEnd); // +LOG_WARP_SIZE because taskNextPhaseCountPPSDown returns result for next iteration
			s_task[threadIdx.y].bestOrder = unfinished;
			unfinished += taskNextPhaseCountPPSDown(level+LOG_WARP_SIZE, triStart, triEnd); // +LOG_WARP_SIZE because taskNextPhaseCountPPSDown returns result for next iteration
			s_task[threadIdx.y].unfinished = unfinished;
			s_task[threadIdx.y].step = level;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
	}

	ASSERT_DIVERGENCE("taskFinishSortPPS1Up bottom", tid);

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

__device__ void taskFinishSortPPS1Down(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortPPS1Down top", tid);

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished = taskNextPhaseCountPPSDown(step, rayStart, rayEnd); // Returns 0 when no new phase is needed
		s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
		unfinished += taskNextPhaseCountPPSDown(step, triStart, triEnd); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step -= LOG_WARP_SIZE;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to SORT1
		{
			int rayRight = rayEnd - ((int*)c_ns_in.ppsRaysBuf)[rayEnd - 1]; // Seems that must be volatile
			int triRight = triEnd - ((int*)c_ns_in.ppsTrisBuf)[triEnd - 1]; // Seems that must be volatile

#ifdef RAYTRI_TEST
			if(rayRight < rayStart || rayRight > rayEnd)
			{
				printf("PPS1 error rayStart %d, rayRight %d, rayEnd %d!\n", rayStart, rayRight, rayEnd);
			}
			if(triRight < triStart || triRight > triEnd)
			{
				printf("PPS1 error triStart %d, triRight %d, triEnd %d!\n", triStart, triRight, triEnd);
			}
#endif

			unfinished = 0; // OPTIMIZE: Should not be needed
			s_task[threadIdx.y].type = TaskType_Sort_SORT1;
			if(rayRight != rayStart && rayRight != rayEnd)
				unfinished += taskWarpSubtasksZero(rayEnd - rayRight); // OPTIMIZE: Choose smaller from rayRight - rayStart and rayEnd - rayRight?
			s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
			if(triRight != triStart && triRight != triEnd)
				unfinished += taskWarpSubtasksZero(triEnd - triRight);

			if(unfinished == 0) // Nothing to sort -> Move to PPS2
			{
				s_task[threadIdx.y].type = TaskType_Sort_PPS2_Up;
				unfinished = taskWarpSubtasksZero(rayRight - rayStart);
				s_task[threadIdx.y].rayDivisor = unfinished;
				unfinished += taskWarpSubtasksZero(rayEnd - rayRight);
				s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
				unfinished += taskWarpSubtasksZero(triRight - triStart);
			}

			s_task[threadIdx.y].rayRight = rayRight;
			s_task[threadIdx.y].triRight = triRight;
			s_task[threadIdx.y].unfinished = unfinished;
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
	}

	ASSERT_DIVERGENCE("taskFinishSortPPS1Down bottom", tid);

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

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayRight = s_task[threadIdx.y].rayRight;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triRight = s_task[threadIdx.y].triRight;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished = 0;

		if(step == 0) // Move to next phase of the current task
		{
			if(rayRight != rayStart && rayRight != rayEnd)
				unfinished = taskWarpSubtasksZero(rayRight - rayStart); // OPTIMIZE: Choose smaller from rayRight - rayStart and rayEnd - rayRight?
			s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
			if(triRight != triStart && triRight != triEnd)
				unfinished += taskWarpSubtasksZero(triRight - triStart);

			s_task[threadIdx.y].unfinished = unfinished;
			step++;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to PPS2
		{
#ifdef RAYTRI_TEST
			if(s_task[threadIdx.y].type != TaskType_RayTriTestSORT1)
			{
				// Move to ray and triangle testing
				unfinished = taskWarpSubtasks(rayEnd - rayStart);
				s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
				unfinished += taskWarpSubtasks(triEnd - triStart);
				s_task[threadIdx.y].unfinished = unfinished;
				s_task[threadIdx.y].type = TaskType_RayTriTestSORT1;
				s_task[threadIdx.y].step = 0;
				s_sharedData[threadIdx.y][0] = -1;
			}
			else
#endif
			{
				unfinished = taskWarpSubtasksZero(rayRight - rayStart);
				s_task[threadIdx.y].rayDivisor = unfinished;
				unfinished += taskWarpSubtasksZero(rayEnd - rayRight);

				s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
				unfinished += taskWarpSubtasksZero(triRight - triStart);

				s_task[threadIdx.y].unfinished = unfinished;
				s_task[threadIdx.y].type = taskChooseScanType2();
				s_task[threadIdx.y].step = 0;
				s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
			}
		}
	}

	ASSERT_DIVERGENCE("taskFinishSortSORT1 bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_SORT1);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

__device__ void taskFinishSortPPS2(int tid, int taskIdx, unsigned countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortPPS2 top", tid);

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished;

		int rayRight = s_task[threadIdx.y].rayRight;
		unfinished = taskNextPhaseCountPPS(step, rayStart, rayRight);
		s_task[threadIdx.y].rayDivisor = unfinished;
		unfinished += taskNextPhaseCountPPS(step, rayRight, rayEnd); // Returns 0 when no new phase is needed

		s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
		int triRight = s_task[threadIdx.y].triRight;
		unfinished += taskNextPhaseCountPPS(step, triStart, triRight); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step++;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to SORT2
		{
			int rayLeft;
			if(rayRight == rayStart)
				rayLeft = rayStart;
			else
				rayLeft = rayRight - ((int*)c_ns_in.ppsRaysBuf)[rayRight - 1];

			int triRight = s_task[threadIdx.y].triRight;
			int triLeft;
			if(triRight == triStart)
				triLeft = triStart;
			else
				triLeft = triRight - ((int*)c_ns_in.ppsTrisBuf)[triRight - 1];

			int removedRays = 0;
			if(rayRight < rayEnd)
				removedRays = ((int*)c_ns_in.ppsRaysBuf)[rayEnd - 1];
			int rayActive = rayEnd - removedRays;
			s_task[threadIdx.y].rayActive = rayActive;
#ifdef RAYTRI_TEST
			if(rayActive < rayRight)
				printf("Too many removed rays!\n");
#endif
			unfinished = 0; // OPTIMIZE: Should not be needed
			s_task[threadIdx.y].type = TaskType_Sort_SORT2;

			if(rayLeft != rayStart && rayLeft != rayRight)
			{
				unfinished += taskWarpSubtasksZero(rayRight - rayLeft);
			}
			s_task[threadIdx.y].rayDivisor = unfinished;
			if(rayActive != rayRight && rayActive != rayEnd)
			{
				unfinished += taskWarpSubtasksZero(rayEnd - rayActive);
			}

			s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
			if(triLeft != triStart && triLeft != triRight)
			{
				unfinished += taskWarpSubtasksZero(triRight - triLeft);
			}

			if(unfinished == 0) // Nothing to sort -> Move to AABB_Min
			{
#ifdef DEBUG_INFO
				s_task[threadIdx.y].rayActive = rayEnd; // Save information about previous end
#endif
				s_task[threadIdx.y].rayEnd = rayActive; // Used for active end position
				unfinished = taskWarpSubtasks(triEnd - triStart);
				s_task[threadIdx.y].type = taskChooseAABBType();
			}

			s_task[threadIdx.y].rayLeft = rayLeft;
			s_task[threadIdx.y].triLeft = triLeft;
			s_task[threadIdx.y].unfinished = unfinished;
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
	}

	ASSERT_DIVERGENCE("taskFinishSortPPS2 bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_PPS2);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

__device__ void taskFinishSortPPS2Up(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortPPS2Up top", tid);

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished;

		int rayRight = s_task[threadIdx.y].rayRight;
		unfinished = taskNextPhaseCountPPSUp(step, rayStart, rayRight);
		s_task[threadIdx.y].rayDivisor = unfinished;
		unfinished += taskNextPhaseCountPPSUp(step, rayRight, rayEnd);

		s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
		int triRight = s_task[threadIdx.y].triRight;
		unfinished += taskNextPhaseCountPPSUp(step, triStart, triRight); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step += LOG_WARP_SIZE;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to PPS2_Down
		{
			// Set the last element to 0 as required by Harris scan
			if(rayRight != rayStart)
				((int*)c_ns_in.ppsRaysBuf)[rayRight - 1] = 0;
			if(rayRight != rayEnd)
				((int*)c_ns_in.ppsRaysBuf)[rayEnd - 1] = 0;
			if(triRight != triStart)
				((int*)c_ns_in.ppsTrisBuf)[triRight - 1] = 0;

			s_task[threadIdx.y].type = TaskType_Sort_PPS2_Down;
			// Make the level multiple of LOG_WARP_SIZE to end at step 0
			int rayLevel = taskTopTreeLevel(rayStart, rayRight);
			int clipLevel = taskTopTreeLevel(rayRight, rayEnd);
			int triLevel = taskTopTreeLevel(triStart, triRight);
			int level = max(max(rayLevel, clipLevel), triLevel);
			// +LOG_WARP_SIZE because taskNextPhaseCountPPSDown returns result for next iteration
			unfinished = taskNextPhaseCountPPSDown(level+LOG_WARP_SIZE, rayStart, rayRight);
			s_task[threadIdx.y].rayDivisor = unfinished;
			unfinished += taskNextPhaseCountPPSDown(level+LOG_WARP_SIZE, rayRight, rayEnd);

			s_task[threadIdx.y].bestOrder = unfinished;
			unfinished += taskNextPhaseCountPPSDown(level+LOG_WARP_SIZE, triStart, triRight);
			s_task[threadIdx.y].unfinished = unfinished;
			s_task[threadIdx.y].step = level;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
	}

	ASSERT_DIVERGENCE("taskFinishSortPPS2Up bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_PPS2_Up);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

__device__ void taskFinishSortPPS2Down(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortPPS2Down top", tid);

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished;

		int rayRight = s_task[threadIdx.y].rayRight;
		unfinished = taskNextPhaseCountPPSDown(step, rayStart, rayRight);
		s_task[threadIdx.y].rayDivisor = unfinished;
		unfinished += taskNextPhaseCountPPSDown(step, rayRight, rayEnd); // Returns 0 when no new phase is needed

		s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
		int triRight = s_task[threadIdx.y].triRight;
		unfinished += taskNextPhaseCountPPSDown(step, triStart, triRight); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step -= LOG_WARP_SIZE;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else // Move to SORT2
		{
			int rayLeft;
			if(rayRight == rayStart)
				rayLeft = rayStart;
			else
				rayLeft = rayRight - ((int*)c_ns_in.ppsRaysBuf)[rayRight - 1];

			int triLeft;
			if(triRight == triStart)
				triLeft = triStart;
			else
				triLeft = triRight - ((int*)c_ns_in.ppsTrisBuf)[triRight - 1];

			int removedRays = 0;
			if(rayRight < rayEnd)
				removedRays = ((int*)c_ns_in.ppsRaysBuf)[rayEnd - 1];
			int rayActive = rayEnd - removedRays;
			s_task[threadIdx.y].rayActive = rayActive;
#ifdef RAYTRI_TEST
			if(rayActive < rayRight)
				printf("Too many removed rays!\n");
#endif
			unfinished = 0; // OPTIMIZE: Should not be needed
			s_task[threadIdx.y].type = TaskType_Sort_SORT2;

			if(rayLeft != rayStart && rayLeft != rayRight)
				unfinished += taskWarpSubtasksZero(rayRight - rayLeft);
			s_task[threadIdx.y].rayDivisor = unfinished;
			if(rayActive != rayRight && rayActive != rayEnd)
				unfinished += taskWarpSubtasksZero(rayEnd - rayActive);

			s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
			if(triLeft != triStart && triLeft != triRight)
				unfinished += taskWarpSubtasksZero(triRight - triLeft);

			if(unfinished == 0) // Nothing to sort -> Move to AABB_Min
			{
#ifdef DEBUG_INFO
				s_task[threadIdx.y].rayActive = rayEnd; // Save information about previous end
#endif
				s_task[threadIdx.y].rayEnd = rayActive; // Used for active end position
				unfinished = taskWarpSubtasks(triEnd - triStart);
				s_task[threadIdx.y].type = taskChooseAABBType();
			}

			s_task[threadIdx.y].rayLeft = rayLeft;
			s_task[threadIdx.y].triLeft = triLeft;
			s_task[threadIdx.y].unfinished = unfinished;
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
	}

	ASSERT_DIVERGENCE("taskFinishSortPPS2Down bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_PPS2_Down);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

__device__ void taskFinishSortSORT2(int tid, int taskIdx, unsigned countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortSORT2 top", tid);

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished = 0;

		if(step == 0) // Move to next phase of the current task
		{
			int rayLeft = s_task[threadIdx.y].rayLeft;
			int rayRight = s_task[threadIdx.y].rayRight;
			if(rayLeft != rayStart && rayLeft != rayRight)
				unfinished += taskWarpSubtasksZero(rayLeft - rayStart);
			s_task[threadIdx.y].rayDivisor = unfinished;

			int rayActive = s_task[threadIdx.y].rayActive;
			if(rayActive != rayRight && rayActive != rayEnd)
				unfinished += taskWarpSubtasksZero(rayActive - rayRight);

			s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
			int triLeft = s_task[threadIdx.y].triLeft;
			int triRight = s_task[threadIdx.y].triRight;
			if(triLeft != triStart && triLeft != triRight)
				unfinished += taskWarpSubtasksZero(triLeft - triStart);

			s_task[threadIdx.y].unfinished = unfinished;
			step++;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else
		{
#ifdef RAYTRI_TEST
			if(s_task[threadIdx.y].type != TaskType_RayTriTestSORT2)
			{
				// Move to ray and triangle testing
				unfinished = taskWarpSubtasks(rayEnd - rayStart);
				s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
				unfinished += taskWarpSubtasks(triEnd - triStart);
				s_task[threadIdx.y].unfinished = unfinished;
				s_task[threadIdx.y].type = TaskType_RayTriTestSORT2;
				s_task[threadIdx.y].step = 0;
				s_sharedData[threadIdx.y][0] = -1;
			}
			else
#endif
			{

				// Change ray end
				s_task[threadIdx.y].rayEnd = s_task[threadIdx.y].rayActive; // Used for active end position

#ifdef DEBUG_INFO
				s_task[threadIdx.y].rayActive = rayEnd; // Save information about previous end
#endif
				s_task[threadIdx.y].unfinished = taskWarpSubtasks(triEnd - triStart);
				s_task[threadIdx.y].type = taskChooseAABBType();
				s_task[threadIdx.y].step = 0;
				s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish				
			}
		}
	}

	ASSERT_DIVERGENCE("taskFinishSortSORT2 bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_SORT2);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

__device__ void taskFinishPartition1(int tid, int taskIdx, unsigned countDown)
{
	ASSERT_DIVERGENCE("taskFinishPartition1 top", tid);

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		// Load correct split positions from global memory
		int rayLeft = *((int*)&g_taskStack.tasks[taskIdx].rayLeft);
		int rayRight = *((int*)&g_taskStack.tasks[taskIdx].rayRight);
		int triLeft = *((int*)&g_taskStack.tasks[taskIdx].triLeft);
		int triRight = *((int*)&g_taskStack.tasks[taskIdx].triRight);

		s_task[threadIdx.y].rayRight = rayRight;
		s_task[threadIdx.y].triRight = triRight;

		// Save starting split positions to global memory
		s_task[threadIdx.y].rayLeft = rayLeft;
		s_task[threadIdx.y].rayActive = rayRight;
		s_task[threadIdx.y].triLeft = triLeft;

#ifndef DEBUG_INFO
		// Make sure the correct values are in the 
		//g_taskStack.tasks[taskIdx].rayActive = s_task[threadIdx.y].rayActive;
		g_taskStack.tasks[taskIdx].bboxLeft.m_mn.x = __int_as_float(rayStart); // Use data item unused in this phase for the counter
		g_taskStack.tasks[taskIdx].bboxRight.m_mn.x = __int_as_float(rayEnd); // Use data item unused in this phase for the counter
		g_taskStack.tasks[taskIdx].bboxMiddle.m_mn.x = __int_as_float(triStart); // Use data item unused in this phase for the counter
#else
		s_task[threadIdx.y].bboxLeft.m_mn.x = __int_as_float(rayStart); // Use data item unused in this phase for the counter
		s_task[threadIdx.y].bboxRight.m_mn.x = __int_as_float(rayEnd); // Use data item unused in this phase for the counter
		s_task[threadIdx.y].bboxMiddle.m_mn.x = __int_as_float(triStart); // Use data item unused in this phase for the counter
#endif

#ifdef RAYTRI_TEST
		if(tid == 0 && rayLeft != rayRight)
			printf("Incorrect partition on ray interval %d - %d (%d x %d)!\n", rayStart, rayEnd, rayLeft, rayRight);

		if(tid == 0 && triLeft != triRight)
			printf("Incorrect partition on tri interval %d - %d (%d x %d)!\n", triStart, triEnd, triLeft, triRight);

		if(s_task[threadIdx.y].type != TaskType_RayTriTestSORT1)
		{
			// Move to ray and triangle testing
			unfinished = taskWarpSubtasks(rayEnd - rayStart);
			s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
			unfinished += taskWarpSubtasks(triEnd - triStart);
			s_task[threadIdx.y].unfinished = unfinished;
			s_task[threadIdx.y].type = TaskType_RayTriTestSORT1;
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1;
		}
		else
#endif
		{
			int unfinished = taskWarpSubtasksZero(rayRight - rayStart);
			s_task[threadIdx.y].rayDivisor = unfinished;
			unfinished += taskWarpSubtasksZero(rayEnd - rayRight);

			s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
			unfinished += taskWarpSubtasksZero(triRight - triStart);
			s_task[threadIdx.y].step = unfinished; // Mark split between unsorted tris and sorted tris
			unfinished += taskWarpSubtasksZero(triEnd - triRight);

			s_task[threadIdx.y].unfinished = unfinished;
			s_task[threadIdx.y].type = taskChooseScanType2();
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
	}

	ASSERT_DIVERGENCE("taskFinishPartition1 bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_SORT1);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

__device__ void taskFinishPartition2(int tid, int taskIdx, unsigned countDown)
{
	ASSERT_DIVERGENCE("taskFinishPartition2 top", tid);

	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		// Load correct split positions from global memory
		int rayMiddle =g_taskStack.tasks[taskIdx].bboxLeft.m_mn.x; // Use data item unused in this phase for the counter
		int rayLeft = g_taskStack.tasks[taskIdx].rayLeft;
		int rayActive = g_taskStack.tasks[taskIdx].rayActive;
		int rayClipped = *(int*)&g_taskStack.tasks[taskIdx].bboxRight.m_mn.x; // Use data item unused in this phase for the counter
		int triMiddle = *(int*)&g_taskStack.tasks[taskIdx].bboxMiddle.m_mn.x; // Use data item unused in this phase for the counter
		int triLeft = g_taskStack.tasks[taskIdx].triLeft;

		s_task[threadIdx.y].rayLeft = rayLeft;
		s_task[threadIdx.y].rayActive = rayActive;
		s_task[threadIdx.y].triLeft = triLeft;

#ifdef RAYTRI_TEST
		if(tid == 0 && rayLeft != rayMiddle)
			printf("Incorrect partition on left ray interval %d - %d (%d x %d, %d)!\n", rayStart, rayEnd, rayMiddle, rayLeft, s_task[threadIdx.y].rayRight);

		if(tid == 0 && rayActive != rayClipped)
			printf("Incorrect partition on active ray interval %d - %d (%d x %d, %d)!\n", rayStart, rayEnd, rayActive, rayClipped, s_task[threadIdx.y].rayRight);

		if(tid == 0 && triLeft != triMiddle)
			printf("Incorrect partition on left tri interval %d - %d (%d x %d, %d)!\n", triStart, triEnd, triMiddle, triLeft, s_task[threadIdx.y].triRight);

		if(s_task[threadIdx.y].type != TaskType_RayTriTestSORT2)
		{
			// Move to ray and triangle testing
			unfinished = taskWarpSubtasks(rayEnd - rayStart);
			s_task[threadIdx.y].bestOrder = unfinished; // Mark split between rays and tris
			unfinished += taskWarpSubtasks(triEnd - triStart);
			s_task[threadIdx.y].unfinished = unfinished;
			s_task[threadIdx.y].type = TaskType_RayTriTestSORT2;
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1;
		}
		else
#endif
		{
			// Change ray end
			s_task[threadIdx.y].rayEnd = s_task[threadIdx.y].rayActive; // Used for active end position

#ifdef DEBUG_INFO
			s_task[threadIdx.y].rayActive = rayEnd; // Save information about previous end
#endif
			s_task[threadIdx.y].unfinished = taskWarpSubtasks(triEnd - triStart);
			s_task[threadIdx.y].type = taskChooseAABBType();
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish	
		}
	}

	ASSERT_DIVERGENCE("taskFinishPartition2 bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_SORT2);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

#if AABB_TYPE < 3

__device__ void taskFinishAABB(int tid, int taskIdx, unsigned countDown)
{
	ASSERT_DIVERGENCE("taskFinishAABB top", tid);

	int start = s_task[threadIdx.y].triStart;
	int end = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		// Returns 0 when no new phase is needed
#if AABB_TYPE == 0
		int unfinished = taskNextPhaseCountReduce(step, start, end);
#elif AABB_TYPE == 1 || AABB_TYPE == 2
		int unfinished = taskNextPhaseCountReduceBlock(step, start, end);
#endif

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
#if AABB_TYPE == 0
			step++;
#elif AABB_TYPE == 1
			step += LOG_WARP_SIZE;
#elif AABB_TYPE == 2
			step += 6;
#endif
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else
		{
			s_sharedData[threadIdx.y][0] = -2;
		}
	}

	ASSERT_DIVERGENCE("taskFinishAABB mid2", tid);

	if(s_sharedData[threadIdx.y][0] == -2) // bbox data need to be modified
	{
		// Data arrays
		float* x = (float*)c_ns_in.ppsTrisBuf;
		float* y = (float*)c_ns_in.ppsTrisIndex;
		float* z = (float*)c_ns_in.sortTris;

		int left = s_task[threadIdx.y].triLeft;
		int right = s_task[threadIdx.y].triRight;

		if(s_task[threadIdx.y].type == TaskType_AABB_Min) // Move to Max
		{
			// Save CudaAABB minimum data from gmem to task
			if(start < left)
			{
				s_task[threadIdx.y].bboxLeft.m_mn.x   = x[start];
				s_task[threadIdx.y].bboxLeft.m_mn.y   = y[start];
				s_task[threadIdx.y].bboxLeft.m_mn.z   = z[start];
			}

			if(left < right)
			{
				s_task[threadIdx.y].bboxMiddle.m_mn.x = x[left];
				s_task[threadIdx.y].bboxMiddle.m_mn.y = y[left];
				s_task[threadIdx.y].bboxMiddle.m_mn.z = z[left];
			}

			if(right < end)
			{
				s_task[threadIdx.y].bboxRight.m_mn.x  = x[right];
				s_task[threadIdx.y].bboxRight.m_mn.y  = y[right];
				s_task[threadIdx.y].bboxRight.m_mn.z  = z[right];
			}

			// Prepare the Max task
			int threads = end - start;
			s_task[threadIdx.y].unfinished = taskWarpSubtasks(threads);
			s_task[threadIdx.y].type = TaskType_AABB_Max;
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish

#ifndef DEBUG_INFO
			taskSaveSecondToGMEM(tid, taskIdx, s_task[threadIdx.y]); // Save bbox data
			// No need for __threadfence() here as it will be called later anyway and the data will not be read for a while
#endif
		}
		else
		{
#ifndef DEBUG_INFO
			taskLoadSecondFromGMEM(tid, taskIdx, s_task[threadIdx.y]); // Load bbox data
#endif

			// Save CudaAABB maximum data from gmem to task
			if(start < left)
			{
				s_task[threadIdx.y].bboxLeft.m_mx.x   = x[start];
				s_task[threadIdx.y].bboxLeft.m_mx.y   = y[start];
				s_task[threadIdx.y].bboxLeft.m_mx.z   = z[start];

#ifdef BBOX_TEST
				// Test that child bounding boxes are inside
				if(tid == 13)
				{
					if(s_task[threadIdx.y].bboxLeft.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x || s_task[threadIdx.y].bboxLeft.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y || s_task[threadIdx.y].bboxLeft.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z
						|| s_task[threadIdx.y].bboxLeft.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x || s_task[threadIdx.y].bboxLeft.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y || s_task[threadIdx.y].bboxLeft.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
					{
						printf("Left child outside! (%f, %f, %f) - (%f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)\n",
							s_task[threadIdx.y].bboxLeft.m_mn.x, s_task[threadIdx.y].bboxLeft.m_mn.y, s_task[threadIdx.y].bboxLeft.m_mn.z,
							s_task[threadIdx.y].bboxLeft.m_mx.x, s_task[threadIdx.y].bboxLeft.m_mx.y, s_task[threadIdx.y].bboxLeft.m_mx.z,
							s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
							s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
					}
				}
#endif
			}

			if(left < right)
			{
				s_task[threadIdx.y].bboxMiddle.m_mx.x = x[left];
				s_task[threadIdx.y].bboxMiddle.m_mx.y = y[left];
				s_task[threadIdx.y].bboxMiddle.m_mx.z = z[left];

#ifdef BBOX_TEST
				// Test that child bounding boxes are inside
				if(tid == 13)
				{
					if(s_task[threadIdx.y].bboxMiddle.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x || s_task[threadIdx.y].bboxMiddle.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y || s_task[threadIdx.y].bboxMiddle.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z
						|| s_task[threadIdx.y].bboxMiddle.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x || s_task[threadIdx.y].bboxMiddle.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y || s_task[threadIdx.y].bboxMiddle.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
					{
						printf("Middle child outside (%f, %f, %f) - (%f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)!\n",
							s_task[threadIdx.y].bboxMiddle.m_mn.x, s_task[threadIdx.y].bboxMiddle.m_mn.y, s_task[threadIdx.y].bboxMiddle.m_mn.z,
							s_task[threadIdx.y].bboxMiddle.m_mx.x, s_task[threadIdx.y].bboxMiddle.m_mx.y, s_task[threadIdx.y].bboxMiddle.m_mx.z,
							s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
							s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
					}
				}
#endif
			}

			if(right < end)
			{
				s_task[threadIdx.y].bboxRight.m_mx.x  = x[right];
				s_task[threadIdx.y].bboxRight.m_mx.y  = y[right];
				s_task[threadIdx.y].bboxRight.m_mx.z  = z[right];

#ifdef BBOX_TEST
				// Test that child bounding boxes are inside
				if(tid == 13)
				{
					if(s_task[threadIdx.y].bboxRight.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x || s_task[threadIdx.y].bboxRight.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y || s_task[threadIdx.y].bboxRight.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z
						|| s_task[threadIdx.y].bboxRight.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x || s_task[threadIdx.y].bboxRight.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y || s_task[threadIdx.y].bboxRight.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
					{
						printf("Right child outside (%f, %f, %f) - (%f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)!\n",
							s_task[threadIdx.y].bboxRight.m_mn.x, s_task[threadIdx.y].bboxRight.m_mn.y, s_task[threadIdx.y].bboxRight.m_mn.z,
							s_task[threadIdx.y].bboxRight.m_mx.x, s_task[threadIdx.y].bboxRight.m_mx.y, s_task[threadIdx.y].bboxRight.m_mx.z,
							s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
							s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
					}
				}
#endif
				}

			// Bounding box computing is finished, finish the whole sort
			s_sharedData[threadIdx.y][0] = -100;
		}
	}

	ASSERT_DIVERGENCE("taskFinishAABB mid3", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_AABB_Min);
	}
	else if(s_sharedData[threadIdx.y][0] == -100) // Finish the whole sort
	{
		taskFinishTask(tid, taskIdx);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}

	ASSERT_DIVERGENCE("taskFinishAABB bottom", tid);
}

#elif AABB_TYPE == 3

__device__ void taskFinishAABB(int tid, int taskIdx, unsigned countDown)
{
	ASSERT_DIVERGENCE("taskFinishAABB top", tid);

	int start = s_task[threadIdx.y].triStart;
	int end = s_task[threadIdx.y].triEnd;

	if(taskCheckFinished(tid, taskIdx, countDown)) // We have finished the task and are responsible for cleaning up
	{
		int step = s_task[threadIdx.y].step;
		int unfinished = taskNextPhaseCountReduceBlock(step, start, end); // Returns 0 when no new phase is needed

		if(unfinished != 0) // Move to next phase of the current task
		{
			s_task[threadIdx.y].unfinished = unfinished;
			step += LOG_WARP_SIZE;
			s_task[threadIdx.y].step = step; // Increase the step
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
		}
		else
		{
			s_sharedData[threadIdx.y][0] = -100;
		}
	}

	ASSERT_DIVERGENCE("taskFinishAABB mid2", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_AABB);
	}
	else if(s_sharedData[threadIdx.y][0] == -100) // Finish the whole sort
	{
#ifdef BBOX_TEST
		int left = s_task[threadIdx.y].triLeft;
		int right = s_task[threadIdx.y].triRight;
		// Test that child bounding boxes are inside
		if(tid == 13)
		{
			if(start < left)
			{
				if(s_task[threadIdx.y].bboxLeft.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x || s_task[threadIdx.y].bboxLeft.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y || s_task[threadIdx.y].bboxLeft.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z
					|| s_task[threadIdx.y].bboxLeft.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x || s_task[threadIdx.y].bboxLeft.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y || s_task[threadIdx.y].bboxLeft.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
				{
					printf("Left child outside! (%f, %f, %f) - (%f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)\n",
						s_task[threadIdx.y].bboxLeft.m_mn.x, s_task[threadIdx.y].bboxLeft.m_mn.y, s_task[threadIdx.y].bboxLeft.m_mn.z,
						s_task[threadIdx.y].bboxLeft.m_mx.x, s_task[threadIdx.y].bboxLeft.m_mx.y, s_task[threadIdx.y].bboxLeft.m_mx.z,
						s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
						s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
					g_taskStack.unfinished = 1;
				}
			}

			if(left < right)
			{
				if(s_task[threadIdx.y].bboxMiddle.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x || s_task[threadIdx.y].bboxMiddle.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y || s_task[threadIdx.y].bboxMiddle.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z
					|| s_task[threadIdx.y].bboxMiddle.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x || s_task[threadIdx.y].bboxMiddle.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y || s_task[threadIdx.y].bboxMiddle.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
				{
					printf("Middle child outside (%f, %f, %f) - (%f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)!\n",
						s_task[threadIdx.y].bboxMiddle.m_mn.x, s_task[threadIdx.y].bboxMiddle.m_mn.y, s_task[threadIdx.y].bboxMiddle.m_mn.z,
						s_task[threadIdx.y].bboxMiddle.m_mx.x, s_task[threadIdx.y].bboxMiddle.m_mx.y, s_task[threadIdx.y].bboxMiddle.m_mx.z,
						s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
						s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
					g_taskStack.unfinished = 1;
				}
			}
			
			if(right < end)
			{
				if(s_task[threadIdx.y].bboxRight.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x || s_task[threadIdx.y].bboxRight.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y || s_task[threadIdx.y].bboxRight.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z
					|| s_task[threadIdx.y].bboxRight.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x || s_task[threadIdx.y].bboxRight.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y || s_task[threadIdx.y].bboxRight.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
				{
					printf("Right child outside (%f, %f, %f) - (%f, %f, %f) not in (%f, %f, %f) - (%f, %f, %f)!\n",
						s_task[threadIdx.y].bboxRight.m_mn.x, s_task[threadIdx.y].bboxRight.m_mn.y, s_task[threadIdx.y].bboxRight.m_mn.z,
						s_task[threadIdx.y].bboxRight.m_mx.x, s_task[threadIdx.y].bboxRight.m_mx.y, s_task[threadIdx.y].bboxRight.m_mx.z,
						s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
						s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
					g_taskStack.unfinished = 1;
				}
			}
		}
#endif

		taskFinishTask(tid, taskIdx);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}

	ASSERT_DIVERGENCE("taskFinishAABB bottom", tid);
}

#endif

//------------------------------------------------------------------------

// Finishes an intersection task
__device__ void taskFinishIntersect(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishIntersect top", tid);

	s_sharedData[threadIdx.y][0] = 0;

	if(taskCheckFinished(tid, taskIdx, countDown) && tid == 0) // We have finished the task and are responsible for cleaning up
	{
#ifndef KEEP_ALL_TASKS
		atomicCAS(&g_taskStack.top, taskIdx, max(taskIdx-1, 0)); // Try decreasing the stack top
#endif

#if PARALLELISM_TEST == 0
		atomicSub(&g_numActive, 1);
#endif

		int status = 0;
		int depend1 = s_task[threadIdx.y].depend1;
		int depend2 = s_task[threadIdx.y].depend2;

		// Unlock task depend1
		if(depend1 == DependType_Root) // Pointer to unfinished, OPTIMIZE: stacks bottom most entry is sentinel?
		{
			atomicAdd(&g_taskStack.unfinished, 1); // Decrease the number of dependencies by increasing the negative number
		}
		else if(depend1 != DependType_None) // Valid link
		{
			status = atomicAdd(&g_taskStack.header[depend1], 1); // Decrease the number of dependencies by increasing the negative number
			if(status == TaskHeader_Locked-1) // We have unlocked the last dependency
			{
				Task *dependTask = &g_taskStack.tasks[depend1]; // Must be volatile
#if PARALLELISM_TEST >= 0
				int active = atomicAdd(&g_numActive, 1);
				int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
#ifndef CUTOFF_DEPTH
				//if(active > ACTIVE_MAX)
				//	printf("Warp %d too much [%d] subtasks\n", warpIdx, active);
				if(active == 0)
				{
					printf("Warp %d no active tasks before intersect opening task with %d subtasks\n", warpIdx, dependTask->unfinished);
					//printf("Distance from stackTop: %d\n", g_taskStack.top - depend1);
					g_taskStack.unfinished = 1;
				}
#endif
#endif
				g_taskStack.header[depend1] = dependTask->unfinished; // Unlock the task

#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
				taskCacheActive(depend1, g_taskStack.active, &g_taskStack.activeTop);
#endif
			}
		}

		// Unlock task depend2
		if(depend2 == DependType_Root) // Pointer to unfinished, OPTIMIZE: stacks bottom most entry is sentinel?
		{
			atomicAdd(&g_taskStack.unfinished, 1); // Decrease the number of dependencies by increasing the negative number
		}
		else if(depend2 != DependType_None) // Valid link
		{
			status = atomicAdd(&g_taskStack.header[depend2], 1); // Decrease the number of dependencies by increasing the negative number
			if(status == TaskHeader_Locked-1) // We have unlocked the last dependency
			{
				Task *dependTask = &g_taskStack.tasks[depend2]; // Must be volatile
#if PARALLELISM_TEST >= 0
				int active = atomicAdd(&g_numActive, 1);
				int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
#ifndef CUTOFF_DEPTH
				//if(active > ACTIVE_MAX)
				//	printf("Warp %d too much [%d] subtasks\n", warpIdx, active);
				if(active == 0)
				{
					printf("Warp %d no active tasks before intersect opening task with %d subtasks\n", warpIdx, dependTask->unfinished);
					//printf("Distance from stackTop: %d\n", g_taskStack.top - depend2);
					g_taskStack.unfinished = 1;
				}
#endif
#endif
				g_taskStack.header[depend2] = dependTask->unfinished; // Unlock the task

#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
				taskCacheActive(depend2, g_taskStack.active, &g_taskStack.activeTop);
#endif
			}
		}

#if PARALLELISM_TEST == 1
		atomicSub(&g_numActive, 1);
#endif

		// We must move the top first and empty this then
#ifndef KEEP_ALL_TASKS
		g_taskStack.header[taskIdx] = TaskHeader_Empty; // Unlock this task

#if ENQUEUE_TYPE == 1
		atomicMin(&g_taskStack.bottom, taskIdx); // Update the stack bottom
#elif ENQUEUE_TYPE == 3
		taskCacheEmpty(taskIdx, g_taskStack.empty, &g_taskStack.emptyTop);
#endif

#ifdef CLIP_INTERSECT
		s_sharedData[threadIdx.y][0] = -1;
#endif
#endif
	}

	s_task[threadIdx.y].lock = LockType_Free;

	if(s_sharedData[threadIdx.y][0] == -1)
	{
#ifndef KEEP_ALL_TASKS
#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
		taskUncacheActive(tid, taskIdx, g_taskStack.active, &g_taskStack.activeTop);
#endif
#endif

#if defined(DEBUG_INFO) && defined(CLIP_INTERSECT)
		taskSaveFirstToGMEM(tid, taskIdx, s_task[threadIdx.y]); // Make sure results are visible in global memory
#endif
	}
}

//------------------------------------------------------------------------

// Compute number of triangles on both sides of the plane
__device__ void trianglesPlanePosition(int triStart, int triEnd, const float4& plane, int& tb, int& tf)
{
	ASSERT_DIVERGENCE("trianglesPlanePosition", threadIdx.x);

	int tris = triEnd - triStart;
	int desiredSamples = getNumberOfSamples(tris);

	tb = 0;
	tf = 0;

	int step = tris / desiredSamples;
	if(step < 1)
		step = 1;

	for(int triPos = triStart; triPos < triEnd; triPos += step)
	{
		int triIdx = ((int*)c_ns_in.trisIndex)[triPos]*3;

		// Fetch triangle
		float3 v0, v1, v2;
		taskFetchTri(c_ns_in.tris, triIdx, v0, v1, v2);

		int pos = getPlanePosition(plane, v0, v1, v2);
		
		if (pos <= 0)
			tb++;
		if (pos >= 0)
			tf++;
	}
}

//------------------------------------------------------------------------

// Compute number of rays on both sides of the plane
__device__ int raysPlanePosition(int rayStart, int rayEnd, const float4& plane, const volatile CudaAABB& bbox, int& rb, int& rf)
{
	ASSERT_DIVERGENCE("raysPlanePosition", threadIdx.x);

	int rays = rayEnd - rayStart;
	int desiredSamples = getNumberOfSamples(rays);

	rb = 0;
	rf = 0;

	int step = rays / desiredSamples;
	if(step < 1)
		step = 1;

	int orderCounter = 0;

	for(int rayPos = rayStart; rayPos < rayEnd; rayPos += step)
	{
		int rayIdx = ((int*)c_ns_in.raysIndex)[rayPos];

		// Fetch ray
		float3 orig, dir;
		float tmin, tmax;
		taskFetchRayVolatile(c_ns_in.rays, rayIdx, orig, dir, tmin, tmax);

		// Update rays tmin and tmax // OPTIMIZE: Do only once per task?
		float3 idir;
		float ooeps = exp2f(-80.0f); // Avoid div by zero.
		idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
		idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
		idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));
		float3 ood = orig * idir;

		float lox = bbox.m_mn.x * idir.x - ood.x;
		float hix = bbox.m_mx.x * idir.x - ood.x;
		float loy = bbox.m_mn.y * idir.y - ood.y;
		float hiy = bbox.m_mx.y * idir.y - ood.y;
		float loz = bbox.m_mn.z * idir.z - ood.z;
		float hiz = bbox.m_mx.z * idir.z - ood.z;
		tmin = max4(fminf(lox, hix), fminf(loy, hiy), fminf(loz, hiz), tmin);
		tmax = min4(fmaxf(lox, hix), fmaxf(loy, hiy), fmaxf(loz, hiz), tmax);

		if(tmin <= tmax)
		{
			int localOrder;
			int pos = getPlanePosition(plane, orig, dir, tmin, tmax, localOrder);
		
			if(!c_ns_in.anyHit) // Shadow rays
				orderCounter += localOrder;
			if(pos <= 0)
				rb++;
			if(pos >= 0)
				rf++;
		}
	}

	return orderCounter;
}

//------------------------------------------------------------------------

// Compute cost of several splitting strategies and choose the best one
__device__ void splitCost(int tid, int subtask, int rayStart, int rayEnd, int triStart, int triEnd, const volatile CudaAABB& bbox,
	volatile float4& bestPlane, volatile float& bestCost, volatile int& bestOrder)
{
	ASSERT_DIVERGENCE("splitCost top", tid);

	// Each thread computes its plane
	int planePos = subtask*WARP_SIZE + tid;

	// Number of evaluated candidates per plane
#if SPLIT_TYPE == 1
	int rays = rayEnd - rayStart;
	int tris = triEnd - triStart;

	int evaluatedCandidates = getNumberOfSamples(rays) + getNumberOfSamples(tris);
	int numPlanes = taskWarpSubtasks(c_env.optPlaneSelectionOverhead * (rays + tris)/evaluatedCandidates)*WARP_SIZE;
#else
	int numPlanes = PLANE_COUNT;
#endif

	float sumWeights = c_env.optAxisAlignedWeight + c_env.optTriangleBasedWeight + c_env.optRayBasedWeight;
	// Assign the planes to different methods
	int numAxisAlignedPlanes = c_env.optAxisAlignedWeight/sumWeights*numPlanes;
	int numTriangleBasedPlanes = c_env.optTriangleBasedWeight/sumWeights*numPlanes;
	//int numRayBasedPlanes = c_env.optRayBasedWeight/sumWeights*numPlanes;

	float4 plane;
	findPlane(planePos, c_ns_in.rays, c_ns_in.raysIndex, rayStart, rayEnd, c_ns_in.tris, c_ns_in.trisIndex, triStart, triEnd, bbox, numAxisAlignedPlanes, numTriangleBasedPlanes, plane);

	// Count the number of rays and triangles on both sides of the plane
	int tb, tf;
	int rb, rf;
	trianglesPlanePosition(triStart, triEnd, plane, tb, tf);
	int order = raysPlanePosition(rayStart, rayEnd, plane, bbox, rb, rf);

	// Compute the plane cost
	float obj_disbalance = 0.5f * fabs((float)tb-tf);
	//float cost = rb*tb + rf*tf + obj_disbalance;
	float cost;
	if(rb + rf == 0)
		cost = tb + tf + obj_disbalance;
	else
		cost = rb*tb + rf*tf + obj_disbalance;

	// Reduce the best cost within the warp
	volatile float* red = (volatile float*)&s_owner[threadIdx.y][0];
	red[tid] = cost;
	reduceWarp(tid, red, min);
	//reduceWarp<float, Min<float>>(tid, red, Min<float>());
	
	// Return the best plane for this warp
	if(__ffs(__ballot(red[tid] == cost)) == tid+1) // First thread with such condition, OPTIMIZE: Can be also computed by overwrite and test: better?
	{
		if(order < 0)
			plane = -plane;

		bestPlane.x = plane.x;
		bestPlane.y = plane.y;
		bestPlane.z = plane.z;
		bestPlane.w = plane.w;
		bestCost = cost;
		bestOrder = order;
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
__device__ void splitCostParallel(int tid, int subtask, int taskIdx, int rayStart, int rayEnd, int triStart, int triEnd, const volatile CudaAABB& bbox)
{
	ASSERT_DIVERGENCE("splitCostParallel top", tid);

	// Each thread computes its ray-plane or tri-plane task
	int warpTask = subtask*WARP_SIZE;
	int task = warpTask + tid;

	int rays = rayEnd - rayStart;
	int tris = triEnd - triStart;

	// Number of evaluated candidates per plane
	int rayTasks = taskWarpSubtasks(getNumberOfSamples(rays));
	int evaluatedRays = min(rayTasks*WARP_SIZE, rays); // Choose either WARP_SIZE multiple of sampled rays or all rays
	int evaluatedTris = min(taskWarpSubtasks(getNumberOfSamples(tris))*WARP_SIZE, tris); // Choose either WARP_SIZE multiple of sampled tris or all tris

	float sumWeights = c_env.optAxisAlignedWeight + c_env.optTriangleBasedWeight + c_env.optRayBasedWeight;
	// Assign the planes to different methods
	int numAxisAlignedPlanes = c_env.optAxisAlignedWeight/sumWeights*PLANE_COUNT;
	int numTriangleBasedPlanes = c_env.optTriangleBasedWeight/sumWeights*PLANE_COUNT;
	//int numRayBasedPlanes = c_env.optRayBasedWeight/sumWeights*PLANE_COUNT;
	
	float4 plane;
	volatile int* red = (volatile int*)&s_sharedData[threadIdx.y][0];
	red[tid] = 0;

	if(task < rayTasks*WARP_SIZE*PLANE_COUNT) // This warp is doing rays vs plane tests
	{
		int planePos = subtask % PLANE_COUNT; // Succesive subtasks do far away planes -> hopefully less lock contention in atomic operations
		findPlane(planePos, c_ns_in.rays, c_ns_in.raysIndex, rayStart, rayEnd, c_ns_in.tris, c_ns_in.trisIndex, triStart, triEnd, bbox, numAxisAlignedPlanes, numTriangleBasedPlanes, plane);

		// Count the number of rays on both sides of the plane
		int rb, rf;
		rb = 0;
		rf = 0;

		task = (subtask/PLANE_COUNT)*WARP_SIZE + tid;
		//int step = rays / evaluatedRays;
		//int rayPos = rayStart + task*step;
		float step = (float)rays / (float)evaluatedRays;
		int rayPos = rayStart + (int)((float)task*step);

		if(rayPos < rayEnd)
		{
			int rayIdx = ((int*)c_ns_in.raysIndex)[rayPos];

			// Fetch ray
			float3 orig, dir;
			float tmin, tmax;
			taskFetchRayVolatile(c_ns_in.rays, rayIdx, orig, dir, tmin, tmax);

			// Update rays tmin and tmax // OPTIMIZE: Do only once per task?
			float3 idir;
			float ooeps = exp2f(-80.0f); // Avoid div by zero.
			idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
			idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
			idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));
			float3 ood = orig * idir;

			float lox = bbox.m_mn.x * idir.x - ood.x;
			float hix = bbox.m_mx.x * idir.x - ood.x;
			float loy = bbox.m_mn.y * idir.y - ood.y;
			float hiy = bbox.m_mx.y * idir.y - ood.y;
			float loz = bbox.m_mn.z * idir.z - ood.z;
			float hiz = bbox.m_mx.z * idir.z - ood.z;
			tmin = max4(fminf(lox, hix), fminf(loy, hiy), fminf(loz, hiz), tmin);
			tmax = min4(fmaxf(lox, hix), fmaxf(loy, hiy), fmaxf(loz, hiz), tmax);

			int localOrder = 0;
			if(tmin <= tmax)
			{
				int pos = getPlanePosition(plane, orig, dir, tmin, tmax, localOrder);

				if (pos <= 0)
					rb++;
				if (pos >= 0)
					rf++;
			}

			// Update the split info
			SplitData *split = &(g_splitStack[taskIdx].splits[planePos]);

			// Reduce the numbers within the warp
			red[tid] = rf;
			reduceWarp(tid, red, plus);
			if(tid == 0 && red[tid] != 0)
				atomicAdd(&split->rf, red[tid]);

			red[tid] = rb;
			reduceWarp(tid, red, plus);
			if(tid == 0 && red[tid] != 0)
				atomicAdd(&split->rb, red[tid]);

			if(!c_ns_in.anyHit) // Shadow rays
			{
				red[tid] = localOrder;
				reduceWarp(tid, red, plus);
				if(tid == 0 && red[tid] != 0)
					atomicAdd(&split->order, red[tid]);
			}
		}
	}
	else// if(task >= rayTasks*WARP_SIZE*PLANE_COUNT) // This warp is doing triangle vs plane tests
	{
		subtask -= rayTasks*PLANE_COUNT; // Recompute subtask for triangle subtasks only
		int planePos = subtask % PLANE_COUNT; // Succesive subtasks do far away planes -> hopefully less lock contention in atomic operations
		findPlane(planePos, c_ns_in.rays, c_ns_in.raysIndex, rayStart, rayEnd, c_ns_in.tris, c_ns_in.trisIndex, triStart, triEnd, bbox, numAxisAlignedPlanes, numTriangleBasedPlanes, plane);

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
			int triIdx = ((int*)c_ns_in.trisIndex)[triPos]*3;

			// Fetch triangle
			float3 v0, v1, v2;
			taskFetchTri(c_ns_in.tris, triIdx, v0, v1, v2);

			int pos = getPlanePosition(plane, v0, v1, v2);

			if (pos <= 0)
				tb++;
			if (pos >= 0)
				tf++;

			// Update the split info
			SplitData *split = &(g_splitStack[taskIdx].splits[planePos]);

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
	}

	ASSERT_DIVERGENCE("splitCostParallel bottom", tid);
}

#endif

//------------------------------------------------------------------------

__device__ int classifyRay(int tid, int subtask, int start, int end, volatile const float4& splitPlane, volatile const CudaAABB& box)
{
	ASSERT_DIVERGENCE("classifyRay", tid);

	int raypos = start + subtask*WARP_SIZE + tid;
	if(raypos < end)
	{
		int rayidx = ((int*)c_ns_in.raysIndex)[raypos];

		// Fetch ray
		float3 orig, dir;
		float tmin, tmax;
		taskFetchRayVolatile(c_ns_in.rays, rayidx, orig, dir, tmin, tmax);

		// Update rays tmin and tmax
		float3 idir;
		float ooeps = exp2f(-80.0f); // Avoid div by zero.
        idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
        idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
        idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));
        float3 ood = orig * idir;

		float lox = box.m_mn.x * idir.x - ood.x;
		float hix = box.m_mx.x * idir.x - ood.x;
		float loy = box.m_mn.y * idir.y - ood.y;
		float hiy = box.m_mx.y * idir.y - ood.y;
		float loz = box.m_mn.z * idir.z - ood.z;
		float hiz = box.m_mx.z * idir.z - ood.z;
		tmin = max4(fminf(lox, hix), fminf(loy, hiy), fminf(loz, hiz), tmin);
		tmax = min4(fmaxf(lox, hix), fmaxf(loy, hiy), fmaxf(loz, hiz), tmax);

		// Clip rays - wrong, runs after first split
		/*if(s_task[threadIdx.y].depth == 0)
		{
			if(tmin < RAY_OFFSET)
				tmin = RAY_OFFSET;
			((float4*)(c_ns_in.rays + rayidx * 32 + 0))->w = tmin;
			((float4*)(c_ns_in.rays + rayidx * 32 + 16))->w = tmax;
		}*/

		int dummy;
		int pos;

		if(tmax < tmin)
			pos = 2;
		else
			pos = getPlanePosition(*((float4*)&splitPlane), orig, dir, tmin, tmax, dummy);

		// Write to auxiliary array
		((int*)c_ns_in.ppsRaysIndex)[raypos] = pos;

		return pos;
	}
	return 3;
}

//------------------------------------------------------------------------

__device__ int classifyTri(int tid, int subtask, int start, int end, volatile const float4& splitPlane)
{
	ASSERT_DIVERGENCE("classifyTri", tid);

	int tripos = start + subtask*WARP_SIZE + tid;
	if(tripos < end)
	{
		int triidx = ((int*)c_ns_in.trisIndex)[tripos]*3;

		// Fetch triangle
		float3 v0, v1, v2;
		taskFetchTri(c_ns_in.tris, triidx, v0, v1, v2);

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
		int pos = getPlanePosition(*((float4*)&splitPlane), v0, v1, v2);

		/*float4 plane; // Without typecast, possible data copy
		plane.x = splitPlane.x;
		plane.y = splitPlane.y;
		plane.z = splitPlane.z;
		plane.w = splitPlane.w;
		int pos = getPlanePosition(plane, v0, v1, v2);*/

		// Write to auxiliary array
		((int*)c_ns_in.ppsTrisIndex)[tripos] = pos;

		return pos;
	}
	return 3;
}

//------------------------------------------------------------------------

__device__ void classifyClip(int tid, int subtask, int start, int end, volatile const CudaAABB& box)
{
	ASSERT_DIVERGENCE("classifyClip", tid);

	int raypos = start + subtask*WARP_SIZE + tid;
	if(raypos < end)
	{
		int rayidx = ((int*)c_ns_in.raysIndex)[raypos];

		// Fetch ray
		float3 orig, dir;
		float tmin, tmax;
		taskFetchRayVolatile(c_ns_in.rays, rayidx, orig, dir, tmin, tmax);

		// Update rays tmin and tmax
		float3 idir;
		float ooeps = exp2f(-80.0f); // Avoid div by zero.
        idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
        idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
        idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));
        float3 ood = orig * idir;

		float lox = box.m_mn.x * idir.x - ood.x;
		float hix = box.m_mx.x * idir.x - ood.x;
		float loy = box.m_mn.y * idir.y - ood.y;
		float hiy = box.m_mx.y * idir.y - ood.y;
		float loz = box.m_mn.z * idir.z - ood.z;
		float hiz = box.m_mx.z * idir.z - ood.z;
		tmin = max4(fminf(lox, hix), fminf(loy, hiy), fminf(loz, hiz), tmin);
		tmax = min4(fmaxf(lox, hix), fmaxf(loy, hiy), fmaxf(loz, hiz), tmax);

		// Clip rays - wrong, runs after first split
		/*if(s_task[threadIdx.y].depth == 0)
		{
			if(tmin < RAY_OFFSET)
				tmin = RAY_OFFSET;
			((float4*)(c_ns_in.rays + rayidx * 32 + 0))->w = tmin;
			((float4*)(c_ns_in.rays + rayidx * 32 + 16))->w = tmax;
		}*/

		int pos;
		if(tmax < tmin)
			pos = 2;
		else
			pos = 0;

		// Write to auxiliary array
		((int*)c_ns_in.ppsRaysIndex)[raypos] = pos;
	}
}

//------------------------------------------------------------------------

#ifdef CLIP_INTERSECT

// comments and variables are with regard to -1,0 to the left, 1 to the right
__device__ void clipSort(int tid, int subtask, CUdeviceptr dataPPSSrc, CUdeviceptr dataPPSBuf, CUdeviceptr dataSort, CUdeviceptr dataIndex, int start, int end, int mid, int test)
{
	ASSERT_DIVERGENCE("clipSort", tid);

	int rayidx = start + subtask*WARP_SIZE + tid;

	if(rayidx < end)
	{
		int value = ((int*)dataPPSSrc)[ rayidx ];
		// Copy values less than 2 (uncliped) to the new array
		if( value < test )
		{
			int newSpot  = rayidx - ((int*)dataPPSBuf)[ rayidx ];
			int curIndex = ((int*)dataIndex)[ rayidx ];
			((int*)dataSort)[ newSpot ] = curIndex;
		}
	}
}

#endif

//------------------------------------------------------------------------

// Computes intersection of the selected range of rays and triangles. Thread == Ray
__device__ void intersect(int tid, int subtask, int taskIdx, int rayStart, int rayEnd, int triStart, int triEnd)
{
	ASSERT_DIVERGENCE("intersect", tid);

	int     rayIdx;                 // Ray index.
    float3  orig;                   // Ray origin.
    float3  dir;                    // Ray direction.
    float   tmin;                   // t-value from which the ray starts. Usually 0.

	int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
    float   hitT;                   // t-value of the closest intersection.
	float   hitU;                   // u-barycentric of the closest intersection.
	float   hitV;                   // v-barycentric of the closest intersection.

	// Pick ray index.
    int rayPos = rayStart + subtask*WARP_SIZE + tid;

#ifndef CLIP_INTERSECT
	if(rayPos < rayEnd)
#else
	if(rayPos < s_task[threadIdx.y].rayActive)
#endif
	{
		// Pick ray from ray index buffer
#ifndef CLIP_INTERSECT
		rayIdx = ((int*)c_ns_in.raysIndex)[rayPos];
#else
		if(s_task[threadIdx.y].rayActive != rayStart && s_task[threadIdx.y].rayActive != rayEnd)
			rayIdx = ((int*)c_ns_in.sortRays)[rayPos];
		else
			rayIdx = ((int*)c_ns_in.raysIndex)[rayPos];
#endif

#ifdef DEBUG_INFO
		taskFetchRayVolatile(c_ns_in.rays, rayIdx, orig, dir, tmin, hitT);

		float tmax = hitT;
		float tbin = tmin;

		const volatile CudaAABB& bbox = s_task[threadIdx.y].bbox;
		// Update rays tmin and tmax
		float3 idir;
		float ooeps = exp2f(-80.0f); // Avoid div by zero.
		idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
		idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
		idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));
		float3 ood = orig * idir;

		float lox = bbox.m_mn.x * idir.x - ood.x;
		float hix = bbox.m_mx.x * idir.x - ood.x;
		float loy = bbox.m_mn.y * idir.y - ood.y;
		float hiy = bbox.m_mx.y * idir.y - ood.y;
		float loz = bbox.m_mn.z * idir.z - ood.z;
		float hiz = bbox.m_mx.z * idir.z - ood.z;
		tbin = max4(fminf(lox, hix), fminf(loy, hiy), fminf(loz, hiz), tbin);
		tmax = min4(fmaxf(lox, hix), fmaxf(loy, hiy), fmaxf(loz, hiz), tmax);

		if(tmax < tbin)
			atomicAdd(&g_taskStack.tasks[taskIdx].clippedRays, 1);
#else
		taskFetchRayVolatile(c_ns_in.rays, rayIdx, orig, dir, tmin, hitT);
#endif

		// Setup
		hitIndex = -1;  // No triangle intersected so far.

		// Trace naive
		for(int triPos = triStart; triPos < triEnd; triPos++)
		//for(int triPos = 0; triPos < 10; triPos++)
		{
			// Convert triAddr using index array
			int triAddr = ((int*)c_ns_in.trisIndex)[triPos]*3; // Times 3 is for moving by whole triangles not just vertices
			float3 v0, v1, v2;
			taskFetchTri(c_ns_in.tris, triAddr, v0, v1, v2);

			float3 nrmN = cross(v1 - v0, v2 - v0);
			const float den = dot(nrmN, dir);

			//if(den >= 0.0f)
			//	continue;

			const float deni = 1.0f / den;
			const float3 org0 = v0 - orig;
			float t = dot(nrmN, org0)*deni;

			if(t > tmin && t < hitT)
			{
				const float3 crossProd = cross(dir, org0);
				const float v = dot(v0 - v2, crossProd)*deni;
				if(v >= 0.0f && v <= 1.0f)
				{
					const float u = 1 - v - (-dot(v0 - v1, crossProd)*deni); // woop
					if(u >= 0.0f && u + v <= 1.0f)
					{
						hitT = t;
						hitU = u;
						hitV = v;
						hitIndex = triAddr;
						if(c_ns_in.anyHit)
						{
							break;
						}
					}
				}
			}
		}

		if(hitIndex != -1)
		{
			// Does it have to be volatile?
			((float4*)(c_ns_in.rays + rayIdx * 32 + 16))->w = hitT;
			hitIndex = hitIndex / 3;
			((int4*)c_ns_in.results)[rayIdx] = make_int4(hitIndex, __float_as_int(hitT), __float_as_int(hitU), __float_as_int(hitV));
			/*int4* result = (int4*)(c_ns_in.results + rayIdx * 16);
			result->x = hitIndex;
			result->y = __float_as_int(hitT);
			result->z = __float_as_int(hitU);
			result->w = __float_as_int(hitV);*/
		}
	}
}

//------------------------------------------------------------------------

// Update best plane in global memory
__device__ __forceinline__ void taskUpdateBestHit(int rayIdx, int hitIndex, float hitT, float hitU, float hitV)
{
	float* g_hitT = (float*)(c_ns_in.rays + rayIdx * 32 + 16 + 12); // rays[rayIdx].hitT
	int* g_idx = (int*)(c_ns_in.results + rayIdx * 16); // hitIndex is used as lock flag, -2 means locked
	int old = *g_idx;
	int locked = 0;

#ifdef UPDATEDEADLOCK_TEST
	int lockCounter = 0;
#endif

	// Atomicaly update the closest distance in global memory
#ifdef UPDATEDEADLOCK_TEST
	while(locked == 0 && hitT < *g_hitT && lockCounter < 1000)
#else
	while(locked == 0 && hitT < *g_hitT)
#endif
	{
		if(old == -2)
			old = *g_idx;

		int retOld = -2;
		if(old > -2 && (retOld = atomicCAS((int*)g_idx, old, -2)) == old) // Potential deadlock because holding one lock and needing other one in a warp?
		{
			old = retOld;
			locked = 1;
			break;
		}
		else
		{
			old = retOld;
		}
#ifdef UPDATEDEADLOCK_TEST
		lockCounter++;
#endif
	}

#ifdef UPDATEDEADLOCK_TEST
	assert(lockCounter < 1000);
#endif

	// Update the best cost
	if(hitT < *g_hitT)
	{
		*g_hitT = hitT;
		//((int4*)c_ns_in.results)[rayIdx] = make_int4(hitIndex, __float_as_int(hitT), __float_as_int(hitU), __float_as_int(hitV));
		int4* result = (int4*)(c_ns_in.results + rayIdx * 16);
		//result->x = hitIndex; // Set as lock flag
		result->y = __float_as_int(hitT);
		result->z = __float_as_int(hitU);
		result->w = __float_as_int(hitV);
		__threadfence();
		*g_idx = hitIndex;
	}
	else if(locked == 1)
	{
		*g_idx = old;
	}
}

//------------------------------------------------------------------------

__device__ void intersectParallel(int tid, int subtask, int rayStart, int rayEnd, int triStart, int triEnd)
{
	ASSERT_DIVERGENCE("intersectParallel", tid);

	int     rayIdx;                 // Ray index.
    float3  orig;                   // Ray origin.
    float3  dir;                    // Ray direction.
    float   tmin;                   // t-value from which the ray starts. Usually 0.

	int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
    float   hitT;                   // t-value of the closest intersection.
	float   hitU;                   // u-barycentric of the closest intersection.
	float   hitV;                   // v-barycentric of the closest intersection.

	// Pick ray index.
#ifndef CLIP_INTERSECT
	int rays = rayEnd - rayStart;
#else
	int rays = s_task[threadIdx.y].rayActive - rayStart;
#endif
	int tris = triEnd - triStart;
    
	int isectPos = subtask*WARP_SIZE + tid;
	int rayPos = isectPos / tris;
	int triPos = isectPos % tris;

	s_owner[threadIdx.y][tid] = 31;     // Mark as outside array
	volatile float* red = (volatile float*)&s_sharedData[threadIdx.y][0];
	red[tid] = 31.0f; // Using same memory should not be a problem

	if(isectPos < rays*tris)
	{
		s_owner[threadIdx.y][tid] = rayPos - ((subtask*WARP_SIZE) / tris); // Index of rayPos inside this warp

		// Mark starts of segments
		if(tid == 0)
			red[s_owner[threadIdx.y][tid]] = tid; // Optimize: Needed only for debug?
		if(tid > 0 && s_owner[threadIdx.y][tid-1] != s_owner[threadIdx.y][tid])
			red[s_owner[threadIdx.y][tid]] = tid;
	}

	// Update owners
	int ownerTmp = s_owner[threadIdx.y][tid];
	s_owner[threadIdx.y][tid] = red[ownerTmp];  // We are done with red, all information is now in owner

	// Setup
	red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data
	hitIndex = -1;  // No triangle intersected so far.
	hitT = CUDART_INF_F;

	// Single ray-triangle intersection
	if(isectPos < rays*tris)
	{
		// Pick ray from ray index buffer
#ifndef CLIP_INTERSECT
		rayIdx = ((int*)c_ns_in.raysIndex)[rayStart + rayPos];
#else
		if(s_task[threadIdx.y].rayActive != rayStart && s_task[threadIdx.y].rayActive != rayEnd)
			rayIdx = ((int*)c_ns_in.sortRays)[rayStart + rayPos];
		else
			rayIdx = ((int*)c_ns_in.raysIndex)[rayStart + rayPos];
#endif

		taskFetchRayVolatile(c_ns_in.rays, rayIdx, orig, dir, tmin, hitT);

		// Convert triAddr using index array
		int triAddr = ((int*)c_ns_in.trisIndex)[triStart + triPos]*3; // Times 3 is for moving by whole triangles not just vertices
		float3 v0, v1, v2;
		taskFetchTri(c_ns_in.tris, triAddr, v0, v1, v2);
#ifdef INTERSECT_TEST
		red[tid] = hitT;
#endif

		float3 nrmN = cross(v1 - v0, v2 - v0);
		const float den = dot(nrmN, dir);

		//if(den < 0.0f)
		{
			const float deni = 1.0f / den;
			const float3 org0 = v0 - orig;
			float t = dot(nrmN, org0)*deni;

			if(t > tmin && t < hitT)
			{
				const float3 crossProd = cross(dir, org0);
				const float v = dot(v0 - v2, crossProd)*deni;
				if(v >= 0.0f && v <= 1.0f)
				{
					const float u = 1 - v - (-dot(v0 - v1, crossProd)*deni); // woop
					if(u >= 0.0f && u + v <= 1.0f)
					{
						hitT = t;
						hitU = u;
						hitV = v;
						hitIndex = triAddr;
					}
				}
			}
		}
	}

	// Reduce the best cost within the warp
	if(hitIndex != -1)
	{
		red[tid] = hitT;
	}
	segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], min);

#ifdef INTERSECT_TEST
	if(hitT < red[s_owner[threadIdx.y][tid]])
		printf("Intersect reduction failed! Tid: %d, Hit: %f, Red: %f, Owner: %d, Rays: %d, Tris: %d\n", tid, hitT, red[s_owner[threadIdx.y][tid]], s_owner[threadIdx.y][tid], rays, tris);
#endif

	if(hitIndex != -1)
	{
		// Select best intersection for each ray
		if(red[s_owner[threadIdx.y][tid]] == hitT)
			red[s_owner[threadIdx.y][tid]] = tid; // Write your vote
	
		if(red[s_owner[threadIdx.y][tid]] == tid)
		{
			taskUpdateBestHit(rayIdx, hitIndex / 3, hitT, hitU, hitV);
		}
	}
}

//------------------------------------------------------------------------

#if SPLIT_TYPE > 0 && SPLIT_TYPE <= 3
__device__ __noinline__ void computeSplit()
{
	int subtasksDone;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		// Run the split task
		splitCost(threadIdx.x, popSubtask, rayStart, rayEnd, triStart, triEnd, s_task[threadIdx.y].bbox, s_task[threadIdx.y].splitPlane, s_task[threadIdx.y].bestCost, s_task[threadIdx.y].bestOrder);
#if SPLIT_TYPE == 1
		if(s_task[threadIdx.y].lock != LockType_None)
			taskUpdateBestPlane(tid, popTaskIdx); // OPTIMIZE: Do not copy through shared memory
#endif
		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	taskFinishSplit(threadIdx.x, popTaskIdx, subtasksDone);
}
#endif

//------------------------------------------------------------------------

#if SPLIT_TYPE == 3
__device__ __noinline__ void computeSplitParallel()
{
	int subtasksDone;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		// Run the split task
		splitCostParallel(threadIdx.x, popSubtask, popTaskIdx, rayStart, rayEnd, triStart, triEnd, s_task[threadIdx.y].bbox);
		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSplitParallel(threadIdx.x, popTaskIdx, subtasksDone);
}
#endif

//------------------------------------------------------------------------

__device__ __noinline__ void computePPS1()
{
	int subtasksDone;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int step = s_task[threadIdx.y].step;
	int bestOrder = s_task[threadIdx.y].bestOrder;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		if(popSubtask < bestOrder)
		{
			if(step == 0) // Classify all rays against the best plane
				classifyRay(threadIdx.x, popSubtask, rayStart, rayEnd, s_task[threadIdx.y].splitPlane, s_task[threadIdx.y].bbox);

			pps<int>(threadIdx.x, popSubtask, s_task[threadIdx.y].step, (int*)c_ns_in.ppsRaysIndex, (int*)c_ns_in.ppsRaysBuf, s_sharedData[threadIdx.y], rayStart, rayEnd, 1);

		}
		else
		{
			int triSubtask = popSubtask - bestOrder; // Lower by the number of ray subtasks
			if(step == 0) // Classify all triangles against the best plane
				classifyTri(threadIdx.x, triSubtask, triStart, triEnd, s_task[threadIdx.y].splitPlane);

			pps<int>(threadIdx.x, triSubtask, step, (int*)c_ns_in.ppsTrisIndex, (int*)c_ns_in.ppsTrisBuf, s_sharedData[threadIdx.y], triStart, triEnd, 1);
		}

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortPPS1(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computePPS2()
{
	int subtasksDone;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int rayRight = s_task[threadIdx.y].rayRight;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int triRight = s_task[threadIdx.y].triRight;
	int step = s_task[threadIdx.y].step;
	int bestOrder = s_task[threadIdx.y].bestOrder;
	int rayDivisor = s_task[threadIdx.y].rayDivisor;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		if(popSubtask < bestOrder)
		{
			if(popSubtask < rayDivisor)
			{
				pps<int>(threadIdx.x, popSubtask, step, (int*)c_ns_in.ppsRaysIndex, (int*)c_ns_in.ppsRaysBuf, s_sharedData[threadIdx.y], rayStart, rayRight, 0);
			}
			else
			{
				int rightSubtask = popSubtask - rayDivisor;
				pps<int>(threadIdx.x, rightSubtask, step, (int*)c_ns_in.ppsRaysIndex, (int*)c_ns_in.ppsRaysBuf, s_sharedData[threadIdx.y], rayRight, rayEnd, 2);
			}
		}
		else
		{
			int triSubtask = popSubtask - bestOrder; // Lower by the number of ray subtasks
			pps<int>(threadIdx.x, triSubtask, step, (int*)c_ns_in.ppsTrisIndex, (int*)c_ns_in.ppsTrisBuf, s_sharedData[threadIdx.y], triStart, triRight, 0);
		}

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortPPS2(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computePPSUp1()
{
	int subtasksDone;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int step = s_task[threadIdx.y].step;
	int bestOrder = s_task[threadIdx.y].bestOrder;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		if(popSubtask < bestOrder)
		{
#ifndef DEBUG_PPS
			if(step == 0) // Classify all rays against the best plane
				classifyRay(threadIdx.x, popSubtask, rayStart, rayEnd, s_task[threadIdx.y].splitPlane, s_task[threadIdx.y].bbox);
#endif

			scanUp<int>(threadIdx.x, popSubtask, step, (int*)c_ns_in.ppsRaysBuf, (int*)c_ns_in.ppsRaysIndex, s_sharedData[threadIdx.y], rayStart, rayEnd, 1, plus, 0);

		}
		else
		{
			int triSubtask = popSubtask - bestOrder; // Lower by the number of ray subtasks
#ifndef DEBUG_PPS
			if(step == 0) // Classify all triangles against the best plane
				classifyTri(threadIdx.x, triSubtask, triStart, triEnd, s_task[threadIdx.y].splitPlane);
#endif

			scanUp<int>(threadIdx.x, triSubtask, step, (int*)c_ns_in.ppsTrisBuf, (int*)c_ns_in.ppsTrisIndex, s_sharedData[threadIdx.y], triStart, triEnd, 1, plus, 0);
		}

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortPPS1Up(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computePPSDown1()
{
	int subtasksDone;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int step = s_task[threadIdx.y].step;
	int bestOrder = s_task[threadIdx.y].bestOrder;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		if(popSubtask < bestOrder)
		{
			scanDown<int>(threadIdx.x, popSubtask, step, (int*)c_ns_in.ppsRaysBuf, (int*)c_ns_in.ppsRaysIndex, s_sharedData[threadIdx.y], rayStart, rayEnd, 1, plus, 0);
		}
		else
		{
			int triSubtask = popSubtask - bestOrder; // Lower by the number of ray subtasks
			scanDown<int>(threadIdx.x, triSubtask, step, (int*)c_ns_in.ppsTrisBuf, (int*)c_ns_in.ppsTrisIndex, s_sharedData[threadIdx.y], triStart, triEnd, 1, plus, 0);
		}

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortPPS1Down(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computePPSUp2()
{
	int subtasksDone;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int rayRight = s_task[threadIdx.y].rayRight;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int triRight = s_task[threadIdx.y].triRight;
	int step = s_task[threadIdx.y].step;
	int bestOrder = s_task[threadIdx.y].bestOrder;
	int rayDivisor = s_task[threadIdx.y].rayDivisor;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		if(popSubtask < bestOrder)
		{
			if(popSubtask < rayDivisor)
			{
				scanUp<int>(threadIdx.x, popSubtask, step, (int*)c_ns_in.ppsRaysBuf, (int*)c_ns_in.ppsRaysIndex, s_sharedData[threadIdx.y], rayStart, rayRight, 0, plus, 0);
			}
			else
			{
				int rightSubtask = popSubtask - rayDivisor;
				scanUp<int>(threadIdx.x, rightSubtask, step, (int*)c_ns_in.ppsRaysBuf, (int*)c_ns_in.ppsRaysIndex, s_sharedData[threadIdx.y], rayRight, rayEnd, 2, plus, 0);
			}
		}
		else
		{
			int triSubtask = popSubtask - bestOrder; // Lower by the number of ray subtasks
			scanUp<int>(threadIdx.x, triSubtask, step, (int*)c_ns_in.ppsTrisBuf, (int*)c_ns_in.ppsTrisIndex, s_sharedData[threadIdx.y], triStart, triRight, 0, plus, 0);
		}

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortPPS2Up(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computePPSDown2()
{
	int subtasksDone;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int rayRight = s_task[threadIdx.y].rayRight;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int triRight = s_task[threadIdx.y].triRight;
	int step = s_task[threadIdx.y].step;
	int bestOrder = s_task[threadIdx.y].bestOrder;
	int rayDivisor = s_task[threadIdx.y].rayDivisor;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		if(popSubtask < bestOrder)
		{
			if(popSubtask < rayDivisor)
			{
				scanDown<int>(threadIdx.x, popSubtask, step, (int*)c_ns_in.ppsRaysBuf, (int*)c_ns_in.ppsRaysIndex, s_sharedData[threadIdx.y], rayStart, rayRight, 0, plus, 0);
			}
			else
			{
				int rightSubtask = popSubtask - rayDivisor;
				scanDown<int>(threadIdx.x, rightSubtask, s_task[threadIdx.y].step, (int*)c_ns_in.ppsRaysBuf, (int*)c_ns_in.ppsRaysIndex, s_sharedData[threadIdx.y], rayRight, rayEnd, 2, plus, 0);
			}
		}
		else
		{
			int triSubtask = popSubtask - bestOrder; // Lower by the number of ray subtasks
			scanDown<int>(threadIdx.x, triSubtask, step, (int*)c_ns_in.ppsTrisBuf, (int*)c_ns_in.ppsTrisIndex, s_sharedData[threadIdx.y], triStart, triRight, 0, plus, 0);
		}

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortPPS2Down(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computeSort1()
{
	int subtasksDone;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int rayRight = s_task[threadIdx.y].rayRight;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int triRight = s_task[threadIdx.y].triRight;
	int step = s_task[threadIdx.y].step;
	int bestOrder = s_task[threadIdx.y].bestOrder;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		if(popSubtask < bestOrder)
		{
			sort(threadIdx.x, popSubtask, step, (int*)c_ns_in.ppsRaysIndex, (int*)c_ns_in.ppsRaysBuf, (int*)c_ns_in.sortRays, (int*)c_ns_in.raysIndex, rayStart, rayEnd, rayRight, 1, true);
		}
		else
		{
			int triSubtask = popSubtask - bestOrder; // Lower by the number of ray subtasks
			sort(threadIdx.x, triSubtask, step, (int*)c_ns_in.ppsTrisIndex, (int*)c_ns_in.ppsTrisBuf, (int*)c_ns_in.sortTris, (int*)c_ns_in.trisIndex, triStart, triEnd, triRight, 1, true);
		}

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortSORT1(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computeSort2()
{
	int subtasksDone;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int rayLeft = s_task[threadIdx.y].rayLeft;
	int rayRight = s_task[threadIdx.y].rayRight;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int triLeft = s_task[threadIdx.y].triLeft;
	int triRight = s_task[threadIdx.y].triRight;
	int step = s_task[threadIdx.y].step;
	int bestOrder = s_task[threadIdx.y].bestOrder;
	int rayDivisor = s_task[threadIdx.y].rayDivisor;
	int rayActive = s_task[threadIdx.y].rayActive;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
		if(popSubtask < bestOrder)
		{
			if(popSubtask < rayDivisor)
			{
				sort(threadIdx.x, popSubtask, step, (int*)c_ns_in.ppsRaysIndex, (int*)c_ns_in.ppsRaysBuf, (int*)c_ns_in.sortRays, (int*)c_ns_in.raysIndex, rayStart, rayRight, rayLeft, 0, false);
			}
			else
			{
				int rightSubtask = popSubtask - rayDivisor;
				sort(threadIdx.x, rightSubtask, step, (int*)c_ns_in.ppsRaysIndex, (int*)c_ns_in.ppsRaysBuf, (int*)c_ns_in.sortRays, (int*)c_ns_in.raysIndex, rayRight, rayEnd, rayActive, 2, false);
			}
		}
		else
		{
			int triSubtask = popSubtask - bestOrder; // Lower by the number of ray subtasks
			sort(threadIdx.x, triSubtask, step, (int*)c_ns_in.ppsTrisIndex, (int*)c_ns_in.ppsTrisBuf, (int*)c_ns_in.sortTris, (int*)c_ns_in.trisIndex, triStart, triRight, triLeft, 0, false);
		}

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishSortSORT2(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computePartition1()
{
	int subtasksDone;
	int tid = threadIdx.x;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int bestOrder = s_task[threadIdx.y].bestOrder;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;
	bool singleWarp = s_task[threadIdx.y].lock == LockType_None;
	//bool singleWarp = false;

	// Set the swap arrays
	int* inIdx;
	int* outIdx;
	int* outClass;
	int* leftCounter;
	int* rightCounter;

	s_owner[threadIdx.y][0] = rayStart;
	s_owner[threadIdx.y][1] = rayEnd;
	s_owner[threadIdx.y][2] = triStart;
	s_owner[threadIdx.y][3] = triEnd;

#if SCAN_TYPE == 2
	do
	{
		// Classify the triangles
		int pos = 3; // Outside of the interval
		int itemPos;
		int itemSum;
		int itemEnd;
		int itemIdx = -1;
		int posIdx;

		if(popSubtask < bestOrder)
		{
			// Setup the unified variables
			inIdx = (int*)c_ns_in.raysIndex;
			outIdx = (int*)c_ns_in.sortRays;
			outClass = (int*)c_ns_in.ppsRaysBuf;
			leftCounter = &g_taskStack.tasks[popTaskIdx].rayLeft;
			rightCounter = &g_taskStack.tasks[popTaskIdx].rayRight;

			itemPos = rayStart + popSubtask*WARP_SIZE + tid;
			itemSum = min(rayEnd - (rayStart + popSubtask*WARP_SIZE), WARP_SIZE); // Rays to process
			itemEnd = rayEnd;
			posIdx = 0;

#ifndef DEBUG_PPS
			//pos = classifyRay(threadIdx.x, popSubtask, rayStart, rayEnd, s_task[threadIdx.y].splitPlane, s_task[threadIdx.y].bbox);
			if(itemPos < itemEnd)
			{
				int rayidx = ((int*)c_ns_in.raysIndex)[itemPos];

				// Fetch ray
				float3 orig, dir;
				float tmin, tmax;
				taskFetchRayVolatile(c_ns_in.rays, rayidx, orig, dir, tmin, tmax);

				// Update rays tmin and tmax
				float3 idir;
				float ooeps = exp2f(-80.0f); // Avoid div by zero.
				idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
				idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
				idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));
				float3 ood = orig * idir;

				float lox = s_task[threadIdx.y].bbox.m_mn.x * idir.x - ood.x;
				float hix = s_task[threadIdx.y].bbox.m_mx.x * idir.x - ood.x;
				float loy = s_task[threadIdx.y].bbox.m_mn.y * idir.y - ood.y;
				float hiy = s_task[threadIdx.y].bbox.m_mx.y * idir.y - ood.y;
				float loz = s_task[threadIdx.y].bbox.m_mn.z * idir.z - ood.z;
				float hiz = s_task[threadIdx.y].bbox.m_mx.z * idir.z - ood.z;
				tmin = max4(fminf(lox, hix), fminf(loy, hiy), fminf(loz, hiz), tmin);
				tmax = min4(fmaxf(lox, hix), fmaxf(loy, hiy), fmaxf(loz, hiz), tmax);

				int dummy;

				if(tmax < tmin)
					pos = 2;
				else
					pos = getPlanePosition(*((float4*)&s_task[threadIdx.y].splitPlane), orig, dir, tmin, tmax, dummy);
			}
#endif
		}
		else
		{
			// Setup the unified variables
			inIdx = (int*)c_ns_in.trisIndex;
			outIdx = (int*)c_ns_in.sortTris;
			outClass = (int*)c_ns_in.ppsTrisBuf;
			leftCounter = &g_taskStack.tasks[popTaskIdx].triLeft;
			rightCounter = &g_taskStack.tasks[popTaskIdx].triRight;

			int triSubtask = popSubtask - bestOrder; // Lower by the number of ray subtasks
			itemPos = triStart + triSubtask*WARP_SIZE + tid;
			itemSum = min(triEnd - (triStart + triSubtask*WARP_SIZE), WARP_SIZE); // Triangles to process
			itemEnd = triEnd;
			posIdx = 2;

#ifndef DEBUG_PPS
			//pos = classifyTri(threadIdx.x, triSubtask, triStart, triEnd, s_task[threadIdx.y].splitPlane);
			if(itemPos < itemEnd)
			{
				int triidx = ((int*)c_ns_in.trisIndex)[itemPos]*3;

				// Fetch triangle
				float3 v0, v1, v2;
				taskFetchTri(c_ns_in.tris, triidx, v0, v1, v2);

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
				pos = getPlanePosition(*((float4*)&s_task[threadIdx.y].splitPlane), v0, v1, v2);
			}
#endif
		}

		if(itemPos < itemEnd)
		{
			itemIdx = inIdx[itemPos];
		}

		// Partition the items to the left and right children intervals

		// Scan the number of items to the left of the splitting plane
		s_sharedData[threadIdx.y][tid] = 0;
		if(pos <= 0)
			s_sharedData[threadIdx.y][tid] = 1;

		scanWarp<int>(tid, s_sharedData[threadIdx.y], plus);
		int exclusiveScan = (s_sharedData[threadIdx.y][tid] - 1);
		
		int itemCnt = s_sharedData[threadIdx.y][WARP_SIZE-1];
		if(!singleWarp && tid == 0 && itemCnt > 0)
			s_owner[threadIdx.y][posIdx+0] = atomicAdd(leftCounter, itemCnt); // Add the number of items to the left of the plane to the global counter

		// Find the output position for each thread as the sum of the output position and the exclusive scanned value
		if(pos <= 0)
		{
			outIdx[s_owner[threadIdx.y][posIdx+0] + exclusiveScan] = itemIdx;
			outClass[s_owner[threadIdx.y][posIdx+0] + exclusiveScan] = pos;
		}
		s_owner[threadIdx.y][posIdx+0] += itemCnt; // Move the position by the number of written nodes

		// Compute the number of items to the right of the splitting plane
		int inverseExclusiveScan = tid - s_sharedData[threadIdx.y][tid]; // The scan of the number of item to the right of the splitting plane
		
		itemCnt = itemSum - itemCnt;
		if(!singleWarp && tid == 0 && itemCnt > 0)
			s_owner[threadIdx.y][posIdx+1] = atomicSub(rightCounter, itemCnt); // Add the number of items to the right of the plane to the global counter

		// Find the output position for each thread as the output position minus the item count plus the scanned value
		if(pos > 0 && pos != 3)
		{
			outIdx[s_owner[threadIdx.y][posIdx+1] - itemCnt + inverseExclusiveScan] = itemIdx;
			outClass[s_owner[threadIdx.y][posIdx+1] - itemCnt + inverseExclusiveScan] = pos;
		}
		s_owner[threadIdx.y][posIdx+1] -= itemCnt; // Move the position by the number of written nodes


		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	// Write out the final positions
	if(singleWarp)
	{
		g_taskStack.tasks[popTaskIdx].rayLeft = s_owner[threadIdx.y][0];
		g_taskStack.tasks[popTaskIdx].rayRight = s_owner[threadIdx.y][1];
		g_taskStack.tasks[popTaskIdx].triLeft = s_owner[threadIdx.y][2];
		g_taskStack.tasks[popTaskIdx].triRight = s_owner[threadIdx.y][3];
	}

#elif SCAN_TYPE == 3
#error Not yet implemented for SubdivisionRT!
#endif

	__threadfence(); // Needed so that next iteration does not read uninitialized data
	taskFinishPartition1(tid, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computePartition2()
{
	int subtasksDone;
	int tid = threadIdx.x;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int rayLeft = s_task[threadIdx.y].rayLeft;
	int rayRight = s_task[threadIdx.y].rayRight;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int triLeft = s_task[threadIdx.y].triLeft;
	int triRight = s_task[threadIdx.y].triRight;
	int bestOrder = s_task[threadIdx.y].bestOrder;
	int rayDivisor = s_task[threadIdx.y].rayDivisor;
	int triDivisor = s_task[threadIdx.y].step;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;
	bool singleWarp = s_task[threadIdx.y].lock == LockType_None;
	//bool singleWarp = false;

	// Set the swap arrays
	int* inIdx;
	int* inClass;
	int* outIdx;
	int* leftCounter;
	int* rightCounter;

	s_owner[threadIdx.y][0] = rayStart;
	s_owner[threadIdx.y][1] = rayLeft;
	s_owner[threadIdx.y][2] = rayRight;
	s_owner[threadIdx.y][3] = rayEnd;
	s_owner[threadIdx.y][4] = triStart;
	s_owner[threadIdx.y][5] = triLeft;

#if SCAN_TYPE == 2
	do
	{
		// Classify the triangles
		int pos = 3; // Outside of the interval
		int itemPos;
		int itemSum;
		int itemEnd;
		int itemIdx = -1;
		int divider;
		int posIdx;

		if(popSubtask < bestOrder)
		{
			// Setup the unified variables
			inIdx = (int*)c_ns_in.sortRays;
			outIdx = (int*)c_ns_in.raysIndex;
			inClass = (int*)c_ns_in.ppsRaysBuf;

			if(popSubtask < rayDivisor)
			{
				leftCounter = (int*)&g_taskStack.tasks[popTaskIdx].bboxLeft.m_mn.x;
				rightCounter = &g_taskStack.tasks[popTaskIdx].rayLeft;

				itemPos = rayStart + popSubtask*WARP_SIZE + tid;
				itemSum = min(rayRight - (rayStart + popSubtask*WARP_SIZE), WARP_SIZE); // Rays to process
				itemEnd = rayRight;
				divider = -1;
				posIdx = 0;
			}
			else
			{
				leftCounter = &g_taskStack.tasks[popTaskIdx].rayActive;
				rightCounter = (int*)&g_taskStack.tasks[popTaskIdx].bboxRight.m_mn.x;

				int rightSubtask = popSubtask - rayDivisor;
				itemPos = rayRight + rightSubtask*WARP_SIZE + tid;
				itemSum = min(rayEnd - (rayRight + rightSubtask*WARP_SIZE), WARP_SIZE); // Rays to process
				itemEnd = rayEnd;
				divider = 1;
				posIdx = 2;
			}
		}
		else
		{
			// Setup the unified variables
			inIdx = (int*)c_ns_in.sortTris;
			outIdx = (int*)c_ns_in.trisIndex;
			inClass = (int*)c_ns_in.ppsTrisBuf;

			if(popSubtask < triDivisor)
			{
				leftCounter = (int*)&g_taskStack.tasks[popTaskIdx].bboxMiddle.m_mn.x;
				rightCounter = &g_taskStack.tasks[popTaskIdx].triLeft;

				int triSubtask = popSubtask - bestOrder; // Lower by the number of ray subtasks
				itemPos = triStart + triSubtask*WARP_SIZE + tid;
				itemSum = min(triRight - (triStart + triSubtask*WARP_SIZE), WARP_SIZE); // Triangles to process
				itemEnd = triRight;
				divider = -1;
				posIdx = 4;
			}
			else
			{
				int rightSubtask = popSubtask - triDivisor;
				itemPos = triRight + rightSubtask*WARP_SIZE + tid;
				itemSum = min(triEnd - (triRight + rightSubtask*WARP_SIZE), WARP_SIZE); // Triangles to process
				itemEnd = triEnd;
				divider = 1;
				posIdx = 6;
			}
		}

		if(itemPos < itemEnd)
		{
			itemIdx = inIdx[itemPos];
			pos = inClass[itemPos];

#ifdef RAYTRI_TEST
			if(posIdx == 0 && pos < -1 || pos > 0)
				printf("Ray error %d should be -1/0 is %d! Start %d, Left %d, Right %d, End %d\n", itemPos, pos, rayStart, rayLeft, rayRight, rayEnd);

			if(posIdx == 2 && pos < 1 || pos > 2)
				printf("Ray error %d should be 1/2 is %d! Start %d, Left %d, Right %d, End %d\n", itemPos, pos, rayStart, rayLeft, rayRight, rayEnd);

			if(posIdx == 4 && pos < -1 || pos > 0)
				printf("Tri error %d should be -1/0 is %d! Start %d, Left %d, Right %d, End %d\n", itemPos, pos, triStart, triLeft, triRight, triEnd);

			if(posIdx == 6 && pos < 1 || pos > 2)
				printf("Tri error %d should be 1/2 is %d! Start %d, Left %d, Right %d, End %d\n", itemPos, pos, triStart, triLeft, triRight, triEnd);
#endif
		}

		// Partition the items to the left and right children intervals

		if(posIdx != 6)
		{
			// Scan the number of items to the left of the splitting plane
			s_sharedData[threadIdx.y][tid] = 0;
			if(pos <= divider)
				s_sharedData[threadIdx.y][tid] = 1;

			scanWarp<int>(tid, s_sharedData[threadIdx.y], plus);
			int exclusiveScan = (s_sharedData[threadIdx.y][tid] - 1);
		
			int itemCnt = s_sharedData[threadIdx.y][WARP_SIZE-1];
			if(!singleWarp && tid == 0 && itemCnt > 0)
				s_owner[threadIdx.y][posIdx+0] = atomicAdd(leftCounter, itemCnt); // Add the number of items to the left of the plane to the global counter

			// Find the output position for each thread as the sum of the output position and the exclusive scanned value
			if(pos <= divider)
				outIdx[s_owner[threadIdx.y][posIdx+0] + exclusiveScan] = itemIdx;
			s_owner[threadIdx.y][posIdx+0] += itemCnt; // Move the position by the number of written nodes

			// Compute the number of items to the right of the splitting plane
			int inverseExclusiveScan = tid - s_sharedData[threadIdx.y][tid]; // The scan of the number of item to the right of the splitting plane
		
			itemCnt = itemSum - itemCnt;
			if(!singleWarp && tid == 0 && itemCnt > 0)
				s_owner[threadIdx.y][posIdx+1] = atomicSub(rightCounter, itemCnt); // Add the number of items to the right of the plane to the global counter

			// Find the output position for each thread as the output position minus the item count plus the scanned value
			if(pos > divider && pos != 3)
				outIdx[s_owner[threadIdx.y][posIdx+1] - itemCnt + inverseExclusiveScan] = itemIdx;
			s_owner[threadIdx.y][posIdx+1] -= itemCnt; // Move the position by the number of written nodes
		}
		else if(pos <= divider)
		{
				outIdx[itemPos] = itemIdx; // Just copy the data
		}


		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	// Write out the final positions
	if(singleWarp)
	{
		g_taskStack.tasks[popTaskIdx].bboxLeft.m_mn.x = __int_as_float(s_owner[threadIdx.y][0]);
		g_taskStack.tasks[popTaskIdx].rayLeft = s_owner[threadIdx.y][1];
		g_taskStack.tasks[popTaskIdx].rayActive = s_owner[threadIdx.y][2];
		g_taskStack.tasks[popTaskIdx].bboxRight.m_mn.x = __int_as_float(s_owner[threadIdx.y][3]);
		g_taskStack.tasks[popTaskIdx].bboxMiddle.m_mn.x = __int_as_float(s_owner[threadIdx.y][4]);
		g_taskStack.tasks[popTaskIdx].triLeft = s_owner[threadIdx.y][5];
	}

#elif SCAN_TYPE == 3
#error Not yet implemented for SubdivisionRT!
#endif

	__threadfence(); // Needed so that next iteration does not read uninitialized data
	taskFinishPartition2(tid, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

__device__ __noinline__ void computeIntersect()
{
	int subtasksDone;
	int rayStart = s_task[threadIdx.y].rayStart;
	int rayEnd = s_task[threadIdx.y].rayEnd;
	int rayRight = s_task[threadIdx.y].rayRight;
	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;
	int triRight = s_task[threadIdx.y].triRight;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	do
	{
#if ISECT_TYPE == 0
		// Intersect each ray in the set with each triangle - must be able to deal with empty tasks if they are enabled
		intersect(threadIdx.x, popSubtask, popTaskIdx, rayStart, rayEnd, triStart, triEnd);
#else
		intersectParallel(threadIdx.x, popSubtask, rayStart, rayEnd, triStart, triEnd);
#endif

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishIntersect(threadIdx.x, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

#if AABB_TYPE == 0

// Local + global reduction on x, y and z coordinates at once
// OPTIMIZE: It could be more efficient to pass op as template parameter instead of function pointer?
template<typename T>
__device__ void computeAABB(int tid, int taskIdx, int subtask, int step, int start, int left, int right, int end, T(*op)(T,T), T identity)
{
	// Data arrays
	T* x = (T*)c_ns_in.ppsTrisBuf;
	T* y = (T*)c_ns_in.ppsTrisIndex;
	T* z = (T*)c_ns_in.sortTris;

	if(step == 0) // Do local reduction, step 0
	{
		ASSERT_DIVERGENCE("computeAABB step0 top", tid);

		int tripos = start + (subtask * WARP_SIZE) + tid;
		volatile T* red = (volatile T*)&s_sharedData[threadIdx.y][0];
		red[tid] = (T)31; // Using same memory should not be a problem
		s_owner[threadIdx.y][tid] = 31;     // Mark as outside array

		ASSERT_DIVERGENCE("computeAABB step0 if1", tid);

		if(tripos < end)
		{
			s_owner[threadIdx.y][tid] = ((tripos - left) >= 0) + ((tripos - right >= 0));

			// Mark starts of segments
			if(tid == 0)
				red[s_owner[threadIdx.y][tid]] = tid; // Optimize: Needed only for debug?
			if(tid > 0 && s_owner[threadIdx.y][tid-1] != s_owner[threadIdx.y][tid])
				red[s_owner[threadIdx.y][tid]] = tid;
		}

		// Update owners
		int ownerTmp = s_owner[threadIdx.y][tid];
		s_owner[threadIdx.y][tid] = red[ownerTmp];  // We are done with red, all information is now in owner

		red[tid] = identity; // Save identities so that we do not work with uninitialized data
		ASSERT_DIVERGENCE("computeAABB step0 if2", tid);

		if(tripos < end)
		{
			int triidx = ((int*)c_ns_in.trisIndex)[tripos]*3;

			// Fetch triangle
			float3 v0, v1, v2;
			taskFetchTri(triidx, v0, v1, v2);

			// Reduce x
			red[tid] = op(op(v0.x, v1.x), v2.x); // OPTIMIZE: Do triangle bounding box computation once and load only the result
			segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);
			x[tripos] = red[tid]; // Copy results to gmem

#ifdef BBOX_TEST
			float thm = op(op(v0.x, v1.x), v2.x);
			if(s_task[threadIdx.y].type == TaskType_AABB_Min)
			{
				if(thm < red[s_owner[threadIdx.y][tid]])
				{
					printf("Min x red error %f!\n", thm);
				}
				if(thm < s_task[threadIdx.y].bbox.m_mn.x)
					printf("Min x step0 bound error task %d!\n", taskIdx);
			}
			else
			{
				if(thm > red[s_owner[threadIdx.y][tid]])
				{
					printf("Max x red error %f!\n", thm);
				}
				if(thm > s_task[threadIdx.y].bbox.m_mx.x)
					printf("Max x step0 bound error task %d!\n", taskIdx);
			}
#endif

			__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

			// Reduce y
			red[tid] = op(op(v0.y, v1.y), v2.y); // OPTIMIZE: Do triangle bounding box computation once and load only the result
			segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);
			y[tripos] = red[tid]; // Copy results to gmem

#ifdef BBOX_TEST
			thm = op(op(v0.y, v1.y), v2.y);
			if(s_task[threadIdx.y].type == TaskType_AABB_Min)
			{
				if(thm < red[s_owner[threadIdx.y][tid]])
				{
					printf("Min y red error %f!\n", thm);
				}
				if(thm < s_task[threadIdx.y].bbox.m_mn.y)
					printf("Min y step0 bound error task %d!\n", taskIdx);
			}
			else
			{
				if(thm > red[s_owner[threadIdx.y][tid]])
				{
					printf("Max y error %f!\n", thm);
				}
				if(thm > s_task[threadIdx.y].bbox.m_mx.y)
					printf("Max y step0 bound error task %d!\n", taskIdx);
			}
#endif

			__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

			// Reduce z
			red[tid] = op(op(v0.z, v1.z), v2.z); // OPTIMIZE: Do triangle bounding box computation once and load only the result
			segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);
			z[tripos] = red[tid]; // Copy results to gmem

#ifdef BBOX_TEST
			thm = op(op(v0.z, v1.z), v2.z);
			if(s_task[threadIdx.y].type == TaskType_AABB_Min)
			{
				if(thm < red[s_owner[threadIdx.y][tid]])
				{
					printf("Min z red error %f!\n", thm);
				}
				if(thm < s_task[threadIdx.y].bbox.m_mn.z)
					printf("Min z step0 bound error task %d!\n", taskIdx);
			}
			else
			{
				if(thm > red[s_owner[threadIdx.y][tid]])
				{
					printf("Max z error %f!\n", thm);
				}
				if(thm > s_task[threadIdx.y].bbox.m_mx.z)
					printf("Max z step0 bound error task %d!\n", taskIdx);
			}
#endif

			// OPTIMIZE: Maybe not needed here
			//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them
		}
	}
	else // Do global reduction, step 1 to n
	{
		ASSERT_DIVERGENCE("computeAABB step1-n top", tid);

		int i          = (subtask * WARP_SIZE) + tid;
		int blockSize  = (1 << (step+LOG_WARP_SIZE));
		int halfBlock  = blockSize >> 1;
		int blockStart = start + blockSize * i;
		int posThis = blockStart;
		int posNext = blockStart + halfBlock;

		if(posNext < end)
		{
			int ownerThis = ((posThis - left) >= 0) + ((posThis - right >= 0));
			int ownerNext = ((posNext - left) >= 0) + ((posNext - right >= 0));

			if(ownerThis == ownerNext)
			{
				x[posThis] = op(x[posThis], x[posNext]);
				y[posThis] = op(y[posThis], y[posNext]);
				z[posThis] = op(z[posThis], z[posNext]);

#ifdef BBOX_TEST
				if(s_task[threadIdx.y].type == TaskType_AABB_Min)
				{
					if(x[posThis] < s_task[threadIdx.y].bbox.m_mn.x)
						printf("Min x step1-n bound error task %d!\n", taskIdx);

					if(y[posThis] < s_task[threadIdx.y].bbox.m_mn.y)
						printf("Min y step1-n bound error task %d!\n", taskIdx);

					if(z[posThis] < s_task[threadIdx.y].bbox.m_mn.z)
						printf("Min z step1-n bound error task %d!\n", taskIdx);
				}
				else
				{
					if(x[posThis] > s_task[threadIdx.y].bbox.m_mx.x)
						printf("Max x step1-n bound error task %d!\n", taskIdx);

					if(y[posThis] > s_task[threadIdx.y].bbox.m_mx.y)
						printf("Max y step1-n bound error task %d!\n", taskIdx);

					if(z[posThis] > s_task[threadIdx.y].bbox.m_mx.z)
						printf("Max z step1-n bound error task %d!\n", taskIdx);
				}
#endif
			}
			else
			{
				int posDivide = (ownerNext == 1) ? left : right;
				x[posDivide] = op(x[posDivide], x[posNext]);
				y[posDivide] = op(y[posDivide], y[posNext]);
				z[posDivide] = op(z[posDivide], z[posNext]);

#ifdef BBOX_TEST
				if(s_task[threadIdx.y].type == TaskType_AABB_Min)
				{
					if(x[posDivide] < s_task[threadIdx.y].bbox.m_mn.x)
						printf("Min x step1-n bound error task %d!\n", taskIdx);

					if(y[posDivide] < s_task[threadIdx.y].bbox.m_mn.y)
						printf("Min y step1-n bound error task %d!\n", taskIdx);

					if(z[posDivide] < s_task[threadIdx.y].bbox.m_mn.z)
						printf("Min z step1-n bound error task %d!\n", taskIdx);
				}
				else
				{
					if(x[posDivide] > s_task[threadIdx.y].bbox.m_mx.x)
						printf("Max x step1-n bound error task %d!\n", taskIdx);

					if(y[posDivide] > s_task[threadIdx.y].bbox.m_mx.y)
						printf("Max y step1-n bound error task %d!\n", taskIdx);

					if(z[posDivide] > s_task[threadIdx.y].bbox.m_mx.z)
						printf("Max z step1-n bound error task %d!\n", taskIdx);
				}
#endif
			}
		}
	}
	__threadfence(); // Optimize: Maybe not needed here
}

#elif AABB_TYPE == 1

// Local + global reduction on x, y and z coordinates at once
// OPTIMIZE: It could be more efficient to pass op as template parameter instead of function pointer?
template<typename T>
__device__ void computeAABB(int tid, int taskIdx, int subtask, int step, int start, int left, int right, int end, T(*op)(T,T), T identity)
{
	// Data arrays
	T* x = (T*)c_ns_in.ppsTrisBuf;
	T* y = (T*)c_ns_in.ppsTrisIndex;
	T* z = (T*)c_ns_in.sortTris;

	int i          = (subtask * WARP_SIZE) + tid;
	int blockSize  = (1 << step);
	int blockStart = start + blockSize * i;
	int pos = blockStart;
	int posDivide = ((pos - right >= 0)) ? right : left;

	ASSERT_DIVERGENCE("computeAABB top", tid);

	volatile T* red = (volatile T*)&s_sharedData[threadIdx.y][0];
	// Mark as outside array
	red[tid] = (T)31; // Using same memory should not be a problem
	s_owner[threadIdx.y][tid] = 31;

	ASSERT_DIVERGENCE("computeAABB if1", tid);

	if(pos < end)
	{
		s_owner[threadIdx.y][tid] = ((pos - left) >= 0) + ((pos - right >= 0));

		// Mark starts of segments
		if(tid == 0)
			red[s_owner[threadIdx.y][tid]] = tid;
		if(tid > 0 && s_owner[threadIdx.y][tid-1] != s_owner[threadIdx.y][tid])
			red[s_owner[threadIdx.y][tid]] = tid;
	}

	// Update owners
	int ownerTmp = s_owner[threadIdx.y][tid];
	s_owner[threadIdx.y][tid] = red[ownerTmp];  // We are done with red, all information is now in owner

	red[tid] = identity; // Save identities so that we do not work with uninitialized data

	ASSERT_DIVERGENCE("computeAABB if2", tid);

	if(pos < end)
	{
		float3 v0, v1, v2;
		//float3 vec;
		// Reduce x
		if(step == 0)
		{
			// Fetch triangle
			int triidx = ((int*)c_ns_in.trisIndex)[pos]*3;
			taskFetchTri(triidx, v0, v1, v2);

			red[tid] = op(op(v0.x, v1.x), v2.x); // OPTIMIZE: Do triangle bounding box computation once and load only the result
			/*int triidx = ((int*)c_ns_in.trisIndex)[pos];
			if(s_task[threadIdx.y].type == TaskType_AABB_Min)
				vec = (((CudaAABB*)c_ns_in.trisBox)[triidx]).m_mn;
			else
				vec = (((CudaAABB*)c_ns_in.trisBox)[triidx]).m_mx;

			red[tid] = vec.x;*/
		}
		else
		{
			red[tid] = x[pos];
		}
		segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);

		if(tid == 0) // Start of first segment in shared memory
		{
			x[pos] = red[tid]; // Copy results to gmem
		}
		else if(tid == s_owner[threadIdx.y][tid]) // Start of other segments in shared memory
		{
			if(step == 0)
				x[posDivide] = red[tid]; // Copy results to gmem
			else
				x[posDivide] = op(x[posDivide], red[tid]);
		}

#ifdef BBOX_TEST
		float thm = op(op(v0.x, v1.x), v2.x);
		if(step == 0 && s_task[threadIdx.y].type == TaskType_AABB_Min)
		{
			if(thm < red[s_owner[threadIdx.y][tid]])
			{
				printf("Min x red error %f!\n", thm);
			}
			if(thm < s_task[threadIdx.y].bbox.m_mn.x)
				printf("Min x step0 bound error task %d!\n", taskIdx);
		}
		else if(step == 0)
		{
			if(thm > red[s_owner[threadIdx.y][tid]])
			{
				printf("Max x red error %f!\n", thm);
			}
			if(thm > s_task[threadIdx.y].bbox.m_mx.x)
				printf("Max x step0 bound error task %d!\n", taskIdx);
		}
#endif

		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		// Reduce y
		if(step == 0)
		{
			red[tid] = op(op(v0.y, v1.y), v2.y); // OPTIMIZE: Do triangle bounding box computation once and load only the result
			//red[tid] = vec.y;
		}
		else
		{
			red[tid] = y[pos];
		}
		segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);

		if(tid == 0) // Start of first segment in shared memory
		{
			y[pos] = red[tid]; // Copy results to gmem
		}
		else if(tid == s_owner[threadIdx.y][tid]) // Start of other segments in shared memory
		{
			if(step == 0)
				y[posDivide] = red[tid]; // Copy results to gmem
			else
				y[posDivide] = op(y[posDivide], red[tid]);
		}

#ifdef BBOX_TEST
		thm = op(op(v0.y, v1.y), v2.y);
		if(step == 0 && s_task[threadIdx.y].type == TaskType_AABB_Min)
		{
			if(thm < red[s_owner[threadIdx.y][tid]])
			{
				printf("Min y red error %f!\n", thm);
			}
			if(thm < s_task[threadIdx.y].bbox.m_mn.y)
				printf("Min y step0 bound error task %d!\n", taskIdx);
		}
		else if(step == 0)
		{
			if(thm > red[s_owner[threadIdx.y][tid]])
			{
				printf("Max y error %f!\n", thm);
			}
			if(thm > s_task[threadIdx.y].bbox.m_mx.y)
				printf("Max y step0 bound error task %d!\n", taskIdx);
		}
#endif

		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		// Reduce z
		if(step == 0)
		{
			red[tid] = op(op(v0.z, v1.z), v2.z); // OPTIMIZE: Do triangle bounding box computation once and load only the result
			//red[tid] = vec.z;
		}
		else
		{
			red[tid] = z[pos];
		}
		segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);

		if(tid == 0) // Start of first segment in shared memory
		{
			z[pos] = red[tid]; // Copy results to gmem
		}
		else if(tid == s_owner[threadIdx.y][tid]) // Start of other segments in shared memory
		{
			if(step == 0)
				z[posDivide] = red[tid]; // Copy results to gmem
			else
				z[posDivide] = op(z[posDivide], red[tid]);
		}

#ifdef BBOX_TEST
		thm = op(op(v0.z, v1.z), v2.z);
		if(step == 0 && s_task[threadIdx.y].type == TaskType_AABB_Min)
		{
			if(thm < red[s_owner[threadIdx.y][tid]])
			{
				printf("Min z red error %f!\n", thm);
			}
			if(thm < s_task[threadIdx.y].bbox.m_mn.z)
				printf("Min z step0 bound error task %d!\n", taskIdx);
		}
		else if(step == 0)
		{
			if(thm > red[s_owner[threadIdx.y][tid]])
			{
				printf("Max z error %f!\n", thm);
			}
			if(thm > s_task[threadIdx.y].bbox.m_mx.z)
				printf("Max z step0 bound error task %d!\n", taskIdx);
		}
#endif
	}
	//__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	// Possibly the problem only manifests when global synchronization is skipped
}

#elif AABB_TYPE == 2

// Local + global reduction on x, y and z coordinates at once
// OPTIMIZE: It could be more efficient to pass op as template parameter instead of function pointer?
template<typename T>
__device__ void computeAABB(int tid, int taskIdx, int subtask, int step, int start, int left, int right, int end, T(*op)(T,T), T identity)
{
	// Data arrays
	T* x = (T*)c_ns_in.ppsTrisBuf;
	T* y = (T*)c_ns_in.ppsTrisIndex;
	T* z = (T*)c_ns_in.sortTris;

	int i          = (subtask * WARP_SIZE) + tid;
	int blockSize  = (1 << step);
	int halfBlock  = blockSize >> 1;
	int blockStart = start + blockSize * i;
	int pos = blockStart;
	int posNext = blockStart + halfBlock;
	int posDivide = ((pos - right >= 0)) ? right : left;
	int posDivideNext = ((posNext - right >= 0)) ? right : left;

	ASSERT_DIVERGENCE("computeAABB top", tid);

	volatile T* red = (volatile T*)&s_sharedData[threadIdx.y][0];
	// Mark as outside array
	red[tid] = (T)31; // Using same memory should not be a problem
	s_owner[threadIdx.y][tid] = 31;

	ASSERT_DIVERGENCE("computeAABB if1", tid);

	if(pos < end)
	{
		s_owner[threadIdx.y][tid] = ((pos - left) >= 0) + ((pos - right >= 0));

		// Mark starts of segments
		if(tid == 0)
			red[s_owner[threadIdx.y][tid]] = tid;
		if(tid > 0 && s_owner[threadIdx.y][tid-1] != s_owner[threadIdx.y][tid])
			red[s_owner[threadIdx.y][tid]] = tid;
	}

	// Update owners
	int ownerTmp = s_owner[threadIdx.y][tid];
	s_owner[threadIdx.y][tid] = red[ownerTmp];  // We are done with red, all information is now in owner

	red[tid] = identity; // Save identities so that we do not work with uninitialized data

	ASSERT_DIVERGENCE("computeAABB if2", tid);

	if(pos < end)
	{
		float3 v0, v1, v2;
		//float3 vec;
		// Reduce x
		if(step == 0)
		{
			// Fetch triangle
			int triidx = ((int*)c_ns_in.trisIndex)[pos]*3;
			taskFetchTri(triidx, v0, v1, v2);

			red[tid] = op(op(v0.x, v1.x), v2.x); // OPTIMIZE: Do triangle bounding box computation once and load only the result
			/*int triidx = ((int*)c_ns_in.trisIndex)[pos];
			if(s_task[threadIdx.y].type == TaskType_AABB_Min)
				vec = (((CudaAABB*)c_ns_in.trisBox)[triidx]).m_mn;
			else
				vec = (((CudaAABB*)c_ns_in.trisBox)[triidx]).m_mx;

			red[tid] = vec.x;*/
		}
		else if(posNext < end && (posNext < left || (pos >= left && posNext < right)  || pos >= right)) // Same owners
		{
			red[tid] = op(x[pos], x[posNext]);
		}
		else
		{
			red[tid] = x[pos];
			if(posNext < end)
			{
				x[posDivideNext] = op(x[posDivideNext], x[posNext]);
			}
		}
		segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);

		if(tid == 0) // Start of first segment in shared memory
		{
			x[pos] = red[tid]; // Copy results to gmem
		}
		else if(tid == s_owner[threadIdx.y][tid]) // Start of other segments in shared memory
		{
			if(step == 0)
				x[posDivide] = red[tid]; // Copy results to gmem
			else
				x[posDivide] = op(x[posDivide], red[tid]);
		}

#ifdef BBOX_TEST
		float thm = op(op(v0.x, v1.x), v2.x);
		if(step == 0 && s_task[threadIdx.y].type == TaskType_AABB_Min)
		{
			if(thm < red[s_owner[threadIdx.y][tid]])
			{
				printf("Min x red error %f!\n", thm);
			}
			if(thm < s_task[threadIdx.y].bbox.m_mn.x)
				printf("Min x step0 bound error task %d!\n", taskIdx);
		}
		else if(step == 0)
		{
			if(thm > red[s_owner[threadIdx.y][tid]])
			{
				printf("Max x red error %f!\n", thm);
			}
			if(thm > s_task[threadIdx.y].bbox.m_mx.x)
				printf("Max x step0 bound error task %d!\n", taskIdx);
		}
#endif

		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		// Reduce y
		if(step == 0)
		{
			red[tid] = op(op(v0.y, v1.y), v2.y); // OPTIMIZE: Do triangle bounding box computation once and load only the result
			//red[tid] = vec.y;
		}
		else if(posNext < end && (posNext < left || (pos >= left && posNext < right)  || pos >= right)) // Same owners
		{
			red[tid] = op(y[pos], y[posNext]);
		}
		else
		{
			red[tid] = y[pos];
			if(posNext < end)
			{
				y[posDivideNext] = op(y[posDivideNext], y[posNext]);
			}
		}
		segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);

		if(tid == 0) // Start of first segment in shared memory
		{
			y[pos] = red[tid]; // Copy results to gmem
		}
		else if(tid == s_owner[threadIdx.y][tid]) // Start of other segments in shared memory
		{
			if(step == 0)
				y[posDivide] = red[tid]; // Copy results to gmem
			else
				y[posDivide] = op(y[posDivide], red[tid]);
		}

#ifdef BBOX_TEST
		thm = op(op(v0.y, v1.y), v2.y);
		if(step == 0 && s_task[threadIdx.y].type == TaskType_AABB_Min)
		{
			if(thm < red[s_owner[threadIdx.y][tid]])
			{
				printf("Min y red error %f!\n", thm);
			}
			if(thm < s_task[threadIdx.y].bbox.m_mn.y)
				printf("Min y step0 bound error task %d!\n", taskIdx);
		}
		else if(step == 0)
		{
			if(thm > red[s_owner[threadIdx.y][tid]])
			{
				printf("Max y error %f!\n", thm);
			}
			if(thm > s_task[threadIdx.y].bbox.m_mx.y)
				printf("Max y step0 bound error task %d!\n", taskIdx);
		}
#endif

		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		// Reduce z
		if(step == 0)
		{
			red[tid] = op(op(v0.z, v1.z), v2.z); // OPTIMIZE: Do triangle bounding box computation once and load only the result
			//red[tid] = vec.z;
		}
		else if(posNext < end && (posNext < left || (pos >= left && posNext < right)  || pos >= right)) // Same owners
		{
			red[tid] = op(z[pos], z[posNext]);
		}
		else
		{
			red[tid] = z[pos];
			if(posNext < end)
			{
				z[posDivideNext] = op(z[posDivideNext], z[posNext]);
			}
		}
		segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);

		if(tid == 0) // Start of first segment in shared memory
		{
			z[pos] = red[tid]; // Copy results to gmem
		}
		else if(tid == s_owner[threadIdx.y][tid]) // Start of other segments in shared memory
		{
			if(step == 0)
				z[posDivide] = red[tid]; // Copy results to gmem
			else
				z[posDivide] = op(z[posDivide], red[tid]);
		}

#ifdef BBOX_TEST
		thm = op(op(v0.z, v1.z), v2.z);
		if(step == 0 && s_task[threadIdx.y].type == TaskType_AABB_Min)
		{
			if(thm < red[s_owner[threadIdx.y][tid]])
			{
				printf("Min z red error %f!\n", thm);
			}
			if(thm < s_task[threadIdx.y].bbox.m_mn.z)
				printf("Min z step0 bound error task %d!\n", taskIdx);
		}
		else if(step == 0)
		{
			if(thm > red[s_owner[threadIdx.y][tid]])
			{
				printf("Max z error %f!\n", thm);
			}
			if(thm > s_task[threadIdx.y].bbox.m_mx.z)
				printf("Max z step0 bound error task %d!\n", taskIdx);
		}
#endif
	}
	//__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	// Possibly the problem only manifests when global synchronization is skipped
}

#elif AABB_TYPE == 3

// Local reduction on x, y and z coordinates at once for both min and max.
// OPTIMIZE: It could be more efficient to pass op as template parameter instead of function pointer?
// OPTIMIZE: SOA layout?
template<typename T>
__device__ __noinline__ void computeAABB()
{
	int subtasksDone;
	int tid = threadIdx.x;
	int start = s_task[threadIdx.y].triStart;
	int end = s_task[threadIdx.y].triEnd;
	int left = s_task[threadIdx.y].triLeft;
	int right = s_task[threadIdx.y].triRight;
	int step = s_task[threadIdx.y].step;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	int floatsPerAABB = (sizeof(CudaAABB)/sizeof(float));
	// Data arrays - only one is used in each phase.
	T* b0 = (T*)c_ns_in.ppsTrisBuf;
	T* b1 = (T*)c_ns_in.ppsTrisIndex;

	// Array for holding partial results caused by segmented reduction - one box (right-end interval)
	T* st = (T*)c_ns_in.ppsTrisIndex; // Use first item of the input/output array

	// Choose the input arrays
	T* in;
	T* out;
	if((step/LOG_WARP_SIZE) % 2 == 0)
	{
		in = b0;
		out = b1;
	}
	else
	{
		in = b1;
		out = b0;
	}

	do
	{
		int warpPos    = (popSubtask * WARP_SIZE);
		int i          = warpPos + tid;
		int blockSize  = (1 << step);
		int firstBlock = start + warpPos*blockSize;
		int pos        = start + blockSize * i;
		// OPTIMIZE: Set the positions so that they do not cross cachelines?
		int inPos      = start + warpPos*floatsPerAABB + 2*floatsPerAABB; // Data are serialized at the beginning of the array
		int outPos     = start + popSubtask*floatsPerAABB + 2*floatsPerAABB + tid; // Threads cooperate in a single CudaAABB store
		int stPos      = start + tid; // Threads cooperate in a single CudaAABB store

		volatile CudaAABB& seg0 = (firstBlock < left) ? s_task[threadIdx.y].bboxLeft : (firstBlock < right) ? s_task[threadIdx.y].bboxMiddle : s_task[threadIdx.y].bboxRight;
		volatile CudaAABB& seg1 = s_task[threadIdx.y].bboxMiddle;
		volatile CudaAABB& seg2 = s_task[threadIdx.y].bboxRight;

		ASSERT_DIVERGENCE("computeAABB top", tid);

		volatile T* redX = (volatile T*)&s_sharedData[threadIdx.y][0];
		volatile T* redY = (volatile T*)&s_newTask[threadIdx.y];
		volatile T* redZ = ((volatile T*)&s_newTask[threadIdx.y])+WARP_SIZE;
		// Mark as outside array
		redX[tid] = (T)31; // Using same memory should not be a problem
		s_owner[threadIdx.y][tid] = 31;
		bool segmented = false;
		bool sgm1, sgm2;
		bool lastLevel = (popSubtask == 0 && s_task[threadIdx.y].origSize == 1); // This call computes the last level of the tree
		int owner = 31;

		ASSERT_DIVERGENCE("computeAABB if1", tid);

		if(pos < end)
		{
			owner = ((pos - left) >= 0) + ((pos - right >= 0));
			s_owner[threadIdx.y][tid] = owner;

			// Mark starts of segments
			if(tid == 0)
				redX[s_owner[threadIdx.y][tid]] = tid;
			if(tid > 0 && s_owner[threadIdx.y][tid-1] != s_owner[threadIdx.y][tid])
			{
				redX[s_owner[threadIdx.y][tid]] = tid;
				segmented = true;
			}
		}

		segmented = __any(segmented);
		sgm1 = segmented && __any(owner == 0) && __any(owner == 1);
		sgm2 = segmented && __any(owner == 2);

		// Load the start of the segment if this warp crosses its boundary
		// OPTIMIZE: If owner1 is start of segment, it will be repeatedly saved twice instead of once
		if(step != 0 && (sgm1 || lastLevel))
		{
			volatile float* bbox = (volatile float*)&seg1;
			if(tid < floatsPerAABB)
			{
				bbox[tid] = st[stPos]; // Load the data to shared memory
			}
		}
		if(step != 0 && (sgm2 || lastLevel))
		{
			volatile float* bbox = (volatile float*)&seg2;
			if(tid < floatsPerAABB)
			{
				bbox[tid] = st[stPos + floatsPerAABB]; // Load the data to shared memory
			}
		}

		// Update owners
		int ownerTmp = s_owner[threadIdx.y][tid];
		s_owner[threadIdx.y][tid] = redX[ownerTmp];  // We are done with red, all information is now in owner

		float3 v0, v1, v2;
		//float3 vec;

		ASSERT_DIVERGENCE("computeAABB if2", tid);

		// Save identities so that we do not work with uninitialized data
		redX[tid] = CUDART_INF_F;
		redY[tid] = CUDART_INF_F;
		redZ[tid] = CUDART_INF_F;

		if(pos < end)
		{		
			// Reduce minimum coordinates
			if(step == 0)
			{
				// Fetch triangle
				int triidx = ((int*)c_ns_in.trisIndex)[pos]*3;
				taskFetchTri(c_ns_in.tris, triidx, v0, v1, v2);

				redX[tid] = fminf(min(v0.x, v1.x), v2.x);
				redY[tid] = fminf(min(v0.y, v1.y), v2.y);
				redZ[tid] = fminf(min(v0.z, v1.z), v2.z);
			
				// OPTIMIZE: Do triangle bounding box computation once and load only the result
				//int triidx = ((int*)c_ns_in.trisIndex)[pos];
				//if(s_task[threadIdx.y].type == TaskType_AABB_Min)
				//	vec = (((CudaAABB*)c_ns_in.trisBox)[triidx]).m_mn;
				//else
				//	vec = (((CudaAABB*)c_ns_in.trisBox)[triidx]).m_mx;

				//red[tid] = vec.x;
			}
			else
			{
				redX[tid] = in[inPos + floatsPerAABB*tid + 0]; // +0 means mn.x
				redY[tid] = in[inPos + floatsPerAABB*tid + 1]; // +1 means mn.y
				redZ[tid] = in[inPos + floatsPerAABB*tid + 2]; // +2 means mn.z
			}
		
			// Reduce min
			if(segmented)
			{
				segReduceWarp(tid, redX, redY, redZ, &s_owner[threadIdx.y][0], min);
			}
			else
			{
				reduceWarp(tid, redX, min);
				reduceWarp(tid, redY, min);
				reduceWarp(tid, redZ, min);
			}

			if(tid == 0) // Start of first segment in shared memory
			{
				seg0.m_mn.x = redX[tid];
				seg0.m_mn.y = redY[tid];
				seg0.m_mn.z = redZ[tid];
			}
			else if(tid == s_owner[threadIdx.y][tid]) // Start of other segments in shared memory
			{
				volatile CudaAABB& seg = (owner == 1) ? seg1 : seg2;
				if(step == 0)
				{
					seg.m_mn.x = redX[tid];
					seg.m_mn.y = redY[tid];
					seg.m_mn.z = redZ[tid];
				}
				else
				{
					seg.m_mn.x = fminf(seg.m_mn.x, redX[tid]);
					seg.m_mn.y = fminf(seg.m_mn.y, redY[tid]);
					seg.m_mn.z = fminf(seg.m_mn.z, redZ[tid]);
				}
			}

	#ifdef BBOX_TEST
			if(step == 0)
			{
				float thm = min(min(v0.x, v1.x), v2.x);
				if(thm < redX[s_owner[threadIdx.y][tid]])
					printf("Min x red error %f!\n", thm);
				if(thm < s_task[threadIdx.y].bbox.m_mn.x)
					printf("Min x step0 bound error task %d!\n", taskIdx);

				thm = min(min(v0.y, v1.y), v2.y);
				if(thm < redY[s_owner[threadIdx.y][tid]])
					printf("Min y red error %f!\n", thm);
				if(thm < s_task[threadIdx.y].bbox.m_mn.y)
					printf("Min y step0 bound error task %d!\n", taskIdx);

				thm = min(min(v0.z, v1.z), v2.z);
				if(thm < redZ[s_owner[threadIdx.y][tid]])
					printf("Min z red error %f!\n", thm);
				if(thm < s_task[threadIdx.y].bbox.m_mn.z)
					printf("Min z step0 bound error task %d!\n", taskIdx);
			}
	#endif
		}

		// Save identities so that we do not work with uninitialized data
		redX[tid] = -CUDART_INF_F;
		redY[tid] = -CUDART_INF_F;
		redZ[tid] = -CUDART_INF_F;

		if(pos < end)
		{
			// Reduce maximum coordinates
			if(step == 0)
			{
				redX[tid] = fmaxf(max(v0.x, v1.x), v2.x);
				redY[tid] = fmaxf(max(v0.y, v1.y), v2.y);
				redZ[tid] = fmaxf(max(v0.z, v1.z), v2.z);
			
				// OPTIMIZE: Do triangle bounding box computation once and load only the result
				//int triidx = ((int*)c_ns_in.trisIndex)[pos];
				//if(s_task[threadIdx.y].type == TaskType_AABB_Min)
				//	vec = (((CudaAABB*)c_ns_in.trisBox)[triidx]).m_mn;
				//else
				//	vec = (((CudaAABB*)c_ns_in.trisBox)[triidx]).m_mx;

				//red[tid] = vec.x;
			}
			else
			{
				redX[tid] = in[inPos + floatsPerAABB*tid + 3]; // +3 means mx.x
				redY[tid] = in[inPos + floatsPerAABB*tid + 4]; // +4 means mx.y
				redZ[tid] = in[inPos + floatsPerAABB*tid + 5]; // +5 means mx.z
			}

			// Reduce max
			if(segmented)
			{
				segReduceWarp(tid, redX, redY, redZ, &s_owner[threadIdx.y][0], max);
			}
			else
			{
				reduceWarp(tid, redX, max);
				reduceWarp(tid, redY, max);
				reduceWarp(tid, redZ, max);
			}

			if(tid == 0) // Start of first segment in shared memory
			{
				seg0.m_mx.x = redX[tid];
				seg0.m_mx.y = redY[tid];
				seg0.m_mx.z = redZ[tid];
			}
			else if(tid == s_owner[threadIdx.y][tid]) // Start of other segments in shared memory
			{
				volatile CudaAABB& seg = (owner == 1) ? seg1 : seg2;
				if(step == 0)
				{
					seg.m_mx.x = redX[tid];
					seg.m_mx.y = redY[tid];
					seg.m_mx.z = redZ[tid];
				}
				else
				{
					seg.m_mx.x = fmaxf(seg.m_mx.x, redX[tid]);
					seg.m_mx.y = fmaxf(seg.m_mx.y, redY[tid]);
					seg.m_mx.z = fmaxf(seg.m_mx.z, redZ[tid]);
				}
			}

	#ifdef BBOX_TEST
			if(step == 0)
			{
				float thm = max(max(v0.x, v1.x), v2.x);
				if(thm > redX[s_owner[threadIdx.y][tid]])
					printf("Max x red error %f!\n", thm);
				if(thm > s_task[threadIdx.y].bbox.m_mx.x)
					printf("Max x step0 bound error task %d!\n", taskIdx);
			
				thm = max(max(v0.y, v1.y), v2.y);
				if(thm > redY[s_owner[threadIdx.y][tid]])
					printf("Max y error %f!\n", thm);
				if(thm > s_task[threadIdx.y].bbox.m_mx.y)
					printf("Max y step0 bound error task %d!\n", taskIdx);

				thm = max(max(v0.z, v1.z), v2.z);
				if(thm > redZ[s_owner[threadIdx.y][tid]])
					printf("Max z error %f!\n", thm);
				if(thm > s_task[threadIdx.y].bbox.m_mx.z)
					printf("Max z step0 bound error task %d!\n", taskIdx);
			}
	#endif
		}

		// Write out the data
		if(!lastLevel) // Nost whole array (last iteration)
		{
			if(tid < floatsPerAABB)
			{
				volatile float* bbox = (volatile float*)&seg0;
				out[outPos] = bbox[tid]; // Write the data to global memory
			}

			// Write the start of the segment if this warp starts with the boundary between segments or crosses it
			// OPTIMIZE: If owner1 is start of segment, it will be repeatedly saved twice instead of once
			if((step == 0 && firstBlock == left) || sgm1)
			{
				if(tid < floatsPerAABB)
				{
					volatile float* bbox = (volatile float*)&seg1;
					st[stPos] = bbox[tid]; // Write the data to global memory
				}
			}
			if((step == 0 && firstBlock == right) || sgm2)
			{
				if(tid < floatsPerAABB)
				{
					volatile float* bbox = (volatile float*)&seg2;
					st[stPos + floatsPerAABB] = bbox[tid]; // Write the data to global memory
				}
			}
		}
		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	__threadfence();
	taskFinishAABB(tid, popTaskIdx, subtasksDone);
}

#endif

//------------------------------------------------------------------------

#ifdef SNAPSHOT_POOL
// Constantly take snapshots of the pool
__device__ void snapshot(int tid, int* header, volatile Task* tasks, int *unfinished, int *stackTop, volatile int* img, volatile int* red)
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
extern "C" __global__ void __launch_bounds__(NUM_THREADS, NUM_BLOCKS_PER_SM) trace(void)
{
	volatile int* taskAddr = (volatile int*)(&s_task[threadIdx.y]);
	int tid = threadIdx.x;

#ifdef SNAPSHOT_POOL
	int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
	if(warpIdx == 0)
	{
		snapshot(tid, g_taskStack.header, g_taskStack.tasks, &g_taskStack.unfinished, &g_taskStack.top, (volatile int*)&s_task[threadIdx.y], (volatile int*)s_sharedData[threadIdx.y]);
		return;
	}
#endif

/*#if PARALLELISM_TEST >= 0
	int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
	if(warpIdx == 0 && tid == 0)
		printf("Output start!\n");
#endif*/

	s_task[threadIdx.y].lock = LockType_Free; // Prepare task

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT) || defined(COUNT_STEPS_DEQUEUE)
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
			if(s_task[threadIdx.y].popTaskIdx >= 0 && s_task[threadIdx.y].popTaskIdx < g_taskStack.sizePool)
			{
				//taskLoadFirstFromGMEM(tid, s_task[threadIdx.y].popTaskIdx, &s_task[threadIdx.y]);
				Task* g_task = &g_taskStack.tasks[s_task[threadIdx.y].popTaskIdx];
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
				g_taskStack.unfinished = 1;
			}
#else
			//taskLoadFirstFromGMEM(tid, taskIdx, &s_task[threadIdx.y]);
			Task* g_task = &g_taskStack.tasks[s_task[threadIdx.y].popTaskIdx];
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
			info.rays = s_task[threadIdx.y].rayEnd - s_task[threadIdx.y].rayStart;
			info.tris = s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triStart;
			info.type = s_task[threadIdx.y].type;
			info.chunks = s_task[threadIdx.y].origSize;
			info.popCount = s_task[threadIdx.y].popCount;
			info.depth = s_task[threadIdx.y].depth;
			info.stackTop = *((int*)&g_taskStack.top);
			info.clockSearch = *(long long int*)&(s_sharedData[threadIdx.y][4]);
			info.clockDequeue = clock64();
#endif
		}

		//if(taskIdx > 7 || (taskIdx == 7 && s_task[threadIdx.y].type == 9 && s_task[threadIdx.y].step > 0))
		/*if(taskIdx > 6 || taskIdx == 6 && s_task[threadIdx.y].type > 9)
		//if(taskIdx > 7 || (taskIdx == 7 && s_task[threadIdx.y].type == 10))
		{
			//if(tid == 0)
			//{
			//	printf("warpId %d\n", blockDim.y*blockIdx.x + threadIdx.y);
			//	printf("Task taskIdx: %d\n", taskIdx);
			//	printf("Global unfinished: %d\n", g_taskStack.unfinished);
			//	//printf("Header: %d\n", subtask);
			//	printf("Unfinished: %d\n", s_task[threadIdx.y].unfinished);
			//	printf("Type: %d\n", s_task[threadIdx.y].type);
			//	printf("RayStart: %d\n", s_task[threadIdx.y].rayStart);
			//	printf("RayEnd: %d\n", s_task[threadIdx.y].rayEnd);
			//	printf("TriStart: %d\n", s_task[threadIdx.y].triStart);
			//	printf("TriEnd: %d\n", s_task[threadIdx.y].triEnd);
			//	printf("Depend1: %d\n", s_task[threadIdx.y].depend1);
			//	printf("Depend2: %d\n", s_task[threadIdx.y].depend2);
			//	printf("Box: (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
			//		s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
			//	printf("\n");
			//}
			return;
		}*/

		ASSERT_DIVERGENCE("taskProcessWorkUntilDone top", tid);

		switch(s_task[threadIdx.y].type) // Decide what to do
		{
#if SPLIT_TYPE > 0 && SPLIT_TYPE <= 3
		case TaskType_Split:
			computeSplit();
			break;
#endif

#if SPLIT_TYPE == 3
		case TaskType_SplitParallel:
			computeSplitParallel();
			break;
#endif

		// --------------------------------------------------

#if SCAN_TYPE == 0
		case TaskType_Sort_PPS1:
			computePPS1();
			break;

		case TaskType_Sort_PPS2:
			computePPS2();
			break;

#elif SCAN_TYPE == 1
		case TaskType_Sort_PPS1_Up:
			computePPSUp1();
			break;

		case TaskType_Sort_PPS1_Down:
			computePPSDown1();
			break;

		case TaskType_Sort_PPS2_Up:
			computePPSUp2();
			break;

		case TaskType_Sort_PPS2_Down:
			computePPSDown2();
			break;
#endif
#if SCAN_TYPE < 2
		case TaskType_Sort_SORT1:
			computeSort1();
			break;

		case TaskType_Sort_SORT2:
			computeSort2();
			break;

#elif SCAN_TYPE == 2 || SCAN_TYPE == 3
		case TaskType_Sort_SORT1:
			computePartition1();
			break;

		case TaskType_Sort_SORT2:
			computePartition2();
			break;
#else
#error Unknown SCAN_TYPE!
#endif

		// --------------------------------------------------

#ifdef RAYTRI_TEST
		case TaskType_RayTriTestSORT1:
			do {
				if(s_task[threadIdx.y].popSubtask < s_task[threadIdx.y].bestOrder)
				{
					int rayidx = s_task[threadIdx.y].rayStart + subtask*WARP_SIZE + tid;

					if(rayidx < s_task[threadIdx.y].rayRight && ((int*)c_ns_in.ppsRaysIndex)[rayidx] > 0)
						printf("Ray error %d should be -1/0 is %d! Start %d, Left %d, Right %d, End %d\n", rayidx, ((int*)c_ns_in.ppsRaysIndex)[rayidx], s_task[threadIdx.y].rayStart, s_task[threadIdx.y].rayLeft, s_task[threadIdx.y].rayRight, s_task[threadIdx.y].rayEnd);

					if(rayidx >= s_task[threadIdx.y].rayRight && rayidx < s_task[threadIdx.y].rayEnd && ((int*)c_ns_in.ppsRaysIndex)[rayidx] < 1)
						printf("Ray error %d should be 1/2 is %d! Start %d, Left %d, Right %d, End %d\n", rayidx, ((int*)c_ns_in.ppsRaysIndex)[rayidx], s_task[threadIdx.y].rayStart, s_task[threadIdx.y].rayLeft, s_task[threadIdx.y].rayRight, s_task[threadIdx.y].rayEnd);
				}
				else
				{
					int triSubtask = s_task[threadIdx.y].popSubtask - s_task[threadIdx.y].bestOrder; // Lower by the number of ray subtasks
					int triidx = s_task[threadIdx.y].triStart + triSubtask*WARP_SIZE + tid;
				
					if(triidx < s_task[threadIdx.y].triRight && ((int*)c_ns_in.ppsTrisIndex)[triidx] > 0)
						printf("Tri error %d should be -1/0 is %d! Start %d, Left %d, Right %d, End %d\n", triidx, ((int*)c_ns_in.ppsTrisIndex)[triidx], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triLeft, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd);

					if(triidx >= s_task[threadIdx.y].triRight && triidx < s_task[threadIdx.y].triEnd && ((int*)c_ns_in.ppsTrisIndex)[triidx] < 1)
						printf("Tri error %d should be 1 is %d! Start %d, Left %d, Right %d, End %d\n", triidx, ((int*)c_ns_in.ppsTrisIndex)[triidx], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triLeft, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd);
				}
				subtasksDone = taskReduceSubtask(s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popStart, s_task[threadIdx.y].popCount);
			} while(subtasksDone == -1);
			
			taskFinishSortSORT1(tid, s_task[threadIdx.y].popTaskIdx, subtasksDone);
			break;

		case TaskType_RayTriTestSORT2:
			do {
				if(s_task[threadIdx.y].popSubtask < s_task[threadIdx.y].bestOrder)
				{
					int rayidx = s_task[threadIdx.y].rayStart + subtask*WARP_SIZE + tid;

					if(rayidx < s_task[threadIdx.y].rayLeft && ((int*)c_ns_in.ppsRaysIndex)[rayidx] != -1)
						printf("Ray error %d should be -1 is %d! Start %d, Left %d, Right %d, End %d\n", rayidx, ((int*)c_ns_in.ppsRaysIndex)[rayidx], s_task[threadIdx.y].rayStart, s_task[threadIdx.y].rayLeft, s_task[threadIdx.y].rayRight, s_task[threadIdx.y].rayEnd);

					if(rayidx >= s_task[threadIdx.y].rayLeft && rayidx < s_task[threadIdx.y].rayRight && ((int*)c_ns_in.ppsRaysIndex)[rayidx] != 0)
						printf("Ray error %d should be 0 is %d! Start %d, Left %d, Right %d, End %d\n", rayidx, ((int*)c_ns_in.ppsRaysIndex)[rayidx], s_task[threadIdx.y].rayStart, s_task[threadIdx.y].rayLeft, s_task[threadIdx.y].rayRight, s_task[threadIdx.y].rayEnd);

					int removedRays = 0;
					if(s_task[threadIdx.y].rayRight < s_task[threadIdx.y].rayEnd)
						removedRays = ((int*)c_ns_in.ppsRaysBuf)[ s_task[threadIdx.y].rayEnd - 1 ];
					if(removedRays < 0 || removedRays > s_task[threadIdx.y].rayEnd - s_task[threadIdx.y].rayRight)
						printf("Errorneous number of clipped rays: %d!\n", removedRays);

					if(rayidx >= s_task[threadIdx.y].rayRight && rayidx < s_task[threadIdx.y].rayEnd-removedRays && ((int*)c_ns_in.ppsRaysIndex)[rayidx] != 1)
						printf("Ray error %d should be 1 is %d! Start %d, Left %d, Right %d, End %d\n", rayidx, ((int*)c_ns_in.ppsRaysIndex)[rayidx], s_task[threadIdx.y].rayStart, s_task[threadIdx.y].rayLeft, s_task[threadIdx.y].rayRight, s_task[threadIdx.y].rayEnd);

					if(rayidx >= s_task[threadIdx.y].rayEnd-removedRays && rayidx < s_task[threadIdx.y].rayEnd && ((int*)c_ns_in.ppsRaysIndex)[rayidx] != 2)
						printf("Ray error %d should be 2 is %d! Start %d, Left %d, Right %d, End %d\n", rayidx, ((int*)c_ns_in.ppsRaysIndex)[rayidx], s_task[threadIdx.y].rayStart, s_task[threadIdx.y].rayLeft, s_task[threadIdx.y].rayRight, s_task[threadIdx.y].rayEnd);
				}
				else
				{
					int triSubtask = s_task[threadIdx.y].popSubtask - s_task[threadIdx.y].bestOrder; // Lower by the number of ray subtasks
					int triidx = s_task[threadIdx.y].triStart + triSubtask*WARP_SIZE + tid;
				
					if(triidx < s_task[threadIdx.y].triLeft && ((int*)c_ns_in.ppsTrisIndex)[triidx] != -1)
						printf("Tri error %d should be -1 is %d! Start %d, Left %d, Right %d, End %d\n", triidx, ((int*)c_ns_in.ppsTrisIndex)[triidx], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triLeft, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd);

					if(triidx >= s_task[threadIdx.y].triLeft && triidx < s_task[threadIdx.y].triRight && ((int*)c_ns_in.ppsTrisIndex)[triidx] != 0)
						printf("Tri error %d should be 0 is %d! Start %d, Left %d, Right %d, End %d\n", triidx, ((int*)c_ns_in.ppsTrisIndex)[triidx], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triLeft, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd);

					if(triidx >= s_task[threadIdx.y].triRight && triidx < s_task[threadIdx.y].triEnd && ((int*)c_ns_in.ppsTrisIndex)[triidx] != 1)
						printf("Tri error %d should be 1 is %d! Start %d, Left %d, Right %d, End %d\n", triidx, ((int*)c_ns_in.ppsTrisIndex)[triidx], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triLeft, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd);
				}
				subtasksDone = taskReduceSubtask(s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popStart, s_task[threadIdx.y].popCount);
			} while(subtasksDone == -1);
			taskFinishSortSORT2(tid, s_task[threadIdx.y].popTaskIdx, subtasksDone);
			break;
#endif
			
		// --------------------------------------------------

#if AABB_TYPE < 3
		case TaskType_AABB_Min:
			// Do segmented reduction on the triangle bounding boxes
			do {
				computeAABB<float>(tid, s_task[threadIdx.y].popTaskIdx, s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].step, s_task[threadIdx.y].triStart, s_task[threadIdx.y].triLeft, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd, min, CUDART_INF_F);
				subtasksDone = taskReduceSubtask(s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popStart, s_task[threadIdx.y].popCount);
			} while(subtasksDone == -1);

			__threadfence(); // Probably needed so that next iteration does not read uninitialized data
			taskFinishAABB(tid, s_task[threadIdx.y].popTaskIdx, subtasksDone);
			break;

		case TaskType_AABB_Max:
			// Do segmented reduction on the triangle bounding boxes
			do {
				computeAABB<float>(tid, s_task[threadIdx.y].popTaskIdx, s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].step, s_task[threadIdx.y].triStart, s_task[threadIdx.y].triLeft, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd, max, -CUDART_INF_F);
				subtasksDone = taskReduceSubtask(s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popStart, s_task[threadIdx.y].popCount);
			} while(subtasksDone == -1);

			__threadfence(); // Probably needed so that next iteration does not read uninitialized data
			taskFinishAABB(tid, s_task[threadIdx.y].popTaskIdx, subtasksDone);
			break;

#elif AABB_TYPE == 3
		case TaskType_AABB:
			// Do segmented reduction on the triangle bounding boxes
			computeAABB<float>();
			break;
#endif

		// --------------------------------------------------

		case TaskType_Intersect:
			computeIntersect();
			break;

		// --------------------------------------------------

#ifdef CLIP_INTERSECT
		case TaskType_ClipPPS:
			// run pss for 2's
			do {
				if(s_task[threadIdx.y].step == 0) // Classify all rays against the best plane
					classifyClip(tid, s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].rayStart, s_task[threadIdx.y].rayEnd, s_task[threadIdx.y].bbox);

				pps<int>(tid, s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].step, c_ns_in.ppsRaysIndex, c_ns_in.ppsRaysBuf, s_task[threadIdx.y].rayStart, s_task[threadIdx.y].rayEnd, 2);
				subtasksDone = taskReduceSubtask(s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popStart, s_task[threadIdx.y].popCount);
			} while(subtasksDone == -1);

			taskFinishClipPPS(tid, s_task[threadIdx.y].popTaskIdx, subtasksDone);
			break;

			case TaskType_ClipSORT:
			// sort 0 to the left and 2 to the right
			do {
				clipSort(tid, s_task[threadIdx.y].popSubtask, c_ns_in.ppsRaysIndex, c_ns_in.ppsRaysBuf, c_ns_in.sortRays, c_ns_in.raysIndex, s_task[threadIdx.y].rayStart, s_task[threadIdx.y].rayEnd, s_task[threadIdx.y].rayActive, 2);
				subtasksDone = taskReduceSubtask(s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popStart, s_task[threadIdx.y].popCount);
			} while(subtasksDone == -1);

			taskFinishClipSORT(tid, s_task[threadIdx.y].popTaskIdx, subtasksDone);
			break;
#endif
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

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT) || defined(COUNT_STEPS_DEQUEUE)
	// Write out work statistics
	if(tid == 0)
	{
		int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
		((float4*)c_ns_in.debug)[warpIdx] = make_float4(numSteps[threadIdx.y]*1.0f, sumSteps[threadIdx.y]*1.0f/numSteps[threadIdx.y], maxSteps[threadIdx.y]*1.0f, numRestarts[threadIdx.y]*1.0f);
		//((float4*)c_ns_in.debug)[warpIdx] = make_float4(numSteps[threadIdx.y]*1.0f, sumSteps[threadIdx.y]*1.0f/numSteps[threadIdx.y], maxSteps[threadIdx.y]*1.0f, numUnderStack[threadIdx.y]*1.0f/numTries[threadIdx.y]);
	}
#endif
}

//------------------------------------------------------------------------

__device__ float4 cross4f3 (const float4& v1, const float4& v2)
{ 
	float4 ret;
	ret.x = v1.y * v2.z - v1.z * v2.y;
	ret.y = v1.z * v2.x - v1.x * v2.z;
	ret.z = v1.x * v2.y - v1.y * v2.x;
	ret.w = 0.f;
	return ret;
}

__device__ float4 minus4f3 (const float4& v1, const float4& v2)
{ 
	float4 ret;
	ret.x = v1.x - v2.x;
	ret.y = v1.y - v2.y;
	ret.z = v1.z - v2.z;
	ret.w = 0.f;
	return ret;
}

__device__ float dot4f3 (const float4& v1, const float4& v2)
{
	return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ float4 retFloat4 (const float& x, const float& y, const float& z)
{
	float4 ret;
	ret.x = x;
	ret.y = y;
	ret.z = z;
	ret.w = 0.f;
	return ret;
}

//------------------------------------------------------------------------

extern "C" __global__ void __naive(void)
{

	int     rayidx;                 // Ray index.
    float   origx, origy, origz;    // Ray origin.
    float   dirx, diry, dirz;       // Ray direction.
    float   tmin;                   // t-value from which the ray starts. Usually 0.

	int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
    float   hitT;                   // t-value of the closest intersection.
	float   hitU;                   // u-barycentric of the closest intersection.
	float   hitV;                   // v-barycentric of the closest intersection.

	// Pick ray index.
    rayidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    if (rayidx >= c_ns_in.numRays)
        return;

    // Fetch ray.
    float4 o = *(float4*)(c_ns_in.rays + rayidx * 32 + 0);
    float4 d = *(float4*)(c_ns_in.rays + rayidx * 32 + 16);
    origx = o.x, origy = o.y, origz = o.z;
    dirx = d.x, diry = d.y, dirz = d.z;
    tmin = o.w;

	// Setup
	hitIndex = -1;  // No triangle intersected so far.
    hitT     = d.w; // tmax

	// Trace naive
	for (int triAddr = 0; triAddr < c_ns_in.numTris * 3 ; triAddr += 3)
    {
        float4 v00 = tex1Dfetch(t_trisA, triAddr + 0);
		float4 v11 = tex1Dfetch(t_trisA, triAddr + 1);
		float4 v22 = tex1Dfetch(t_trisA, triAddr + 2);

		float4 nrmN = cross4f3(minus4f3(v11,v00),minus4f3(v22,v00));
		const float den = dot4f3(nrmN,retFloat4(dirx,diry,dirz));

		//if(den >= 0.0f)
		//	continue;

		const float deni = 1.0f / den;
		const float4 org0 = minus4f3(v00,retFloat4(origx,origy,origz));
		float t = dot4f3(nrmN,org0)*deni;

		if (t > tmin && t < hitT)
		{
			const float4 crossProd = cross4f3(retFloat4(dirx,diry,dirz),org0);
			const float v = dot4f3(minus4f3(v00,v22),crossProd)*deni;
			if (v >= 0.0f && v <= 1.0f)
			{
				const float u = 1 - v - (-dot4f3(minus4f3(v00,v11),crossProd)*deni); // woop
				if (u >= 0.0f && u + v <= 1.0f)
				{
					hitT = t;
					hitU = u;
					hitV = v;
					hitIndex = triAddr;
					if(c_ns_in.anyHit)
					{
						break;
					}
				}
			}
		}
    }

	if(hitIndex != -1)
		hitIndex = hitIndex / 3;
	((int4*)c_ns_in.results)[rayidx] = make_int4(hitIndex, __float_as_int(hitT), __float_as_int(hitU), __float_as_int(hitV));
}

//------------------------------------------------------------------------
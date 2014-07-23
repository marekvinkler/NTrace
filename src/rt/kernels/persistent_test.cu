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
    Quick sort specialization of the framework.

    "Massively Parallel Hierarchical Scene Sorting with Applications in Rendering",
    Marek Vinkler, Michal Hapala, Jiri Bittner and Vlastimil Havran,
    Computer Graphics Forum 2012
*/

#include "rt_common.cu"
#include "CudaBuilderKernels.hpp"

//------------------------------------------------------------------------
// Shared variables.
//------------------------------------------------------------------------

__shared__ volatile TaskBVH s_task[NUM_WARPS_PER_BLOCK]; // Memory holding information about the currently processed task
__shared__ volatile TaskBVH s_newTask[NUM_WARPS_PER_BLOCK]; // Memory for the new task to be created in
__shared__ volatile int s_sharedInt[NUM_WARPS_PER_BLOCK]; // We need shared memory to distribute data from one thread to all the others in the same warp
__shared__ volatile int s_sharedData[NUM_WARPS_PER_BLOCK][WARP_SIZE]; // Shared memory for inside warp use
__shared__ volatile int s_owner[NUM_WARPS_PER_BLOCK][WARP_SIZE]; // Another shared pool

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
	volatile TaskBVH* g_task = &g_taskStackBVH.tasks[taskIdx];
	taskAddr[tid] = ((volatile int*)g_task)[tid]; // Every thread copies one word of task data
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
	volatile TaskBVH* g_task = &g_taskStackBVH.tasks[taskIdx];
	int offset = 128/sizeof(int); // 128B offset
	taskAddr[tid+offset] = ((volatile int*)g_task)[tid+offset]; // Every thread copies one word of task data
	ASSERT_DIVERGENCE("taskLoadSecondFromGMEM bottom", tid);
}

//------------------------------------------------------------------------

// Copies first cache line of the task to Task taskIdx
__device__ __forceinline__ void taskSaveFirstToGMEM(int tid, int taskIdx, const volatile TaskBVH& task)
{
	ASSERT_DIVERGENCE("taskSaveFirstToGMEM top", tid);
	// Copy the data to global memory
	volatile int* taskAddr = (volatile int*)(&g_taskStackBVH.tasks[taskIdx]);
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
	volatile int* taskAddr = (volatile int*)(&g_taskStackBVH.tasks[taskIdx]);
	int offset = 128/sizeof(int); // 128B offset
	taskAddr[tid+offset] = ((const volatile int*)&task)[tid+offset]; // Every thread copies one word of data of its task
	ASSERT_DIVERGENCE("taskSaveSecondToGMEM bottom", tid);
}

//------------------------------------------------------------------------

__device__ __forceinline__ int taskPopCount(int status)
{
	//return 1;
	return c_env.popCount;
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
	int endAfter = TaskType_Max;

	if(phase == endAfter && s_task[threadIdx.y].type != phase)
	{
#ifdef KEEP_ALL_TASKS
		taskSaveFirstToGMEM(tid, taskIdx, s_task[threadIdx.y]);
#endif
		*((volatile int*)&g_taskStackBVH.unfinished) = 0;
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

// Decides what type of task should be created
__device__ int taskDecideType(int tid, volatile TaskBVH* newTask, int& childrenIdx)
{
	int trisLeft = newTask->triRight - newTask->triStart;
	int trisRight = newTask->triEnd - newTask->triRight;
	int tris = trisLeft + trisRight;

	if(s_task[threadIdx.y].depth > (c_env.optMaxDepth-1))
	{
		return 1;
	}

	if(tris < 2)
		return 1;
	else
		return 0;
}

//------------------------------------------------------------------------

// Creates child task
__device__ void taskChildTask(volatile TaskBVH* newTask)
{
	newTask->lock = LockType_Free;	

	newTask->unfinished = taskWarpSubtasks(newTask->triEnd-newTask->triStart);

	if(newTask->unfinished == 1) // Leaf
	{
		newTask->type = TaskType_Intersect;
	}
	else
	{
#if SCAN_TYPE == 0
		newTask->type = TaskType_Sort_PPS1;
#elif SCAN_TYPE == 1
		if(newTask->unfinished < 8) // Value of 8 corresponds to 256 items where there is a crossover between naive and Harris
			newTask->type = TaskType_Sort_PPS1;
		else
			newTask->type = TaskType_Sort_PPS1_Up;
#endif
		newTask->pivot = ((volatile int*)c_bvh_in.trisIndex)[newTask->triStart + (newTask->triEnd-newTask->triStart)/2];
	}
	newTask->step = 0;
	newTask->depth = s_task[threadIdx.y].depth+1;

#ifdef DEBUG_INFO
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

	//if(tid == 4)
	{
		switch(subIdx) // Choose which subtask we are creating
		{
		case 1:
			newTask->triStart   = s_task[threadIdx.y].triRight;
			newTask->triEnd     = s_task[threadIdx.y].triEnd;
			break;

		case 0:
			newTask->triStart   = s_task[threadIdx.y].triStart;
			newTask->triEnd     = s_task[threadIdx.y].triRight;
			break;
		}
	}

	//if(tid == 30)
	{
		taskChildTask(newTask);
	}

	newTask->taskID = subIdx;
	newTask->origSize = newTask->unfinished;
}

//------------------------------------------------------------------------

// Adds subtasks of a task into a global task queue
__device__ void __noinline__ taskEnqueueSubtasks(int tid, int taskIdx, int childrenIdx) // Must be __noinline__
{
	ASSERT_DIVERGENCE("taskEnqueueSubtasks top", tid);

	volatile int *stackTop = &g_taskStackBVH.top;
#if ENQUEUE_TYPE == 0
	int beg = *stackTop;
	bool goRight = true;
#elif ENQUEUE_TYPE == 1
	int beg = g_taskStackBVH.bottom;
#elif ENQUEUE_TYPE == 2 || ENQUEUE_TYPE == 3
	int beg = *stackTop;
#endif

//#pragma unroll 2 // OPTIMIZE: Is this beneficial?
	for(int i = 0; i < 2; i++)
	{
		taskCreateSubtask(tid, &s_newTask[threadIdx.y], i); // Fill newTask with valid task for ID=i
		if(s_newTask[threadIdx.y].triEnd - s_newTask[threadIdx.y].triStart <= 1) // Skip leaf
			continue;

		s_newTask[threadIdx.y].parentIdx = s_task[threadIdx.y].nodeIdx;
		s_newTask[threadIdx.y].nodeIdx = childrenIdx+i;
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
			taskEnqueueLeft(tid, g_taskStackBVH.header, s_sharedData[threadIdx.y], newStatus, beg, &g_taskStackBVH.unfinished, g_taskStackBVH.size); // Go left of beg and fill empty tasks
		}
#else
		taskEnqueueLeft(tid, g_taskStackBVH.header, s_sharedData[threadIdx.y], newStatus, beg, &g_taskStackBVH.unfinished, g_taskStackBVH.size); // Go left of beg and fill empty tasks
#endif

		// All threads
		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle1", tid);

		/*if(s_task[threadIdx.y].depth > 0 && taskIdx > 5 && tid == 0)
		{
			printf("warpId %d\n", blockDim.y*blockIdx.x + threadIdx.y);
			printf("task taskIdx: %d\n", beg);
			printf("Header: %d\n", newStatus);
			printf("Unfinished: %d\n", newTask[threadIdx.y].unfinished);
			printf("Type: %d\n", newTask[threadIdx.y].type);
			printf("RayStart: %d\n", newTask[threadIdx.y].rayStart);
			printf("RayEnd: %d\n", newTask[threadIdx.y].rayEnd);
			printf("TriStart: %d\n", newTask[threadIdx.y].triStart);
			printf("TriEnd: %d\n", newTask[threadIdx.y].triEnd);
			printf("Depend1: %d\n", newTask[threadIdx.y].depend1);
			printf("Depend2: %d\n", newTask[threadIdx.y].depend2);
			printf("Box: (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n", newTask[threadIdx.y].bbox.m_mn.x, newTask[threadIdx.y].bbox.m_mn.y, newTask[threadIdx.y].bbox.m_mn.z,
				newTask[threadIdx.y].bbox.m_mx.x, newTask[threadIdx.y].bbox.m_mx.y, newTask[threadIdx.y].bbox.m_mx.z);
			printf("\n");
		}*/

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT)
		maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], numReads[threadIdx.y]);
		sumSteps[threadIdx.y] += numReads[threadIdx.y];
		numSteps[threadIdx.y]++;

		/*if(numReads[threadIdx.y] > 30) // Test long waiting
		{
			g_taskStackBVH.unfinished = 1;
			int warpIdx = (blockDim.y*blockIdx.x + threadIdx.y);
			printf("Warp %d ended on task: %d, bottom %d\n", warpIdx, beg, g_taskStackBVH.bottom);
			break;
		}*/
#endif

		if(tid == 24)
		{
			atomicMax(&g_taskStackBVH.top, beg); // Update the stack top to be a larger position than all nonempty tasks
#if ENQUEUE_TYPE == 1
			atomicMax(&g_taskStackBVH.bottom, beg);  // Update the stack bottom
#endif
		}

#ifdef DIVERGENCE_TEST
		if(beg >= 0 && beg < g_taskStackBVH.size) // TESTING ONLY - WRITE WILL CAUSE "UNKNOWN ERROR" IF WARP DIVERGES
			taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
		else
			printf("task adding on invalid index: %d, Tid %d\n", beg, tid);
#else
		taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
#endif

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

#if DEQUEUE_TYPE == 3
		if(tid == 24)
		{
			int pos = atomicInc(&g_taskStackBVH.activeTop, ACTIVE_MAX);  // Update the active task
			g_taskStackBVH.active[pos] = beg;
		}
#endif

		// Unlock the task - set the task status
#ifdef DIVERGENCE_TEST
		if(beg >= 0 && beg < g_taskStackBVH.size) // TESTING ONLY - WRITE WILL CAUSE "UNKNOWN ERROR" IF WARP DIVERGES
		{
#ifdef CUTOFF_DEPTH
			if(s_newTask[threadIdx.y].depth > c_env.optCutOffDepth)
				g_taskStackBVH.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks
			else
				g_taskStackBVH.header[beg] = newStatus; // This operation is atomic anyway
#else
			g_taskStackBVH.header[beg] = newStatus; // This operation is atomic anyway
			//g_taskStackBVH.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks
#endif
		}
		else
			printf("task adding on invalid index: %d, Tid %d\n", beg, tid);
#else
#ifdef CUTOFF_DEPTH
		if(s_newTask[threadIdx.y].depth > c_env.optCutOffDepth)
			g_taskStackBVH.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks
		else
			g_taskStackBVH.header[beg] = newStatus; // This operation is atomic anyway
#else
		g_taskStackBVH.header[beg] = newStatus; // This operation is atomic anyway
		//g_taskStackBVH.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks
#endif
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

#define WAIT_COUNT 1000000

#if DEQUEUE_TYPE <= 3

__device__ __noinline__ bool taskDequeue(int tid, int& subtask, int& taskIdx, int &popCount)
{
	ASSERT_DIVERGENCE("taskDequeue", tid);

#if PARALLELISM_TEST >= 0
	volatile int* active = &g_numActive;
#endif

	if(tid == 13) // Only thread 0 acquires the work
	{
/*#if PARALLELISM_TEST >= 0
		volatile int *active = &g_numActive;
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
		volatile int* header = g_taskStackBVH.header;
		volatile int *unfinished = &g_taskStackBVH.unfinished;
		volatile int *stackTop = &g_taskStackBVH.top;

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
		volatile unsigned int *activeTop = &g_taskStackBVH.activeTop;
		volatile int* act = g_taskStackBVH.active;
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
				popCount = taskPopCount(*((volatile int*)&g_taskStackBVH.tasks[beg].origSize));
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
				*unfinished = 0;
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
			popCount = taskPopCount(*((volatile int*)&g_taskStackBVH.tasks[beg].origSize));
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

__device__ __noinline__ bool taskDequeue(int tid, int& subtask, int& taskIdx, int &popCount)
{
	ASSERT_DIVERGENCE("taskDequeue", tid);

	// Initiate variables
	volatile int* header = g_taskStackBVH.header;
	volatile int* unfinished = &g_taskStackBVH.unfinished;
	volatile int* stackTop = &g_taskStackBVH.top;
	volatile int* red = (volatile int*)&s_newTask[threadIdx.y];
	volatile int* owner = s_owner[threadIdx.y];

	int status = TaskHeader_Active;
	int counter = 0; // TESTING ONLY: Allows to undeadlock a failed run!
#ifdef COUNT_STEPS_DEQUEUE
	int readCounter = 1;
#endif

#ifdef SNAPSHOT_WARP
	long long int clock = clock64();
	*(long long int*)&(s_sharedData[threadIdx.y][4]) = clock;
#endif

	//int beg = *stackTop - tid;
	int warpIdx = (blockDim.y*blockIdx.x + threadIdx.y);
	int top = *stackTop;
	int beg = (warpIdx % taskWarpSubtasks(top)) * WARP_SIZE - tid;

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
			beg = *stackTop - tid; // Try again from a new beginning
			counter++;
			continue;
		}

		// Initialize memory for reduction
		red[tid] = 0;
		if(status > TaskHeader_Active)
		{
			red[tid] = *((volatile int*)&g_taskStackBVH.tasks[beg].origSize);
			//red[tid] = status;
		}

		reduceWarp<int>(tid, red, plus);

		popCount = max((red[tid] / NUM_WARPS) + 1, taskPopCount(status));

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
				s_sharedData[threadIdx.y][0] = atomicSub(&g_taskStackBVH.header[beg], popCount); // Try to update and return the current value
				s_sharedData[threadIdx.y][1] = beg;
				s_sharedData[threadIdx.y][2] = popCount;

#ifdef INTERSECT_TEST
				if(s_sharedData[threadIdx.y][0] == TaskHeader_Empty)
					printf("Latency error!\n");
#endif
			}
		}
		status = s_sharedData[threadIdx.y][0];

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
		// status now holds the information about what tasks this warp has to do but in a reversed order and offseted by 1
		// (status starts at the number of subtasks and ends at 1)
		subtask = s_sharedData[threadIdx.y][0]-1;
		taskIdx = s_sharedData[threadIdx.y][1];
		popCount = s_sharedData[threadIdx.y][2];
		return true;
	}
}

#endif

//------------------------------------------------------------------------

// Finishes a sort task
__device__ void taskFinishSort(int tid, int taskIdx, int leaf)
{
	// We should update the dependencies before we start adding new tasks because these subtasks may be finished before this is done

#ifndef DEBUG_INFO
	atomicCAS(&g_taskStackBVH.top, taskIdx, max(taskIdx-1, 0)); // Try decreasing the stack top
#endif

#if PARALLELISM_TEST == 0
	atomicSub(&g_numActive, 1);
#endif

	int fullLeft = (s_task[threadIdx.y].triRight - s_task[threadIdx.y].triStart) > 1;
	int fullRight = (s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triRight) > 1;
	int numSubtasks = leaf ? 0 : fullLeft+fullRight;
#ifdef CUTOFF_DEPTH
	if(s_task[threadIdx.y].depth == c_env.optCutOffDepth)
		numSubtasks = 0;
#endif

	// Decrease the waiting counters (increase the number of tasks the task is waiting on) for tasks dependent on this one
	// Update taskStack.unfinished
	atomicSub(&g_taskStackBVH.unfinished, numSubtasks-1);

#ifndef DEBUG_INFO
	g_taskStackBVH.header[taskIdx] = TaskHeader_Empty; // Empty this task

#if ENQUEUE_TYPE == 1
	atomicMin(&g_taskStackBVH.bottom, taskIdx); // Update the stack bottom
#endif

#if ENQUEUE_TYPE == 3
	int pos = atomicInc(&g_taskStackBVH.emptyTop, EMPTY_MAX);  // Update the empty task
	g_taskStackBVH.empty[pos] = taskIdx;
#endif
#endif
}

//------------------------------------------------------------------------

__device__ void taskFinishTask(int tid, int taskIdx)
{
#ifndef DEBUG_INFO
		taskLoadSecondFromGMEM(tid, taskIdx, s_task[threadIdx.y]); // Load bbox data
#endif
#ifdef DEBUG_INFO
		if(tid == 1)
			s_task[threadIdx.y].sync++;
#endif

		s_task[threadIdx.y].lock = LockType_Free;

		//*((volatile int*)&g_taskStackBVH.unfinished) = 0;
		//return; // Measure time without enqueue

		ASSERT_DIVERGENCE("taskFinishTask sort", tid);

		int childrenIdx;
		int leaf = taskDecideType(tid, &s_task[threadIdx.y], childrenIdx);

		if(tid == 15)
		{
			taskFinishSort(tid, taskIdx, leaf);
		}

		if(!leaf) // Subdivide
		{
			ASSERT_DIVERGENCE("taskFinishTask enqueue", tid);

			// Enqueue the new tasks
			taskEnqueueSubtasks(tid, taskIdx, childrenIdx);
		}

#ifdef DEBUG_INFO
		taskSaveFirstToGMEM(tid, taskIdx, s_task[threadIdx.y]); // Make sure results are visible in global memory
#endif

#if PARALLELISM_TEST == 1
		if(tid == 0)
			atomicSub(&g_numActive, 1);
#endif
}

//#if SCAN_TYPE == 0

//------------------------------------------------------------------------

__device__ void taskFinishSortPPS1(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortPPS1 top", tid);

	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(tid == 0)
	{
		int finished;
		if(s_task[threadIdx.y].lock == LockType_None)
			finished = countDown;
		else
			finished = atomicSub(&g_taskStackBVH.tasks[taskIdx].unfinished, countDown); // Lower the number of unfinished tasks

		if(finished == countDown) // Finished is the value before Dec, thus == countDown means last. We have finished the task and are responsible for cleaning up
		{
			int step = s_task[threadIdx.y].step;
			int unfinished = taskNextPhaseCountPPS(step, triStart, triEnd); // Returns 0 when no new phase is needed

			if(unfinished != 0) // Move to next phase of the current task
			{
				s_task[threadIdx.y].unfinished = unfinished;
				step++;
				s_task[threadIdx.y].step = step; // Increase the step
				finished = -1; // Make the other threads join for the finish
			}
			else // Move to SORT1
			{
				int triRight = triEnd - ((volatile int*)c_bvh_in.ppsTrisBuf)[triEnd - 1]; // Seems that must be volatile
#ifdef RAYTRI_TEST
				if(triRight < triStart || triRight > triEnd)
				{
					printf("PPS1 error triStart %d, triRight %d, triEnd %d!\n", triStart, triRight, triEnd);
					//triRight = triStart;
				}
#endif

				unfinished = 0; // OPTIMIZE: Should not be needed
				if(triRight != triStart)
				{
					s_task[threadIdx.y].type = TaskType_Sort_SORT1;
					unfinished = taskWarpSubtasksZero(triEnd - triRight);
					finished = -1;
				}

				/*if(triRight == triEnd)
				{
					printf("WTF\n");
					for(int i = triStart; i < triEnd; i++)
						printf("Pivot %d Pos %d = %d (class %d pps %d test %d)\n", s_task[threadIdx.y].pivot, i, ((volatile int*)c_bvh_in.trisIndex)[i], ((volatile int*)c_bvh_in.ppsTrisIndex)[i], ((volatile int*)c_bvh_in.ppsTrisBuf)[i], ((volatile int*)c_bvh_in.trisIndex)[i] >= s_task[threadIdx.y].pivot);
				}*/

				if(unfinished == 0) // Nothing to sort -> Move to Sort_PPS1
				{
					int pivotPos = triStart + (triEnd-triStart)/2;
					// Swap the lowest element to the beginning
					int temp = ((volatile int*)c_bvh_in.trisIndex)[triStart];
					((volatile int*)c_bvh_in.trisIndex)[triStart] = ((volatile int*)c_bvh_in.trisIndex)[pivotPos];
					((volatile int*)c_bvh_in.trisIndex)[pivotPos] = temp;
					//printf("Start %d Right %d End %d\n", triStart, triRight, triEnd);
					
					if(triEnd - triStart <= 2) // End computation
					{
						triRight = triStart+1;
						finished = -100;
					}
					else
					{
						triStart++; // Force split on unsubdivided task
						triRight = -1;
						s_task[threadIdx.y].triStart = triStart;
						s_task[threadIdx.y].type = TaskType_Sort_PPS1;
						s_task[threadIdx.y].pivot = ((volatile int*)c_bvh_in.trisIndex)[triStart + (triEnd-triStart)/2];
						unfinished = taskWarpSubtasksZero(triEnd - triStart);
						finished = -1; // Make the other threads join for the finish
						//printf("Start %d Right %d End %d\n", triStart, triRight, triEnd);
					}
				}

				s_task[threadIdx.y].triRight = triRight;
				s_task[threadIdx.y].unfinished = unfinished;
				s_task[threadIdx.y].step = 0;
			}
		}

		s_sharedInt[threadIdx.y] = finished;
	}

	ASSERT_DIVERGENCE("taskFinishSortPPS1 bottom", tid);

	if(s_sharedInt[threadIdx.y] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_PPS1);
	}
	else if(s_sharedInt[threadIdx.y] == -100) // Finish the whole sort
	{
		taskFinishTask(tid, taskIdx);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

#/*el*/if SCAN_TYPE == 1

//------------------------------------------------------------------------

__device__ void taskFinishSortPPSUp(int tid, int taskIdx, int countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortPPSUp top", tid);

	int triStart = s_task[threadIdx.y].triStart;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(tid == 0)
	{
		int finished;
		if(s_task[threadIdx.y].lock == LockType_None)
			finished = countDown;
		else
			finished = atomicSub(&g_taskStackBVH.tasks[taskIdx].unfinished, countDown); // Lower the number of unfinished tasks

		if(finished == countDown) // Finished is the value before Dec, thus == countDown means last. We have finished the task and are responsible for cleaning up
		{
			int step = s_task[threadIdx.y].step;
			int unfinished = taskNextPhaseCountPPSUp(step, triStart, triEnd); // Returns 0 when no new phase is needed

			if(unfinished != 0) // Move to next phase of the current task
			{
				s_task[threadIdx.y].unfinished = unfinished;
				step += LOG_WARP_SIZE;
				s_task[threadIdx.y].step = step; // Increase the step
				finished = -1; // Make the other threads join for the finish
			}
			else // Move to PPS1_Down
			{
				((volatile int*)c_bvh_in.ppsTrisBuf)[triEnd - 1] = 0; // Set the last element to 0 as required by Harris scan

				s_task[threadIdx.y].type = TaskType_Sort_PPS1_Down;
				// Make the level multiple of LOG_WARP_SIZE to end at step 0
				int level = taskTopTreeLevel(triStart, triEnd);
				s_task[threadIdx.y].unfinished = 1; // Top levels for tris
				s_task[threadIdx.y].step = level;
				finished = -1; // Make the other threads join for the finish
			}
		}

		s_sharedInt[threadIdx.y] = finished;
	}

	ASSERT_DIVERGENCE("taskFinishSortPPSUp bottom", tid);

	if(s_sharedInt[threadIdx.y] == -1)
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

	if(tid == 0)
	{
		int finished;
		if(s_task[threadIdx.y].lock == LockType_None)
			finished = countDown;
		else
			finished = atomicSub(&g_taskStackBVH.tasks[taskIdx].unfinished, countDown); // Lower the number of unfinished tasks

		if(finished == countDown) // Finished is the value before Dec, thus == countDown means last. We have finished the task and are responsible for cleaning up
		{
			int step = s_task[threadIdx.y].step;
			int unfinished = taskNextPhaseCountPPSDown(step, triStart, triEnd); // Returns 0 when no new phase is needed

			if(unfinished != 0) // Move to next phase of the current task
			{
				s_task[threadIdx.y].unfinished = unfinished;
				step -= LOG_WARP_SIZE;
				s_task[threadIdx.y].step = step; // Increase the step
				finished = -1; // Make the other threads join for the finish
			}
			else // Move to SORT1
			{
				int triRight = triEnd - ((volatile int*)c_bvh_in.ppsTrisBuf)[triEnd - 1]; // Seems that must be volatile
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
					finished = -1; // Make the other threads join for the finish
				}

				/*if(triRight == triEnd)
				{
					printf("WTF\n");
					for(int i = triStart; i < triEnd; i++)
						printf("Pivot %d Pos %d = %d (class %d pps %d test %d)\n", s_task[threadIdx.y].pivot, i, ((volatile int*)c_bvh_in.trisIndex)[i], ((volatile int*)c_bvh_in.ppsTrisIndex)[i], ((volatile int*)c_bvh_in.ppsTrisBuf)[i], ((volatile int*)c_bvh_in.trisIndex)[i] >= s_task[threadIdx.y].pivot);
				}*/

				if(unfinished == 0) // Nothing to sort -> Move to Sort_PPS1
				{
					int pivotPos = triStart + (triEnd-triStart)/2;
					// Swap the lowest element to the beginning
					int temp = ((volatile int*)c_bvh_in.trisIndex)[triStart];
					((volatile int*)c_bvh_in.trisIndex)[triStart] = ((volatile int*)c_bvh_in.trisIndex)[pivotPos];
					((volatile int*)c_bvh_in.trisIndex)[pivotPos] = temp;
					//printf("Start %d Right %d End %d\n", triStart, triRight, triEnd);
					
					if(triEnd - triStart <= 2) // End computation
					{
						triRight = triStart+1;
						finished = -100;
					}
					else
					{
						triStart++; // Force split on unsubdivided task
						triRight = -1;
						s_task[threadIdx.y].triStart = triStart;
						s_task[threadIdx.y].type = TaskType_Sort_PPS1;
						s_task[threadIdx.y].pivot = ((volatile int*)c_bvh_in.trisIndex)[triStart + (triEnd-triStart)/2];
						unfinished = taskWarpSubtasksZero(triEnd - triStart);
						finished = -1; // Make the other threads join for the finish
						//printf("Start %d Right %d End %d\n", triStart, triRight, triEnd);
					}
				}

				s_task[threadIdx.y].triRight = triRight;
				s_task[threadIdx.y].unfinished = unfinished;
				s_task[threadIdx.y].step = 0;
				finished = -1; // Make the other threads join for the finish
			}
		}

		s_sharedInt[threadIdx.y] = finished;
	}

	ASSERT_DIVERGENCE("taskFinishSortPPSDown bottom", tid);

	if(s_sharedInt[threadIdx.y] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_PPS1_Down);
	}
	else if(s_sharedInt[threadIdx.y] == -100) // Finish the whole sort
	{
		taskFinishTask(tid, taskIdx);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

#endif

//------------------------------------------------------------------------

__device__ void taskFinishSortSORT1(int tid, int taskIdx, unsigned countDown)
{
	ASSERT_DIVERGENCE("taskFinishSortSORT1 top", tid);

	int triStart = s_task[threadIdx.y].triStart;
	int triRight = s_task[threadIdx.y].triRight;
	int triEnd = s_task[threadIdx.y].triEnd;

	if(tid == 0)
	{
		int finished;
		if(s_task[threadIdx.y].lock == LockType_None)
			finished = countDown;
		else
			finished = atomicSub(&g_taskStackBVH.tasks[taskIdx].unfinished, countDown); // Lower the number of unfinished tasks

		if(finished == countDown) // Finished is the value before Dec, thus == countDown means last. We have finished the task and are responsible for cleaning up
		{
			int step = s_task[threadIdx.y].step;
			int unfinished;

			if(step == 0) // Move to next phase of the current task
			{
				unfinished = taskWarpSubtasksZero(triRight - triStart);

				s_task[threadIdx.y].unfinished = unfinished;
				step++;
				s_task[threadIdx.y].step = step; // Increase the step
				finished = -1; // Make the other threads join for the finish
			}
			else // Move to PPS2
			{
#ifdef RAYTRI_TEST
				if(s_task[threadIdx.y].type != TaskType_RayTriTestSORT1)
				{
					unfinished = taskWarpSubtasks(triEnd - triStart);
					s_task[threadIdx.y].unfinished = unfinished;
					s_task[threadIdx.y].type = TaskType_RayTriTestSORT1;
					s_task[threadIdx.y].step = 0;
					finished = -1;
				}
				else
#endif

				{
				finished = -100; // Make the other threads join for the finish
				}
			}
		}

		s_sharedInt[threadIdx.y] = finished;
	}

	ASSERT_DIVERGENCE("taskFinishSortSORT1 bottom", tid);

	if(s_sharedInt[threadIdx.y] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_SORT1);
	}
	else if(s_sharedInt[threadIdx.y] == -100) // Finish the whole sort
	{
		taskFinishTask(tid, taskIdx);
	}
	else
	{
		s_task[threadIdx.y].lock = LockType_Free;
	}
}

//------------------------------------------------------------------------

// Finishes a sort task
__device__ void taskFinishWarpSort(int tid, int taskIdx, int leaf)
{
	if(tid == 0)
	{
		// We should update the dependencies before we start adding new tasks because these subtasks may be finished before this is done
		s_task[threadIdx.y].lock = LockType_Free;

#ifndef DEBUG_INFO
		atomicCAS(&g_taskStackBVH.top, taskIdx, max(taskIdx-1, 0)); // Try decreasing the stack top
#endif

#if PARALLELISM_TEST == 0
		atomicSub(&g_numActive, 1);
#endif

		// Decrease the waiting counters (increase the number of tasks the task is waiting on) for tasks dependent on this one
		// Update taskStack.unfinished
		atomicAdd(&g_taskStackBVH.unfinished, 1);

#ifndef DEBUG_INFO
		g_taskStackBVH.header[taskIdx] = TaskHeader_Empty; // Empty this task

#if ENQUEUE_TYPE == 1
		atomicMin(&g_taskStackBVH.bottom, taskIdx); // Update the stack bottom
#endif

#if ENQUEUE_TYPE == 3
		int pos = atomicInc(&g_taskStackBVH.emptyTop, EMPTY_MAX);  // Update the empty task
		g_taskStackBVH.empty[pos] = taskIdx;
#endif
#endif
	}
}

//------------------------------------------------------------------------

__device__ void classify(int tid, int subtask, int start, int end, int pivot)
{
	ASSERT_DIVERGENCE("classify", tid);

	int pos = start + subtask*WARP_SIZE + tid;
	if(pos < end)
	{
		int value = ((volatile int*)c_bvh_in.trisIndex)[pos];
		int flag = (value >= pivot) ? 1 : 0;
		((volatile int*)c_bvh_in.ppsTrisIndex)[pos] = flag;
	}
}

//------------------------------------------------------------------------

__device__ void warpSort(int tid, int start, int end)
{
	ASSERT_DIVERGENCE("warpSort", tid);
	
	int pos = start + tid;
	if(pos < end)
	{
		int value = ((volatile int*)c_bvh_in.trisIndex)[pos];
		s_sharedData[threadIdx.y][tid] = value;
	}

	transposition_sort<int>(s_sharedData[threadIdx.y], tid, end-start);
	if(pos < end)
	{
		((volatile int*)c_bvh_in.trisIndex)[pos] = s_sharedData[threadIdx.y][tid];
	}
}

//------------------------------------------------------------------------

#ifdef SNAPSHOT_POOL
// Constantly take snapshots of the pool
__device__ void snapshot(int tid, volatile int* header,	volatile TaskBVH* tasks, volatile int *unfinished,	volatile int *stackTop, volatile int* img, volatile int* red)
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

// Main function of the programme
extern "C" __global__ void __launch_bounds__(NUM_THREADS, NUM_BLOCKS_PER_SM) sort(void)
{
	volatile int* taskAddr = (volatile int*)(&s_task[threadIdx.y]);

	int subtasksDone = 1;
	int tid = threadIdx.x;
	int subtask;
	int taskIdx;
	int count;
	int popCount;

#ifdef SNAPSHOT_POOL
	int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
	if(warpIdx == 0)
	{
		snapshot(tid, g_taskStackBVH.header, g_taskStackBVH.tasks, &g_taskStackBVH.unfinished, &g_taskStackBVH.top, (volatile int*)&s_task[threadIdx.y], (volatile int*)s_sharedData[threadIdx.y]);
		return;
	}
#endif

	/*if(tid == 0)
	{
		for(int i = 0; i < 56; i++)
		{
			atomicAdd(&g_taskStackBVH.top, 1);
			__threadfence();
		}
	}
	return;*/

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

	/*if(blockDim.y*blockIdx.x + threadIdx.y + tid == 0)
	{
		printf("task: %dB\n", sizeof(task));
		printf("Int %dB\n", sizeof(int));
		printf("Float %dB\n", sizeof(float));
		printf("Float3 %dB\n", sizeof(float3));
		printf("Float4 %dB\n", sizeof(float4));
		printf("CudaAABB %dB\n", sizeof(CudaAABB));
		printf("Shared tasks %dB\n", sizeof(s_task));
		printf("GMEM task %dB\n", sizeof(g_taskStackBVH.tasks[0]));
		printf("task stack %dB\n", sizeof(taskStack));
	}
	return;*/

	while(s_task[threadIdx.y].lock != LockType_Free || taskDequeue(tid, subtask, taskIdx, popCount)) // Main loop of the programme, while there is some task to do
	{
#ifdef SNAPSHOT_WARP
		WarpInfo info;
		s_task[threadIdx.y].dequeued = 0;
#endif
		if(s_task[threadIdx.y].lock == LockType_Free)
		{
		// Copy first cache line of task to shared memory
#ifdef DIVERGENCE_TEST
			if(taskIdx >= 0 && taskIdx < g_taskStackBVH.size)
			{
				//taskLoadFirstFromGMEM(tid, taskIdx, &s_task[threadIdx.y]);
				volatile TaskBVH* g_task = &g_taskStackBVH.tasks[taskIdx];
				taskAddr[tid] = ((volatile int*)g_task)[tid]; // Every thread copies one word of task data
#ifdef DEBUG_INFO
				int offset = 128/sizeof(int); // 128B offset
				taskAddr[tid+offset] = ((volatile int*)g_task)[tid+offset]; // Every thread copies one word of task data
#endif
				
				count = subtask;
				s_task[threadIdx.y].poped = popCount;// If we have poped all of the task's work we do not have to update unfinished atomicaly
				if(subtask == s_task[threadIdx.y].origSize-1 && popCount >= s_task[threadIdx.y].origSize)
					s_task[threadIdx.y].lock = LockType_None;
			}
			else
			{
				printf("Fetched task %d out of range!\n", taskIdx);
				g_taskStackBVH.unfinished = 1;
			}
#else
			//taskLoadFirstFromGMEM(tid, taskIdx, &s_task[threadIdx.y]);
			volatile TaskBVH* g_task = &g_taskStackBVH.tasks[taskIdx];
			taskAddr[tid] = ((volatile int*)g_task)[tid]; // Every thread copies one word of task data
#ifdef DEBUG_INFO
			int offset = 128/sizeof(int); // 128B offset
			taskAddr[tid+offset] = ((volatile int*)g_task)[tid+offset]; // Every thread copies one word of task data
#endif
			count = subtask;
			s_task[threadIdx.y].popCount = popCount;
			// If we have poped all of the task's work we do not have to update unfinished atomicaly
			if(subtask == s_task[threadIdx.y].origSize-1 && popCount >= s_task[threadIdx.y].origSize)
				s_task[threadIdx.y].lock = LockType_None;
#endif

#ifdef SNAPSHOT_WARP
			s_task[threadIdx.y].dequeued = 1;
			// Write out information about this dequeue
			info.reads = s_sharedData[threadIdx.y][3];
			info.tris = s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triStart;
			info.type = s_task[threadIdx.y].type;
			info.chunks = s_task[threadIdx.y].origSize;
			info.popCount = s_task[threadIdx.y].poped;
			info.depth = s_task[threadIdx.y].depth;
			info.idx = s_task[threadIdx.y].nodeIdx;
			info.stackTop = *((volatile int*)&g_taskStackBVH.top);
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
			//	printf("task taskIdx: %d\n", taskIdx);
			//	printf("Global unfinished: %d\n", g_taskStackBVH.unfinished);
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
//#if SCAN_TYPE == 0
		case TaskType_Sort_PPS1:
			// run pss for 1's
			do {
				if(s_task[threadIdx.y].step == 0) // Classify all triangles against the best plane
					classify(tid, subtask, s_task[threadIdx.y].triStart, s_task[threadIdx.y].triEnd, s_task[threadIdx.y].pivot);
				
				pps<int>(tid, subtask, s_task[threadIdx.y].step, c_bvh_in.ppsTrisIndex, c_bvh_in.ppsTrisBuf, s_sharedData[threadIdx.y], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triEnd, 1);
				subtasksDone = taskReduceSubtask(subtask, count, popCount);
			} while(subtasksDone == -1);

			__threadfence(); // Probably needed so that next iteration does not read uninitialized data
			taskFinishSortPPS1(tid, taskIdx, subtasksDone);
			break;
#/*el*/if SCAN_TYPE == 1
			case TaskType_Sort_PPS1_Up:
			// run pss for 1's
			do {
#ifndef DEBUG_PPS
				if(s_task[threadIdx.y].step == 0) // Classify all triangles against the best plane
					classify(tid, subtask, s_task[threadIdx.y].triStart, s_task[threadIdx.y].triEnd, s_task[threadIdx.y].pivot);
#endif
				
				scanUp<int>(tid, subtask, s_task[threadIdx.y].step, (volatile int*)c_bvh_in.ppsTrisBuf, (volatile int*)c_bvh_in.ppsTrisIndex, s_sharedData[threadIdx.y], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triEnd, 1, plus, 0);
				subtasksDone = taskReduceSubtask(subtask, count, popCount);
			} while(subtasksDone == -1);

			__threadfence(); // Probably needed so that next iteration does not read uninitialized data
			taskFinishSortPPSUp(tid, taskIdx, subtasksDone);
			break;

			case TaskType_Sort_PPS1_Down:
			// run pss for 1's
			do {
				scanDown<int>(tid, subtask, s_task[threadIdx.y].step, (volatile int*)c_bvh_in.ppsTrisBuf, (volatile int*)c_bvh_in.ppsTrisIndex, s_sharedData[threadIdx.y], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triEnd, 1, plus, 0);
				subtasksDone = taskReduceSubtask(subtask, count, popCount);
			} while(subtasksDone == -1);

			__threadfence(); // Probably needed so that next iteration does not read uninitialized data
			taskFinishSortPPSDown(tid, taskIdx, subtasksDone);
			break;
#endif

		case TaskType_Sort_SORT1:
			// sort -1,0 to the left and 1 to the right
			do {
				sort(tid, subtask, s_task[threadIdx.y].step, c_bvh_in.ppsTrisIndex, c_bvh_in.ppsTrisBuf, c_bvh_in.sortTris, c_bvh_in.trisIndex, s_task[threadIdx.y].triStart, s_task[threadIdx.y].triEnd, s_task[threadIdx.y].triRight, 1, false);
				subtasksDone = taskReduceSubtask(subtask, count, popCount);
			} while(subtasksDone == -1);

			__threadfence(); // Probably needed so that next iteration does not read uninitialized data
			taskFinishSortSORT1(tid, taskIdx, subtasksDone);
			break;

		case TaskType_Intersect:
			do {
				// Sort the less than 32 values in one run
				warpSort(tid, s_task[threadIdx.y].triStart, s_task[threadIdx.y].triEnd);
				subtasksDone = taskReduceSubtask(subtask, count, popCount);
			} while(subtasksDone == -1);

			//__threadfence(); // Probably needed so that next iteration does not read uninitialized data
			taskFinishWarpSort(tid, taskIdx, subtasksDone);
			break;
		}

#ifdef SNAPSHOT_WARP
		if(s_task[threadIdx.y].dequeued == 1 && numSteps[threadIdx.y] - 1 < SNAPSHOT_WARP)
		{
			info.clockFinished = clock64();

			if(numSteps[threadIdx.y] <= SNAPSHOT_WARP)
			{
				// Save the snapshot
				int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
				int *g_ptr = (int*)&g_snapshots[warpIdx*SNAPSHOT_WARP + numSteps[threadIdx.y] - 1];
				int *ptr = (int*)&info;
				if(tid < sizeof(WarpInfo)/sizeof(int))
					g_ptr[tid] = ptr[tid];
			}
		}
#endif

		// Last finished had work for only one warp
		if(s_task[threadIdx.y].lock == LockType_None || s_task[threadIdx.y].lock == LockType_Subtask)
		{
			// Convert to the multiple subtask solution
			//subtask = s_task[threadIdx.y].unfinished-1;
			subtask = s_task[threadIdx.y].origSize-1;
			count = subtask;
			popCount = s_task[threadIdx.y].popCount;
			//popCount = c_env.popCount;
		}

		ASSERT_DIVERGENCE("taskProcessWorkUntilDone bottom", tid);
	}

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT) || defined(COUNT_STEPS_DEQUEUE)
	// Write out work statistics
	if(tid == 0)
	{
		int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
		((float4*)c_bvh_in.debug)[warpIdx] = make_float4(numSteps[threadIdx.y]*1.0f, sumSteps[threadIdx.y]*1.0f/numSteps[threadIdx.y], maxSteps[threadIdx.y]*1.0f, numRestarts[threadIdx.y]*1.0f);
	}
#endif
}

//------------------------------------------------------------------------

// Compare performance of various caching operation in loading and saving to global memory
extern "C" __global__ void __launch_bounds__(NUM_THREADS, NUM_BLOCKS_PER_SM) testCacheOps(void)
{
	int* g_mem = g_taskStackBVH.header;
	volatile int* g_memV = g_taskStackBVH.header;
	//int idx = threadIdx.x + threadIdx.y * WARP_SIZE + blockIdx.x * blockDim.x;
	int idx = threadIdx.y * WARP_SIZE + blockIdx.x * blockDim.x;
	//int idx = threadIdx.x;
	//int idx = 0;

	// It seems that all these methods are the same, caching in L2
	// The only difference is whether the compiler is smart enough to optimize out some reads or writes
	
	// Writing to the same locations slows down the computation
	// Moderately for all threads in a warp writing to the same location (in warp collision resolving?)
	// Significantly for all threads writing to the same location (partition camping? broadcast? in warp collision resolving?)
	// Brutally for same threads in different warps writing to the same location (partition camping?)

	// Writing to the same locations with atomic operation slows down the computation
	// Is on par with nonatomic write (no conflicts occur)
	// Moderately for same threads in different warps writing to the same location (15 conflicting writes per iteration)
	// Significantly for all threads in a warp writing to the same location (32 conflicting writes per iteration)
	// Brutally for all threads writing to the same location (300 conflicting writes per iteration)

	for(int i = 0; i < 10000; i++)
	{
		//int t = g_mem[idx]; // Every thread copies one word of task data
		int t = g_memV[idx]; // Every thread copies one word of task data
		//int t = loadCG(&g_mem[idx]); // Every thread copies one word of task data
		//int t = loadCS(&g_mem[idx]); // Every thread copies one word of task data

		//g_mem[idx] = t+1;
		g_memV[idx] = t+1;
		//saveCG(&g_mem[idx], t+1);
		//atomicAdd(&g_mem[idx], 1);
	}

	g_taskStackBVH.unfinished = 1;
}

//------------------------------------------------------------------------

// Test influence of partition camping on the memory load and store performance
extern "C" __global__ void __launch_bounds__(NUM_THREADS, NUM_BLOCKS_PER_SM) testMemoryCamping(void)
{
	int* g_mem = g_taskStackBVH.header;
	volatile int* g_memV = g_taskStackBVH.header;
	//int idx = threadIdx.x + threadIdx.y * WARP_SIZE + blockIdx.x * blockDim.x;
	int idx = threadIdx.x + (threadIdx.y * WARP_SIZE + blockIdx.x * blockDim.x)*2;

	// Skipping one cache line causes partition camping

	for(int i = 0; i < 1000000; i++)
	{
		//int t = g_mem[idx]; // Every thread copies one word of task data
		int t = g_memV[idx]; // Every thread copies one word of task data
		//int t = loadCG(&g_mem[idx]); // Every thread copies one word of task data
		//int t = loadCS(&g_mem[idx]); // Every thread copies one word of task data

		//g_mem[idx] = t+1;
		g_memV[idx] = t+1;
		//saveCG(&g_mem[idx], t+1);
		//atomicAdd(&g_mem[idx], 1);
	}

	g_taskStackBVH.unfinished = 1;
}

//------------------------------------------------------------------------

// Test the new shuffle instructions in a odd-even merge sort
extern "C" __global__ void __launch_bounds__(NUM_THREADS, NUM_BLOCKS_PER_SM) testKeplerSort(void)
{
	int tid = threadIdx.x;
	int key = blockDim.x-1 - tid;
	int value = blockDim.x-1 - tid;
	int segmentID = -1;

	if(tid < 5)
		segmentID = 4;
	else if(tid >= 5 && tid < 10)
		segmentID = 5;
	if(tid >= 10 && tid < 11)
		segmentID = 6;
	if(tid >= 11 && tid < 30)
		segmentID = 7;

	if(threadIdx.y * WARP_SIZE + blockIdx.x * blockDim.x == 0) // First warp
	{
		printf("Unsorted tid %d key %d value %d segment %d\n", tid, key, value, segmentID);

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)
		sortWarpSegmented(tid, key, value, segmentID, 30);
#endif

		printf("Sorted tid %d key %d value %d segment %d\n", tid, key, value, segmentID);
	}
}
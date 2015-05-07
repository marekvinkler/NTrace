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
#include "CudaPool.hpp"
#include "warp_common.cu"

//------------------------------------------------------------------------

#ifdef MALLOC_SCRATCHPAD

// Allocate memory for the root node
__global__ void allocFreeableMemory(int numTris, int numRays)
{
	// Save the base pointer (hopefully) to the heap

#if (MALLOC_TYPE == CUDA_MALLOC)
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
#endif

//------------------------------------------------------------------------

// Return the number of warp sized subtasks but at least 1 task
__device__ __forceinline__ int taskWarpSubtasks(int threads)
{
	return max((threads + WARP_SIZE - 1) / WARP_SIZE, 1); // Do not create empty tasks - at least on warp gets to clean this task
}

//------------------------------------------------------------------------

// Return the number of warp sized subtasks
__device__ __forceinline__ int taskWarpSubtasksZero(int threads)
{
	return (threads + WARP_SIZE - 1) / WARP_SIZE;
}

//------------------------------------------------------------------------

// Computes the number of warps needed for the naive PPS
__device__ __forceinline__ int taskNextPhaseCountPPS(int step, int start, int end)
{
	int all = end - start;
	int allTasks = taskWarpSubtasks(all);

	if(all > (1 << (LOG_WARP_SIZE+step)))
		return allTasks >> 1;
	else
		return 0;
}

//------------------------------------------------------------------------

// Computes the number of levels in the tree defined by interval [start, end) and ceils the value to the multiple of LOG_WARP_SIZE
__device__ __forceinline__ int taskTopTreeLevel(int start, int end)
{
	if(end - start > 1)
	{
		return (((int)__log2f(end - start - 1))/LOG_WARP_SIZE) * LOG_WARP_SIZE;
	}
	else
	{
		return 0;
	}
}

//------------------------------------------------------------------------

// Computes the number of warps needed for a specific level of a hierarchical algorithm
__device__ __forceinline__ int taskNextPhaseCountTree(int level, int start, int end)
{
	int all = end - start;

	if(level >= 0 && all > ((level == 0) ? 0 : (1 << level)))
	{
		int nodes = ((all-1) >> level)+1; // Threads needed
		return taskWarpSubtasks(nodes);
		//return taskWarpSubtasksZero(nodes); // OPTIMIZE: Does this work as well?
	}
	else
	{
		return 0;
	}
}

//------------------------------------------------------------------------

// Computes the number of warps needed for the PPS up phase
__device__ __forceinline__ int taskNextPhaseCountPPSUp(int step, int start, int end)
{
	// return taskNextPhaseCountTree(step+6, start, end); // +6 for work done by single warp
	return taskNextPhaseCountTree(step+LOG_WARP_SIZE, start, end); // +LOG_WARP_SIZE for work done by single warp
}

//------------------------------------------------------------------------

// Computes the number of warps needed for the PPS down phase
//__device__ __forceinline__ int taskNextPhaseCountPPSDown(int step, int start, int end)
//{
//	int all = end - start;
//
//	if(step-LOG_WARP_SIZE >= 0) // -LOG_WARP_SIZE for work done by single warp
//	{
//		int threads = ((all-1) >> (step-LOG_WARP_SIZE))+1; // Threads needed
//		return taskWarpSubtasks(threads);
//		//return taskWarpSubtasksZero(nodes); // OPTIMIZE: Does this work as well?
//	}
//	else
//	{
//		return 0;
//	}
//}

__device__ __forceinline__ int taskNextPhaseCountPPSDown(int step, int start, int end)
{
	return taskNextPhaseCountTree(step-LOG_WARP_SIZE, start, end); // -LOG_WARP_SIZE for work done by single warp
}

//------------------------------------------------------------------------

// Computes the number of warps needed for the reduction level by level
__device__ __forceinline__ int taskNextPhaseCountReduce(int step, int start, int end)
{
	int all = end - start;

	int threads = ((all-1) >> (step+LOG_WARP_SIZE+1))+1; // Threads needed
	if(all > (1 << (LOG_WARP_SIZE+step))) // +LOG_WARP_SIZE for work done by single warp
		return taskWarpSubtasks(threads);
	else
		return 0;
}

//------------------------------------------------------------------------

// Computes the number of warps needed for the reduction multiplied by a constant
__device__ __forceinline__ int taskNextPhaseCountReduceMultiplied(int step, int start, int end, int multiply)
{
	int all = end - start;

	int threads = ((all-1) >> (step+LOG_WARP_SIZE+1))+1; // Threads needed
	if(all > (1 << (LOG_WARP_SIZE+step))) // +LOG_WARP_SIZE for work done by single warp
		return taskWarpSubtasks(threads*multiply);
	else
		return 0;
}

//------------------------------------------------------------------------

// Computes the number of warps needed for the reduction
__device__ __forceinline__ int taskNextPhaseCountReduceBlock(int step, int start, int end)
{
	return taskNextPhaseCountTree(step+LOG_WARP_SIZE, start, end); // +LOG_WARP_SIZE for work done by single warp
}

//------------------------------------------------------------------------

// Converts float to an orderable int
__device__ __forceinline__ int floatToOrderedInt(float floatVal)
{
	int intVal = __float_as_int(floatVal);
	return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

//------------------------------------------------------------------------

// Converts an orderable int back to a float
__device__ __forceinline__ float orderedIntToFloat(int intVal)
{
	return __int_as_float((intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF);
}

//------------------------------------------------------------------------

// Atomic minimum on floats
__device__ __forceinline__ float atomicMinFloat(int* address, float value)
{
	return atomicMin(address, floatToOrderedInt(value));
}

//------------------------------------------------------------------------

// Atomic maximum on floats
__device__ __forceinline__ float atomicMaxFloat(int* address, float value)
{
	return atomicMax(address, floatToOrderedInt(value));
}

/*__device__ __forceinline__ unsigned int floatToOrderedInt(float floatVal)
{
	unsigned int f = __float_as_int(floatVal);
	unsigned int mask = -(int)(f >> 31) | 0x80000000;
	return f ^ mask;
}

//------------------------------------------------------------------------

__device__ __forceinline__ float orderedIntToFloat(unsigned int intVal)
{
	unsigned int mask = ((intVal >> 31) - 1) | 0x80000000;
	return __int_as_float(intVal ^ mask);
}

//------------------------------------------------------------------------

__device__ __forceinline__ float atomicMin(float* address, float value)
{
	return atomicMin((unsigned int*)address, floatToOrderedInt(value));
}

//------------------------------------------------------------------------

__device__ __forceinline__ float atomicMax(float* address, float value)
{
	return atomicMax((unsigned int*)address, floatToOrderedInt(value));
}*/

//------------------------------------------------------------------------

// Adds a task into a global task queue in place of an empty task with position lower or equal of taskIdx and returns its position in taskIdx
__device__ void taskEnqueueRight(int tid, int *head, volatile int* sharedData, int taskStatus, int& beg, int end)
{
	ASSERT_DIVERGENCE("taskEnqueueRight top", tid);

	int* header = head;
	int status = TaskHeader_Locked;
	//beg = beg - tid; // From single location to parallel
	beg = (taskWarpSubtasks(beg)*WARP_SIZE) - tid; // From single location to parallel

	while(status == TaskHeader_Locked) // while we have not found an empty task
	{
		// Find first empty task, end if we have reached start of the array
#ifndef PTX_LOAD_CACHE
		while(__any(beg >= end) && __all(beg < 0 || (status = header[beg]) > TaskHeader_Empty))
#else
		while(__any(beg >= end) && __all(beg < 0 || (status = loadCG(&head[beg])) > TaskHeader_Empty))
#endif
		{
			beg -= WARP_SIZE;
#ifdef COUNT_STEPS_RIGHT
			numReads[threadIdx.y]++;
#endif
		}

		if(__all(status > TaskHeader_Empty)) // We have found no empty task down from taskIdx
		{
			sharedData[WARP_SIZE-1] = -1;
#ifdef COUNT_STEPS_RIGHT
			numRestarts[threadIdx.y]++;
#endif
			break;
		}

		// Try all tasks that might be empty
		sharedData[WARP_SIZE-1] = TaskHeader_Locked;
		while(sharedData[WARP_SIZE-1] == TaskHeader_Locked && __any(status <= TaskHeader_Empty)) // We have not locked a task yet and there is still some to try
		{
			ASSERT_DIVERGENCE("taskEnqueueRight loop", tid);

			// Choose some with active task
			if(status <= TaskHeader_Empty)
				sharedData[WARP_SIZE-2] = tid;

			if(sharedData[WARP_SIZE-2] == tid)
			{
				// Try acquire the current task - lock it
				if(atomicCAS(&head[beg], status, TaskHeader_Locked) != status) // Try to lock and return the current value
				{
					status = TaskHeader_Locked; // Let other threads try
				}
				else
				{
					sharedData[WARP_SIZE-1] = beg;
				}
			}
		}
		ASSERT_DIVERGENCE("taskEnqueueRight loopEnd", tid);
		status = sharedData[WARP_SIZE-1]; // Next run?

		if(status == TaskHeader_Locked) // We have not succeeded
		{
			// Move to next task
			beg -= WARP_SIZE;
			// OPTIMIZE: we shall move beg -= WARP_SIZE; as the first statement in the outer while and start with g_taskStack.top+1.
		}
	}
	beg = sharedData[WARP_SIZE-1]; // Distribute the locked task position

	//ASSERT_DIVERGENCE("taskEnqueueRight bottom", tid); // There is divergence here but it is solved by the reconvergence point at the end of the __noinline__ function
}

//------------------------------------------------------------------------

// Adds a task into a global task queue in place of an empty task with position lower or equal of taskIdx and returns its position in taskIdx
__device__ void taskEnqueueLeft(int tid, int *head, volatile int* sharedData, int taskStatus, int& beg, int* unfinished, int size)
{
	ASSERT_DIVERGENCE("taskEnqueueLeft top", tid);

	int* header = head;
	int status = TaskHeader_Locked;
	//beg = beg + (WARP_SIZE-1) - tid; // From single location to parallel.
	beg = ((taskWarpSubtasks(beg)-1)*WARP_SIZE) + (WARP_SIZE-1) - tid; // From single location to parallel.
	// Reverse order because with single bank write the last thread updates the position.

	while(status == TaskHeader_Locked) // while we have not found an empty task
	{
		// Find first empty task, end if we have reached start of the array
#ifndef PTX_LOAD_CACHE
		while(__any(beg < size) && __all(beg >= size || (status = header[beg]) > TaskHeader_Empty))
#else
		while(__any(beg < size) && __all(beg >= size || (status = loadCG(&head[beg])) > TaskHeader_Empty))
#endif
		{
			beg += WARP_SIZE;
#ifdef COUNT_STEPS_LEFT
			numReads[threadIdx.y]++;
#endif
		}

		if(__all(status > TaskHeader_Empty)) // Should never succeed
		{
			sharedData[WARP_SIZE-1] = -1;
			//printf("Pool overflow\n");
			*unfinished = 1; // Ends the algorithm
			break;
		}

		// Try all tasks that might be empty
		sharedData[WARP_SIZE-1] = TaskHeader_Locked;
		while(sharedData[WARP_SIZE-1] == TaskHeader_Locked && __any(status <= TaskHeader_Empty)) // We have not locked a task yet and there is still some to try
		{
			ASSERT_DIVERGENCE("taskEnqueueLeft loop", tid);

			// Choose some with active task
			if(status <= TaskHeader_Empty)
				sharedData[WARP_SIZE-2] = tid;

			if(sharedData[WARP_SIZE-2] == tid)
			{
				// Try acquire the current task - lock it
				if(atomicCAS(&head[beg], status, TaskHeader_Locked) != status) // Try to lock and return the current value
				{
					status = TaskHeader_Locked; // Let other threads try
				}
				else
				{
					sharedData[WARP_SIZE-1] = beg;
				}
			}
		}
		ASSERT_DIVERGENCE("taskEnqueueLeft loopEnd", tid);
		status = sharedData[WARP_SIZE-1]; // Next run?

		if(status == TaskHeader_Locked) // We have not succeeded
		{
			// Move to next task
			beg += WARP_SIZE;
			// OPTIMIZE: we shall move beg -= WARP_SIZE; as the first statement in the outer while and start with g_taskStack.top+1.
		}
	}
	beg = sharedData[WARP_SIZE-1]; // Distribute the locked task position

	//ASSERT_DIVERGENCE("taskEnqueueLeft bottom", tid); // There is divergence here but it is solved by the reconvergence point at the end of the __noinline__ function
}

//------------------------------------------------------------------------

// Adds a task into a global task queue in place of an empty task
__device__ void taskEnqueueCache(int tid, TaskStackBase* task, volatile int* s_sharedData, int& status, int& pos, int& beg, int& top)
{
	ASSERT_DIVERGENCE("taskEnqueueCache top", tid);
	
	int* empty = task->empty;

	if(tid == 0)
	{
		//pos = atomicDec(&task->emptyTop, EMPTY_MAX);  // Update the empty task
		int pop = atomicInc(&task->emptyBottom, EMPTY_MAX);  // Update the empty task

		int beg = empty[pop]; // UNSAFE: May read from a position behind top
		int status = atomicCAS(&task->header[beg], TaskHeader_Empty, TaskHeader_Locked); // UNSAFE: Does not take into account tasks < TaskHeader_Empty

		s_sharedData[WARP_SIZE-1] = status; // Distribute the inserted task status
		s_sharedData[WARP_SIZE-2] = pop; // Distribute the inserted task cache position
		s_sharedData[WARP_SIZE-3] = beg; // Distribute the inserted task pool position
	}

	status = s_sharedData[WARP_SIZE-1];
	pos = s_sharedData[WARP_SIZE-2];
	beg = s_sharedData[WARP_SIZE-3];

	//ASSERT_DIVERGENCE("taskEnqueueCache bottom", tid); // There is divergence here but it is solved by the reconvergence point at the end of the __noinline__ function
}

// Adds a task into a global task queue in place of an empty task
//__device__ bool taskEnqueueCache(int tid, TaskStackBase* task, volatile int* s_sharedData, int& status, int& pos, int& beg, int& top)
//{
//	ASSERT_DIVERGENCE("taskEnqueueCache top", tid);
//	
//	int* empty = task->empty;
//	unsigned int *emptyTop = &g_taskStackBVH.emptyTop;
//	unsigned int *emptyBottom = &g_taskStackBVH.emptyBottom;
//
//	if(tid == 0)
//	{
//		//pos = atomicDec(&task->emptyTop, EMPTY_MAX);  // Update the empty task
//		int pop = atomicInc(&task->emptyBottom, EMPTY_MAX);  // Update the empty task
//#ifdef COUNT_STEPS_CACHE
//			numReads[threadIdx.y]++;
//#endif
//
//		if((pop < pos && (pos <= top || top <= pop)) || (pos <= top && top <= pop)) // We got behind top
//		{
//			atomicDec(&task->emptyBottom, EMPTY_MAX); // Get back
//			s_sharedData[threadIdx.y][WARP_SIZE-1] = TaskHeader_Locked; // Distribute the inserted task status
//			s_sharedData[threadIdx.y][WARP_SIZE-2] = pop; // Distribute the inserted task cache position
//
//			int tmp = *emptyTop;
//			s_sharedData[threadIdx.y][WARP_SIZE-4] = tmp;
//			if(top != tmp)
//			{
//				s_sharedData[threadIdx.y][WARP_SIZE-2] = *emptyBottom;
//				s_sharedData[threadIdx.y][WARP_SIZE-5] = 1;
//			}
//			else
//			{
//				s_sharedData[threadIdx.y][WARP_SIZE-5] = 0;
//			}
//		}
//		else
//		{
//			int beg = empty[pop]; // UNSAFE: May read from a position behind top
//			int status = atomicCAS(&task->header[beg], TaskHeader_Empty, TaskHeader_Locked); // UNSAFE: Does not take into account tasks < TaskHeader_Empty
//		
//			s_sharedData[threadIdx.y][WARP_SIZE-1] = status; // Distribute the inserted task status
//			s_sharedData[threadIdx.y][WARP_SIZE-2] = pop; // Distribute the inserted task cache position
//			s_sharedData[threadIdx.y][WARP_SIZE-3] = beg; // Distribute the inserted task pool position
//			s_sharedData[threadIdx.y][WARP_SIZE-4] = top;
//			s_sharedData[threadIdx.y][WARP_SIZE-5] = 1;
//		}
//	}
//
//	status = s_sharedData[threadIdx.y][WARP_SIZE-1];
//	pos = s_sharedData[threadIdx.y][WARP_SIZE-2];
//	beg = s_sharedData[threadIdx.y][WARP_SIZE-3];
//	top = s_sharedData[threadIdx.y][WARP_SIZE-4];
//
//	return s_sharedData[threadIdx.y][WARP_SIZE-5];
//
//	//ASSERT_DIVERGENCE("taskEnqueueCache bottom", tid); // There is divergence here but it is solved by the reconvergence point at the end of the __noinline__ function
//}

//------------------------------------------------------------------------

// Caches an item in the active cache
__device__ __forceinline__ void taskCacheActive(int taskIdx, int* activeCache, unsigned int* activeCacheTop)
{
#if CACHE_TYPE == 0
	int pos = atomicInc(activeCacheTop, ACTIVE_MAX);
	activeCache[pos] = taskIdx;
#elif CACHE_TYPE == 1
	int status = TaskHeader_Active;
	int count = 0;
	while(status >= TaskHeader_Active && count <= ACTIVE_MAX)
	{
		int pos = atomicInc(activeCacheTop, ACTIVE_MAX);
		status = atomicCAS(&activeCache[pos], TaskHeader_Locked, taskIdx);
		count++;
	}

	/*if(count == ACTIVE_MAX)
	{
		printf("Failed to cache task %d\n", taskIdx);
	}*/
#else
#error Unsupported cache type!
#endif
}

//------------------------------------------------------------------------

// Uncaches an item in the active cache
__device__ __forceinline__ void taskUncacheActive(int tid, int taskIdx, int* activeCache, unsigned int* activeCacheTop)
{
#if CACHE_TYPE == 1
	//int cleared = 0;
	//int pos = -1;
	for(int i = 0; i < (ACTIVE_MAX+1)/WARP_SIZE; i++)
	{
#if 1
		int status = activeCache[i*WARP_SIZE + tid];
		if(status == taskIdx) // Remove the task from the cache
		{
			activeCache[i*WARP_SIZE + tid] = TaskHeader_Locked;
			//atomicCAS(&activeCache[i*WARP_SIZE + tid], taskIdx, TaskHeader_Locked);
			//cleared++;
			//pos = i*WARP_SIZE + tid;
		}
#else
		int status = atomicCAS(&activeCache[i*WARP_SIZE + tid], taskIdx, TaskHeader_Locked);
		/*if(status == taskIdx) // Remove the task from the cache
		{
			cleared++;
			pos = i*WARP_SIZE + tid;
		}*/
#endif

		if(__any(status == taskIdx))
			break;
	}
	
	/*reduceWarp(cleared, plus);
	if(cleared != 1)
	{
		if(tid == 0)
			printf("Cleared task %d: %d\n", taskIdx, cleared);
		if(pos != -1)
			printf("\t%d: on pos %d\n", taskIdx, pos);
	}*/
#endif
}

//------------------------------------------------------------------------

// Caches an item in the empty cache
__device__ __forceinline__ void taskCacheEmpty(int taskIdx, int* emptyCache, unsigned int* emptyCacheTop)
{
	int pos = atomicInc(emptyCacheTop, EMPTY_MAX);  // Update the empty task
	emptyCache[pos] = taskIdx;
}

//------------------------------------------------------------------------

// Checks whether there is more work chunks to process
__device__ __forceinline__ int taskReduceSubtask(int &subtask, const int& count, const int& popCount)
{
	subtask--;
	int subtasksDone = count-subtask;

	if( subtask == -1 || subtasksDone == popCount )
	{
		return subtasksDone;
	}
	else
	{
		return -1;
	}
}

//------------------------------------------------------------------------

// Naive inclusive prefix scan
template<typename T>
__device__ void pps(int tid, int subtask, int step, T* dataSrc, T* dataBuff, volatile T* red, unsigned int start, unsigned int end, int test)
{
	ASSERT_DIVERGENCE("pps", tid);

	if(step == 0)
	{
		// Do local, step 0
		unsigned int posInArr  = start + (subtask * WARP_SIZE) + tid;
		unsigned int posInWarp = tid;

		if(posInArr < end)
		{
			T val = dataSrc[posInArr] >= test;
			red[posInWarp] = val;

			// Local scan with shared memory
			if (posInWarp >=  1) { val = red[posInWarp -  1] + val; } red[posInWarp] = val;
			if (posInWarp >=  2) { val = red[posInWarp -  2] + val; } red[posInWarp] = val;
			if (posInWarp >=  4) { val = red[posInWarp -  4] + val; } red[posInWarp] = val;
			if (posInWarp >=  8) { val = red[posInWarp -  8] + val; } red[posInWarp] = val;
			if (posInWarp >= 16) { val = red[posInWarp - 16] + val; } red[posInWarp] = val;

#ifdef RAYTRI_TEST
			if(val < 0 || val > end-start)
			{
				printf("Warp pps failure rayidx %d: value %d\n", posInArr, val);
				//g_taskStackBVH.unfinished = 1;
			}
#endif

			dataBuff[posInArr] = val;
		}
	}
	else
	{
		// do global, step 1 to n
		unsigned int blockSize  = WARP_SIZE * (1 << step); // base is 64 (32 threads will get a value from neighboring 32), doubled every step
		unsigned int halfBlock  = blockSize >> 1; // half it
		unsigned int blockStart = start + blockSize * (subtask >> (step - 1)); // warps will merge every step to a bigger base block
		unsigned int inBlockOffset = WARP_SIZE * (subtask & ((1 << (step - 1)) - 1)) + tid; // dtto
		unsigned int posInArr = blockStart + halfBlock + inBlockOffset;

		if(posInArr < end)
		{
			T rhs = dataBuff[posInArr];
			T lhs = dataBuff[blockStart + halfBlock - 1];
			T val = lhs + rhs;

#ifdef RAYTRI_TEST
			if(val < 0 || val > end-start)
			{
				printf("Merge pps failure start %d, end %d, lhs %d (pos %d), rhs %d (float %f pos %d): value %d, test %d\n", start, end, lhs, blockStart + halfBlock - 1, rhs, *(float*)&rhs, posInArr, val, lhs+rhs);
				//g_taskStackBVH.unfinished = 1;
			}
#endif

			dataBuff[posInArr] = val;
		}
	}

	//__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	// Possibly the problem only manifests when global synchronization is skipped
}

//------------------------------------------------------------------------

// Inclusive scan by Harris: up-sweep phase
template<typename T>
__device__ void scanUp(int tid, int subtask, int step, T* data, T* cls, volatile T* red, int start, int end, T test, T(*op)(T,T), T identity)
{
	int i          = (subtask * WARP_SIZE) + tid;
	int blockSize  = (1 << step);
	int blockEnd   = start + blockSize * (i+1) - 1;
	int pos        = blockEnd;

	red[tid] = identity; // Save identities so that we do not work with uninitialized data

	if(pos < end) // OPTIMIZE: Can these ifs be merged?
	{
		red[tid] = (step == 0) ? (cls[pos] >= test) : data[pos];
	}
	reduceWarpTree(tid, &red[0], op);

	if(pos < end)
	{
		data[pos] = red[tid]; // Copy results to gmem
	}
}

//------------------------------------------------------------------------

// Inclusive scan by Harris: down-sweep phase
template<typename T>
__device__ void scanDown(int tid, int subtask, int step, T* data, T* cls, volatile T* red, int start, int end, T test, T(*op)(T,T), T identity)
{
	int i          = (subtask * WARP_SIZE) + tid;
	int blockSize  = (1 << step);
	int blockEnd   = start + blockSize * (i+1) - 1;
	int pos        = blockEnd;

	red[tid] = identity; // Save identities so that we do not work with uninitialized data

	if(pos < end-1) // -1 so that it is not loaded by thread other than WARP_SIZE-1 which would cause incorrect results
	{
		red[tid] = data[pos];
	}
	else if(tid == WARP_SIZE-1)
	{
		red[tid] = data[end-1];
	}

	if((tid & 31) == 31)
	{
		T tmp = red[tid];
		red[tid] = op(red[tid], red[tid - 16]);
		red[tid - 16] = tmp;
	}
	if((tid & 15) == 15)
	{
		T tmp = red[tid];
		red[tid] = op(red[tid], red[tid -  8]);
		red[tid -  8] = tmp;
	}
	if((tid &  7) == 7)
	{
		T tmp = red[tid];
		red[tid] = op(red[tid], red[tid -  4]);
		red[tid -  4] = tmp;
	}
	if((tid &  3) == 3)
	{
		T tmp = red[tid];
		red[tid] = op(red[tid], red[tid -  2]);
		red[tid -  2] = tmp;
	}
	if((tid &  1) == 1)
	{
		T tmp = red[tid];
		red[tid] = op(red[tid], red[tid -  1]);
		red[tid -  1] = tmp;
	}

	if(pos < end-1)
	{
		if(step == 0) // Add each element's value to make inclusive scan from the computed exclusive scan
			red[tid] += (cls[pos] >= test);
		data[pos] = red[tid]; // Copy results to gmem
	}
	else if(__ffs(__ballot(1)) == tid+1) // Leftmost thread outside of end
	{
		if(step == 0) // Add each element's value to make inclusive scan from the computed exclusive scan
			red[tid] += (cls[end-1] >= test);
		data[end-1] = red[tid];
	}
}

//------------------------------------------------------------------------

// Comments and variables are with regard to -1,0 to the left, 1 to the right
__device__ void sort(int tid, int subtask, int step, int* dataPPSSrc, int* dataPPSBuf, int* dataSort, int* dataIndex, int start, int end, int mid, int test, bool swapIndex)
{
	// Code launching this sort phase is responsible for ensuring there is something to sort
	ASSERT_DIVERGENCE("sort", tid);

	if(step == 0)
	{
		int rayidx = mid + subtask*WARP_SIZE + tid;

		if(rayidx < end)
		{
			int left1s = 0;
			if(start < mid)
				left1s = dataPPSBuf[ mid - 1 ];
			int value = dataPPSSrc[ rayidx ];

#ifdef RAYTRI_TEST
			if(value < -1 || value > 2)
				printf("Invalid value %d in interval mid %d end %d", value, mid, end);
#endif

			// Step 0: copy all indexes of non-1's in range [mid, end] to a separate array starting at index mid
			// Each fits to a spot computed as (save only larger than 0 ones) = position - ( onesBefore - onesBeforeMid )
			int pps = dataPPSBuf[ rayidx ];
			int spot = rayidx - (pps - left1s);

#ifdef RAYTRI_TEST
			if(pps < left1s)
			{
				printf("Crazy pps values for rayidx %d: left1s %d (%d), PPS[rayidx] %d\n", rayidx, left1s, ((int*)dataPPSSrc)[ mid - 1 ], pps);
				//g_taskStackBVH.unfinished = 1;
			}

			if(value < test && (spot < mid || spot >= end))
			{
				int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
				printf("Incorrect low value warp %d, spot %d, rayidx %d, start %d, mid %d, end %d, pps %d, left1s %d, value %d\n", warpIdx, spot, rayidx, start, mid, end, pps, left1s, value);
				//g_taskStackBVH.unfinished = 1;
			}
#endif

			if( value < test && spot >= mid ) // OPTIMIZE: spot >= mid is useless?
				dataSort[spot] = rayidx;
		}
	}
	else
	{
		int rayidx = start + subtask*WARP_SIZE + tid;

		if(rayidx < mid)
		{
			int value = dataPPSSrc[ rayidx ];
			// Step 1: go through all items in range [start, mid] and exchange each 1 with with
			// item in separate array that is on a position mid + my_pss_value - 1
			// We are switching 1s from the left (the start) for the others on the right (the end)

#ifdef RAYTRI_TEST
			if(value < -1 || value > 2)
				printf("Invalid value %d in interval start %d mid %d", value, start, mid);
#endif

			if( value >= test )
			{
				int pps = dataPPSBuf[ rayidx ];
				int repspot  = mid + (pps - 1);
				int repindex = dataSort[ repspot ];

#ifdef RAYTRI_TEST
				if(repindex < start || repindex >= end || dataPPSSrc[ repindex ] >= test)
				{
					int testPps = 0;
					int testSum = 0;
					for(int i = start; i <= rayidx; i++) // Check PPS
					{
						testPps += (dataPPSSrc[ i ] + 1) / 2;
					}
					testSum = testPps;
					for(int i = rayidx+1; i < end; i++) // Check PPS
					{
						testSum += (dataPPSSrc[ i ] + 1) / 2;
					}
					int tp = 0;
					/*if(tid == 0)
					{
						for(int i = start; i < end; i++)
						{
							tp += (((int*)dataPPSSrc)[ i ] + 1) / 2;
							printf("Errorneous value of repindex: Start %d, Mid %d, End %d, rayIdx %d, pps %d, testPps %d\n", start, mid, end, i, ((int*)dataPPSBuf)[ i ], tp);
						}
					}*/
					printf("PPS results tid %d: parallel this %d (%f), sum %d |vs| sequential this %d, sum %d\n", tid, pps, *(float*)&pps, end-mid, testPps, testSum);
					printf("Errorneous value %d of repindex with value %d replaced for %d! Start %d, Mid %d, End %d, rayIdx %d, pps %d, repspot %d, test %d\n", repindex, value, (repindex < start || repindex >= end) ? 123 : ((int*)dataPPSSrc)[ repindex ], start, mid, end, rayidx, pps, repspot, test);
					//g_taskStackBVH.unfinished = 1;
				}
#endif

#if !defined(DEBUG_INFO) && !defined(RAYTRI_TEST)
				if(swapIndex)
#endif
				{
					int temp_buff = dataPPSSrc[ rayidx ];
					dataPPSSrc[ rayidx ] = dataPPSSrc[ repindex ];
					dataPPSSrc[ repindex ] = temp_buff;
				}

				int temp = dataIndex[ rayidx ];
				dataIndex[ rayidx ] = dataIndex[ repindex ];
				dataIndex[ repindex ] = temp;
			}
		}
	}

	//__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	// Possibly the problem only manifests when global synchronization is skipped
}
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
    On demand BVH building specialization of the framework.

    "On demand BVH constuction on many-core processors",
    Marek Vinkler,
    None
*/

#include "rt_common.cu"
#include "CudaBuilderKernels.hpp"

//#define COUNT_INTERUPTS // Outputs the number of build checks, number of unsuccessful checks and number of traversal interupts
						// Use in combination with COUNT_STEPS_DEQUEUE
//#define RAY_STATS

#define PACKET_MULTIPLIER 2
#define SPECULATIVE_OPEN
//#define RANDOM_OPEN
#define ALL_OPEN
//#define SWAP_STACK 1
//#define SKIP_COUNT 16
//#define CRITERION_AABB
//#define PRECOMPUTE_ISECT
//#define SKIP_HIT

//------------------------------------------------------------------------
// Shared variables.
//------------------------------------------------------------------------

__shared__ volatile TaskBVH s_task[NUM_WARPS_PER_BLOCK]; // Memory holding information about the currently processed task
__shared__ volatile TaskBVH s_newTask[NUM_WARPS_PER_BLOCK]; // Memory for the new task to be created in
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

// Other
__device__ bool restartTraversal(int& nodeAddr);
__device__ void taskCreateInnerNode(int tid, volatile TaskBVH* newTask);

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
	if(tid < TASK_GLOBAL) // Prevent overwriting local data saved in task
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

__device__ __forceinline__ int* getTriIdxPtr(int triIdxCtr)
{
#if SCAN_TYPE < 2
	return (int*)c_bvh_in.trisIndex;
#else
	if((triIdxCtr % 2) == 0)
		return (int*)c_bvh_in.trisIndex;
	else
		return (int*)c_bvh_in.sortTris;
#endif
}

//------------------------------------------------------------------------
__device__ __forceinline__ void backcopy(int tid, int triIdxCtr, int triStart, int triEnd)
{
	if(getTriIdxPtr(triIdxCtr) != (int*)c_bvh_in.trisIndex)
	{
		int* inIdx = (int*)c_bvh_in.sortTris + triStart + tid;
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

	//if(tid == 1)
	//	printf("Task %d; phase %d\n", taskIdx, phase);

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
__device__ bool taskTerminationCriteria(int trisLeft, int trisRight, volatile CudaAABB& bbox, volatile CudaAABB& bboxLeft, volatile CudaAABB& bboxRight, int& termCrit, bool& leftLeaf, bool& rightLeaf)
{
	//if(s_task[threadIdx.y].depth < 2) // Unknown error if we try to split an empty task
	//	return false;

#if defined(SAH_TERMINATION) && SPLIT_TYPE != 0
	if(trisLeft+trisRight <= c_env.triMaxLimit)
	{
		float leafCost, leftCost, rightCost;
		// Evaluate if the termination criteria are met
		leafCost = c_env.optCi * (float)(trisLeft+trisRight);
		leftCost = areaAABB(bboxLeft)/areaAABB(bbox)*(float)trisLeft;
		rightCost = areaAABB(bboxRight)/areaAABB(bbox)*(float)trisRight;
		float subdivisionCost = c_env.optCt + c_env.optCi*(leftCost + rightCost);

		if(leafCost < subdivisionCost)
		{
			termCrit = TerminatedBy_Cost;
			return true; // Trivial computation
		}
	}
#endif

	leftLeaf = trisLeft <= c_env.triLimit || s_task[threadIdx.y].depth > (c_env.optMaxDepth-2);
	rightLeaf = trisRight <= c_env.triLimit || s_task[threadIdx.y].depth > (c_env.optMaxDepth-2);

#ifdef CRITERION_AABB
	float saL = areaAABB(bboxLeft);
	//if(tid == 0 && trisLeft > c_env.triLimit && saL < c_env.subdivThreshold)
	//	printf("Left %d saL %f threshold %f\n", childLeft, saL, c_env.subdivThreshold);

	// Check if children should be leaves
	leftLeaf = leftLeaf || saL < c_env.subdivThreshold;

	float saR = areaAABB(bboxRight);
	//if(tid == 0 && trisRight > c_env.triLimit && saR < c_env.subdivThreshold)
	//	printf("Right %d saL %f threshold %f\n", childRight, saR, c_env.subdivThreshold);

	rightLeaf = rightLeaf || saR < c_env.subdivThreshold;
#endif

	return false; // Continue subdivision
}

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
	int numLeft = triRight - triStart;
	int numRight = triEnd - triRight;
	bool leftLeaf, rightLeaf;
	int triIdxCtr = newTask->triIdxCtr;
	int parentIdx = newTask->parentIdx;
	int nodeIdx = newTask->nodeIdx;

	ASSERT_DIVERGENCE("taskDecideType top", tid);
	
	if(taskTerminationCriteria(numLeft, numRight, newTask->bbox, newTask->bboxLeft, newTask->bboxRight, termCrit, leftLeaf, rightLeaf))
	{
#ifndef COMPACT_LAYOUT
		volatile CudaBVHNode* node = (CudaBVHNode*)&s_newTask[threadIdx.y];

		// Leaf -> same bounding boxes
		*((float4*)&node->c0xy) = make_float4(newTask->bbox.m_mn.x, newTask->bbox.m_mx.x, newTask->bbox.m_mn.y, newTask->bbox.m_mx.y);
		*((float4*)&node->c1xy) = make_float4(newTask->bbox.m_mn.x, newTask->bbox.m_mx.x, newTask->bbox.m_mn.y, newTask->bbox.m_mx.y);
		*((float4*)&node->c01z) = make_float4(newTask->bbox.m_mn.z, newTask->bbox.m_mx.z, newTask->bbox.m_mn.z, newTask->bbox.m_mx.z);
		*((int4*)&node->children) = make_int4(triStart, triEnd, parentIdx, 0);
		taskSaveNodeToGMEM(g_bvh, tid, nodeIdx, *node);

#if SCAN_TYPE == 2 || SCAN_TYPE == 3
		// Back-copy triangles to the correct array
		backcopy(tid, triIdxCtr, triStart, triEnd);
#endif
		__threadfence(); // Make sure the node is in the hierarchy before we update the parent

		if(tid == 0)
			taskUpdateParentPtr(g_bvh, parentIdx, newTask->taskID, ~nodeIdx); // Mark this node as leaf in the hierarchy	
#else
		if(tid == 0)
		{
			//printf("Left %d, right %d, depth %d\n", numLeft, numRight, newTask->depth);
			s_sharedData[threadIdx.y][3] = atomicAdd(&g_taskStackBVH.triTop, (numLeft+numRight)*3+1); // Atomically acquire leaf space, +1 is for the triangle sentinel
		}
		int triOfs = s_sharedData[threadIdx.y][3];
#ifndef WOOP_TRIANGLES
		int triIdx = createLeaf(tid, triOfs, (float*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, triStart, triEnd, (float*)c_bvh_in.tris, getTriIdxPtr(triIdxCtr)); 
#else
		int triIdx = createLeafWoop(tid, triOfs, (float4*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, triStart, triEnd, (float4*)c_bvh_in.tris, getTriIdxPtr(triIdxCtr));
#endif
		__threadfence(); // Make sure the node is in the hierarchy before we update the parent

		if(tid == 0)
			taskUpdateParentPtr(g_bvh, parentIdx, newTask->taskID, triIdx); // Mark this node as leaf in the hierarchy
#endif

		// Mark children as leaves -> correct update of unfinished counter
		s_sharedData[threadIdx.y][0] = -1;
		s_sharedData[threadIdx.y][1] = -1;

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
		
#ifndef COMPACT_LAYOUT
		volatile CudaBVHNode* node0 = ((CudaBVHNode*)&s_newTask[threadIdx.y])+0;
		volatile CudaBVHNode* node1 = ((CudaBVHNode*)&s_newTask[threadIdx.y])+1;

		if(tid == 0)
		{
			// Inner node -> create new subtasks in the final array
			s_sharedData[threadIdx.y][2] = atomicAdd(&g_taskStackBVH.nodeTop, 2);
		}
		int childrenIdx = s_sharedData[threadIdx.y][2];

		int childLeft = childrenIdx+0;
		int childRight = childrenIdx+1;
#else
		int innerNodes = (int)(!leftLeaf) + (int)(!rightLeaf);
		int leafChunks = ((leftLeaf) ? (numLeft*3+1) : 0) + ((rightLeaf) ? (numRight*3+1) : 0); // Atomically acquire leaf space, +1 is for the triangle sentinel
		if(tid == 0)
		{
			if(innerNodes > 0)
				s_sharedData[threadIdx.y][2] = atomicAdd(&g_taskStackBVH.nodeTop, innerNodes); // Inner node -> create new subtasks in the final array
			if(innerNodes < 2)
				s_sharedData[threadIdx.y][3] = atomicAdd(&g_taskStackBVH.triTop, leafChunks); // Atomically acquire leaf space

			// Check if there is enough memory to write the nodes and triangles
			if((innerNodes > 0 && s_sharedData[threadIdx.y][2]+innerNodes >= g_taskStackBVH.sizeNodes) || (leafChunks > 0 && s_sharedData[threadIdx.y][3]+leafChunks >= g_taskStackBVH.sizeTris))
			{
				g_taskStackBVH.unfinished = 1;
			}
		}
		int childrenIdx = s_sharedData[threadIdx.y][2];
		int triOfs = s_sharedData[threadIdx.y][3];

		int childLeft = childrenIdx+0;
		int childRight = childrenIdx+((leftLeaf) ? 0 : 1);

		int leftOfs = triOfs;
		int rightOfs = triOfs+((leftLeaf) ? (numLeft*3+1) : 0);
#endif

		ASSERT_DIVERGENCE("taskDecideType node2", tid);

		// Check if children should be leaves
		if(leftLeaf)
		{
#ifndef COMPACT_LAYOUT
			// Leaf -> same bounding boxes
			*((float4*)&node0->c0xy) = make_float4(newTask->bboxLeft.m_mn.x, newTask->bboxLeft.m_mx.x, newTask->bboxLeft.m_mn.y, newTask->bboxLeft.m_mx.y);
			*((float4*)&node0->c1xy) = make_float4(newTask->bboxLeft.m_mn.x, newTask->bboxLeft.m_mx.x, newTask->bboxLeft.m_mn.y, newTask->bboxLeft.m_mx.y);
			*((float4*)&node0->c01z) = make_float4(newTask->bboxLeft.m_mn.z, newTask->bboxLeft.m_mx.z, newTask->bboxLeft.m_mn.z, newTask->bboxLeft.m_mx.z);
			*((int4*)&node0->children) = make_int4(triStart, triRight, nodeIdx, 0);
			taskSaveNodeToGMEM(g_bvh, tid, childLeft, *node0);
			childLeft = ~childLeft;

#if SCAN_TYPE == 2 || SCAN_TYPE == 3
			// Back-copy triangles to the correct array
			backcopy(tid, triIdxCtr, triStart, triRight);
#endif
#else
			// OPTIMIZE: Write out both children in one call?
#ifndef WOOP_TRIANGLES
			childLeft = createLeaf(tid, leftOfs, (float*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, triStart, triRight, (float*)c_bvh_in.tris, getTriIdxPtr(triIdxCtr));
#else
			childLeft = createLeafWoop(tid, leftOfs, (float4*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, triStart, triRight, (float4*)c_bvh_in.tris, getTriIdxPtr(triIdxCtr));
#endif
#endif

			if(tid == 0)
			{
#ifdef BVH_COUNT_NODES
				atomicAdd(&g_taskStackBVH.numLeaves, 1);
				//printf("Leaf left (%d, %d)\n", triStart, triRight);
#endif
#ifdef LEAF_HISTOGRAM
				atomicAdd(&g_taskStackBVH.leafHist[numLeft], 1); // Update histogram
#endif
			}
		}
		s_sharedData[threadIdx.y][0] = childLeft;

		ASSERT_DIVERGENCE("taskDecideType node3", tid);

		if(rightLeaf)
		{
#ifndef COMPACT_LAYOUT
			// Leaf -> same bounding boxes
			*((float4*)&node1->c0xy) = make_float4(newTask->bboxRight.m_mn.x, newTask->bboxRight.m_mx.x, newTask->bboxRight.m_mn.y, newTask->bboxRight.m_mx.y);
			*((float4*)&node1->c1xy) = make_float4(newTask->bboxRight.m_mn.x, newTask->bboxRight.m_mx.x, newTask->bboxRight.m_mn.y, newTask->bboxRight.m_mx.y);
			*((float4*)&node1->c01z) = make_float4(newTask->bboxRight.m_mn.z, newTask->bboxRight.m_mx.z, newTask->bboxRight.m_mn.z, newTask->bboxRight.m_mx.z);
			*((int4*)&node1->children) = make_int4(triRight, triEnd, nodeIdx, 0);
			taskSaveNodeToGMEM(g_bvh, tid, childRight, *node1);
			childRight = ~childRight;

#if SCAN_TYPE == 2 || SCAN_TYPE == 3
			// Back-copy triangles to the correct array
			backcopy(tid, triIdxCtr, triRight, triEnd);
#endif
#else
			// OPTIMIZE: Write out both children in one call?
#ifndef WOOP_TRIANGLES
			childRight = createLeaf(tid, rightOfs, (float*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, triRight, triEnd, (float*)c_bvh_in.tris, getTriIdxPtr(triIdxCtr));
#else
			childRight = createLeafWoop(tid, rightOfs, (float4*)c_bvh_in.trisOut, (int*)c_bvh_in.trisIndexOut, triRight, triEnd, (float4*)c_bvh_in.tris, getTriIdxPtr(triIdxCtr));
#endif
#endif

			if(tid == 0)
			{
#ifdef BVH_COUNT_NODES
				atomicAdd(&g_taskStackBVH.numLeaves, 1);
				//printf("Leaf right (%d, %d)\n", triRight, triEnd);
#endif
#ifdef LEAF_HISTOGRAM
				atomicAdd(&g_taskStackBVH.leafHist[numRight], 1); // Update histogram
#endif
			}
		}
		s_sharedData[threadIdx.y][1] = childRight;

		ASSERT_DIVERGENCE("taskDecideType nodeBottom", tid);

		// Create the parent immediately
		if(leftLeaf && rightLeaf)
		{
			taskCreateInnerNode(tid, &s_task[threadIdx.y]);
		}

		return leftLeaf && rightLeaf;
	}
}

//------------------------------------------------------------------------

// Creates a parent node in final array
__device__ void taskCreateInnerNode(int tid, volatile TaskBVH* newTask)
{
	int childLeft = s_sharedData[threadIdx.y][0];
	int childRight = s_sharedData[threadIdx.y][1];
	int triStart = newTask->triStart;
	int triEnd = newTask->triEnd;
	int parentIdx = newTask->parentIdx;
	int taskID = newTask->taskID;
	int nodeIdx = newTask->nodeIdx;
#ifndef COMPACT_LAYOUT
	int childIdx = s_sharedData[threadIdx.y][2]-1;
#else
	int childIdx = (nodeIdx != 0) ? ((childLeft >= 0) ? s_sharedData[threadIdx.y][2]-1 : s_sharedData[threadIdx.y][2]-2) : 0;
#endif

	volatile CudaBVHNode* node = (CudaBVHNode*)&s_newTask[threadIdx.y];

	//if(tid == 1)
	//	printf("Constructed node %d; warpCounter %d %s %s\n", nodeIdx, *(int*)&g_taskStackBVH.warpCounter, (childLeft >= 0) ? "L" : "", (childRight >= 0) ? "R" : "");

	*((float4*)&node->c0xy) = make_float4(newTask->bboxLeft.m_mn.x, newTask->bboxLeft.m_mx.x, newTask->bboxLeft.m_mn.y, newTask->bboxLeft.m_mx.y);
	*((float4*)&node->c1xy) = make_float4(newTask->bboxRight.m_mn.x, newTask->bboxRight.m_mx.x, newTask->bboxRight.m_mn.y, newTask->bboxRight.m_mx.y);
	*((float4*)&node->c01z) = make_float4(newTask->bboxLeft.m_mn.z, newTask->bboxLeft.m_mx.z, newTask->bboxRight.m_mn.z, newTask->bboxRight.m_mx.z);
	*((int4*)&node->children) = make_int4(childLeft, childRight, parentIdx, childIdx); // Sets the leaf child pointers or the waiting status
	taskSaveNodeToGMEM(g_bvh, tid, nodeIdx, *node);
	__threadfence(); // Make sure the node is in the hierarchy before we update the parent

#ifndef COMPACT_LAYOUT
	taskUpdateParentPtr(g_bvh, parentIdx, taskID, nodeIdx); // Mark this node as finished in the hierarchy
#else
	taskUpdateParentPtr(g_bvh, parentIdx, taskID, nodeIdx*64); // Mark this node as finished in the hierarchy
#endif

#ifdef BVH_COUNT_NODES
	if(tid == 0)
	{
		atomicAdd(&g_taskStackBVH.numNodes, 1);
		atomicAdd(&g_taskStackBVH.numSortedTris, triEnd - triStart);
	//	printf("Node (%d, %d)\n", triStart, triEnd);
	}
#endif
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
	newTask->triIdxCtr = s_task[threadIdx.y].triIdxCtr;
	newTask->origSize = newTask->unfinished;
#if SCAN_TYPE == 2 || SCAN_TYPE == 3
	newTask->triLeft = newTask->triStart;
	newTask->triRight = newTask->triEnd;
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
			newTask->triStart   = s_task[threadIdx.y].triRight;
			newTask->triEnd     = s_task[threadIdx.y].triEnd;
			srcBox = (volatile const float*)&(s_task[threadIdx.y].bboxRight);
			break;

		case 0:
			newTask->triStart   = s_task[threadIdx.y].triStart;
			newTask->triEnd     = s_task[threadIdx.y].triRight;
			srcBox = (volatile const float*)&(s_task[threadIdx.y].bboxLeft);
			break;
		}
	}

	// Copy CudaAABB from corresponding task
	if(tid < sizeof(CudaAABB)/sizeof(float))
	{
		bbox[tid] = srcBox[tid];
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
		if(s_sharedData[threadIdx.y][i] < 0) // Skip leaf
			continue;
		taskCreateSubtask(tid, &s_newTask[threadIdx.y], i); // Fill newTask with valid task for ID=i

		s_newTask[threadIdx.y].parentIdx = s_task[threadIdx.y].nodeIdx;
		s_newTask[threadIdx.y].nodeIdx = s_sharedData[threadIdx.y][i];
		int newStatus = TaskHeader_Waiting | s_sharedData[threadIdx.y][i];
#ifdef SPECULATIVE_OPEN
#if !defined(RANDOM_OPEN) && !defined(ALL_OPEN)
		bool unlocked = s_task[threadIdx.y].pivot > 0 || s_newTask[threadIdx.y].triEnd - s_newTask[threadIdx.y].triStart <= c_env.childLimit;
		s_sharedData[threadIdx.y][i+3] = (unlocked) ? s_newTask[threadIdx.y].unfinished : -1;
#else
		bool unlocked = true;
#if defined(RANDOM_OPEN)
		s_sharedData[threadIdx.y][i+3] = newStatus;
#elif defined(ALL_OPEN)
		s_sharedData[threadIdx.y][i+3] = s_newTask[threadIdx.y].unfinished;
		if(s_task[threadIdx.y].cached != LockType_Free && s_task[threadIdx.y].cached != LockType_Subtask) // Requested node
			s_newTask[threadIdx.y].pivot = c_env.subtreeLimit;
		else
#endif
#endif
			s_newTask[threadIdx.y].pivot = s_task[threadIdx.y].pivot - 1; // Lower the number of to-be-build levels
		//s_newTask[threadIdx.y].pivot = (s_task[threadIdx.y].pivot > 0) ? s_task[threadIdx.y].pivot - 1 : c_env.subtreeLimit; // Lower the number of to-be-build levels or restart the counter
#endif

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
				return;
		}
#else
		taskEnqueueLeft(tid, g_taskStackBVH.header, s_sharedData[threadIdx.y], newStatus, beg, &g_taskStackBVH.unfinished, g_taskStackBVH.sizePool); // Go left of beg and fill empty tasks
		if(beg == -1)
				return;
#endif

		// All threads
#ifdef SPECULATIVE_OPEN
#if !defined(RANDOM_OPEN) && !defined(ALL_OPEN)
		s_sharedData[threadIdx.y][i] = ((unlocked) ? UNBUILD_UNMASK : beg) | UNBUILD_FLAG; // Save the position of the waiting task and mark it as unbuild
#else
		s_sharedData[threadIdx.y][i] = (beg | UNBUILD_FLAG); // Save the position of the waiting task and mark it as unbuild
#endif
		s_sharedData[threadIdx.y][i+5] = beg;
#else
		s_sharedData[threadIdx.y][i] = (beg | UNBUILD_FLAG); // Save the position of the waiting task and mark it as unbuild
#endif
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
		if(beg >= 0 && beg < g_taskStackBVH.sizePool) // TESTING ONLY - WRITE WILL CAUSE "UNKNOWN ERROR" IF WARP DIVERGES
			taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
		else
			printf("task adding on invalid index: %d, Tid %d\n", beg, tid);
#else
		taskSaveFirstToGMEM(tid, beg, s_newTask[threadIdx.y]);
#endif
		/*if((s_task[threadIdx.y].cached != LockType_Free && s_task[threadIdx.y].cached != LockType_Subtask) || s_task[threadIdx.y].pivot > 0)
			g_taskStackBVH.tasks[beg].cached = LockType_Subtask; // Mark the task as cached
		else*/
			g_taskStackBVH.tasks[beg].cached = LockType_Free; // Mark the task as uncached

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
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
#endif

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle2", tid);

		
//		if(newStatus > TaskHeader_Active) // No need to wait if the task is inactive anyway
//		{
//#if PARALLELISM_TEST >= 0
//			if(tid == 0)
//			{
//#ifdef CUTOFF_DEPTH
//				int active;
//				if(s_newTask[threadIdx.y].depth > c_env.optCutOffDepth)
//					active = atomicAdd(&g_numActive, 0);
//				else
//					active = atomicAdd(&g_numActive, 1)+1;
//#else
//				int active = atomicAdd(&g_numActive, 1);
//#endif
//				int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
//				//if(active > ACTIVE_MAX)
//				//	printf("Warp %d too much [%d] subtasks\n", warpIdx, active);
//#ifdef CUTOFF_DEPTH
//				if(/*beg == 124 || */(active == 0 && i == table.count-1))
//#else
//				if(active == 0)
//#endif
//				{
//					//printf("Warp %d no active tasks before adding task with %d subtasks\n", warpIdx, newStatus);
//					g_taskStackBVH.unfinished = 1;
//				}
//			}
//#endif
//			__threadfence(); // Make sure task is copied to the global memory before we unlock it
//
//#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
//			if(tid == 24)
//			{
//				taskCacheActive(beg, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
//			}
//#endif
//		}

		// Unlock the task - set the task status
#ifdef CUTOFF_DEPTH
		if(s_newTask[threadIdx.y].depth > c_env.optCutOffDepth)
			g_taskStackBVH.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks
		else
			g_taskStackBVH.header[beg] = newStatus; // Task will be opened on demand
#else

#ifdef SPECULATIVE_OPEN
		if(!unlocked) // Open it immediately as waiting
#endif
			g_taskStackBVH.header[beg] = newStatus; // Task will be opened on demand
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
		if(s_sharedData[threadIdx.y][i] < 0) // Skip leaf
			continue;
		taskCreateSubtask(tid, &s_newTask[threadIdx.y], i); // Fill newTask with valid task for ID=i

		s_newTask[threadIdx.y].parentIdx = s_task[threadIdx.y].nodeIdx;
		s_newTask[threadIdx.y].nodeIdx = s_sharedData[threadIdx.y][i];
		int newStatus = TaskHeader_Waiting | s_sharedData[threadIdx.y][i];
#ifdef SPECULATIVE_OPEN
#if !defined(RANDOM_OPEN) && !defined(ALL_OPEN)
		bool unlocked = s_task[threadIdx.y].pivot > 0 || s_newTask[threadIdx.y].triEnd - s_newTask[threadIdx.y].triStart <= c_env.childLimit;
		s_sharedData[threadIdx.y][i+3] = (unlocked) ? s_newTask[threadIdx.y].unfinished : -1;
#else
		bool unlocked = true;
#if defined(RANDOM_OPEN)
		s_sharedData[threadIdx.y][i+3] = newStatus;
#elif defined(ALL_OPEN)
		s_sharedData[threadIdx.y][i+3] = s_newTask[threadIdx.y].unfinished;
		if(s_task[threadIdx.y].cached != LockType_Free && s_task[threadIdx.y].cached != LockType_Subtask) // Requested node
			s_newTask[threadIdx.y].pivot = c_env.subtreeLimit;
		else
#endif
#endif
			s_newTask[threadIdx.y].pivot = s_task[threadIdx.y].pivot - 1; // Lower the number of to-be-build levels
		//s_newTask[threadIdx.y].pivot = (s_task[threadIdx.y].pivot > 0) ? s_task[threadIdx.y].pivot - 1 : c_env.subtreeLimit; // Lower the number of to-be-build levels or restart the counter
#endif

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
				return;
		}

		// All threads
#ifdef SPECULATIVE_OPEN
#if !defined(RANDOM_OPEN) && !defined(ALL_OPEN)
		s_sharedData[threadIdx.y][i] = ((unlocked) ? UNBUILD_UNMASK : beg) | UNBUILD_FLAG; // Save the position of the waiting task and mark it as unbuild
#else
		s_sharedData[threadIdx.y][i] = (beg | UNBUILD_FLAG); // Save the position of the waiting task and mark it as unbuild
#endif
		s_sharedData[threadIdx.y][i+5] = beg;
#else
		s_sharedData[threadIdx.y][i] = (beg | UNBUILD_FLAG); // Save the position of the waiting task and mark it as unbuild
#endif
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
		/*if((s_task[threadIdx.y].cached != LockType_Free && s_task[threadIdx.y].cached != LockType_Subtask) || s_task[threadIdx.y].pivot > 0)
			g_taskStackBVH.tasks[beg].cached = LockType_Subtask; // Mark the task as cached
		else*/
			g_taskStackBVH.tasks[beg].cached = LockType_Free; // Mark the task as uncached

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
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
#endif

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle2", tid);

//		if(newStatus > TaskHeader_Active) // No need to wait if the task is inactive anyway
//		{
//#if PARALLELISM_TEST >= 0
//			if(tid == 0)
//			{
//				int active = atomicAdd(&g_numActive, 1);
//				int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
//				if(active == 0)
//					printf("Warp %d no active tasks before adding task with %d subtasks\n", warpIdx, newStatus);
//			}
//#endif
//			__threadfence(); // Make sure task is copied to the global memory before we unlock it
//
//#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
//			if(tid == 24)
//			{
//				taskCacheActive(beg, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
//			}
//#endif
//		}

		// Unlock the task - set the task status
#ifdef SPECULATIVE_OPEN
		if(!unlocked) // Open it immediately as waiting
#endif
			g_taskStackBVH.header[beg] = newStatus; // This operation is atomic anyway
		//g_taskStackBVH.header[beg] = TaskHeader_Locked; // Stop the algorithm by not activating tasks

		beg++; // Move for next item

		ASSERT_DIVERGENCE("taskEnqueueSubtasks forcycle3", tid);
	}

	ASSERT_DIVERGENCE("taskEnqueueSubtasks aftercycle", tid);


	//ASSERT_DIVERGENCE("taskEnqueueSubtasks bottom", tid); // Tid 24 diverges here but it converges at the end of this function
}

#endif // ENQUEUE_TYPE == 3

//------------------------------------------------------------------------

#if DEQUEUE_TYPE <= 5
__device__ __noinline__ bool taskDequeue(int tid, int& nodeAddr)
{
	ASSERT_DIVERGENCE("taskDequeue", tid);

	// Initiate variables
	int* header = g_taskStackBVH.header;
	int* unfinished = &g_taskStackBVH.unfinished;
	int* stackTop = &g_taskStackBVH.top;
	volatile int* red = (volatile int*)&s_newTask[threadIdx.y];

	int status = TaskHeader_Active;
#ifdef RANDOM_OPEN
	int check = TaskHeader_Active;
#endif
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

	int cnt = 0;
#if defined(TRAVERSAL_TEST) && defined(ALL_OPEN)
	do
	{
	counter = 0;
#endif

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
		cnt++;

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
		if(s_task[threadIdx.y].rayPackets != -1 && restartTraversal(nodeAddr))
			return true;

		int topChunk = taskWarpSubtasks(*stackTop);
		beg = (topChunk - (warpIdx % topChunk)) * WARP_SIZE - tid;
	}
	//else
	//	cached = true;

#if defined(TRAVERSAL_TEST) && defined(ALL_OPEN)
	} while(g_taskStackBVH.launchFlag != 0 && status <= TaskHeader_Active && (s_task[threadIdx.y].rayPackets != -1 || *unfinished < 0));
#endif
	ASSERT_DIVERGENCE("taskDequeue cache", tid);
#endif

	//---------------------------- REVERT TO POOL SEARCH ----------------------------//

	while(counter < WAIT_COUNT && status <= TaskHeader_Active && (s_task[threadIdx.y].rayPackets != -1 || *unfinished < 0)) // while g_taskStack is not empty and we have not found ourselves a task
	{
		// Find first active task, end if we have reached start of the array
#ifndef RANDOM_OPEN
		while(__any(beg >= 0) && __all(beg < 0 || (status = header[beg]) <= TaskHeader_Active) /*&& restartPos != tryCounter*/)
#else
		while(__any(beg >= 0) && __all(beg < 0 || (status = header[beg]) <= check || (status >= TaskHeader_Dependent && status <= TaskHeader_Active)))
#endif
		{				
			beg -= WARP_SIZE;
#ifdef COUNT_STEPS_DEQUEUE
			readCounter++;
#endif
		}

		// OPTIMIZE: How to use the threads with beg < 0 in a clever way?
#ifndef RANDOM_OPEN
		if(__all(status <= TaskHeader_Active)) // We have found no active task
#else
		if(__all(status <= check || (status >= TaskHeader_Dependent && status <= TaskHeader_Active))) // We have found no active task
#endif
		{
			if(s_task[threadIdx.y].rayPackets != -1 && restartTraversal(nodeAddr))
			{
#ifndef COUNT_INTERUPTS
#ifdef COUNT_STEPS_DEQUEUE
				maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], readCounter);
				sumSteps[threadIdx.y] += readCounter;
				//numSteps[threadIdx.y]++;
				numRestarts[threadIdx.y] += counter;
#endif
#endif
				return true;
			}

			beg = (taskWarpSubtasks(*stackTop))*WARP_SIZE - tid; // Try again from a new beginning
			counter++;
#ifdef RANDOM_OPEN
			check = TaskHeader_Waiting;
			status = TaskHeader_Active; // Reset state of all threads
#endif
			continue;
		}

		// OPTIMIZE: On HappyBuddha the build can be slightly sped up by eliminating reduction and using status
		// Initialize memory for reduction
		/*red[tid] = 0;
		if(status > TaskHeader_Active)
		{
			//red[tid] = *((int*)&g_taskStackBVH.tasks[beg].origSize);
			red[tid] = status;
		}

		reduceWarp<int>(tid, red, plus);*/

		int popCount;
#ifdef RANDOM_OPEN
		if(status > TaskHeader_Active)
		{
#endif
#if 1 // Simple but overestimating strategy
#ifndef ALL_OPEN
		popCount = max((status / NUM_WARPS) + 1, taskPopCount(status));
#else
		popCount = 3; // Use minimal work amount for pool dequeues
#endif
		//popCount = max((red[tid] / NUM_WARPS) + 1, taskPopCount(status));
#else // Precise strategy
		int div = red[tid] / NUM_WARPS;
		int rest = red[tid] - div*NUM_WARPS;
		div += (warpIdx < rest) ? 1 : 0;
		s_task[threadIdx.y].popCount = max(div, taskPopCount(status));
#endif
#ifdef RANDOM_OPEN
		}
		else if(check == TaskHeader_Waiting && (status >= TaskHeader_Waiting && status < TaskHeader_Dependent)) 
		{
			int origSize = *((int*)&g_taskStackBVH.tasks[beg].origSize);
			//printf("Beg %d, Status %d, origSize %d\n", beg, status, origSize);
			popCount = max((origSize / NUM_WARPS) + 1, taskPopCount(origSize));
			atomicCAS(&g_taskStackBVH.header[beg], status, origSize); // Open some task
			status = origSize;
			//printf("Beg %d, Status %d, origSize %d\n", beg, status, origSize);
			s_task[threadIdx.y].popSubtask = TaskHeader_Active;
		}
#endif

		// Choose the right position
		/*int tidPos = warpIdx % WARP_SIZE; // Position of the tid to take work from

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
#ifdef RANDOM_OPEN
		check = TaskHeader_Active;
#endif

		if(status <= TaskHeader_Active) // We have not succeeded
		{
#ifdef COUNT_STEPS_DEQUEUE
			readCounter++;
#endif
			/*if(tid == 0 && status < TaskHeader_Dependent)
			{
				atomicAdd(&g_taskStackBVH.header[beg], s_task[threadIdx.y].popCount); // Revert if we have accidentaly changed waiting task
			}*/

			// Move to next task
			beg -= WARP_SIZE;
			// OPTIMIZE: we shall move beg -= WARP_SIZE; as the first statement in the outer while and start with g_taskStack.top+1.
		}
		//else
		//	cached = false;
	}

#ifdef SNAPSHOT_WARP
	s_sharedData[threadIdx.y][3] = readCounter;
#endif

#ifdef COUNT_STEPS_DEQUEUE
#ifndef COUNT_INTERUPTS
	//if(cached)
	//{
	maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], readCounter);
	sumSteps[threadIdx.y] += readCounter;
	numSteps[threadIdx.y]++;
	numRestarts[threadIdx.y] += counter;
	//}
#endif

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

#elif DEQUEUE_TYPE == 6
__device__ __noinline__ bool taskDequeue(int tid, int& nodeAddr)
{
	ASSERT_DIVERGENCE("taskDequeue", tid);

	// Initiate variables
	int* header = g_taskStackBVH.header;
	int* unfinished = &g_taskStackBVH.unfinished;
	int* stackTop = &g_taskStackBVH.top;
	volatile int* red = (volatile int*)&s_newTask[threadIdx.y];

	int status = TaskHeader_Active;
#ifdef RANDOM_OPEN
	int check = TaskHeader_Active;
#endif
	int counter = 0; // TESTING ONLY: Allows to undeadlock a failed run!
#ifdef COUNT_STEPS_DEQUEUE
	int readCounter = 1;
#endif

#ifdef SNAPSHOT_WARP
	long long int clock = clock64();
	*(long long int*)&(s_sharedData[threadIdx.y][4]) = clock;
#endif

	int warpIdx = (blockDim.y*blockIdx.x + threadIdx.y);

	//unsigned int *activeTop = &g_taskStackBVH.activeTop;
	int* cache = g_taskStackBVH.active;

	int beg = -1;
	bool reset = true;
	while(counter < WAIT_COUNT && status <= TaskHeader_Active && (s_task[threadIdx.y].rayPackets != -1 || *unfinished < 0)) // while g_taskStack is not empty and we have not found ourselves a task
	{
		s_task[threadIdx.y].popSubtask = status;
		//int item = pos - tid;
		//if(item >= 0)
		int item = tid;
		if(reset /*&& *stackTop > 4096*/ && item <= ACTIVE_MAX)
		{
			beg = cache[item];
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

		if(__all(status <= TaskHeader_Active)) // We have found no active task
		{
			if(reset)
			{
				beg = taskWarpSubtasks(*stackTop) * WARP_SIZE - tid;
			}

			ASSERT_DIVERGENCE("taskDequeue cache", tid);

			//---------------------------- REVERT TO POOL SEARCH ----------------------------//

			// Find first active task, end if we have reached start of the array
#ifndef RANDOM_OPEN
			while(__any(beg >= 0) && __all(beg < 0 || (status = header[beg]) <= TaskHeader_Active) /*&& restartPos != tryCounter*/)
#else
			while(__any(beg >= 0) && __all(beg < 0 || (status = header[beg]) <= check || (status >= TaskHeader_Dependent && status <= TaskHeader_Active)))
#endif
			{				
				beg -= WARP_SIZE;
#ifdef COUNT_STEPS_DEQUEUE
				readCounter++;
#endif
			}
		}

		// OPTIMIZE: How to use the threads with beg < 0 in a clever way?
#ifndef RANDOM_OPEN
		if(__all(status <= TaskHeader_Active)) // We have found no active task
#else
		if(__all(status <= check || (status >= TaskHeader_Dependent && status <= TaskHeader_Active))) // We have found no active task
#endif
		{
			if(s_task[threadIdx.y].rayPackets != -1 && restartTraversal(nodeAddr))
			{
#ifndef COUNT_INTERUPTS
#ifdef COUNT_STEPS_DEQUEUE
				maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], readCounter);
				sumSteps[threadIdx.y] += readCounter;
				//numSteps[threadIdx.y]++;
				numRestarts[threadIdx.y] += counter;
#endif
#endif
				return true;
			}

			counter++;
#ifdef RANDOM_OPEN
			check = TaskHeader_Waiting;
			status = TaskHeader_Active; // Reset state of all threads
#endif
			reset = true;
			continue;
		}

		// OPTIMIZE: On HappyBuddha the build can be slightly sped up by eliminating reduction and using status
		// Initialize memory for reduction
		/*red[tid] = 0;
		if(status > TaskHeader_Active)
		{
			//red[tid] = *((int*)&g_taskStackBVH.tasks[beg].origSize);
			red[tid] = status;
		}

		reduceWarp(tid, red, plus);*/

		int popCount;
#ifdef RANDOM_OPEN
		if(status > TaskHeader_Active)
		{
#endif
#if 1 // Simple but overestimating strategy
		popCount = max((status / NUM_WARPS) + 1, taskPopCount(status));
		//popCount = max((red[tid] / NUM_WARPS) + 1, taskPopCount(status));
#else // Precise strategy
		int div = red[tid] / NUM_WARPS;
		int rest = red[tid] - div*NUM_WARPS;
		div += (warpIdx < rest) ? 1 : 0;
		s_task[threadIdx.y].popCount = max(div, taskPopCount(status));
#endif
#ifdef RANDOM_OPEN
		}
		else if(check == TaskHeader_Waiting && (status >= TaskHeader_Waiting && status < TaskHeader_Dependent)) 
		{
			int origSize = *((int*)&g_taskStackBVH.tasks[beg].origSize);
			//printf("Beg %d, Status %d, origSize %d\n", beg, status, origSize);
			popCount = max((origSize / NUM_WARPS) + 1, taskPopCount(origSize));
			atomicCAS(&g_taskStackBVH.header[beg], status, origSize); // Open some task
			status = origSize;
			//printf("Beg %d, Status %d, origSize %d\n", beg, status, origSize);
			s_task[threadIdx.y].popSubtask = TaskHeader_Active;
		}
#endif

		// Choose the right position
		/*int tidPos = warpIdx % WARP_SIZE; // Position of the tid to take work from

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
		scanWarp(tid, red, plus);

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
#ifdef RANDOM_OPEN
		check = TaskHeader_Active;
#endif

		if(status <= TaskHeader_Active) // We have not succeeded
		{
#ifdef COUNT_STEPS_DEQUEUE
			readCounter++;
#endif
			/*if(tid == 0 && status < TaskHeader_Dependent)
			{
				atomicAdd(&g_taskStackBVH.header[beg], s_task[threadIdx.y].popCount); // Revert if we have accidentaly changed waiting task
			}*/

			// Move to next task
			beg -= WARP_SIZE;
			reset = false;
			// OPTIMIZE: we shall move beg -= WARP_SIZE; as the first statement in the outer while and start with g_taskStack.top+1.
		}
	}

#ifdef SNAPSHOT_WARP
	s_sharedData[threadIdx.y][3] = readCounter;
#endif

#ifdef COUNT_STEPS_DEQUEUE
#ifndef COUNT_INTERUPTS
	maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], readCounter);
	sumSteps[threadIdx.y] += readCounter;
	numSteps[threadIdx.y]++;
	numRestarts[threadIdx.y] += counter;
#endif

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
	// We should update the dependencies before we start adding new tasks because these subtasks may be finished before this is done

#ifndef KEEP_ALL_TASKS
	atomicCAS(&g_taskStackBVH.top, taskIdx, max(taskIdx-1, 0)); // Try decreasing the stack top
#endif

#if PARALLELISM_TEST == 0
	atomicSub(&g_numActive, 1);
#endif

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

		// Prevent other warps from caching a task that is finished
		if(tid == 0)
			s_task[threadIdx.y].cached = atomicCAS(&g_taskStackBVH.tasks[taskIdx].cached, LockType_Free, LockType_None);

#if 1
		while(s_task[threadIdx.y].cached == LockType_Set) // Wait for the task to be finished caching
		{
			s_task[threadIdx.y].cached = g_taskStackBVH.tasks[taskIdx].cached;
		}
#elif 0
		if(s_task[threadIdx.y].cached == LockType_Set)
		{
			volatile int* test = &g_taskStackBVH.tasks[taskIdx].cached;

			while(*test == LockType_Set) // Wait for the task to be finished caching
			{
				;
			}
			s_task[threadIdx.y].cached = *test;
		}
#else
		if(s_task[threadIdx.y].cached == LockType_Set)
		{
			while(atomicCAS(&g_taskStackBVH.tasks[taskIdx].cached, LockType_None, LockType_None) == LockType_Set) // Wait for the task to be finished caching
			{
				;
			}
			s_task[threadIdx.y].cached = LockType_None;
		}
#endif

#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
		if(s_task[threadIdx.y].cached != LockType_Free) // Free the task if it is in cache
			taskUncacheActive(tid, taskIdx, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
#endif

		s_task[threadIdx.y].lock = LockType_Free;

		//g_taskStackBVH.unfinished = 1;
		//return; // Measure time without enqueue

		ASSERT_DIVERGENCE("taskFinishTask top", tid);

		bool leaf = taskDecideType(tid, &s_task[threadIdx.y]);

		if(!leaf) // Subdivide
		{
			ASSERT_DIVERGENCE("taskFinishTask node", tid);

			// Enqueue the new tasks
#if ENQUEUE_TYPE != 3
			taskEnqueueSubtasks(tid, taskIdx);
#else
			taskEnqueueSubtasksCache(tid, taskIdx);
#endif

			__threadfence(); // The parent shouldn't be updated before the child entries are marked as waiting
			taskCreateInnerNode(tid, &s_task[threadIdx.y]);
		}

		__threadfence(); // The pool entry shouldn't be freed before the parent is updated

#ifdef SPECULATIVE_OPEN
		if(!leaf && tid == 0)
		{
#if !defined(RANDOM_OPEN) && !defined(ALL_OPEN)
			if(s_sharedData[threadIdx.y][0] > 0 && s_sharedData[threadIdx.y][3] > 0)
#else
			if(s_sharedData[threadIdx.y][0] > 0)
#endif
			{
				int pos0 = s_sharedData[threadIdx.y][5];
				int unfinished0 = s_sharedData[threadIdx.y][3];
				//printf("Opening %d: %d\n", pos0, unfinished0);
/*#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
				//if(unfinished0 < POP_MULTIPLIER)
				if((s_task[threadIdx.y].cached != LockType_Free && s_task[threadIdx.y].cached != LockType_Subtask) || s_task[threadIdx.y].pivot > 0)
					taskCacheActive(pos0, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
#endif*/
				g_taskStackBVH.header[pos0] = unfinished0;
			}

#if !defined(RANDOM_OPEN) && !defined(ALL_OPEN)
			if(s_sharedData[threadIdx.y][1] > 0 && s_sharedData[threadIdx.y][4] > 0)
#else
			if(s_sharedData[threadIdx.y][1] > 0)
#endif
			{
				int pos1 = s_sharedData[threadIdx.y][6];
				int unfinished1 = s_sharedData[threadIdx.y][4];
				//printf("Opening %d: %d\n", pos1, unfinished1);
/*#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
				//if(unfinished1 < POP_MULTIPLIER)
				if((s_task[threadIdx.y].cached != LockType_Free && s_task[threadIdx.y].cached != LockType_Subtask) || s_task[threadIdx.y].pivot > 0)
					taskCacheActive(pos1, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
#endif*/
				g_taskStackBVH.header[pos1] = unfinished1;
			}
		}
#endif

		if(tid == 0)
		{
			taskFinishSort(tid, taskIdx);
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

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
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
		SplitArray *splitArray = &(g_redSplits[taskIdx]);
		// Reduced left and right data
		ChildData* left = (ChildData*)&splitArray->splits[tid].children[0];
		ChildData* right = (ChildData*)&splitArray->splits[tid].children[1];
		volatile float* red = (volatile float*)&s_sharedData[threadIdx.y][0];
		red[tid] = CUDART_INF_F;
		
		if(tid < PLANE_COUNT)
		{
#ifdef SPLIT_TEST
			if(left->cnt+right->cnt != triEnd - triStart)
			{
				printf("Failed reduction in task %d (%d x %d)!\n", taskIdx, left->cnt+right->cnt, triEnd - triStart);
				g_taskStackBVH.unfinished = 1;
			}
#endif

			// Compute cost
			CudaAABB bboxLeft;
			bboxLeft.m_mn.x = orderedIntToFloat(left->bbox.m_mn.x);
			bboxLeft.m_mn.y = orderedIntToFloat(left->bbox.m_mn.y);
			bboxLeft.m_mn.z = orderedIntToFloat(left->bbox.m_mn.z);

			bboxLeft.m_mx.x = orderedIntToFloat(left->bbox.m_mx.x);
			bboxLeft.m_mx.y = orderedIntToFloat(left->bbox.m_mx.y);
			bboxLeft.m_mx.z = orderedIntToFloat(left->bbox.m_mx.z);

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

			float leftCnt = (float)left->cnt;
			float leftCost = areaAABB(bboxLeft)*leftCnt;
			CudaAABB bboxRight;
			bboxRight.m_mn.x = orderedIntToFloat(right->bbox.m_mn.x);
			bboxRight.m_mn.y = orderedIntToFloat(right->bbox.m_mn.y);
			bboxRight.m_mn.z = orderedIntToFloat(right->bbox.m_mn.z);

			bboxRight.m_mx.x = orderedIntToFloat(right->bbox.m_mx.x);
			bboxRight.m_mx.y = orderedIntToFloat(right->bbox.m_mx.y);
			bboxRight.m_mx.z = orderedIntToFloat(right->bbox.m_mx.z);

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

			float rightCnt = (float)right->cnt;
			float rightCost = areaAABB(bboxRight)*rightCnt;
			float cost = leftCost + rightCost;

			// Reduce the best cost within the warp
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
			if(triEnd - triStart < c_env.childLimit)
				findPlaneTriAA(tid, c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].triIdxCtr), triStart, triEnd, plane);
			else
				findPlaneAABB(tid, bbox, plane);
#endif

			s_owner[threadIdx.y][0] = -1; // Mark as no split
			// Return the best plane for this warp
			if(__ffs(__ballot(red[tid] == cost)) == tid+1) // First thread with such condition, OPTIMIZE: Can be also computed by overwrite and test: better?
			{
				s_task[threadIdx.y].splitPlane.x = plane.x;
				s_task[threadIdx.y].splitPlane.y = plane.y;
				s_task[threadIdx.y].splitPlane.z = plane.z;
				s_task[threadIdx.y].splitPlane.w = plane.w;
				s_task[threadIdx.y].bestCost = cost;
				s_owner[threadIdx.y][0] = tid;
			}
		}

		if(s_owner[threadIdx.y][0] == -1) // No split found, do object median
		{
			s_task[threadIdx.y].triRight = triStart + (triEnd - triStart) / 2; // Force split on unsubdivided task
			s_task[threadIdx.y].unfinished = taskWarpSubtasksZero(triEnd - triStart);
#ifdef COMPUTE_MEDIAN_BOUNDS
			s_task[threadIdx.y].type = taskChooseAABBType();
#else
			// Copy boxes
			volatile float* bbox = (volatile float*)&(s_task[threadIdx.y].bbox);
			volatile float* bboxCpy = (volatile float*)((tid < sizeof(CudaAABB)/sizeof(float)) ? &s_task[threadIdx.y].bboxLeft : &s_task[threadIdx.y].bboxRight);
			//volatile float* bboxLeft = (volatile float*)&(s_task[threadIdx.y].bboxLeft);
			//volatile float* bboxRight = (volatile float*)&(s_task[threadIdx.y].bboxRight);
			/*if(tid < sizeof(CudaAABB)/sizeof(float))
			{
				bboxLeft[tid] = bbox[tid];
				bboxRight[tid] = bbox[tid];
			}*/
			if(tid < 2*sizeof(CudaAABB)/sizeof(float))
			{
				bboxCpy[tid] = bbox[tid];
			}

			s_task[threadIdx.y].type = taskChooseScanType(s_task[threadIdx.y].unfinished);
#endif
		}
		else
		{
			// Copy boxes
			left = (ChildData*)&splitArray->splits[s_owner[threadIdx.y][0]].children[0];
			right = (ChildData*)&splitArray->splits[s_owner[threadIdx.y][0]].children[1];

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
#endif
		}

		taskPrepareNext(tid, taskIdx, TaskType_BinTriangles);	
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
				findPlaneTriAA(tid, c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].triIdxCtr), s_task[threadIdx.y].triStart, s_task[threadIdx.y].triEnd, plane);
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
			int triRight = triEnd - ((int*)c_bvh_in.ppsTrisBuf)[triEnd - 1]; // Must be volatile
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
			((int*)c_bvh_in.ppsTrisBuf)[triEnd - 1] = 0; // Set the last element to 0 as required by Harris scan

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
			int triRight = triEnd - ((int*)c_bvh_in.ppsTrisBuf)[triEnd - 1]; // Must be volatile
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
#if SPLIT_TYPE < 4 || SPLIT_TYPE > 6
				s_task[threadIdx.y].unfinished = taskWarpSubtasks(triEnd - triStart);
				s_task[threadIdx.y].type = taskChooseAABBType();
				s_task[threadIdx.y].step = 0;
				s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
#else
				s_sharedData[threadIdx.y][0] = -100; // Make the other threads join for the finish
#endif
			}
		}
	}

	//ASSERT_DIVERGENCE("taskFinishSortSORT1 bottom", tid);

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_SORT1);
	}
#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
	else if(s_sharedData[threadIdx.y][0] == -100) // Finish the whole sort
	{
#ifndef DEBUG_INFO
		// Ensure we save valid child box data
		taskLoadSecondFromGMEM(tid, taskIdx, s_task[threadIdx.y]); // Load bbox data
#endif

		taskFinishTask(tid, taskIdx);
	}
#endif
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

		if(triRight == triStart || triRight == triEnd)
		{
			triRight = triStart + (triEnd - triStart) / 2; // Force split on unsubdivided task
			triLeft = triRight;
		}

		s_task[threadIdx.y].triLeft = triLeft;
		s_task[threadIdx.y].triRight = triRight;
		s_task[threadIdx.y].triIdxCtr++; // The output array should be used

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
#if SPLIT_TYPE < 4 || SPLIT_TYPE > 6
			s_task[threadIdx.y].unfinished = taskWarpSubtasks(triEnd - triStart);
			s_task[threadIdx.y].type = taskChooseAABBType();
			s_task[threadIdx.y].step = 0;
			s_sharedData[threadIdx.y][0] = -1; // Make the other threads join for the finish
#else
			s_sharedData[threadIdx.y][0] = -100; // Make the other threads join for the finish
#endif
		}
	}

	if(s_sharedData[threadIdx.y][0] == -1)
	{
		taskPrepareNext(tid, taskIdx, TaskType_Sort_SORT1);
	}
#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
	else if(s_sharedData[threadIdx.y][0] == -100) // Finish the whole sort
	{
#ifndef DEBUG_INFO
		// Ensure we save valid child box data
		taskLoadSecondFromGMEM(tid, taskIdx, s_task[threadIdx.y]); // Load bbox data
#endif

		taskFinishTask(tid, taskIdx);
	}
#endif
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
		float* x = (float*)c_bvh_in.ppsTrisBuf;
		float* y = (float*)c_bvh_in.ppsTrisIndex;
		float* z = (float*)c_bvh_in.sortTris;

		int right = s_task[threadIdx.y].triRight;

		if(s_task[threadIdx.y].type == TaskType_AABB_Min) // Move to Max
		{
			// Save CudaAABB minimum data from gmem to task
			if(start < right)
			{
				s_task[threadIdx.y].bboxLeft.m_mn.x   = x[start];
				s_task[threadIdx.y].bboxLeft.m_mn.y   = y[start];
				s_task[threadIdx.y].bboxLeft.m_mn.z   = z[start];
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
			if(start < right)
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
		// Test that child bounding boxes are inside
		if(tid == 13)
		{
			if(s_task[threadIdx.y].bboxLeft.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x || s_task[threadIdx.y].bboxLeft.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y || s_task[threadIdx.y].bboxLeft.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z
				|| s_task[threadIdx.y].bboxLeft.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x || s_task[threadIdx.y].bboxLeft.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y || s_task[threadIdx.y].bboxLeft.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
			{
				printf("Left child outside! (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f) not in (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)\n",
					s_task[threadIdx.y].bboxLeft.m_mn.x, s_task[threadIdx.y].bboxLeft.m_mn.y, s_task[threadIdx.y].bboxLeft.m_mn.z,
					s_task[threadIdx.y].bboxLeft.m_mx.x, s_task[threadIdx.y].bboxLeft.m_mx.y, s_task[threadIdx.y].bboxLeft.m_mx.z,
					s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
					s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
				//g_taskStackBVH.unfinished = 1;
			}
			
			if(s_task[threadIdx.y].bboxRight.m_mn.x < s_task[threadIdx.y].bbox.m_mn.x || s_task[threadIdx.y].bboxRight.m_mn.y < s_task[threadIdx.y].bbox.m_mn.y || s_task[threadIdx.y].bboxRight.m_mn.z < s_task[threadIdx.y].bbox.m_mn.z
				|| s_task[threadIdx.y].bboxRight.m_mx.x > s_task[threadIdx.y].bbox.m_mx.x || s_task[threadIdx.y].bboxRight.m_mx.y > s_task[threadIdx.y].bbox.m_mx.y || s_task[threadIdx.y].bboxRight.m_mx.z > s_task[threadIdx.y].bbox.m_mx.z)
			{
				printf("Right child outside (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f) not in (%.2f, %.2f, %.2f) - (%.2f, %.2f, %.2f)!\n",
					s_task[threadIdx.y].bboxRight.m_mn.x, s_task[threadIdx.y].bboxRight.m_mn.y, s_task[threadIdx.y].bboxRight.m_mn.z,
					s_task[threadIdx.y].bboxRight.m_mx.x, s_task[threadIdx.y].bboxRight.m_mx.y, s_task[threadIdx.y].bboxRight.m_mx.z,
					s_task[threadIdx.y].bbox.m_mn.x, s_task[threadIdx.y].bbox.m_mn.y, s_task[threadIdx.y].bbox.m_mn.z,
					s_task[threadIdx.y].bbox.m_mx.x, s_task[threadIdx.y].bbox.m_mx.y, s_task[threadIdx.y].bbox.m_mx.z);
				//g_taskStackBVH.unfinished = 1;
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

#if defined(OBJECT_SAH)/* && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)*/

__device__ void taskFinishObjectSplitTree(int tid, int taskIdx)
{
	ASSERT_DIVERGENCE("taskFinishObjectSplitTree top", tid);

	// End the subdivision
	s_sharedData[threadIdx.y][0] = -1;
	s_sharedData[threadIdx.y][1] = -1;

	if(tid == 0)
	{
		taskFinishSort(tid, taskIdx);
	}

	s_task[threadIdx.y].lock = LockType_Free;

#ifdef DEBUG_INFO
	taskSaveFirstToGMEM(tid, taskIdx, s_task[threadIdx.y]); // Make sure results are visible in global memory
#endif

#if PARALLELISM_TEST == 1
	if(tid == 0)
		atomicSub(&g_numActive, 1);
#endif
}

#endif

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
		int* inIdx = getTriIdxPtr(s_task[threadIdx.y].triIdxCtr);
		int triIdx = inIdx[triPos]*3;

		// Fetch triangle
		float3 v0, v1, v2;
		taskFetchTri(c_bvh_in.tris, triIdx, v0, v1, v2);

		CudaAABB dummy;
		int pos = getPlaneCentroidPosition(plane, v0, v1, v2, dummy);
		
		if(pos < 0)
			tb++;
		else
			tf++;
	}
}

//------------------------------------------------------------------------

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
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
		findPlaneTriAA(tid, c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].triIdxCtr), triStart, triEnd, plane);
	else
		findPlaneAABB(tid, bbox, plane);
#endif

	//CudaAABB tbox;
	//int pos;

	int triPos = triStart + subtask*WARP_SIZE;
	for(int i = 0; i < WARP_SIZE && triPos < triEnd; i++, triPos++)
	{
		int* inIdx = getTriIdxPtr(s_task[threadIdx.y].triIdxCtr);
		int triIdx = inIdx[triPos]*3;

		// Fetch triangle
		float3 v0, v1, v2;
		taskFetchTri(triIdx, v0, v1, v2);

		CudaAABB tbox;
		int pos = getPlaneCentroidPosition(plane, v0, v1, v2, tbox);
		//pos = getPlaneCentroidPosition(plane, v0, v1, v2, tbox);

		// Write immediately - LOT OF MEMORY TRANSACTIONS (SLOW) but little memory usage

		// Update the bounding boxes and the counts
		pos = (pos+1) / 2; // Convert from -1,1 to 0,1
		split->children[pos].bbox.m_mn = fminf(split->children[pos].bbox.m_mn, tbox.m_mn-c_env.epsilon);
		split->children[pos].bbox.m_mx = fmaxf(split->children[pos].bbox.m_mx, tbox.m_mx+c_env.epsilon);
		split->children[pos].cnt++;
	}

	// Reduce in thread, write in the end - FAST BUT WITH EXTENSIVE MEMORY USAGE
	// Can be further accelerated by using less splitting planes and parallelizing the for cycle

	// Update the bounding boxes and the counts
	/*pos = (pos+1) / 2; // Convert from -1,1 to 0,1
	split->children[pos].bbox.m_mn = fminf(split->children[pos].bbox.m_mn, tbox.m_mn);
	split->children[pos].bbox.m_mx = fmaxf(split->children[pos].bbox.m_mx, tbox.m_mx);
	split->children[pos].cnt++;*/
}

//------------------------------------------------------------------------

// Compute all data needed for each bin in parallel over triangles and planes
__device__ void binTrianglesParallel(int tid, int subtask, int taskIdx, int triStart, int triEnd, const volatile CudaAABB& bbox, int axis)
{	
	// Compute binning data
	CudaAABB tbox;
	int pos = -1; // Mark inactive threads
	//__shared__ volatile SplitRed out[NUM_WARPS_PER_BLOCK];
	float3 v0, v1, v2;

	int triPos = triStart + subtask*WARP_SIZE + tid;
	if(triPos < triEnd)
	{
		// Fetch triangle
		int* inIdx = getTriIdxPtr(s_task[threadIdx.y].triIdxCtr);
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
			findPlaneTriAA(tid, c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].triIdxCtr), triStart, triEnd, plane);
		else
			findPlaneAABB(tid, bbox, plane);
#endif

		if(triPos < triEnd)
		{
			pos = getPlaneCentroidPosition(plane, v0, v1, v2, tbox);
			pos = (pos+1) / 2; // Convert from -1,1 to 0,1
		}

		// Warpwide update of the left child
		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		// OPTIMIZE: Segmented scan on left and right child simultaneously?
		// Reduce min
		if(pos == 0)
			red[tid] = tbox.m_mn.x - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[0].bbox.m_mn.x)
			split->children[0].bbox.m_mn.x = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.x = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos == 0)
			red[tid] = tbox.m_mn.y - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[0].bbox.m_mn.y)
			split->children[0].bbox.m_mn.y = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.y = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos == 0)
			red[tid] = tbox.m_mn.z - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[0].bbox.m_mn.z)
			split->children[0].bbox.m_mn.z = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.z = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		// Reduce max
		if(pos == 0)
			red[tid] = tbox.m_mx.x + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[0].bbox.m_mx.x)
			split->children[0].bbox.m_mx.x = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.x = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data
		
		if(pos == 0)
			red[tid] = tbox.m_mx.y + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[0].bbox.m_mx.y)
			split->children[0].bbox.m_mx.y = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.y = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos == 0)
			red[tid] = tbox.m_mx.z + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[0].bbox.m_mx.z)
			split->children[0].bbox.m_mx.z = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.z = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		redI[tid] = 0; // Save identities so that we do not work with uninitialized data
		
		// Reduce cnt
		if(pos == 0)
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
		if(pos == 1)
			red[tid] = tbox.m_mn.x - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[1].bbox.m_mn.x)
			split->children[1].bbox.m_mn.x = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.x = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos == 1)
			red[tid] = tbox.m_mn.y - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[1].bbox.m_mn.y)
			split->children[1].bbox.m_mn.y = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.y = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos == 1)
			red[tid] = tbox.m_mn.z - c_env.epsilon;
		reduceWarp(tid, &red[0], min);
		if(red[0] < split->children[1].bbox.m_mn.z)
			split->children[1].bbox.m_mn.z = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mn.z = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		// Reduce max
		if(pos == 1)
			red[tid] = tbox.m_mx.x + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[1].bbox.m_mx.x)
			split->children[1].bbox.m_mx.x = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.x = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos == 1)
			red[tid] = tbox.m_mx.y + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[1].bbox.m_mx.y)
			split->children[1].bbox.m_mx.y = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.y = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		red[tid] = -CUDART_INF_F; // Save identities so that we do not work with uninitialized data

		if(pos == 1)
			red[tid] = tbox.m_mx.z + c_env.epsilon;
		reduceWarp(tid, &red[0], max);
		if(red[0] > split->children[1].bbox.m_mx.z)
			split->children[1].bbox.m_mx.z = red[tid]; // Copy results to gmem
		//out[threadIdx.y].children[pos].bbox.m_mx.z = red[tid]; // Copy results to gmem
		//__threadfence(); // We have to wait for the data to copy to gmem before we overwrite them

		redI[tid] = 0; // Save identities so that we do not work with uninitialized data

		// Reduce cnt
		if(pos == 1)
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
		float4 plane;
#if SPLIT_TYPE == 4
		findPlaneRobin(tid, bbox, axis, plane);
#elif SPLIT_TYPE == 5
		findPlaneAABB(tid, bbox, plane);
#elif SPLIT_TYPE == 6
		if(triEnd - triStart < c_env.childLimit)
			findPlaneTriAA(tid, c_bvh_in.tris, getTriIdxPtr(s_task[threadIdx.y].triIdxCtr), triStart, triEnd, plane);
		else
			findPlaneAABB(tid, bbox, plane);
#endif

		CudaAABB bboxLeft, bboxRight;
		int cntLeft, cntRight;

		// Initialize boxes
		bboxLeft.m_mn.x = bboxLeft.m_mn.y = bboxLeft.m_mn.z = CUDART_INF_F;
		bboxRight.m_mn.x = bboxRight.m_mn.y = bboxRight.m_mn.z = CUDART_INF_F;
		bboxLeft.m_mx.x = bboxLeft.m_mx.y = bboxLeft.m_mx.z = -CUDART_INF_F;
		bboxRight.m_mx.x = bboxRight.m_mx.y = bboxRight.m_mx.z = -CUDART_INF_F;
		cntLeft = cntRight = 0;

		int* inIdx = getTriIdxPtr(s_task[threadIdx.y].triIdxCtr);
		int triPos = triStart + subtaskFirst*WARP_SIZE*BIN_MULTIPLIER;
		int triLast = min(triStart + subtaskLast*WARP_SIZE*BIN_MULTIPLIER, triEnd);
		for(;triPos < triLast; triPos++)
		{
			// OPTIMIZE: Can the data be loaded into shared memory at once by all the threads?
			int triIdx = inIdx[triPos]*3;

			/*if(tid < 12 && (tid % 4) != 3) // Threads corresponding to the data
			{
			volatile float *shared = ((volatile float*)&s_task[threadIdx.y].v0) + tid - (tid / 4);
			float *global = &((float*)c_bvh_in.tris)[triIdx*4] + tid;
			*shared = *global;
			}
			__threadfence_block();*/

			// Fetch triangle
			float3 v0, v1, v2;
			taskFetchTri(c_bvh_in.tris, triIdx, v0, v1, v2);

			CudaAABB tbox;
			int pos = getPlaneCentroidPosition(plane, v0, v1, v2, tbox);
			//int pos = getPlaneCentroidPosition(plane, *((float3*)&s_task[threadIdx.y].v0), *((float3*)&s_task[threadIdx.y].v1), *((float3*)&s_task[threadIdx.y].v2), tbox);

			if(pos == -1)
			{
				bboxLeft.m_mn = fminf(bboxLeft.m_mn, tbox.m_mn/*-c_env.epsilon*/);
				bboxLeft.m_mx = fmaxf(bboxLeft.m_mx, tbox.m_mx/*+c_env.epsilon*/);

				cntLeft++;
			}
			else
			{
				bboxRight.m_mn = fminf(bboxRight.m_mn, tbox.m_mn/*-c_env.epsilon*/);
				bboxRight.m_mx = fmaxf(bboxRight.m_mx, tbox.m_mx/*+c_env.epsilon*/);

				cntRight++;
			}
		}

		// OPTIMIZE: Can possibly be further accelerated by using less splitting planes and parallelizing the for cycle

		// Update the bins
		SplitRed *split = &(g_redSplits[taskIdx].splits[tid]);

		if(cntLeft != 0)
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
				atomicMinFloat(&split->children[0].bbox.m_mn.x, bboxLeft.m_mn.x);
				atomicMinFloat(&split->children[0].bbox.m_mn.y, bboxLeft.m_mn.y);
				atomicMinFloat(&split->children[0].bbox.m_mn.z, bboxLeft.m_mn.z);

				atomicMaxFloat(&split->children[0].bbox.m_mx.x, bboxLeft.m_mx.x);
				atomicMaxFloat(&split->children[0].bbox.m_mx.y, bboxLeft.m_mx.y);
				atomicMaxFloat(&split->children[0].bbox.m_mx.z, bboxLeft.m_mx.z);

				atomicAdd(&split->children[0].cnt, cntLeft);
			}
			else
			{
				split->children[0].bbox.m_mn.x = floatToOrderedInt(bboxLeft.m_mn.x);
				split->children[0].bbox.m_mn.y = floatToOrderedInt(bboxLeft.m_mn.y);
				split->children[0].bbox.m_mn.z = floatToOrderedInt(bboxLeft.m_mn.z);

				split->children[0].bbox.m_mx.x = floatToOrderedInt(bboxLeft.m_mx.x);
				split->children[0].bbox.m_mx.y = floatToOrderedInt(bboxLeft.m_mx.y);
				split->children[0].bbox.m_mx.z = floatToOrderedInt(bboxLeft.m_mx.z);

				split->children[0].cnt = cntLeft;
			}
		}

		if(cntRight != 0)
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
				atomicMinFloat(&split->children[1].bbox.m_mn.x, bboxRight.m_mn.x);
				atomicMinFloat(&split->children[1].bbox.m_mn.y, bboxRight.m_mn.y);
				atomicMinFloat(&split->children[1].bbox.m_mn.z, bboxRight.m_mn.z);

				atomicMaxFloat(&split->children[1].bbox.m_mx.x, bboxRight.m_mx.x);
				atomicMaxFloat(&split->children[1].bbox.m_mx.y, bboxRight.m_mx.y);
				atomicMaxFloat(&split->children[1].bbox.m_mx.z, bboxRight.m_mx.z);

				atomicAdd(&split->children[1].cnt, cntRight);
			}
			else
			{
				split->children[1].bbox.m_mn.x = floatToOrderedInt(bboxRight.m_mn.x);
				split->children[1].bbox.m_mn.y = floatToOrderedInt(bboxRight.m_mn.y);
				split->children[1].bbox.m_mn.z = floatToOrderedInt(bboxRight.m_mn.z);

				split->children[1].bbox.m_mx.x = floatToOrderedInt(bboxRight.m_mx.x);
				split->children[1].bbox.m_mx.y = floatToOrderedInt(bboxRight.m_mx.y);
				split->children[1].bbox.m_mx.z = floatToOrderedInt(bboxRight.m_mx.z);

				split->children[1].cnt = cntRight;
			}
		}
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
		int triidx = (getTriIdxPtr(s_task[threadIdx.y].triIdxCtr))[tripos]*3;

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
		CudaAABB dummy;
		int pos = getPlaneCentroidPosition(*((float4*)&splitPlane), v0, v1, v2, dummy);

		/*float4 plane; // Without typecast, possible data copy
		plane.x = splitPlane.x;
		plane.y = splitPlane.y;
		plane.z = splitPlane.z;
		plane.w = splitPlane.w;
		int pos = getPlanePosition(plane, v0, v1, v2);*/

#if SCAN_TYPE == 3
		// Write to auxiliary array
		((int*)c_bvh_in.ppsTrisIndex)[tripos] = pos;
#endif

		return pos;
	}
	return 2;
}

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

		pps<int>(threadIdx.x, popSubtask, step, (int*)c_bvh_in.ppsTrisIndex, (int*)c_bvh_in.ppsTrisBuf, s_sharedData[threadIdx.y], triStart, triEnd, 1);
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

		scanUp<int>(threadIdx.x, popSubtask, step, (int*)c_bvh_in.ppsTrisBuf, (int*)c_bvh_in.ppsTrisIndex, s_sharedData[threadIdx.y], triStart, triEnd, 1, plus, 0);
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
		scanDown<int>(threadIdx.x, popSubtask, step, (int*)c_bvh_in.ppsTrisBuf, (int*)c_bvh_in.ppsTrisIndex, s_sharedData[threadIdx.y], triStart, triEnd, 1, plus, 0);
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
		sort(threadIdx.x, popSubtask, step, (int*)c_bvh_in.ppsTrisIndex, (int*)c_bvh_in.ppsTrisBuf, (int*)c_bvh_in.sortTris, (int*)c_bvh_in.trisIndex, triStart, triEnd, triRight, 1, false);
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
	int triIdxCtr = s_task[threadIdx.y].triIdxCtr;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;
	bool singleWarp = s_task[threadIdx.y].lock == LockType_None;

	// Set the swap arrays
	int* inIdx = getTriIdxPtr(triIdxCtr);
	int* outIdx = getTriIdxPtr(triIdxCtr+1);

	s_owner[threadIdx.y][0] = triStart;
	s_owner[threadIdx.y][1] = triEnd;

#if SCAN_TYPE == 2
	do
	{
		// Classify the triangles
		int pos = 2; // Outside of the interval
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
			CudaAABB dummy;
			pos = getPlaneCentroidPosition(*((float4*)&s_task[threadIdx.y].splitPlane), v0, v1, v2, dummy);
		}

		// Partition the triangles to the left and right children intervals

		// Scan the number of triangles to the left of the splitting plane
		s_sharedData[threadIdx.y][tid] = 0;
		if(pos == -1)
			s_sharedData[threadIdx.y][tid] = 1;

		scanWarp<int>(tid, s_sharedData[threadIdx.y], plus);
		int exclusiveScan = (s_sharedData[threadIdx.y][tid] - 1);
		
		int triCnt = s_sharedData[threadIdx.y][WARP_SIZE-1];
		if(!singleWarp && tid == 0 && triCnt > 0)
			s_owner[threadIdx.y][0] = atomicAdd(&g_taskStackBVH.tasks[popTaskIdx].triLeft, triCnt); // Add the number of triangles to the left of the plane to the global counter

		// Find the output position for each thread as the sum of the output position and the exclusive scanned value
		if(pos == -1)
			outIdx[s_owner[threadIdx.y][0] + exclusiveScan] = triIdx;
		s_owner[threadIdx.y][0] += triCnt; // Move the position by the number of written nodes

		// Compute the number of triangles to the right of the splitting plane
		int inverseExclusiveScan = tid - s_sharedData[threadIdx.y][tid]; // The scan of the number of triangles to the right of the splitting plane
		
		triCnt = triSum - triCnt;
		if(!singleWarp && tid == 0 && triCnt > 0)
			s_owner[threadIdx.y][1] = atomicSub(&g_taskStackBVH.tasks[popTaskIdx].triRight, triCnt); // Add the number of triangles to the right of the plane to the global counter

		// Find the output position for each thread as the output position minus the triangle count plus the scanned value
		if(pos == 1)
			outIdx[s_owner[threadIdx.y][1] - triCnt + inverseExclusiveScan] = triIdx;
		s_owner[threadIdx.y][1] -= triCnt; // Move the position by the number of written nodes


		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);

	// Write out the final positions
	if(singleWarp)
	{
		g_taskStackBVH.tasks[popTaskIdx].triLeft = s_owner[threadIdx.y][0];
		g_taskStackBVH.tasks[popTaskIdx].triRight = s_owner[threadIdx.y][1];
	}

#elif SCAN_TYPE == 3
	int cntLeft = 0;
	int cntRight = 0;
	int* inPos = (int*)c_bvh_in.ppsTrisIndex;

	do
	{
		// Classify the triangles
		int pos = classifyTri(threadIdx.x, popSubtask, triStart, triEnd, s_task[threadIdx.y].splitPlane);

		// Update the counters
		if(pos == -1)
			cntLeft++;
		else if(pos == 1)
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

		int pos = 2; // Outside of the interval
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
		if(pos == -1)
			s_sharedData[threadIdx.y][tid] = 1;

		scanWarp(tid, s_sharedData[threadIdx.y], plus);
		int exclusiveScan = (s_sharedData[threadIdx.y][tid] - 1);
		int triCnt = s_sharedData[threadIdx.y][WARP_SIZE-1];

		// Find the output position for each thread as the sum of the output position and the exclusive scanned value
		if(pos == -1)
			outIdx[warpAtomicLeft + exclusiveScan] = triIdx;
		warpAtomicLeft += triCnt; // Move the position by the number of written nodes

		// Compute the number of triangles to the right of the splitting plane
		int inverseExclusiveScan = tid - s_sharedData[threadIdx.y][tid]; // The scan of the number of triangles to the right of the splitting plane

		triCnt = triSum - triCnt;
		// Find the output position for each thread as the output position minus the triangle count plus the scanned value
		if(pos == 1)
			outIdx[warpAtomicRight - triCnt + inverseExclusiveScan] = triIdx;
		warpAtomicRight -= triCnt; // Move the position by the number of written nodes

		subtasksDone = taskReduceSubtask(popSubtask, popStart, popCount);
	} while(subtasksDone == -1);
#endif

	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
	taskFinishPartition(tid, popTaskIdx, subtasksDone);
}

//------------------------------------------------------------------------

#if AABB_TYPE == 0

// Local + global reduction on x, y and z coordinates at once
// OPTIMIZE: It could be more efficient to pass op as template parameter instead of function pointer?
template<typename T>
__device__ void computeAABB(int tid, int taskIdx, int subtask, int step, int start, int right, int end, T(*op)(T,T), T identity, T epsilon)
{
	// Data arrays
	T* x = (T*)c_bvh_in.ppsTrisBuf;
	T* y = (T*)c_bvh_in.ppsTrisIndex;
	T* z = (T*)c_bvh_in.sortTris;

	if(step == 0) // Do local reduction, step 0
	{
		ASSERT_DIVERGENCE("computeAABB step0 top", tid);

		int tripos = start + (subtask * WARP_SIZE) + tid;
		volatile T* red = (volatile T*)&s_sharedData[threadIdx.y][0];
		s_owner[threadIdx.y][tid] = 31;     // Mark as outside array

		ASSERT_DIVERGENCE("computeAABB step0 if1", tid);

		if(tripos < end)
		{
			s_owner[threadIdx.y][tid] = ((tripos - right >= 0));

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
			int* inIdx = getTriIdxPtr(s_task[threadIdx.y].triIdxCtr);
			int triidx = inIdx[tripos]*3;

			// Fetch triangle
			float3 v0, v1, v2;
			taskFetchTri(triidx, v0, v1, v2);

			// Reduce x
			red[tid] = op(op(v0.x, v1.x), v2.x)+epsilon; // OPTIMIZE: Do triangle bounding box computation once and load only the result
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
			red[tid] = op(op(v0.y, v1.y), v2.y)+epsilon; // OPTIMIZE: Do triangle bounding box computation once and load only the result
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
			red[tid] = op(op(v0.z, v1.z), v2.z)+epsilon; // OPTIMIZE: Do triangle bounding box computation once and load only the result
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
			int ownerThis = ((posThis - right >= 0));
			int ownerNext = ((posNext - right >= 0));

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
				int posDivide = right;
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
	__threadfence(); // Probably needed so that next iteration does not read uninitialized data
}

#elif AABB_TYPE == 1

// Local reduction on x, y and z coordinates at once
// OPTIMIZE: It could be more efficient to pass op as template parameter instead of function pointer?
template<typename T>
__device__ void computeAABB(int tid, int taskIdx, int subtask, int step, int start, int right, int end, T(*op)(T,T), T identity, T epsilon)
{
	// Data arrays
	T* x = (T*)c_bvh_in.ppsTrisBuf;
	T* y = (T*)c_bvh_in.ppsTrisIndex;
	T* z = (T*)c_bvh_in.sortTris;

	int i          = (subtask * WARP_SIZE) + tid;
	int blockSize  = (1 << step);
	int blockStart = start + blockSize * i;
	int pos = blockStart;

	ASSERT_DIVERGENCE("computeAABB top", tid);

	volatile T* red = (volatile T*)&s_sharedData[threadIdx.y][0];
	// Mark as outside array
	red[tid] = (T)31; // Using same memory should not be a problem
	s_owner[threadIdx.y][tid] = 31;

	ASSERT_DIVERGENCE("computeAABB if1", tid);

	if(pos < end)
	{
		s_owner[threadIdx.y][tid] = ((pos - right >= 0));

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
			int* inIdx = getTriIdxPtr(s_task[threadIdx.y].triIdxCtr);
			int triidx = inIdx[pos]*3;
			taskFetchTri(triidx, v0, v1, v2);

			red[tid] = op(op(v0.x, v1.x), v2.x)+epsilon; // OPTIMIZE: Do triangle bounding box computation once and load only the result
			//int triidx = ((int*)c_bvh_in.trisIndex)[pos];
			//if(s_task[threadIdx.y].type == TaskType_AABB_Min)
			//	vec = (((CudaAABB*)c_bvh_in.trisBox)[triidx]).m_mn;
			//else
			//	vec = (((CudaAABB*)c_bvh_in.trisBox)[triidx]).m_mx;

			//red[tid] = vec.x;
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
			int posDivide = right;
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
			red[tid] = op(op(v0.y, v1.y), v2.y)+epsilon; // OPTIMIZE: Do triangle bounding box computation once and load only the result
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
			int posDivide = right;
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
			red[tid] = op(op(v0.z, v1.z), v2.z)+epsilon; // OPTIMIZE: Do triangle bounding box computation once and load only the result
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
			int posDivide = right;
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

// Local reduction on x, y and z coordinates at once
// OPTIMIZE: It could be more efficient to pass op as template parameter instead of function pointer?
template<typename T>
__device__ void computeAABB(int tid, int taskIdx, int subtask, int step, int start, int right, int end, T(*op)(T,T), T identity, T epsilon)
{
	// Data arrays
	T* x = (T*)c_bvh_in.ppsTrisBuf;
	T* y = (T*)c_bvh_in.ppsTrisIndex;
	T* z = (T*)c_bvh_in.sortTris;

	int i          = (subtask * WARP_SIZE) + tid;
	int blockSize  = (1 << step);
	int halfBlock  = blockSize >> 1;
	int blockStart = start + blockSize * i;
	int pos = blockStart;
	int posNext = blockStart + halfBlock;

	ASSERT_DIVERGENCE("computeAABB top", tid);

	volatile T* red = (volatile T*)&s_sharedData[threadIdx.y][0];
	// Mark as outside array
	red[tid] = (T)31; // Using same memory should not be a problem
	s_owner[threadIdx.y][tid] = 31;

	ASSERT_DIVERGENCE("computeAABB if1", tid);

	if(pos < end)
	{
		s_owner[threadIdx.y][tid] = ((pos - right >= 0));

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
			int* inIdx = getTriIdxPtr(s_task[threadIdx.y].triIdxCtr);
			int triidx = inIdx[pos]*3;
			taskFetchTri(triidx, v0, v1, v2);

			red[tid] = op(op(v0.x, v1.x), v2.x)+epsilon; // OPTIMIZE: Do triangle bounding box computation once and load only the result
			//int triidx = ((int*)c_bvh_in.trisIndex)[pos];
			//if(s_task[threadIdx.y].type == TaskType_AABB_Min)
			//	vec = (((CudaAABB*)c_bvh_in.trisBox)[triidx]).m_mn;
			//else
			//	vec = (((CudaAABB*)c_bvh_in.trisBox)[triidx]).m_mx;

			//red[tid] = vec.x;
		}
		else if(posNext < end && (posNext < right || pos >= right)) // Same owners
		{
			red[tid] = op(x[pos], x[posNext]);
		}
		else
		{
			red[tid] = x[pos];
			if(posNext < end)
			{
				int posDivide = right;
				x[posDivide] = op(x[posDivide], x[posNext]);
			}
		}
		segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);

		if(tid == 0) // Start of first segment in shared memory
		{
			x[pos] = red[tid]; // Copy results to gmem
		}
		else if(tid == s_owner[threadIdx.y][tid]) // Start of other segments in shared memory
		{
			int posDivide = right;
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
			red[tid] = op(op(v0.y, v1.y), v2.y)+epsilon; // OPTIMIZE: Do triangle bounding box computation once and load only the result
			//red[tid] = vec.y;
		}
		else if(posNext < end && (posNext < right || pos >= right))
		{
			red[tid] = op(y[pos], y[posNext]);
		}
		else
		{
			red[tid] = y[pos];
			if(posNext < end)
			{
				int posDivide = right;
				y[posDivide] = op(y[posDivide], y[posNext]);
			}
		}
		segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);

		if(tid == 0) // Start of first segment in shared memory
		{
			y[pos] = red[tid]; // Copy results to gmem
		}
		else if(tid == s_owner[threadIdx.y][tid]) // Start of other segments in shared memory
		{
			int posDivide = right;
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
			red[tid] = op(op(v0.z, v1.z), v2.z)+epsilon; // OPTIMIZE: Do triangle bounding box computation once and load only the result
			//red[tid] = vec.z;
		}
		else if(posNext < end && (posNext < right || pos >= right))
		{
			red[tid] = op(z[pos], z[posNext]);
		}
		else
		{
			red[tid] = z[pos];
			if(posNext < end)
			{
				int posDivide = right;
				z[posDivide] = op(z[posDivide], z[posNext]);
			}
		}
		segReduceWarp(tid, &red[0], &s_owner[threadIdx.y][0], op);

		if(tid == 0) // Start of first segment in shared memory
		{
			z[pos] = red[tid]; // Copy results to gmem
		}
		else if(tid == s_owner[threadIdx.y][tid]) // Start of other segments in shared memory
		{
			int posDivide = right;
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
	int right = s_task[threadIdx.y].triRight;
	int step = s_task[threadIdx.y].step;
	int triIdxCtr = s_task[threadIdx.y].triIdxCtr;

	int popStart = s_task[threadIdx.y].popStart;
	int popSubtask = s_task[threadIdx.y].popSubtask;
	int popCount = s_task[threadIdx.y].popCount;
	int popTaskIdx = s_task[threadIdx.y].popTaskIdx;

	int floatsPerAABB = (sizeof(CudaAABB)/sizeof(float));
	// Data arrays - only one is used in each phase.
	T* b0 = (T*)c_bvh_in.ppsTrisBuf;
	T* b1 = (T*)c_bvh_in.ppsTrisIndex;

	// Array for holding partial results caused by segmented reduction - one box (right-end interval)
	T* st = b1; // Use first item of the input/output array

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
		int inPos      = start + warpPos*floatsPerAABB + floatsPerAABB; // Data are serialized at the beginning of the array
		int outPos     = start + popSubtask*floatsPerAABB + floatsPerAABB + tid; // Threads cooperate in a single CudaAABB store
		int stPos      = start + tid; // Threads cooperate in a single CudaAABB store

		volatile CudaAABB& seg0 = (firstBlock < right) ? s_task[threadIdx.y].bboxLeft : s_task[threadIdx.y].bboxRight;
		volatile CudaAABB& seg1 = s_task[threadIdx.y].bboxRight;

		ASSERT_DIVERGENCE("computeAABB top", tid);

		volatile T* redX = (volatile T*)&s_sharedData[threadIdx.y][0];
		volatile T* redY = (volatile T*)&s_newTask[threadIdx.y];
		volatile T* redZ = ((volatile T*)&s_newTask[threadIdx.y])+WARP_SIZE;
		// Mark as outside array
		redX[tid] = (T)31; // Using same memory should not be a problem
		s_owner[threadIdx.y][tid] = 31;
		bool segmented = false;
		bool lastLevel = (popSubtask == 0 && s_task[threadIdx.y].origSize == 1); // This call computes the last level of the tree

		ASSERT_DIVERGENCE("computeAABB if1", tid);

		if(pos < end)
		{
			s_owner[threadIdx.y][tid] = ((pos - right >= 0));

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

		// Load the start of the segment if this warp crosses its boundary
		// OPTIMIZE: If owner1 is start of segment, it will be repeatedly saved twice instead of once
		if(step != 0 && (segmented || lastLevel))
		{
			volatile float* bbox = (volatile float*)&seg1;
			if(tid < floatsPerAABB)
			{
				bbox[tid] = st[stPos]; // Load the data to shared memory
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
				int* inIdx = getTriIdxPtr(triIdxCtr);
				int triidx = inIdx[pos]*3;
				taskFetchTri(c_bvh_in.tris, triidx, v0, v1, v2);

				redX[tid] = fminf(min(v0.x, v1.x), v2.x)-c_env.epsilon;
				redY[tid] = fminf(min(v0.y, v1.y), v2.y)-c_env.epsilon;
				redZ[tid] = fminf(min(v0.z, v1.z), v2.z)-c_env.epsilon;
			
				// OPTIMIZE: Do triangle bounding box computation once and load only the result
				//int triidx = ((int*)c_bvh_in.trisIndex)[pos];
				//if(s_task[threadIdx.y].type == TaskType_AABB_Min)
				//	vec = (((CudaAABB*)c_bvh_in.trisBox)[triidx]).m_mn;
				//else
				//	vec = (((CudaAABB*)c_bvh_in.trisBox)[triidx]).m_mx;

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
				if(step == 0)
				{
					seg1.m_mn.x = redX[tid];
					seg1.m_mn.y = redY[tid];
					seg1.m_mn.z = redZ[tid];
				}
				else
				{
					seg1.m_mn.x = fminf(seg1.m_mn.x, redX[tid]);
					seg1.m_mn.y = fminf(seg1.m_mn.y, redY[tid]);
					seg1.m_mn.z = fminf(seg1.m_mn.z, redZ[tid]);
				}
			}

#ifdef BBOX_TEST
			if(step == 0)
			{
				float thm = min(min(v0.x, v1.x), v2.x);
				if(thm < redX[s_owner[threadIdx.y][tid]])
					printf("Min x red error %f!\n", thm);
				if(thm < s_task[threadIdx.y].bbox.m_mn.x)
					printf("Min x step0 bound error task %d!\n", popTaskIdx);

				thm = min(min(v0.y, v1.y), v2.y);
				if(thm < redY[s_owner[threadIdx.y][tid]])
					printf("Min y red error %f!\n", thm);
				if(thm < s_task[threadIdx.y].bbox.m_mn.y)
					printf("Min y step0 bound error task %d!\n", popTaskIdx);

				thm = min(min(v0.z, v1.z), v2.z);
				if(thm < redZ[s_owner[threadIdx.y][tid]])
					printf("Min z red error %f!\n", thm);
				if(thm < s_task[threadIdx.y].bbox.m_mn.z)
					printf("Min z step0 bound error task %d!\n", popTaskIdx);
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
				redX[tid] = fmaxf(max(v0.x, v1.x), v2.x)+c_env.epsilon;
				redY[tid] = fmaxf(max(v0.y, v1.y), v2.y)+c_env.epsilon;
				redZ[tid] = fmaxf(max(v0.z, v1.z), v2.z)+c_env.epsilon;
			
				// OPTIMIZE: Do triangle bounding box computation once and load only the result
				//int triidx = ((int*)c_bvh_in.trisIndex)[pos];
				//if(s_task[threadIdx.y].type == TaskType_AABB_Min)
				//	vec = (((CudaAABB*)c_bvh_in.trisBox)[triidx]).m_mn;
				//else
				//	vec = (((CudaAABB*)c_bvh_in.trisBox)[triidx]).m_mx;

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
				if(step == 0)
				{
					seg1.m_mx.x = redX[tid];
					seg1.m_mx.y = redY[tid];
					seg1.m_mx.z = redZ[tid];
				}
				else
				{
					seg1.m_mx.x = fmaxf(seg1.m_mx.x, redX[tid]);
					seg1.m_mx.y = fmaxf(seg1.m_mx.y, redY[tid]);
					seg1.m_mx.z = fmaxf(seg1.m_mx.z, redZ[tid]);
				}
			}

#ifdef BBOX_TEST
			if(step == 0)
			{
				float thm = max(max(v0.x, v1.x), v2.x);
				if(thm > redX[s_owner[threadIdx.y][tid]])
					printf("Max x red error %f!\n", thm);
				if(thm > s_task[threadIdx.y].bbox.m_mx.x)
					printf("Max x step0 bound error task %d!\n", popTaskIdx);
			
				thm = max(max(v0.y, v1.y), v2.y);
				if(thm > redY[s_owner[threadIdx.y][tid]])
					printf("Max y error %f!\n", thm);
				if(thm > s_task[threadIdx.y].bbox.m_mx.y)
					printf("Max y step0 bound error task %d!\n", popTaskIdx);

				thm = max(max(v0.z, v1.z), v2.z);
				if(thm > redZ[s_owner[threadIdx.y][tid]])
					printf("Max z error %f!\n", thm);
				if(thm > s_task[threadIdx.y].bbox.m_mx.z)
					printf("Max z step0 bound error task %d!\n", popTaskIdx);
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
			if((step == 0 && firstBlock == right) || segmented)
			{
				if(tid < floatsPerAABB)
				{
					volatile float* bbox = (volatile float*)&seg1;
					st[stPos] = bbox[tid]; // Write the data to global memory
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
		int* inIdx = getTriIdxPtr(s_task[threadIdx.y].triIdxCtr);
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
		taskSaveNodeToGMEM(g_bvh, tid, nodeIdx, *node);

#if SCAN_TYPE == 2 || SCAN_TYPE == 3
		// Back-copy triangles to the correct array
		backcopy(tid, s_task[threadIdx.y].triIdxCtr, triStart, triEnd);
#endif
		__threadfence(); // Make sure the node is in the hierarchy before we update the parent

		if(tid == 0)
		{
			taskUpdateParentPtr(g_bvh, parentIdx, taskID, ~nodeIdx); // Mark this node as leaf in the hierarchy	
#ifdef BVH_COUNT_NODES
			// Beware this node has a preallocated space and is thus also counted as an inner node
			atomicAdd(&g_taskStackBVH.numNodes, 1);
			atomicAdd(&g_taskStackBVH.numLeaves, 1);
			atomicAdd(&g_taskStackBVH.numSortedTris, triEnd - triStart);
			//printf("Split Leaf (%d, %d)\n", triStart, triEnd);
#endif
		}

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
					g_bvh[nodeIdx].c0xy = make_float4(bbox.m_mn.x, bbox.m_mx.x, bbox.m_mn.y, bbox.m_mx.y);
					g_bvh[nodeIdx].c1xy = make_float4(bbox.m_mn.x, bbox.m_mx.x, bbox.m_mn.y, bbox.m_mx.y);
					g_bvh[nodeIdx].c01z = make_float4(bbox.m_mn.z, bbox.m_mx.z, bbox.m_mn.z, bbox.m_mx.z);
					g_bvh[nodeIdx].children = make_int4(triStart+segmentStart, triStart+segmentEnd, parentIdx, 0); // Sets the leaf child pointers

					taskUpdateParentPtr(g_bvh, parentIdx, taskID, ~nodeIdx); // Mark this node as finished in the hierarchy
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
				g_bvh[childLeft].c0xy = make_float4(bboxLeft.m_mn.x, bboxLeft.m_mx.x, bboxLeft.m_mn.y, bboxLeft.m_mx.y);
				g_bvh[childLeft].c1xy = make_float4(bboxLeft.m_mn.x, bboxLeft.m_mx.x, bboxLeft.m_mn.y, bboxLeft.m_mx.y);
				g_bvh[childLeft].c01z = make_float4(bboxLeft.m_mn.z, bboxLeft.m_mx.z, bboxLeft.m_mn.z, bboxLeft.m_mx.z);
				g_bvh[childLeft].children = make_int4(triStart+segmentStart, triStart+bestPos, parentIdx, 0);

				childLeft = ~childLeft;
			}

			if(tid == segmentStart && segmentEnd-bestPos <= c_env.triLimit) // Only the first thread of the segment may write the leafs because it may be the only thread in the segment
			{
				// Leaf -> same bounding boxes
				g_bvh[childRight].c0xy = make_float4(bboxRight.m_mn.x, bboxRight.m_mx.x, bboxRight.m_mn.y, bboxRight.m_mx.y);
				g_bvh[childRight].c1xy = make_float4(bboxRight.m_mn.x, bboxRight.m_mx.x, bboxRight.m_mn.y, bboxRight.m_mx.y);
				g_bvh[childRight].c01z = make_float4(bboxRight.m_mn.z, bboxRight.m_mx.z, bboxRight.m_mn.z, bboxRight.m_mx.z);
				g_bvh[childRight].children = make_int4(triStart+bestPos, triStart+segmentEnd, parentIdx, 0);

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

				g_bvh[nodeIdx].c0xy = make_float4(bboxLeft.m_mn.x, bboxLeft.m_mx.x, bboxLeft.m_mn.y, bboxLeft.m_mx.y);
				g_bvh[nodeIdx].c1xy = make_float4(bboxRight.m_mn.x, bboxRight.m_mx.x, bboxRight.m_mn.y, bboxRight.m_mx.y);
				g_bvh[nodeIdx].c01z = make_float4(bboxLeft.m_mn.z, bboxLeft.m_mx.z, bboxRight.m_mn.z, bboxRight.m_mx.z);
#ifndef COMPACT_LAYOUT
				g_bvh[nodeIdx].children = make_int4(childLeft, childRight, parentIdx, 0); // Sets the leaf child pointers
#else
				g_bvh[nodeIdx].children = make_int4(childLeft*64, childRight*64, parentIdx, 0); // Sets the leaf child pointers
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

#ifdef BVH_COUNT_NODES
	int leafCount = 0;
	if(tid < end)
		leafCount = (tid == segmentEnd-1) ? 1 : 0; // Count the number of vertices + the number of segments
	reduceWarp<int>(leafCount, plus);

	if(tid == 0)
	{
		atomicAdd(&g_taskStackBVH.numNodes, leafCount-1);
		atomicAdd(&g_taskStackBVH.numSortedTris, __log2f(leafCount)*(s_task[threadIdx.y].triEnd - s_task[threadIdx.y].triStart));
		atomicAdd(&g_taskStackBVH.numLeaves, leafCount);
	}
#endif

	// Link this subtree to the already build part of the hierarchy
	__threadfence(); // Make sure the node is in the hierarchy before we update the parent
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

	// Link this subtree to the already build part of the hierarchy
	__threadfence(); // Make sure the node is in the hierarchy before we update the parent

	// Update the parents with the starts of the triangle list
	if(tid < end && tid == segmentStart)
	{
		//printf("Segment %d: Parent %d <- node %d (%d)!\n", segmentStart, parentIdx, triPosition, taskID);
		taskUpdateParentPtr(g_bvh, parentIdx, taskID, ~triPosition); // Mark this node as leaf in the hierarchy
#ifdef LEAF_HISTOGRAM
		atomicAdd(&g_taskStackBVH.leafHist[segmentEnd-segmentStart], 1); // Update histogram
#endif
	}
#endif // COMPACT_LAYOUT

	ASSERT_DIVERGENCE("computeObjectSplitTree bottom", tid);

	// Link the root of the subtree
	if(tid == 0 && segmentEnd != end) // Do not write the inner node ptr if there is a single leaf. The leaf is updated by the previous if.
	{
#ifndef COMPACT_LAYOUT
		taskUpdateParentPtr(g_bvh, s_task[threadIdx.y].parentIdx, s_task[threadIdx.y].taskID, s_task[threadIdx.y].nodeIdx); // Mark this node as finished in the hierarchy
#else
		taskUpdateParentPtr(g_bvh, s_task[threadIdx.y].parentIdx, s_task[threadIdx.y].taskID, s_task[threadIdx.y].nodeIdx*64); // Mark this node as finished in the hierarchy
#endif
	}

	taskFinishObjectSplitTree(tid, s_task[threadIdx.y].popTaskIdx);
}
#endif

//------------------------------------------------------------------------

// Unlocks a BVH build task in the global memory
__device__ void buildNode(int taskIdx, int nodeIdx, int parentIdx, int childOffset, bool traverse)
{

#if defined(ALL_OPEN)
#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
	int lock = atomicCAS(&g_taskStackBVH.tasks[taskIdx].cached, LockType_Free, LockType_Set);

	if(lock == LockType_Free) // The thread unlocked the task
	{
		taskCacheActive(taskIdx, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
		g_taskStackBVH.tasks[taskIdx].cached = LockType_None;
		//atomicExch(&g_taskStackBVH.tasks[taskIdx].cached, LockType_None);
		//atomicCAS(&g_taskStackBVH.tasks[taskIdx].cached, LockType_Set, LockType_None);
	}
#endif
#else
#ifdef SPECULATIVE_OPEN
	if(taskIdx == UNBUILD_UNMASK)
		return;
#endif

	int unfinished = *((int*)&g_taskStackBVH.tasks[taskIdx].unfinished); // Read the amount of work to activate the entry with

/*#ifdef SPECULATIVE_OPEN
	if(!traverse && unfinished > c_env.siblingLimit)
		return;
#endif*/

#if 0
	// Many threads in the warp may want to build the same nodes -> atomic conflicts
	s_owner[threadIdx.y][threadIdx.x] = taskIdx;
	for(int i = threadIdx.x+1; i < WARP_SIZE; i++)
	{
		if(s_owner[threadIdx.y][i] == taskIdx) // Duplicate
		{
			s_owner[threadIdx.y][i] = -1;
			break; // Further duplicates will be dealt with by the thread we have just eliminated
		}
	}

	if(s_owner[threadIdx.y][threadIdx.x] == taskIdx) // Last man standing
#elif 0
	bool taskUndone = true;
	//bool onlySet = false;

	// Many threads in the warp may want to build the same nodes -> atomic conflicts
	while(taskUndone)
	{
		s_sharedData[threadIdx.y][0] = taskIdx;
		if(s_sharedData[threadIdx.y][0] == taskIdx) // Process this taskIdx
		{
			taskUndone = false;
			s_sharedData[threadIdx.y][1] = threadIdx.x;
			if(s_sharedData[threadIdx.y][1] == threadIdx.x) // This thread processes the task
			{
				// Activate the task if it haven't already been activated
				int active = atomicCAS(&g_taskStackBVH.header[taskIdx], TaskHeader_Waiting | nodeIdx, unfinished); // Unlock the task, waiting flag bear the nodeIdx to customize the unlock.
				// This prevents opening a different waiting task, which wreaks havok in the computation.
				if(active == (TaskHeader_Waiting | nodeIdx)) // We have activated the task
				{
#ifdef SPECULATIVE_OPEN
					*(((int*)&g_bvh[parentIdx].children) + childOffset) = UNBUILD_UNMASK | UNBUILD_FLAG; // Mark the node as unlocked
#endif

#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
					taskCacheActive(taskIdx, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
#endif

#if PARALLELISM_TEST == 1
					atomicAdd(&g_numActive, 1);
#endif
				}
			}
		}
	}

	//if(onlySet)
#else
	// COMMENT: This should not work as different nodes may want to be unlocked at the same time
	//s_sharedData[threadIdx.y][0] = threadIdx.x;
	//if(s_sharedData[threadIdx.y][0] == threadIdx.x) // This thread processes the task
#endif
	{
		// Activate the task if it haven't already been activated
		int active = atomicCAS(&g_taskStackBVH.header[taskIdx], TaskHeader_Waiting | nodeIdx, unfinished); // Unlock the task, waiting flag bear the nodeIdx to customize the unlock.
																										   // This prevents opening a different waiting task, which wreaks havok in the computation.
		if(active == (TaskHeader_Waiting | nodeIdx)) // We have activated the task
		{
/*#ifdef SPECULATIVE_OPEN
			*(((int*)&g_bvh[parentIdx].children) + childOffset) = UNBUILD_UNMASK | UNBUILD_FLAG; // Mark the node as unlocked
#endif*/

#if DEQUEUE_TYPE == 3 || DEQUEUE_TYPE == 5 || DEQUEUE_TYPE == 6
			taskCacheActive(taskIdx, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
#endif

#if PARALLELISM_TEST == 1
			atomicAdd(&g_numActive, 1);
#endif
		}
	}

#endif // ALL_OPEN
}

//------------------------------------------------------------------------

// Checks whether the node has already been build
__device__ __noinline__ void checkBuildDone(int& nodeAddr) // Must be noinline to avoid warp divergence
{
	if((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG) // The poped node was not build at the time of push
	{
#ifdef COUNT_INTERUPTS
		numSteps[threadIdx.y]++;
#endif
		int cleanAddr = nodeAddr & UNBUILD_UNMASK; // Node address without the flags
		int field = (nodeAddr >> 28) & 0x3; // The field in the link vector

#ifndef COMPACT_LAYOUT
		int newAddr = *(int*)((CUdeviceptr)g_bvh + cleanAddr*sizeof(CudaBVHNode) + 48 + field*sizeof(int));
#else
		int newAddr = *(int*)((CUdeviceptr)g_bvh + cleanAddr + 48 + field*sizeof(int));
#endif

		if((newAddr & UNBUILD_MASK) != UNBUILD_FLAG) // If any thread wants to traverse unbuild node
			nodeAddr = newAddr;
		
#ifdef COUNT_INTERUPTS
		if(__any((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG))
			sumSteps[threadIdx.y]++;
#endif

		//if(threadIdx.x == 0)
		//printf("Node 0 built nodeAddr %d, cleanAddr %d, field %d, newAddr %d\n", nodeAddr, cleanAddr, field, newAddr);
	}
}

//------------------------------------------------------------------------

// Unlocks a BVH build task in the global memory
__device__ void popStack(int& nodeAddr, int* &stackPtr)
{
	nodeAddr = *stackPtr;
	--stackPtr;

	// Do not synchronize rays through call to noinline function "checkBuildDone(nodeAddr);"
	// Produces 4x faster traversal code!!!
	if((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG) // The poped node was not build at the time of push
	{
#ifdef COUNT_INTERUPTS
		numSteps[threadIdx.y]++;
#endif
		int cleanAddr = nodeAddr & UNBUILD_UNMASK; // Node address without the flags
		int field = (nodeAddr >> 28) & 0x3; // The field in the link vector

		int newAddr = *(int*)((CUdeviceptr)g_bvh + cleanAddr*sizeof(CudaBVHNode) + 48 + field*sizeof(int));

		if((newAddr & UNBUILD_MASK) != UNBUILD_FLAG) // If any thread wants to traverse unbuild node
			nodeAddr = newAddr;

#ifdef COUNT_INTERUPTS
		if(__any((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG))
			sumSteps[threadIdx.y]++;
#endif

		//if(threadIdx.x == 0)
		//printf("Node 0 built nodeAddr %d, cleanAddr %d, field %d, newAddr %d\n", nodeAddr, cleanAddr, field, newAddr);
	}
}

//------------------------------------------------------------------------
#if !defined(COMPACT_LAYOUT) && !defined(WOOP_TRIANGLES)

// Traverses the yet build BVH and returns the index of the task to further subdivide
/*__device__ __noinline__ void traverse(const int& rayIdx, const float3& orig, const float3& dir, const float3& idir, const float3& ood, float& tmin, int& nodeAddr, int* &stackPtr, int& hitIndex, float& hitT, float& hitU, float& hitV) // Must be noinline to avoid warp divergence
{
	ASSERT_DIVERGENCE("traverse top", threadIdx.x);

	// Process rays in a loop.
	while(nodeAddr != EntrypointSentinel)
	{
		while(nodeAddr >= 0 && nodeAddr < EntrypointSentinel)
		{
			// Fetch AABBs of the two child nodes.

			CudaBVHNode node;
			taskFetchNode((CUdeviceptr)g_bvh, nodeAddr, node);

			// Intersect the ray against the child nodes.

			float c0lox = node.c0xy.x * idir.x - ood.x;
			float c0hix = node.c0xy.y * idir.x - ood.x;
			float c0loy = node.c0xy.z * idir.y - ood.y;
			float c0hiy = node.c0xy.w * idir.y - ood.y;
			float c0loz = node.c01z.x * idir.z - ood.z;
			float c0hiz = node.c01z.y * idir.z - ood.z;
			float c1loz = node.c01z.z * idir.z - ood.z;
			float c1hiz = node.c01z.w * idir.z - ood.z;
			float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), tmin);
			float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
			float c1lox = node.c1xy.x * idir.x - ood.x;
			float c1hix = node.c1xy.y * idir.x - ood.x;
			float c1loy = node.c1xy.z * idir.y - ood.y;
			float c1hiy = node.c1xy.w * idir.y - ood.y;
			float c1min = max4(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), tmin);
			float c1max = min4(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz), hitT);

			// Decide where to go next.

			bool traverseChild0 = (c0max >= c0min);
			bool traverseChild1 = (c1max >= c1min);

			// Neither child was intersected => pop stack.

			if(!traverseChild0 && !traverseChild1)
            {
                popStack(nodeAddr, stackPtr);
            }

            // Otherwise => fetch child pointers.

            else
            {
#ifndef SPECULATIVE_OPEN
				if(traverseChild0 && (node.children.x & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild left child
#else
				if((node.children.x & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild left child
#endif
				{
					buildNode(node.children.x & UNBUILD_UNMASK, node.children.w+1, nodeAddr, 0, traverseChild0); // Clear the unbuild flag to get the pool entry
					node.children.x = nodeAddr | (UNBUILD_FLAG | 0x0); // Save the code of the location where to check for completion
				}

#ifndef SPECULATIVE_OPEN
				if(traverseChild1 && (node.children.y & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild right child
#else
				if((node.children.y & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild right child
#endif
				{
					buildNode(node.children.y & UNBUILD_UNMASK, node.children.w+2, nodeAddr, 1, traverseChild1); // Clear the unbuild flag to get the pool entry
					node.children.y = nodeAddr | (UNBUILD_FLAG | 0x10000000); // Save the code of the location where to check for completion
				}

                nodeAddr = (traverseChild0) ? node.children.x : node.children.y;

                // Both children were intersected => push the farther one.

                if(traverseChild0 && traverseChild1)
                {
					if(c1min < c0min)
                        swap(nodeAddr, node.children.y);
                    
					// Place child pointer onto the stack
					++stackPtr;
					*stackPtr = node.children.y; // Save the BVH index
                }
            }

			//if(nodeAddr == 1073741824 || nodeAddr == 1073741825)
			//	printf("Node %d, left %d, right %d\n", nodeAddr, node.children.x, node.children.y);

			//if(__any((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) // If any thread wants to traverse unbuild node
			//{
			//	return; // Skip traversal for build
			//}

			// Speculative traversal does not pay off as we may speculatively build nodes that we do not need
			
			// First leaf => postpone and continue traversal.

			//if(nodeAddr < 0 && leafAddr >= 0)
			//{
			//	leafAddr = nodeAddr;
			//	popStack();
			//}

			//if(__any((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) // If any thread wants to traverse unbuild node
			//{
			//	return; // Skip traversal for build
			//}

			// All SIMD lanes have found a leaf => process them.

			//if(!__any(leafAddr >= 0))
			//	break;
		}

		// Process postponed leaf nodes.

		//while(leafAddr < 0)
		while(nodeAddr < 0)
		{
			// Fetch the start and end of the triangle list.

			//int4* leaf = (int4*)((CUdeviceptr)g_bvh + (-leafAddr-1)*sizeof(CudaBVHNode) + 48);
			int4* leaf = (int4*)((CUdeviceptr)g_bvh + (-nodeAddr-1)*sizeof(CudaBVHNode) + 48);

			int triAddr  = leaf->x;
			int triAddr2 = leaf->y;

			// Intersect the ray against each triangle using Sven Woop's algorithm.

			for( ; triAddr < triAddr2; triAddr++)
			{
				// Compute and check intersection t-value.

				int triIdx = ((int*)c_bvh_in.trisIndex)[triAddr];

				float3 v0, v1, v2;
				taskFetchTri(c_bvh_in.tris, triIdx*3, v0, v1, v2);

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
							hitIndex = triIdx;
							if(c_in.anyHit)
							{
								nodeAddr = EntrypointSentinel;
								break;
							}
						}
					}
				}
			} // triangle

			// Another leaf was postponed => process it as well.

			//leafAddr = nodeAddr;
			//if(nodeAddr < 0)
			//{
			//bool ret = popStack();
			//if(!ret)
			//return 0; // OPTIMIZE: The traversal can be interupted after the leaf have be processed
			//}

			popStack(nodeAddr, stackPtr);
		} // leaf

		if(__any((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) // If any thread wants to traverse unbuild node
		{
#ifdef COUNT_INTERUPTS
			maxSteps[threadIdx.y]++;
#endif
			return; // Skip traversal for build
		}
	} // traversal

	// Remap intersected triangle index, and store the result.

	//if(hitIndex != -1)
	//	hitIndex = FETCH_TEXTURE(triIndices, hitIndex, int);
	STORE_RESULT(rayIdx, hitIndex, hitT, hitU, hitV);
}*/

//------------------------------------------------------------------------

// Traverses the yet build BVH and returns the index of the task to further subdivide
#ifdef PRECOMPUTE_ISECT
__device__ __forceinline__ void traverseSemiReg(int& _rayIdx, const float3& _orig, const float3& _dir, const float3& _idir, const float3& _ood, float& _tmin, int& _nodeAddr, int* &_stackPtr, int& _hitIndex, float& _hitT, float& _hitU, float& _hitV) // Must be noinline to avoid warp divergence
#else
__device__ __forceinline__ void traverseSemiReg(int& _rayIdx, const float3& _orig, const float3& _dir, float& _tmin, int& _nodeAddr, int* &_stackPtr, int& _hitIndex, float& _hitT, float& _hitU, float& _hitV) // Must be noinline to avoid warp divergence
#endif
{
	// Live state during traversal, stored in registers.
	// IMPORTANT: This caching is crucial as without it each write is possibly written to the the param space
	int     rayIdx = _rayIdx;                 // Ray index.
    float3  orig   = _orig;					  // Ray origin.
    float3  dir    = _dir;					  // Ray direction.
#ifdef PRECOMPUTE_ISECT
	float3  idir   = _idir;                   // 1 / dir
    float3  ood    = _ood;                    // orig / dir
#else
	float3  idir;
	float ooeps = exp2f(-80.0f); // Avoid div by zero.
	idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
	idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
	idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));

	float3 ood  = orig * idir;
#endif

	float   tmin   = _tmin;                   // t-value from which the ray starts. Usually 0.
	float   hitT   = _hitT;                   // t-value of the closest intersection.

    int*    stackPtr = _stackPtr; //&traversalStack[0];  // Current position in traversal stack.
    int     leafAddr = 0;               // First postponed leaf, non-negative if none.
    int     nodeAddr = _nodeAddr;               // Non-negative: current internal node, negative: second postponed leaf.

    int     hitIndex = _hitIndex;               // Triangle index of the closest intersection, -1 if none.
	float   hitU = _hitU;                   // u-barycentric of the closest intersection.
	float   hitV = _hitV;                   // v-barycentric of the closest intersection.

	//int     leafAddr = 0;                             // First postponed leaf, non-negative if none.
	ASSERT_DIVERGENCE("traverse top", threadIdx.x);

	//if(rayIdx == 208147)
	//	printf("Restart %d, hit %f\n", nodeAddr, hitT);

	if(nodeAddr < 0) // The built node may have been a leaf
	{
		leafAddr = nodeAddr;
		//if(rayIdx == 208147)
		//	printf("RestartLeaf %d | %d, hit %f\n", nodeAddr, *stackPtr, hitT);
		nodeAddr = *stackPtr;
		--stackPtr;
	}

	// Process rays in a loop.
	while(nodeAddr != EntrypointSentinel || leafAddr < 0)
	{
		while(nodeAddr >= 0 && nodeAddr < EntrypointSentinel)
		{
			// Fetch AABBs of the two child nodes.

			CudaBVHNode node;
			taskFetchNode((CUdeviceptr)g_bvh, nodeAddr, node);

			// Intersect the ray against the child nodes.

			float c0lox = node.c0xy.x * idir.x - ood.x;
			float c0hix = node.c0xy.y * idir.x - ood.x;
			float c0loy = node.c0xy.z * idir.y - ood.y;
			float c0hiy = node.c0xy.w * idir.y - ood.y;
			float c0loz = node.c01z.x * idir.z - ood.z;
			float c0hiz = node.c01z.y * idir.z - ood.z;
			float c1loz = node.c01z.z * idir.z - ood.z;
			float c1hiz = node.c01z.w * idir.z - ood.z;
			float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), tmin);
			float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
			float c1lox = node.c1xy.x * idir.x - ood.x;
			float c1hix = node.c1xy.y * idir.x - ood.x;
			float c1loy = node.c1xy.z * idir.y - ood.y;
			float c1hiy = node.c1xy.w * idir.y - ood.y;
			float c1min = max4(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), tmin);
			float c1max = min4(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz), hitT);

			// Decide where to go next.

			bool traverseChild0 = (c0max >= c0min);
			bool traverseChild1 = (c1max >= c1min);

#ifdef RAY_STATS
			atomicAdd(&g_NumNodes, 2);
#endif

			// Neither child was intersected => pop stack.

			if(!traverseChild0 && !traverseChild1)
            {
				//if(rayIdx == 208147)
				//	printf("NoTraverse %d | %d (%f - %f) (%f - %f) hit %f\n", nodeAddr, *stackPtr, c0min, c0max, c1min, c1max, hitT);
                nodeAddr = *stackPtr;
				--stackPtr;
            }

            // Otherwise => fetch child pointers.

            else
            {
//#if !defined(SPECULATIVE_OPEN) || defined(ALL_OPEN)
				if(traverseChild0 && (node.children.x & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild left child
/*#else
				if((node.children.x & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild left child
#endif*/
				{
/*#ifdef TRAVERSAL_TEST
					if(g_taskStackBVH.launchFlag != 0)
					{
						if(rayIdx == 208147)
						{
							int unfinished = *((int*)&g_taskStackBVH.tasks[node.children.x & UNBUILD_UNMASK].unfinished);
							int active = *((int*)&g_taskStackBVH.header[node.children.x & UNBUILD_UNMASK]);
							printf("Traversal test violation ray %d left %d -> %d (%d | %d) Task: u %d a %d\n", rayIdx, nodeAddr, node.children.w+1, node.children.x, node.children.y, unfinished, active);
							int n = nodeAddr;
							int p = node.children.x;
							while(n != 0)
							{
								float4 temp = tex1Dfetch(t_nodesA, n*4+3);
								printf("Traversal backtrack %d: %d -> %d (%c), left %d right %d\n", rayIdx, n, p, (p == __float_as_int(temp.x)) ? 'L' : 'R', __float_as_int(temp.x), __float_as_int(temp.y), __float_as_int(temp.z));
								p = n;
								n = __float_as_int(temp.z);
							}
							float4 temp = tex1Dfetch(t_nodesA, n*4+3);
							printf("Traversal backtrack %d: %d -> %d (%c), left %d right %d\n", rayIdx, n, p, (p == __float_as_int(temp.x)) ? 'L' : 'R', __float_as_int(temp.x), __float_as_int(temp.y), __float_as_int(temp.z));
							printf("Task %d, Parent %d, Node %d, Child %c\n", node.children.x & UNBUILD_UNMASK, nodeAddr, node.children.w+1, 'L');
						}
						node.children.x = EntrypointSentinel;
					}
					else
					{
						if(rayIdx == 208147)
							printf("Task %d, Parent %d, Node %d, Child %c\n", node.children.x & UNBUILD_UNMASK, nodeAddr, node.children.w+1, 'L');
						buildNode(node.children.x & UNBUILD_UNMASK, node.children.w+1, nodeAddr, 0, traverseChild0); // Clear the unbuild flag to get the pool entry
						node.children.x = nodeAddr | (UNBUILD_FLAG | 0x0); // Save the code of the location where to check for completion
					}
#else*/
					buildNode(node.children.x & UNBUILD_UNMASK, node.children.w+1, nodeAddr, 0, traverseChild0); // Clear the unbuild flag to get the pool entry
					node.children.x = nodeAddr | (UNBUILD_FLAG | 0x0); // Save the code of the location where to check for completion
//#endif
				}

//#if !defined(SPECULATIVE_OPEN) || defined(ALL_OPEN)
				if(traverseChild1 && (node.children.y & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild right child
/*#else
				if((node.children.y & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild right child
#endif*/
				{
/*#ifdef TRAVERSAL_TEST
					if(g_taskStackBVH.launchFlag != 0)
					{
						//printf("Traversal test violation ray %d right %d -> %d (%d | %d)\n", rayIdx, nodeAddr, node.children.w+2, node.children.x, node.children.y);
						//printf("Task %d, Parent %d, Node %d, Child %c\n", node.children.y & UNBUILD_UNMASK, nodeAddr, node.children.w+2, 'R');
						node.children.y = EntrypointSentinel;
					}
					else
					{
						buildNode(node.children.y & UNBUILD_UNMASK, node.children.w+2, nodeAddr, 1, traverseChild1); // Clear the unbuild flag to get the pool entry
						node.children.y = nodeAddr | (UNBUILD_FLAG | 0x10000000); // Save the code of the location where to check for completion
					}
#else*/
					buildNode(node.children.y & UNBUILD_UNMASK, node.children.w+2, nodeAddr, 1, traverseChild1); // Clear the unbuild flag to get the pool entry
					node.children.y = nodeAddr | (UNBUILD_FLAG | 0x10000000); // Save the code of the location where to check for completion
//#endif
				}

				//if(rayIdx == 208147)
				//	printf("%d -> ", nodeAddr);
                nodeAddr = (traverseChild0) ? node.children.x : node.children.y;

                // Both children were intersected => push the farther one.

                if(traverseChild0 && traverseChild1)
                {
					if(c1min < c0min)
                        swap(nodeAddr, node.children.y);
                    ++stackPtr;

					// Place child pointer onto the stack
					*stackPtr = node.children.y; // Save the BVH index
                }
				//if(rayIdx == 208147)
				//	printf("%d (%c) (%f - %f) (%f - %f) hit %f\n", nodeAddr, (nodeAddr == node.children.x) ? 'L' : 'R', c0min, c0max, c1min, c1max, hitT);

#if SWAP_STACK == 1
				if((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG) // If a thread wants to traverse unbuild node
				{
					// Swap it with the top of the stack
					int tmp = *stackPtr;
					if(tmp != EntrypointSentinel && (tmp & UNBUILD_MASK) != UNBUILD_FLAG)
					{
						*stackPtr = nodeAddr;
						nodeAddr = tmp;
					}
				}

#elif SWAP_STACK == 2
				if((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG) // If a thread wants to traverse unbuild node
				{
					// Swap it with the top of the stack
					int* ptr = stackPtr;
					int tmp = *ptr;
					while(tmp != EntrypointSentinel)
					{
						if((tmp & UNBUILD_MASK) != UNBUILD_FLAG)
						{
							*ptr = nodeAddr;
							nodeAddr = tmp;
							break;
						}
						--ptr;
						tmp = *ptr;
					}
				}
#endif

            }

			// Speculative traversal does not pay off as we may speculatively build nodes that we do not need
			
			// First leaf => postpone and continue traversal.

			if(nodeAddr < 0 && leafAddr >= 0)
			{
				leafAddr = nodeAddr;
				//if(rayIdx == 208147)
				//	printf("Postpone %d | %d\n", nodeAddr, *stackPtr);
				nodeAddr = *stackPtr;
				--stackPtr;
			}

			// All SIMD lanes have found a leaf => process them.

			if(!__any(leafAddr >= 0))
				break;

#if SKIP_COUNT > 0
			if(__popc(__ballot((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) > SKIP_COUNT)
			{
				_hitT = hitT;
				_stackPtr = stackPtr;
				_nodeAddr = nodeAddr;

				_hitIndex = hitIndex;
				_hitU = hitU;
				_hitV = hitV;

#ifdef COUNT_INTERUPTS
				maxSteps[threadIdx.y]++;
#endif
				return; // Skip traversal for build
			}
#endif

		}

		// Process postponed leaf nodes.
		while(leafAddr < 0)
		//while(nodeAddr < 0)
		{
			// Fetch the start and end of the triangle list.

			//int4* leaf = (int4*)((CUdeviceptr)g_bvh + (-leafAddr-1)*sizeof(CudaBVHNode) + 48);
			//int4* leaf = (int4*)((CUdeviceptr)g_bvh + (-nodeAddr-1)*sizeof(CudaBVHNode) + 48);

			//int triAddr  = leaf->x;
			//int triAddr2 = leaf->y;

			float4 temp = tex1Dfetch(t_nodesA, (-leafAddr-1)*4+3);

			int triAddr =__float_as_int(temp.x);
			int triAddr2 =__float_as_int(temp.y);

			// Intersect the ray against each triangle using Sven Woop's algorithm.

			for( ; triAddr < triAddr2; triAddr++)
			{
				// Compute and check intersection t-value.

				//int triIdx = ((volatile int*)c_bvh_in.trisIndex)[triAddr];
				int triIdx = FETCH_GLOBAL(triIndices, triAddr, int);
				//int triIdx = loadCG(((int*)c_bvh_in.trisIndex)+triAddr);

				float3 v0, v1, v2;
				taskFetchTri(c_bvh_in.tris, triIdx*3, v0, v1, v2);

#ifdef RAY_STATS
				atomicAdd(&g_NumTris, 1);
#endif

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
							//if(rayIdx == 208147)
							//	printf("Intersect %f -> %f\n", hitT, t);
							hitT = t;
							hitU = u;
							hitV = v;
							hitIndex = triIdx;
							if(c_in.anyHit)
							{
								nodeAddr = EntrypointSentinel;
								break;
							}
						}
					}
				}
			} // triangle

			// Another leaf was postponed => process it as well.

			leafAddr = nodeAddr;
			if(nodeAddr < 0)
			{
				//if(rayIdx == 208147)
				//	printf("LeafDone %d | %d\n", nodeAddr, *stackPtr);
				nodeAddr = *stackPtr;
				--stackPtr;
			}

			//nodeAddr = *stackPtr;
			//--stackPtr;
		} // leaf

		if(__any((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) // If any thread wants to traverse unbuild node
		{
			//if(rayIdx == 208147)
			//	printf("Break %d | %d, hit %f\n", nodeAddr, *stackPtr, hitT);
			_hitT = hitT;
			_stackPtr = stackPtr;
			_nodeAddr = nodeAddr;

			_hitIndex = hitIndex;
			_hitU = hitU;
			_hitV = hitV;

#ifdef COUNT_INTERUPTS
			maxSteps[threadIdx.y]++;
#endif
			return; // Skip traversal for build
		}
	} // traversal

	// Remap intersected triangle index, and store the result.

	if(rayIdx >= 0)
	{
		//if(hitIndex != -1)
		//	hitIndex = FETCH_TEXTURE(triIndices, hitIndex, int);
		STORE_RESULT(rayIdx, hitIndex, hitT, hitU, hitV);
		_rayIdx = -1;
		_nodeAddr = nodeAddr;
	}

}

//------------------------------------------------------------------------

// Traverses the yet build BVH and returns the index of the task to further subdivide
#ifdef PRECOMPUTE_ISECT
__device__ void traverseFullReg(const int& _rayIdx, const float3& _orig, const float3& _dir, const float3& _idir, const float3& _ood, float& _tmin, int& _nodeAddr, int* &_stackPtr, int& _hitIndex, float& _hitT, float& _hitU, float& _hitV)
#else
__device__ void traverseFullReg(const int& _rayIdx, const float3& _orig, const float3& _dir, float& _tmin, int& _nodeAddr, int* &_stackPtr, int& _hitIndex, float& _hitT, float& _hitU, float& _hitV)
#endif
{
	// Traversal stack in CUDA thread-local memory.
    //int traversalStack[TRAVERSAL_STACK];
	//traversalStack[0] = EntrypointSentinel; // Bottom-most entry.

    // Live state during traversal, stored in registers.
	int     rayIdx = _rayIdx;                 // Ray index.
    float3  orig   = _orig;					  // Ray origin.
    float3  dir    = _dir;					  // Ray direction.
#ifdef PRECOMPUTE_ISECT
	float3  idir   = _idir;                   // 1 / dir
    float3  ood    = _ood;                    // orig / dir
#else
	float3  idir;
	float ooeps = exp2f(-80.0f); // Avoid div by zero.
	idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
	idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
	idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));

	float3 ood  = orig * idir;
#endif

	float   tmin   = _tmin;                   // t-value from which the ray starts. Usually 0.
	float   hitT   = _hitT;                   // t-value of the closest intersection.

    int*    stackPtr = _stackPtr; //&traversalStack[0];  // Current position in traversal stack.
    int     leafAddr = 0;               // First postponed leaf, non-negative if none.
    int     nodeAddr = _nodeAddr;               // Non-negative: current internal node, negative: second postponed leaf.

    int     hitIndex = _hitIndex;               // Triangle index of the closest intersection, -1 if none.
	float   hitU = _hitU;                   // u-barycentric of the closest intersection.
	float   hitV = _hitV;                   // v-barycentric of the closest intersection.

	//int     leafAddr = 0;                             // First postponed leaf, non-negative if none.

	ASSERT_DIVERGENCE("traverse top", threadIdx.x);

	// Process rays in a loop.
	while(nodeAddr != EntrypointSentinel)
	{
		while(nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
		{
			// Fetch AABBs of the two child nodes.

			CudaBVHNode node;
			taskFetchNode((CUdeviceptr)g_bvh, nodeAddr, node);

			// Intersect the ray against the child nodes.

			float c0lox = node.c0xy.x * idir.x - ood.x;
			float c0hix = node.c0xy.y * idir.x - ood.x;
			float c0loy = node.c0xy.z * idir.y - ood.y;
			float c0hiy = node.c0xy.w * idir.y - ood.y;
			float c0loz = node.c01z.x * idir.z - ood.z;
			float c0hiz = node.c01z.y * idir.z - ood.z;
			float c1loz = node.c01z.z * idir.z - ood.z;
			float c1hiz = node.c01z.w * idir.z - ood.z;
			float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), tmin);
			float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
			float c1lox = node.c1xy.x * idir.x - ood.x;
			float c1hix = node.c1xy.y * idir.x - ood.x;
			float c1loy = node.c1xy.z * idir.y - ood.y;
			float c1hiy = node.c1xy.w * idir.y - ood.y;
			float c1min = max4(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), tmin);
			float c1max = min4(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz), hitT);

			// Decide where to go next.

			bool traverseChild0 = (c0max >= c0min);
			bool traverseChild1 = (c1max >= c1min);

			// Neither child was intersected => pop stack.

			if(!traverseChild0 && !traverseChild1)
            {
                nodeAddr = *stackPtr;
				--stackPtr;
            }

            // Otherwise => fetch child pointers.

            else
            {
                nodeAddr = (traverseChild0) ? node.children.x : node.children.y;

                // Both children were intersected => push the farther one.

                if(traverseChild0 && traverseChild1)
                {
					if(c1min < c0min)
                        swap(nodeAddr, node.children.y);
                    ++stackPtr;

					// Place child pointer onto the stack
					*stackPtr = node.children.y; // Save the BVH index
                }
            }

			// Speculative traversal does not pay off as we may speculatively build nodes that we do not need
			
			// First leaf => postpone and continue traversal.

			if(nodeAddr < 0 && leafAddr >= 0)
			{
				leafAddr = nodeAddr;
				nodeAddr = *stackPtr;
				--stackPtr;
			}

			// All SIMD lanes have found a leaf => process them.

			if(!__any(leafAddr >= 0))
				break;
		}

		// Process postponed leaf nodes.
		while(leafAddr < 0)
		//while(nodeAddr < 0)
		{
			// Fetch the start and end of the triangle list.

			//int4* leaf = (int4*)((CUdeviceptr)g_bvh + (-leafAddr-1)*sizeof(CudaBVHNode) + 48);
			//int4* leaf = (int4*)((CUdeviceptr)g_bvh + (-nodeAddr-1)*sizeof(CudaBVHNode) + 48);

			//int triAddr  = leaf->x;
			//int triAddr2 = leaf->y;

			float4 temp = tex1Dfetch(t_nodesA, (-leafAddr-1)*4+3);

			int triAddr =__float_as_int(temp.x);
			int triAddr2 =__float_as_int(temp.y);

			// Intersect the ray against each triangle using Sven Woop's algorithm.

			for( ; triAddr < triAddr2; triAddr++)
			{
				// Compute and check intersection t-value.

				//int triIdx = ((volatile int*)c_bvh_in.trisIndex)[triAddr];
				int triIdx = FETCH_GLOBAL(triIndices, triAddr, int);

				float3 v0, v1, v2;
				taskFetchTri(c_bvh_in.tris, triIdx*3, v0, v1, v2);

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
							hitIndex = triIdx;
							if(c_in.anyHit)
							{
								nodeAddr = EntrypointSentinel;
								break;
							}
						}
					}
				}
			} // triangle

			// Another leaf was postponed => process it as well.

			leafAddr = nodeAddr;
			if(nodeAddr < 0)
			{
				nodeAddr = *stackPtr;
				--stackPtr;
			}

			//nodeAddr = *stackPtr;
			//--stackPtr;
		} // leaf
	} // traversal

	// Remap intersected triangle index, and store the result.

	//if(hitIndex != -1)
	//	hitIndex = FETCH_TEXTURE(triIndices, hitIndex, int);
	STORE_RESULT(rayIdx, hitIndex, hitT, hitU, hitV);

	_hitT = hitT;
    _stackPtr = stackPtr;
	_nodeAddr = nodeAddr;

    _hitIndex = hitIndex;
	_hitU = hitU;
	_hitV = hitV;
}

//------------------------------------------------------------------------
#elif defined(COMPACT_LAYOUT) && defined(WOOP_TRIANGLES)

#ifndef SKIP_HIT
// Traverses the yet build BVH and returns the index of the task to further subdivide
#ifdef PRECOMPUTE_ISECT
__device__ __forceinline__ void traverseSemiReg(int& _rayIdx, const float3& _orig, const float3& _dir, const float3& _idir, const float3& _ood, float& _tmin, int& _nodeAddr, int* &_stackPtr, int& _hitIndex, float& _hitT, float& _hitU, float& _hitV) // Must be noinline to avoid warp divergence
#else
__device__ __forceinline__ void traverseSemiReg(int& _rayIdx, const float3& _orig, const float3& _dir, float& _tmin, int& _nodeAddr, int* &_stackPtr, int& _hitIndex, float& _hitT, float& _hitU, float& _hitV) // Must be noinline to avoid warp divergence
#endif
{
	// Live state during traversal, stored in registers.
	// IMPORTANT: This caching is crucial as without it each write is possibly written to the the param space
	int     rayIdx = _rayIdx;                 // Ray index.
    float3  orig   = _orig;					  // Ray origin.
    float3  dir    = _dir;					  // Ray direction.
#ifdef PRECOMPUTE_ISECT
	float3  idir   = _idir;                   // 1 / dir
    float3  ood    = _ood;                    // orig / dir
#else
	float3  idir;
	float ooeps = exp2f(-80.0f); // Avoid div by zero.
	idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
	idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
	idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));

	float3 ood  = orig * idir;
#endif

	float   tmin   = _tmin;                   // t-value from which the ray starts. Usually 0.
	float   hitT   = _hitT;                   // t-value of the closest intersection.

    int*    stackPtr = _stackPtr; //&traversalStack[0];  // Current position in traversal stack.
    int     leafAddr = 0;               // First postponed leaf, non-negative if none.
    int     nodeAddr = _nodeAddr;               // Non-negative: current internal node, negative: second postponed leaf.

    int     hitIndex = _hitIndex;               // Triangle index of the closest intersection, -1 if none.
	float   hitU = _hitU;                   // u-barycentric of the closest intersection.
	float   hitV = _hitV;                   // v-barycentric of the closest intersection.

	//int     leafAddr = 0;                             // First postponed leaf, non-negative if none.
	ASSERT_DIVERGENCE("traverse top", threadIdx.x);

	if(nodeAddr < 0) // The built node may have been a leaf
	{
		leafAddr = nodeAddr;
		nodeAddr = *stackPtr;
		--stackPtr;
	}

	// Process rays in a loop.
	while(nodeAddr != EntrypointSentinel || leafAddr < 0)
	{
		while(nodeAddr >= 0 && nodeAddr < EntrypointSentinel)
		{
			// Fetch AABBs of the two child nodes.

			CudaBVHNode node;
			taskFetchNodeAddr((CUdeviceptr)g_bvh, nodeAddr, node);

			// Intersect the ray against the child nodes.

			float c0lox = node.c0xy.x * idir.x - ood.x;
			float c0hix = node.c0xy.y * idir.x - ood.x;
			float c0loy = node.c0xy.z * idir.y - ood.y;
			float c0hiy = node.c0xy.w * idir.y - ood.y;
			float c0loz = node.c01z.x * idir.z - ood.z;
			float c0hiz = node.c01z.y * idir.z - ood.z;
			float c1loz = node.c01z.z * idir.z - ood.z;
			float c1hiz = node.c01z.w * idir.z - ood.z;
			float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), tmin);
			float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
			float c1lox = node.c1xy.x * idir.x - ood.x;
			float c1hix = node.c1xy.y * idir.x - ood.x;
			float c1loy = node.c1xy.z * idir.y - ood.y;
			float c1hiy = node.c1xy.w * idir.y - ood.y;
			float c1min = max4(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), tmin);
			float c1max = min4(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz), hitT);

			// Decide where to go next.

			bool traverseChild0 = (c0max >= c0min);
			bool traverseChild1 = (c1max >= c1min);

			// Neither child was intersected => pop stack.

			if(!traverseChild0 && !traverseChild1)
            {
				nodeAddr = *stackPtr;
				--stackPtr;
            }

            // Otherwise => fetch child pointers.

            else
            {
//#if !defined(SPECULATIVE_OPEN) || defined(ALL_OPEN)
				if(traverseChild0 && (node.children.x & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild left child
/*#else
				if((node.children.x & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild left child
#endif*/
				{
					buildNode(node.children.x & UNBUILD_UNMASK, node.children.w+1, nodeAddr, 0, traverseChild0); // Clear the unbuild flag to get the pool entry
					node.children.x = nodeAddr | (UNBUILD_FLAG | 0x0); // Save the code of the location where to check for completion
				}

//#if !defined(SPECULATIVE_OPEN) || defined(ALL_OPEN)
				if(traverseChild1 && (node.children.y & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild right child
/*#else
				if((node.children.y & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild right child
#endif*/
				{
					buildNode(node.children.y & UNBUILD_UNMASK, node.children.w+2, nodeAddr, 1, traverseChild1); // Clear the unbuild flag to get the pool entry
					node.children.y = nodeAddr | (UNBUILD_FLAG | 0x10000000); // Save the code of the location where to check for completion
				}

                nodeAddr = (traverseChild0) ? node.children.x : node.children.y;

                // Both children were intersected => push the farther one.

                if(traverseChild0 && traverseChild1)
                {
					if(c1min < c0min)
                        swap(nodeAddr, node.children.y);
                    ++stackPtr;

					// Place child pointer onto the stack
					*stackPtr = node.children.y; // Save the BVH index
                }

#if SWAP_STACK == 1
				if((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG) // If a thread wants to traverse unbuild node
				{
					// Swap it with the top of the stack
					int tmp = *stackPtr;
					if(tmp != EntrypointSentinel && (tmp & UNBUILD_MASK) != UNBUILD_FLAG)
					{
						*stackPtr = nodeAddr;
						nodeAddr = tmp;
					}
				}

#elif SWAP_STACK == 2
				if((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG) // If a thread wants to traverse unbuild node
				{
					// Swap it with the top of the stack
					int* ptr = stackPtr;
					int tmp = *ptr;
					while(tmp != EntrypointSentinel)
					{
						if((tmp & UNBUILD_MASK) != UNBUILD_FLAG)
						{
							*ptr = nodeAddr;
							nodeAddr = tmp;
							break;
						}
						--ptr;
						tmp = *ptr;
					}
				}
#endif

            }

			// Speculative traversal does not pay off as we may speculatively build nodes that we do not need
			
			// First leaf => postpone and continue traversal.

			if(nodeAddr < 0 && leafAddr >= 0)
			{
				leafAddr = nodeAddr;
				nodeAddr = *stackPtr;
				--stackPtr;
			}

			// All SIMD lanes have found a leaf => process them.

			if(!__any(leafAddr >= 0))
				break;

#if SKIP_COUNT > 0
			if(__popc(__ballot((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) > SKIP_COUNT)
			{
				_hitT = hitT;
				_stackPtr = stackPtr;
				_nodeAddr = nodeAddr;

				_hitIndex = hitIndex;
				_hitU = hitU;
				_hitV = hitV;

#ifdef COUNT_INTERUPTS
				maxSteps[threadIdx.y]++;
#endif
				return; // Skip traversal for build
			}
#endif

		}

		// Process postponed leaf nodes.
		while(leafAddr < 0)
		//while(nodeAddr < 0)
		{
			// Intersect the ray against each triangle using Sven Woop's algorithm.
            for(int triAddr = ~leafAddr;; triAddr += 3)
			{
				// Read first 16 bytes of the triangle.
                // End marker (negative zero) => all triangles processed.

                //float4 v00 = tex1Dfetch(t_trisAOut, triAddr + 0);
				float4 v00 = *(((float4*)c_bvh_in.trisOut) + triAddr + 0);
				//float4 v00 = loadfloat4V(((volatile float4*)c_bvh_in.trisOut) + triAddr + 0);
                if(__float_as_int(v00.x) == 0x80000000)
                    break;

                // Compute and check intersection t-value.

                float Oz = v00.w - orig.x*v00.x - orig.y*v00.y - orig.z*v00.z;
                float invDz = 1.0f / (dir.x*v00.x + dir.y*v00.y + dir.z*v00.z);
                float t = Oz * invDz;

                if(t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

					//float4 v11 = tex1Dfetch(t_trisAOut, triAddr + 1);
					float4 v11 = *(((float4*)c_bvh_in.trisOut) + triAddr + 1);
                    //float4 v11 = loadfloat4V(((volatile float4*)c_bvh_in.trisOut) + triAddr + 1);
                    float Ox = v11.w + orig.x*v11.x + orig.y*v11.y + orig.z*v11.z;
                    float Dx = dir.x*v11.x + dir.y*v11.y + dir.z*v11.z;
                    float u = Ox + t*Dx;

                    if(u >= 0.0f && u <= 1.0f)
                    {
                        // Compute and check barycentric v.

						//float4 v22 = tex1Dfetch(t_trisAOut, triAddr + 2);
						float4 v22 = *(((float4*)c_bvh_in.trisOut) + triAddr + 2);
                        //float4 v22 = loadfloat4V(((volatile float4*)c_bvh_in.trisOut) + triAddr + 2);
                        float Oy = v22.w + orig.x*v22.x + orig.y*v22.y + orig.z*v22.z;
                        float Dy = dir.x*v22.x + dir.y*v22.y + dir.z*v22.z;
                        float v = Oy + t*Dy;

                        if(v >= 0.0f && u + v <= 1.0f)
                        {
                            // Record intersection.
                            // Closest intersection not required => terminate.

                            hitT = t;
							hitU = u;
							hitV = v;
                            hitIndex = triAddr;
                            if(c_in.anyHit)
                            {
                                nodeAddr = EntrypointSentinel;
                                break;
                            }
                        }
                    }
				}                
			} // triangle

			// Another leaf was postponed => process it as well.

			leafAddr = nodeAddr;
			if(nodeAddr < 0)
			{
				nodeAddr = *stackPtr;
				--stackPtr;
			}

			//nodeAddr = *stackPtr;
			//--stackPtr;
		} // leaf

		if(__any((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) // If any thread wants to traverse unbuild node
		{
			_hitT = hitT;
			_stackPtr = stackPtr;
			_nodeAddr = nodeAddr;

			_hitIndex = hitIndex;
			_hitU = hitU;
			_hitV = hitV;

#ifdef COUNT_INTERUPTS
			maxSteps[threadIdx.y]++;
#endif
			return; // Skip traversal for build
		}
	} // traversal

	// Remap intersected triangle index, and store the result.

	if(rayIdx >= 0)
	{
		if(hitIndex != -1)
			//hitIndex = tex1Dfetch(t_triIndicesOut, hitIndex);
			hitIndex = *(((int*)c_bvh_in.trisIndexOut) + hitIndex);
		STORE_RESULT(rayIdx, hitIndex, hitT, hitU, hitV);
		_rayIdx = -1;
		_nodeAddr = nodeAddr;
	}

}
#else
// Traverses the yet build BVH and returns the index of the task to further subdivide
#ifdef PRECOMPUTE_ISECT
__device__ __forceinline__ void traverseSemiReg(int& _rayIdx, const float3& _orig, const float3& _dir, const float3& _idir, const float3& _ood, float& _tmin, int& _nodeAddr, int* &_stackPtr, int& _hitIndex, float& _hitT, float& _hitU, float& _hitV) // Must be noinline to avoid warp divergence
#else
__device__ __forceinline__ void traverseSemiReg(int& _rayIdx, const float3& _orig, const float3& _dir, float& _tmin, int& _nodeAddr, int* &_stackPtr, float* &_hitPtr, int& _hitIndex, float& _hitT, float& _hitU, float& _hitV) // Must be noinline to avoid warp divergence
#endif
{
	// Live state during traversal, stored in registers.
	// IMPORTANT: This caching is crucial as without it each write is possibly written to the the param space
	int     rayIdx = _rayIdx;                 // Ray index.
    float3  orig   = _orig;					  // Ray origin.
    float3  dir    = _dir;					  // Ray direction.
#ifdef PRECOMPUTE_ISECT
	float3  idir   = _idir;                   // 1 / dir
    float3  ood    = _ood;                    // orig / dir
#else
	float3  idir;
	float ooeps = exp2f(-80.0f); // Avoid div by zero.
	idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
	idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
	idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));

	float3 ood  = orig * idir;
#endif

	float   tmin   = _tmin;                   // t-value from which the ray starts. Usually 0.
	float   hitT   = _hitT;                   // t-value of the closest intersection.

    int*    stackPtr = _stackPtr; //&traversalStack[0];  // Current position in traversal stack.
	float*  hitPtr = _hitPtr; //&traversalStack[0];  // Current position in traversal stack.
    int     leafAddr = 0;               // First postponed leaf, non-negative if none.
    int     nodeAddr = _nodeAddr;               // Non-negative: current internal node, negative: second postponed leaf.

    int     hitIndex = _hitIndex;               // Triangle index of the closest intersection, -1 if none.
	float   hitU = _hitU;                   // u-barycentric of the closest intersection.
	float   hitV = _hitV;                   // v-barycentric of the closest intersection.

	//int     leafAddr = 0;                             // First postponed leaf, non-negative if none.
	ASSERT_DIVERGENCE("traverse top", threadIdx.x);

	if(nodeAddr < 0) // The built node may have been a leaf
	{
		leafAddr = nodeAddr;
		while(*hitPtr > hitT) // Cannot improve the solution
		{
			--stackPtr;
			--hitPtr;
		}
		nodeAddr = *stackPtr;
		--stackPtr;
		--hitPtr;
	}

	// Process rays in a loop.
	while(nodeAddr != EntrypointSentinel || leafAddr < 0)
	{
		while(nodeAddr >= 0 && nodeAddr < EntrypointSentinel)
		{
			// Fetch AABBs of the two child nodes.

			CudaBVHNode node;
			taskFetchNodeAddr((CUdeviceptr)g_bvh, nodeAddr, node);

			// Intersect the ray against the child nodes.

			float c0lox = node.c0xy.x * idir.x - ood.x;
			float c0hix = node.c0xy.y * idir.x - ood.x;
			float c0loy = node.c0xy.z * idir.y - ood.y;
			float c0hiy = node.c0xy.w * idir.y - ood.y;
			float c0loz = node.c01z.x * idir.z - ood.z;
			float c0hiz = node.c01z.y * idir.z - ood.z;
			float c1loz = node.c01z.z * idir.z - ood.z;
			float c1hiz = node.c01z.w * idir.z - ood.z;
			float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), tmin);
			float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
			float c1lox = node.c1xy.x * idir.x - ood.x;
			float c1hix = node.c1xy.y * idir.x - ood.x;
			float c1loy = node.c1xy.z * idir.y - ood.y;
			float c1hiy = node.c1xy.w * idir.y - ood.y;
			float c1min = max4(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), tmin);
			float c1max = min4(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz), hitT);

			// Decide where to go next.

			bool traverseChild0 = (c0max >= c0min);
			bool traverseChild1 = (c1max >= c1min);

			// Neither child was intersected => pop stack.

			if(!traverseChild0 && !traverseChild1)
            {
				while(*hitPtr > hitT) // Cannot improve the solution
				{
					--stackPtr;
					--hitPtr;
				}
				nodeAddr = *stackPtr;
				--stackPtr;
				--hitPtr;
            }

            // Otherwise => fetch child pointers.

            else
            {
//#if !defined(SPECULATIVE_OPEN) || defined(ALL_OPEN)
				if(traverseChild0 && (node.children.x & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild left child
/*#else
				if((node.children.x & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild left child
#endif*/
				{
					//buildNode(node.children.x & UNBUILD_UNMASK, node.children.w+1, nodeAddr, 0, traverseChild0); // Clear the unbuild flag to get the pool entry
					node.children.x = nodeAddr | (UNBUILD_FLAG | 0x0); // Save the code of the location where to check for completion
				}

//#if !defined(SPECULATIVE_OPEN) || defined(ALL_OPEN)
				if(traverseChild1 && (node.children.y & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild right child
/*#else
				if((node.children.y & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild right child
#endif*/
				{
					//buildNode(node.children.y & UNBUILD_UNMASK, node.children.w+2, nodeAddr, 1, traverseChild1); // Clear the unbuild flag to get the pool entry
					node.children.y = nodeAddr | (UNBUILD_FLAG | 0x10000000); // Save the code of the location where to check for completion
				}

                nodeAddr = (traverseChild0) ? node.children.x : node.children.y;

                // Both children were intersected => push the farther one.

                if(traverseChild0 && traverseChild1)
                {
					if(c1min < c0min)
					{
                        swap(nodeAddr, node.children.y);
						swap(c0min, c1min);
					}
                    ++stackPtr;
					++hitPtr;

					// Place child pointer onto the stack
					*stackPtr = node.children.y; // Save the BVH index
					*hitPtr = c1min; // Save the node distance
                }

#if SWAP_STACK == 1
				if((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG) // If a thread wants to traverse unbuild node
				{
					// Swap it with the top of the stack
					int tmp = *stackPtr;
					if(tmp != EntrypointSentinel && (tmp & UNBUILD_MASK) != UNBUILD_FLAG)
					{
						*stackPtr = nodeAddr;
						nodeAddr = tmp;
					}
				}

#elif SWAP_STACK == 2
				if((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG) // If a thread wants to traverse unbuild node
				{
					// Swap it with the top of the stack
					int* ptr = stackPtr;
					int tmp = *ptr;
					while(tmp != EntrypointSentinel)
					{
						if((tmp & UNBUILD_MASK) != UNBUILD_FLAG)
						{
							*ptr = nodeAddr;
							nodeAddr = tmp;
							break;
						}
						--ptr;
						tmp = *ptr;
					}
				}
#endif

            }

			// Speculative traversal does not pay off as we may speculatively build nodes that we do not need
			
			// First leaf => postpone and continue traversal.

			if(nodeAddr < 0 && leafAddr >= 0)
			{
				leafAddr = nodeAddr;
				while(*hitPtr > hitT) // Cannot improve the solution
				{
					--stackPtr;
					--hitPtr;
				}
				nodeAddr = *stackPtr;
				--stackPtr;
				--hitPtr;
			}

			// All SIMD lanes have found a leaf => process them.

			if(!__any(leafAddr >= 0))
				break;

#if SKIP_COUNT > 0
			if(__popc(__ballot((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) > SKIP_COUNT)
			{
				_hitT = hitT;
				_stackPtr = stackPtr;
				_hitPtr = hitPtr;
				_nodeAddr = nodeAddr;

				_hitIndex = hitIndex;
				_hitU = hitU;
				_hitV = hitV;

#ifdef COUNT_INTERUPTS
				maxSteps[threadIdx.y]++;
#endif
				return; // Skip traversal for build
			}
#endif

		}

		// Process postponed leaf nodes.
		while(leafAddr < 0)
		//while(nodeAddr < 0)
		{
			// Intersect the ray against each triangle using Sven Woop's algorithm.
            for(int triAddr = ~leafAddr;; triAddr += 3)
			{
				// Read first 16 bytes of the triangle.
                // End marker (negative zero) => all triangles processed.

                float4 v00 = tex1Dfetch(t_trisAOut, triAddr + 0);
				//float4 v00 = *(((float4*)c_bvh_in.trisOut) + triAddr + 0);
				//float4 v00 = loadfloat4V(((volatile float4*)c_bvh_in.trisOut) + triAddr + 0);
                if(__float_as_int(v00.x) == 0x80000000)
                    break;

                // Compute and check intersection t-value.

                float Oz = v00.w - orig.x*v00.x - orig.y*v00.y - orig.z*v00.z;
                float invDz = 1.0f / (dir.x*v00.x + dir.y*v00.y + dir.z*v00.z);
                float t = Oz * invDz;

                if(t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

					float4 v11 = tex1Dfetch(t_trisAOut, triAddr + 1);
					//float4 v11 = *(((float4*)c_bvh_in.trisOut) + triAddr + 1);
                    //float4 v11 = loadfloat4V(((volatile float4*)c_bvh_in.trisOut) + triAddr + 1);
                    float Ox = v11.w + orig.x*v11.x + orig.y*v11.y + orig.z*v11.z;
                    float Dx = dir.x*v11.x + dir.y*v11.y + dir.z*v11.z;
                    float u = Ox + t*Dx;

                    if(u >= 0.0f && u <= 1.0f)
                    {
                        // Compute and check barycentric v.

						float4 v22 = tex1Dfetch(t_trisAOut, triAddr + 2);
						//float4 v22 = *(((float4*)c_bvh_in.trisOut) + triAddr + 2);
                        //float4 v22 = loadfloat4V(((volatile float4*)c_bvh_in.trisOut) + triAddr + 2);
                        float Oy = v22.w + orig.x*v22.x + orig.y*v22.y + orig.z*v22.z;
                        float Dy = dir.x*v22.x + dir.y*v22.y + dir.z*v22.z;
                        float v = Oy + t*Dy;

                        if(v >= 0.0f && u + v <= 1.0f)
                        {
                            // Record intersection.
                            // Closest intersection not required => terminate.

                            hitT = t;
							hitU = u;
							hitV = v;
                            hitIndex = triAddr;
                            if(c_in.anyHit)
                            {
                                nodeAddr = EntrypointSentinel;
                                break;
                            }
                        }
                    }
				}                
			} // triangle

			// Another leaf was postponed => process it as well.

			leafAddr = nodeAddr;
			if(nodeAddr < 0)
			{
				while(*hitPtr > hitT) // Cannot improve the solution
				{
					--stackPtr;
					--hitPtr;
				}
				nodeAddr = *stackPtr;
				--stackPtr;
				--hitPtr;
			}

			//nodeAddr = *stackPtr;
			//--stackPtr;
		} // leaf

		if(__any((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) // If any thread wants to traverse unbuild node
		{
			if((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG) // If a thread wants to traverse unbuild node, mark it for build
			{
				int cleanAddr = nodeAddr & UNBUILD_UNMASK; // Node address without the flags
				int field = (nodeAddr >> 28) & 0x3; // The field in the link vector

#ifndef COMPACT_LAYOUT
				int newAddr = *(int*)((CUdeviceptr)g_bvh + cleanAddr*sizeof(CudaBVHNode) + 48 + field*sizeof(int));
#else
				int newAddr = *(int*)((CUdeviceptr)g_bvh + cleanAddr + 48 + field*sizeof(int));
#endif

				if((newAddr & UNBUILD_MASK) != UNBUILD_FLAG) // If any thread wants to traverse unbuild node
				{
					nodeAddr = newAddr;
				}
				else
				{
					int nodeOfs = *(int*)((CUdeviceptr)g_bvh + cleanAddr + 48 + 3*sizeof(int));
					buildNode(newAddr & UNBUILD_UNMASK, nodeOfs+field+1, cleanAddr, field, true); // Clear the unbuild flag to get the pool entry
				}
			}

			_hitT = hitT;
			_stackPtr = stackPtr;
			_hitPtr = hitPtr;
			_nodeAddr = nodeAddr;

			_hitIndex = hitIndex;
			_hitU = hitU;
			_hitV = hitV;

#ifdef COUNT_INTERUPTS
			maxSteps[threadIdx.y]++;
#endif
			return; // Skip traversal for build
		}
	} // traversal

	// Remap intersected triangle index, and store the result.

	if(rayIdx >= 0)
	{
		if(hitIndex != -1)
			hitIndex = tex1Dfetch(t_triIndicesOut, hitIndex);
			//hitIndex = *(((int*)c_bvh_in.trisIndexOut) + hitIndex);
		STORE_RESULT(rayIdx, hitIndex, hitT, hitU, hitV);
		_rayIdx = -1;
		_nodeAddr = nodeAddr;
	}

}
#endif
#elif defined(COMPACT_LAYOUT) && !defined(WOOP_TRIANGLES)

//------------------------------------------------------------------------

// Traverses the yet build BVH and returns the index of the task to further subdivide
// Traverses the yet build BVH and returns the index of the task to further subdivide
#ifdef PRECOMPUTE_ISECT
__device__ __forceinline__ void traverseSemiReg(int& _rayIdx, const float3& _orig, const float3& _dir, const float3& _idir, const float3& _ood, float& _tmin, int& _nodeAddr, int* &_stackPtr, int& _hitIndex, float& _hitT, float& _hitU, float& _hitV) // Must be noinline to avoid warp divergence
#else
__device__ __forceinline__ void traverseSemiReg(int& _rayIdx, const float3& _orig, const float3& _dir, float& _tmin, int& _nodeAddr, int* &_stackPtr, int& _hitIndex, float& _hitT, float& _hitU, float& _hitV) // Must be noinline to avoid warp divergence
#endif
{
	// Live state during traversal, stored in registers.
	// IMPORTANT: This caching is crucial as without it each write is possibly written to the the param space
	int     rayIdx = _rayIdx;                 // Ray index.
    float3  orig   = _orig;					  // Ray origin.
    float3  dir    = _dir;					  // Ray direction.
#ifdef PRECOMPUTE_ISECT
	float3  idir   = _idir;                   // 1 / dir
    float3  ood    = _ood;                    // orig / dir
#else
	float3  idir;
	float ooeps = exp2f(-80.0f); // Avoid div by zero.
	idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
	idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
	idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));

	float3 ood  = orig * idir;
#endif

	float   tmin   = _tmin;                   // t-value from which the ray starts. Usually 0.
	float   hitT   = _hitT;                   // t-value of the closest intersection.

    int*    stackPtr = _stackPtr; //&traversalStack[0];  // Current position in traversal stack.
    int     leafAddr = 0;               // First postponed leaf, non-negative if none.
    int     nodeAddr = _nodeAddr;               // Non-negative: current internal node, negative: second postponed leaf.

    int     hitIndex = _hitIndex;               // Triangle index of the closest intersection, -1 if none.
	float   hitU = _hitU;                   // u-barycentric of the closest intersection.
	float   hitV = _hitV;                   // v-barycentric of the closest intersection.

	//int     leafAddr = 0;                             // First postponed leaf, non-negative if none.
	ASSERT_DIVERGENCE("traverse top", threadIdx.x);

	if(nodeAddr < 0) // The built node may have been a leaf
	{
		leafAddr = nodeAddr;
		nodeAddr = *stackPtr;
		--stackPtr;
	}

	// Process rays in a loop.
	while(nodeAddr != EntrypointSentinel || leafAddr < 0)
	{
		while(nodeAddr >= 0 && nodeAddr < EntrypointSentinel)
		{
			// Fetch AABBs of the two child nodes.

			CudaBVHNode node;
			taskFetchNodeAddr((CUdeviceptr)g_bvh, nodeAddr, node);

			// Intersect the ray against the child nodes.

			float c0lox = node.c0xy.x * idir.x - ood.x;
			float c0hix = node.c0xy.y * idir.x - ood.x;
			float c0loy = node.c0xy.z * idir.y - ood.y;
			float c0hiy = node.c0xy.w * idir.y - ood.y;
			float c0loz = node.c01z.x * idir.z - ood.z;
			float c0hiz = node.c01z.y * idir.z - ood.z;
			float c1loz = node.c01z.z * idir.z - ood.z;
			float c1hiz = node.c01z.w * idir.z - ood.z;
			float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), tmin);
			float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
			float c1lox = node.c1xy.x * idir.x - ood.x;
			float c1hix = node.c1xy.y * idir.x - ood.x;
			float c1loy = node.c1xy.z * idir.y - ood.y;
			float c1hiy = node.c1xy.w * idir.y - ood.y;
			float c1min = max4(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), tmin);
			float c1max = min4(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz), hitT);

			// Decide where to go next.

			bool traverseChild0 = (c0max >= c0min);
			bool traverseChild1 = (c1max >= c1min);

			// Neither child was intersected => pop stack.

			if(!traverseChild0 && !traverseChild1)
            {
                nodeAddr = *stackPtr;
				--stackPtr;
            }

            // Otherwise => fetch child pointers.

            else
            {
//#if !defined(SPECULATIVE_OPEN) || defined(ALL_OPEN)
				if(traverseChild0 && (node.children.x & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild left child
/*#else
				if((node.children.x & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild left child
#endif*/
				{
					buildNode(node.children.x & UNBUILD_UNMASK, node.children.w+1, nodeAddr, 0, traverseChild0); // Clear the unbuild flag to get the pool entry
					node.children.x = nodeAddr | (UNBUILD_FLAG | 0x0); // Save the code of the location where to check for completion
				}

//#if !defined(SPECULATIVE_OPEN) || defined(ALL_OPEN)
				if(traverseChild1 && (node.children.y & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild right child
/*#else
				if((node.children.y & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild right child
#endif*/
				{
					buildNode(node.children.y & UNBUILD_UNMASK, node.children.w+2, nodeAddr, 1, traverseChild1); // Clear the unbuild flag to get the pool entry
					node.children.y = nodeAddr | (UNBUILD_FLAG | 0x10000000); // Save the code of the location where to check for completion
				}

                nodeAddr = (traverseChild0) ? node.children.x : node.children.y;

                // Both children were intersected => push the farther one.

                if(traverseChild0 && traverseChild1)
                {
					if(c1min < c0min)
                        swap(nodeAddr, node.children.y);
                    ++stackPtr;

					// Place child pointer onto the stack
					*stackPtr = node.children.y; // Save the BVH index
                }

#if SWAP_STACK == 1
				if((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG) // If a thread wants to traverse unbuild node
				{
					// Swap it with the top of the stack
					int tmp = *stackPtr;
					if(tmp != EntrypointSentinel && (tmp & UNBUILD_MASK) != UNBUILD_FLAG)
					{
						*stackPtr = nodeAddr;
						nodeAddr = tmp;
					}
				}

#elif SWAP_STACK == 2
				if((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG) // If a thread wants to traverse unbuild node
				{
					// Swap it with the top of the stack
					int* ptr = stackPtr;
					int tmp = *ptr;
					while(tmp != EntrypointSentinel)
					{
						if((tmp & UNBUILD_MASK) != UNBUILD_FLAG)
						{
							*ptr = nodeAddr;
							nodeAddr = tmp;
							break;
						}
						--ptr;
						tmp = *ptr;
					}
				}
#endif

            }

			// Speculative traversal does not pay off as we may speculatively build nodes that we do not need
			
			// First leaf => postpone and continue traversal.

			if(nodeAddr < 0 && leafAddr >= 0)
			{
				leafAddr = nodeAddr;
				nodeAddr = *stackPtr;
				--stackPtr;
			}

			// All SIMD lanes have found a leaf => process them.

			if(!__any(leafAddr >= 0))
				break;

#if SKIP_COUNT > 0
			if(__popc(__ballot((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) > SKIP_COUNT)
			{
				_hitT = hitT;
				_stackPtr = stackPtr;
				_nodeAddr = nodeAddr;

				_hitIndex = hitIndex;
				_hitU = hitU;
				_hitV = hitV;

#ifdef COUNT_INTERUPTS
				maxSteps[threadIdx.y]++;
#endif
				return; // Skip traversal for build
			}
#endif

		}

		// Process postponed leaf nodes.
		while(leafAddr < 0)
			//while(nodeAddr < 0)
		{
			// Intersect the ray against each triangle using Sven Woop's algorithm.
			for(int triAddr = ~leafAddr;; triAddr += 3)
			{
				// Read first 16 bytes of the triangle.
				// End marker (negative zero) => all triangles processed.

				float3 v0 = make_float3(tex1Dfetch(t_trisAOut, triAddr + 0));
				//float4 v00 = loadfloat4V(((volatile float4*)c_bvh_in.trisOut) + triAddr + 0);
				//float3 v0 = make_float3(v00);
				if(__float_as_int(v0.x) == 0x80000000 && __float_as_int(v0.y) == 0x80000000 && __float_as_int(v0.z) == 0x80000000)
					break;

				float3 v1 = make_float3(tex1Dfetch(t_trisAOut, triAddr + 1));
				float3 v2 = make_float3(tex1Dfetch(t_trisAOut, triAddr + 2));
				//float4 v11 = loadfloat4V(((volatile float4*)c_bvh_in.trisOut) + triAddr + 1);
				//float3 v1 = make_float3(v11);
				//float4 v22 = loadfloat4V(((volatile float4*)c_bvh_in.trisOut) + triAddr + 2);
				//float3 v2 = make_float3(v22);

				// Compute and check intersection t-value.

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
							if(c_in.anyHit)
							{
								nodeAddr = EntrypointSentinel;
								break;
							}
						}
					}
				}
			} // triangle

			// Another leaf was postponed => process it as well.

			leafAddr = nodeAddr;
			if(nodeAddr < 0)
			{
				nodeAddr = *stackPtr;
				--stackPtr;
			}

			//nodeAddr = *stackPtr;
			//--stackPtr;
		} // leaf

		if(__any((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) // If any thread wants to traverse unbuild node
		{
			_hitT = hitT;
			_stackPtr = stackPtr;
			_nodeAddr = nodeAddr;

			_hitIndex = hitIndex;
			_hitU = hitU;
			_hitV = hitV;

#ifdef COUNT_INTERUPTS
			maxSteps[threadIdx.y]++;
#endif
			return; // Skip traversal for build
		}
	} // traversal

	// Remap intersected triangle index, and store the result.

	if(rayIdx >= 0)
	{
		if(hitIndex != -1)
			hitIndex = tex1Dfetch(t_triIndicesOut, hitIndex);
			//hitIndex = *(((int*)c_bvh_in.trisIndexOut) + hitIndex);
		STORE_RESULT(rayIdx, hitIndex, hitT, hitU, hitV);
		_rayIdx = -1;
		_nodeAddr = nodeAddr;
	}

}
#else
#error Unsupported traversal kernel!
#endif // !defined(COMPACT_LAYOUT) && !defined(WOOP_TRIANGLES)

//------------------------------------------------------------------------

// Check if new rays have to be read from the global pool
#ifdef PRECOMPUTE_ISECT
__device__ __forceinline__ void loadRaysOrExit(int tid, int* traversalStack, int& rayIdx, float3& orig, float3& dir, float3& idir, float3& ood, float& tmin, int& nodeAddr, int* &stackPtr, int& hitIndex, float& hitT, float& hitU, float& hitV)
#else
#ifdef SKIP_HIT
__device__ __forceinline__ void loadRaysOrExit(int tid, int* traversalStack, float* hitStack, int& rayIdx, float3& orig, float3& dir, float& tmin, int& nodeAddr, int* &stackPtr, float* &hitPtr, int& hitIndex, float& hitT, float& hitU, float& hitV)
#else
__device__ __forceinline__ void loadRaysOrExit(int tid, int* traversalStack, int& rayIdx, float3& orig, float3& dir, float& tmin, int& nodeAddr, int* &stackPtr, int& hitIndex, float& hitT, float& hitU, float& hitV)
#endif
#endif
{
	ASSERT_DIVERGENCE("loadRaysOrExit", tid);

	s_task[threadIdx.y].lock = LockType_Free; // Find build work in pool

	// All rays are done
	if(__all(nodeAddr == EntrypointSentinel))
	{
		// Local pool is empty => fetch new rays from the global pool using lane 0
		if(tid == 0 && s_task[threadIdx.y].rayCount <= 0)
		{
			s_task[threadIdx.y].nextRay = atomicSub(&g_taskStackBVH.warpCounter, PACKET_MULTIPLIER*WARP_SIZE); // Get new rays
			s_task[threadIdx.y].rayCount = PACKET_MULTIPLIER*WARP_SIZE;
		}

		// Out of work => done.
		if(s_task[threadIdx.y].nextRay - WARP_SIZE < -(WARP_SIZE-1))
		{
			s_task[threadIdx.y].rayPackets = -1;
			if(tid == 0)
				atomicAdd(&g_taskStackBVH.unfinished, 1); // Update the number of warps not finished with traversal
			return;
		}

		// Pick 32 rays from the local pool.
		rayIdx = s_task[threadIdx.y].nextRay - WARP_SIZE + tid; // Lower by WARP_SIZE to start from the taken rays

		// Update the ray pool
		s_task[threadIdx.y].nextRay -= WARP_SIZE;
		s_task[threadIdx.y].rayCount-= WARP_SIZE;

		// Fetch ray.

		if(rayIdx >= 0)
			taskFetchRay(c_in.rays, rayIdx, orig, dir, tmin, hitT);

		/*if(tid == 0)
		{
			printf("Ray (%f, %f, %f), (%f, %f, %f), %f, %f\n", orig.x, orig.y, orig.z, dir.x, dir.y, dir.z, tmin, hitT);
		}*/

#ifdef PRECOMPUTE_ISECT
		float ooeps = exp2f(-80.0f); // Avoid div by zero.
		idir.x = 1.0f / (fabsf(dir.x) > ooeps ? dir.x : copysignf(ooeps, dir.x));
		idir.y = 1.0f / (fabsf(dir.y) > ooeps ? dir.y : copysignf(ooeps, dir.y));
		idir.z = 1.0f / (fabsf(dir.z) > ooeps ? dir.z : copysignf(ooeps, dir.z));

		ood  = orig * idir;
#endif

		// Setup traversal.

		//traversalStack[0] = EntrypointSentinel; // Bottom-most entry.
        stackPtr = &traversalStack[0];
#ifdef SKIP_HIT
		hitPtr = &hitStack[0];
#endif
		//*stackPtr = EntrypointSentinel; // Bottom-most entry.
		hitIndex = -1;  // No triangle intersected so far.
		//leafAddr = 0;   // No postponed leaf.
		if(rayIdx < 0) // No ray
		{
			nodeAddr = EntrypointSentinel;
		}
		else if(g_taskStackBVH.nodeTop == 1)
		{
			nodeAddr = UNBUILD_FLAG | 0x30000000;   // Start from the unbuild root (we expect the 4th value is 0).
			s_task[threadIdx.y].lock = LockType_Free; // Load root task from the pool
		}
		else
		{
			nodeAddr = 0;   // Start from the root.
			s_task[threadIdx.y].type = TaskType_Intersect; // Continue traversal
			s_task[threadIdx.y].lock = LockType_None; // Bypass pool
		}
	}
}

//------------------------------------------------------------------------

// Restart traversal after it has been suspended due to unbuild node
__device__ bool restartTraversal(int& nodeAddr) // Making the function __noinline__ produced a better code
{
	// Try restarting the traversal
	checkBuildDone(nodeAddr);
	if(__all((nodeAddr & UNBUILD_MASK) != UNBUILD_FLAG))
	{
		ASSERT_DIVERGENCE("restartTraversal", threadIdx.x);
		/*if(nodeAddr < 0 && leafAddr >= 0)
		{
			leafAddr = nodeAddr;
			popStack();
		}

		if(__any((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) // If any thread wants to traverse unbuild node
		{
			ASSERT_DIVERGENCE("restartTraversal speculative", threadIdx.x);
			return 0; // Skip traversal for build
		}

		if(__any(leafAddr >= 0) || __all(intersect())) // Check for intersections in the loaded node
		{*/
		s_task[threadIdx.y].type = TaskType_Intersect; // Continue traversal
		s_task[threadIdx.y].lock = LockType_None; // Bypass pool
		return true;
		//}
	}

	return false;
}

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

	s_task[threadIdx.y].nextRay = 0;
	s_task[threadIdx.y].rayCount = 0;
	s_task[threadIdx.y].rayPackets = 1;

	//------------------------------------------------------------------------
	// Ray state.
	//------------------------------------------------------------------------

	// Traversal stack in CUDA thread-local memory.
	int traversalStack[TRAVERSAL_STACK];
#ifdef SKIP_HIT
	float hitStack[TRAVERSAL_STACK];
#endif

	// Live state during traversal, stored in registers.
	int     rayIdx;                 // Ray index.
	float3  orig;					// Ray origin.
	float3  dir;					// Ray direction.
	float   tmin;                   // t-value from which the ray starts. Usually 0.
#ifdef PRECOMPUTE_ISECT
	float3  idir;                   // 1 / dir
	float3  ood;                    // orig / dir
#endif

	int     nodeAddr;               // Non-negative: current internal node, negative: second postponed leaf.
	//int     leafAddr;               // First postponed leaf, non-negative if none.
	int*    stackPtr;               // Current position in traversal stack.
#ifdef SKIP_HIT
	float*  hitPtr;                 // Current position in hit stack.
#endif
	int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
	float   hitT;                   // t-value of the closest intersection.
	float   hitU;                   // u-barycentric of the closest intersection.
	float   hitV;                   // v-barycentric of the closest intersection.

	traversalStack[0] = EntrypointSentinel; // Bottom-most entry.
	nodeAddr = EntrypointSentinel;
	stackPtr = &traversalStack[0];
#ifdef SKIP_HIT
	hitStack[0] = -1.0f; // Bottom-most entry.
	hitPtr = &hitStack[0];
#endif

	// Initialize rays
#ifdef PRECOMPUTE_ISECT
	loadRaysOrExit(tid, traversalStack, rayIdx, orig, dir, idir, ood, tmin, nodeAddr, stackPtr, hitIndex, hitT, hitU, hitV);
#else
#ifdef SKIP_HIT
	loadRaysOrExit(tid, traversalStack, hitStack, rayIdx, orig, dir, tmin, nodeAddr, stackPtr, hitPtr, hitIndex, hitT, hitU, hitV);
#else
	loadRaysOrExit(tid, traversalStack, rayIdx, orig, dir, tmin, nodeAddr, stackPtr, hitIndex, hitT, hitU, hitV);
#endif
#endif

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

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT) || defined(COUNT_STEPS_DEQUEUE)
	//if(tid == 0)
	{
		maxSteps[threadIdx.y] = 0;
		sumSteps[threadIdx.y] = 0;
		numSteps[threadIdx.y] = 0;
		numRestarts[threadIdx.y] = 0;
	}
#endif

	while(s_task[threadIdx.y].lock != LockType_Free || taskDequeue(tid, nodeAddr)) // Main loop of the programme, while there is some task to do
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
#if SPLIT_TYPE > 0 && SPLIT_TYPE <= 3
#error Unsupported SPLIT_TYPE!
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
				
				if(triidx < s_task[threadIdx.y].triRight && ((int*)c_bvh_in.ppsTrisIndex)[triidx] > 0)
					printf("Tri error should be -1/0 is %d! Start %d, Left %d, Right %d, End %d\n", ((int*)c_bvh_in.ppsTrisIndex)[triidx], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triLeft, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd);

				if(triidx >= s_task[threadIdx.y].triRight && triidx < s_task[threadIdx.y].triEnd && ((int*)c_bvh_in.ppsTrisIndex)[triidx] < 1)
					printf("Tri error should be 1 is %d! Start %d, Left %d, Right %d, End %d\n", ((int*)c_bvh_in.ppsTrisIndex)[triidx], s_task[threadIdx.y].triStart, s_task[threadIdx.y].triLeft, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd);
				subtasksDone = taskReduceSubtask(s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popStart, s_task[threadIdx.y].popCount);
			} while(subtasksDone == -1);
			
			taskFinishSortSORT1(tid, s_task[threadIdx.y].popTaskIdx, subtasksDone);
			break;
#endif
			
		// --------------------------------------------------

#if SPLIT_TYPE == 0 || defined(COMPUTE_MEDIAN_BOUNDS)
#if AABB_TYPE < 3
		case TaskType_AABB_Min:
			// Do segmented reduction on the triangle bounding boxes
			do {
				computeAABB<float>(tid, s_task[threadIdx.y].popTaskIdx, s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].step, s_task[threadIdx.y].triStart, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd, min, CUDART_INF_F, -c_env.epsilon);
				subtasksDone = taskReduceSubtask(s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].popStart, s_task[threadIdx.y].popCount);
			} while(subtasksDone == -1);
			
			__threadfence(); // Probably needed so that next iteration does not read uninitialized data
			taskFinishAABB(tid, s_task[threadIdx.y].popTaskIdx, subtasksDone);
			break;

		case TaskType_AABB_Max:
			// Do segmented reduction on the triangle bounding boxes
			do {
				computeAABB<float>(tid, s_task[threadIdx.y].popTaskIdx, s_task[threadIdx.y].popSubtask, s_task[threadIdx.y].step, s_task[threadIdx.y].triStart, s_task[threadIdx.y].triRight, s_task[threadIdx.y].triEnd, max, -CUDART_INF_F, c_env.epsilon);
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
#endif

#if defined(OBJECT_SAH)/* && defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 300)*/
		case TaskType_BuildObjectSAH:
			// Build the subtree with object SAH
			computeObjectSplitTree();
			break;
#endif

		// --------------------------------------------------

		case TaskType_Intersect:
			// Traverse the rays until a nonbuild node is encountered
#ifdef PRECOMPUTE_ISECT
			traverseSemiReg(rayIdx, orig, dir, idir, ood, tmin, nodeAddr, stackPtr, hitIndex, hitT, hitU, hitV);
			loadRaysOrExit(tid, traversalStack, rayIdx, orig, dir, idir, ood, tmin, nodeAddr, stackPtr, hitIndex, hitT, hitU, hitV);
#else
#ifdef TRAVERSAL_TEST
#ifdef SKIP_HIT
			//if(g_taskStackBVH.launchFlag != 2)
				traverseSemiReg(rayIdx, orig, dir, tmin, nodeAddr, stackPtr, hitPtr, hitIndex, hitT, hitU, hitV);
			//else
			//	traverseFullReg(rayIdx, orig, dir, tmin, nodeAddr, stackPtr, hitIndex, hitT, hitU, hitV);
#else
			traverseSemiReg(rayIdx, orig, dir, tmin, nodeAddr, stackPtr, hitIndex, hitT, hitU, hitV);
#endif
#else
			traverseSemiReg(rayIdx, orig, dir, tmin, nodeAddr, stackPtr, hitIndex, hitT, hitU, hitV);
#endif
#ifdef SKIP_HIT
			loadRaysOrExit(tid, traversalStack, hitStack, rayIdx, orig, dir, tmin, nodeAddr, stackPtr, hitPtr, hitIndex, hitT, hitU, hitV);
#else
			loadRaysOrExit(tid, traversalStack, rayIdx, orig, dir, tmin, nodeAddr, stackPtr, hitIndex, hitT, hitU, hitV);
#endif
#endif

			continue; // No need to try to restart traversal
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
		else if(s_task[threadIdx.y].rayPackets != -1)
		{
			restartTraversal(nodeAddr);
		}

		ASSERT_DIVERGENCE("taskProcessWorkUntilDone bottom", tid);
	}

#if defined(COUNT_STEPS_LEFT) || defined(COUNT_STEPS_RIGHT) || defined(COUNT_STEPS_DEQUEUE)
	// Write out work statistics
	if(tid == 0)
	{
		int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
#ifndef COUNT_INTERUPTS
		((float4*)c_bvh_in.debug)[warpIdx] = make_float4(numSteps[threadIdx.y]*1.0f, sumSteps[threadIdx.y]*1.0f/numSteps[threadIdx.y], maxSteps[threadIdx.y]*1.0f, numRestarts[threadIdx.y]*1.0f);
#else
		((float4*)c_bvh_in.debug)[warpIdx] = make_float4(numSteps[threadIdx.y]*1.0f, sumSteps[threadIdx.y]*1.0f, maxSteps[threadIdx.y]*1.0f, numRestarts[threadIdx.y]*1.0f);
#endif
	}
#endif
}

//------------------------------------------------------------------------
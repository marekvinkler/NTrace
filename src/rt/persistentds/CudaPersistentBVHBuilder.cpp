#include "base/Random.hpp"
#include "persistentds/CudaPersistentBVHBuilder.hpp"
#include "persistentds/PersistentHelper.hpp"
#include "AppEnvironment.h"

using namespace FW;

#define TASK_SIZE 150000
#define BENCHMARK

//------------------------------------------------------------------------

CudaPersistentBVHBuilder::CudaPersistentBVHBuilder(Scene& scene, F32 epsilon) : CudaBVH(BVHLayout_Compact), m_epsilon(epsilon), m_numTris(scene.getNumTriangles()), m_trisCompact(scene.getTriCompactBuffer())
{
	// init
	CudaModule::staticInit();
	//m_compiler.addOptions("-use_fast_math");
	m_compiler.addOptions("-use_fast_math -Xptxas -dlcm=cg");

	m_trisCompactIndex.resizeDiscard(m_numTris * sizeof(S32));
	scene.getBBox(m_bboxMin, m_bboxMax);

	m_sizeTask = 0.f;
	m_sizeSplit = 0.f;
	m_sizeADS = 0.f;
	m_sizeTri = 0.f;
	m_sizeTriIdx = 0.f;
	m_heap = 0.f;

#ifndef BENCHMARK
	Debug.open("persistent_bvh_debug.log");
#endif
}

//------------------------------------------------------------------------

CudaPersistentBVHBuilder::~CudaPersistentBVHBuilder()
{
#ifndef BENCHMARK
	Debug.close();
#endif
}

//------------------------------------------------------------------------

F32 CudaPersistentBVHBuilder::build()
{
	// Compile the kernel
	m_compiler.setSourceFile("src/rt/kernels/persistent_bvh.cu");
	m_compiler.clearDefines();
	m_module = m_compiler.compile();
	failIfError();

#ifdef DEBUG_PPS
	Random rand;
	m_numTris = rand.getU32(1, 1000000);
#endif

	// Set triangle index buffer
	S32* tiout = (S32*)m_trisCompactIndex.getMutablePtr();
#ifdef DEBUG_PPS
	S32* pout = (S32*)m_ppsTris.getMutablePtr();
	S32* clout = (S32*)m_ppsTrisIndex.getMutablePtr();
	S32* sout = (S32*)m_sortTris.getMutablePtr();
#endif
	for(int i=0;i<m_numTris;i++)
	{
#ifndef DEBUG_PPS
		// indices 
		*tiout = i;
		tiout++;
#else
		int rnd = rand.getU32(0, 2);
		//*pout = rnd;
		*clout = rnd;
		*sout = (rnd >= 1);
		//pout++;
		clout++;
		sout++;
#endif
	}

	// Start the timer
	m_timer.unstart();
	m_timer.start();

	// Create the taskData
	m_taskData.resizeDiscard(TASK_SIZE * (sizeof(TaskBVH) + sizeof(int)));
	m_taskData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
	m_splitData.resizeDiscard((S64)(TASK_SIZE+1) * (S64)sizeof(SplitArray));
	m_splitData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#endif

	// Node and triangle data
	S64 bvhSize = align<S64, 4096>(m_numTris * sizeof(CudaBVHNode));
	m_nodes.resizeDiscard(bvhSize);
	m_nodes.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
	//m_nodes.clearRange32(0, 0, bvhSize); // Mark all tasks as 0 (important for debug)
#ifdef COMPACT_LAYOUT
	m_triWoop.resizeDiscard(m_numTris * (3+1) * sizeof(Vec4f));
	m_triIndex.resizeDiscard(m_numTris * (3+1) * sizeof(S32));
#endif

	m_gpuTime = buildCuda();
	m_cpuTime = m_timer.end();

	// Resize to exact memory
	trimBuffers();

#ifdef DEBUG_PPS
	exit(0);
#endif

	return m_gpuTime;
}

//------------------------------------------------------------------------

void CudaPersistentBVHBuilder::updateConstants()
{
	RtEnvironment& cudaEnv = *(RtEnvironment*)m_module->getGlobal("c_env").getMutablePtr();

	Environment::GetSingleton()->GetIntValue("PersistentBVH.maxDepth", cudaEnv.optMaxDepth);

	Environment::GetSingleton()->GetFloatValue("PersistentBVH.ci", cudaEnv.optCi);

	Environment::GetSingleton()->GetFloatValue("PersistentBVH.ct", cudaEnv.optCt);

	Environment::GetSingleton()->GetIntValue("PersistentBVH.triLimit", cudaEnv.triLimit);

	Environment::GetSingleton()->GetIntValue("PersistentBVH.triMaxLimit", cudaEnv.triMaxLimit);

	Environment::GetSingleton()->GetIntValue("PersistentBVH.popCount", cudaEnv.popCount);

	Environment::GetSingleton()->GetFloatValue("PersistentBVH.granularity", cudaEnv.granularity);
	
	cudaEnv.epsilon = m_epsilon;
	//cudaEnv.epsilon = 0.f;
}

//------------------------------------------------------------------------

void CudaPersistentBVHBuilder::initPool(Buffer* nodeBuffer)
{
	// Prepare the task data
	updateConstants();

	// Set PPS buffers
	m_ppsTris.resizeDiscard(sizeof(int)*m_numTris);
	m_ppsTrisIndex.resizeDiscard(sizeof(int)*m_numTris);
	m_sortTris.resizeDiscard(sizeof(int)*m_numTris);

#if defined(SNAPSHOT_POOL) || defined(SNAPSHOT_WARP)
	// Prepare snapshot memory
	Buffer snapData;
	allocateSnapshots(snapData);
#endif

	// Set all headers empty
	m_taskData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#ifdef BENCHMARK
	m_taskData.clearRange32(0, TaskHeader_Empty, TASK_SIZE * sizeof(int)); // Mark all tasks as empty
#else
	m_taskData.clearRange32(0, TaskHeader_Empty, TASK_SIZE * (sizeof(int)+sizeof(TaskBVH))); // Mark all tasks as empty (important for debug)
#endif

	// Set texture references.
	if(nodeBuffer != NULL)
	{
		m_module->setTexRef("t_nodesA", *nodeBuffer, CU_AD_FORMAT_FLOAT, 4);
	}
	m_module->setTexRef("t_trisA", m_trisCompact, CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_triIndices", m_trisCompactIndex, CU_AD_FORMAT_SIGNED_INT32, 1);
}

//------------------------------------------------------------------------

void CudaPersistentBVHBuilder::deinitPool()
{
	m_ppsTris.reset();
	m_ppsTrisIndex.reset();
	m_sortTris.reset();
}

//------------------------------------------------------------------------

void CudaPersistentBVHBuilder::printPoolHeader(TaskStackBase* tasks, int* header, int numWarps, FW::String state)
{
#if defined(SNAPSHOT_POOL) || defined(SNAPSHOT_WARP)
	printSnapshots(snapData);
#endif

#ifdef DEBUG_INFO
	Debug << "\nPRINTING DEBUG_INFO STATISTICS" << "\n\n";
#else
	Debug << "\nPRINTING STATISTICS" << "\n\n";
#endif

	float4* debugData = (float4*)m_debug.getPtr();
	float minAll[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
	float maxAll[4] = {0, 0, 0, 0};
	float sumAll[4] = {0, 0, 0, 0};
	int countDead = 0;
	Debug << "Warp No. cnt_task_queues Avg. #Reads Max #Reads #Restarts" << "\n";
	for(int i = 0; i < numWarps; i++)
	{
		Debug << "Warp " << i << ": (" << debugData[i].x << ", " << debugData[i].y << ", " << debugData[i].z << ", " << debugData[i].w << ")" << "\n";

		//fabs is because we do not care whether the warp stopped prematurely or not
		minAll[0] = min(fabs(debugData[i].x), minAll[0]);
		minAll[1] = min(fabs(debugData[i].y), minAll[1]);
		minAll[2] = min(fabs(debugData[i].z), minAll[2]);
		minAll[3] = min(fabs(debugData[i].w), minAll[3]);

		maxAll[0] = max(fabs(debugData[i].x), maxAll[0]);
		maxAll[1] = max(fabs(debugData[i].y), maxAll[1]);
		maxAll[2] = max(fabs(debugData[i].z), maxAll[2]);
		maxAll[3] = max(fabs(debugData[i].w), maxAll[3]);

		sumAll[0] += fabs(debugData[i].x);
		sumAll[1] += fabs(debugData[i].y);
		sumAll[2] += fabs(debugData[i].z);
		sumAll[3] += fabs(debugData[i].w);

		if(debugData[i].x < 0)
			countDead++;
	}
	Debug << "Dead=" << countDead << " / All=" << numWarps << " = " << (float)countDead/(float)numWarps << "\n";
	Debug << "Min: " << minAll[0] << ", " << minAll[1] << ", " << minAll[2] << ", " << minAll[3] << "\n";
	Debug << "Max: " << maxAll[0] << ", " << maxAll[1] << ", " << maxAll[2] << ", " << maxAll[3] << "\n";
	Debug << "Sum: " << sumAll[0] << ", " << sumAll[1] << ", " << sumAll[2] << ", " << sumAll[3] << "\n";
	Debug << "Avg: " << sumAll[0]/numWarps << ", " << sumAll[1]/numWarps << ", " << sumAll[2]/numWarps << ", " << sumAll[3]/numWarps << "\n\n" << "\n";
	Debug << "cnt_task_queues per object = " << sumAll[0]/(float)m_numTris << "\n";

	Debug << "Pool" << "\n";
	Debug << "Top = " << tasks->top << "; Bottom = " << tasks->bottom << "; Unfinished = " << tasks->unfinished << "; Size = " << tasks->sizePool << "; ";
	Debug << state.getPtr() << "\n";
	Debug << "ActiveTop = " << tasks->activeTop << "; Active = ";
	for(int i = 0; i < ACTIVE_MAX+1; i++)
		Debug << tasks->active[i] << " ";
	Debug << "\n" << "\n";
	Debug << "EmptyTop = " << tasks->emptyTop << "; EmptyBottom = " << tasks->emptyBottom  << "\nEmpty\n";
	for(int i = 0; i < EMPTY_MAX+1; i++)
	{
		if(i % 50 == 0)
			Debug << "\n";
		else
			Debug << " ";
		Debug << tasks->empty[i];
	}

	Debug << "\n" << "\n";

	int emptyItems = 0;
	int bellowEmpty = 0;
	Debug << "Header" << "\n";
	for(int i = 0; i < TASK_SIZE; i++)
	{
		if(i % 50 == 0)
			Debug << "\n";
		else
			Debug << " ";
		if(header[i] != TaskHeader_Empty)
		{
			Debug << header[i];
		}
		else
		{
			Debug << TaskHeader_Active;
			if(i < tasks->top)
				emptyItems++;
		}

		if(header[i] < TaskHeader_Empty)
			bellowEmpty++;
	}

	Debug << "\n\nEmptyItems = " << emptyItems << "\n";
	Debug << "BellowEmpty = " << bellowEmpty << "\n";
}

//------------------------------------------------------------------------

void CudaPersistentBVHBuilder::printPool(TaskStackBVH &tasks, int numWarps)
{
#ifdef LEAF_HISTOGRAM
	printf("Leaf histogram\n");
	unsigned int leafSum = 0;
	unsigned int triSum = 0;
	for(S32 i = 0; i <= Environment::GetSingleton()->GetInt("PersistentBVH.triLimit"); i++)
	{
		printf("%d: %d\n", i, tasks.leafHist[i]);
		leafSum += tasks.leafHist[i];
		triSum += i*tasks.leafHist[i];
	}
	printf("Leafs total %d, average leaf %.2f\n", leafSum, (float)triSum/(float)leafSum);
#endif

	int* header = (int*)m_taskData.getPtr();
	FW::String state = sprintf("BVH Top = %d; Tri Top = %d; Warp counter = %d; ", tasks.nodeTop, tasks.triTop, tasks.warpCounter);
#ifdef COUNT_NODES
	state.appendf("Number of inner nodes = %d; Number of leaves = %d; Sorted tris = %d; ", tasks.numNodes, tasks.numLeaves, tasks.numSortedTris);
#endif
	printPoolHeader(&tasks, header, numWarps, state);

	Debug << "\n\nTasks" << "\n";
	TaskBVH* task = (TaskBVH*)m_taskData.getPtr(TASK_SIZE*sizeof(int));
	int stackMax = 0;
	int maxDepth = 0;
	int syncCount = 0;
	int maxTaskId = -1;
	long double sumTris = 0;
	long double maxTris = 0;

	int sortTasks = 0;
	long double cntSortTris = 0;

	int subFailed = 0;

#ifdef DEBUG_INFO
	char terminatedNames[TerminatedBy_Max][255] = {
		"None", "Depth","TotalLimit","OverheadLimit","Cost","FailureCounter"
	};

	int terminatedBy[TerminatedBy_Max];
	memset(&terminatedBy,0,sizeof(int)*TerminatedBy_Max);
#endif

	for(int i = 0; i < TASK_SIZE; i++)
	{
		if(task[i].nodeIdx != TaskHeader_Empty || task[i].parentIdx != TaskHeader_Empty)
		{
#ifdef DEBUG_INFO
			_ASSERT(task[i].terminatedBy >= 0 && task[i].terminatedBy < TerminatedBy_Max);
			terminatedBy[ task[i].terminatedBy ]++;
#endif

			Debug << "Task " << i << "\n";
			Debug << "Header: " << header[i] << "\n";
			Debug << "Unfinished: " << task[i].unfinished << "\n";
			Debug << "Type: " << task[i].type << "\n";
			Debug << "TriStart: " << task[i].triStart << "\n";
			Debug << "TriLeft: " << task[i].triLeft << "\n";
			Debug << "TriRight: " << task[i].triRight << "\n";
			Debug << "TriEnd: " << task[i].triEnd << "\n";
			Debug << "ParentIdx: " << task[i].parentIdx << "\n";
			Debug << "NodeIdx: " << task[i].nodeIdx << "\n";
			Debug << "TaskID: " << task[i].taskID << "\n";
			Debug << "Split: (" << task[i].splitPlane.x << ", " << task[i].splitPlane.y << ", " << task[i].splitPlane.z << ", " << task[i].splitPlane.w << ")\n";
			Debug << "Box: (" << task[i].bbox.m_mn.x << ", " << task[i].bbox.m_mn.y << ", " << task[i].bbox.m_mn.z << ") - ("
				<< task[i].bbox.m_mx.x << ", " << task[i].bbox.m_mx.y << ", " << task[i].bbox.m_mx.z << ")\n";
			Debug << "BoxLeft: (" << task[i].bboxLeft.m_mn.x << ", " << task[i].bboxLeft.m_mn.y << ", " << task[i].bboxLeft.m_mn.z << ") - ("
				<< task[i].bboxLeft.m_mx.x << ", " << task[i].bboxLeft.m_mx.y << ", " << task[i].bboxLeft.m_mx.z << ")\n";
			Debug << "BoxRight: (" << task[i].bboxRight.m_mn.x << ", " << task[i].bboxRight.m_mn.y << ", " << task[i].bboxRight.m_mn.z << ") - ("
				<< task[i].bboxRight.m_mx.x << ", " << task[i].bboxRight.m_mx.y << ", " << task[i].bboxRight.m_mx.z << ")\n";
			Debug << "Axis: " << task[i].axis << "\n";
			Debug << "Depth: " << task[i].depth << "\n";
			Debug << "Step: " << task[i].step << "\n";
#ifdef DEBUG_INFO
			//Debug << "Step: " << task[i].step << "\n";
			//Debug << "Lock: " << task[i].lock << "\n";
			Debug << "GMEMSync: " << task[i].sync << "\n";
			Debug << "Parent: " << task[i].parent << "\n";
#endif

#ifdef DEBUG_INFO
			Debug << "TerminatedBy: " << task[i].terminatedBy << "\n";
#endif
			if(task[i].terminatedBy != TerminatedBy_None)
				Debug << "Triangles: " << task[i].triEnd - task[i].triStart << "\n";

			Debug << "\n";
			stackMax = i;

			if(header[i] > (int)0xFF800000) // Not waiting
			{
				long double tris = task[i].triEnd - task[i].triStart;
				if(task[i].terminatedBy != TerminatedBy_None)
				{
					if(tris > maxTris)
					{
						maxTris = tris;
						maxTaskId = i;
					}
					sumTris += tris;
				}
				sortTasks++;
				cntSortTris += tris;

#ifdef DEBUG_INFO
				maxDepth = max(task[i].depth, maxDepth);
				syncCount += task[i].sync;
#endif
			}
		}
	}

	if(stackMax == TASK_SIZE-1)
		printf("\aIncomplete result!\n");
	Debug << "\n\n";

#ifdef DEBUG_INFO
	Debug << "Avg naive task height (tris) = " << sumTris/(long double)sortTasks << "\n";
	Debug << "Max naive task height (tris) = " << maxTris << ", taskId: " << maxTaskId << "\n";
	Debug << "Cnt sorted operations = " << sortTasks << "\n";
	double cntTrisLog2Tris = (double(m_numTris) * (double)(logf(m_numTris)/logf(2.0f)));
	Debug << "Cnt sorted triangles = " << cntSortTris << "\n";	
	Debug << "Cnt sorted triangles/(N log N), N=#tris = " << cntSortTris/cntTrisLog2Tris << "\n";
	Debug << "\n";
	Debug << "Max task depth = " << maxDepth << "\n";
	Debug << "Cnt gmem synchronizations: " << syncCount << "\n";
	Debug << "Leafs failed to subdivide = " << subFailed << " (*3) => total useless tasks " << subFailed * 3 << "\n";
	Debug << "Terminated by:" << "\n";
	for(int i = 0; i < TerminatedBy_Max; i++)
	{
		Debug << terminatedNames[i] << ": " << terminatedBy[i] << "\n";
	}
#endif

	Debug << "max_queue_length = " << stackMax << "\n\n" << "\n";
}

//------------------------------------------------------------------------

F32 CudaPersistentBVHBuilder::buildCuda()
{
	CudaKernel kernel;
	kernel = m_module->getKernel("build");

	// Prepare the task data
	initPool();

	// Set input.
	KernelInputBVH& in = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
	in.numTris		= m_numTris;
	in.tris         = m_trisCompact.getCudaPtr();
	in.trisIndex    = m_trisCompactIndex.getMutableCudaPtr();
	//in.trisBox      = m_trisBox.getCudaPtr();
	in.ppsTrisBuf   = m_ppsTris.getMutableCudaPtr();
	in.ppsTrisIndex = m_ppsTrisIndex.getMutableCudaPtr();
	in.sortTris     = m_sortTris.getMutableCudaPtr();
#ifdef COMPACT_LAYOUT
	in.trisOut      = m_triWoop.getMutableCudaPtr();
	in.trisIndexOut = m_triIndex.getMutableCudaPtr();
#endif

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
	SplitRed split;
	for(int i = 0; i < 2; i++)
	{
		split.children[i].bbox.m_mn = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		split.children[i].bbox.m_mx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		split.children[i].cnt = 0;
	}

	SplitArray sArray;
	for(int i = 0; i < NUM_WARPS; i++)
	{
		for(int j = 0; j < PLANE_COUNT; j++)
			sArray.splits[i][j] = split;
	}
#else
	SplitRed split;
	for(int i = 0; i < 2; i++)
	{
		//split.children[i].bbox.m_mn = make_float3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
		//split.children[i].bbox.m_mx = make_float3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
		split.children[i].bbox.m_mn = make_int3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
		split.children[i].bbox.m_mx = make_int3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
		split.children[i].cnt = 0;
	}

	SplitArray sArray;
	for(int j = 0; j < PLANE_COUNT; j++)
		sArray.splits[j] = split;

	m_splitData.setRange(0, &sArray, sizeof(SplitArray)); // Set the first splits
#endif

	m_splitData.setRange(TASK_SIZE * sizeof(SplitArray), &sArray, sizeof(SplitArray)); // Set the last splits for copy
#endif

	CudaAABB bbox;
	memcpy(&bbox.m_mn, &m_bboxMin, sizeof(float3));
	memcpy(&bbox.m_mx, &m_bboxMax, sizeof(float3));

	// Set parent task containing all the work
	TaskBVH all;
	all.triStart     = 0;
	all.triLeft      = 0;
	all.triRight     = m_numTris;
	all.triEnd       = m_numTris;
	all.bbox         = bbox;
	all.step         = 0;
	all.lock         = LockType_Free;
	all.bestCost     = 1e38f;
	all.depth        = 0;
	all.dynamicMemory= 0;
	all.triIdxCtr    = 0;
	all.parentIdx    = -1;
	all.nodeIdx      = 0;
	all.taskID       = 0;
	Vec3f size       = m_bboxMax - m_bboxMin;
	all.axis         = size.x > size.y ? (size.x > size.z ? 0 : 2) : (size.y > size.z ? 1 : 2);
	all.terminatedBy = TerminatedBy_None;
#ifdef DEBUG_INFO
	all.sync         = 0;
	all.parent       = -1;
	all.clockStart   = 0;
	all.clockEnd     = 0;
#endif

#if SPLIT_TYPE == 0
#if SCAN_TYPE == 0
	all.type         = TaskType_Sort_PPS1;
#elif SCAN_TYPE == 1
	all.type         = TaskType_Sort_PPS1_Up;
#elif SCAN_TYPE == 2 ||  SCAN_TYPE == 3
	all.type         = TaskType_Sort_SORT1;
#endif
	all.unfinished   = warpSubtasks(m_numTris);
	float pos = m_bbox.min[all.axis] + m_bbox.Size(all.axis)/2.0f;
	if(all.axis == 0)
		all.splitPlane   = make_float4(1.f, 0.f, 0.f, -pos);
	else if(all.axis == 1)
		all.splitPlane   = make_float4(0.f, 1.f, 0.f, -pos);
	else
		all.splitPlane   = make_float4(0.f, 0.f, 1.f, -pos);
#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
	all.type         = TaskType_InitMemory;
	all.unfinished   = warpSubtasks(sizeof(SplitArray)/sizeof(int));
#else
	all.type         = TaskType_BinTriangles;
	all.unfinished   = (warpSubtasks(m_numTris)+BIN_MULTIPLIER-1)/BIN_MULTIPLIER;
	/*all.type         = TaskType_BuildObjectSAH;
	all.unfinished   = 1;*/
#endif
#endif
	all.origSize     = all.unfinished;

	m_taskData.setRange(TASK_SIZE * sizeof(int), &all, sizeof(TaskBVH)); // Set the first task

	// Set parent task header
	m_taskData.setRange(0, &all.unfinished, sizeof(int)); // Set the first task

	// Prepare the task stack
	TaskStackBVH& tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getMutablePtr();
	tasks.header     = (int*)m_taskData.getMutableCudaPtr();
	tasks.tasks      = (TaskBVH*)m_taskData.getMutableCudaPtr(TASK_SIZE * sizeof(int));
	tasks.nodeTop    = 1;
	tasks.triTop     = 0;
	tasks.top        = 0;
	tasks.bottom     = 0;
	//memset(tasks.active, 0, sizeof(int)*(ACTIVE_MAX+1));
	memset(tasks.active, -1, sizeof(int)*(ACTIVE_MAX+1));
	tasks.active[0] = 0;
	/*for(int i = 0; i < ACTIVE_MAX+1; i++)
	tasks.active[i] = i;*/
	tasks.activeTop = 1;
	//tasks.empty[0] = 0;
	//int j = 1;
	//for(int i = EMPTY_MAX; i > 0; i--, j++)
	//	tasks.empty[i] = j;
	memset(tasks.empty, 0, sizeof(int)*(EMPTY_MAX+1));
	tasks.emptyTop = 0;
	tasks.emptyBottom = 0;
	tasks.unfinished = -1; // We are waiting for one task to finish = task all
	tasks.numSortedTris = 0;
	tasks.numNodes = 0;
	tasks.numLeaves = 0;
	tasks.sizePool = TASK_SIZE;
	tasks.sizeNodes = m_nodes.getSize()/sizeof(CudaBVHNode);
	tasks.sizeTris = m_triIndex.getSize()/sizeof(S32);
	memset(tasks.leafHist, 0, sizeof(tasks.leafHist));

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
	// Prepare split stack
	SplitArray* &splits = *(SplitArray**)m_module->getGlobal("g_redSplits").getMutablePtr();
	splits = (SplitArray*)m_splitData.getMutableCudaPtr();
#endif

	CudaBVHNode* &bvh = *(CudaBVHNode**)m_module->getGlobal("g_bvh").getMutablePtr();
	bvh = (CudaBVHNode*)m_nodes.getMutableCudaPtr();

	// Determine block and grid sizes.
#ifdef ONE_WARP_RUN
	Vec2i blockSize(WARP_SIZE, 1); // threadIdx.x must equal the thread lane in warp
	Vec2i gridSize(1, 1); // Number of SMs * Number of blocks?
	int numWarps = 1;
#else
	int numWarpsPerBlock = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numWarpsPerBlock");
	int numBlocksPerSM = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numBlockPerSM");
	Vec2i blockSize(WARP_SIZE, numWarpsPerBlock); // threadIdx.x must equal the thread lane in warp
	int gridSizeX = NUM_SM*numBlocksPerSM;
	int numWarps = numWarpsPerBlock*gridSizeX;
	Vec2i gridSize(gridSizeX, 1); // Number of SMs * Number of blocks?

	if(gridSizeX*numWarpsPerBlock != NUM_WARPS)
		printf("\aNUM_WARPS constant does not match the launch parameters\n");
#endif

	m_debug.resizeDiscard(blockSize.y*gridSize.x*sizeof(float4));
	m_debug.clear();
	in.debug = m_debug.getMutableCudaPtr();

	// Launch.
	kernel.setGridExact(blockSize, gridSize);
	float tKernel = kernel.launchTimed();

#ifndef BENCHMARK
	cuCtxSynchronize(); // Flushes printfs
#endif

#ifdef DEBUG_PPS
	pout = (S32*)m_ppsTris.getPtr();
	sout = (S32*)m_sortTris.getPtr();
	S32 sum = 0;
	S32 error = 0;
	int j = 0;
	for(int i=0;i<m_numTris;i++)
	{
		sum += *sout; // Here for inclusive scan
		if(*pout != sum)
		{
			cout << "PPS error for item " << i << ", CPU=" << sum << ", GPU=" << *pout << " for " << m_numTris << " triangles!" << "\n";
			error = 1;
			if(j == 10)
				break;
			j++;
		}
		if(*sout != 0 && *sout != 1)
		{
			cout << "\nWTF " << i << " of " << m_numTris << ": " << *sout << "!\n" << "\n";
			break;
		}
		//sum += *sout; // Here for exclusive scan
		pout++;
		sout++;
	}

	if(!error)
		cout << "PPS correct for " << m_numTris << " triangles!" << "\n";
	return 0;
#endif

	tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();
	if(tasks.unfinished != 0 || tasks.top >= tasks.sizePool || tasks.nodeTop >= m_nodes.getSize() / sizeof(CudaBVHNode) || tasks.triTop >= m_triIndex.getSize() / sizeof(S32)) // Something went fishy
	{
		tKernel = 1e38f;
		fail("%d (%d x %d) (%d x %d) (%d x %d)\n", tasks.unfinished != 0, tasks.top, tasks.sizePool, tasks.nodeTop, m_nodes.getSize() / sizeof(CudaBVHNode), tasks.triTop, m_triIndex.getSize() / sizeof(S32));
	}

	//Debug << "\nBuild in " << tKernel << "\n\n";

#ifndef BENCHMARK
	printPool(tasks, numWarps);

	/*Debug << "\n\nBVH" << "\n";
	CudaBVHNode* nodes = (CudaBVHNode*)m_nodes.getPtr();

	for(int i = 0; i < tasks.nodeTop; i++)
	{
		Debug << "Node " << i << "\n";
		Debug << "BoxLeft: (" << nodes[i].c0xy.x << ", " << nodes[i].c0xy.z << ", " << nodes[i].c01z.x << ") - ("
				<< nodes[i].c0xy.y << ", " << nodes[i].c0xy.w << ", " << nodes[i].c01z.y << ")\n";
		Debug << "BoxRight: (" << nodes[i].c1xy.x << ", " << nodes[i].c1xy.z << ", " << nodes[i].c01z.z << ") - ("
				<< nodes[i].c1xy.y << ", " << nodes[i].c1xy.w << ", " << nodes[i].c01z.w << ")\n";
		Debug << "Children: " << nodes[i].children.x << ", " << nodes[i].children.y << "\n\n";
	}*/

	// Free data
	deinitPool();
#endif

	return tKernel;
}

//------------------------------------------------------------------------

void CudaPersistentBVHBuilder::saveBufferSizes(bool ads, bool aux)
{
	float MB = (float)(1024*1024);

	if(ads)
	{
		m_sizeADS    = m_nodes.getSize()/MB;
#ifndef COMPACT_LAYOUT
		m_sizeTri    = m_trisCompact.getSize()/MB;
		m_sizeTriIdx = m_trisCompactIndex.getSize()/MB;
#else
		m_sizeTri    = m_triWoop.getSize()/MB;
		m_sizeTriIdx = m_triIndex.getSize()/MB;
#endif
	}

	if(aux)
	{
		m_sizeTask   = m_taskData.getSize()/MB;
		m_sizeSplit  = m_splitData.getSize()/MB;
		m_heap       = 0.f;
	}
}

//------------------------------------------------------------------------

void CudaPersistentBVHBuilder::resetBuffers(bool resetADSBuffers)
{
	// Reset buffers so that reuse of space does not cause timing disturbs
	if(resetADSBuffers)
	{
		m_nodes.reset();
		m_triWoop.reset();
		m_triIndex.reset();
	}

	m_taskData.reset();
	m_splitData.reset();

	m_ppsTris.reset();
	m_ppsTrisIndex.reset();
	m_sortTris.reset();
}

//------------------------------------------------------------------------

void CudaPersistentBVHBuilder::trimBuffers()
{
	// Save sizes of auxiliary buffers so that they can be printed
	saveBufferSizes(false, true);
	// Free auxiliary buffers
	resetBuffers(false);

	// Resize to exact memory
	U32 nN, nL, sT, bT, tT, sTr; 
	getStats(nN, nL, sT, bT, tT, sTr);
#ifdef COMPACT_LAYOUT
	m_nodes.resize(nN * sizeof(CudaBVHNode));
	m_triWoop.resize(tT*3*sizeof(float4) + nL*sizeof(float4));
	m_triIndex.resize(tT*3*sizeof(int) + nL*sizeof(int));
#else
	m_nodes.resize((nN + nL) * sizeof(CudaBVHNode));
#endif

	// Save sizes of ads buffers so that they can be printed
	saveBufferSizes(true, false);
}

//------------------------------------------------------------------------

void CudaPersistentBVHBuilder::getStats(U32& nodes, U32& leaves, U32& stackTop, U32& nodeTop, U32& tris, U32& sortedTris, bool sub)
{
	TaskStackBVH tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();

#ifndef COUNT_NODES
#ifndef COMPACT_LAYOUT
	nodes = tasks.nodeTop / 2;
	leaves = tasks.nodeTop - nodes;
#else
	nodes = tasks.nodeTop;
	leaves = tasks.triTop;
#endif
#else // COUNT_NODES
	nodes = tasks.numNodes;
	leaves = tasks.numLeaves;
#endif // COUNT_NODES

#ifdef COMPACT_LAYOUT
	tris = tasks.triTop;
	if(sub)
		tris -= leaves;
#ifdef DUPLICATE_REFERENCES
	tris /= 3;
#endif
#else
	if(sub)
	{
		tris = m_numTris;
	}
	else
	{
		tris = tasks.triTop;
		tris /= 3;
	}
#endif

	nodeTop = tasks.nodeTop;
	sortedTris = tasks.numSortedTris;
	stackTop = tasks.top;
}

//------------------------------------------------------------------------

void CudaPersistentBVHBuilder::getSizes(F32& task, F32& split, F32& ads, F32& tri, F32& triIdx, F32& heap)
{
	task = m_sizeTask;
	split = m_sizeSplit;
	ads = m_sizeADS;
	tri = m_sizeTri;
	triIdx = m_sizeTriIdx;
	heap = m_heap;
}

//------------------------------------------------------------------------
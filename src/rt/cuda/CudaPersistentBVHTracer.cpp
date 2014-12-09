#include <sstream>
#include <iomanip>
#include "base/Random.hpp"
#include "CudaPersistentBVHTracer.hpp"
#include "kernels/CudaTracerKernels.hpp"
#include "persistentbvh/CudaBuilderKernels.hpp"
#include "../AppEnvironment.h"

#define TASK_SIZE 150000

using namespace FW;

#define CIRCULAR_MALLOC_PREALLOC 3
#define CPU 0
#define GPU 1
#define CIRCULAR_MALLOC_INIT GPU
#define CIRCULAR_MALLOC_PREALLOC_SPECIAL

#define BENCHMARK
#define TEST_TASKS

#ifndef TEST_TASKS
#include "kernels/thrustTest.hpp"
#endif

ofstream Debug;

// Alignment to multiply of S
template<typename T, int S>
T align(T a)
{
	 return (a+S-1) & -S;
}

//------------------------------------------------------------------------

CudaPersistentBVHTracer::CudaPersistentBVHTracer(Scene& scene, F32 epsilon) : CudaBVH(BVHLayout_Compact), m_epsilon(epsilon)
{
	// init
	CudaModule::staticInit();
	//m_compiler.addOptions("-use_fast_math");
	m_compiler.addOptions("-use_fast_math -Xptxas -dlcm=cg");
	m_compiler.clearDefines();
	if (CudaModule::getComputeCapability() == 20 || CudaModule::getComputeCapability() == 21)
		m_compiler.define("FERMI");

	// convert from scene
	Vec3f light = Vec3f(1.0f, 2.0f, 3.0f).normalized();

	m_numTris = scene.getTriVtxIndexBuffer().getSize() / sizeof(Vec3i);
	m_numVerts          = m_numTris * 3;
	
	m_tris.resizeDiscard(m_numTris * sizeof(Vec3i));

	m_trisCompact.resizeDiscard(m_numTris * 3 * sizeof(Vec4f));
	m_trisIndex.resizeDiscard(m_numTris * sizeof(S32));

	Vec3i*			tout  = (Vec3i*)m_tris.getMutablePtr();

	Vec4f* tcout  = (Vec4f*)m_trisCompact.getMutablePtr();
	S32*   tiout  = (S32*)m_trisIndex.getMutablePtr();

	m_bboxMin = Vec3f(2e30f);
	m_bboxMax = Vec3f(-2e30f);

	// Load vertices
	for (int i = 0; i < m_numTris; i++)
	{
		Vec3i& tri = ((Vec3i*)scene.getTriVtxIndexBuffer().getPtr())[i];
		Vec3f& a = ((Vec3f*)scene.getVtxPosBuffer().getPtr())[tri.x];
		Vec3f& b = ((Vec3f*)scene.getVtxPosBuffer().getPtr())[tri.y];
		Vec3f& c = ((Vec3f*)scene.getVtxPosBuffer().getPtr())[tri.z];
		*tcout = Vec4f(a.x, a.y, a.z, 0.0f); tcout++;
		*tcout = Vec4f(b.x, b.y, b.z, 0.0f); tcout++;
		*tcout = Vec4f(c.x, c.y, c.z, 0.0f); tcout++;
		
		m_bboxMin = FW::min(m_bboxMin, a);
		m_bboxMin = FW::min(m_bboxMin, b);
		m_bboxMin = FW::min(m_bboxMin, c);
		m_bboxMax = FW::max(m_bboxMax, a);
		m_bboxMax = FW::max(m_bboxMax, b);
		m_bboxMax = FW::max(m_bboxMax, c);
	}

	// load triangles
	for(int i=0,j=0;i<m_numTris;i++,j+=3)
	{
		// triangle data
		tout[i] = Vec3i(j,j+1,j+2);
	}

	m_sizeTask = 0.f;
	m_sizeSplit = 0.f;
	m_sizeADS = 0.f;
	m_sizeTri = 0.f;
	m_sizeTriIdx = 0.f;
	m_heap = 0.f;
}

F32 CudaPersistentBVHTracer::buildBVH(bool sbvh)
{
#ifdef MALLOC_SCRATCHPAD
	// Set the memory limit according to triangle count
#ifndef BENCHMARK
	printf("Setting dynamic memory limit to %fMB\n", (float)(m_trisIndex.getSize()*5*3)/(float)(1024*1024));
#endif
	 cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, m_trisIndex.getSize()*5*3);
#endif

	// Compile the kernel
	if(!sbvh)
		m_kernelFile = "src/rt/kernels/persistent_bvh.cu";
	else
		m_kernelFile = "src/rt/kernels/persistent_sbvh.cu";

	m_compiler.setSourceFile(m_kernelFile);
	m_module = m_compiler.compile();
	failIfError();

#ifdef DEBUG_PPS
	Random rand;
	m_numTris = rand.getU32(1, 1000000);
#endif

	// Set triangle index buffer
	S32* tiout = (S32*)m_trisIndex.getMutablePtr();
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
	S64 bvhSize = ((m_numTris * sizeof(CudaBVHNode)) + 4096 - 1) & -4096;
	//S64 bvhSize = ((m_numTris/2 * sizeof(CudaBVHNode)) + 4096 - 1) & -4096;
	m_bvhData.resizeDiscard(bvhSize);
	m_bvhData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
	//m_bvhData.clearRange32(0, 0, bvhSize); // Mark all tasks as 0 (important for debug)
#ifdef COMPACT_LAYOUT
	if(!sbvh)
	{
		m_trisCompactOut.resizeDiscard(m_numTris * (3+1) * sizeof(Vec4f));
		m_trisIndexOut.resizeDiscard(m_numTris * (3+1) * sizeof(S32));
	}
	else
	{
		m_trisCompactOut.resizeDiscard(m_numTris*2 * (3+1) * sizeof(Vec4f));
		m_trisIndexOut.resizeDiscard(m_numTris*2 * (3+1) * sizeof(S32));
	}
#endif

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
	m_splitData.resizeDiscard((S64)(TASK_SIZE+1) * (S64)sizeof(SplitArray));
	m_splitData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#endif

	m_gpuTime = buildCudaBVH();
	m_cpuTime = m_timer.end();

	// Resize to exact memory
	trimBVHBuffers();

#ifdef DEBUG_PPS
	exit(0);
#endif

	return m_gpuTime;
}

F32 CudaPersistentBVHTracer::traceBatchBVH(RayBuffer& rays, RayStats* stats)
{
#ifdef TRACE_L1
	// Set compiler options
	m_compiler.clearOptions();
#endif

	m_compiler.setCachePath("cudacache"); // On the first compilation the cache path becames absolute which kills the second compilation
#ifdef COMPACT_LAYOUT
#ifdef WOOP_TRIANGLES
	String kernelName("src/rt/kernels/fermi_speculative_while_while");
#else
	String kernelName("src/rt/kernels/fermi_speculative_while_while_inter");
#endif
#ifdef TRACE_L1
	m_compiler.addOptions("-use_fast_math");
#endif
#else
	String kernelName("src/rt/kernels/fermi_persistent_speculative_while_while_inter");
#ifdef TRACE_L1
	m_compiler.addOptions("-use_fast_math -maxrregcount 40");
#endif
#endif
	if(stats != NULL)
	{
		kernelName += "_statistics";
	}
	kernelName += ".cu";
	m_compiler.setSourceFile(kernelName);

	m_module = m_compiler.compile();
	failIfError();

	CudaKernel queryKernel = m_module->getKernel("queryConfig");
	if (!queryKernel.getHandle())
        fail("Config query kernel not found!");

    // Initialize config with default values.
	KernelConfig& kernelConfig = *(KernelConfig*)m_module->getGlobal("g_config").getMutablePtr();
	kernelConfig.bvhLayout             = BVHLayout_Max;
	kernelConfig.blockWidth            = 0;
	kernelConfig.blockHeight           = 0;
	kernelConfig.usePersistentThreads  = 0;

    // Query config.

	queryKernel.launch(1, 1);
    kernelConfig = *(const KernelConfig*)m_module->getGlobal("g_config").getPtr();

	CudaKernel kernel;
	if(stats != NULL)
		kernel = m_module->getKernel("trace_stats");
	else
		kernel = m_module->getKernel("trace");
	if (!kernel.getHandle())
		fail("Trace kernel not found!");

	KernelInput& in = *(KernelInput*)m_module->getGlobal("c_in").getMutablePtr();
	// Start the timer
	m_timer.unstart();
	m_timer.start();

	CUdeviceptr nodePtr     = m_bvhData.getCudaPtr();
	Vec2i       nodeOfsA    = Vec2i(0, (S32)m_bvhData.getSize());

#ifdef COMPACT_LAYOUT
	CUdeviceptr triPtr      = m_trisCompactOut.getCudaPtr();
	Vec2i       triOfsA     = Vec2i(0, (S32)m_trisCompactOut.getSize());
	Buffer&     indexBuf    = m_trisIndexOut;
#else
	CUdeviceptr triPtr      = m_trisCompact.getCudaPtr();
	Vec2i       triOfsA     = Vec2i(0, (S32)m_trisCompact.getSize());
	Buffer&     indexBuf    = m_trisIndex;
#endif	

	// Set input.
	// The new version has it via parameters, not const memory
	in.numRays      = rays.getSize();
	in.anyHit       = (rays.getNeedClosestHit() == false);
	in.nodesA       = nodePtr + nodeOfsA.x;
	in.trisA        = triPtr + triOfsA.x;
	in.rays         = rays.getRayBuffer().getCudaPtr();
	in.results      = rays.getResultBuffer().getMutableCudaPtr();
	in.triIndices   = indexBuf.getCudaPtr();

	// Set texture references.
	m_module->setTexRef("t_rays", rays.getRayBuffer(), CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_nodesA", nodePtr + nodeOfsA.x, nodeOfsA.y, CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_trisA", triPtr + triOfsA.x, triOfsA.y, CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_triIndices", indexBuf, CU_AD_FORMAT_SIGNED_INT32, 1);

	// Determine block and grid sizes.
	int desiredWarps = (rays.getSize() + 31) / 32;
	if (kernelConfig.usePersistentThreads != 0)
	{
		*(S32*)m_module->getGlobal("g_warpCounter").getMutablePtr() = 0;
		desiredWarps = 720; // Tesla: 30 SMs * 24 warps, Fermi: 15 SMs * 48 warps
	}

	Vec2i blockSize(kernelConfig.blockWidth, kernelConfig.blockHeight);
	int blockWarps = (blockSize.x * blockSize.y + 31) / 32;
	Vec2i gridSize((desiredWarps + blockWarps - 1) / blockWarps, 1);

	if(stats != NULL)
	{
		m_module->getGlobal("g_NumNodes").clear();
		m_module->getGlobal("g_NumLeaves").clear();
		m_module->getGlobal("g_NumEmptyLeaves").clear();
		m_module->getGlobal("g_NumTris").clear();
		m_module->getGlobal("g_NumFailedTris").clear();
		m_module->getGlobal("g_NumHitTrisOutside").clear();
	}

	// Launch.
	F32 launchTime = kernel.launchTimed(blockSize, gridSize);

	if(stats != NULL)
	{
		stats->numNodeTests += *(U32*)m_module->getGlobal("g_NumNodes").getPtr();
		stats->numLeavesVisited += *(U32*)m_module->getGlobal("g_NumLeaves").getPtr();
		stats->numEmptyLeavesVisited += *(U32*)m_module->getGlobal("g_NumEmptyLeaves").getPtr();
		stats->numTriangleTests += *(U32*)m_module->getGlobal("g_NumTris").getPtr();
		stats->numFailedTriangleTests += *(U32*)m_module->getGlobal("g_NumFailedTris").getPtr();
		stats->numSuccessTriangleTestsOutside += *(U32*)m_module->getGlobal("g_NumHitTrisOutside").getPtr();
		stats->numRays += rays.getSize();
	}

	m_gpuTime = launchTime;
	m_cpuTime = m_timer.end();

#ifdef TRACE_L1
	// reset options
	m_compiler.clearOptions();
	m_compiler.addOptions("-use_fast_math -Xptxas -dlcm=cg");
#endif

	return launchTime;
}

void CudaPersistentBVHTracer::updateConstants()
{
	RtEnvironment& cudaEnv = *(RtEnvironment*)m_module->getGlobal("c_env").getMutablePtr();

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.maxDepth", cudaEnv.optMaxDepth);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.planeSelectionOverhead", cudaEnv.optPlaneSelectionOverhead);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.ci", cudaEnv.optCi);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.ct", cudaEnv.optCt);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.ctr", cudaEnv.optCtr);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.ctt", cudaEnv.optCtt);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.triangleBasedWeight", cudaEnv.optTriangleBasedWeight);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.rayBasedWeight", cudaEnv.optRayBasedWeight);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.axisAlignedWeight", cudaEnv.optAxisAlignedWeight);

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.cutOffDepth", cudaEnv.optCutOffDepth);
	m_cutOffDepth = cudaEnv.optCutOffDepth;

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.rayLimit", cudaEnv.rayLimit);

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.triLimit", cudaEnv.triLimit);
	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.triMaxLimit", cudaEnv.triMaxLimit);

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.popCount", cudaEnv.popCount);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.granularity", cudaEnv.granularity);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.failRq", cudaEnv.failRq);

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.failureCount", cudaEnv.failureCount);

	int siblingLimit;
	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.siblingLimit", siblingLimit);
	cudaEnv.siblingLimit = siblingLimit / WARP_SIZE;

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.childLimit", cudaEnv.childLimit);

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.subtreeLimit", cudaEnv.subtreeLimit);

	Vec3f d = m_bboxMax - m_bboxMin;
	Vec3f d_s = Vec3f(d.x, d.x, d.y) * Vec3f(d.y, d.z, d.z);
	cudaEnv.subdivThreshold = ((d_s.x + d_s.y + d_s.z) / (float)m_numRays) * ((float)cudaEnv.optCt/10.0f);
	
	cudaEnv.epsilon = m_epsilon;
	//cudaEnv.epsilon = 0.f;
}

//------------------------------------------------------------------------

int CudaPersistentBVHTracer::warpSubtasks(int threads)
{
	//return (threads + WARP_SIZE - 1) / WARP_SIZE;
	return max((threads + WARP_SIZE - 1) / WARP_SIZE, 1); // Do not create empty tasks - at least on warp gets to clean this task
}

//------------------------------------------------------------------------

int CudaPersistentBVHTracer::floatToOrderedInt(float floatVal)
{
	int intVal = *((int*)&floatVal);
	return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

//------------------------------------------------------------------------

void CudaPersistentBVHTracer::initPool(int numRays, Buffer* rayBuffer, Buffer* nodeBuffer)
{
	// Prepare the task data
	updateConstants();
#if PARALLELISM_TEST >= 0
	int& numActive = *(int*)m_module->getGlobal("g_numActive").getMutablePtr();
	numActive = 1;
#endif

#ifndef MALLOC_SCRATCHPAD
	// Set PPS buffers
	m_ppsTris.resizeDiscard(sizeof(int)*m_numTris);
	m_ppsTrisIndex.resizeDiscard(sizeof(int)*m_numTris);
	m_sortTris.resizeDiscard(sizeof(int)*m_numTris);

	if(numRays > 0)
	{
		m_ppsRays.resizeDiscard(sizeof(int)*numRays);
		m_ppsRaysIndex.resizeDiscard(sizeof(int)*numRays);
		m_sortRays.resizeDiscard(sizeof(int)*numRays);
	}
#endif

#if defined(SNAPSHOT_POOL) || defined(SNAPSHOT_WARP)
	// Prepare snapshot memory
	Buffer snapData;
	allocateSnapshots(snapData);
#endif

	// Set all headers empty
#ifdef TEST_TASKS
	m_taskData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#ifdef BENCHMARK
	m_taskData.clearRange32(0, TaskHeader_Empty, TASK_SIZE * sizeof(int)); // Mark all tasks as empty
#else
	m_taskData.clearRange32(0, TaskHeader_Empty, TASK_SIZE * (sizeof(int)+sizeof(Task))); // Mark all tasks as empty (important for debug)
#endif
#endif

	// Increase printf output size so that more can fit
	//cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 536870912);

	/*cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED); // Driver does not seem to care and preffers L1
	cuFuncSetCacheConfig(kernel, CU_FUNC_CACHE_PREFER_SHARED);
	CUfunc_cache test;
	cuCtxGetCacheConfig(&test);
	if(test != CU_FUNC_CACHE_PREFER_SHARED)
		printf("Error\n");*/

	// Set texture references.
	if(rayBuffer != NULL)
	{
		m_module->setTexRef("t_rays", *rayBuffer, CU_AD_FORMAT_FLOAT, 4);
	}
	if(nodeBuffer != NULL)
	{
		m_module->setTexRef("t_nodesA", *nodeBuffer, CU_AD_FORMAT_FLOAT, 4);
	}
	m_module->setTexRef("t_trisA", m_trisCompact, CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_triIndices", m_trisIndex, CU_AD_FORMAT_SIGNED_INT32, 1);

/*#ifdef COMPACT_LAYOUT
	if(numRays == 0)
	{
		m_module->setTexRef("t_trisAOut", m_trisCompactOut, CU_AD_FORMAT_FLOAT, 4);
		m_module->setTexRef("t_triIndicesOut", m_trisIndexOut, CU_AD_FORMAT_SIGNED_INT32, 1);
	}
#endif*/
}

//------------------------------------------------------------------------

void CudaPersistentBVHTracer::deinitPool(int numRays)
{
	m_ppsTris.reset();
	m_ppsTrisIndex.reset();
	m_sortTris.reset();

	if(numRays > 0)
	{
		m_ppsRays.reset();
		m_ppsRaysIndex.reset();
		m_sortRays.reset();
	}
}

//------------------------------------------------------------------------

/*void CudaPersistentBVHTracer::printPoolHeader(TaskStackBase* tasks, int* header, int numWarps, FW::String state)
{
#if PARALLELISM_TEST >= 0
	numActive = *(int*)m_module->getGlobal("g_numActive").getPtr();
	printf("Active: %d\n", numActive);
#endif


#if defined(SNAPSHOT_POOL) || defined(SNAPSHOT_WARP)
	printSnapshots(snapData);
#endif

#ifdef DEBUG_INFO
	Debug << "\nPRINTING DEBUG_INFO STATISTICS" << "\n\n";
#else
	Debug << "\nPRINTING STATISTICS" << "\n\n";
#endif

	float4* debugData = (float4*)m_debug.getPtr();
	float minAll[4] = {MAX_FLOAT, MAX_FLOAT, MAX_FLOAT, MAX_FLOAT};
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
}*/

//------------------------------------------------------------------------

/*void CudaPersistentBVHTracer::printPool(TaskStackBVH &tasks, int numWarps)
{
#ifdef LEAF_HISTOGRAM
	printf("Leaf histogram\n");
	unsigned int leafSum = 0;
	unsigned int triSum = 0;
	for(S32 i = 0; i <= Environment::GetSingleton()->GetInt("SubdivisionRayCaster.triLimit"); i++)
	{
		printf("%d: %d\n", i, tasks.leafHist[i]);
		leafSum += tasks.leafHist[i];
		triSum += i*tasks.leafHist[i];
	}
	printf("Leafs total %d, average leaf %.2f\n", leafSum, (float)triSum/(float)leafSum);
#endif

	int* header = (int*)m_taskData.getPtr();
	FW::String state = sprintf("BVH Top = %d; Tri Top = %d; Warp counter = %d; ", tasks.nodeTop, tasks.triTop, tasks.warpCounter);
#ifdef BVH_COUNT_NODES
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
			//Debug << "BoxLeft: (" << task[i].bboxLeft.m_mn.x << ", " << task[i].bboxLeft.m_mn.y << ", " << task[i].bboxLeft.m_mn.z << ") - ("
			//	<< task[i].bboxLeft.m_mx.x << ", " << task[i].bboxLeft.m_mx.y << ", " << task[i].bboxLeft.m_mx.z << ")\n";
			//Debug << "BoxRight: (" << task[i].bboxRight.m_mn.x << ", " << task[i].bboxRight.m_mn.y << ", " << task[i].bboxRight.m_mn.z << ") - ("
			//	<< task[i].bboxRight.m_mx.x << ", " << task[i].bboxRight.m_mx.y << ", " << task[i].bboxRight.m_mx.z << ")\n";
			Debug << "Axis: " << task[i].axis << "\n";
			Debug << "Depth: " << task[i].depth << "\n";
			Debug << "Step: " << task[i].step << "\n";
#ifdef DEBUG_INFO
			//Debug << "Step: " << task[i].step << "\n";
			//Debug << "Lock: " << task[i].lock << "\n";
#ifdef MALLOC_SCRATCHPAD
			Debug << "SubFailure: " << task[i].subFailureCounter << "\n";
#endif
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
#ifdef CUTOFF_DEPTH
				if(task[i].depth == m_cutOffDepth)
				{
#endif
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
#ifdef CUTOFF_DEPTH
				}
#endif

#ifdef DEBUG_INFO
				maxDepth = max(task[i].depth, maxDepth);
				syncCount += task[i].sync;
#endif
			}
		}
	}

	if(stackMax == TASK_SIZE-1)
		printf("\aIncomplete result!\n");
#ifdef CUTOFF_DEPTH
	Debug << "\n\nStatistics for cutoff depth " << m_cutOffDepth << "\n\n";
#else
	Debug << "\n\n";
#endif

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

void CudaPersistentBVHTracer::printPool(TaskStack &tasks, int numWarps)
{
	tasks = *(TaskStack*)m_module->getGlobal("g_taskStack").getPtr();
	int* header = (int*)m_taskData.getPtr();
	printPoolHeader(&tasks, header, numWarps, FW::sprintf(""));

	Debug << "\n\nTasks" << "\n";
	Task* task = (Task*)m_taskData.getPtr(TASK_SIZE*sizeof(int));
	int stackMax = 0;
	int maxDepth = 0;
	int syncCount = 0;
	int maxTaskId = -1;
	int rayIssues = 0;
	int triIssues = 0;
	long double sumRays = 0;
	long double maxRays = 0;
	long double sumTris = 0;
	long double maxTris = 0;
	
	int isectTasks = 0;
	long double cntIsect = 0;
	long double maxIsect = 0;
	long double clippedIsect = 0;

	int sortTasks = 0;
	long double cntSortRays = 0;
	long double cntClippedRays = 0;
	long double cntSortTris = 0;

	int subFailed = 0;
	int failureCount = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.failureCount");

#ifdef DEBUG_INFO
	char terminatedNames[TerminatedBy_Max][255] = {
		"None", "Depth","TotalLimit","OverheadLimit","Cost","FailureCounter"
	};

	int terminatedBy[TerminatedBy_Max];
	memset(&terminatedBy,0,sizeof(int)*TerminatedBy_Max);
#endif

	for(int i = 0; i < TASK_SIZE; i++)
	{
		if(task[i].depend1 != TaskHeader_Empty || task[i].depend2 != TaskHeader_Empty)
		{
#ifdef DEBUG_INFO
			_ASSERT(task[i].terminatedBy >= 0 && task[i].terminatedBy < TerminatedBy_Max);
			terminatedBy[ task[i].terminatedBy ]++;
#endif

			Debug << "Task " << i << "\n";
			Debug << "Header: " << header[i] << "\n";
			Debug << "Unfinished: " << task[i].unfinished << "\n";
			Debug << "Type: " << task[i].type << "\n";
			Debug << "RayStart: " << task[i].rayStart << "\n";
			Debug << "RayEnd: " << task[i].rayEnd << "\n";
			if(task[i].type != TaskType_Intersect) // Splitted
			{
				Debug << "RayLeft: " << task[i].rayLeft << "\n";
				Debug << "RayRight: " << task[i].rayRight << "\n";
				Debug << "RayActive: " << task[i].rayActive << "\n";
			}
#ifdef CLIP_INTERSECT
			if(task[i].type == TaskType_Intersect)
			Debug << "RayActive: " << task[i].rayActive << "\n";
#endif
			Debug << "TriStart: " << task[i].triStart << "\n";
			Debug << "TriEnd: " << task[i].triEnd << "\n";
			if(task[i].type != TaskType_Intersect) // Splitted
			{
				//Debug << "BestOrder: " << task[i].bestOrder << "\n";
				Debug << "TriLeft: " << task[i].triLeft << "\n";
				Debug << "TriRight: " << task[i].triRight << "\n";
			}
			Debug << "Depend1: " << task[i].depend1 << "\n";
			Debug << "Depend2: " << task[i].depend2 << "\n";
			if(task[i].type != TaskType_Intersect) // Splitted
			{
				Debug << "Split: (" << task[i].splitPlane.x << ", " << task[i].splitPlane.y << ", " << task[i].splitPlane.z << ", " << task[i].splitPlane.w << ")\n";
			}
			Debug << "Box: (" << task[i].bbox.m_mn.x << ", " << task[i].bbox.m_mn.y << ", " << task[i].bbox.m_mn.z << ") - ("
				<< task[i].bbox.m_mx.x << ", " << task[i].bbox.m_mx.y << ", " << task[i].bbox.m_mx.z << ")\n";
			//Debug << "BoxLeft: (" << task[i].bboxLeft.m_mn.x << ", " << task[i].bboxLeft.m_mn.y << ", " << task[i].bboxLeft.m_mn.z << ") - ("
			//	<< task[i].bboxLeft.m_mx.x << ", " << task[i].bboxLeft.m_mx.y << ", " << task[i].bboxLeft.m_mx.z << ")\n";
			//Debug << "BoxMiddle (" << task[i].bboxMiddle.m_mn.x << ", " << task[i].bboxMiddle.m_mn.y << ", " << task[i].bboxMiddle.m_mn.z << ") - ("
			//	<< task[i].bboxMiddle.m_mx.x << ", " << task[i].bboxMiddle.m_mx.y << ", " << task[i].bboxMiddle.m_mx.z << ")\n";
			//Debug << "BoxRight: (" << task[i].bboxRight.m_mn.x << ", " << task[i].bboxRight.m_mn.y << ", " << task[i].bboxRight.m_mn.z << ") - ("
			//	<< task[i].bboxRight.m_mx.x << ", " << task[i].bboxRight.m_mx.y << ", " << task[i].bboxRight.m_mx.z << ")\n";
			Debug << "Depth: " << task[i].depth << "\n";
#ifdef DEBUG_INFO
			//Debug << "Step: " << task[i].step << "\n";
			//Debug << "Lock: " << task[i].lock << "\n";
			Debug << "SubFailure: " << task[i].subFailureCounter << "\n";
			Debug << "GMEMSync: " << task[i].sync << "\n";
			Debug << "TaskID: " << task[i].taskID << "\n";
			Debug << "Parent: " << task[i].parent << "\n";
#if AABB_TYPE < 3
			if(task[i].type == TaskType_AABB_Max)
#elif AABB_TYPE == 3
			if(task[i].type == TaskType_AABB)
#endif
			{
				Debug << "SubtaskIdx: " << task[i].subtaskIdx << "\n";
				Debug << "Clipped rays: " << task[i].rayEnd-task[i].rayActive << "\n";
			}
#endif

#ifdef CUTOFF_DEPTH
			if(task[i].depth == m_cutOffDepth)
#endif
			if(task[i].type == TaskType_Intersect)
			{
#ifdef CLIP_INTERSECT
				long double locRays = task[i].rayActive - task[i].rayStart;
#else
				long double locRays = task[i].rayEnd - task[i].rayStart;
#endif
				long double locTris = task[i].triEnd - task[i].triStart; 
				Debug << "Intersections: " << locRays * locTris << "\n";
				//if(locRays > 1000 || locTris > 1000 )
				{
					if( locRays < sqrt((double)locTris) )
						triIssues++;
					if( locTris < sqrt((double)locRays) )
						rayIssues++;
				}

				Debug << "ClippedIntersections: " << task[i].clippedRays * locTris << "\n";
				clippedIsect += task[i].clippedRays * locTris;
			}

#ifdef ONE_WARP_RUN
			//Debug << "Clock: " << task[i].clockEnd - task[i].clockStart << "\n";
			Debug << "Clock: " << task[i].clockEnd << "\n";
#endif
#ifdef DEBUG_INFO
			Debug << "TerminatedBy: " << task[i].terminatedBy << "\n";
#endif
			
			Debug << "\n";
			stackMax = i;

#ifdef CUTOFF_DEPTH
			if(task[i].depth == m_cutOffDepth)
			{
#endif

#ifdef CLIP_INTERSECT
			long double rays = task[i].rayActive - task[i].rayStart;
#else
			long double rays = task[i].rayEnd - task[i].rayStart;
#endif
			
			long double tris = task[i].triEnd - task[i].triStart;
			if(task[i].type == TaskType_Intersect)
			{
				isectTasks++;
				cntIsect += rays*tris;
				maxIsect = max<long double>(rays*tris, maxIsect);
				if(maxIsect==(rays*tris)) maxTaskId = i;
				sumRays += rays;
				maxRays = max<long double>(rays, maxRays);
				sumTris += tris;
				maxTris = max<long double>(tris, maxTris);
				if(task[i].subFailureCounter > failureCount)
					subFailed++;
			}
#if AABB_TYPE < 3
			if(task[i].type == TaskType_AABB_Max)
#elif AABB_TYPE == 3
			if(task[i].type == TaskType_AABB)
#endif
			{
				sortTasks++;
				cntSortRays += rays;
				cntClippedRays += task[i].rayEnd-task[i].rayActive;
				cntSortTris += tris;
			}
#ifdef CUTOFF_DEPTH
			}
#endif

#ifdef DEBUG_INFO
			maxDepth = max(task[i].depth, maxDepth);
			syncCount += task[i].sync;
#endif
		}
	}

	if(stackMax == TASK_SIZE-1)
		printf("\aIncomplete result!\n");
#ifdef CUTOFF_DEPTH
	Debug << "\n\nStatistics for cutoff depth " << m_cutOffDepth << "\n\n";
#else
	Debug << "\n\n";
#endif

#ifdef DEBUG_INFO
	Debug << "ray_obj_intersections per ray = " << cntIsect/m_numRays << "\n";
	Debug << "cnt_leaves = " << isectTasks << "\n";
	Debug << "cnt_leaves per obj = " << (float)isectTasks/(float)m_numTris << "\n";
	Debug << "ray_obj_intersections = " << cntIsect << "\n";
	Debug << "Useless ray_obj_intersections = " << clippedIsect << "\n";
	Debug << "Avg ray_obj_intersections per leaf = " << cntIsect/(long double)isectTasks << "\n";
	Debug << "Max ray_obj_intersections per leaf = " << maxIsect << ", taskId: " << maxTaskId << "\n";
	Debug << "reduction [%] = " << 100.0f * (cntIsect/((long double)m_numRays*(long double)m_numTris)) << "\n";
	Debug << "Avg naive task width (rays) = " << sumRays/(long double)isectTasks << "\n";
	Debug << "Max naive task width (rays) = " << maxRays << "\n";
	Debug << "Avg naive task height (tris) = " << sumTris/(long double)isectTasks << "\n";
	Debug << "Max naive task height (tris) = " << maxTris << "\n";
	Debug << "Cnt sorted operations = " << sortTasks << "\n";
	double cntTrisLog2Tris = (double(m_numTris) * (double)(logf(m_numTris)/logf(2.0f)));
	double cntRaysLog2Tris = (double(m_numRays) * (double)(logf(m_numTris)/logf(2.0f)));
	Debug << "Cnt sorted triangles = " << cntSortTris << "\n";	
	Debug << "Cnt sorted triangles/(N log N), N=#tris = " << cntSortTris/cntTrisLog2Tris << "\n";
	Debug << "Cnt sorted rays = " << cntSortRays << " BEFORE CLIPPING\n";
	Debug << "Cnt sorted rays/(log N)/R, N=#tris,R=#rays = " << cntSortRays/cntRaysLog2Tris << " BEFORE CLIPPING\n";
	Debug << "Cnt clipped rays = " << cntClippedRays << "\n";
	Debug << "\n";
	Debug << "Max task depth = " << maxDepth << "\n";
	Debug << "Cnt gmem synchronizations: " << syncCount << "\n";
	Debug << "Ray issues = " << rayIssues << ", tris issues = " << triIssues << "\n";
	Debug << "Leafs failed to subdivide = " << subFailed << " (*3) => total useless tasks " << subFailed * 3 << "\n";

	Debug << "Terminated by:" << "\n";
	for(int i = 0; i < TerminatedBy_Max; i++)
	{
		Debug << terminatedNames[i] << ": " << terminatedBy[i] << "\n";
	}
#endif

	Debug << "max_queue_length = " << stackMax << "\n\n" << "\n";
}*/

//------------------------------------------------------------------------

F32 CudaPersistentBVHTracer::buildCudaBVH()
{
	CudaKernel kernel;
	kernel = m_module->getKernel("build");
	if (!kernel.getHandle())
		fail("Build kernel not found!");

#ifdef MALLOC_SCRATCHPAD
	KernelInputBVH& in = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
	in.numTris	    = m_numTris;
	in.tris         = m_trisCompact.getCudaPtr();
	in.trisIndex    = m_trisIndex.getMutableCudaPtr();
#ifdef COMPACT_LAYOUT
	in.trisOut      = m_trisCompactOut.getMutableCudaPtr();
	in.trisIndexOut = m_trisIndexOut.getMutableCudaPtr();
#endif
#endif

	// Prepare the task data
	initPool();

#ifndef MALLOC_SCRATCHPAD
	// Set input.
	KernelInputBVH& in = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
	in.numTris		= m_numTris;
	in.tris         = m_trisCompact.getCudaPtr();
	in.trisIndex    = m_trisIndex.getMutableCudaPtr();
	//in.trisBox      = m_trisBox.getCudaPtr();
	in.ppsTrisBuf   = m_ppsTris.getMutableCudaPtr();
	in.ppsTrisIndex = m_ppsTrisIndex.getMutableCudaPtr();
	in.sortTris     = m_sortTris.getMutableCudaPtr();
#ifdef COMPACT_LAYOUT
	in.trisOut      = m_trisCompactOut.getMutableCudaPtr();
	in.trisIndexOut = m_trisIndexOut.getMutableCudaPtr();
#endif
#else
	CUfunction kernelAlloc = m_module->getKernel("allocFreeableMemory", 2*sizeof(int));
	if (!kernelAlloc)
		fail("Memory allocation kernel not found!");

	int offset = 0;
	offset += m_module->setParami(kernelAlloc, offset, m_numTris);
	offset += m_module->setParami(kernelAlloc, offset, 0);
	F32 allocTime = m_module->launchKernelTimed(kernelAlloc, Vec2i(1,1), Vec2i(1, 1));

#ifndef BENCHMARK
	printf("Memory allocated in %f\n", allocTime);
#endif

	CUfunction kernelMemCpyIndex = m_module->getKernel("MemCpyIndex", sizeof(CUdeviceptr)+sizeof(int));
	if (!kernelMemCpyIndex)
		fail("Memory copy kernel not found!");

	int memSize = m_trisIndex.getSize()/sizeof(int);
	offset = 0;
	offset += m_module->setParamPtr(kernelMemCpyIndex, offset, m_trisIndex.getCudaPtr());
	offset += m_module->setParami(kernelMemCpyIndex, offset, memSize);
	F32 memcpyTime = m_module->launchKernelTimed(kernelMemCpyIndex, Vec2i(256,1), Vec2i((memSize-1+256)/256, 1));

#ifndef BENCHMARK
	printf("Triangle indices copied in %f\n", memcpyTime);
#endif
	in = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
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
#ifndef MALLOC_SCRATCHPAD
	all.triRight     = m_numTris;
#else
	all.triRight     = 0;
#endif
	all.triEnd       = m_numTris;
	all.bbox         = bbox;
	all.step         = 0;
	all.lock         = LockType_Free;
	all.bestCost     = 1e38f;
	all.depth        = 0;
	all.dynamicMemory= 0;
#ifndef MALLOC_SCRATCHPAD
	all.triIdxCtr    = 0;
#endif
	all.parentIdx    = -1;
	all.nodeIdx      = 0;
	all.taskID       = 0;
	Vec3f size     = m_bboxMax - m_bboxMin;
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
	tasks.numEmptyLeaves = 0;
	tasks.sizePool = TASK_SIZE;
	tasks.sizeNodes = m_bvhData.getSize()/sizeof(CudaKdtreeNode);
	tasks.sizeTris = m_trisIndexOut.getSize()/sizeof(S32);
	memset(tasks.leafHist, 0, sizeof(tasks.leafHist));

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
	// Prepare split stack
	SplitArray* &splits = *(SplitArray**)m_module->getGlobal("g_redSplits").getMutablePtr();
	splits = (SplitArray*)m_splitData.getMutableCudaPtr();
#endif

	CudaBVHNode* &bvh = *(CudaBVHNode**)m_module->getGlobal("g_bvh").getMutablePtr();
	bvh = (CudaBVHNode*)m_bvhData.getMutableCudaPtr();

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
	float tKernel = kernel.launchTimed(blockSize, gridSize);

/*#ifdef MALLOC_SCRATCHPAD
	CUfunction kernelDealloc = m_module->getKernel("deallocFreeableMemory", 0);
	if (!kernelDealloc)
		fail("Memory allocation kernel not found!");

	F32 deallocTime = m_module->launchKernelTimed(kernelDealloc, Vec2i(1,1), Vec2i(1, 1));

	printf("Memory freed in %f\n", deallocTime);
#endif*/

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
	if(tasks.unfinished != 0 || tasks.top >= tasks.sizePool || tasks.nodeTop >= m_bvhData.getSize() / sizeof(CudaBVHNode) || tasks.triTop >= m_trisIndexOut.getSize() / sizeof(S32)) // Something went fishy
	{
		tKernel = 1e38f;
		fail("%d (%d x %d) (%d x %d) (%d x %d)\n", tasks.unfinished != 0, tasks.top, tasks.sizePool, tasks.nodeTop, m_bvhData.getSize() / sizeof(CudaBVHNode), tasks.triTop, m_trisIndexOut.getSize() / sizeof(S32));
	}

	//Debug << "\nBuild in " << tKernel << "\n\n";

#ifndef BENCHMARK
	printPool(tasks, numWarps);

	/*Debug << "\n\nBVH" << "\n";
	CudaBVHNode* nodes = (CudaBVHNode*)m_bvhData.getPtr();

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

/*void CudaPersistentBVHTracer::saveBufferSizes(bool ads, bool aux)
{
	float MB = (float)(1024*1024);

	if(ads)
	{
		m_sizeADS    = m_bvhData.getSize()/MB;
#ifndef COMPACT_LAYOUT
		m_sizeTri    = m_trisCompact.getSize()/MB;
		m_sizeTriIdx = m_trisIndex.getSize()/MB;
#else
		m_sizeTri    = m_trisCompactOut.getSize()/MB;
		m_sizeTriIdx = m_trisIndexOut.getSize()/MB;
#endif
	}

	if(aux)
	{
		m_sizeTask   = m_taskData.getSize()/MB;
		m_sizeSplit  = m_splitData.getSize()/MB;
#ifdef MALLOC_SCRATCHPAD
#if (MALLOC_TYPE == CUDA_MALLOC) || (MALLOC_TYPE == FDG_MALLOC)
		size_t heapSize;
		cuCtxGetLimit(&heapSize, CU_LIMIT_MALLOC_HEAP_SIZE);
		m_heap       = heapSize/MB; 
#else
		m_heap       = (m_mallocData.getSize()+m_mallocData2.getSize())/MB;
#endif
#else
		m_heap       = 0.f;
#endif
	}
}*/

void CudaPersistentBVHTracer::resetBuffers(bool resetADSBuffers)
{
	// Reset buffers so that reuse of space does not cause timing disturbs
	if(resetADSBuffers)
	{
		m_bvhData.reset();
		m_trisCompactOut.reset();
		m_trisIndexOut.reset();
	}

	//m_mallocData.reset();
	//m_mallocData2.reset();
	m_taskData.reset();
	m_splitData.reset();

	//m_raysIndex.reset();

	m_ppsTris.reset();
	m_ppsTrisIndex.reset();
	m_sortTris.reset();
	m_ppsRays.reset();
	m_ppsRaysIndex.reset();
	m_sortRays.reset();
}

void CudaPersistentBVHTracer::trimBVHBuffers()
{
	// Save sizes of auxiliary buffers so that they can be printed
	//saveBufferSizes(false, true);
	// Free auxiliary buffers
	resetBuffers(false);

	// Resize to exact memory
	U32 nN, nL, eL, sT, bT, tT, sTr; 
	getStats(nN, nL, eL, sT, bT, tT, sTr);
#ifdef COMPACT_LAYOUT
	m_bvhData.resize(nN * sizeof(CudaBVHNode));
	m_trisCompactOut.resize(tT*3*sizeof(float4) + nL*sizeof(float4));
	m_trisIndexOut.resize(tT*3*sizeof(int) + nL*sizeof(int));
#else
	m_bvhData.resize((nN + nL) * sizeof(CudaBVHNode));
#endif

	// Save sizes of ads buffers so that they can be printed
	//saveBufferSizes(true, false);
}

void CudaPersistentBVHTracer::getStats(U32& nodes, U32& leaves, U32& emptyLeaves, U32& stackTop, U32& nodeTop, U32& tris, U32& sortedTris, bool sub)
{
	TaskStackBVH tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();

#ifndef INTERLEAVED_LAYOUT
#ifndef BVH_COUNT_NODES
#ifndef COMPACT_LAYOUT
	nodes = tasks.nodeTop / 2;
	leaves = tasks.nodeTop - nodes;
#else
	nodes = tasks.nodeTop;
	leaves = tasks.triTop;
	emptyLeaves = 0;
#endif
#else // BVH_COUNT_NODES
	nodes = tasks.numNodes;
	leaves = tasks.numLeaves;
	emptyLeaves = tasks.numEmptyLeaves;
#endif // BVH_COUNT_NODES

#ifdef COMPACT_LAYOUT
	tris = tasks.triTop;
	if(sub)
		tris -= (leaves-emptyLeaves);
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
#else
#ifndef BVH_COUNT_NODES
	nodes = tasks.nodeTop / 2;
	leaves = tasks.nodeTop - nodes;
	emptyLeaves = 0;
#else // BVH_COUNT_NODES
	nodes = tasks.numNodes;
	leaves = tasks.numLeaves;
	emptyLeaves = tasks.numEmptyLeaves;
#endif // BVH_COUNT_NODES

	tris = tasks.nodeTop - (nodes+leaves)*sizeof(CudaKdtreeNode); // Subtract node memory
	tris /= 3*sizeof(float4)+sizeof(int); // Only approximate because of padding
#endif

	nodeTop = tasks.nodeTop;
	sortedTris = tasks.numSortedTris;
	stackTop = tasks.top;
}

void CudaPersistentBVHTracer::getSizes(F32& task, F32& split, F32& ads, F32& tri, F32& triIdx, F32& heap)
{
	task = m_sizeTask;
	split = m_sizeSplit;
	ads = m_sizeADS;
	tri = m_sizeTri;
	triIdx = m_sizeTriIdx;
	heap = m_heap;
}
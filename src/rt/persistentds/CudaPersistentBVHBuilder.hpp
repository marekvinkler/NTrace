#pragma once

#include "gpu/Buffer.hpp"
#include "ray/RayBuffer.hpp"
#include "bvh/BVH.hpp"
#include "Util.hpp"
#include "3d/Mesh.hpp"
#include "gpu/CudaCompiler.hpp"
#include "base/Timer.hpp"
#include "cuda/CudaBVH.hpp"

#include "Scene.hpp"

namespace FW
{

class CudaPersistentBVHTracer : public CudaBVH
{
	// Scene data
	F32 m_epsilon;
	S32 m_numTris;
	Vec3f m_bboxMin, m_bboxMax;
	S32 m_buildNodes;
	S32 m_buildLeafs;

	// Input buffers (from the scene)
	Buffer m_trisCompactIndex;					
	Buffer& m_trisCompact;
	Buffer m_trisBox;

	// GPU task data
	Buffer   m_taskData;
	Buffer   m_splitData;
	int      m_cutOffDepth;
	int      m_numRays;

	CudaCompiler m_compiler;
	CudaModule*  m_module;

	// Auxiliary buffers
	Buffer m_ppsRays;
	Buffer m_ppsTris;
	Buffer m_ppsRaysIndex;
	Buffer m_ppsTrisIndex;
	Buffer m_sortRays;
	Buffer m_sortTris;

	// Debug buffers
	Buffer m_debug;

	// Statistics
	Timer  m_timer;
	F32    m_cpuTime;
	F32    m_gpuTime;

	F32    m_sizeTask;
	F32    m_sizeSplit;
	F32    m_sizeADS;
	F32    m_sizeTri;
	F32    m_sizeTriIdx;
	F32    m_heap;
	String m_kernelFile;

	void updateConstants();
	int warpSubtasks(int threads);
	int floatToOrderedInt(float floatVal);
	void initPool(int numRays = 0, Buffer* rayBuffer = NULL, Buffer* nodeBuffer = NULL);
	void deinitPool(int numRays = 0);
	//void printPoolHeader(TaskStackBase* tasks, int* header, int numWarps, FW::String state);
	//void printPool(TaskStackBVH& tasks, int numWarps);
	//void printPool(TaskStack& tasks, int numWarps);

	F32 buildCudaBVH();
public:
	CudaPersistentBVHTracer(Scene& scene, F32 epsilon);

	F32 buildBVH();
	F32 traceBatchBVH(RayBuffer& rays, RayStats* stats = NULL);

	void resetBuffers(bool resetADSBuffers); // Resets all buffers for timing purposes
	void trimBVHBuffers(); // Sets all buffer sizes to their used extent

	F32 getCPUTime() { return m_cpuTime; }
	F32 getGPUTime() { return m_gpuTime; }
	void getStats(U32& nodes, U32& leaves, U32& emptyLeaves, U32& stackTop, U32& nodeTop, U32& tris, U32& sortedTris, bool sub = true);
	void getSizes(F32& task, F32& split, F32& ads, F32& tri, F32& triIdx, F32& heap);
};

}
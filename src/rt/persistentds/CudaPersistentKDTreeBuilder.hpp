#pragma once

#include "gpu/Buffer.hpp"
#include "gpu/CudaCompiler.hpp"
#include "base/Timer.hpp"
#include "cuda/CudaKDTree.hpp"
#include "persistentds/CudaPoolKDTree.hpp"
#include "Scene.hpp"

#include <fstream>

namespace FW
{

class CudaPersistentKDTreeBuilder : public CudaKDTree
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
	Buffer m_taskData;
	Buffer m_splitData;

	CudaCompiler m_compiler;
	CudaModule*  m_module;

	// Debug buffers
	Buffer m_debug;
	std::ofstream Debug;
	Buffer m_snapData;

	// Allocator data
	Buffer m_mallocData;
	Buffer m_mallocData2;
	Buffer m_multiOffset;
	Buffer m_interFragSum;
	char* halloc_base;

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
	F32    m_fragIntern;
	F32    m_fragExtern;

	void updateConstants();
	void initPool(Buffer* nodeBuffer = NULL);
	void deinitPool();
	void printPoolHeader(TaskStackBase* tasks, int* header, int numWarps, FW::String state);
	void printPool(TaskStackKdtree& tasks, int numWarps);

	void saveBufferSizes(bool ads = true, bool aux = true);

	F32 buildCuda();
	void prepareDynamicMemory();
	void setCircularMallocHeader(bool set, U32 ofs, U32 prevOfs, U32 nextOfs);
	int setDynamicMemory();
	void computeFragmentation();
	F32 convertWoop(); // Convert regular triangles to Woop triangles

public:
	CudaPersistentKDTreeBuilder(Scene& scene, F32 epsilon);
	~CudaPersistentKDTreeBuilder();

	F32 build();

	void resetBuffers(bool resetADSBuffers); // Resets all buffers for timing purposes
	void trimBuffers(); // Sets all buffer sizes to their used extent

	F32 getCPUTime() { return m_cpuTime; }
	F32 getGPUTime() { return m_gpuTime; }
	void getStats(U32& nodes, U32& leaves, U32& emptyLeaves, U32& stackTop, U32& nodeTop, U32& tris, U32& sortedTris, bool sub = true);
	void getSizes(F32& task, F32& split, F32& ads, F32& tri, F32& triIdx, F32& heap);
	void getAllocStats(U32& numAllocs, F32& allocSum, F32& allocSumSquare);
	void getFragmentationStats(F32& fInt, F32& fExt);
};

}
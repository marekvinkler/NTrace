#pragma once

#include "gpu/Buffer.hpp"
#include "ray/RayBuffer.hpp"
#include "bvh/BVH.hpp"
#include "Util.hpp"
#include "3d/Mesh.hpp"
#include "gpu/CudaCompiler.hpp"
#include "base/Timer.hpp"
/*#include "materials/MaterialCommon.h"
#include "textures/TexArray.h"
#include "kernels/CudaPool.hpp"
#include "kernels/CudaNoStructKernels.hpp"
#include "kernels/CudaBuilderKernels.hpp"*/

#include "Scene.hpp"

namespace FW
{

class CudaPersistentBVHTracer
{
	// scene data
	F32 m_epsilon;
	S32 m_numVerts;
	S32 m_numTris;
	S32 m_numMaterials;
	S32 m_numShadingNormals;
	S32 m_numTextureCoords;
	S32 m_numLights;
	Vec3f m_bboxMin, m_bboxMax;
	S32 m_buildNodes;
	S32 m_buildLeafs;

	Buffer m_verts;
	Buffer m_tris;
	Buffer m_triNormals;
	Buffer m_materials;
	Buffer m_shadingNormals;
	Buffer m_shadedColor;
	Buffer m_materialColor;

	Buffer m_raysIndex;
	Buffer m_trisIndex;
	Buffer m_trisCompact;
	Buffer m_trisBox;

	Buffer m_trisCompactOut; // Used for COMPACT_LAYOUT
	Buffer m_trisIndexOut; // Used for COMPACT_LAYOUT

	Buffer m_lights;

	/*TexArray m_texArray;
	Buffer   m_textureCoords;*/

	// GPU task data
	Buffer   m_mallocData;
	Buffer   m_mallocData2;
	Buffer   m_taskData;
	Buffer   m_splitData;
	Buffer   m_bvhData;
	int      m_cutOffDepth;
	int      m_numRays;

	CudaCompiler m_compiler;
	CudaModule*  m_module;

	// buffers
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
	//unsigned int floatToOrderedInt(float floatVal);
	void allocateSnapshots(Buffer &snapData);
	void printSnapshots(Buffer &snapData);
	void initPool(int numRays = 0, Buffer* rayBuffer = NULL, Buffer* nodeBuffer = NULL);
	void deinitPool(int numRays = 0);
	/*void printPoolHeader(TaskStackBase* tasks, int* header, int numWarps, FW::String state);
	void printPool(TaskStackBVH& tasks, int numWarps);
	void printPool(TaskStack& tasks, int numWarps);*/
	void traceCpuRay(const Ray& r, RayResult& result, bool anyHit);

	F32 traceCudaRayBuffer(RayBuffer& rb);
	F32 traceOnDemandBVHRayBuffer(RayBuffer& rb, bool rebuild);
	F32 traceOnDemandKdtreeRayBuffer(RayBuffer& rb, bool rebuild);
	F32 buildCudaBVH();
	F32 buildCudaKdtree();
	F32 testSort(S32 arraySize);
	F32 traceCpuRayBuffer(RayBuffer& rb);
	void saveBufferSizes(bool ads = true, bool aux = true);
	void prepareDynamicMemory();
	int setDynamicMemory();
public:
	CudaPersistentBVHTracer(const Scene& scene, F32 epsilon);

	F32 traceBatch(RayBuffer& rays);
	F32 buildBVH(bool sbvh);
	F32 buildKdtree();
	F32 traceBatchBVH(RayBuffer& rays, RayStats* stats = NULL);
	F32 traceBatchKdtree(RayBuffer& rays, RayStats* stats = NULL);
	F32 traceOnDemandBVH(RayBuffer& rays, bool rebuild, int numRays = 0);
	F32 traceOnDemandKdtree(RayBuffer& rays, bool rebuild, int numRays = 0);
	void traceOnDemandTrace(RayBuffer& rays, F32& GPUmegakernel, F32& CPUmegakernel, F32& GPUtravKernel, F32& CPUtravKernel, int& buildNodes, RayStats* stats = NULL);
	F32 test();

	F32 convertWoop(); // Convert regular triangles to Woop triangles
	void resetBuffers(bool resetADSBuffers); // Resets all buffers for timing purposes
	void trimBVHBuffers(); // Sets all buffer sizes to their used extent
	void trimKdtreeBuffers(); // Sets all buffer sizes to their used extent

	F32 getCPUTime() { return m_cpuTime; }
	F32 getGPUTime() { return m_gpuTime; }
	void getStats(U32& nodes, U32& leaves, U32& emptyLeaves, U32& stackTop, U32& nodeTop, U32& tris, U32& sortedTris, bool sub = true);
	void getSizes(F32& task, F32& split, F32& ads, F32& tri, F32& triIdx, F32& heap);

	/*S32 getNumLights() { return m_numLights; }

	Buffer& getTriangleBuffer() { return m_tris; }
	Buffer& getTriangleNormalBuffer() { return m_triNormals; }
	Buffer& getTriangleOutBuffer() { return m_trisCompactOut; }
	Buffer& getTriangleIndexOutBuffer() { return m_trisIndexOut; }
	Buffer& getNodeBuffer() { return m_bvhData; }
	Buffer& getMaterialsBuffer() { return m_materials; }
	Buffer& getLightBuffer() { return m_lights; }
	Buffer& getShadingNormalsBuffer() { return m_shadingNormals; }
	Buffer& getShadedColorBuffer() { return m_shadedColor; }
	Buffer& getMaterialColorBuffer() { return m_materialColor; }
	Buffer& getTextureCoordsBuffer() { return m_textureCoords; }*/
};

}
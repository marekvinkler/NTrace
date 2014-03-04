#pragma once
//#include "base/DLLImports.hpp"
//#include "base/Math.hpp"
//#include "Scene.hpp"
//#include "Util.hpp"
#include <cuda.h>
#include "../../../framework/base/Defs.hpp"

namespace FW {

#define BIN_CNT 8

//#define BLOCK_SIZE 192
#define BLOCK_SIZE 128

#define WARP_SIZE 32

//#define DOPRINTF
//#define MEASURE_STATS
//#define LEAF_HISTOGRAM
#define WOOP_TRIANGLES
#define COMPACT_LAYOUT // NOT WORKING WITHOUT THIS
					   // Would require generating one more level of nodes through the queue
#define CLUSTER_AABB 3 // 0 means thread
					   // 1 means warp
					   // 2 means block
					   // 3 means parallel

#define NUM_SM 8 // Number of SMs on the device
#define NUM_BLOCKS_PER_SM 12
#define NUM_BLOCKS (NUM_SM*NUM_BLOCKS_PER_SM) // Number of blocks on the device

#if FW_CUDA
extern "C"
{
__constant__ int                        c_leafSize;
__constant__ float                      c_epsilon;

#ifdef LEAF_HISTOGRAM
__device__ U32							g_leafHist[32];
#endif

__constant__ __device__ float3*			g_verts; // vertices
__constant__ __device__ CUdeviceptr		g_tris;  // struct Triangle

__device__ U32*							g_inTriMem;    // morton
__device__ U32*							g_inTriIdxMem; // index
__device__ int4*						g_inWoopMem;   // woop

__device__ S32							g_inQueuePtr;
__device__ S32*							g_inQueueMem;

__device__ U32							g_outQueuePtr;
__device__ S32*							g_outQueueMem;

__device__ CUdeviceptr					g_outNodes;
__device__ CUdeviceptr					g_outIdxMem;
//__device__ int3*						g_outIdxMem;
__device__ int4*						g_outWoopMem;

__device__ U64							g_leafsPtr;

__device__ S32*							g_rangeMem;

#ifdef MEASURE_STATS
__device__ S32 	g_ga,g_gb,g_gc,g_gd;
#endif

__global__ void emitTreeKernel(S32 level, U32 nodeCnt, S32 inOfs);
__global__ void calcWoopKernel(U32 triCnt);
__global__ void calcMorton(U32 triCnt,
							F32 lo_x,
							F32 lo_y,
							F32 lo_z,
							F32 step_x,
							F32 step_y,
							F32 step_z);
__global__ void calcAABB(S32 start, S32 cnt);

///////////// SAH

// bins
__device__ CUdeviceptr	g_binAABB;
__device__ S32*			g_binCnt;

// clusters
#if CLUSTER_AABB <= 2
__device__ float3*		g_clsAABB;
#else
__device__ int3*		g_clsAABB;
#endif
__device__ S32*			g_clsBinId;
__device__ S32*			g_clsSplitId;

// split tasks
__device__ CUdeviceptr	g_qsiAABB;
__device__ S32*			g_qsiCnt;
__device__ S32*			g_qsiId;
__device__ S32*			g_qsiPlane;
__device__ S32*			g_qsiChildId;

__device__ CUdeviceptr	g_qsoAABB;
__device__ S32*			g_qsoCnt;
__device__ S32*			g_qsoId;
__device__ S32*			g_qsoPlane;
__device__ S32*			g_qsoChildId;

__device__ U32			g_sahCreated;

__device__ U32			g_oofs;
__device__ S32*			g_ooq;
__device__ S32*			g_clsStart;

__global__ void initBins(U32 qsiCnt);
__global__ void fillBins(U32 clsCnt);
__global__ void findSplit(U32 qsiCnt, U32 inOfs);
__global__ void distribute(U32 clsCnt, S32 inOfs);

__device__ S32 g_cls_cnt;
__global__ void clusterCreate(S32 cnt, S32 shift);
__global__ void clusterAABB(S32 cnt, S32 tris);

__device__ F32 g_sahCost;
}
#endif

};
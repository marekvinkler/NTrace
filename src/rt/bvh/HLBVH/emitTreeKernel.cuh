#pragma once
//#include "base/DLLImports.hpp"
//#include "base/Math.hpp"
//#include "Scene.hpp"
//#include "Util.hpp"
#include <cuda.h>
//#include "../../../framework/base/Defs.hpp"

#define BIN_CNT 8

//#define BLOCK_SIZE 192
#define BLOCK_SIZE 128

#define WARP_SIZE 32

//#define DOPRINTF
//#define LOWMEM
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

#ifdef __CUDACC__
extern "C"
{
#define FW_F32_MAX          (3.402823466e+38f)

typedef unsigned int uint;
typedef unsigned long long int ullint;

__constant__ int                        c_leafSize;
__constant__ float                      c_epsilon;

#ifdef LEAF_HISTOGRAM
__device__ uint							g_leafHist[32];
#endif

__constant__ __device__ float3*			g_verts; // vertices
__constant__ __device__ CUdeviceptr		g_tris;  // struct Triangle

__device__ uint*							g_inTriMem;    // morton
__device__ uint*							g_inTriIdxMem; // index
__device__ int4*						g_inWoopMem;   // woop

__device__ int							g_inQueuePtr;
__device__ int*							g_inQueueMem;

__device__ uint							g_outQueuePtr;
__device__ int*							g_outQueueMem;

__device__ CUdeviceptr					g_outNodes;
__device__ CUdeviceptr					g_outIdxMem;
//__device__ int3*						g_outIdxMem;
__device__ int4*						g_outWoopMem;

__device__ ullint							g_leafsPtr;

__device__ int*							g_rangeMem;

#ifdef MEASURE_STATS
__device__ int 	g_ga,g_gb,g_gc,g_gd;
#endif

__global__ void emitTreeKernel(int level, uint nodeCnt, int inOfs);
__global__ void calcWoopKernel(uint triCnt);
__global__ void calcMorton(uint triCnt,
							float lo_x,
							float lo_y,
							float lo_z,
							float step_x,
							float step_y,
							float step_z);
__global__ void calcAABB(int start, int cnt);

///////////// SAH

// bins
__device__ CUdeviceptr	g_binAABB;
__device__ int*			g_binCnt;

// clusters
#if CLUSTER_AABB <= 2
__device__ float3*		g_clsAABB;
#else
__device__ int3*		g_clsAABB;
#endif
__device__ int*			g_clsBinId;
__device__ int*			g_clsSplitId;

// split tasks
__device__ CUdeviceptr	g_qsiAABB;
__device__ int*			g_qsiCnt;
__device__ int*			g_qsiId;
__device__ int*			g_qsiPlane;
__device__ int*			g_qsiChildId;

__device__ CUdeviceptr	g_qsoAABB;
__device__ int*			g_qsoCnt;
__device__ int*			g_qsoId;
__device__ int*			g_qsoPlane;
__device__ int*			g_qsoChildId;

__device__ uint			g_sahCreated;

__device__ uint			g_oofs;
__device__ int*			g_ooq;
__device__ int*			g_clsStart;

__global__ void initBins(uint qsiCnt);
__global__ void fillBins(uint clsCnt);
__global__ void findSplit(uint qsiCnt, uint inOfs);
__global__ void distribute(uint clsCnt, int inOfs);

__device__ int g_cls_cnt;
__global__ void clusterCreate(int cnt, int shift);
__global__ void clusterAABB(int cnt, int tris);

__device__ float g_sahCost;
}
#endif
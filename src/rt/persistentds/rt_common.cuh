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
    Common functionality for ray tracing based framework specializations.

    "Massively Parallel Hierarchical Scene Sorting with Applications in Rendering",
    Marek Vinkler, Michal Hapala, Jiri Bittner and Vlastimil Havran,
    Computer Graphics Forum 2012
*/

#pragma once
#ifdef __CUDACC__
#include "pool_common.cu"
#endif
#include "CudaTracerDefines.h"
#include "kernels/CudaTracerKernels.hpp"

//------------------------------------------------------------------------
// Common and debugging constants.
//------------------------------------------------------------------------


#define EPS 1e-8f
#define RAY_OFFSET 1e-3f
#define TRAVERSAL_STACK 128  // Size of the traversal stack in local memory.

// Macros for testing parts of the code. Do not give a definitive answer whether everything is OK
//#define RAYTRI_TEST
//#define SPLIT_TEST
//#define BBOX_TEST
//#define LUTINDEX_TEST
//#define UPDATEDEADLOCK_TEST
//#define INTERSECT_TEST
//#define PHASE_TEST


//------------------------------------------------------------------------
// Common types.
//------------------------------------------------------------------------

// Simple AABB Struct
struct CudaAABB
{
	float3 m_mn;
	float3 m_mx;

#ifdef __CUDACC__
	__device__ __forceinline__    void            grow        (const float3& pt)      { m_mn = fminf(m_mn, pt); m_mx = fmaxf(m_mx, pt); }
    __device__ __forceinline__    void            grow        (const CudaAABB& aabb)  { grow(aabb.m_mn); grow(aabb.m_mx); }
	__device__ __forceinline__    bool            isEmpty     ()                      { return (m_mx.x < m_mn.x || m_mx.y < m_mn.y || m_mx.z < m_mn.z); }
#endif
};

//------------------------------------------------------------------------

// A structure holding CudaAABB in int format capable of atomic min and max of floats
struct CudaAABBInt
{
	int3 m_mn;
	int3 m_mx;
};

//------------------------------------------------------------------------

struct __align__(32) Reference
{
	CudaAABB	bbox;
	int			idx;
	int			pdd;
};

//------------------------------------------------------------------------

struct CudaBVHNode
{
	float4 c0xy;
	float4 c1xy;
	float4 c01z;
	int4   children;
};

//------------------------------------------------------------------------

// Wald Kd-tree node
/*struct CudaKdtreeNode
{
	union
	{
		float spos; // position of the split
		int ntris; // number of triangles in the leaf
	}
	int flags;
	// Inner Node
	// bits 30..31 : flag whether node is a leaf / splitting dimension
	// bits 0..28 : offset bits
	// Leaf
	// bits 0..30 : offset to first son
};*/

typedef int4 CudaKdtreeNode;
// Kdtree
//x flag + childPtr or ntris
//y spos or triPtr
//z parentIdx
//w nodeIdx

// Ondemand or empty leafs
//x childLeft or ntris
//y childRight or triStart
//z spos
//w flag + nodeIdx

//------------------------------------------------------------------------

struct TraversalInfo
{
	// Traversal stack in CUDA thread-local memory.
    int traversalStack[TRAVERSAL_STACK];

    // Live state during traversal, stored in registers.
    int     rayIdx;                 // Ray index.
    float3  orig;					// Ray origin.
    float3  dir;					// Ray direction.
    float   tmin;                   // t-value from which the ray starts. Usually 0.
    float3  idir;                   // 1 / dir
    float3  ood;                    // orig / dir

    int*    stackPtr;               // Current position in traversal stack.
    //int     leafAddr;               // First postponed leaf, non-negative if none.
    int     nodeAddr;               // Non-negative: current internal node, negative: second postponed leaf.
    int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
    float   hitT;                   // t-value of the closest intersection.
	float   hitU;                   // u-barycentric of the closest intersection.
	float   hitV;                   // v-barycentric of the closest intersection.
};

//------------------------------------------------------------------------

enum TaskType
{
	TaskType_Split,
#if SPLIT_TYPE == 3
	TaskType_SplitParallel,
#endif
	TaskType_Sort_PPS1,
	TaskType_Sort_SORT1,
	TaskType_Sort_PPS2,
	TaskType_Sort_SORT2,
	TaskType_Sort_PPS1_Up,
	TaskType_Sort_PPS1_Down,
	TaskType_Sort_PPS2_Up,
	TaskType_Sort_PPS2_Down,
#if AABB_TYPE < 3
	TaskType_AABB_Min,        // Computes child bounding boxes min in x, y and z simultaneously
	TaskType_AABB_Max,        // Computes child bounding boxes max in x, y and z simultaneously
#else
	TaskType_AABB,
#endif
	TaskType_Intersect,
	TaskType_ClipPPS,         // Computes the scan on clipped rays
	TaskType_ClipSORT,        // Compacts the array of rays
	TaskType_InitMemory,      // Prepares the memory for the task
	TaskType_BinTriangles,    // Bins triangles to gmem accorting to WARP_split planes
	TaskType_BinSpatial,
	TaskType_ReduceBins,      // Reduces the counters in all bins
	TaskType_BuildObjectSAH,  // Builds the subtree with object SAH
	TaskType_RayTriTestPPS1,  // Test the output of PPS1 phase
	TaskType_RayTriTestSORT1, // Test whether rays are distributed correctly into -1+0,1+2 and triangles are distributed into -1+0,1
	TaskType_RayTriTestSORT2, // Test whether rays are distributed correctly into -1,0,1,2 and triangles are distributed into -1,0,1
	TaskType_Max
};

//------------------------------------------------------------------------

enum TerminatedBy
{
	TerminatedBy_None = 0,
	TerminatedBy_Depth,
	TerminatedBy_TotalLimit,
	TerminatedBy_OverheadLimit,
	TerminatedBy_Cost,
	TerminatedBy_FailureCounter,
	TerminatedBy_Max
};

//------------------------------------------------------------------------
// Constants.
//------------------------------------------------------------------------

// Constants
struct RtEnvironment
{
	float epsilon;
	float optPlaneSelectionOverhead;
	float optAxisAlignedWeight;
	float optTriangleBasedWeight;
	float optRayBasedWeight;
	float optCi;
	float optCt;
	float optCtr;
	float optCtt;
	int   optMaxDepth;
	int   optCutOffDepth;
	int   rayLimit;
	int   triLimit;
	int   triMaxLimit;
	int   popCount;
	float granularity;
	float failRq;
	int   failureCount;
	int   siblingLimit;
	int   childLimit;
	int   subtreeLimit;
	float subdivThreshold;
};

#ifdef __CUDACC__
__constant__ RtEnvironment c_env;


//------------------------------------------------------------------------
// Auxilliary control flow functions.
//------------------------------------------------------------------------

// Returns the right TaskType based on the SCAN_TYPE for PPS1
__device__ __forceinline__ TaskType taskChooseScanType(int unfinished);

// Returns the right TaskType based on the SCAN_TYPE for PPS1
__device__ __forceinline__ TaskType taskChooseScanType1();

// Returns the right TaskType based on the SCAN_TYPE for PPS2
__device__ __forceinline__ TaskType taskChooseScanType2();

// Returns the right TaskType based on the AABB_TYPE
__device__ __forceinline__ TaskType taskChooseAABBType();


//------------------------------------------------------------------------
// Data loading functions.
//------------------------------------------------------------------------


// Fetches ray from global memory
__device__ __forceinline__ void taskFetchRay(CUdeviceptr rays, int rayIdx, float3 &orig, float3 &dir, float &tmin, float &tmax);

// Fetches ray from global memory
__device__ __forceinline__ void taskFetchRayVolatile(CUdeviceptr rays, int rayIdx, float3 &orig, float3 &dir, float &tmin, float &tmax);

// Fetches triangle from global memory
__device__ __forceinline__ void taskFetchTri(CUdeviceptr tris, int triIdx, float3 &v0, float3 &v1, float3 &v2);

// Fetches node from global memory
__device__ __forceinline__ void taskFetchNodeAddr(CUdeviceptr nodes, int nodeIdx, CudaBVHNode &node);

// Fetches node from global memory
__device__ __forceinline__ void taskFetchNode(CUdeviceptr nodes, int nodeIdx, CudaBVHNode &node);

// Fetches node from global memory
__device__ __forceinline__ void taskFetchNodeVolatile(CUdeviceptr nodes, int nodeIdx, CudaBVHNode &node);


//------------------------------------------------------------------------
// Data saving functions.
//------------------------------------------------------------------------


// Copies node to the node array
__device__ __forceinline__ void taskSaveNodeToGMEM(CudaBVHNode* g_bvh, int tid, int nodeIdx, const volatile CudaBVHNode& node);

// Update the pointer in the parent to point to this node
__device__ void taskUpdateParentPtr(CudaBVHNode* g_bvh, int parentIdx, int taskID, int newValue);

// Update the pointer in the parent to point to this node
__device__ void taskUpdateParentPtr(CudaKdtreeNode* g_kdtree, int parentIdx, int taskID, int newValue);


//------------------------------------------------------------------------
// Geometric computations.
//------------------------------------------------------------------------

// Computes plane dimension of axis aligned planes
__device__ __forceinline__ int getPlaneDimension(const float4& plane);

// Computes distance of a point from a plane
__device__ __forceinline__ float planeDistance(const float3& normal, const float& d, const float3& p);

// Creates plane from three points
__device__ __forceinline__ float4 set3PointPlane(const float3& v0, const float3& v1, const float3& v2);

// Computes which side of the plane is a triangle on
__device__ __forceinline__ int getPlanePosition(const float4& plane, const float3& v0, const float3& v1, const float3& v2);

// Computes the bounding box of a triangle
__device__ __forceinline__ void getAABB(const float3& v0, const float3& v1, const float3& v2, CudaAABB& tbox);

// Computes the box and the centroid of a triangle
__device__ __forceinline__ float3 getCentroid(const float3& v0, const float3& v1, const float3& v2, CudaAABB& tbox);

// Computes which side of the plane is the point on based on its centroid
__device__ __forceinline__ int getPlaneCentroidPosition(const float4& plane, const float3& v0, const float3& v1, const float3& v2, CudaAABB& tbox);

// Split triangle bounding box based on spatil split location
__device__ __forceinline__ int getPlanePositionClipped(const float4& plane, const float3& v0, const float3& v1, const float3& v2);

// Computes which side of the plane is a ray on
__device__ __forceinline__ int getPlanePosition(const float4& plane, const float3& orig, const float3& dir, const float& tmin, const float& tmax, int& orderCounter);

// Computes the number of samples for the cost function
__device__ __host__ __forceinline__ int getNumberOfSamples(const int& number);


//------------------------------------------------------------------------
// AABB computing functions.
//------------------------------------------------------------------------


// Computes area of the bounding box
__device__ __forceinline__ float areaAABB(const volatile CudaAABB& bbox);

// Computes areas of left and right parts of bounding box divided by x
__device__ __forceinline__ void areaAABBX(const volatile CudaAABB& bbox, float pos, float& areaLeft, float& areaRight);

// Computes areas of left and right parts of bounding box divided by y
__device__ __forceinline__ void areaAABBY(const volatile CudaAABB& bbox, float pos, float& areaLeft, float& areaRight);

// Computes areas of left and right parts of bounding box divided by x
__device__ __forceinline__ void areaAABBZ(const volatile CudaAABB& bbox, float pos, float& areaLeft, float& areaRight);


//------------------------------------------------------------------------
// Axis selection and plane computation.
//------------------------------------------------------------------------


// Choose axis based on Havran's lonest-axis + round-robin mixture
__device__ __forceinline__ int taskAxis(volatile float4& plane, const volatile CudaAABB& bbox, volatile int &sharedInt, int axis);

// Splits the node with bounding box's spatial median along the longest axis
__device__ void splitMedian(int tid, int axis, volatile float4& plane, const volatile CudaAABB& bbox);

// Compute a splitting plane for each thread based on AABB
__device__ void findPlaneAABB(int planePos, const volatile CudaAABB& bbox, float4& plane, int numAxisAlignedPlanes = PLANE_COUNT);

// Compute a splitting plane for each thread
__device__ void findPlaneAABB(int planePos, const volatile CudaAABB& bbox, float& areaLeft, float& areaRight, float4& plane, int numAxisAlignedPlanes = PLANE_COUNT);

// Compute a splitting plane for each thread based on triangle division
__device__ void findPlaneTriAA(int planePos, CUdeviceptr tris, CUdeviceptr trisIndex, int triStart, int triEnd, float4& plane, int numAxisAlignedPlanes = PLANE_COUNT);

// Compute a splitting plane for each thread based on triangle division
__device__ void findPlaneTri(int planePos, CUdeviceptr tris, CUdeviceptr trisIndex, int triStart, int triEnd, float4& plane);

// Compute a splitting plane for each thread based on ray division
__device__ void findPlaneRay(int planePos, CUdeviceptr rays, CUdeviceptr raysIndex, int rayStart, int rayEnd, float4& plane);

// Compute a splitting plane for each thread
__device__ void findPlane(int planePos, CUdeviceptr rays, CUdeviceptr raysIndex,int rayStart, int rayEnd, CUdeviceptr tris, CUdeviceptr trisIndex, int triStart, int triEnd, const volatile CudaAABB& bbox, int numAxisAlignedPlanes, int numTriangleBasedPlanes, float4& plane);

// Compute a splitting plane for each thread in the chosen axis
__device__ void findPlaneRobin(int planePos, const volatile CudaAABB& bbox, int axis, float4& plane);


//------------------------------------------------------------------------
// Leaf creation.
//------------------------------------------------------------------------

// Computes Woop triangle from a regular one
__device__ void calcWoop(float3& v0, float3& v1, float3& v2, float4& o0, float4& o1, float4& o2);

// Creates a leaf in the compact layout
__device__ int createLeaf(int tid, int outOfs, float* outTriMem, int* outIdxMem, int start, int end, float* inTriMem, int* inIdxMem);

// Creates a leaf in the compact layout, with Woop triangles
__device__ int createLeafWoop(int tid, int outOfs, float4* outTriMem, int* outIdxMem, int start, int end, float4* inTriMem, Reference* inIdxMem);

// Creates a leaf in the compact layout, with Woop triangles
//_device__ int createLeafWoop(int tid, int outOfs, float4* outTriMem, int* outIdxMem, int start, int end, float4* inTriMem, int* inIdxMem);

// Creates a leaf in the compact layout, with references to triangles
__device__ int createLeafReference(int tid, int outOfs, int* outIdxMem, int start, int end, int* inIdxMem);

// Creates a leaf for a Kdtree, with Woop triangles
__device__ int createKdtreeLeafWoop(int tid, int outOfs, float4* outTriMem, int* outIdxMem, int start, int end, float4* inTriMem, int* inIdxMem);

// Creates a leaf for a Kdtree, with Woop triangles
__device__ int createKdtreeInterleavedLeafWoop(int tid, int outOfs, char* outTriMem, int start, int end, float4* inTriMem, int* inIdxMem);

// Kernel converting regular triangles to Woop triangles
extern "C" __global__ void createWoop(CUdeviceptr tri, CUdeviceptr woop, int numTris);

// Returns true if the node is a leaf
__device__ bool isKdLeaf(int flag);

#endif // __CUDACC__
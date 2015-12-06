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
#include "rt_common.cuh"
#include "tri_box_overlap.cuh"

//------------------------------------------------------------------------

// Returns the right TaskType based on the SCAN_TYPE
__device__ __forceinline__ TaskType taskChooseScanType(int unfinished)
{
#if SCAN_TYPE == 0
	return TaskType_Sort_PPS1;
#elif SCAN_TYPE == 1
	if(unfinished < 8) // Value of 8 corresponds to 256 items where there is a crossover between naive and Harris
		return TaskType_Sort_PPS1;
	else
		return TaskType_Sort_PPS1_Up;
#elif SCAN_TYPE == 2 || SCAN_TYPE == 3
	return TaskType_Sort_SORT1;
#else
#error Unknown SCAN_TYPE!
#endif
}

//------------------------------------------------------------------------

// Returns the right TaskType based on the SCAN_TYPE for PPS1
__device__ __forceinline__ TaskType taskChooseScanType1()
{
#if SCAN_TYPE == 0
	return TaskType_Sort_PPS1;
#elif SCAN_TYPE == 1
	return TaskType_Sort_PPS1_Up;
#elif SCAN_TYPE == 2 || SCAN_TYPE == 3
	return TaskType_Sort_SORT1;
#else
#error Unknown SCAN_TYPE!
#endif
}

//------------------------------------------------------------------------

// Returns the right TaskType based on the AABB_TYPE
__device__ __forceinline__ TaskType taskChooseAABBType()
{
#if AABB_TYPE < 3
	return TaskType_AABB_Min;
#elif AABB_TYPE == 3
	return TaskType_AABB;
#endif
}

//------------------------------------------------------------------------

// Returns the right TaskType based on the SCAN_TYPE for PPS1
__device__ __forceinline__ TaskType taskChooseScanType2()
{
#if SCAN_TYPE == 0
	return TaskType_Sort_PPS2;
#elif SCAN_TYPE == 1
	return TaskType_Sort_PPS2_Up;
#elif SCAN_TYPE == 2 || SCAN_TYPE == 3
	return TaskType_Sort_SORT2;
#else
#error Unknown SCAN_TYPE!
#endif
}

//------------------------------------------------------------------------

// Fetches ray from global memory
__device__ __forceinline__ void taskFetchRay(CUdeviceptr rays, int rayIdx, float3 &orig, float3 &dir, float &tmin, float &tmax)
{
	float4 o = *((float4*)(rays + rayIdx * 32 + 0));
	float4 d = *((float4*)(rays + rayIdx * 32 + 16));
	orig = make_float3(o);
	tmin = o.w;
	dir = make_float3(d);
	tmax = d.w;
}

//------------------------------------------------------------------------

// Fetches ray from global memory
__device__ __forceinline__ void taskFetchRayVolatile(CUdeviceptr rays, int rayIdx, float3 &orig, float3 &dir, float &tmin, float &tmax)
{
	// We must read data as volatile or we can get deprected data
	volatile float4 *po = (volatile float4*)(rays + rayIdx * 32 + 0);
	volatile float4 *pd = (volatile float4*)(rays + rayIdx * 32 + 16);
	orig.x = po->x, orig.y = po->y, orig.z = po->z;
	dir.x = pd->x, dir.y = pd->y, dir.z = pd->z;
	tmin = po->w;
	tmax = pd->w;
}

//------------------------------------------------------------------------

// Fetches triangle from global memory
__device__ __forceinline__ void taskFetchTri(CUdeviceptr tris, int triIdx, float3 &v0, float3 &v1, float3 &v2)
{
#if 1
	v0 = make_float3(tex1Dfetch(t_trisA, triIdx + 0));
	v1 = make_float3(tex1Dfetch(t_trisA, triIdx + 1));
	v2 = make_float3(tex1Dfetch(t_trisA, triIdx + 2));
#elif 0
	v0 = make_float3(((float4*)tris)[triIdx + 0]);
	v1 = make_float3(((float4*)tris)[triIdx + 1]);
	v2 = make_float3(((float4*)tris)[triIdx + 2]);
#else
	v0 = make_float3(*(float4*)&(((volatile float4*)tris)[triIdx + 0]));
	v1 = make_float3(*(float4*)&(((volatile float4*)tris)[triIdx + 1]));
	v2 = make_float3(*(float4*)&(((volatile float4*)tris)[triIdx + 2]));
#endif
}

//------------------------------------------------------------------------

// Fetches reference from global memory
__device__ __forceinline__ void taskFetchReference(CUdeviceptr refs, int refIdx, CudaAABB& bbox, int &idx)
{
#if 0
	v0 = make_float3(tex1Dfetch(t_trisA, triIdx + 0));
	v1 = make_float3(tex1Dfetch(t_trisA, triIdx + 1));
	v2 = make_float3(tex1Dfetch(t_trisA, triIdx + 2));
#elif 1
	bbox = (((Reference*)refs)[refIdx]).bbox;
	idx = (((Reference*)refs)[refIdx]).idx;
#else
	v0 = make_float3(*(float4*)&(((volatile float4*)tris)[triIdx + 0]));
	v1 = make_float3(*(float4*)&(((volatile float4*)tris)[triIdx + 1]));
	v2 = make_float3(*(float4*)&(((volatile float4*)tris)[triIdx + 2]));
#endif
}

//------------------------------------------------------------------------

// Fetches node from global memory
__device__ __forceinline__ void taskFetchNodeAddr(CUdeviceptr nodes, int nodeIdx, CudaBVHNode &node)
{
#if 0
	// We must read data as volatile or we can get deprected data
	volatile float4 *vc0xy = (volatile float4*)(nodes + nodeIdx + 0);      // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
	volatile float4 *vc1xy = (volatile float4*)(nodes + nodeIdx + 16);     // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
	volatile float4 *vc01z = (volatile float4*)(nodes + nodeIdx + 32);     // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
	volatile int4 *vchildren = (volatile int4*)(nodes + nodeIdx + 48);       // (leftAddr, rightAddr, parentAddr, buildState)
	
	node.c0xy.x = vc0xy->x, node.c0xy.y = vc0xy->y, node.c0xy.z = vc0xy->z, node.c0xy.w = vc0xy->w;
	node.c1xy.x = vc1xy->x, node.c1xy.y = vc1xy->y, node.c1xy.z = vc1xy->z, node.c1xy.w = vc1xy->w;
	node.c01z.x = vc01z->x, node.c01z.y = vc01z->y, node.c01z.z = vc01z->z, node.c01z.w = vc01z->w;
	node.children.x = vchildren->x, node.children.y = vchildren->y, node.children.z = vchildren->z, node.children.w = vchildren->w;
#elif 0
	CUdeviceptr addr = (nodes + nodeIdx);
	asm("{\n\t"
		"ld.volatile.v4.f32\t{%0, %1, %2, %3}, [%16];\n\t"
		"ld.volatile.v4.f32\t{%4, %5, %6, %7}, [%16+16];\n\t"
		"ld.volatile.v4.f32\t{%8, %9, %10, %11}, [%16+32];\n\t"
		"ld.volatile.v4.u32\t{%12, %13, %14, %15}, [%16+48];\n\t"
		"}"
		: "=f"(node.c0xy.x), "=f"(node.c0xy.y), "=f"(node.c0xy.z), "=f"(node.c0xy.w),
		"=f"(node.c1xy.x), "=f"(node.c1xy.y), "=f"(node.c1xy.z), "=f"(node.c1xy.w),
		"=f"(node.c01z.x), "=f"(node.c01z.y), "=f"(node.c01z.z), "=f"(node.c01z.w),
		"=r"(node.children.x), "=r"(node.children.y), "=r"(node.children.z), "=r"(node.children.w) : "r"(addr));
#elif 0 // Must be used with -Xptxas -dlcm=cg for correctness
	node.c0xy = *((float4*)(nodes + nodeIdx + 0));      // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
	node.c1xy = *((float4*)(nodes + nodeIdx + 16));     // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
	node.c01z = *((float4*)(nodes + nodeIdx + 32));     // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
	node.children = *((int4*)(nodes + nodeIdx + 48));       // (leftAddr, rightAddr, parentAddr, buildState)
#else
	node.c0xy = tex1Dfetch(t_nodesA, nodeIdx/16+0);  // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
	node.c1xy = tex1Dfetch(t_nodesA, nodeIdx/16+1);  // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
	node.c01z = tex1Dfetch(t_nodesA, nodeIdx/16+2);  // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
	float4 temp = tex1Dfetch(t_nodesA, nodeIdx/16+3);

	node.children.x =__float_as_int(temp.x);
	node.children.y =__float_as_int(temp.y);
	node.children.z =__float_as_int(temp.z);
	node.children.w =__float_as_int(temp.w);
#endif
}

//------------------------------------------------------------------------

// Fetches node from global memory
__device__ __forceinline__ void taskFetchNode(CUdeviceptr nodes, int nodeIdx, CudaBVHNode &node)
{
#if 0
	// We must read data as volatile or we can get deprected data
	volatile float4 *vc0xy = (volatile float4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 0);      // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
	volatile float4 *vc1xy = (volatile float4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 16);     // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
	volatile float4 *vc01z = (volatile float4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 32);     // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
	volatile int4 *vchildren = (volatile int4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 48);       // (leftAddr, rightAddr, parentAddr, buildState)
	
	node.c0xy.x = vc0xy->x, node.c0xy.y = vc0xy->y, node.c0xy.z = vc0xy->z, node.c0xy.w = vc0xy->w;
	node.c1xy.x = vc1xy->x, node.c1xy.y = vc1xy->y, node.c1xy.z = vc1xy->z, node.c1xy.w = vc1xy->w;
	node.c01z.x = vc01z->x, node.c01z.y = vc01z->y, node.c01z.z = vc01z->z, node.c01z.w = vc01z->w;
	node.children.x = vchildren->x, node.children.y = vchildren->y, node.children.z = vchildren->z, node.children.w = vchildren->w;
#elif 0
	CUdeviceptr addr = (nodes + nodeIdx * sizeof(CudaBVHNode));
	asm("{\n\t"
		"ld.volatile.v4.f32\t{%0, %1, %2, %3}, [%16];\n\t"
		"ld.volatile.v4.f32\t{%4, %5, %6, %7}, [%16+16];\n\t"
		"ld.volatile.v4.f32\t{%8, %9, %10, %11}, [%16+32];\n\t"
		"ld.volatile.v4.u32\t{%12, %13, %14, %15}, [%16+48];\n\t"
		"}"
		: "=f"(node.c0xy.x), "=f"(node.c0xy.y), "=f"(node.c0xy.z), "=f"(node.c0xy.w),
		"=f"(node.c1xy.x), "=f"(node.c1xy.y), "=f"(node.c1xy.z), "=f"(node.c1xy.w),
		"=f"(node.c01z.x), "=f"(node.c01z.y), "=f"(node.c01z.z), "=f"(node.c01z.w),
		"=r"(node.children.x), "=r"(node.children.y), "=r"(node.children.z), "=r"(node.children.w) : "r"(addr));
#elif 0 // Incorrect for some volativity reason
	node.c0xy = *((float4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 0));      // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
	node.c1xy = *((float4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 16));     // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
	node.c01z = *((float4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 32));     // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
	node.children = *((int4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 48));       // (leftAddr, rightAddr, parentAddr, buildState)
#else
	node.c0xy = tex1Dfetch(t_nodesA, nodeIdx*4+0);  // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
	node.c1xy = tex1Dfetch(t_nodesA, nodeIdx*4+1);  // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
	node.c01z = tex1Dfetch(t_nodesA, nodeIdx*4+2);  // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
	float4 temp = tex1Dfetch(t_nodesA, nodeIdx*4+3);

	node.children.x =__float_as_int(temp.x);
	node.children.y =__float_as_int(temp.y);
	node.children.z =__float_as_int(temp.z);
	node.children.w =__float_as_int(temp.w);
#endif
}

//------------------------------------------------------------------------

// Fetches node from global memory
__device__ __forceinline__ void taskFetchNodeVolatile(CUdeviceptr nodes, int nodeIdx, CudaBVHNode &node)
{
	// We must read data as volatile or we can get deprected data
	volatile float4 *vc0xy = (volatile float4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 0);      // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
	volatile float4 *vc1xy = (volatile float4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 16);     // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
	volatile float4 *vc01z = (volatile float4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 32);     // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
	volatile int4 *vchildren = (volatile int4*)(nodes + nodeIdx * sizeof(CudaBVHNode) + 48);       // (leftAddr, rightAddr, parentAddr, buildState)
	
	node.c0xy.x = vc0xy->x, node.c0xy.y = vc0xy->y, node.c0xy.z = vc0xy->z, node.c0xy.w = vc0xy->w;
	node.c1xy.x = vc1xy->x, node.c1xy.y = vc1xy->y, node.c1xy.z = vc1xy->z, node.c1xy.w = vc1xy->w;
	node.c01z.x = vc01z->x, node.c01z.y = vc01z->y, node.c01z.z = vc01z->z, node.c01z.w = vc01z->w;
	node.children.x = vchildren->x, node.children.y = vchildren->y, node.children.z = vchildren->z, node.children.w = vchildren->w;
}

//------------------------------------------------------------------------

// Copies node to the node array
__device__ __forceinline__ void taskSaveNodeToGMEM(CudaBVHNode* g_bvh, int tid, int nodeIdx, const volatile CudaBVHNode& node)
{
	ASSERT_DIVERGENCE("taskSaveNodeToGMEM top", tid);
	// Copy the data to global memory
	int* nodeAddr = (int*)(&g_bvh[nodeIdx]);
	if(tid < sizeof(CudaBVHNode)/sizeof(int))
		nodeAddr[tid] = ((const volatile int*)&node)[tid]; // Every thread copies one word of data of its task
	ASSERT_DIVERGENCE("taskSaveNodeToGMEM bottom", tid);
}

//------------------------------------------------------------------------

// Update the pointer in the parent to point to this node
__device__ void taskUpdateParentPtr(CudaBVHNode* g_bvh, int parentIdx, int taskID, int newValue)
{
	// Update the parent pointers
	if(parentIdx != -1) // Not for the root
	{
#if 0
		if(newTask->taskID == 0) // Left child
		{
			atomicExch(&g_bvh[parentIdx].children.x, newValue); // Inform the parent of the position of the child
			//g_bvh[parentIdx].children.x = newValue;
			//atomicAnd(&g_bvh[parentIdx].children.w, 0xFFFFFFFD); // Inform the parent the left child is ready
		}
		else
		{
			atomicExch(&g_bvh[parentIdx].children.y, newValue); // Inform the parent of the position of the child
			//g_bvh[parentIdx].children.y = newValue;
			//atomicAnd(&g_bvh[parentIdx].children.w, 0xFFFFFFFE); // Inform the parent the right child is ready
		}
#else
		//atomicExch(((int*)&g_bvh[parentIdx].children) + taskID , newValue);
		*(((int*)&g_bvh[parentIdx].children) + taskID) = newValue;
#endif
	}
}

//------------------------------------------------------------------------

// Update the pointer in the parent to point to this node
__device__ void taskUpdateParentPtr(CudaKdtreeNode* g_kdtree, int parentIdx, int taskID, int newValue)
{
	// Update the parent pointers
	if(parentIdx != -1) // Not for the root
	{
		//atomicExch(((int*)&g_bvh[parentIdx].children) + taskID , newValue);
		*(((int*)&g_kdtree[parentIdx]) + taskID) = newValue;
	}
}

//------------------------------------------------------------------------

// Computes plane dimension of axis aligned planes
__device__ __forceinline__ int getPlaneDimension(const float4& plane)
{
	return -plane.y - plane.z*2;
}

//------------------------------------------------------------------------

// Computes distance of a point from a plane
__device__ __forceinline__ float planeDistance(const float3& normal, const float& d, const float3& p)
{
	return dot(normal, p) + d;
}

//------------------------------------------------------------------------

// Creates plane from three points
__device__ __forceinline__ float4 set3PointPlane(const float3& v0, const float3& v1, const float3& v2)
{
	float3 normal = normalize(cross(v0-v1, v2-v1));
	float d = -dot(normal, v1);
	return make_float4(normal, d);
}

//------------------------------------------------------------------------

// Computes which side of the plane is a triangle on
__device__ __forceinline__ int getPlanePosition(const float4& plane, const float3& v0, const float3& v1, const float3& v2)
{
	// Fetch plane
	float3 normal;
	normal.x = plane.x;
	normal.y = plane.y;
	normal.z = plane.z;
	float d = plane.w;

	int mn = 0;
	int mx = 0;

	float vd0, vd1, vd2; // Vertex distance

#if 1
	// OPTIMIZE: Get rid of conditionals?
	vd0 = planeDistance(normal, d, v0);
	if(vd0 < EPS)
		mn = -1;
	if(vd0 > -EPS)
		mx = 1;

	vd1 = planeDistance(normal, d, v1);
	if(vd1 < EPS)
		mn = -1;
	if(vd1 > -EPS)
		mx = 1;

	vd2 = planeDistance(normal, d, v2);
	if(vd2 < EPS)
		mn = -1;
	if(vd2 > -EPS)
		mx = 1;
#else
	if(normal.x == -1.f)
	{
		int sgn1, sgn2;
		sgn1 = signbit(v0.x - d + EPS);
		mn = min(2*sgn1-1, mn);
		sgn2 = signbit(v0.x - d - EPS);
		mx = max(2*sgn2-1, mx);

		sgn1 = signbit(v1.x - d + EPS);
		mn = min(2*sgn1-1, mn);
		sgn2 = signbit(v1.x - d - EPS);
		mx = max(2*sgn2-1, mx);

		sgn1 = signbit(v2.x - d + EPS);
		mn = min(2*sgn1-1, mn);
		sgn2 = signbit(v2.x - d - EPS);
		mx = max(2*sgn2-1, mx);
	}
	else if(normal.y == -1.f)
	{
		int sgn1, sgn2;
		sgn1 = signbit(v0.y - d + EPS);
		mn = min(2*sgn1-1, mn);
		sgn2 = signbit(v0.y - d - EPS);
		mx = max(2*sgn2-1, mx);

		sgn1 = signbit(v1.y - d + EPS);
		mn = min(2*sgn1-1, mn);
		sgn2 = signbit(v1.y - d - EPS);
		mx = max(2*sgn2-1, mx);

		sgn1 = signbit(v2.y - d + EPS);
		mn = min(2*sgn1-1, mn);
		sgn2 = signbit(v2.y - d - EPS);
		mx = max(2*sgn2-1, mx);
	}
	else
	{
		int sgn1, sgn2;
		sgn1 = signbit(v0.z - d + EPS);
		mn = min(2*sgn1-1, mn);
		sgn2 = signbit(v0.z - d - EPS);
		mx = max(2*sgn2-1, mx);

		sgn1 = signbit(v1.z - d + EPS);
		mn = min(2*sgn1-1, mn);
		sgn2 = signbit(v1.z - d - EPS);
		mx = max(2*sgn2-1, mx);

		sgn1 = signbit(v2.z - d + EPS);
		mn = min(2*sgn1-1, mn);
		sgn2 = signbit(v2.z - d - EPS);
		mx = max(2*sgn2-1, mx);
	}
#endif

	return -(mn + mx);
}

//------------------------------------------------------------------------

__device__ __forceinline__ void getAABB(const float3& v0, const float3& v1, const float3& v2, CudaAABB& tbox)
{
	tbox.m_mn.x = fminf(fminf(v0.x, v1.x), v2.x);
	tbox.m_mn.y = fminf(fminf(v0.y, v1.y), v2.y);
	tbox.m_mn.z = fminf(fminf(v0.z, v1.z), v2.z);

	tbox.m_mx.x = fmaxf(fmaxf(v0.x, v1.x), v2.x);
	tbox.m_mx.y = fmaxf(fmaxf(v0.y, v1.y), v2.y);
	tbox.m_mx.z = fmaxf(fmaxf(v0.z, v1.z), v2.z);
}

//------------------------------------------------------------------------

// Computes the box and the centroid of a triangle
__device__ __forceinline__ float3 getCentroid(const float3& v0, const float3& v1, const float3& v2, CudaAABB& tbox)
{
	getAABB(v0, v1, v2, tbox);
	return (tbox.m_mn + tbox.m_mx)*0.5f;
}

// Computes the centroid of a bounding box
__device__ __forceinline__ float3 getCentroid(const CudaAABB& tbox)
{
	return (tbox.m_mn + tbox.m_mx)*0.5f;
}

//------------------------------------------------------------------------

__device__ __forceinline__ float getBoxSize(const CudaAABB& tbox, const float3& axis)
{
	if(axis.x < 0.f)
		return tbox.m_mx.x - tbox.m_mn.x;
	else if(axis.y < 0.f)
		return tbox.m_mx.y - tbox.m_mn.y;
	else
		return tbox.m_mx.z - tbox.m_mn.z;
}


// Computes which side of the plane is the point on based on its centroid
__device__ __forceinline__ int getPlaneCentroidPosition(const float4& plane, const float3& v0, const float3& v1, const float3& v2, CudaAABB& tbox)
{
	// Fetch plane
	float3 normal;
	normal.x = plane.x;
	normal.y = plane.y;
	normal.z = plane.z;
	float d = plane.w;

	int pos;

	float3 centroid = getCentroid(v0, v1, v2, tbox);

	float ctd = planeDistance(normal, d, centroid);
	if(ctd < EPS)
		pos = -1;
	else
		pos = 1;

	return pos;
}

// Computes which side of the plane is the point on based on its centroid
__device__ __forceinline__ int getPlaneCentroidPosition(const float4& plane, const CudaAABB& tbox)
{
	// Fetch plane
	float3 normal;
	normal.x = plane.x;
	normal.y = plane.y;
	normal.z = plane.z;
	float d = plane.w;

	int pos;

	float3 centroid = getCentroid(tbox);

	float ctd = planeDistance(normal, d, centroid);
	if(ctd < EPS)
		pos = -1;
	else
		pos = 1;

	return pos;
}

// Computes which side of the plane is the point on based on its centroid. Returns -2 or 2 if plane intersects bbox.
__device__ __forceinline__ int getPlaneCentroidPositionHitMiss(const float4& plane, const CudaAABB& tbox)
{
	// Fetch plane
	float3 normal;
	normal.x = plane.x;
	normal.y = plane.y;
	normal.z = plane.z;
	float d = plane.w;

	int pos;

	float size = getBoxSize(tbox, normal);

	float3 centroid = getCentroid(tbox);

	float ctd = planeDistance(normal, d, centroid);
	if(ctd < EPS)
		pos = 1;
	else
		pos = -1;

	if(abs(ctd) < size/2)
		pos *= 2;

	//printf("hitmiss: pos %i ctd %f size %f | plane %.3f %.3f %.3f %.3f {} box %.3f %.3f %.3f %.3f %.3f %.3f\n",
	//	pos, ctd, size, plane.x, plane.y, plane.z, plane.w, tbox.m_mn.x, tbox.m_mn.y, tbox.m_mn.z, tbox.m_mx.x, tbox.m_mx.y, tbox.m_mx.z);

	return pos;
}

//------------------------------------------------------------------------

// Split triangle bounding box based on spatial split location
__device__ __forceinline__ void computeClippedBoxes(const float4& plane, const float3& v0, const float3& v1, const float3& v2, const volatile CudaAABB& nodeBox, CudaAABB& leftBox, CudaAABB& rightBox)
{
	int dim = getPlaneDimension(plane);
	float split = plane.w;
	CudaAABB triBox, triBoxL, triBoxR;

	getAABB(v0, v1, v2, triBox);

	// Because GPUs do not support register indexing we have to switch execution based on dimension
	switch(dim)
	{
	case 0:
		//initializing tight AABBs only  for splitting dimension 
		triBoxL.m_mn.x = triBox.m_mn.x;
		triBoxR.m_mx.x = triBox.m_mx.x;
		triBoxL.m_mx.x = triBoxR.m_mn.x = split;

		//two remaining dimensions are recomputed 
		{
			//reordering vertices’ indices 
			const float3* _min  = (v1.x <= v0.x) ? &v1 : &v0;
			const float3* _max  = (v1.x <= v0.x) ? &v0 : &v1;
			const float3* vertMin = (v2.x <  _min->x) ? &v2 : _min;
			const float3* vertMax = (v2.x >= _max->x) ? &v2 : _max;
			const float3* vertMid = (&v0 != vertMin && &v0 != vertMax) ? &v0 : ((&v1 != vertMin && &v1 != vertMax) ? &v1 : &v2);
			const bool conda = split <= vertMid->x;
			const float3* iA = conda ? vertMin : vertMax;
			const float3* iB = vertMid;
			const float3* iC = conda ? vertMax : vertMin;

			const float ratio_ab = (split-iA->x)/(iB->x-iA->x);
			const float ratio_cd = (split-iA->x)/(iC->x-iA->x);

			const float x0 = iA->y + ratio_ab*(iB->y-iA->y);
			const float x1 = iA->y + ratio_cd*(iC->y-iA->y);
			const float xmin = fminf(x0, x1);
			const float xmax = fmaxf(x0, x1);

			if(conda){
			triBoxL.m_mn.y = fminf(xmin, iA->y);
			triBoxL.m_mx.y = fmaxf(xmax, iA->y);
			triBoxR.m_mn.y = fminf(xmin, fminf(iB->y, iC->y));
			triBoxR.m_mx.y = fmaxf(xmax, fmaxf(iB->y, iC->y));
			}else{
			triBoxR.m_mn.y = fminf(xmin, iA->y);
			triBoxR.m_mx.y = fmaxf(xmax, iA->y);
			triBoxL.m_mn.y = fminf(xmin, fminf(iB->y, iC->y));
			triBoxL.m_mx.y = fmaxf(xmax, fmaxf(iB->y, iC->y));
			}

			const float y0 = iA->z + ratio_ab*(iB->z-iA->z);
			const float y1 = iA->z + ratio_cd*(iC->z-iA->z);
			const float ymin = fminf(y0, y1);
			const float ymax = fmaxf(y0, y1);

			if(conda){
			triBoxL.m_mn.z = fminf(ymin, iA->z);
			triBoxL.m_mx.z = fmaxf(ymax, iA->z);
			triBoxR.m_mn.z = fminf(ymin, fminf(iB->z, iC->z));
			triBoxR.m_mx.z = fmaxf(ymax, fmaxf(iB->z, iC->z));
			}else{
			triBoxR.m_mn.z = fminf(ymin, iA->z);
			triBoxR.m_mx.z = fmaxf(ymax, iA->z);
			triBoxL.m_mn.z = fminf(ymin, fminf(iB->z, iC->z));
			triBoxL.m_mx.z = fmaxf(ymax, fmaxf(iB->z, iC->z));
			}
		}

		break;

	case 1:
		//initializing tight AABBs only  for splitting dimension 
		triBoxL.m_mn.y = triBox.m_mn.y;
		triBoxR.m_mx.y = triBox.m_mx.y;
		triBoxL.m_mx.y = triBoxR.m_mn.y = split;
		
		//two remaining dimensions are recomputed 
		{
			//reordering vertices’ indices 
			const float3* _min  = (v1.y <= v0.y) ? &v1 : &v0;
			const float3* _max  = (v1.y <= v0.y) ? &v0 : &v1;
			const float3* vertMin = (v2.y <  _min->y) ? &v2 : _min;
			const float3* vertMax = (v2.y >= _max->y) ? &v2 : _max;
			const float3* vertMid = (&v0 != vertMin && &v0 != vertMax) ? &v0 : ((&v1 != vertMin && &v1 != vertMax) ? &v1 : &v2);
			const bool conda = split <= vertMid->y;
			const float3* iA = conda ? vertMin : vertMax;
			const float3* iB = vertMid;
			const float3* iC = conda ? vertMax : vertMin;

			const float ratio_ab = (split-iA->y)/(iB->y-iA->y);
			const float ratio_cd = (split-iA->y)/(iC->y-iA->y);

			const float x0 = iA->x + ratio_ab*(iB->x-iA->x);
			const float x1 = iA->x + ratio_cd*(iC->x-iA->x);
			const float xmin = fminf(x0, x1);
			const float xmax = fmaxf(x0, x1);

			if(conda){
			triBoxL.m_mn.x = fminf(xmin, iA->x);
			triBoxL.m_mx.x = fmaxf(xmax, iA->x);
			triBoxR.m_mn.x = fminf(xmin, fminf(iB->x, iC->x));
			triBoxR.m_mx.x = fmaxf(xmax, fmaxf(iB->x, iC->x));
			}else{
			triBoxR.m_mn.x = fminf(xmin, iA->x);
			triBoxR.m_mx.x = fmaxf(xmax, iA->x);
			triBoxL.m_mn.x = fminf(xmin, fminf(iB->x, iC->x));
			triBoxL.m_mx.x = fmaxf(xmax, fmaxf(iB->x, iC->x));
			}

			const float y0 = iA->z + ratio_ab*(iB->z-iA->z);
			const float y1 = iA->z + ratio_cd*(iC->z-iA->z);
			const float ymin = fminf(y0, y1);
			const float ymax = fmaxf(y0, y1);

			if(conda){
			triBoxL.m_mn.z = fminf(ymin, iA->z);
			triBoxL.m_mx.z = fmaxf(ymax, iA->z);
			triBoxR.m_mn.z = fminf(ymin, fminf(iB->z, iC->z));
			triBoxR.m_mx.z = fmaxf(ymax, fmaxf(iB->z, iC->z));
			}else{
			triBoxR.m_mn.z = fminf(ymin, iA->z);
			triBoxR.m_mx.z = fmaxf(ymax, iA->z);
			triBoxL.m_mn.z = fminf(ymin, fminf(iB->z, iC->z));
			triBoxL.m_mx.z = fmaxf(ymax, fmaxf(iB->z, iC->z));
			}
		}

		break;

	case 2:
		//initializing tight AABBs only  for splitting dimension 
		triBoxL.m_mn.z = triBox.m_mn.z;
		triBoxR.m_mx.z = triBox.m_mx.z;
		triBoxL.m_mx.z = triBoxR.m_mn.z = split;
		
		//two remaining dimensions are recomputed
		{
			//reordering vertices’ indices 
			const float3* _min  = (v1.z <= v0.z) ? &v1 : &v0;
			const float3* _max  = (v1.z <= v0.z) ? &v0 : &v1;
			const float3* vertMin = (v2.z <  _min->z) ? &v2 : _min;
			const float3* vertMax = (v2.z >= _max->z) ? &v2 : _max;
			const float3* vertMid = (&v0 != vertMin && &v0 != vertMax) ? &v0 : ((&v1 != vertMin && &v1 != vertMax) ? &v1 : &v2);
			const bool conda = split <= vertMid->z;
			const float3* iA = conda ? vertMin : vertMax;
			const float3* iB = vertMid;
			const float3* iC = conda ? vertMax : vertMin;

			const float ratio_ab = (split-iA->z)/(iB->z-iA->z);
			const float ratio_cd = (split-iA->z)/(iC->z-iA->z);

			const float x0 = iA->y + ratio_ab*(iB->y-iA->y);
			const float x1 = iA->y + ratio_cd*(iC->y-iA->y);
			const float xmin = fminf(x0, x1);
			const float xmax = fmaxf(x0, x1);

			if(conda){
			triBoxL.m_mn.y = fminf(xmin, iA->y);
			triBoxL.m_mx.y = fmaxf(xmax, iA->y);
			triBoxR.m_mn.y = fminf(xmin, fminf(iB->y, iC->y));
			triBoxR.m_mx.y = fmaxf(xmax, fmaxf(iB->y, iC->y));
			}else{
			triBoxR.m_mn.y = fminf(xmin, iA->y);
			triBoxR.m_mx.y = fmaxf(xmax, iA->y);
			triBoxL.m_mn.y = fminf(xmin, fminf(iB->y, iC->y));
			triBoxL.m_mx.y = fmaxf(xmax, fmaxf(iB->y, iC->y));
			}

			const float y0 = iA->x + ratio_ab*(iB->x-iA->x);
			const float y1 = iA->x + ratio_cd*(iC->x-iA->x);
			const float ymin = fminf(y0, y1);
			const float ymax = fmaxf(y0, y1);

			if(conda){
			triBoxL.m_mn.x = fminf(ymin, iA->x);
			triBoxL.m_mx.x = fmaxf(ymax, iA->x);
			triBoxR.m_mn.x = fminf(ymin, fminf(iB->x, iC->x));
			triBoxR.m_mx.x = fmaxf(ymax, fmaxf(iB->x, iC->x));
			}else{
			triBoxR.m_mn.x = fminf(ymin, iA->x);
			triBoxR.m_mx.x = fmaxf(ymax, iA->x);
			triBoxL.m_mn.x = fminf(ymin, fminf(iB->x, iC->x));
			triBoxL.m_mx.x = fmaxf(ymax, fmaxf(iB->x, iC->x));
			}
		}

		break;
	}

	leftBox = triBoxL;
	rightBox = triBoxR;
}

// Split triangle bounding box based on spatial split location
__device__ __forceinline__ int getPlanePositionClipped(const float4& plane, const float3& v0, const float3& v1, const float3& v2, const CudaAABB& nodeBox)
{
	int dim = getPlaneDimension(plane);
	float split = plane.w;
	CudaAABB triBox, triBoxL, triBoxR;

	getAABB(v0, v1, v2, triBox);

	// Because GPUs do not support register indexing we have to switch execution based on dimension
	switch(dim)
	{
	case 0:
		//initializing tight AABBs only  for splitting dimension 
		triBoxL.m_mn.x = triBox.m_mn.x;
		triBoxR.m_mx.x = triBox.m_mx.x;
		triBoxL.m_mx.x = triBoxR.m_mn.x = split;

		//two remaining dimensions are recomputed 
		{
			//reordering vertices’ indices 
			const float3* _min  = (v1.x <= v0.x) ? &v1 : &v0;
			const float3* _max  = (v1.x <= v0.x) ? &v0 : &v1;
			const float3* vertMin = (v2.x <  _min->x) ? &v2 : _min;
			const float3* vertMax = (v2.x >= _max->x) ? &v2 : _max;
			const float3* vertMid = (&v0 != vertMin && &v0 != vertMax) ? &v0 : ((&v1 != vertMin && &v1 != vertMax) ? &v1 : &v2);
			const bool conda = split <= vertMid->x;
			const float3* iA = conda ? vertMin : vertMax;
			const float3* iB = vertMid;
			const float3* iC = conda ? vertMax : vertMin;

			const float ratio_ab = (split-iA->x)/(iB->x-iA->x);
			const float ratio_cd = (split-iA->x)/(iC->x-iA->x);

			const float x0 = iA->y + ratio_ab*(iB->y-iA->y);
			const float x1 = iA->y + ratio_cd*(iC->y-iA->y);
			const float xmin = fminf(x0, x1);
			const float xmax = fmaxf(x0, x1);

			if(conda){
			triBoxL.m_mn.y = fminf(xmin, iA->y);
			triBoxL.m_mx.y = fmaxf(xmax, iA->y);
			triBoxR.m_mn.y = fminf(xmin, fminf(iB->y, iC->y));
			triBoxR.m_mx.y = fmaxf(xmax, fmaxf(iB->y, iC->y));
			}else{
			triBoxR.m_mn.y = fminf(xmin, iA->y);
			triBoxR.m_mx.y = fmaxf(xmax, iA->y);
			triBoxL.m_mn.y = fminf(xmin, fminf(iB->y, iC->y));
			triBoxL.m_mx.y = fmaxf(xmax, fmaxf(iB->y, iC->y));
			}

			const float y0 = iA->z + ratio_ab*(iB->z-iA->z);
			const float y1 = iA->z + ratio_cd*(iC->z-iA->z);
			const float ymin = fminf(y0, y1);
			const float ymax = fmaxf(y0, y1);

			if(conda){
			triBoxL.m_mn.z = fminf(ymin, iA->z);
			triBoxL.m_mx.z = fmaxf(ymax, iA->z);
			triBoxR.m_mn.z = fminf(ymin, fminf(iB->z, iC->z));
			triBoxR.m_mx.z = fmaxf(ymax, fmaxf(iB->z, iC->z));
			}else{
			triBoxR.m_mn.z = fminf(ymin, iA->z);
			triBoxR.m_mx.z = fmaxf(ymax, iA->z);
			triBoxL.m_mn.z = fminf(ymin, fminf(iB->z, iC->z));
			triBoxL.m_mx.z = fmaxf(ymax, fmaxf(iB->z, iC->z));
			}
		}

		break;

	case 1:
		//initializing tight AABBs only  for splitting dimension 
		triBoxL.m_mn.y = triBox.m_mn.y;
		triBoxR.m_mx.y = triBox.m_mx.y;
		triBoxL.m_mx.y = triBoxR.m_mn.y = split;
		
		//two remaining dimensions are recomputed 
		{
			//reordering vertices’ indices 
			const float3* _min  = (v1.y <= v0.y) ? &v1 : &v0;
			const float3* _max  = (v1.y <= v0.y) ? &v0 : &v1;
			const float3* vertMin = (v2.y <  _min->y) ? &v2 : _min;
			const float3* vertMax = (v2.y >= _max->y) ? &v2 : _max;
			const float3* vertMid = (&v0 != vertMin && &v0 != vertMax) ? &v0 : ((&v1 != vertMin && &v1 != vertMax) ? &v1 : &v2);
			const bool conda = split <= vertMid->y;
			const float3* iA = conda ? vertMin : vertMax;
			const float3* iB = vertMid;
			const float3* iC = conda ? vertMax : vertMin;

			const float ratio_ab = (split-iA->y)/(iB->y-iA->y);
			const float ratio_cd = (split-iA->y)/(iC->y-iA->y);

			const float x0 = iA->x + ratio_ab*(iB->x-iA->x);
			const float x1 = iA->x + ratio_cd*(iC->x-iA->x);
			const float xmin = fminf(x0, x1);
			const float xmax = fmaxf(x0, x1);

			if(conda){
			triBoxL.m_mn.x = fminf(xmin, iA->x);
			triBoxL.m_mx.x = fmaxf(xmax, iA->x);
			triBoxR.m_mn.x = fminf(xmin, fminf(iB->x, iC->x));
			triBoxR.m_mx.x = fmaxf(xmax, fmaxf(iB->x, iC->x));
			}else{
			triBoxR.m_mn.x = fminf(xmin, iA->x);
			triBoxR.m_mx.x = fmaxf(xmax, iA->x);
			triBoxL.m_mn.x = fminf(xmin, fminf(iB->x, iC->x));
			triBoxL.m_mx.x = fmaxf(xmax, fmaxf(iB->x, iC->x));
			}

			const float y0 = iA->z + ratio_ab*(iB->z-iA->z);
			const float y1 = iA->z + ratio_cd*(iC->z-iA->z);
			const float ymin = fminf(y0, y1);
			const float ymax = fmaxf(y0, y1);

			if(conda){
			triBoxL.m_mn.z = fminf(ymin, iA->z);
			triBoxL.m_mx.z = fmaxf(ymax, iA->z);
			triBoxR.m_mn.z = fminf(ymin, fminf(iB->z, iC->z));
			triBoxR.m_mx.z = fmaxf(ymax, fmaxf(iB->z, iC->z));
			}else{
			triBoxR.m_mn.z = fminf(ymin, iA->z);
			triBoxR.m_mx.z = fmaxf(ymax, iA->z);
			triBoxL.m_mn.z = fminf(ymin, fminf(iB->z, iC->z));
			triBoxL.m_mx.z = fmaxf(ymax, fmaxf(iB->z, iC->z));
			}
		}

		break;

	case 2:
		//initializing tight AABBs only  for splitting dimension 
		triBoxL.m_mn.z = triBox.m_mn.z;
		triBoxR.m_mx.z = triBox.m_mx.z;
		triBoxL.m_mx.z = triBoxR.m_mn.z = split;
		
		//two remaining dimensions are recomputed
		{
			//reordering vertices’ indices 
			const float3* _min  = (v1.z <= v0.z) ? &v1 : &v0;
			const float3* _max  = (v1.z <= v0.z) ? &v0 : &v1;
			const float3* vertMin = (v2.z <  _min->z) ? &v2 : _min;
			const float3* vertMax = (v2.z >= _max->z) ? &v2 : _max;
			const float3* vertMid = (&v0 != vertMin && &v0 != vertMax) ? &v0 : ((&v1 != vertMin && &v1 != vertMax) ? &v1 : &v2);
			const bool conda = split <= vertMid->z;
			const float3* iA = conda ? vertMin : vertMax;
			const float3* iB = vertMid;
			const float3* iC = conda ? vertMax : vertMin;

			const float ratio_ab = (split-iA->z)/(iB->z-iA->z);
			const float ratio_cd = (split-iA->z)/(iC->z-iA->z);

			const float x0 = iA->y + ratio_ab*(iB->y-iA->y);
			const float x1 = iA->y + ratio_cd*(iC->y-iA->y);
			const float xmin = fminf(x0, x1);
			const float xmax = fmaxf(x0, x1);

			if(conda){
			triBoxL.m_mn.y = fminf(xmin, iA->y);
			triBoxL.m_mx.y = fmaxf(xmax, iA->y);
			triBoxR.m_mn.y = fminf(xmin, fminf(iB->y, iC->y));
			triBoxR.m_mx.y = fmaxf(xmax, fmaxf(iB->y, iC->y));
			}else{
			triBoxR.m_mn.y = fminf(xmin, iA->y);
			triBoxR.m_mx.y = fmaxf(xmax, iA->y);
			triBoxL.m_mn.y = fminf(xmin, fminf(iB->y, iC->y));
			triBoxL.m_mx.y = fmaxf(xmax, fmaxf(iB->y, iC->y));
			}

			const float y0 = iA->x + ratio_ab*(iB->x-iA->x);
			const float y1 = iA->x + ratio_cd*(iC->x-iA->x);
			const float ymin = fminf(y0, y1);
			const float ymax = fmaxf(y0, y1);

			if(conda){
			triBoxL.m_mn.x = fminf(ymin, iA->x);
			triBoxL.m_mx.x = fmaxf(ymax, iA->x);
			triBoxR.m_mn.x = fminf(ymin, fminf(iB->x, iC->x));
			triBoxR.m_mx.x = fmaxf(ymax, fmaxf(iB->x, iC->x));
			}else{
			triBoxR.m_mn.x = fminf(ymin, iA->x);
			triBoxR.m_mx.x = fmaxf(ymax, iA->x);
			triBoxL.m_mn.x = fminf(ymin, fminf(iB->x, iC->x));
			triBoxL.m_mx.x = fmaxf(ymax, fmaxf(iB->x, iC->x));
			}
		}

		break;
	}

	float3 intersectMn = fmaxf(triBoxL.m_mn, nodeBox.m_mn);
	float3 intersectMx = fminf(triBoxL.m_mx, nodeBox.m_mx);
	bool leftIsect = (intersectMn.x <= intersectMx.x) && (intersectMn.y <= intersectMx.y) && (intersectMn.z <= intersectMx.z);
	intersectMn = fmaxf(triBoxR.m_mn, nodeBox.m_mn);
	intersectMx = fminf(triBoxR.m_mx, nodeBox.m_mx);
	bool rightIsect = (intersectMn.x <= intersectMx.x) && (intersectMn.y <= intersectMx.y) && (intersectMn.z <= intersectMx.z);
	return -1*leftIsect + 1*rightIsect;
}

inline __host__ __device__ double3 operator+(double3 a, double3 b)
{
    return make_double3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __host__ __device__ double3 operator-(double3 a, double3 b)
{
    return make_double3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __host__ __device__ double3 operator*(double3 a, float b)
{
    return make_double3(a.x * b, a.y * b, a.z * b);
}

//------------------------------------------------------------------------
__device__ __forceinline__ void boxCenterHalfSize(const CudaAABB& nodeBox, float3& center, float3& halfSize)
{
	center = (nodeBox.m_mn + nodeBox.m_mx)*0.5f;
	halfSize = (nodeBox.m_mx - nodeBox.m_mn)*0.5f/* + 2000*EPS*/;
	/*double3 cD, hD;
	cD = (make_double3(nodeBox.m_mn.x, nodeBox.m_mn.y, nodeBox.m_mn.z) + make_double3(nodeBox.m_mx.x, nodeBox.m_mx.y, nodeBox.m_mx.z))*0.5;
	hD = (make_double3(nodeBox.m_mx.x, nodeBox.m_mx.y, nodeBox.m_mx.z) - make_double3(nodeBox.m_mn.x, nodeBox.m_mn.y, nodeBox.m_mn.z))*0.5;
	center = make_float3(cD.x, cD.y, cD.z);
	halfSize = make_float3(hD.x, hD.y, hD.z);*/
}

//------------------------------------------------------------------------

// Compute triangle's position wrt splitting plane by computing its intersection with children bounding boxes
__device__ __forceinline__ int getTriChildOverlap(const float4& plane, const float3& v0, const float3& v1, const float3& v2, const CudaAABB& nodeBox)
{
	int dim = getPlaneDimension(plane);
	float split = plane.w;
	CudaAABB nodeBoxL, nodeBoxR;

	nodeBoxL = nodeBoxR = nodeBox;

	// Because GPUs do not support register indexing we have to switch execution based on dimension
	switch(dim)
	{
	case 0:
		nodeBoxL.m_mx.x = nodeBoxR.m_mn.x = split;
		break;

	case 1:
		nodeBoxL.m_mx.y = nodeBoxR.m_mn.y = split;
		break;
		
	case 2:
		nodeBoxL.m_mx.z = nodeBoxR.m_mn.z = split;
		break;
	}

	float3 boxCenterL, boxHalfSizeL;
	boxCenterHalfSize(nodeBoxL, boxCenterL, boxHalfSizeL);
	int leftIsect = triBoxOverlap(boxCenterL, boxHalfSizeL, v0, v1, v2, nodeBoxL.m_mn, nodeBoxL.m_mx);

	float3 boxCenterR, boxHalfSizeR;
	boxCenterHalfSize(nodeBoxR, boxCenterR, boxHalfSizeR);
	int rightIsect = triBoxOverlap(boxCenterR, boxHalfSizeR, v0, v1, v2, nodeBoxR.m_mn, nodeBoxR.m_mx);

	if(leftIsect == 0 && rightIsect == 0) // Should not happen, but happens due to numerical imprecision
	{
		//printf("Cannot happen!\n");
		return -1;
	}

	return -1*leftIsect + 1*rightIsect;
}

//------------------------------------------------------------------------

// Computes which side of the plane is a ray on
__device__ __forceinline__ int getPlanePosition(const float4& plane, const float3& orig, const float3& dir, const float& tmin, const float& tmax, int& orderCounter)
{
	// Fetch plane
	float3 normal;
	normal.x = plane.x;
	normal.y = plane.y;
	normal.z = plane.z;
	float d = plane.w;

	int retVal;

#if 0
	int min = 0;
	int max = 0;

	float d1 = planeDistance(normal, d, orig + tmin*dir);
	float d2 = planeDistance(normal, d, orig + tmax*dir);

	// OPTIMIZE: Get rid of conditionals?
	if (d1 < EPS)
		min = -1;
	if (d1 > -EPS)
		max = 1;

	if (d2 < EPS)
		min = -1;
	if (d2 > -EPS)
		max = 1;

	retVal = min + max;
#else
	float dv = dot(dir, normal);
	orderCounter = 0;

#define COPLANAR_EPS 1e-30f
	if(dv < -COPLANAR_EPS)
	{
		// the ray will hit from the front side
		float t = -planeDistance(normal, d, orig) / dv; 

		if (t > tmax + EPS)
			retVal = 1;
		else if (t < tmin - EPS)
			retVal = -1;
		else 
		{
			// hits the plane from front to back
			orderCounter = -1;
			retVal = 0;
		}
	}
	else if(dv > COPLANAR_EPS)
	{
		// the ray will hit from the front side
		float t = -planeDistance(normal, d, orig) / dv; 

		if (t > tmax + EPS)
			retVal = -1;
		else if (t < tmin - EPS)
			retVal = 1;
		else
		{
			// hits the plane from back to front
			orderCounter = 1;
			retVal = 0;
		}
	}  
	else
	{
		int min = 0;
		int max = 0;

		float d1 = planeDistance(normal, d, orig + tmin*dir);
		float d2 = planeDistance(normal, d, orig + tmax*dir);

		// OPTIMIZE: Get rid of conditionals?
		if (d1 < EPS)
			min = -1;
		if (d1 > -EPS)
			max = 1;

		if (d2 < EPS)
			min = -1;
		if (d2 > -EPS)
			max = 1;

		retVal = min + max;
	}
#endif

	return retVal;
}

//------------------------------------------------------------------------

// Computes the number of samples for the cost function
__device__ __host__ __forceinline__ int getNumberOfSamples(const int& number)
{
	return (int)sqrtf(number);
}

//------------------------------------------------------------------------

// Computes area of the bounding box
__device__ __forceinline__ float areaAABB(const volatile CudaAABB& bbox)
{
	float3 d;
	d.x = bbox.m_mx.x - bbox.m_mn.x;
	d.y = bbox.m_mx.y - bbox.m_mn.y;
	d.z = bbox.m_mx.z - bbox.m_mn.z;
	return (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f;
}

//------------------------------------------------------------------------

// Computes areas of left and right parts of bounding box divided by x
__device__ __forceinline__ void areaAABBX(const volatile CudaAABB& bbox, float pos, float& areaLeft, float& areaRight)
{
	float3 d;
	d.x = pos - bbox.m_mn.x;
	d.y = bbox.m_mx.y - bbox.m_mn.y;
	d.z = bbox.m_mx.z - bbox.m_mn.z;
	areaLeft = (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f;

	d.x = bbox.m_mx.x - pos;
	areaRight = (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f;
}

//------------------------------------------------------------------------

// Computes areas of left and right parts of bounding box divided by y
__device__ __forceinline__ void areaAABBY(const volatile CudaAABB& bbox, float pos, float& areaLeft, float& areaRight)
{
	float3 d;
	d.x = bbox.m_mx.x - bbox.m_mn.x;
	d.y = pos - bbox.m_mn.y;
	d.z = bbox.m_mx.z - bbox.m_mn.z;
	areaLeft = (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f;

	d.y = bbox.m_mx.y - pos;
	areaRight = (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f;
}

//------------------------------------------------------------------------

// Computes areas of left and right parts of bounding box divided by x
__device__ __forceinline__ void areaAABBZ(const volatile CudaAABB& bbox, float pos, float& areaLeft, float& areaRight)
{
	float3 d;
	d.x = bbox.m_mx.x - bbox.m_mn.x;
	d.y = bbox.m_mx.y - bbox.m_mn.y;
	d.z = pos - bbox.m_mn.z;
	areaLeft = (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f;

	d.z = bbox.m_mx.z - pos;
	areaRight = (d.x*d.y + d.y*d.z + d.z*d.x)*2.0f;
}

//------------------------------------------------------------------------

// Choose axis based on Havran's lonest-axis + round-robin mixture
__device__ __forceinline__ int taskAxis(volatile float4& plane, const volatile CudaAABB& bbox, volatile int &sharedInt, int axis)
{

	volatile float* tPln = ((volatile float*)&plane)+threadIdx.x;
	volatile float* tMin = ((volatile float*)&bbox.m_mn)+threadIdx.x;
	volatile float* tMax = ((volatile float*)&bbox.m_mx)+threadIdx.x;
	
	// Compute longest axis
	if(threadIdx.x < 3)
	{
		*tPln = *tMax - *tMin;
		float dMax = max3(plane.x, plane.y, plane.z);
		if(__ffs(__ballot(dMax == *tPln)) == threadIdx.x+1) // First thread with such condition
		{
			sharedInt = threadIdx.x; // Longest axis
		}
	}

	int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
	return ((warpIdx & 0x3) != 0) ? axis : sharedInt;
}

//------------------------------------------------------------------------

// Splits the node with bounding box's spatial median along the longest axis
__device__ void splitMedian(int tid, int axis, volatile float4& plane, const volatile CudaAABB& bbox)
{
	ASSERT_DIVERGENCE("splitMedian", tid);

	volatile float* tPln = ((volatile float*)&plane)+tid;
	volatile float* tMin = ((volatile float*)&bbox.m_mn)+tid;
	volatile float* tMax = ((volatile float*)&bbox.m_mx)+tid;
#if 0 // Longest axis
	
	// Compute spatial median
	if(tid < 3)
	{
#if 1
		*tPln = *tMax - *tMin;
		float dMax = max3(plane.x, plane.y, plane.z);
		if(__ffs(__ballot(dMax == *tPln)) == tid+1) // First thread with such condition
		{
			plane.w = -(*tMin + *tMax) / 2.0f;
			*tPln = 1;
		}
		else
		{
			*tPln = 0;
		}
#else
		if(tid == 0) // Single thread median split
		{
			if(dMax == plane[threadIdx.y].x)
			{
				plane[threadIdx.y].x = 1;
				plane[threadIdx.y].w = -(bbox.m_mn.x + bbox.m_mx.x) / 2.0f;
			}
			else
				plane[threadIdx.y].x = 0;
			
			if(dMax == plane[threadIdx.y].y)
			{
				plane[threadIdx.y].y = 1;
				plane[threadIdx.y].w = -(bbox.m_mn.y + bbox.m_mx.y) / 2.0f;
			}
			else
				plane[threadIdx.y].y = 0;

			if(dMax == plane[threadIdx.y].z)
			{
				plane[threadIdx.y].z = 1;
				plane[threadIdx.y].w = -(bbox.m_mn.z + bbox.m_mx.z) / 2.0f;
			}
			else
				plane[threadIdx.y].z = 0;
		}
#endif
	}
#else // Round robin

	//int axis = depth % 3;
	if(tid < 3)
	{
		*tPln = *tMax - *tMin;
		if(tid == axis)
		{
			plane.w = -(*tMin + *tMax) / 2.0f;
			*tPln = 1;
		}
		else
		{
			*tPln = 0;
		}
	}	
#endif
}

//------------------------------------------------------------------------

// Compute a splitting plane for each thread based on AABB
__device__ void findPlaneAABB(int planePos, const volatile CudaAABB& bbox, float4& plane, int numAxisAlignedPlanes)
{
	//ASSERT_DIVERGENCE("findPlaneAABB", threadIdx.x);

#if 1 // Equal number of planes in each dimension
	int planesPerAxis = ((numAxisAlignedPlanes+2) / 3);
	int axis = planePos / planesPerAxis;
	float rpos = (float)(1 + (planePos % planesPerAxis))/(float)(planesPerAxis+1);

	if(axis == 0)
	{
		float pos = bbox.m_mn.x + (bbox.m_mx.x - bbox.m_mn.x) * rpos;
		plane = make_float4(-1.f, 0.f, 0.f, pos);
	}
	else if(axis == 1)
	{
		float pos = bbox.m_mn.y + (bbox.m_mx.y - bbox.m_mn.y) * rpos;
		plane = make_float4(0.f, -1.f, 0.f, pos);
	}
	else
	{
		float pos = bbox.m_mn.z + (bbox.m_mx.z - bbox.m_mn.z) * rpos;
		plane = make_float4(0.f, 0.f, -1.f, pos);
	}
#else		
	float lX = bbox.m_mx.x - bbox.m_mn.x;
	float lY = bbox.m_mx.y - bbox.m_mn.y;
	float lZ = bbox.m_mx.z - bbox.m_mn.z;

	float sumLengths = lX + lY + lZ;
	// Assign the planes to different methods
	int numX = lX/sumLengths*PLANE_COUNT+0.5f;
	int numY = lY/sumLengths*PLANE_COUNT+0.5f;
	int numZ = lZ/sumLengths*PLANE_COUNT+0.5f;

	//int axis = (planePos < numX) ? 0 : (planePos < numX+numY) ? 1 : 2;
	int axis = (planePos >= numX) + (planePos >= numX+numY);

	if(axis == 0)
	{
		float rpos = (float)(planePos+1) / (float)(numX+1);
		float pos = bbox.m_mn.x + lX * rpos;
		plane = make_float4(-1.f, 0.f, 0.f, pos);
	}
	else if(axis == 1)
	{
		float rpos = (float)(planePos-numX+1) / (float)(numY+1);
		float pos = bbox.m_mn.y + lY * rpos;
		plane = make_float4(0.f, -1.f, 0.f, pos);
	}
	else
	{
		float rpos = (float)(planePos-numX-numY+1) / (float)(numZ+1);
		float pos = bbox.m_mn.z + lZ * rpos;
		plane = make_float4(0.f, 0.f, -1.f, pos);
	}
#endif
}

//------------------------------------------------------------------------

// Compute a splitting plane for each thread
__device__ void findPlaneAABB(int planePos, const volatile CudaAABB& bbox, float& areaLeft, float& areaRight, float4& plane, int numAxisAlignedPlanes)
{
	//ASSERT_DIVERGENCE("findPlaneAABB", threadIdx.x);

#if 1 // Equal number of planes in each dimension
	int planesPerAxis = ((numAxisAlignedPlanes+2) / 3);
	int axis = planePos / planesPerAxis;
	float rpos = (float)( 1 + (planePos % planesPerAxis))/(float)(planesPerAxis+1);

	if(axis == 0)
	{
		float pos = bbox.m_mn.x + (bbox.m_mx.x - bbox.m_mn.x) * rpos;
		plane = make_float4(-1.f, 0.f, 0.f, pos);
		areaAABBX(bbox, pos, areaLeft, areaRight);
	}
	else if(axis == 1)
	{
		float pos = bbox.m_mn.y + (bbox.m_mx.y - bbox.m_mn.y) * rpos;
		plane = make_float4(0.f, -1.f, 0.f, pos);
		areaAABBY(bbox, pos, areaLeft, areaRight);
	}
	else
	{
		float pos = bbox.m_mn.z + (bbox.m_mx.z - bbox.m_mn.z) * rpos;
		plane = make_float4(0.f, 0.f, -1.f, pos);
		areaAABBZ(bbox, pos, areaLeft, areaRight);
	}
#else		
	float lX = bbox.m_mx.x - bbox.m_mn.x;
	float lY = bbox.m_mx.y - bbox.m_mn.y;
	float lZ = bbox.m_mx.z - bbox.m_mn.z;

	float sumLengths = lX + lY + lZ;
	// Assign the planes to different methods
	int numX = lX/sumLengths*PLANE_COUNT+0.5f;
	int numY = lY/sumLengths*PLANE_COUNT+0.5f;
	int numZ = lZ/sumLengths*PLANE_COUNT+0.5f;

	//int axis = (planePos < numX) ? 0 : (planePos < numX+numY) ? 1 : 2;
	int axis = (planePos >= numX) + (planePos >= numX+numY);

	if(axis == 0)
	{
		float rpos = (float)(planePos+1) / (float)(numX+1);
		float pos = bbox.m_mn.x + lX * rpos;
		plane = make_float4(-1.f, 0.f, 0.f, pos);
		areaAABBX(bbox, pos, areaLeft, areaRight);
	}
	else if(axis == 1)
	{
		float rpos = (float)(planePos-numX+1) / (float)(numY+1);
		float pos = bbox.m_mn.y + lY * rpos;
		plane = make_float4(0.f, -1.f, 0.f, pos);
		areaAABBY(bbox, pos, areaLeft, areaRight);
	}
	else
	{
		float rpos = (float)(planePos-numX-numY+1) / (float)(numZ+1);
		float pos = bbox.m_mn.z + lZ * rpos;
		plane = make_float4(0.f, 0.f, -1.f, pos);
		areaAABBZ(bbox, pos, areaLeft, areaRight);
	}
#endif
}

//------------------------------------------------------------------------

// Compute a splitting plane for each thread
__device__ void findPlaneTriAABB(int planePos, float4* tris, int* trisIndex, int triStart, const volatile CudaAABB& bbox, float& areaLeft, float& areaRight, float4& plane, int numAxisAlignedPlanes)
{
	//ASSERT_DIVERGENCE("findPlaneTriAABB", threadIdx.x);

	int tri = planePos / 6;
	int axis = (planePos % 6) / 2;
	int lim = (planePos % 6) - axis;

	int triidx = trisIndex[triStart + tri]*3;

	// Fetch triangle
	float3 v0, v1, v2;
	taskFetchTri((CUdeviceptr)tris, triidx, v0, v1, v2);

	// Get bounding box
	CudaAABB tbox;
	getAABB(v0, v1, v0, tbox);

	if(axis == 0)
	{
		float pos;
		if(lim == 0)
			pos = tbox.m_mn.x;
		else
			pos = tbox.m_mx.x;
		plane = make_float4(-1.f, 0.f, 0.f, pos);
		areaAABBX(bbox, pos, areaLeft, areaRight);
	}
	else if(axis == 1)
	{
		float pos;
		if(lim == 0)
			pos = tbox.m_mn.y;
		else
			pos = tbox.m_mx.y;
		plane = make_float4(0.f, -1.f, 0.f, pos);
		areaAABBY(bbox, pos, areaLeft, areaRight);
	}
	else
	{
		float pos;
		if(lim == 0)
			pos = tbox.m_mn.z;
		else
			pos = tbox.m_mx.z;
		plane = make_float4(0.f, 0.f, -1.f, pos);
		areaAABBZ(bbox, pos, areaLeft, areaRight);
	}
}

//------------------------------------------------------------------------

// Compute a splitting plane for each thread based on triangle division
__device__ void findPlaneTriAA(int planePos, CUdeviceptr tris, CUdeviceptr trisIndex, int triStart, int triEnd, float4& plane, int numAxisAlignedPlanes)
{
	int planesPerAxis = ((numAxisAlignedPlanes+2) / 3);
	int axis = planePos / planesPerAxis;

	int triNum = triEnd - triStart;
	/*unsigned int hashA = planePos;
	unsigned int hashB = 0x9e3779b9u;
	unsigned int hashC = 0x9e3779b9u;
	jenkinsMix(hashA, hashB, hashC);
	jenkinsMix(hashA, hashB, hashC);
	int triidx = ((int*)trisIndex)[triStart + (hashC % triNum)]*3;*/
	float tpos = (float)(planePos % planesPerAxis)/(float)(planesPerAxis-1);
	int triidx = ((int*)trisIndex)[triStart + (int)(tpos * (triNum-1))]*3;

	// Fetch triangle
	float3 v0, v1, v2;
	taskFetchTri(tris, triidx, v0, v1, v2);
	// Compute triangle centroid
	CudaAABB tbox;
	float3 cent = getCentroid(v0, v1, v2, tbox);

	// Compute axis aligned plane through its centoid
	if(axis == 0)
	{
		plane = make_float4(-1.f, 0.f, 0.f, cent.x);
	}
	else if(axis == 1)
	{
		plane = make_float4(0.f, -1.f, 0.f, cent.y);
	}
	else
	{
		plane = make_float4(0.f, 0.f, -1.f, cent.z);
	}
}

//------------------------------------------------------------------------

// Compute a splitting plane for each thread based on triangle division
__device__ void findPlaneTri(int planePos, CUdeviceptr tris, CUdeviceptr trisIndex, int triStart, int triEnd, float4& plane)
{
	ASSERT_DIVERGENCE("findPlaneTri", threadIdx.x);

	int triNum = triEnd - triStart;
	unsigned int hashA = planePos;
	unsigned int hashB = 0x9e3779b9u;
	unsigned int hashC = 0x9e3779b9u;
	jenkinsMix(hashA, hashB, hashC);
	jenkinsMix(hashA, hashB, hashC);
	int triidx = ((int*)trisIndex)[triStart + (hashC % triNum)]*3;

	// Fetch triangle
	float3 v0, v1, v2;
	taskFetchTri(tris, triidx, v0, v1, v2);
	plane = set3PointPlane(v0, v1, v2);
}

//------------------------------------------------------------------------

// Compute a splitting plane for each thread based on ray division
__device__ void findPlaneRay(int planePos, CUdeviceptr rays, CUdeviceptr raysIndex, int rayStart, int rayEnd, float4& plane)
{
	ASSERT_DIVERGENCE("findPlaneRay", threadIdx.x);

	// BUG: Fails because of unclipped rays
	// Good strategy - only for primary rays
	// partitioning using an edge of random triangle and camera origin
	// RAY1 min / RAY1 max / RAY2 min
	int rayNum = rayEnd - rayStart;
	unsigned int hashA = planePos;
	unsigned int hashB = 0x9e3779b9u;
	unsigned int hashC = 0x9e3779b9u;
	jenkinsMix(hashA, hashB, hashC);
	jenkinsMix(hashA, hashB, hashC);
	int raypos1 = rayStart + (hashC % rayNum);
	int rayidx1 = ((int*)raysIndex)[raypos1];

	float3 orig, dir;
	float tmin, tmax;
	taskFetchRay(rays, rayidx1, orig, dir, tmin, tmax);

	float3 v0 = orig + tmin*dir;
	float3 v1 = orig + tmax*dir;

	int raypos2 = raypos1+1;
	if(raypos2 >= rayEnd)
		raypos2 = rayStart;
	int rayidx2 = ((int*)raysIndex)[raypos2];
	taskFetchRay(rays, rayidx2, orig, dir, tmin, tmax);

	float3 v2 = orig + tmax*dir;
	if(hashA & 0x1)
		v2 = v1 + cross(v1-v0, v2-v1);

	plane = set3PointPlane(v0, v1, v2);
}

//------------------------------------------------------------------------

// Compute a splitting plane for each thread
__device__ void findPlane(int planePos, CUdeviceptr rays, CUdeviceptr raysIndex, int rayStart, int rayEnd, CUdeviceptr tris, CUdeviceptr trisIndex, int triStart, int triEnd, const volatile CudaAABB& bbox, int numAxisAlignedPlanes, int numTriangleBasedPlanes, float4& plane)
{
	ASSERT_DIVERGENCE("findPlane", threadIdx.x);

	if(planePos < numAxisAlignedPlanes) // Choose axis aligned plane
	{
		findPlaneAABB(planePos, bbox, plane, numAxisAlignedPlanes);
	}
	else if(planePos < numAxisAlignedPlanes + numTriangleBasedPlanes) // Choose triangle based plane
	{
		findPlaneTri(planePos, tris, trisIndex, triStart, triEnd, plane);
	}
	else // Choose ray based plane
	{
		findPlaneRay(planePos, rays, raysIndex, rayStart, rayEnd, plane);
	}
}

//------------------------------------------------------------------------

// Compute a splitting plane for each thread in the chosen axis
__device__ void findPlaneRobin(int planePos, const volatile CudaAABB& bbox, int axis, float4& plane)
{
	ASSERT_DIVERGENCE("findPlaneRobin", threadIdx.x);

	float rpos = (float)(planePos+1) / (float)(WARP_SIZE+1);

	if(axis == 0)
	{
		float pos = bbox.m_mn.x + (bbox.m_mx.x - bbox.m_mn.x) * rpos;
		plane = make_float4(-1.f, 0.f, 0.f, pos);
	}
	else if(axis == 1)
	{
		float pos = bbox.m_mn.y + (bbox.m_mx.y - bbox.m_mn.y) * rpos;
		plane = make_float4(0.f, -1.f, 0.f, pos);
	}
	else
	{
		float pos = bbox.m_mn.z + (bbox.m_mx.z - bbox.m_mn.z) * rpos;
		plane = make_float4(0.f, 0.f, -1.f, pos);
	}
}

//------------------------------------------------------------------------

// Computes Woop triangle from a regular one
__device__ void calcWoop(float3& v0, float3& v1, float3& v2, float4& o0, float4& o1, float4& o2)
{
	// Compute woop
	float3 c0 = v0 - v2;
	float3 c1 = v1 - v2;
	float3 c2 = cross(c0,c1);
	
	// division by 0 ???
	float det = 1.0/(c0.x*(c2.z*c1.y-c1.z*c2.y) - c0.y*(c2.z*c1.x-c1.z*c2.x) + c0.z*(c2.y*c1.x-c1.y*c2.x));

	float3 i0,i1,i2;
	//i0 =
	i0.x =  (c2.z*c1.y-c1.z*c2.y)*det;
	i0.y = -(c2.z*c1.x-c1.z*c2.x)*det;
	i0.z =  (c2.y*c1.x-c1.y*c2.x)*det;
	
	//i1 =
	i1.x = -(c2.z*c0.y-c0.z*c2.y)*det;
	i1.y =  (c2.z*c0.x-c0.z*c2.x)*det;
	i1.z = -(c2.y*c0.x-c0.y*c2.x)*det;
	
	//i2 = 
	i2.x =  (c1.z*c0.y-c0.z*c1.y)*det;
	i2.y = -(c1.z*c0.x-c0.z*c1.x)*det;
	i2.z =  (c1.y*c0.x-c0.y*c1.x)*det;
	
	// Final values
	o0.x = i2.x;
	o0.y = i2.y;
	o0.z = i2.z;
	o0.w = -dot(-i2,v2);
	o1.x = i0.x;
	o1.y = i0.y;
	o1.z = i0.z;
	o1.w = dot(-i0,v2);
	o2.x = i1.x;
	o2.y = i1.y;
	o2.z = i1.z;
	o2.w = dot(-i1,v2);

	if (o0.x == 0.0f)
		o0.x = 0.0f;
}

//------------------------------------------------------------------------

// Creates a node in the compact layout
__device__ int createLeaf(int tid, int outOfs, float* outTriMem, int* outIdxMem, int start, int end, float* inTriMem, int* inIdxMem)
{
	// Compute output data pointers
	int numTris = end-start;
	float4 triData;
	int idxData;

	int* inIdx = inIdxMem + start; // Memory for the first triangle index

	float4* outTri = ((float4*)outTriMem) + outOfs; // Memory for the first triangle data
	int* outIdx = outIdxMem + outOfs; // Memory for the first triangle index


	// Write out all triangles and the triangle sentinel per vertex
	int numIters = taskWarpSubtasksZero(numTris*3+1); // Number of written out data chunks divided by WARP_SIZE
	for(int i = 0; i < numIters; i++)
	{
		int pos = i*WARP_SIZE + tid;
		int tri = pos/3;
		int item = pos % 3;

		if(tri < numTris) // Regular triangle
		{
			idxData = inIdx[tri];
			/*float4* inTri = ((float4*)inTriMem) + idxData*3; // Memory for the first triangle data
			//triData.x = inTri[item].x;
			//triData.y = inTri[item].y;
			//triData.z = inTri[item].z;
			//triData.w = inTri[item].w;
			triData = inTri[item];*/
			triData = tex1Dfetch(t_trisA, idxData*3 + item);
		}
		else // Sentinel
		{
			idxData = 0;
			triData = make_float4(__int_as_float(0x80000000));
		}

		// Write out the data
		if(tri < numTris || (tri == numTris && item == 0))
		{
			outTri[pos] = triData;
			outIdx[pos] = idxData;
		}
	}

	return ~outOfs;
}

//------------------------------------------------------------------------

// Creates a leaf in the compact layout, with Woop triangles
__device__ int createLeafWoop(int tid, int outOfs, float4* outTriMem, int* outIdxMem, int start, int end, float4* inTriMem, Reference* inRefMem)// int* inIdxMem)
{
	// Compute output data pointers
	int numTris = end-start;
	int idxData;

	Reference* inIdx = inRefMem;// + start; // Memory for the first triangle index

	float4* outTri = outTriMem + outOfs; // Memory for the first triangle data
	int* outIdx = outIdxMem + outOfs; // Memory for the first triangle index


	// Write out all triangles and the triangle sentinel per vertex
	int numIters = taskWarpSubtasksZero(numTris); // Number of written out data chunks divided by WARP_SIZE
	for(int i = 0; i < numIters; i++)
	{
		int tri = i*WARP_SIZE + tid;
		int pos = tri*3;

		if(tri < numTris) // Regular triangle
		{
			//idxData = inIdx[tri];
			idxData = inIdx[tri].idx;
			//printf("cwl: tri %i -> idxData %i (start offset = %i)\n", tri, idxData, start);
			float3 v0, v1, v2;
			float4 o0, o1, o2;
			taskFetchTri((CUdeviceptr)inTriMem, idxData*3, v0, v1, v2);
			//printf("cwl: fetchTri %i | %f %f %f | %f %f %f | %f %f %f\n", idxData, v0.x, v0.y, v0.z, v1.x, v1.y, v1.z, v2.x, v2.y, v2.z);

			calcWoop(v0, v1, v2, o0, o1, o2);

			outTri[pos+0] = o0;
			outTri[pos+1] = o1;
			outTri[pos+2] = o2;
			outIdx[pos] = idxData;
		}
	}

	if(tid == 0)
	{
		outTri[numTris*3].x = __int_as_float(0x80000000);
		outIdx[numTris*3] = 0;
	}

	return ~outOfs;
}

//------------------------------------------------------------------------

// Creates a leaf in the compact layout, with references to triangles
__device__ int createLeafReference(int tid, int outOfs, int* outIdxMem, int start, int end, int* inIdxMem)
{
	// Compute output data pointers
	int numTris = end-start;
	int idxData;

	int* inIdx = inIdxMem + start; // Memory for the first triangle index
	int* outIdx = outIdxMem + outOfs; // Memory for the first triangle index


	// Write out all triangles and the triangle sentinel per vertex
	int numIters = taskWarpSubtasksZero(numTris); // Number of written out data chunks divided by WARP_SIZE
	for(int i = 0; i < numIters; i++)
	{
		int tri = i*WARP_SIZE + tid;

		if(tri < numTris) // Regular triangle
		{
			idxData = inIdx[tri];
			outIdx[tri] = idxData;
		}
	}

	if(tid == 0)
	{
		outIdx[numTris] = 0x80000000;
	}

	return ~outOfs;
}

//------------------------------------------------------------------------

// Creates a leaf for a Kdtree, with Woop triangles
__device__ int createKdtreeLeafWoop(int tid, int outOfs, float4* outTriMem, int* outIdxMem, int start, int end, float4* inTriMem, int* inIdxMem)
{
	// Compute output data pointers
	int numTris = end-start;
	int idxData;

	int* inIdx = inIdxMem + start; // Memory for the first triangle index

	float4* outTri = outTriMem + outOfs; // Memory for the first triangle data
	int* outIdx = outIdxMem + outOfs; // Memory for the first triangle index


	// Write out all triangles and the triangle sentinel per vertex
	int numIters = taskWarpSubtasksZero(numTris); // Number of written out data chunks divided by WARP_SIZE
	for(int i = 0; i < numIters; i++)
	{
		int tri = i*WARP_SIZE + tid;
		int pos = tri*3;

		if(tri < numTris) // Regular triangle
		{
			idxData = inIdx[tri];
			float3 v0, v1, v2;
			float4 o0, o1, o2;
			taskFetchTri((CUdeviceptr)inTriMem, idxData*3, v0, v1, v2);

			calcWoop(v0, v1, v2, o0, o1, o2);

			outTri[pos+0] = o0;
			outTri[pos+1] = o1;
			outTri[pos+2] = o2;
			outIdx[pos] = idxData;
		}
	}

	return numTris | KDTREE_LEAF;
}

//------------------------------------------------------------------------

// Creates a leaf for a Kdtree, with Woop triangles
__device__ int createKdtreeInterleavedLeafWoop(int tid, int outOfs, char* outTriMem, int start, int end, float4* inTriMem, int* inIdxMem)
{
	// Compute output data pointers
	int numTris = end-start;
	int idxData;

	int* inIdx = inIdxMem + start; // Memory for the first triangle index

	float4* outTri = (float4*)(outTriMem + outOfs); // Memory for the first triangle data
	int* outIdx = (int*)(outTriMem + outOfs + numTris*3*sizeof(float4)); // Memory for the first triangle index


	// Write out all triangles and the triangle sentinel per vertex
	int numIters = taskWarpSubtasksZero(numTris); // Number of written out data chunks divided by WARP_SIZE
	for(int i = 0; i < numIters; i++)
	{
		int tri = i*WARP_SIZE + tid;
		int pos = tri*3;

		if(tri < numTris) // Regular triangle
		{
			idxData = inIdx[tri];
			float3 v0, v1, v2;
			float4 o0, o1, o2;
			taskFetchTri((CUdeviceptr)inTriMem, idxData*3, v0, v1, v2);

			calcWoop(v0, v1, v2, o0, o1, o2);

			outTri[pos+0] = o0;
			outTri[pos+1] = o1;
			outTri[pos+2] = o2;
			outIdx[tri] = idxData;
		}
	}

	return numTris | KDTREE_LEAF;
}

//------------------------------------------------------------------------

// Kernel converting regular triangles to Woop triangles
extern "C" __global__ void createWoop(CUdeviceptr tri, CUdeviceptr woop, int numTris)
{
	// Compute output data pointers
	int idx = blockDim.x * blockIdx.x + threadIdx.x; // 1D index

	if(idx < numTris)
	{
		float3 v0, v1, v2;
		float4 o0, o1, o2;
		taskFetchTri(tri, idx*3, v0, v1, v2);

		calcWoop(v0, v1, v2, o0, o1, o2);

		float4* woopData = (float4*)woop;
		woopData[idx*3+0] = o0;
		woopData[idx*3+1] = o1;
		woopData[idx*3+2] = o2;
	}
}

//------------------------------------------------------------------------

// Returns true if the node is a leaf
__device__ bool isKdLeaf(int flag)
{
#if defined(COMPACT_LAYOUT) && defined(WOOP_TRIANGLES)
	return flag < 0;
#else
	return flag & KDTREE_LEAF;
#endif
}

//------------------------------------------------------------------------
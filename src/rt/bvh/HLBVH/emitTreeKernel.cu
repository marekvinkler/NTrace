#include "emitTreeKernel.cuh"

#include <stdio.h>
//#include <cuda.h>
//#include <cutil_math.h>
#include "../../Util.hpp"

using namespace FW;

typedef SceneTriangle Triangle;

typedef union mint4 {
	int4 st;
	int ar[4];
} MINT4;

inline __device__ int3 ifloorf(float3 a) {
	return make_int3((int)::floorf(a.x),(int)::floorf(a.y),(int)::floorf(a.z));
}

inline __device__ float3 fminf(float3 a, float3 b) {
	return make_float3(::fminf(a.x,b.x), ::fminf(a.y,b.y),:: fminf(a.z,b.z));
}

inline __device__ float3 fmaxf(float3 a, float3 b) {
	return make_float3(::fmaxf(a.x,b.x), ::fmaxf(a.y,b.y), ::fmaxf(a.z,b.z));
}

inline __device__ int4 minmax(int4 a, int4 b) {
    return make_int4(__float_as_int(::min(__int_as_float(a.x),__int_as_float(b.x))), 
					 __float_as_int(::max(__int_as_float(a.y),__int_as_float(b.y))), 
					 __float_as_int(::min(__int_as_float(a.z),__int_as_float(b.z))), 
					 __float_as_int(::max(__int_as_float(a.w),__int_as_float(b.w))));
}

inline __device__ float4 minmax2(float4 a, float4 b) {
    return make_float4(::min(a.x,b.x), ::max(a.y,b.y), ::min(a.z,b.z), ::max(a.w,b.w));		
}

inline __device__ float fdot(float3 a, float3 b) {
	return a.x*b.x + a.y*b.y + a.z*b.z;
}

inline __device__ float3 fcross(float3 a, float3 b) {
	return make_float3(a.y*b.z - a.z*b.y, a.z*b.x - a.x*b.z, a.x*b.y - a.y*b.x);
}

inline __device__ float3 operator-(float3 &a) {
    return make_float3(-a.x, -a.y, -a.z);
}

inline __device__ float3 operator-(float3 a, float3 b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

inline __device__ float3 operator+(float3 a, float3 b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

inline __device__ float3 operator/(float3 a, float b) {
    return make_float3(a.x / b, a.y / b, a.z / b);
}

inline __device__ float3 operator/(float3 a, float3 b) {
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

inline __device__ uint3 floor(float3 a) {
    return make_uint3((U32)(::floor(a.x)), (U32)(::floor(a.y)), (U32)(::floor(a.z)));
}

inline __device__ int f2i( float floatVal ) {
	int intVal = __float_as_int( floatVal );
	return (intVal >= 0 ) ? intVal : intVal ^ 0x7FFFFFFF;
}

inline __device__ float i2f( int intVal ) {
	return __int_as_float( (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF );
}

inline __device__ int3 f3i3(float3 a) {
	//return make_int3(__float_as_int(a.x),__float_as_int(a.y),__float_as_int(a.z));
	return make_int3(f2i(a.x),f2i(a.y),f2i(a.z));
}

inline __device__ float3 i3f3(int3 a) {
	//return make_float3(__int_as_float(a.x),__int_as_float(a.y),__int_as_float(a.z));
	return make_float3(i2f(a.x),i2f(a.y),i2f(a.z));
}
/*
__device__ float atomicMin(float* address, float val) {
	float old = *address, assumed;
	if (old <= val) return old;
	do {
		assumed = old;
		old = atomicCAS((int*)address, __float_as_int(assumed), __float_as_int(val));
	} while (old != assumed);

	return old;
}

__device__ float atomicMax(float* address, float val) {
	float old = *address, assumed;
	if (old >= val) return old;
	do {
		assumed = old;
		old = atomicCAS((int*)address, __float_as_int(assumed), __float_as_int(val));
	} while (old != assumed);

	return old;
}
*/
__device__ F32 area(float3 v) {
	return (v.x*v.y + v.y*v.z + v.z*v.x)*2.0;
}

//      nodes[innerOfs + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
//      nodes[innerOfs + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
//      nodes[innerOfs + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
//      nodes[innerOfs + 48] = Vec4i(c0.innerOfs or ~c0.triOfs, c1.innerOfs or ~c1.triOfs, bitFlag, 0)

__device__ void inclusiveScan(volatile S32* childrenIdx, int tid) {
	if(tid >=  1)
		childrenIdx[1+tid] += childrenIdx[1+tid-1];
	if(tid >=  2)
		childrenIdx[1+tid] += childrenIdx[1+tid-2]; 
	if(tid >=  4)
		childrenIdx[1+tid] += childrenIdx[1+tid-4]; 
	if(tid >=  8)
		childrenIdx[1+tid] += childrenIdx[1+tid-8]; 
	if(tid >= 16)
		childrenIdx[1+tid] += childrenIdx[1+tid-16];
}

// Warp sized reduction with butterfly pattern
template<typename T>
__device__ __forceinline__ void reduceWarp(int tid, volatile T* data, T(*op)(T,T))
{
	data[tid] = op(data[tid], data[tid ^ 1]);
	data[tid] = op(data[tid], data[tid ^ 2]);
	data[tid] = op(data[tid], data[tid ^ 4]);
	data[tid] = op(data[tid], data[tid ^ 8]);
	data[tid] = op(data[tid], data[tid ^ 16]);
}

// Block sized reduction with butterfly pattern
template<typename T>
__device__ __forceinline__ void reduceBlock(int tid, volatile T* data, T(*op)(T,T))
{
	data[tid] = op(data[tid], data[tid ^ 1]);
	data[tid] = op(data[tid], data[tid ^ 2]);
	data[tid] = op(data[tid], data[tid ^ 4]);
	data[tid] = op(data[tid], data[tid ^ 8]);
	data[tid] = op(data[tid], data[tid ^ 16]);
	__syncthreads();
	data[tid] = op(data[tid], data[tid ^ 32]);
	__syncthreads();
	data[tid] = op(data[tid], data[tid ^ 64]);
	__syncthreads();
}

__device__ void calcWoop(S32 tid, int4* out);

extern "C" __device__ S32 createLeaf(S32 start, S32 end) {	// exclusive
	U32 numTris = end - start;

	U64 add = numTris;
	add <<= 32;
	add += 1;
	U64 leafPtr = atomicAdd(&g_leafsPtr, add);
	
#ifdef LEAF_HISTOGRAM
	atomicAdd(&g_leafHist[numTris], 1); // Update histogram
#endif
	
	U32 numLeafs = (leafPtr & 0xFFFFFFFF);
	U32 allTris = (leafPtr >> 32) & 0xFFFFFFFF;

#ifdef COMPACT_LAYOUT
	S32 outWoop = allTris * 3 + numLeafs; // Extra memory for triangle sentinel
	S32 outIdx = allTris * 3*sizeof(S32) + numLeafs * sizeof(S32); // Extra memory for triangle sentinel
#else
	S32 outWoop = allTris * 3;
	S32 outIdx = allTris * 3*sizeof(S32);
#endif

	int4* outWoopMem = g_outWoopMem + outWoop;
	int3* outIdxMem = (int3*)(g_outIdxMem + outIdx);
	
	//S32 outWoop = allTris * 3 + numLeafs;
	//S32 outIdx = allTris + numLeafs;

	//int4* outWoopMem = g_outWoopMem + outWoop;
	//int3* outIdxMem = g_outIdxMem + outIdx;

	for (S32 i = 0; i < numTris; i++) {
		//calcWoop(start+i, &outWoopMem[i*3]);
#ifdef WOOP_TRIANGLES
		// Copy woop triangles
		outWoopMem[i*3] = g_inWoopMem[g_inTriIdxMem[start+i]*3];
		outWoopMem[i*3+1] = g_inWoopMem[g_inTriIdxMem[start+i]*3+1];
		outWoopMem[i*3+2] = g_inWoopMem[g_inTriIdxMem[start+i]*3+2];
#else
		// Copy vertex data
		int3 vidx = *(int3*)(g_tris + sizeof(Triangle)*g_inTriIdxMem[start+i]);
		float3 v0 = g_verts[vidx.x];
		outWoopMem[i*3] = make_int4(__float_as_int(v0.x), __float_as_int(v0.y), __float_as_int(v0.z), 0);
		float3 v1 = g_verts[vidx.y];
		outWoopMem[i*3+1] = make_int4(__float_as_int(v1.x), __float_as_int(v1.y), __float_as_int(v1.z), 0);
		float3 v2 = g_verts[vidx.z];
		outWoopMem[i*3+2] = make_int4(__float_as_int(v2.x), __float_as_int(v2.y), __float_as_int(v2.z), 0);
#endif
		outIdxMem[i] = make_int3(g_inTriIdxMem[start+i],0,0);
	}

#ifdef COMPACT_LAYOUT
	// Write out triangle sentinel
	outWoopMem[numTris*3] = make_int4(0x80000000,0x80000000,0x80000000,0x80000000);
	outIdxMem[numTris].x = 0;

	return ~(allTris*3 + numLeafs);
#else
	return ~(allTris*3);
#endif
}

extern "C" __global__ void emitTreeKernel(S32 level, U32 nodeCnt, S32 inOfs) {
	int tid = threadIdx.x % WARP_SIZE;
	int wid = threadIdx.x / WARP_SIZE;
	int qid = threadIdx.x + blockIdx.x * blockDim.x;

	__shared__ S32 s_childrenIdx[BLOCK_SIZE+BLOCK_SIZE/WARP_SIZE];
	volatile S32* childrenIdx = s_childrenIdx + wid*(WARP_SIZE+1);
	childrenIdx[1+tid] = 0;

	// get WARP_SIZE nodes from queue
#if 0
	if (tid == 0)
		childrenIdx[0] = atomicAdd(&g_inQueuePtr, WARP_SIZE);

	int inOff = childrenIdx[0] + tid;
#else
	int inOff = qid;
#endif

	if (inOff >= nodeCnt) // spare threads
		return ;
	
	// read in 32 nodes
	S32 nIdx = g_inQueueMem[inOff*3];
	S32 nStart = g_inQueueMem[inOff*3+1]; // inclusive
	S32 nEnd = g_inQueueMem[inOff*3+2];   // exclusive	
	
	// find split
	S32 split = -1;
	int oldLevel = level;
	while(level >= 0 && (((g_inTriMem[nStart] >> level)&1) == ((g_inTriMem[nEnd-1] >> level)&1)))
		level--;

	//if(((g_inTriMem[nStart] >> level)&1) != ((g_inTriMem[nEnd-1] >> level)&1)) {
	if(level >= 0) {  // Split found
		S32 startBit = (g_inTriMem[nStart] >> level)&1;
		S32 a = nStart, b = nEnd;
		for (;;) {
			split = (a + b) >> 1;
			U32 splitBit = ((g_inTriMem[split] >> level)&1);
			if (((g_inTriMem[split-1] >> level)&1) != splitBit) {				
				break;
			} else {
				if (splitBit == startBit)
					a = split;
				else b = split;
			}
		}
	} else { // No spatial split exist on any level
		split = (nStart+nEnd) >> 1;
		//if(oldLevel < 1)
		//	printf("Object split level %d (%d / %d)\n", level, (split - nStart), (nEnd - split));
	}

	childrenIdx[1+tid] = 2;
#ifdef COMPACT_LAYOUT
	if ((split - nStart) <= c_leafSize || oldLevel == 0)
		childrenIdx[1+tid]--;
	if ((nEnd - split) <= c_leafSize || oldLevel == 0)
		childrenIdx[1+tid]--;
#endif

	// Exclusive scan the childrenIdx and childrenOfs
	inclusiveScan(childrenIdx, tid);
		
	// update output queue head
	if (tid == 0) {
		childrenIdx[0] = 0;
		S32 lid = (inOff+WARP_SIZE) <= nodeCnt ? WARP_SIZE : (nodeCnt%WARP_SIZE); // because not all threads participate in the scan
			
		childrenIdx[WARP_SIZE] = atomicAdd(&g_outQueuePtr, childrenIdx[lid]);
	}

	// all levels before + cur level size
	S32 outOff = childrenIdx[WARP_SIZE] + childrenIdx[tid]; // position in output queue
	S32 outIdx = inOfs + childrenIdx[WARP_SIZE] + childrenIdx[tid];

	// write output data
	/*if (split < 0) { // no split on current level
		split = (nStart+nEnd) >> 1;
		if ((nEnd - nStart) <= c_leafSize || level == 0) { // can ONLY happen on FIRST level in LBVH without SAH
#ifdef DOPRINTF
			if(split - nStart >= c_leafSize)
				printf("Left %d\n", split - nStart);
			if(nEnd - split >= c_leafSize)
				printf("Left %d\n", nEnd - split);
#endif
#ifdef COMPACT_LAYOUT
			((S32*)g_outNodes)[nIdx*16 + 12] = createLeaf(nStart, split);
			((S32*)g_outNodes)[nIdx*16 + 13] = createLeaf(split, nEnd);
			((S32*)g_outNodes)[nIdx*16 + 14] = level % 3;
			((S32*)g_outNodes)[nIdx*16 + 15] = 0;

			((S32*)g_outNodes)[nIdx*16 + 0] = nStart;
			((S32*)g_outNodes)[nIdx*16 + 1] = split;
			((S32*)g_outNodes)[nIdx*16 + 4] = split;
			((S32*)g_outNodes)[nIdx*16 + 5] = nEnd;
#else
			// TODO
#endif

			return ;
		}
	} else */{
		S32 c0, c1;
		if ((split - nStart) <= c_leafSize || oldLevel == 0) {// create left leaf
#ifdef DOPRINTF
			if(split - nStart > c_leafSize)
				printf("Left level %d oldLevel %d %d (%d)\n", level, oldLevel, split - nStart, nEnd - split);
#endif
			c0 = createLeaf(nStart, split);
			((S32*)g_outNodes)[nIdx*16 + 0] = nStart;
			((S32*)g_outNodes)[nIdx*16 + 1] = split;
		}
		else {		
			//if(oldLevel >= 25)
			//	printf("Left node level %d left %d right %d\n", level, split - nStart, nEnd - split);
			g_outQueueMem[outOff*3] = outIdx;
			g_outQueueMem[outOff*3+1] = nStart;
			g_outQueueMem[outOff*3+2] = split;
			c0 = outIdx * 64;
			outOff++;
			outIdx++;
		}

		if ((nEnd - split) <= c_leafSize || oldLevel == 0) {// create right leaf
#ifdef DOPRINTF
			if(nEnd - split > c_leafSize)
				printf("Right level %d oldLevel %d %d (%d)\n", level, oldLevel, nEnd - split, split - nStart);
#endif
			c1 = createLeaf(split, nEnd);
			((S32*)g_outNodes)[nIdx*16 + 4] = split;
			((S32*)g_outNodes)[nIdx*16 + 5] = nEnd;
		}
		else {
			//if(oldLevel >= 25)
			//	printf("Right node level %d left %d right %d\n", level, nEnd - split, split - nStart);
			g_outQueueMem[outOff*3] = outIdx;
			g_outQueueMem[outOff*3+1] = split;
			g_outQueueMem[outOff*3+2] = nEnd;
			c1 = outIdx * 64;
		}

		((S32*)g_outNodes)[nIdx*16 + 12] = c0;
		((S32*)g_outNodes)[nIdx*16 + 13] = c1;
		((S32*)g_outNodes)[nIdx*16 + 14] = level % 3;
		((S32*)g_outNodes)[nIdx*16 + 15] = 0;
	}
}

__device__ void calcLeaf(S32 start, S32 end, float3& lo, float3& hi) { // exclusive
	//int3 vidx;
	//float3 vpos;
#ifdef MEASURE_STATS
	atomicMax(&g_gd, end-start);
#endif

	for (S32 i = start; i < end; i++) {
		int3 vidx = *(int3*)(g_tris + sizeof(Triangle)*g_inTriIdxMem[i]);
		float3 a = g_verts[vidx.x];
		float3 b = g_verts[vidx.y];
		float3 c = g_verts[vidx.z];

		/*float3 lo0 = fminf(a, fminf(b,c)-make_float3(c_epsilon, c_epsilon, c_epsilon));
		float3 hi0 = fmaxf(a, fmaxf(b,c)+make_float3(c_epsilon, c_epsilon, c_epsilon));
		if(area(hi0-lo0) == 0.f)
			printf("Degenerate tri %f (%f, %f, %f) - (%f, %f, %f)!\n", c_epsilon, lo0.x, lo0.y, lo0.z, hi0.x, hi0.y, hi0.z);*/
		
		lo = fminf(lo, fminf(a, fminf(b,c))-make_float3(c_epsilon, c_epsilon, c_epsilon));
		hi = fmaxf(hi, fmaxf(a, fmaxf(b,c))+make_float3(c_epsilon, c_epsilon, c_epsilon));
		
		//vpos = g_verts[vidx.y];
		//lo = fminf(lo, b);
		//hi = fmaxf(hi, b);
	}
}

/*
	nodes[innerOfs + 0 ] = Vec4f(c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
	nodes[innerOfs + 16] = Vec4f(c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
	nodes[innerOfs + 32] = Vec4f(c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
	nodes[innerOfs + 48] = Vec4i(c0.innerOfs or ~c0.triOfs, c1.innerOfs or ~c1.triOfs, bitFlag, 0)
*/

extern "C" __global__ void calcAABB(S32 start, S32 cnt) {
	S32 qid = threadIdx.x + blockIdx.x * blockDim.x;

	if (qid >= cnt)
		return ;
		
	S32 nIdx = start + qid;
	//printf("cuda XX: %d,%d,%d\n", nIdx,start,qid);
	float4* node = (float4*)(g_outNodes + nIdx*64);

	S32 nl = *(S32*)(g_outNodes + nIdx*64 + 12*4);
	S32 nr = *(S32*)(g_outNodes + nIdx*64 + 13*4);

#ifdef MEASURE_STATS
	//printf("cuda START: %d,%d\n", nl,nr);

	atomicAdd(&g_ga, 1);
#endif

#if 0
	// left child
	if (nl < 0) {		
		float3 lo = make_float3(FW_F32_MAX, FW_F32_MAX, FW_F32_MAX);
		float3 hi = make_float3(-FW_F32_MAX, -FW_F32_MAX, -FW_F32_MAX);		

		S32 a = ((S32*)g_outNodes)[nIdx*16 + 0];
		S32 b = ((S32*)g_outNodes)[nIdx*16 + 1];

#ifdef MEASURE_STATS
		//printf("cuda cl1: %d -> %d\n", a,b);

		atomicAdd(&g_gb, 1);
		atomicAdd(&g_gc, b - a);
#endif

		calcLeaf(a, b, lo, hi);
				
		node[0] = make_float4(lo.x,hi.x,lo.y,hi.y);
		node[2].x = lo.z;
		node[2].y = hi.z;
	} else {
		//printf("cuda cl1A: %d\n", nl);
		float4* childNode = (float4*)(g_outNodes + nl);
		node[0] = ::minmax2(childNode[0],childNode[1]);
		node[2].x = ::min(childNode[2].x,childNode[2].z);
		node[2].y = ::max(childNode[2].y,childNode[2].w);
	}
	
	// right child
	if (nr < 0) {		
		float3 lo = make_float3(FW_F32_MAX, FW_F32_MAX, FW_F32_MAX);
		float3 hi = make_float3(-FW_F32_MAX, -FW_F32_MAX, -FW_F32_MAX);

		S32 a = ((S32*)g_outNodes)[nIdx*16 + 4];
		S32 b = ((S32*)g_outNodes)[nIdx*16 + 5];

#ifdef MEASURE_STATS
		//printf("cuda cl2: %d -> %d\n", a,b);
		atomicAdd(&g_gb, 1);	
		atomicAdd(&g_gc, b - a);
#endif

		calcLeaf(a, b, lo, hi);
		
		node[1] = make_float4(lo.x,hi.x,lo.y,hi.y);
		node[2].z = lo.z;
		node[2].w = hi.z;
	} else {
		//printf("cuda cl2A: %d\n", nr);
		float4* childNode = (float4*)(g_outNodes + nr);
		node[1] = ::minmax2(childNode[0],childNode[1]);
		node[2].z = ::min(childNode[2].x,childNode[2].z);
		node[2].w = ::max(childNode[2].y,childNode[2].w);
	}
#else
	float4 n0, n1, n2;
	// left child
	if (nl < 0) {		
		float3 lo = make_float3(FW_F32_MAX, FW_F32_MAX, FW_F32_MAX);
		float3 hi = make_float3(-FW_F32_MAX, -FW_F32_MAX, -FW_F32_MAX);		

		S32 a = ((S32*)g_outNodes)[nIdx*16 + 0];
		S32 b = ((S32*)g_outNodes)[nIdx*16 + 1];

#ifdef MEASURE_STATS
		//printf("cuda cl1: %d -> %d\n", a,b);

		atomicAdd(&g_gb, 1);
		atomicAdd(&g_gc, b - a);
#endif

		calcLeaf(a, b, lo, hi);
		/*if(area(hi-lo) == 0.f)
			printf("Degenerate box %d-%d(%f, %f, %f) - (%f, %f, %f)!\n", a, b, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);*/
		//printf("Leaf (%f, %f, %f) - (%f, %f, %f)\n", lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
				
		n0 = make_float4(lo.x,hi.x,lo.y,hi.y);
		n2.x = lo.z;
		n2.y = hi.z;
	} else {
		//printf("cuda cl1A: %d\n", nl);
		float4* childNode = (float4*)(g_outNodes + nl);
		n0 = ::minmax2(childNode[0],childNode[1]);
		n2.x = ::min(childNode[2].x,childNode[2].z);
		n2.y = ::max(childNode[2].y,childNode[2].w);
	}
	node[0] = n0; // Write out loX, hiX, loY, hiY for left child
	
	// right child
	if (nr < 0) {		
		float3 lo = make_float3(FW_F32_MAX, FW_F32_MAX, FW_F32_MAX);
		float3 hi = make_float3(-FW_F32_MAX, -FW_F32_MAX, -FW_F32_MAX);

		S32 a = ((S32*)g_outNodes)[nIdx*16 + 4];
		S32 b = ((S32*)g_outNodes)[nIdx*16 + 5];

#ifdef MEASURE_STATS
		//printf("cuda cl2: %d -> %d\n", a,b);
		atomicAdd(&g_gb, 1);	
		atomicAdd(&g_gc, b - a);
#endif

		calcLeaf(a, b, lo, hi);
		/*if(area(hi-lo) == 0.f)
			printf("Degenerate box %d-%d (%f, %f, %f) - (%f, %f, %f)!\n", a, b, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);*/
		//printf("Leaf (%f, %f, %f) - (%f, %f, %f)\n", lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
		
		n1 = make_float4(lo.x,hi.x,lo.y,hi.y);
		n2.z = lo.z;
		n2.w = hi.z;
	} else {
		//printf("cuda cl2A: %d\n", nr);
		float4* childNode = (float4*)(g_outNodes + nr);
		n1 = ::minmax2(childNode[0],childNode[1]);
		n2.z = ::min(childNode[2].x,childNode[2].z);
		n2.w = ::max(childNode[2].y,childNode[2].w);
	}
	/*float3 bboxLeft = make_float3(n0.y-n0.x, n0.w-n0.z, n2.y-n2.x);
	float3 bboxRight = make_float3(n1.y-n1.x, n1.w-n1.z, n2.w-n2.z);
	if(area(bboxLeft) == 0.f || area(bboxRight) == 0.f)
		printf("Degenerate box!\n");*/

	node[1] = n1; // Write out loX, hiX, loY, hiY for right child
	node[2] = n2; // Write out loZ, hiZ and loZ, hiZ for left and right child respectively
#endif
}

/*
inv(A) = [ inv(M)   -inv(M) * b ]
         [   0            1     ]
		 
| a11 a12 a13 |-1             |   a33a22-a32a23  -(a33a12-a32a13)   a23a12-a22a13  |
| a21 a22 a23 |    =  1/DET * | -(a33a21-a31a23)   a33a11-a31a13  -(a23a11-a21a13) |
| a31 a32 a33 |               |   a32a21-a31a22  -(a32a11-a31a12)   a22a11-a21a12  |

DET  =  a11(a33a22-a32a23)-a21(a33a12-a32a13)+a31(a23a12-a22a13)
*/
__device__ void calcWoop(S32 tid, int4* out) {
	//int3 vidx = *(int3*)(g_tris + sizeof(Scene::Triangle)*tid);
	//int3 vidx = *(int3*)(g_tris + sizeof(Triangle)*(  *(S32*)(g_inTriIdxMem + tid*4)  ));
	int3 vidx = *(int3*)(g_tris + sizeof(Triangle)*tid);

	float3 v0 = g_verts[vidx.x];
	float3 v1 = g_verts[vidx.y];
	float3 v2 = g_verts[vidx.z];
	
	float3 c0 = v0 - v2;
	float3 c1 = v1 - v2;
	float3 c2 = fcross(c0,c1);
	//float3 c3 = v2;
	
	// division by 0 ???
	float det = 1.0/(c0.x*(c2.z*c1.y-c1.z*c2.y) - c0.y*(c2.z*c1.x-c1.z*c2.x) + c0.z*(c2.y*c1.x-c1.y*c2.x));

	float3 i0,i1,i2;
	//c0 =
	i0.x =  (c2.z*c1.y-c1.z*c2.y)*det;
	i0.y = -(c2.z*c1.x-c1.z*c2.x)*det;
	i0.z =  (c2.y*c1.x-c1.y*c2.x)*det;
	
	//c1 =
	i1.x = -(c2.z*c0.y-c0.z*c2.y)*det;
	i1.y =  (c2.z*c0.x-c0.z*c2.x)*det;
	i1.z = -(c2.y*c0.x-c0.y*c2.x)*det;
	
	//c2 = 
	i2.x =  (c1.z*c0.y-c0.z*c1.y)*det;
	i2.y = -(c1.z*c0.x-c0.z*c1.x)*det;
	i2.z =  (c1.y*c0.x-c0.y*c1.x)*det;
	
	float4 o0,o1,o2;
	o0.x = i2.x;
	o0.y = i2.y;
	o0.z = i2.z;
	o0.w = -fdot(-i2,v2);
	o1.x = i0.x;
	o1.y = i0.y;
	o1.z = i0.z;
	o1.w = fdot(-i0,v2);
	o2.x = i1.x;
	o2.y = i1.y;
	o2.z = i1.z;
	o2.w = fdot(-i1,v2);

	if (o0.x == 0.0f)
		o0.x = 0.0f;
	
	out[0] = make_int4(__float_as_int(o0.x),__float_as_int(o0.y),__float_as_int(o0.z),__float_as_int(o0.w));
	out[1] = make_int4(__float_as_int(o1.x),__float_as_int(o1.y),__float_as_int(o1.z),__float_as_int(o1.w));
	out[2] = make_int4(__float_as_int(o2.x),__float_as_int(o2.y),__float_as_int(o2.z),__float_as_int(o2.w));

	//if (i2.x == 0.0f)
	//	i2.x = 0.0f;
	
	//out[0] = make_int4(__float_as_int(i2),__float_as_int(-fdot(-i2,v2)));
	//out[0] = make_int4(__float_as_int(i2.x),__float_as_int(i2.y),__float_as_int(i2.z),__float_as_int(-fdot(-i2,v2)));
	//out[1] = make_int4(__float_as_int(i0.x),__float_as_int(i0.y),__float_as_int(i0.z),__float_as_int( fdot(-i0,v2)));
	//out[2] = make_int4(__float_as_int(i1.x),__float_as_int(i1.y),__float_as_int(i1.z),__float_as_int( fdot(-i1,v2)));	
}

extern "C" __global__ void calcWoopKernel(U32 triCnt) {
	int tid = threadIdx.x + blockIdx.x*blockDim.x;

	if (tid >= triCnt)
		return ;
	
	//int4* out = (int4*)(g_inWoopMem + tid*3*sizeof(int4)); // Vec4i		
	calcWoop(tid, g_inWoopMem + tid*3);	
}

__device__ __inline__ U32 spread(U32 n) {
	n &= 0x3ff;
	n = (n ^ (n << 16)) & 0xff0000ff;
	n = (n ^ (n << 8)) & 0x0300f00f;
	n = (n ^ (n << 4)) & 0x030c30c3;
	return (n ^ (n << 2)) & 0x09249249;
}

extern "C" __global__ void calcMorton(
	U32 triCnt,
	F32 lo_x,
	F32 lo_y,
	F32 lo_z,
	F32 step_x,
	F32 step_y,
	F32 step_z) 
{
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	
	if (tid >= triCnt)
		return ;

	float3 lo = make_float3(lo_x, lo_y, lo_z);
	float3 step = make_float3(step_x, step_y, step_z);

	int3 vidx = *(int3*)(g_tris + sizeof(Triangle)*tid);
	float3 a = g_verts[vidx.x];
	float3 b = g_verts[vidx.y];
	float3 c = g_verts[vidx.z];

	//float3 bary = (a+b+c)/3.0f;
	//uint3 vtx = floor((bary - lo) / step);

	float3 abcLo = fminf(a,fminf(b,c));
	float3 abcHi = fmaxf(a,fmaxf(b,c));
	float3 aabb_mid = abcLo + (abcHi - abcLo)/2.0f;
	int3 vtx = ifloorf((aabb_mid - lo) / step);
	vtx.x = clamp(vtx.x, 0, 1024-1); // Expects n=10
	vtx.y = clamp(vtx.y, 0, 1024-1);
	vtx.z = clamp(vtx.z, 0, 1024-1);
	
	g_inTriMem[tid] = spread(vtx.x) | (spread(vtx.y) << 1) | (spread(vtx.z) << 2);

	g_inTriIdxMem[tid] = tid;
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// SAH
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

extern "C" __global__ void initBins(U32 qsiCnt) {
	int nid = threadIdx.x + blockIdx.x * blockDim.x;

	if (nid >= qsiCnt)
		return ;
	
	int3* aabb = (int3*)(g_binAABB + nid*sizeof(int3)*2); //x,y,z * AABB	
	S32* cnt = g_binCnt + nid;  //x,y,z * S32

	aabb[0] = make_int3(f2i(FW_F32_MAX), f2i(FW_F32_MAX), f2i(FW_F32_MAX));
	aabb[1] = make_int3(f2i(-FW_F32_MAX), f2i(-FW_F32_MAX), f2i(-FW_F32_MAX));
	*cnt = 0;
}

extern "C" __global__ void fillBins(U32 clsCnt) {
	int cid = threadIdx.x + blockIdx.x * blockDim.x;

	if (cid >= clsCnt)
		return ;	
		
	S32 node_id = g_clsSplitId[cid];	
	
	if (node_id < 0) // skip disabled clusters 
		return ;

	//float3 *cls_aabb = (float3*)(g_clsAABB + cid*sizeof(float3)*2);
#if CLUSTER_AABB <= 2
	float3 *cls_aabb = g_clsAABB + cid*2;	
	float3 cls_aabb_0 = cls_aabb[0];
	float3 cls_aabb_1 = cls_aabb[1];
#else
	int3 *cls_aabb = g_clsAABB + cid*2;	
	float3 cls_aabb_0 = i3f3(cls_aabb[0]);
	float3 cls_aabb_1 = i3f3(cls_aabb[1]);
	//printf("Cluster %d (%f, %f, %f) - (%f, %f, %f)\n", cid, cls_aabb_0.x, cls_aabb_0.y, cls_aabb_0.z, cls_aabb_1.x, cls_aabb_1.y, cls_aabb_1.z);
#endif
	float3 cls_mid = cls_aabb_0 + (cls_aabb_1 - cls_aabb_0)/2.0f; // FIX?

	int3* cls_bin = (int3*)(g_clsBinId + cid*3);
	
	float3* qsi_aabb = (float3*)(g_qsiAABB + node_id*sizeof(float3)*2);
	float3 step = (qsi_aabb[1] - qsi_aabb[0])/(1.f*BIN_CNT);
	
	//float3 *bin_aabb = (float3*)(g_binAABB + node_id*sizeof(float3)*2*BIN_CNT*3);
	int3* bin_aabb = (int3*)(g_binAABB + node_id*sizeof(int3)*2*BIN_CNT*3);
	S32* bin_cnt = g_binCnt + node_id*BIN_CNT*3;
	
	int3 bin_id = ifloorf((cls_mid-qsi_aabb[0])/step);
	bin_id.x = clamp(bin_id.x, 0, BIN_CNT-1);
	bin_id.y = clamp(bin_id.y, 0, BIN_CNT-1);
	bin_id.z = clamp(bin_id.z, 0, BIN_CNT-1);
	
	*cls_bin = bin_id;

	atomicMin(&bin_aabb[(BIN_CNT*0 + bin_id.x)*2].x, f2i(cls_aabb_0.x));
	atomicMin(&bin_aabb[(BIN_CNT*0 + bin_id.x)*2].y, f2i(cls_aabb_0.y));
	atomicMin(&bin_aabb[(BIN_CNT*0 + bin_id.x)*2].z, f2i(cls_aabb_0.z));
	atomicMax(&bin_aabb[(BIN_CNT*0 + bin_id.x)*2+1].x, f2i(cls_aabb_1.x));
	atomicMax(&bin_aabb[(BIN_CNT*0 + bin_id.x)*2+1].y, f2i(cls_aabb_1.y));
	atomicMax(&bin_aabb[(BIN_CNT*0 + bin_id.x)*2+1].z, f2i(cls_aabb_1.z));
	
	atomicMin(&bin_aabb[(BIN_CNT*1 + bin_id.y)*2].x, f2i(cls_aabb_0.x));
	atomicMin(&bin_aabb[(BIN_CNT*1 + bin_id.y)*2].y, f2i(cls_aabb_0.y));
	atomicMin(&bin_aabb[(BIN_CNT*1 + bin_id.y)*2].z, f2i(cls_aabb_0.z));
	atomicMax(&bin_aabb[(BIN_CNT*1 + bin_id.y)*2+1].x, f2i(cls_aabb_1.x));
	atomicMax(&bin_aabb[(BIN_CNT*1 + bin_id.y)*2+1].y, f2i(cls_aabb_1.y));
	atomicMax(&bin_aabb[(BIN_CNT*1 + bin_id.y)*2+1].z, f2i(cls_aabb_1.z));
	
	atomicMin(&bin_aabb[(BIN_CNT*2 + bin_id.z)*2].x, f2i(cls_aabb_0.x));
	atomicMin(&bin_aabb[(BIN_CNT*2 + bin_id.z)*2].y, f2i(cls_aabb_0.y));
	atomicMin(&bin_aabb[(BIN_CNT*2 + bin_id.z)*2].z, f2i(cls_aabb_0.z));
	atomicMax(&bin_aabb[(BIN_CNT*2 + bin_id.z)*2+1].x, f2i(cls_aabb_1.x));
	atomicMax(&bin_aabb[(BIN_CNT*2 + bin_id.z)*2+1].y, f2i(cls_aabb_1.y));
	atomicMax(&bin_aabb[(BIN_CNT*2 + bin_id.z)*2+1].z, f2i(cls_aabb_1.z));
	
	atomicAdd(&bin_cnt[BIN_CNT*0 + bin_id.x], 1);
	atomicAdd(&bin_cnt[BIN_CNT*1 + bin_id.y], 1);	
	atomicAdd(&bin_cnt[BIN_CNT*2 + bin_id.z], 1);	
}

extern "C" __global__ void findSplit(U32 qsiCnt, U32 inOfs) {
	S32 nid = threadIdx.x + blockIdx.x*blockDim.x;

	if (nid >= qsiCnt)
		return ;

	float3* qsi_aabb = (float3*)(g_qsiAABB + nid*sizeof(float3)*2);
	S32* bin_cnt = g_binCnt + nid*BIN_CNT*3;
	int3* bin_aabb = (int3*)(g_binAABB + nid*sizeof(int3)*2*BIN_CNT*3);		

	float3 mn[BIN_CNT-1],mx[BIN_CNT-1];
	S32 cnt[BIN_CNT-1];
		
	float3 mn_left, mx_left, mn_right, mx_right;
	S32 cnt_left, cnt_right, axis;
	
	F32 sah = FW_F32_MAX;

	S32 split = -1;
	for (S32 a = 0; a < 3; a++) {
		float3 mnr = make_float3(FW_F32_MAX,FW_F32_MAX,FW_F32_MAX);
		float3 mxr = make_float3(-FW_F32_MAX,-FW_F32_MAX,-FW_F32_MAX);
		S32 c = 0;
		for (S32 b = BIN_CNT-1; b > 0; b--) {
			mnr = fminf(mnr,i3f3(bin_aabb[(BIN_CNT*a+b)*2]));
			mxr = fmaxf(mxr,i3f3(bin_aabb[(BIN_CNT*a+b)*2+1]));
			mn[b-1] = mnr;
			mx[b-1] = mxr;
			c += bin_cnt[BIN_CNT*a+b];
			cnt[b-1] = c;
		}	
						
		float3 mnl = make_float3(FW_F32_MAX,FW_F32_MAX,FW_F32_MAX);
		float3 mxl = make_float3(-FW_F32_MAX,-FW_F32_MAX,-FW_F32_MAX);
		c = 0;
		for (S32 b = 0; b < BIN_CNT-1; b++) {
			mnl = fminf(mnl,i3f3(bin_aabb[(BIN_CNT*a+b)*2]));
			mxl = fmaxf(mxl,i3f3(bin_aabb[(BIN_CNT*a+b)*2+1]));
			c += bin_cnt[BIN_CNT*a+b];

			F32 s = c*area(mxl - mnl) + cnt[b]*area(mx[b] - mn[b]);
				
			if (s < sah) {
				sah = s;
				split = b;
				axis = a;
				cnt_left = c;
				cnt_right = cnt[b];

				mn_left = mnl;
				mx_left = mxl;
				mn_right = mn[b];
				mx_right = mx[b];
			}
		}
	}

	S32 qsi_id = g_qsiId[nid];
	S32 in_cnt = g_qsiCnt[nid];
	//S32 in_child = g_qsiChildId[nid];	

	if (split == -1) { // split missed		
		// Find the bin all clusters are in
		for (S32 i = 0; i < BIN_CNT; i++)
			if (bin_cnt[BIN_CNT*0 + i] != 0) {
				mn_left = mn_right = i3f3(bin_aabb[(BIN_CNT*0+i)*2]);
				mx_left = mn_right = i3f3(bin_aabb[(BIN_CNT*0+i)*2+1]);
				break;
			}

		//if(in_cnt > 1)
		//	printf("Too many clusters %d\n", in_cnt);

		// Split the clusters into halves
		cnt_right = in_cnt / 2;
		cnt_left = in_cnt - cnt_right;

		split = -cnt_left; // Mark for object split and save the number of left clusters
		axis = 0;
	}
#if 0
	else if (split != -1) { // split found, create 2 split tasks
		float3 step = (qsi_aabb[1] - qsi_aabb[0])/(1.f*BIN_CNT);
		

		mn_right = mn_left = qsi_aabb[0];
		mx_right = mx_left = qsi_aabb[1];

		if (axis == 0)
			mn_right.x = mx_left.x = mn_left.x + step.x*(split+1);
		else if (axis == 1)
			mn_right.y = mx_left.y = mn_left.y + step.y*(split+1);
		else
			mn_right.z = mx_left.z = mn_left.z + step.z*(split+1);
	}
#endif

	S32 nodes = 0;
	if (cnt_left > 1)
		nodes++;
	if (cnt_right > 1)
		nodes++;

#if 0
	U32 idx = atomicAdd(&g_sahCreated, nodes);
	U32 ofs = idx;
	idx += inOfs;
#else
	int tid = threadIdx.x % WARP_SIZE;
	int wid = threadIdx.x / WARP_SIZE;
	__shared__ S32 s_childrenIdx[BLOCK_SIZE+BLOCK_SIZE/WARP_SIZE];
	volatile S32* childrenIdx = s_childrenIdx + wid*(WARP_SIZE+1);
	childrenIdx[1+tid] = nodes;

	// Exclusive scan the childrenIdx and childrenOfs
	inclusiveScan(childrenIdx, tid);
		
	// update output queue head
	if (tid == 0) {
		childrenIdx[0] = 0;
		S32 lid = (nid+WARP_SIZE) <= qsiCnt ? WARP_SIZE : (qsiCnt%WARP_SIZE); // because not all threads participate in the scan
			
		childrenIdx[WARP_SIZE] = atomicAdd(&g_sahCreated, childrenIdx[lid]);
	}

	// all levels before + cur level size
	S32 ofs = childrenIdx[WARP_SIZE] + childrenIdx[tid]; // position in output queue
	S32 idx = inOfs + childrenIdx[WARP_SIZE] + childrenIdx[tid];
#endif

	float3* qso_aabb = (float3*)(g_qsoAABB + ofs*sizeof(float3)*2);

	S32 val = 0;
	S32 l = 0,r = 0;
	if (cnt_left > 1) {
		l = idx*64;
		g_qsoId[ofs] = idx;
		g_qsoCnt[ofs] = cnt_left;
		g_qsoChildId[ofs] = -1;
		qso_aabb[0] = mn_left;
		qso_aabb[1] = mx_left;		
		val++;
	}

	if (cnt_right > 1) {
		r = (idx+val)*64;
		g_qsoId[ofs+val] = idx+val;
		g_qsoCnt[ofs+val] = cnt_right;
		g_qsoChildId[ofs+val] = -1;
		qso_aabb[val*2] = mn_right;
		qso_aabb[val*2+1] = mx_right;		
	}

	g_qsiCnt[nid] = axis | ((((cnt_left <= 1) << 1) | (cnt_right <= 1)) << 2); // care with count of axis
	g_qsiPlane[nid] = split;
	g_qsiChildId[nid] = (S32)ofs;

	//if (cnt_left > 1 && cnt_right > 1)
	((int4*)g_outNodes)[qsi_id*4+3] = make_int4(l, r, axis, 0);
}

extern "C" __global__ void distribute(U32 clsCnt, S32 inOfs) {
	S32 cid = threadIdx.x + blockIdx.x * blockDim.x;

	if (cid >= clsCnt)
		return ;
	
	S32 old_id = g_clsSplitId[cid];	

	if (old_id < 0) // skip disabled clusters
		return ;

	S32 split_id = g_qsiPlane[old_id];
	S32 child_id = g_qsiChildId[old_id];
	S32 qsi_id = g_qsiId[old_id];
	
	S32 bin_id;
	S32 axis = g_qsiCnt[old_id] & 0xF;
	S32 leafs = axis >> 2;
	axis &= 0x3;

	S32 cnt = g_clsStart[cid+1] - g_clsStart[cid];

	if (split_id < 0) { // Object split
		split_id = (-split_id) - 1; // Because <= 0
		bin_id = (atomicAdd(g_qsiCnt + old_id, 1<<4)) >> 4; // Choose left or right child based on atomic order		
	} else { // split found				
		bin_id = g_clsBinId[cid*3 + axis];
	}

	if (leafs == 0) {
		g_clsSplitId[cid] = child_id + (bin_id <= split_id ? 0 : 1);
		/*if(bin_id <= split_id)
			printf("%d - %d left\n", g_clsStart[cid], g_clsStart[cid+1]);
		else
			printf("%d - %d right\n", g_clsStart[cid], g_clsStart[cid+1]);*/
	} else { // terminate

		if (bin_id <= split_id) { // left
			//printf("%d - %d left\n", g_clsStart[cid], g_clsStart[cid+1]);
			if (leafs & 2) { // only one cluster left of the splitting plane (this cluster)
				if (cnt <= c_leafSize) { // create leaf in the final hierarchy
					((int4*)g_outNodes)[qsi_id*4+3].x = createLeaf(g_clsStart[cid], g_clsStart[cid+1]);

					((int4*)g_outNodes)[qsi_id*4+0].x = g_clsStart[cid];
					((int4*)g_outNodes)[qsi_id*4+0].y = g_clsStart[cid+1];
					//printf("%d ---> %d [%d] %d\n", qsi_id, ((int4*)g_outNodes)[qsi_id*4+3].x, leafs, axis);
				} else { // create node for LBVH
					U32 idx = atomicAdd(&g_sahCreated, 1);

					U32 oofs = atomicAdd(&g_oofs, 1);

					g_ooq[oofs*3+0] = inOfs+idx;
					g_ooq[oofs*3+1] = g_clsStart[cid];
					g_ooq[oofs*3+2] = g_clsStart[cid+1];

					((int4*)g_outNodes)[qsi_id*4+3].x = (inOfs+idx)*64;
					//printf("%d -> %d\n", qsi_id, ((int4*)g_outNodes)[qsi_id*4+3].x);
				}
				g_clsSplitId[cid] = -1;
			} else g_clsSplitId[cid] = child_id;
		}			
		else {//if (bin_id > split_id) { // right
			//printf("%d - %d right\n", g_clsStart[cid], g_clsStart[cid+1]);
			if (leafs & 1) { // only one cluster right of the splitting plane (this cluster)
				if (cnt <= c_leafSize) { // create leaf in the final hierarchy
					((int4*)g_outNodes)[qsi_id*4+3].y = createLeaf(g_clsStart[cid], g_clsStart[cid+1]);

					((int4*)g_outNodes)[qsi_id*4+1].x = g_clsStart[cid];
					((int4*)g_outNodes)[qsi_id*4+1].y = g_clsStart[cid+1];
					//printf("%d ---> %d [%d] %d\n", qsi_id, ((int4*)g_outNodes)[qsi_id*4+3].y, leafs, axis);
				} else { // create node for LBVH
					U32 idx = atomicAdd(&g_sahCreated, 1);

					U32 oofs = atomicAdd(&g_oofs, 1);

					g_ooq[oofs*3+0] = inOfs+idx;
					g_ooq[oofs*3+1] = g_clsStart[cid];
					g_ooq[oofs*3+2] = g_clsStart[cid+1];						

					((int4*)g_outNodes)[qsi_id*4+3].y = (inOfs+idx)*64;
					//printf("%d -> %d\n", qsi_id, ((int4*)g_outNodes)[qsi_id*4+3].y);
				}
				g_clsSplitId[cid] = -1;
			} else g_clsSplitId[cid] = child_id;
		}			

	}
}

__device__ void reduceBlockAABB(int tid, volatile float* s_sharedData, int i, int cStart, int cEnd, float3& lo, float3& hi)
{
	// Reduce minimum
	if(cStart <= i && i < cEnd)
		s_sharedData[tid] = lo.x;
	else
		s_sharedData[tid] = FW_F32_MAX;
	reduceBlock<float>(tid, s_sharedData, fminf);
	lo.x = s_sharedData[tid];

	if(cStart <= i && i < cEnd)
		s_sharedData[tid] = lo.y;
	else
		s_sharedData[tid] = FW_F32_MAX;
	reduceBlock<float>(tid, s_sharedData, fminf);
	lo.y = s_sharedData[tid];

	if(cStart <= i && i < cEnd)
		s_sharedData[tid] = lo.z;
	else
		s_sharedData[tid] = FW_F32_MAX;
	reduceBlock<float>(tid, s_sharedData, fminf);
	lo.z = s_sharedData[tid];

	// Reduce maximum
	if(cStart <= i && i < cEnd)
		s_sharedData[tid] = hi.x;
	else
		s_sharedData[tid] = -FW_F32_MAX;
	reduceBlock<float>(tid, s_sharedData, fmaxf);
	hi.x = s_sharedData[tid];

	if(cStart <= i && i < cEnd)
		s_sharedData[tid] = hi.y;
	else
		s_sharedData[tid] = -FW_F32_MAX;
	reduceBlock<float>(tid, s_sharedData, fmaxf);
	hi.y = s_sharedData[tid];

	if(cStart <= i && i < cEnd)
		s_sharedData[tid] = hi.z;
	else
		s_sharedData[tid] = -FW_F32_MAX;
	reduceBlock<float>(tid, s_sharedData, fmaxf);
	hi.z = s_sharedData[tid];
}

#if CLUSTER_AABB >= 3
extern "C" __global__ void initClusterAABB(S32 cnt) {
	int nid = threadIdx.x + blockIdx.x * blockDim.x;

	if (nid >= cnt)
		return ;
	
	int3* aabb = g_clsAABB + nid*2; //x,y,z * AABB	

	aabb[0] = make_int3(f2i(FW_F32_MAX), f2i(FW_F32_MAX), f2i(FW_F32_MAX));
	aabb[1] = make_int3(f2i(-FW_F32_MAX), f2i(-FW_F32_MAX), f2i(-FW_F32_MAX));
}
#endif

extern "C" __global__ void clusterAABB(S32 cnt, S32 tris) {
#if CLUSTER_AABB == 0
	S32 tid = threadIdx.x + blockDim.x * blockIdx.x;
	
	if (tid >= cnt)
		return ;
		
	S32 start = g_clsStart[tid];
	S32 end = g_clsStart[tid+1];

	float3 lo = make_float3(FW_F32_MAX,FW_F32_MAX,FW_F32_MAX);
	float3 hi = make_float3(-FW_F32_MAX,-FW_F32_MAX,-FW_F32_MAX);

	for (S32 i = start; i < end; i++) {
		int3 vidx = *(int3*)(g_tris + sizeof(Triangle)*g_inTriIdxMem[i]);
		float3 a = g_verts[vidx.x];
		float3 b = g_verts[vidx.y];
		float3 c = g_verts[vidx.z];
		
		lo = fminf(lo, fminf(a, fminf(b,c))/*-make_float3(c_epsilon, c_epsilon, c_epsilon)*/);
		hi = fmaxf(hi, fmaxf(a, fmaxf(b,c))/*+make_float3(c_epsilon, c_epsilon, c_epsilon)*/);
	}

	//printf("Cluster %d (%f, %f, %f) - (%f, %f, %f)\n", tid, lo.x, lo.y, lo.z, hi.x, hi.y, hi.z);
	g_clsAABB[tid*2] = lo;
	g_clsAABB[tid*2+1] = hi;
#elif CLUSTER_AABB == 1
	__shared__ volatile float s_sharedData[BLOCK_SIZE/WARP_SIZE][WARP_SIZE]; // Shared memory for inside warp use
	int tid = threadIdx.x;
	int wid = threadIdx.y;
	int cid = threadIdx.y + blockDim.y * blockIdx.x;
	
	if (cid >= cnt)
		return ;
		
	int start = g_clsStart[cid];
	int end = g_clsStart[cid+1];

	float3 lo = make_float3(FW_F32_MAX,FW_F32_MAX,FW_F32_MAX);
	float3 hi = make_float3(-FW_F32_MAX,-FW_F32_MAX,-FW_F32_MAX);

	// Each thread processes its part of the array
	for (int i = start+tid; i < end; i+=WARP_SIZE) {
		int3 vidx = *(int3*)(g_tris + sizeof(Triangle)*g_inTriIdxMem[i]);
		float3 a = g_verts[vidx.x];
		float3 b = g_verts[vidx.y];
		float3 c = g_verts[vidx.z];
		
		lo = fminf(lo, fminf(a, fminf(b,c))/*-make_float3(c_epsilon, c_epsilon, c_epsilon)*/);
		hi = fmaxf(hi, fmaxf(a, fmaxf(b,c))/*+make_float3(c_epsilon, c_epsilon, c_epsilon)*/);
	}

	// The bounding boxes are reduced and written out
	// Reduce minimum
	s_sharedData[wid][tid] = lo.x;
	reduceWarp<float>(tid, s_sharedData[wid], fminf);
	lo.x = s_sharedData[wid][tid];

	s_sharedData[wid][tid] = lo.y;
	reduceWarp<float>(tid, s_sharedData[wid], fminf);
	lo.y = s_sharedData[wid][tid];

	s_sharedData[wid][tid] = lo.z;
	reduceWarp<float>(tid, s_sharedData[wid], fminf);
	lo.z = s_sharedData[wid][tid];

	// Reduce maximum
	s_sharedData[wid][tid] = hi.x;
	reduceWarp<float>(tid, s_sharedData[wid], fmaxf);
	hi.x = s_sharedData[wid][tid];

	s_sharedData[wid][tid] = hi.y;
	reduceWarp<float>(tid, s_sharedData[wid], fmaxf);
	hi.y = s_sharedData[wid][tid];

	s_sharedData[wid][tid] = hi.z;
	reduceWarp<float>(tid, s_sharedData[wid], fmaxf);
	hi.z = s_sharedData[wid][tid];

	g_clsAABB[cid*2] = lo;
	g_clsAABB[cid*2+1] = hi;
#elif CLUSTER_AABB == 2
	__shared__ volatile float s_sharedData[BLOCK_SIZE]; // Shared memory for inside warp use
	int tid = threadIdx.x;
	int cid = blockIdx.x;
	
	if (cid >= cnt)
		return ;
		
	int start = g_clsStart[cid];
	int end = g_clsStart[cid+1];

	float3 lo = make_float3(FW_F32_MAX,FW_F32_MAX,FW_F32_MAX);
	float3 hi = make_float3(-FW_F32_MAX,-FW_F32_MAX,-FW_F32_MAX);

	// Each thread processes its part of the array
	for (int i = start+tid; i < end; i+=BLOCK_SIZE) {
		int3 vidx = *(int3*)(g_tris + sizeof(Triangle)*g_inTriIdxMem[i]);
		float3 a = g_verts[vidx.x];
		float3 b = g_verts[vidx.y];
		float3 c = g_verts[vidx.z];
		
		lo = fminf(lo, fminf(a, fminf(b,c))/*-make_float3(c_epsilon, c_epsilon, c_epsilon)*/);
		hi = fmaxf(hi, fmaxf(a, fmaxf(b,c))/*+make_float3(c_epsilon, c_epsilon, c_epsilon)*/);
	}

	// The bounding boxes are reduced and written out
	// Reduce minimum
	s_sharedData[tid] = lo.x;
	reduceBlock<float>(tid, s_sharedData, fminf);
	lo.x = s_sharedData[tid];

	s_sharedData[tid] = lo.y;
	reduceBlock<float>(tid, s_sharedData, fminf);
	lo.y = s_sharedData[tid];

	s_sharedData[tid] = lo.z;
	reduceBlock<float>(tid, s_sharedData, fminf);
	lo.z = s_sharedData[tid];

	// Reduce maximum
	s_sharedData[tid] = hi.x;
	reduceBlock<float>(tid, s_sharedData, fmaxf);
	hi.x = s_sharedData[tid];

	s_sharedData[tid] = hi.y;
	reduceBlock<float>(tid, s_sharedData, fmaxf);
	hi.y = s_sharedData[tid];

	s_sharedData[tid] = hi.z;
	reduceBlock<float>(tid, s_sharedData, fmaxf);
	hi.z = s_sharedData[tid];

	g_clsAABB[cid*2] = lo;
	g_clsAABB[cid*2+1] = hi;
#elif CLUSTER_AABB == 3
	__shared__ volatile float s_sharedData[BLOCK_SIZE]; // Shared memory for inside warp use
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	// Compute the triangle interval to process
	int dataBlocks = (tris + (BLOCK_SIZE-1)) / BLOCK_SIZE;
	int processedBlocks = (dataBlocks+(NUM_BLOCKS-1))/NUM_BLOCKS;
	int triStart = bid * processedBlocks * BLOCK_SIZE;
	int triEnd = min((bid+1) * processedBlocks * BLOCK_SIZE, dataBlocks*BLOCK_SIZE);

	if(triStart >= tris)
		return;

	// Find the starting cluster id with binary search
	int a = 0;
	int b = cnt;
	int s;
	int cStart;
	int cEnd;

	while(true)
	{
		s = (a+b)/2;
		cStart = g_clsStart[s];
		cEnd = g_clsStart[s+1];

		if(cStart <= triStart && triStart < cEnd)
		{
			break; // Starting cluster found
		}
		else
		{
			if(triStart < cStart)
				b = s;
			else
				a = s;
		}
	}
	
	int cid = s;

	float3 lo = make_float3(FW_F32_MAX,FW_F32_MAX,FW_F32_MAX);
	float3 hi = make_float3(-FW_F32_MAX,-FW_F32_MAX,-FW_F32_MAX);

	// Each thread processes its part of the array
	for (int i = triStart+tid; i < triEnd; i+=BLOCK_SIZE) {

		float3 a, b, c;
		if(i < tris)
		{
			int3 vidx = *(int3*)(g_tris + sizeof(Triangle)*g_inTriIdxMem[i]);
			a = g_verts[vidx.x];
			b = g_verts[vidx.y];
			c = g_verts[vidx.z];


			if(cStart <= i && i < cEnd) // Update box only when inside the current cluster
			{
				lo = fminf(lo, fminf(a, fminf(b,c))/*-make_float3(c_epsilon, c_epsilon, c_epsilon)*/);
				hi = fmaxf(hi, fmaxf(a, fmaxf(b,c))/*+make_float3(c_epsilon, c_epsilon, c_epsilon)*/);
			}
		}

		// Write out all clusters that ended in this interval
		int lastBlockTri = i + ((BLOCK_SIZE-1)-tid);
		bool isLast = lastBlockTri + BLOCK_SIZE >= triEnd;
		while(lastBlockTri >= cEnd || isLast) // When cluster processed -> update its bounds
		{
			int testI = (i < cEnd) ? i : i-BLOCK_SIZE; // Position of the data stored in lo, high for current cluster
			reduceBlockAABB(tid, s_sharedData, testI, cStart, cEnd, lo, hi); // Reduce the bounds
			if(tid == 0)
			{
				if(triStart <= cStart && cEnd <= triEnd) // When entire cluster computed in this block
				{
					//printf("%d Setting cluster %d [%d, %d] with values [%d, %d]\n", bid, cid, cStart, cEnd, max(triStart, cStart), min(triEnd, cEnd));
					g_clsAABB[cid*2] = f3i3(lo);
					g_clsAABB[cid*2+1] = f3i3(hi);
				}
				else // When only part of the cluster computed in this block
				{
					//printf("%d Updating cluster %d [%d, %d] with values [%d, %d]\n", bid, cid, cStart, cEnd, max(triStart, cStart), min(triEnd, cEnd));
					int3 loI = f3i3(lo);
					atomicMin(&g_clsAABB[cid*2].x, loI.x);
					atomicMin(&g_clsAABB[cid*2].y, loI.y);
					atomicMin(&g_clsAABB[cid*2].z, loI.z);

					int3 hiI = f3i3(hi);
					atomicMax(&g_clsAABB[cid*2+1].x, hiI.x);
					atomicMax(&g_clsAABB[cid*2+1].y, hiI.y);
					atomicMax(&g_clsAABB[cid*2+1].z, hiI.z);
				}
			}

			// Prepare for next cluster

			// Set boxes
			if(i < cEnd || i >= tris)
			{
				lo = make_float3(FW_F32_MAX,FW_F32_MAX,FW_F32_MAX);
				hi = make_float3(-FW_F32_MAX,-FW_F32_MAX,-FW_F32_MAX);
			}
			else /*if(i < tris)*/ // Prevent repeated computation when multiple clusters end
			{
				lo = fminf(a, fminf(b,c))/*-make_float3(c_epsilon, c_epsilon, c_epsilon)*/;
				hi = fmaxf(a, fmaxf(b,c))/*+make_float3(c_epsilon, c_epsilon, c_epsilon)*/;
			}

			// Move cluster bounds
			cid++;
			if(cid < cnt)
			{
				isLast = isLast && cEnd <= lastBlockTri && lastBlockTri < tris; // There are some unprocessed triangles
				cStart = cEnd;
				cEnd = g_clsStart[cid+1];
			}
			else
			{
				break;
			}
		}
	}
#endif
}

__device__ S32 calcLeafs(S32 n) {
	int4* woop = g_outWoopMem + (~n);

	S32 cnt = 0;
	while (woop[cnt*3].x != 0x80000000)
		cnt++;

	return cnt;
}

__device__ F32 calcSAHNode(S32 n) {
	float4* node = (float4*)(g_outNodes + n*64);

	S32 nl = *(S32*)(g_outNodes + n*64 + 12*4);
	S32 nr = *(S32*)(g_outNodes + n*64 + 13*4);	

	F32 xi = fminf(node[0].x,node[1].x);
	F32 xa = fmaxf(node[0].y,node[1].y);
	F32 yi = fminf(node[0].z,node[1].z);
	F32 ya = fmaxf(node[0].w,node[1].w);
	F32 zi = fminf(node[2].x,node[2].z);
	F32 za = fmaxf(node[2].y,node[2].w);

	F32 pa = 2*((xa-xi)*(ya-yi) + (ya-yi)*(za-zi) + (za-zi)*(xa-xi));
	F32 pl = 2*((node[0].y-node[0].x)*(node[0].w-node[0].z) + (node[0].w-node[0].z)*(node[2].y-node[2].x) + (node[2].y-node[2].x)*(node[0].y-node[0].x));
	F32 pr = 2*((node[1].y-node[1].x)*(node[1].w-node[1].z) + (node[1].w-node[1].z)*(node[2].w-node[2].z) + (node[2].w-node[2].z)*(node[1].y-node[1].x));

	F32 sah_left, sah_right;

	if (nl < 0) {		
		S32 nleafs = calcLeafs(nl);
		sah_left = nleafs;
	} else sah_left = calcSAHNode(nl/64);

	if (nr < 0) {		
		S32 nleafs = calcLeafs(nr);
		sah_right = nleafs;
	} else sah_right = calcSAHNode(nr/64);
	
	return 1 + (pl/pa) * sah_left + (pr/pa) * sah_right;
}

extern "C" __global__ void calcSAH() {
	S32 tid = threadIdx.x;

	if (tid != 0)
		return ;

	g_sahCost = calcSAHNode(0);
}

extern "C" __global__ void test() {

}
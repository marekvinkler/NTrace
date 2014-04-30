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
    GF100-optimized variant of the "Speculative while-while"
    kernel used in:

    "Understanding the Efficiency of Ray Traversal on GPUs",
    Timo Aila and Samuli Laine,
    Proc. High-Performance Graphics 2009
*/

#include "CudaTracerKernels.hpp"
#include <stdio.h>
#include <math_constants.h>

//------------------------------------------------------------------------

#define STACK_SIZE  128  // Size of the traversal stack in local memory.
//#define BRANCHLESS
#define SHORTSTACK 0
//#define SPECULATIVE
//#define ONDEMAND_LIKE

#if SHORTSTACK == 3
#define SHORTSTACK_SIZE 3
__shared__ int2 s_stack[128*SHORTSTACK_SIZE]; // Short stack for each thread
#endif

#ifdef ONDEMAND_LIKE
__shared__ int s_limit[320];
#endif

//------------------------------------------------------------------------

extern "C" __global__ void queryConfig(void)
{
    g_config.bvhLayout = BVHLayout_Compact;
	//g_config.bvhLayout = BVHLayout_Compact | BVHLayout_ShadowCompact;
    g_config.blockWidth = 32; // One warp per row.
    g_config.blockHeight = 4; // 4*32 = 128 threads, optimal for GTX480
}

//------------------------------------------------------------------------

#if SHORTSTACK == 0
__device__ __forceinline__ void pushNode(char*& stackPtr, int& second, float& tmin, float& tmax)
{
	stackPtr += 8;
	*(int2*)stackPtr = make_int2(second, __float_as_int(tmax));
	/*stackPtr += 8;
	unsigned long long ft = (unsigned long long)__float_as_int(tmax);
	unsigned long long nd = (unsigned long long)second & 0xFFFFFFFFULL;
	*(unsigned long long*)stackPtr = nd | (ft << 32);*/
}

#elif SHORTSTACK == 1
__device__ __forceinline__ void pushNode(char*& stackPtr, int& second, float& tmin, float& tmax, int& nodeAddr1, float& tmax1)
{
	if(nodeAddr1 == 0)
	{
		nodeAddr1 = second;
		tmax1 = tmax;
	}
	else
	{
		stackPtr += 8;
		*(int2*)stackPtr = make_int2(nodeAddr1, __float_as_int(tmax1));
		nodeAddr1 = second;
		tmax1 = tmax;
	}
}

#elif SHORTSTACK == 2
__device__ __forceinline__ void pushNode(char*& stackPtr, int& second, float& tmin, float& tmax, int& nodeAddr1, float& tmax1, int& nodeAddr2, float& tmax2)
{
	if(nodeAddr1 == 0)
	{
		nodeAddr1 = second;
		tmax1 = tmax;
	}
	else
	{
		if(nodeAddr2 == 0)
		{
			nodeAddr2 = nodeAddr1;
			tmax2 = tmax1;
			nodeAddr1 = second;
			tmax1 = tmax;
		}
		else
		{
			stackPtr += 8;
			*(int2*)stackPtr = make_int2(nodeAddr2, __float_as_int(tmax2));
			nodeAddr2 = nodeAddr1;
			tmax2 = tmax1;
			nodeAddr1 = second;
			tmax1 = tmax;
		}
	}
}

#elif SHORTSTACK == 3
__device__ __forceinline__ void pushNode(char*& stackPtr, int& shortStack, int& second, float& tmin, float& tmax)
{
	int head = shortStack & 0xF;
	int tail = shortStack >> 4;
	int newHead = ((head+1) % SHORTSTACK_SIZE);
	int newTail = ((tail+1) % SHORTSTACK_SIZE);
	int2& pushVal = s_stack[(threadIdx.y*blockDim.x + threadIdx.x)*SHORTSTACK_SIZE + newHead];

	if(newHead == tail) // Write to global stack
	{
		stackPtr += 8;
		*(int2*)stackPtr = pushVal;
		tail = newTail;
	}

	pushVal = make_int2(second, __float_as_int(tmax));
	head = newHead;
	shortStack = (tail << 4) | head;
}
#endif

//------------------------------------------------------------------------

#if defined(BRANCHLESS) || SHORTSTACK == 0
__device__ __forceinline__ void popNode(char*& stackPtr, int& nodeAddr, float& tmin, float& tmax)
{
#ifndef BRANCHLESS
	tmin = tmax;

	int2 item = *(int2*)stackPtr;
	stackPtr -= 8;
	nodeAddr = item.x;
	tmax = __int_as_float(item.y);
	/*unsigned long long item = *(unsigned long long*)stackPtr;
	stackPtr -= 8;
	nodeAddr = item & 0xFFFFFFFFULL;
	int ft = item >> 32;
	tmax = __int_as_float(ft);*/

#else
	int4 item = *(int4*)stackPtr;
	stackPtr -= 16;
	nodeAddr = item.x;
	tmin = __int_as_float(item.y);
	tmax = __int_as_float(item.z);
#endif
}

#elif SHORTSTACK == 1
__device__ void popNode(char*& stackPtr, int& nodeAddr, float& tmin, float& tmax, int& nodeAddr1, float& tmax1)
{
	tmin = tmax;

	if(nodeAddr1 == 0)
	{
		int2 item = *(int2*)stackPtr;
		stackPtr -= 8;
		nodeAddr = item.x;
		tmax = __int_as_float(item.y);
	}
	else
	{
		nodeAddr = nodeAddr1;
		tmax = tmax1;

		/*if(nodeAddr1 != EntrypointSentinel)
		{
		int2 item = *(int2*)stackPtr;
		stackPtr -= 8;
		nodeAddr1 = item.x;
		tmax1 = __int_as_float(item.y);
		}*/
		nodeAddr1 = 0;
	}
}

#elif SHORTSTACK == 2
__device__ void popNode(char*& stackPtr, int& nodeAddr, float& tmin, float& tmax, int& nodeAddr1, float& tmax1, int& nodeAddr2, float& tmax2)
{
	tmin = tmax;

	if(nodeAddr1 == 0)
	{
		int2 item = *(int2*)stackPtr;
		stackPtr -= 8;
		nodeAddr = item.x;
		tmax = __int_as_float(item.y);
	}
	else
	{
		nodeAddr = nodeAddr1;
		tmax = tmax1;

		nodeAddr1 = nodeAddr2;
		tmax1 = tmax2;

		nodeAddr2 = 0;
	}
}

#elif SHORTSTACK == 3
__device__ __forceinline__ void popNode(char*& stackPtr, int& shortStack, int& nodeAddr, float& tmin, float& tmax)
{
	int head = shortStack & 0xF;
	int tail = shortStack >> 4;
	int newHead = ((head-1+SHORTSTACK_SIZE) % SHORTSTACK_SIZE);
	int2& popVal = s_stack[(threadIdx.y*blockDim.x + threadIdx.x)*SHORTSTACK_SIZE + head];

	tmin = tmax;
	nodeAddr = popVal.x;
	tmax = __int_as_float(popVal.y);

	if(head == tail) // Read from global stack
	{		
		popVal = *(int2*)stackPtr;
		stackPtr -= 8;
	}
	else
	{
		head = newHead;
		shortStack = (tail << 4) | head;
	}
}
#endif

//------------------------------------------------------------------------

TRACE_FUNC_KDTREE
{
    // Traversal stack in CUDA thread-local memory.

#ifndef BRANCHLESS
    int2 traversalStack[STACK_SIZE];
	//unsigned long long traversalStack[STACK_SIZE];
#else
	int4 traversalStack[STACK_SIZE];
#endif

    // Live state during traversal, stored in registers.

    int     rayidx;                 // Ray index.
    float   origx, origy, origz;    // Ray origin.
    float   dirx, diry, dirz;       // Ray direction.
    float   tmin;                   // t-value from which the cell starts.
	float   tmax;                   // t-value where the cell ends.
    float   idirx, idiry, idirz;    // 1 / dir
    float   oodx, oody, oodz;       // orig / dir

    char*   stackPtr;               // Current position in traversal stack.
#ifdef SPECULATIVE
	int     leafAddr;               // First postponed leaf, non-negative if none.
	float   leafTmin;               // First postponed leaf tmin.
	float   leafTmax;               // First postponed leaf tmax.
#endif
    int     nodeAddr;               // Non-negative: current internal node, negative: second postponed leaf.
    int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
	float   hitU;                   // u-barycentric of the closest intersection.
	float   hitV;                   // v-barycentric of the closest intersection.

#if SHORTSTACK == 1
	int nodeAddr1;
	float tmax1;
#elif SHORTSTACK == 2
	int nodeAddr1, nodeAddr2;
	float tmax1, tmax2;
#elif SHORTSTACK == 3
	s_stack[(threadIdx.y*blockDim.x + threadIdx.x)*SHORTSTACK_SIZE + 0] = make_int2(EntrypointSentinel, __float_as_int(-1.f)); // Bottom-most entry.
	int shortStack = 0;
#endif

    // Initialize.
    {
#ifdef ONDEMAND_LIKE
		s_limit[threadIdx.y*blockDim.x + threadIdx.x] = 1;
#endif

        // Pick ray index.

        rayidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
        if (rayidx >= numRays)
            return;

		//if(rayidx == 0)
		//	printf("Num %d (%f %f %f) - (%f %f %f)\n", numRays, bmin[0], bmin[1], bmin[2], bmax[0], bmax[1], bmax[2]);

        // Fetch ray.

		float4 o = rays[rayidx * 2 + 0];
        float4 d = rays[rayidx * 2 + 1];
        origx = o.x, origy = o.y, origz = o.z;
        dirx = d.x, diry = d.y, dirz = d.z;
        tmin = o.w;
		float hitT = d.w; // t-value of the closest intersection.

		//printf("(%f %f %f %f) - (%f %f %f %f)\n", o.x, o.y, o.z, o.w, d.x, d.y, d.z, d.w);

        float ooeps = exp2f(-80.0f); // Avoid div by zero.
        idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
        idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
        idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
        oodx = origx * idirx, oody = origy * idiry, oodz = origz * idirz;

		// Intersect the ray against the scene bounding box.
		float clox = bmin[0] * idirx - oodx;
		float chix = bmax[0] * idirx - oodx;
		float cloy = bmin[1] * idiry - oody;
		float chiy = bmax[1] * idiry - oody;
		float cloz = bmin[2] * idirz - oodz;
		float chiz = bmax[2] * idirz - oodz;
		tmin = max4(fminf(clox, chix), fminf(cloy, chiy), fminf(cloz, chiz), tmin) - 1e-4f;
		tmax = min4(fmaxf(clox, chix), fmaxf(cloy, chiy), fmaxf(cloz, chiz), hitT) + 1e-4f;

        // Setup traversal.

#ifndef BRANCHLESS
#if SHORTSTACK != 3
		traversalStack[0] = make_int2(EntrypointSentinel, __float_as_int(-1.f)); // Bottom-most entry.
		/*unsigned long long nd = (unsigned long long)EntrypointSentinel & 0xFFFFFFFFULL;
		unsigned long long ft = (unsigned long long)__float_as_int(-1.f);
		traversalStack[0] = nd | (ft << 32); // Bottom-most entry.*/
#endif
#else
		traversalStack[0] = make_int4(EntrypointSentinel, __float_as_int(CUDART_INF_F), __float_as_int(-1.f), 0); // Bottom-most entry.
#endif
        stackPtr = (char*)&traversalStack[0];
#ifdef SPECULATIVE
		leafAddr = 0;   // No postponed leaf.
#endif
        nodeAddr = 0;   // Start from the root.
        hitIndex = -1;  // No triangle intersected so far.

#if SHORTSTACK == 1
		nodeAddr1 = 0;
#elif SHORTSTACK == 2
		nodeAddr1 = nodeAddr2 = 0;
#endif
    }

	// Traversal loop.
    while(hitIndex == -1 && tmax > tmin)
    {
#ifdef SPECULATIVE
		bool searchingLeaf = true;
#endif
		while(nodeAddr >= 0 && tmax > tmin)
		{
			// Fetch Kdtree node.
			int4 cell = FETCH_GLOBAL(nodesA, nodeAddr, int4);
			//int4 cell = tex1Dfetch(t_nodesI, nodeAddr);
			unsigned int type = cell.w & KDTREE_MASK;

			float split = __int_as_float(cell.z);

			float origDim;
			float idirDim;

			// Gather data for split plane intersection.
			int dim = (type >> KDTREE_DIMPOS);
			switch(dim)	
			{
			case 0: origDim = origx; idirDim = idirx; break;
			case 1: origDim = origy; idirDim = idiry; break;
			case 2: origDim = origz; idirDim = idirz; break;
			}

			float t = (split - origDim) * idirDim;
			
#ifdef PRINT_TRACE_PATH
			if(rayidx == testRay)
				printf("Node %d %f(%d): %f %f %f\n", nodeAddr, split, dim, tmin, t, tmax);
#endif
			
			// Choose first/second based on ray direction
			bool nfd = ((*(unsigned int*)&idirDim) >> 31); // Choose based on the sign bit
			int first = nfd ? cell.y : cell.x;
			int second = nfd ? cell.x : cell.y;

			// Now find, which of the child nodes are intersected by ray.
#ifndef BRANCHLESS
			if(t > tmax) // Only the first child is intersected
			{
				nodeAddr = first;
#ifdef PRINT_TRACE_PATH
				if(rayidx == testRay)
					printf("Near %d %f %f\n", first, tmin, tmax);
#endif
			}
			else if(t < tmin) // Only the second child is intersected
			{
				nodeAddr = second;
#ifdef PRINT_TRACE_PATH
				if(rayidx == testRay)
					printf("Far %d %f %f\n", second, tmin, tmax);
#endif
			}
			else // Both child need to be traversed. Push second on the stack and move to first.
			{
				nodeAddr = first;

#ifdef PRINT_TRACE_PATH
				if(rayidx == testRay)
					printf("Both %d %d %f %f\n", first, second, tmin, tmax);
#endif

				// Push the farther node on stack.
#if SHORTSTACK == 0
				pushNode(stackPtr, second, tmin, tmax);
#elif SHORTSTACK == 1
				pushNode(stackPtr, second, tmin, tmax, nodeAddr1, tmax1);
#elif SHORTSTACK == 2
				pushNode(stackPtr, second, tmin, tmax, nodeAddr1, tmax1, nodeAddr2, tmax2);
#elif SHORTSTACK == 3
				pushNode(stackPtr, shortStack, second, tmin, tmax);
#endif

#ifdef PRINT_TRACE_PATH
				if(rayidx == testRay)
					printf("Push %d %f %f\n", second, tmin, tmax);
#endif

				tmax = t;
			}

#else
			int c = (t < tmax);
			int d = (t < tmin);
			stackPtr += 16;
			*(int4*)stackPtr = make_int4(second, __float_as_int(max(t, tmin)), __float_as_int(tmax), 0);
			stackPtr += 16*c;
			*(int4*)stackPtr = make_int4(first, __float_as_int(tmin), __float_as_int(min(t, tmax)), 0);
			stackPtr -= 16*d;
			
			int4 item = *(int4*)stackPtr;
			stackPtr -= 16;
			nodeAddr = item.x;
			tmin = __int_as_float(item.y);
			tmax = __int_as_float(item.z);
#endif

			// First leaf => postpone and continue traversal.

#ifdef SPECULATIVE
			if(nodeAddr < 0 && leafAddr >= 0)
			{
				searchingLeaf = false;
				leafAddr = nodeAddr;
				leafTmin = tmin;
				leafTmax = tmax;
#if defined(BRANCHLESS) || SHORTSTACK == 0
				popNode(stackPtr, nodeAddr, tmin, tmax);
#elif SHORTSTACK == 1
				popNode(stackPtr, nodeAddr, tmin, tmax, nodeAddr1, tmax1);
#elif SHORTSTACK == 2
				popNode(stackPtr, nodeAddr, tmin, tmax, nodeAddr1, tmax1, nodeAddr2, tmax2);
#elif SHORTSTACK == 3
				popNode(stackPtr, shortStack, nodeAddr, tmin, tmax, nodeAddr1, tmax1, nodeAddr2, tmax2);
#endif
			}

			// All SIMD lanes have found a leaf => process them.

			if(!__any(searchingLeaf))
				break;
#endif
        }

		// Process leaf nodes.

#ifndef SPECULATIVE
		while(nodeAddr < 0)
		{
#ifdef PRINT_TRACE_PATH
			if(rayidx == testRay)
				printf("Leaf %d %f %f\n", nodeAddr, tmin, tmax);
#endif
			if((nodeAddr & KDTREE_MASK) != KDTREE_EMPTYLEAF)
			{
				// Intersect the ray against each triangle using Sven Woop's algorithm.
				for (int triAddr = ~nodeAddr;; triAddr++)
#else
		while(leafAddr < 0)
		{
			if((leafAddr & KDTREE_MASK) != KDTREE_EMPTYLEAF)
			{
				// Intersect the ray against each triangle using Sven Woop's algorithm.
				for (int triAddr = ~leafAddr;; triAddr++)
#endif
				{
					// Read the triangle index
					int triIdx = triIndices[triAddr]; // Do not trash cache by reading index through it

					if (triIdx == 0x80000000)
						break;

					// Read first 16 bytes of the triangle.

					float4 v00 = FETCH_GLOBAL(trisA, triIdx*3 + 0, float4);
					//float4 v00 = tex1Dfetch(t_trisA, triIdx*3 + 0);

					// Compute and check intersection t-value.

					float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
					float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
					float t = Oz * invDz;

#ifdef PRINT_TRACE_PATH
					if(rayidx == testRay)
						printf("Tri %d->%d %f %f %f (%f %f %f %f)\n", triAddr, triIdx, tmin, t, tmax, v00.x, v00.y, v00.z, v00.w);
#endif
#ifndef SPECULATIVE
					if(t >= tmin - delta && t <= tmax + delta)
#else
					if(t >= leafTmin && t <= leafTmax)
#endif
					{
						// Compute and check barycentric u.

						float4 v11 = FETCH_GLOBAL(trisA, triIdx*3 + 1, float4);
						//float4 v11 = tex1Dfetch(t_trisA, triIdx*3 + 1);
						float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
						float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
						float u = Ox + t*Dx;

#ifdef PRINT_TRACE_PATH
						if(rayidx == testRay)
							printf("Tri %d->%d %f %f %f %f\n", triAddr, triIdx, tmin, t, tmax, u);
#endif
						if(u >= 0.0f && u <= 1.0f)
						{
							// Compute and check barycentric v.

							float4 v22 = FETCH_GLOBAL(trisA, triIdx*3 + 2, float4);
							//float4 v22 = tex1Dfetch(t_trisA, triIdx*3 + 2);
							float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
							float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
							float v = Oy + t*Dy;

#ifdef PRINT_TRACE_PATH
							if(rayidx == testRay)
								printf("Tri %d->%d %f %f %f %f %f\n", triAddr, triIdx, tmin, t, tmax, u, v);
#endif
							if(v >= 0.0f && u + v <= 1.0f)
							{
#ifdef PRINT_TRACE_PATH
								if(rayidx == testRay)
									printf("Hit %d->%d %f\n", triAddr, triIdx, t);
#endif
								// Record intersection.
								tmax = t;
								hitU = u;
								hitV = v;
								hitIndex = triIdx;
							}
						}
					}
				} // triangle
			}

			// Pop node
			if(hitIndex == -1)
			{
#ifndef SPECULATIVE
#if defined(BRANCHLESS) || SHORTSTACK == 0
				popNode(stackPtr, nodeAddr, tmin, tmax);
#elif SHORTSTACK == 1
				popNode(stackPtr, nodeAddr, tmin, tmax, nodeAddr1, tmax1);
#elif SHORTSTACK == 2
				popNode(stackPtr, nodeAddr, tmin, tmax, nodeAddr1, tmax1, nodeAddr2, tmax2);
#elif SHORTSTACK == 3
				popNode(stackPtr, shortStack, nodeAddr, tmin, tmax);
#endif
#else
				leafAddr = nodeAddr;
				leafTmin = tmin;
				leafTmax = tmax;
				if(nodeAddr < 0)
				{
#if defined(BRANCHLESS) || SHORTSTACK == 0
					popNode(stackPtr, nodeAddr, tmin, tmax);
#elif SHORTSTACK == 1
					popNode(stackPtr, nodeAddr, tmin, tmax, nodeAddr1, tmax1);
#elif SHORTSTACK == 2
					popNode(stackPtr, nodeAddr, tmin, tmax, nodeAddr1, tmax1, nodeAddr2, tmax2);
#elif SHORTSTACK == 3
					popNode(stackPtr, shortStack, nodeAddr, tmin, tmax);
#endif
				}
#endif
#ifdef PRINT_TRACE_PATH
				if(rayidx == testRay)
					printf("Pop %d %f %f\n", nodeAddr, tmin, tmax);
#endif
			}
			else
			{
				break;
			}
			
			//if(rayidx == 0)
			//	printf("Node %d, %f %f)\n", nodeAddr, tmin, tmax);
        } // leaf
    } // traversal

#ifdef ONDEMAND_LIKE
	if (s_limit[threadIdx.y*blockDim.x + threadIdx.x] == 1)
#endif
    STORE_RESULT(rayidx, hitIndex, tmax);
}


//------------------------------------------------------------------------

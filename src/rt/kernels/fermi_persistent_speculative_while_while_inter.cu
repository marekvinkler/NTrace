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
    "Persistent speculative while-while" kernel used in:

    "Understanding the Efficiency of Ray Traversal on GPUs",
    Timo Aila and Samuli Laine,
    Proc. High-Performance Graphics 2009
*/

#include "CudaTracerKernels.hpp"

//------------------------------------------------------------------------

#define NODES_ARRAY_OF_STRUCTURES           // Define for AOS, comment out for SOA.
#define TRIANGLES_ARRAY_OF_STRUCTURES       // Define for AOS, comment out for SOA.

#define LOAD_BALANCER_BATCH_SIZE        64  // Number of rays to fetch at a time. Must be a multiple of 32.
#define STACK_SIZE                      128  // Size of the traversal stack in local memory.

//#define SIMILAR_TRAVERSAL

extern "C" __device__ int g_warpCounter;    // Work counter for persistent threads.

//------------------------------------------------------------------------

extern "C" __global__ void queryConfig(void)
{
#if (defined(NODES_ARRAY_OF_STRUCTURES) && defined(TRIANGLES_ARRAY_OF_STRUCTURES))
    g_config.bvhLayout = BVHLayout_AOS_AOS;
#elif (defined(NODES_ARRAY_OF_STRUCTURES) && !defined(TRIANGLES_ARRAY_OF_STRUCTURES))
    g_config.bvhLayout = BVHLayout_AOS_SOA;
#elif (!defined(NODES_ARRAY_OF_STRUCTURES) && defined(TRIANGLES_ARRAY_OF_STRUCTURES))
    g_config.bvhLayout = BVHLayout_SOA_AOS;
#elif (!defined(NODES_ARRAY_OF_STRUCTURES) && !defined(TRIANGLES_ARRAY_OF_STRUCTURES))
    g_config.bvhLayout = BVHLayout_SOA_SOA;
#endif

    g_config.blockWidth = 32; // One warp per row.
    g_config.blockHeight = 4; // 6*32 = 192 threads, optimal for GTX285.
    g_config.usePersistentThreads = 1;
}

//------------------------------------------------------------------------

 __device__ float4 cross (const float4& v1, const float4& v2)
 { 
	 float4 ret;
	 ret.x = v1.y * v2.z - v1.z * v2.y;
	 ret.y = v1.z * v2.x - v1.x * v2.z;
	 ret.z = v1.x * v2.y - v1.y * v2.x;
	 ret.w = 0.f;
	 return ret;
 }

 __device__ float4 minus (const float4& v1, const float4& v2)
 { 
	 float4 ret;
	 ret.x = v1.x - v2.x;
	 ret.y = v1.y - v2.y;
	 ret.z = v1.z - v2.z;
	 ret.w = 0.f;
	 return ret;
 }

 __device__ float dot (const float4& v1, const float4& v2)
 {
	 return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
 }

 __device__ float4 retFloat4 (const float& x, const float& y, const float& z)
 {
	 float4 ret;
	 ret.x = x;
	 ret.y = y;
	 ret.z = z;
	 ret.w = 0.f;
	 return ret;
 }

 #ifdef SIMILAR_TRAVERSAL
 // Unlocks a BVH build task in the global memory
__device__ void buildNode(int taskIdx, int nodeIdx, int parentIdx, int childOffset, bool traverse)
{
	int lock = atomicCAS(&g_taskStackBVH.tasks[taskIdx].depend1, LockType_Free, LockType_Set);

	if(lock == LockType_Free) // The thread unlocked the task
	{
		taskCacheActive(taskIdx, g_taskStackBVH.active, &g_taskStackBVH.activeTop);
	}
}
#endif

//------------------------------------------------------------------------

extern "C" __global__ void trace(void)
{
    // Traversal stack in CUDA thread-local memory.
    // Allocate 3 additional entries for spilling rarely used variables.

    int traversalStack[STACK_SIZE];

    // Live state during traversal, stored in registers.

	int     rayidx;                 // Ray index.
    float   origx, origy, origz;    // Ray origin.
	float   dirx, diry, dirz;       // Ray direction.
    float   tmin;                   // t-value from which the ray starts. Usually 0.
    float   idirx, idiry, idirz;    // 1 / dir
    float   oodx, oody, oodz;       // orig / dir
    
	char*   stackPtr;               // Current position in traversal stack.
    int     leafAddr;               // First postponed leaf, non-negative if none.
    int     nodeAddr;               // Non-negative: current internal node, negative: second postponed leaf.
    int     hitIndex;               // Triangle index of the closest intersection, -1 if none.
    float   hitT;                   // t-value of the closest intersection.
	float   hitU;                   // u-barycentric of the closest intersection.
	float   hitV;                   // v-barycentric of the closest intersection.

    // Initialize persistent threads.

    __shared__ volatile int nextRayArray[MaxBlockHeight]; // Current ray index in global buffer.
    __shared__ volatile int rayCountArray[MaxBlockHeight]; // Number of rays in the local pool.
    nextRayArray[threadIdx.y] = 0;
    rayCountArray[threadIdx.y] = 0;

    // Persistent threads: fetch and process rays in a loop.

  do
  {
        int tidx = threadIdx.x; // threadIdx.x
        int widx = threadIdx.y; // threadIdx.y
		volatile int& localPoolRayCount = rayCountArray[widx];
		volatile int& localPoolNextRay = nextRayArray[widx];

        // Local pool is empty => fetch new rays from the global pool using lane 0.

        if (tidx == 0 && localPoolRayCount <= 0)
		{
            localPoolNextRay = atomicAdd(&g_warpCounter, LOAD_BALANCER_BATCH_SIZE);
            localPoolRayCount = LOAD_BALANCER_BATCH_SIZE;
		}

        // Pick 32 rays from the local pool.
        // Out of work => done.
        {
            rayidx = localPoolNextRay + tidx;
			if(rayidx >= c_in.numRays)
				break;

            if (tidx == 0)
			{
                localPoolNextRay += 32;
                localPoolRayCount -= 32;
			}

            // Fetch ray.

			float4 o = FETCH_GLOBAL(rays, rayidx * 2 + 0, float4);
			float4 d = FETCH_GLOBAL(rays, rayidx * 2 + 1, float4);
            origx = o.x, origy = o.y, origz = o.z;
			tmin = o.w;
			dirx = d.x, diry = d.y, dirz = d.z;
			tmin = o.w;


            float ooeps = exp2f(-80.0f); // Avoid div by zero.
			idirx = 1.0f / (fabsf(d.x) > ooeps ? d.x : copysignf(ooeps, d.x));
			idiry = 1.0f / (fabsf(d.y) > ooeps ? d.y : copysignf(ooeps, d.y));
			idirz = 1.0f / (fabsf(d.z) > ooeps ? d.z : copysignf(ooeps, d.z));
			oodx = origx * idirx, oody = origy * idiry, oodz = origz * idirz;

            // Setup traversal.

            traversalStack[0] = EntrypointSentinel; // Bottom-most entry.
            stackPtr = (char*)&traversalStack[0];
            leafAddr = 0;   // No postponed leaf.
            nodeAddr = 0;   // Start from the root.
            hitIndex = -1;  // No triangle intersected so far.
            hitT     = d.w; // tmax
        }

#ifdef SIMILAR_TRAVERSAL
		if(nodeAddr < 0) // The built node may have been a leaf
		{
			leafAddr = nodeAddr;
			nodeAddr = *(int*)stackPtr;
			stackPtr -= 4;
		}
#endif

		while(nodeAddr!=EntrypointSentinel)
		{
			// Traverse internal nodes until all SIMD lanes have found a leaf.

			bool searchingLeaf = true;
			while(nodeAddr>=0 && nodeAddr!=EntrypointSentinel)
			{
				// Fetch AABBs of the two child nodes.

				float4 n0xy = FETCH_TEXTURE(nodesA, nodeAddr*4+0, float4);  // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
				float4 n1xy = FETCH_TEXTURE(nodesA, nodeAddr*4+1, float4);  // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
				float4 nz   = FETCH_TEXTURE(nodesA, nodeAddr*4+2, float4);  // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)
				float4 cnodesA=FETCH_TEXTURE(nodesA, nodeAddr*4+3, float4);
				int2 cnodes; cnodes.x = __float_as_int(cnodesA.x); cnodes.y = __float_as_int(cnodesA.y);

				// Intersect the ray against the child nodes.

				float c0lox = n0xy.x * idirx - oodx;
				float c0hix = n0xy.y * idirx - oodx;
				float c0loy = n0xy.z * idiry - oody;
				float c0hiy = n0xy.w * idiry - oody;
				float c0loz = nz.x   * idirz - oodz;
				float c0hiz = nz.y   * idirz - oodz;
				float c1loz = nz.z   * idirz - oodz;
				float c1hiz = nz.w   * idirz - oodz;
				float c0min = max4(fminf(c0lox, c0hix), fminf(c0loy, c0hiy), fminf(c0loz, c0hiz), tmin);
				float c0max = min4(fmaxf(c0lox, c0hix), fmaxf(c0loy, c0hiy), fmaxf(c0loz, c0hiz), hitT);
				float c1lox = n1xy.x * idirx - oodx;
				float c1hix = n1xy.y * idirx - oodx;
				float c1loy = n1xy.z * idiry - oody;
				float c1hiy = n1xy.w * idiry - oody;
				float c1min = max4(fminf(c1lox, c1hix), fminf(c1loy, c1hiy), fminf(c1loz, c1hiz), tmin);
				float c1max = min4(fmaxf(c1lox, c1hix), fmaxf(c1loy, c1hiy), fmaxf(c1loz, c1hiz), hitT);

				// Decide where to go next.
				// Differs from "while-while" because this just happened to produce better code here.

				bool traverseChild0 = (c0max >= c0min);
				bool traverseChild1 = (c1max >= c1min);

				// Neither child was intersected => pop stack.

				if (!traverseChild0 && !traverseChild1)
				{
					nodeAddr = *(int*)stackPtr;
					stackPtr -= 4;
				}

				// Otherwise => fetch child pointers.

				else
				{
#ifdef SIMILAR_TRAVERSAL
					if(traverseChild0 && (cnodes.x & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild left child
					{
						buildNode(cnodes.x & UNBUILD_UNMASK, cnodes.x, nodeAddr, 0, traverseChild0); // Clear the unbuild flag to get the pool entry
						cnodes.x = nodeAddr | (UNBUILD_FLAG | 0x0); // Save the code of the location where to check for completion
					}

					if(traverseChild1 && (cnodes.y & UNBUILD_MASK) == UNBUILD_FLAG) // Unbuild right child
					{
						buildNode(cnodes.y & UNBUILD_UNMASK, cnodes.y, nodeAddr, 1, traverseChild1); // Clear the unbuild flag to get the pool entry
						cnodes.y = nodeAddr | (UNBUILD_FLAG | 0x10000000); // Save the code of the location where to check for completion
					}
#endif

					nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

					// Both children were intersected => push the farther one.

					if (traverseChild0 && traverseChild1)
					{
						// going to larger SA for shadowrays does not really help at all
						// slows down
						if (c1min < c0min)
							swap(nodeAddr, cnodes.y);
						stackPtr += 4;
						*(int*)stackPtr = cnodes.y;
					}
				}

				// First leaf => postpone and continue traversal.

				if(nodeAddr<0 && leafAddr>=0)
				{
					searchingLeaf = false;
					leafAddr = nodeAddr;
					nodeAddr = *(int*)stackPtr;
					stackPtr -= 4;
				}

				// All SIMD lanes have found a leaf => process them.

				if(!__any(searchingLeaf))
					break;
			}

			// Process postponed leaf nodes.

			while(leafAddr<0)
			{
				// Fetch the start and end of the triangle list.

				float4 leaf=FETCH_TEXTURE(nodesA, (-leafAddr-1)*4+3, float4);

				int triAddr  = __float_as_int(leaf.x);              // stored as int
				int triAddr2 = __float_as_int(leaf.y);              // stored as int

				// Intersect the ray against each triangle using Sven Woop's algorithm.

				for( ;triAddr < triAddr2; triAddr++)
				{
					// Compute and check intersection t-value.

					//int triIdx = FETCH_TEXTURE(triIndices, triAddr, int);
					int triIdx = FETCH_GLOBAL(triIndices, triAddr, int);

					//float4 v00 = FETCH_GLOBAL(trisA, triIdx*3+0, float4);
					float4 v00 = FETCH_TEXTURE(trisA, triIdx*3+0, float4);
					//float4 v11 = FETCH_GLOBAL(trisA, triIdx*3+1, float4);
					float4 v11 = FETCH_TEXTURE(trisA, triIdx*3+1, float4);
					//float4 v22 = FETCH_GLOBAL(trisA, triIdx*3+2, float4);
					float4 v22 = FETCH_TEXTURE(trisA, triIdx*3+2, float4);

					// shirley
					float4 nrmN = cross(minus(v11,v00),minus(v22,v00));
					const float den = dot(nrmN,retFloat4(dirx,diry,dirz));

					//if(den >= 0.0f)
					//	continue;

					const float deni = 1.0f / den;
					const float4 org0 = minus(v00,retFloat4(origx,origy,origz));
					float t = dot(nrmN,org0)*deni;

					if (t > tmin && t < hitT)
					{
						const float4 crossProd = cross(retFloat4(dirx,diry,dirz),org0);
						const float v = dot(minus(v00,v22),crossProd)*deni;
						if (v >= 0.0f && v <= 1.0f)
						{
							const float u = 1 - v - (-dot(minus(v00,v11),crossProd)*deni); // woop
							if (u >= 0.0f && u + v <= 1.0f)
							{
								hitT = t;
								hitU = u;
								hitV = v;
								hitIndex = triIdx;
								if(c_in.anyHit)
								{
									nodeAddr = EntrypointSentinel;
									break;
								}
							}
						}
					}
				} // triangle

				// Another leaf was postponed => process it as well.

				leafAddr = nodeAddr;
				if(nodeAddr<0)
				{
					nodeAddr = *(int*)stackPtr;
					stackPtr -= 4;
				}
			} // leaf

#ifdef SIMILAR_TRAVERSAL
			if(__any((nodeAddr & UNBUILD_MASK) == UNBUILD_FLAG)) // If any thread wants to traverse unbuild node
			{
				hitT = 0;
				stackPtr = 0;
				nodeAddr = 0;

				hitIndex = 0;
				hitU = 0;
				hitV = 0;
				break; // Skip traversal for build
			}
#endif
		} // traversal

		// Remap intersected triangle index, and store the result.

#ifdef SIMILAR_TRAVERSAL
		if(rayidx >= 0)
		{
#endif
		//if (hitIndex != -1)
		//hitIndex = FETCH_TEXTURE(triIndices, hitIndex, int);
		STORE_RESULT(rayidx, hitIndex, hitT, hitU, hitV);
#ifdef SIMILAR_TRAVERSAL
		rayidx = -1;
		nodeAddr = 0;
		}
#endif
    } while(1); // persistent threads (always true)
}

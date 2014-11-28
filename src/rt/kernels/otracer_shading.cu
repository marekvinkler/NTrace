/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
 *        names of its contributors may be used to endorse or promote products
 *        derived from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 *  ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 *  WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 *  DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 *  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 *  (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 *  ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 *  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 *  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

/*
    GF100-optimized variant of the "Speculative while-while"
    kernel used in:

    "Understanding the Efficiency of Ray Traversal on GPUs",
    Timo Aila and Samuli Laine,
    Proc. High-Performance Graphics 2009
*/

#include "CudaTracerKernels.hpp"
#include "helper_math.h"
#include <curand.h>

//------------------------------------------------------------------------

#define STACK_SIZE  64  // Size of the traversal stack in local memory.

//------------------------------------------------------------------------

extern "C" __global__ void queryConfig(void)
{
    g_config.bvhLayout = BVHLayout_Compact;
    g_config.blockWidth = 32; // One warp per row.
    g_config.blockHeight = 4; // 4*32 = 128 threads, optimal for GTX480
}

//------------------------------------------------------------------------

__device__ bool traversal(int rayidx, int numRays, float4* rays, float4* nodesA, bool anyHit, int4* results, int* hitIdx, float* dist, float* bU, float* bV)
{
    // Traversal stack in CUDA thread-local memory.

    int traversalStack[STACK_SIZE];

    // Live state during traversal, stored in registers.

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

	float	hitU, hitV;				// Barycentric coordinates

    // Initialize.
    {
        // Pick ray index.

        // Fetch ray.

        float4 o = rays[rayidx * 2 + 0];
        float4 d = rays[rayidx * 2 + 1];
        origx = o.x, origy = o.y, origz = o.z;
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

    // Traversal loop.
    while (nodeAddr != EntrypointSentinel)
    {
        // Traverse internal nodes until all SIMD lanes have found a leaf.

        bool searchingLeaf = true;
        while (nodeAddr >= 0 && nodeAddr != EntrypointSentinel)
        {
            // Fetch AABBs of the two child nodes.

            float4* ptr = (float4*)((char*)nodesA + nodeAddr);
            float4 n0xy = ptr[0]; // (c0.lo.x, c0.hi.x, c0.lo.y, c0.hi.y)
            float4 n1xy = ptr[1]; // (c1.lo.x, c1.hi.x, c1.lo.y, c1.hi.y)
            float4 nz   = ptr[2]; // (c0.lo.z, c0.hi.z, c1.lo.z, c1.hi.z)

            // Intersect the ray against the child nodes.

            float c0lox = n0xy.x * idirx - oodx;
            float c0hix = n0xy.y * idirx - oodx;
            float c0loy = n0xy.z * idiry - oody;
            float c0hiy = n0xy.w * idiry - oody;
            float c0loz = nz.x   * idirz - oodz;
            float c0hiz = nz.y   * idirz - oodz;
            float c1loz = nz.z   * idirz - oodz;
            float c1hiz = nz.w   * idirz - oodz;
			float c0min = spanBeginFermi(c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, tmin);
			float c0max = spanEndFermi  (c0lox, c0hix, c0loy, c0hiy, c0loz, c0hiz, hitT);
            float c1lox = n1xy.x * idirx - oodx;
            float c1hix = n1xy.y * idirx - oodx;
            float c1loy = n1xy.z * idiry - oody;
            float c1hiy = n1xy.w * idiry - oody;
			float c1min = spanBeginFermi(c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, tmin);
			float c1max = spanEndFermi  (c1lox, c1hix, c1loy, c1hiy, c1loz, c1hiz, hitT);

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
                int2 cnodes = *(int2*)&ptr[3];
                nodeAddr = (traverseChild0) ? cnodes.x : cnodes.y;

                // Both children were intersected => push the farther one.

                if (traverseChild0 && traverseChild1)
                {
                    if (c1min < c0min)
                        swap(nodeAddr, cnodes.y);
                    stackPtr += 4;
                    *(int*)stackPtr = cnodes.y;
                }
            }

            // First leaf => postpone and continue traversal.

            if (nodeAddr < 0 && leafAddr >= 0)
            {
                searchingLeaf = false;
                leafAddr = nodeAddr;
                nodeAddr = *(int*)stackPtr;
                stackPtr -= 4;
            }

            // All SIMD lanes have found a leaf => process them.

            if (!__any(searchingLeaf))
                break;
        }

        // Process postponed leaf nodes.

        while (leafAddr < 0)
        {
            // Intersect the ray against each triangle using Sven Woop's algorithm.

            for (int triAddr = ~leafAddr;; triAddr += 3)
            {
                // Read first 16 bytes of the triangle.
                // End marker (negative zero) => all triangles processed.

                float4 v00 = tex1Dfetch(t_trisA, triAddr + 0);
                if (__float_as_int(v00.x) == 0x80000000)
                    break;

                // Compute and check intersection t-value.

                float Oz = v00.w - origx*v00.x - origy*v00.y - origz*v00.z;
                float invDz = 1.0f / (dirx*v00.x + diry*v00.y + dirz*v00.z);
                float t = Oz * invDz;

                if (t > tmin && t < hitT)
                {
                    // Compute and check barycentric u.

                    float4 v11 = tex1Dfetch(t_trisA, triAddr + 1);
                    float Ox = v11.w + origx*v11.x + origy*v11.y + origz*v11.z;
                    float Dx = dirx*v11.x + diry*v11.y + dirz*v11.z;
                    float u = Ox + t*Dx;

                    if (u >= 0.0f && u <= 1.0f)
                    {
                        // Compute and check barycentric v.

                        float4 v22 = tex1Dfetch(t_trisA, triAddr + 2);
                        float Oy = v22.w + origx*v22.x + origy*v22.y + origz*v22.z;
                        float Dy = dirx*v22.x + diry*v22.y + dirz*v22.z;
                        float v = Oy + t*Dy;

                        if (v >= 0.0f && u + v <= 1.0f)
                        {
                            // Record intersection.
                            // Closest intersection not required => terminate.

                            hitT = t;
							hitU = u;
							hitV = v;
                            hitIndex = triAddr;
                            if (anyHit)
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
    } // traversal

    // Remap intersected triangle index, and store the result.

    if (hitIndex != -1)
	{
        hitIndex = tex1Dfetch(t_triIndices, hitIndex);
		*hitIdx = hitIndex;
		*dist = hitT;
		*bU = hitU;
		*bV = hitV;
		return true;
	}

	return false;
}

//------------------------------------------------------------------------

__device__ inline float4 fromABGR(unsigned int abgr)
{
    return make_float4(
		(float)(abgr & 0xFF) * (1.0f / 255.0f),
		(float)((abgr >> 8) & 0xFF) * (1.0f / 255.0f),
		(float)((abgr >> 16) & 0xFF) * (1.0f / 255.0f),
		(float)(abgr >> 24) * (1.0f / 255.0f));
}

//-----------------------------------------------------------------------

#define SAMPLES_PER_PIXEL 1
#define MAX_PATH_LENGTH 4

extern "C" __device__ inline float interpolateAttribute1f(float3 bary, float attribA, float attribB, float attribC)
{
	return attribA * bary.x + attribB * bary.y + attribC * bary.z;
}

extern "C" __device__ inline float2 interpolateAttribute2f(float3 bary, float2 attribA, float2 attribB, float2 attribC)
{
	return make_float2(
			attribA.x * bary.x + attribB.x * bary.y + attribC.x * bary.z,
			attribA.y * bary.x + attribB.y * bary.y + attribC.y * bary.z
		);
}

extern "C" __device__ inline float3 interpolateAttribute3f(float3 bary, float3 attribA, float3 attribB, float3 attribC)
{
	return make_float3(
			attribA.x * bary.x + attribB.x * bary.y + attribC.x * bary.z,
			attribA.y * bary.x + attribB.y * bary.y + attribC.y * bary.z,
			attribA.z * bary.x + attribB.z * bary.y + attribC.z * bary.z
		);
}

extern "C" __device__ inline float4 interpolateAttribute4f(float3 bary, float4 attribA, float4 attribB, float4 attribC)
{
	return make_float4(
			attribA.x * bary.x + attribB.x * bary.y + attribC.x * bary.z,
			attribA.y * bary.x + attribB.y * bary.y + attribC.y * bary.z,
			attribA.z * bary.x + attribB.z * bary.y + attribC.z * bary.z,
			attribA.w * bary.x + attribB.w * bary.y + attribC.w * bary.z
		);
}

extern "C" __device__ inline float4 sample2D(float3 bary, float2 tc, float4 texAtlasInfo, texture<float4, 2> tex)
{
	tc.x = tc.x - floorf(tc.x);
	tc.y = tc.y - floorf(tc.y);
	tc.x = tc.x * texAtlasInfo.z + texAtlasInfo.x;
	tc.y = tc.y * texAtlasInfo.w + texAtlasInfo.y;
	return tex2D(tex, tc.x, tc.y);
}

__device__ inline void jenkinsMix(int& a, int& b, int& c)
{
    a -= b; a -= c; a ^= (c>>13);
    b -= c; b -= a; b ^= (a<<8);
    c -= a; c -= b; c ^= (b>>13);
    a -= b; a -= c; a ^= (c>>12);
    b -= c; b -= a; b ^= (a<<16);
    c -= a; c -= b; c ^= (b>>5);
    a -= b; a -= c; a ^= (c>>3);
    b -= c; b -= a; b ^= (a<<10);
    c -= a; c -= b; c ^= (b>>15);	// ~36 instructions
}

//------------------------------------------------------------------------

__device__ inline float random(int seed, int offset)
{
	int hashA = seed + offset;
	int hashB = 0x9e3779b9u;
	int hashC = 0x9e3779b9u;
	jenkinsMix(hashA, hashB, hashC);
	jenkinsMix(hashA, hashB, hashC);
	jenkinsMix(hashA, hashB, hashC);
	return ((float)hashA * exp2f(-32)) + 0.5f;
}

//------------------------------------------------------------------------

__device__ inline int randomInt(int seed, int offset)
{
	int hashA = seed + offset;
	int hashB = 0x9e3779b9u;
	int hashC = 0x9e3779b9u;
	jenkinsMix(hashA, hashB, hashC);
	return hashA;
}

//------------------------------------------------------------------------

extern "C" __global__ void otrace_kernel(void)
{
	// Get information from constant memory
    const OtraceInput& in = c_OtraceInput;
	int numRays = in.numRays;
	bool anyHit = in.anyHit;
	float4* rays = (float4*)in.rays;
	int4* results = (int4*)in.results;
	float4* nodesA = (float4*)in.nodesA;
	float4* nodesB = (float4*)in.nodesB;
    float4* nodesC = (float4*)in.nodesC;
    float4* nodesD = (float4*)in.nodesD;
    float4* trisA = (float4*)in.trisA;
    float4* trisB = (float4*)in.trisB;
    float4* trisC = (float4*)in.trisC;
	int* triIndices = (int*)in.triIndices;
	float2*	texCoords = (float2*)in.texCoords;
	float3* normals = (float3*)in.normals;
	int3* vertIdx = (int3*)in.triVertIndex;
	float4* atlasInfo = (float4*)in.atlasInfo;
	int* matId = (int*)in.matId;
	float4* matInfo = (float4*)in.matInfo;
	int emissiveCount = in.emissiveNum;
	int3* emissive = (int3*)in.emissive;
	int trisCount = in.trisCount;
	int3* tris = (int3*)in.tris;
	int vertsCount = in.vertsCount;
	float3* verts = (float3*)in.verts;
	int randomSeed = in.randomSeed;
	unsigned int* triMaterialColor    = (unsigned int*)in.matColor;

    int rayidx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    if (rayidx >= numRays)
        return;

	// Cast primary ray
	int hitIndex;
	float hitU;
	float hitV;
	float hitT;
	float4 result = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
	float4 primO = rays[rayidx * 2 + 0];
	float4 primD = rays[rayidx * 2 + 1];
	float4 tempDir;
	float mult_val = 1.0f;

	/*result = make_float4(-random(in.randomSeed, rayidx),
		-random(in.randomSeed, rayidx),
		-random(in.randomSeed, rayidx),
		1.0f);
    STORE_RESULT(rayidx, hitIndex, result.x, result.y, result.z);
	return;*/

	// For each sample per pixel
	for (int spp = 0; spp < SAMPLES_PER_PIXEL; spp++)
	{
		// Grab primitive origin and direction
		rays[rayidx * 2 + 0] = primO;
		rays[rayidx * 2 + 1] = primD;

		int depth = 0;
		float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);				// Accumulated color
		float4 reflectance = make_float4(1.0f, 1.0f, 1.0f, 1.0f);		// Accumulated reflectance
		while(true)
		{
			// If we miss, return black
			bool hit = traversal(rayidx, numRays, rays, nodesA, anyHit, results, &hitIndex, &hitT, &hitU, &hitV);
			if (hit == false || hitIndex == -1)
			{
				break;
			}
			
			// Get hit triangle index, compute barycentric coordinates, texture coordinates, hitpoint and normal; break on hitting light
			int tri = hitIndex;
			if (matInfo[matId[tri]].x > 0.0f)
			{
				color = color + reflectance * make_float4(matInfo[matId[tri]].x, matInfo[matId[tri]].x, matInfo[matId[tri]].x, 1.0f);
				break;
			}
			float3 bary = make_float3(hitU, hitV, 1.0f - hitU - hitV);
			float4 hitPoint = rays[rayidx * 2 + 0] + rays[rayidx * 2 + 1] * hitT * 0.99f;
			float4 hitPointBehind = rays[rayidx * 2 + 0] + rays[rayidx * 2 + 1] * hitT * 1.01f;
			float3 normal = interpolateAttribute3f(bary, normals[vertIdx[tri].x], normals[vertIdx[tri].y], normals[vertIdx[tri].z]);
			float2 tc = interpolateAttribute2f(bary, texCoords[vertIdx[tri].x], texCoords[vertIdx[tri].y], texCoords[vertIdx[tri].z]);
			// Calculate normal (always directing towards ray origin)
			float3 normalDir = dot(normal, make_float3(rays[rayidx * 2 + 1])) < 0.0f ? normal : normal * -1;
			// Sample color	
			float4 f = matInfo[matId[tri]].x > 0.0f ? make_float4(matInfo[matId[tri]].x, matInfo[matId[tri]].x, matInfo[matId[tri]].x, 1.0f) : 
				matInfo[matId[tri]].w == 0.0f ? fromABGR(triMaterialColor[tri]) : sample2D(bary, tc, atlasInfo[tri], t_textureAtlas);
			f.x = powf(f.x, 2.2f);
			f.y = powf(f.y, 2.2f);
			f.z = powf(f.z, 2.2f);
			// Calculate maximum reflectance
			float p = f.x > f.y && f.x > f.z ? f.x : f.y > f.z ? f.y : f.z;
			
			// Store previous ray dir
			float4 raydir = rays[rayidx * 2 + 1];

			// Explicit step
			if (matInfo[matId[tri]].y == 0.0f && matInfo[matId[tri]].z == 0.0f)
			{
				int emiId = (uint)randomInt(in.randomSeed, rayidx + (spp + 127) * (rayidx + (depth + 334) * (rayidx + 72))) % emissiveCount;
				float3 baryRand = make_float3(random(in.randomSeed, rayidx + (spp + 17) * (rayidx + (depth + 68) * (rayidx + 62))), 
					random(in.randomSeed, rayidx + (spp + 63) * (rayidx + (depth + 68) * (rayidx + 628))), 
					0.0f);
				if(baryRand.x + baryRand.y > 1.0f)
				{
					baryRand.x = 1.0f - baryRand.x;
					baryRand.y = 1.0f - baryRand.y;
				}
				baryRand.z = 1.0f - baryRand.x - baryRand.y;
				float3 sp = interpolateAttribute3f(baryRand, verts[emissive[emiId].x], verts[emissive[emiId].y], verts[emissive[emiId].z]);
				float3 l = sp - make_float3(hitPoint);
				float atten = sqrtf(1.0f / (length(l) + 1.0f));
				l = normalize(l);
				rays[rayidx * 2 + 0] = hitPoint;
				rays[rayidx * 2 + 0].w = 0.01f;
				rays[rayidx * 2 + 1] = make_float4(l, 10000.0f);
				hit = traversal(rayidx, numRays, rays, nodesA, anyHit, results, &hitIndex, &hitT, &hitU, &hitV);
				float4 expl = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
				if (hit && matInfo[matId[hitIndex]].x > 0.0f)
				{
					expl = matInfo[matId[hitIndex]].x * f * dot(normalDir, l) * atten;
				}

				color = color + reflectance * (matInfo[matId[tri]].x > 0.0f ? matInfo[matId[tri]].x : 0.0f) + reflectance * expl;
			}
			else
			{
				color = color + reflectance * (matInfo[matId[tri]].x > 0.0f ? matInfo[matId[tri]].x : 0.0f);
			}

			// Russian roulette
			if (++depth > 3)
			{
				if (random(in.randomSeed, rayidx + spp * (rayidx + depth * (rayidx + 256))) < p)
				{
					f = f * (1.0f / p);

					if (depth > MAX_PATH_LENGTH)
					{
						break;
					}
				}
				else
				{
					break;
				}
			}
			
			reflectance = reflectance * f;
			// Diffuse
			if (matInfo[matId[tri]].y == 0.0f && matInfo[matId[tri]].z == 0.0f)
			{
				float r1 = 2.0f * 3.141592654f * random(in.randomSeed, rayidx + (spp + 37) * (rayidx + (depth + 17) * (rayidx + 171)));
				float r2 = random(in.randomSeed, rayidx + (spp + 63) * (rayidx + (depth + 94) * (rayidx + 317)));
				float r2s = sqrtf(r2);
				float3 w = normalDir;
				float3 u = normalize(cross((fabsf(w.x) > 0.1f ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f)), w));
				float3 v = cross(w, u);
				float3 d = normalize(u * cosf(r1) * r2s + v * sinf(r1) * r2s + w * sqrtf(1.0f - r2));
				rays[rayidx * 2 + 0] = hitPoint;
				rays[rayidx * 2 + 0].w = 0.01f;
				rays[rayidx * 2 + 1] = make_float4(d, 10000.0f);
			}
			// Ideal dielectric reflection
			else if (matInfo[matId[tri]].z == 0.0f)
			{
				rays[rayidx * 2 + 0] = hitPoint;
				rays[rayidx * 2 + 0].w = 0.01f;
				rays[rayidx * 2 + 1] = raydir - 2.0f * make_float4(normal, 0.0f) * dot(make_float4(normal, 0.0f), raydir);
			}
			// Ideal dielectric refraction
			else if (matInfo[matId[tri]].y == 0.0f)
			{
				rays[rayidx * 2 + 0] = hitPoint;
				rays[rayidx * 2 + 0].w = 0.01f;
				rays[rayidx * 2 + 1] = raydir - 2.0f * make_float4(normal, 0.0f) * dot(make_float4(normal, 0.0f), raydir);

				bool into = dot(normal, normalDir) > 0.0f;
				float nc = 1.0f;
				float nt = 1.5f;
				float nnt = into ? nc / nt : nt / nc;
				float ddn = dot(make_float3(raydir), normalDir);
				float cos2t = 1.0f - nnt * nnt * (1.0f - ddn * ddn);
				if (cos2t < 0.0f)
				{
					continue;
				}
				float3 tdir = normalize(make_float3(raydir) * nnt - normal * ((into ? 1.0f : -1.f) * (ddn * nnt + sqrtf(cos2t))));
				//float3 tdir = normalize(make_float3(raydir));
				float a = nt - nc;
				float b = nt + nc;
				float c = 1.0f - (into ? -ddn : dot(tdir, normal));
				float Re = matInfo[matId[tri]].y + (1.0f - matInfo[matId[tri]].y) * c * c * c * c * c;
				float Tr = 1.0f - Re;
				float P = 0.25f + 0.5f * Re;
				float RP = Re / P;
				float TP = Tr / (1.0f - P);

				if (random(in.randomSeed, rayidx + (spp + 63) * (rayidx + (depth + 94) * (rayidx + 317))) < P)
				{
					reflectance = reflectance * RP;
				}
				else
				{
					reflectance = reflectance * TP;
					rays[rayidx * 2 + 0] = hitPointBehind;
					rays[rayidx * 2 + 0].w = 0.01f;
					rays[rayidx * 2 + 1] = make_float4(tdir, 10000.0f);
				}
			}
		}
		
		result = result + color;

		// Until path is terminated or path maximum length is reached
		/*for (int path = 0; path < MAX_PATH_LENGTH; path++)
		{
			float4 color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
			float3 normal = make_float3(0.0f, 0.0f, 0.0f);
			float len;
			bool hit;
		
			tempDir = rays[rayidx * 2 + 1];

			// Check hit
			hit = traversal(rayidx, numRays, rays, nodesA, anyHit, results, &hitIndex, &hitT, &hitU, &hitV);
			if(hit)
			{
				int tri = hitIndex;
				if (tri == -1)
				{
					// Invalid triangle hit - return black
					color = make_float4(0.0f, 0.0f, 0.0f, 1.0f);
				}
				else
				{
					if(matInfo[matId[tri]].w == 0.0f)
					{
						// We hit a light
						color = make_float4(1.0f, 1.0f, 1.0f, 1.0f);
						normal = make_float3(rays[rayidx * 2 + 1]);
					}
					else
					{
						// We hit a triangle
						float3 bary = make_float3(hitU, hitV, 1.0f - hitU - hitV);

						// Sample diffuse texture for color
						float2 tc = interpolateAttribute2f(bary, texCoords[vertIdx[tri].x], texCoords[vertIdx[tri].y], texCoords[vertIdx[tri].z]);
						color = sample2D(bary, tc, atlasInfo[tri], t_textureAtlas);

						// Interpolate normal
						normal = interpolateAttribute3f(bary, normals[vertIdx[tri].x], normals[vertIdx[tri].y], normals[vertIdx[tri].z]);
				
						// Generate random value for emissive triangle ID
						int hashA = in.randomSeed + path * spp + spp + path + rayidx * (path + threadIdx.x + 17);
						int hashB = 0x9e3779b9u;
						int hashC = 0x9e3779b9u;
						jenkinsMix(hashA, hashB, hashC);
						int emiId = (uint)hashC % emissiveCount;

						// Generate random barycentric coordinate
						float3 baryRand = make_float3((float)hashA * exp2f(-32), (float)hashB * exp2f(-32), 0.0f);
						if(baryRand.x + baryRand.y > 1.0f)
						{
							baryRand.x = 1.0f - baryRand.x;
							baryRand.y = 1.0f - baryRand.y;
						}
						baryRand.z = 1.0f - baryRand.x - baryRand.y;

						// Calculate new origin and direction to light
						rays[rayidx * 2 + 0] = rays[rayidx * 2 + 0] + rays[rayidx * 2 + 1] * hitT * 0.99f;
						rays[rayidx * 2 + 0].w = 0.01f;
						float3 pos = interpolateAttribute3f(baryRand, verts[emissive[emiId].x], verts[emissive[emiId].y], verts[emissive[emiId].z]);
						float3 dir = pos - make_float3(rays[rayidx * 2 + 0]);
						len = length(dir);
						dir = normalize(dir);
						rays[rayidx * 2 + 1] = make_float4(dir, 10000.0f);

						// Calculate BRDF
						
						color = color * max(dot(dir, normal), 0.0f);
					}
				}
			}

			// Cast a ray against random light source position
			int hitIdx;
			hit = traversal(rayidx, numRays, rays, nodesA, anyHit, results, &hitIdx, &hitT, &hitU, &hitV);
			if (hit)
			{
				// Hit means shadow, unless emissive material
				if (hitIdx != -1)
				{
					if(matInfo[matId[hitIdx]].w != 0.0f)
					{
						color *= 0.0f;
					}
				}
			}
		
			// Generate random value for next step
			int hashA = in.randomSeed + path * spp + spp + path * 4 + 27 + rayidx * (path + threadIdx.x + 34);
			int hashB = 0x9e3779b9u;
			int hashC = 0x9e3779b9u;
			jenkinsMix(hashA, hashB, hashC);
			rays[rayidx * 2 + 0].w = 0.01f;
			rays[rayidx * 2 + 1] = normalize(make_float4((float)hashA * exp2f(-32), (float)hashB * exp2f(-32), (float)hashC * exp2f(-32), 0.0f));
			rays[rayidx * 2 + 1].w = 10000.0f;
		
			mult_val = dot(tempDir, make_float4(-normal, 0.0f));
			color.x = max(color.x, 0.0f);
			color.y = max(color.y, 0.0f);
			color.z = max(color.z, 0.0f);
			result += color * mult_val;
		}*/
	}
	result /= SAMPLES_PER_PIXEL;

    STORE_RESULT(rayidx, hitIndex, result.x, result.y, result.z);
}

//------------------------------------------------------------------------

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

#include "cuda/RendererKernels.hpp"
#include "base/Math.hpp"

#include "3d/Light.hpp"

using namespace FW;

//------------------------------------------------------------------------

__device__ inline Vec4f fromABGR(U32 abgr)
{
    return Vec4f(
        (F32)(abgr & 0xFF) * (1.0f / 255.0f),
        (F32)((abgr >> 8) & 0xFF) * (1.0f / 255.0f),
        (F32)((abgr >> 16) & 0xFF) * (1.0f / 255.0f),
        (F32)(abgr >> 24) * (1.0f / 255.0f));
}

//------------------------------------------------------------------------

__device__ inline U32 toABGR(Vec4f v)
{
    return
        (U32)(fminf(fmaxf(v.x, 0.0f), 1.0f) * 255.0f) |
        ((U32)(fminf(fmaxf(v.y, 0.0f), 1.0f) * 255.0f) << 8) |
        ((U32)(fminf(fmaxf(v.z, 0.0f), 1.0f) * 255.0f) << 16) |
        ((U32)(fminf(fmaxf(v.w, 0.0f), 1.0f) * 255.0f) << 24);
}

//------------------------------------------------------------------------

extern "C" __global__ void reconstructKernel(void)
{
    // Get parameters.

    const ReconstructInput& in = c_ReconstructInput;
    int taskIdx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));
    if (taskIdx >= in.numPrimary)
        return;

    // Initialize.

    int                     primarySlot     = in.firstPrimary + taskIdx;
    int                     primaryID       = ((const S32*)in.primarySlotToID)[primarySlot];
    const RayResult&        primaryResult   = ((const RayResult*)in.primaryResults)[primarySlot];
    const S32*              batchSlots      = (const S32*)in.batchIDToSlot + ((in.isPrimary || in.isTextured || in.isPathTraced) ? primaryID : taskIdx * in.numRaysPerPrimary);
    const RayResult*        batchResults    = (const RayResult*)in.batchResults;
    const U32*				triMaterialColor    = (const U32*)in.triMaterialColor;
    const U32*				triShadedColor      = (const U32*)in.triShadedColor;
    U32&                    pixel           = ((U32*)in.pixels)[primaryID];
    Vec4f                   bgColor         = Vec4f(0.2f, 0.4f, 0.8f, 1.0f);
	const Vec2f*			texCoords		= (const Vec2f*)in.texCoords;
	const Vec3f*			normals			= (const Vec3f*)in.normals;
	const Vec3i*			vertIdx			= (const Vec3i*)in.triVertIndex;
	const Vec4f*			atlasInfo		= (const Vec4f*)in.atlasInfo;
	const U32*				matId			= (const U32*)in.matId;
	const Vec4f*			matInfo			= (const Vec4f*)in.matInfo;
	Vec4f*					outColor		= (Vec4f*)in.outputColor;
	float					samples			= in.samplesCount;

	if (samples == 1.0f)
	{
		outColor[primaryID] = Vec4f(0.0f);
	}

    // Accumulate color from each ray in the batch.

    Vec4f color = Vec4f(0.0f);
    for (int i = 0; i < in.numRaysPerPrimary; i++)
    {
        int tri = batchResults[batchSlots[i]].id;					// hit index
        if (tri == -1)
		{
			if(in.isPrimary || in.isTextured || in.isPathTraced)	
			{
				color += bgColor;									// Primary: missed the scene, use background color
			}
			else				color += Vec4f(1.0f);				// AO: not blocked, use white (should be light color). Arbitrary choice for Diffuse.
		}
        else
		{
			if(in.isAO)
			{
				color += Vec4f(0,0,0,1);			// AO: blocked, use white
			}
			else if(in.isTextured)
			{
				float u = __int_as_float(primaryResult.padA);
				float v = __int_as_float(primaryResult.padB);
				float w = 1.0f - u - v;

				float tU = texCoords[vertIdx[tri].x].x * u + texCoords[vertIdx[tri].y].x * v + texCoords[vertIdx[tri].z].x * w;
				float tV = texCoords[vertIdx[tri].x].y * u + texCoords[vertIdx[tri].y].y * v + texCoords[vertIdx[tri].z].y * w;
		
				tU = tU - floorf(tU);
				tV = tV - floorf(tV);
		
				tU = tU * atlasInfo[tri].z + atlasInfo[tri].x;
				tV = tV * atlasInfo[tri].w + atlasInfo[tri].y;
				
				float4 diffuseColor;

				if(matInfo[matId[tri]].w == 0.0f)
				{
					diffuseColor = fromABGR(triMaterialColor[tri]);
					color = Vec4f(diffuseColor.x, diffuseColor.y, diffuseColor.z, 1.0f);
				}
				else
				{
					diffuseColor = tex2D(t_textures, tU, tV);
					color = Vec4f(diffuseColor.x, diffuseColor.y, diffuseColor.z, 1.0f);// * shadow * diffuse;
				}
			}
			else if(in.isPathTraced)
			{
				float u = primaryResult.t;
				float v = __int_as_float(primaryResult.padA);
				float w = __int_as_float(primaryResult.padB);
				color = Vec4f(u, v, w, 1.0f);
			}
			else
				color += fromABGR(triShadedColor[tri]);
		}
    }
    color *= 1.0f / (F32)in.numRaysPerPrimary;

    // Diffuse: modulate with primary hit color.

    int tri = primaryResult.id;
    if (in.isAO && tri == -1)   color = bgColor;
    if (in.isDiffuse)			color *= (tri == -1) ? bgColor : fromABGR(triMaterialColor[tri]);
	if(in.isPathTraced)
	{
		outColor[primaryID] = outColor[primaryID] + color;
		color = outColor[primaryID] / samples;
		color.x = powf(color.x, 1.0f / 2.2f);
		color.y = powf(color.y, 1.0f / 2.2f);
		color.z = powf(color.z, 1.0f / 2.2f);
	}
    // Write result.

    pixel = toABGR(color);
}

//------------------------------------------------------------------------

extern "C" __global__ void countHitsKernel(void)
{
    // Pick a bunch of rays for the block.

    const CountHitsInput& in = c_CountHitsInput;

    int bidx        = blockIdx.x + blockIdx.y * gridDim.x;
    int tidx        = threadIdx.x + threadIdx.y * CountHits_BlockWidth;
    int blockSize   = CountHits_BlockWidth * CountHits_BlockHeight;
    int blockStart  = bidx * blockSize * in.raysPerThread;
    int blockEnd    = ::min(blockStart + blockSize * in.raysPerThread, in.numRays);

    if (blockStart >= blockEnd)
        return;

    // Count hits by each thread.

    S32 threadTotal = 0;
    for (int i = blockStart + tidx; i < blockEnd; i += blockSize)
        if (((const RayResult*)in.rayResults)[i].id >= 0)
            threadTotal++;

    // Perform reduction within the warp.

    __shared__ volatile S32 red[CountHits_BlockWidth * CountHits_BlockHeight];
    red[tidx] = threadTotal;
    red[tidx] += red[tidx ^ 1];
    red[tidx] += red[tidx ^ 2];
    red[tidx] += red[tidx ^ 4];
    red[tidx] += red[tidx ^ 8];
    red[tidx] += red[tidx ^ 16];

    // Perform reduction within the block.

    __syncthreads();
    if ((tidx & 32) == 0)
        red[tidx] += red[tidx ^ 32];

    __syncthreads();
    if ((tidx & 64) == 0)
        red[tidx] += red[tidx ^ 64];

    __syncthreads();
    if ((tidx & 128) == 0)
        red[tidx] += red[tidx ^ 128];

    // Accumulate globally.

    if (tidx == 0)
        atomicAdd(&g_CountHitsOutput, red[tidx]);
}

//------------------------------------------------------------------------

extern "C" __global__ void getVisibility(const float4* rayResults, int numRays, int* visibility)
{
    // Pick a bunch of rays for the block.

    int bidx        = blockIdx.x + blockIdx.y * gridDim.x;
    int tidx        = threadIdx.x + threadIdx.y * Visibility_BlockWidth;
    int blockSize   = Visibility_BlockWidth * Visibility_BlockHeight;
    int blockStart  = bidx * blockSize;
    int blockEnd    = ::min(blockStart + blockSize, numRays);

    if (blockStart >= blockEnd)
        return;

    // Mark hit triangles.

    for (int i = blockStart + tidx; i < blockEnd; i += blockSize)
	{
		int hitIndex = ((const RayResult*)rayResults)[i].id;
        if (hitIndex >= 0)
            visibility[hitIndex] = 1;
	}
}

//------------------------------------------------------------------------

// For VPLs

extern "C" __device__ bool isCloserThan(const Ray& ray, Vec3f point, float value) {

	// Direction is normalized so
	float t = dot(ray.direction, point - ray.origin);
	if(t <= 0.0) {
		return false;
	}
	Vec3f closest = ray.origin + t * ray.direction;

	return (closest - point).length() <= value;

}

extern "C" __global__ void vplReconstructKernel() 
{

	const VPLReconstructInput& in = c_VPLReconstructInput;
    int taskIdx = threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * (blockIdx.x + gridDim.x * blockIdx.y));

	int                     primarySlot     = in.firstPrimary + taskIdx;

	if(primarySlot >= in.numPrimary) {
		return;
	}

    int                     primaryID       = ((const S32*)in.primarySlotToID)[primarySlot];

    const RayResult&        primaryResult   = ((const RayResult*)in.primaryResults)[primarySlot];
	const Ray&				primaryRay		= ((const Ray*)in.primaryRays)[primarySlot];

	int						shadowSlot		= taskIdx;
	const RayResult&        shadowResult	= ((const RayResult*)in.shadowResults)[taskIdx];

	U32&                    pixel           = ((U32*)in.pixels)[primaryID];
	Vec4f                   bgColor         = Vec4f(0.2f, 0.4f, 0.8f, 1.0f);
	const Vec3i*			vertIdx			= (const Vec3i*)in.triVertIndex;
	const Vec3f*			normals			= (const Vec3f*)in.normals;
	const Vec3f*			vertices		= (const Vec3f*)in.vertices;

	const Light&			light			= ((const Light*)in.lights)[in.currentLight];
	const Vec3f				lightPos		= light.position;
	int						lightCount		= in.lightCount;
	
	if(isCloserThan(primaryRay, lightPos, 0.1)) {
		pixel = toABGR(Vec4f(1,0,0,1));
		return;
	}

	int tri = primaryResult.id;
	if(tri == -1) {
		pixel = toABGR(bgColor);
		return;
	}


	float u = __int_as_float(primaryResult.padA);
	float v = __int_as_float(primaryResult.padB);
	float w = 1.0f - u - v;

	Vec4f color = Vec4f(0,0,0,1);

	Vec3f normal = (normals[vertIdx[tri].x] * u + normals[vertIdx[tri].y] * v + normals[vertIdx[tri].z] * w).normalized();
	Vec3f position = vertices[vertIdx[tri].x] * u + vertices[vertIdx[tri].y] * v + vertices[vertIdx[tri].z] * w;

	Vec3f	lightDir = (lightPos - position);
	float invDist = 1.0f / lightDir.length();
	lightDir = lightDir.normalized();


	float nDotDir = dot(normal, lightDir);

	if(nDotDir > 0 && (!in.shadow || shadowResult.id == -1)) {
		color += nDotDir * Vec4f(light.intensity,0);
	}
	

	Vec4f prevPixel = fromABGR(pixel);

	pixel = toABGR(prevPixel + color / (in.shadow ? lightCount : 10));
}
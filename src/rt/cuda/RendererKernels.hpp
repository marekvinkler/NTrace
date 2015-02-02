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

#pragma once
#include "base/DLLImports.hpp"
#include "Util.hpp"

namespace FW
{
//------------------------------------------------------------------------

enum
{
    CountHits_BlockWidth    = 32,
    CountHits_BlockHeight   = 8,
};

enum
{
    Visibility_BlockWidth    = 32,
    Visibility_BlockHeight   = 8,
};

//------------------------------------------------------------------------

struct ReconstructInput
{
    S32         numRaysPerPrimary;
    S32         firstPrimary;
    S32         numPrimary;
    bool        isPrimary;
    bool        isAO;
    bool        isDiffuse;
    bool        isTextured;
	bool		isPathTraced;
    CUdeviceptr primarySlotToID;    // const S32*
    CUdeviceptr primaryResults;     // const RayResult*
    CUdeviceptr batchIDToSlot;      // const S32*
    CUdeviceptr batchResults;       // const RayResult*
    CUdeviceptr triMaterialColor;   // const U32* ABGR
    CUdeviceptr triShadedColor;     // const U32* ABGR
    CUdeviceptr pixels;             // U32* ABGR

	CUdeviceptr texCoords;			// const Vec2f*
	CUdeviceptr normals;			// const Vec3f*
	CUdeviceptr triVertIndex;		// const Vec3i*
	CUdeviceptr atlasInfo;			// const Vec4f*
	CUdeviceptr matId;				// const U32*
	CUdeviceptr matInfo;			// const Vec4f*
	CUdeviceptr outputColor;		// const Vec4f*
	float		samplesCount;	
};


//------------------------------------------------------------------------

struct VPLReconstructInput {
    S32         numPrimary;
	S32         firstPrimary;
	S32			lightCount;
	S32			currentLight;
	CUdeviceptr lights;				// const Light*

    CUdeviceptr primarySlotToID;    // const S32*
    CUdeviceptr primaryResults;     // const RayResult*
	CUdeviceptr primaryRays;		// const Ray*
    CUdeviceptr pixels;             // U32* ABGR

	CUdeviceptr shadowResults;		// const RayResult*
	CUdeviceptr shadowIdToSlot;     // const S32*


	CUdeviceptr texCoords;			// const Vec2f*
	CUdeviceptr normals;			// const Vec3f*
	CUdeviceptr triVertIndex;		// const Vec3i*
	CUdeviceptr vertices;	     	// const Vec3i*
	CUdeviceptr triShadedColor;	   
	bool		shadow;
};


//------------------------------------------------------------------------

struct CountHitsInput
{
    S32         numRays;
    CUdeviceptr rayResults;         // const RayResult*
    S32         raysPerThread;
};

//------------------------------------------------------------------------

#if FW_CUDA
extern "C"
{

__constant__ ReconstructInput c_ReconstructInput;
__global__ void reconstructKernel(void);
__constant__ VPLReconstructInput c_VPLReconstructInput;
__global__ void vplReconstructKernel(void);

__constant__ CountHitsInput c_CountHitsInput;
__device__ S32 g_CountHitsOutput;
__global__ void countHitsKernel(void);

__global__ void getVisibility(const float4* rayResults, int numRays, int* visibility);

texture<float4, 2> t_textures;

}
#endif

//------------------------------------------------------------------------
}

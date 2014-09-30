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

#include "ray/RayGen.hpp"
#include "ray/RayGenKernels.hpp"
#include "base/Random.hpp"

namespace FW
{

RayGen::RayGen(S32 maxBatchSize)
:   m_maxBatchSize(maxBatchSize)
{
    m_compiler.setSourceFile("src/rt/ray/RayGenKernels.cu");
    m_compiler.addOptions("-use_fast_math");
    m_compiler.include("src/rt");
    m_compiler.include("src/framework");
}

void RayGen::primary(RayBuffer& orays, const Vec3f& origin, const Mat4f& nscreenToWorld, S32 w,S32 h,float maxDist, U32 randomSeed)
{
    // doesn't do batching

    m_pixelTable.setSize(Vec2i(w, h));
    orays.resize(w * h);
    orays.setNeedClosestHit(true);

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Setup input struct.

    RayGenPrimaryInput& in  = *(RayGenPrimaryInput*)module->getGlobal("c_RayGenPrimaryInput").getMutablePtr();
    in.origin               = origin;
    in.nscreenToWorld       = nscreenToWorld;
    in.w                    = w;
    in.h                    = h;
    in.maxDist              = maxDist;
    in.rays                 = orays.getRayBuffer().getMutableCudaPtr();
    in.idToSlot             = orays.getIDToSlotBuffer().getMutableCudaPtr();
    in.slotToID             = orays.getSlotToIDBuffer().getMutableCudaPtr();
    in.indexToPixel         = m_pixelTable.getIndexToPixel().getCudaPtr();
    in.randomSeed			= Random(randomSeed).getU32();

    // Launch.

    module->getKernel("rayGenPrimaryKernel").launch(orays.getSize());
}

void RayGen::primaryCPU(RayBuffer& orays, const Vec3f& origin, const Mat4f& nscreenToWorld, S32 w,S32 h,float maxDist)
{
    // doesn't do batching

    m_pixelTable.setSize(Vec2i(w, h));
    orays.resize(w * h);
    orays.setNeedClosestHit(true);

    // Compute end position.
	for(S32 i = 0; i < orays.getSize(); i++)
	{
		int pixel = ((const S32*)m_pixelTable.getIndexToPixel().getMutablePtr())[i];
		int posIdx = pixel;
	//    int posIdx = ((const S32*)in.indexToPixel)[(taskIdx & 31) + 1024 * 768 / 2 + 1234];
	//    int posIdx = ((const S32*)in.indexToPixel)[(taskIdx & 0) + 1024 * 768 / 2 + 1234];

		Vec4f nscreenPos;
		nscreenPos.x = 2.0f * ((F32)(posIdx % w) + 0.5f) / (F32)w - 1.0f;
		nscreenPos.y = 2.0f * ((F32)(posIdx / w) + 0.5f) / (F32)h - 1.0f;
		nscreenPos.z = 0.0f;
		nscreenPos.w = 1.0f;

		Vec4f worldPos4D = nscreenToWorld * nscreenPos;
		Vec3f worldPos = worldPos4D.getXYZ() / worldPos4D.w;

		// Write results.

		Ray& ray = ((Ray*)orays.getRayBuffer().getMutablePtr())[i];
		((S32*)orays.getSlotToIDBuffer().getMutablePtr())[i] = pixel;
		((S32*)orays.getIDToSlotBuffer().getMutablePtr())[pixel] = i;

		ray.origin      = origin;
		ray.direction   = (worldPos - origin).normalized();
		ray.tmin        = 0.0f;
		ray.tmax        = maxDist;
	}
}

bool RayGen::shadow(RayBuffer& orays, RayBuffer& irays, int numSamples, const Vec3f& lightPos, float lightRadius, bool& newBatch, U32 randomSeed)
{
    // batching
    S32 lo,hi;
    if( !batching(irays.getSize(),numSamples, m_shadowStartIdx,newBatch, lo,hi) )
        return false;

    // allocate output array
    orays.resize((hi-lo)*numSamples);
    orays.setNeedClosestHit(false);

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Setup input struct.

    RayGenShadowInput& in   = *(RayGenShadowInput*)module->getGlobal("c_RayGenShadowInput").getMutablePtr();
    in.firstInputSlot   = lo;
    in.numInputRays     = hi - lo;
    in.numSamples       = numSamples;
    in.lightPositionX   = lightPos.x;
    in.lightPositionY   = lightPos.y;
    in.lightPositionZ   = lightPos.z;
	in.lightRadius		= lightRadius;
    in.randomSeed       = Random(randomSeed).getU32();
    in.inRays           = irays.getRayBuffer().getCudaPtr();
    in.inResults        = irays.getResultBuffer().getCudaPtr();
    in.outRays          = orays.getRayBuffer().getMutableCudaPtr();
    in.outIDToSlot      = orays.getIDToSlotBuffer().getMutableCudaPtr();
    in.outSlotToID      = orays.getSlotToIDBuffer().getMutableCudaPtr();

    // Launch.

	module->getKernel("rayGenShadowKernel").launch(in.numInputRays);
    return true;
}

bool RayGen::shadowCPU(RayBuffer& orays, RayBuffer& irays, int numSamples, const Vec3f& lightPos, float lightRadius, bool& newBatch, U32 randomSeed)
{
    const float epsilon = 1e-3f;

    // batching
    S32 lo,hi;
    if( !batching(irays.getSize(), numSamples, m_shadowStartIdx, newBatch, lo, hi) )
        return false;

    // allocate output array
    const S32 numOutputRays = (hi-lo)*numSamples;
    orays.resize(numOutputRays);
    Random rnd(randomSeed);

    // raygen
    for(int i=lo;i<hi;i++)
    {
        const Ray& iray = irays.getRayForSlot(i);
        const RayResult& irayres = irays.getResultForSlot(i);

        const float t = max(0.f,(irayres.t-epsilon));               // backtrack a little bit
        const Vec3f origin = iray.origin + t*iray.direction;

        for(int j=0;j<numSamples;j++)
        {
            Vec3f target = lightPos;
            Vec3f direction = target - origin;

            Ray oray;
            oray.origin     = origin;
            oray.direction  = direction.normalized();
            oray.tmin       = 0.f;
            oray.tmax       = direction.length();

            if(!irayres.hit())
                oray.degenerate();

            const S32 oindex = (i-lo)*numSamples+j;
            orays.setRay(oindex, oray);
        }
    }

    orays.setNeedClosestHit(false);
    return true;
}

bool RayGen::ao(RayBuffer& orays, RayBuffer& irays, Scene& scene, int numSamples, float maxDist, bool& newBatch, U32 randomSeed)
{
    // Perform batching and setup output array.

    S32 lo, hi;
    if(!batching(irays.getSize(), numSamples, m_aoStartIdx, newBatch, lo, hi))
        return false;

    orays.resize((hi - lo) * numSamples);
    orays.setNeedClosestHit(false);

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Setup input struct.

    RayGenAOInput& in   = *(RayGenAOInput*)module->getGlobal("c_RayGenAOInput").getMutablePtr();
    in.firstInputSlot   = lo;
    in.numInputRays     = hi - lo;
    in.numSamples       = numSamples;
    in.maxDist          = maxDist;
    in.randomSeed       = Random(randomSeed).getU32();
    in.inRays           = irays.getRayBuffer().getCudaPtr();
    in.inResults        = irays.getResultBuffer().getCudaPtr();
    in.outRays          = orays.getRayBuffer().getMutableCudaPtr();
    in.outIDToSlot      = orays.getIDToSlotBuffer().getMutableCudaPtr();
    in.outSlotToID      = orays.getSlotToIDBuffer().getMutableCudaPtr();
    in.normals          = scene.getTriNormalBuffer().getCudaPtr();

    // Launch.

	module->getKernel("rayGenAOKernel").launch(in.numInputRays);
    return true;
}

bool RayGen::aoCPU(RayBuffer& orays, RayBuffer& irays, Scene& scene, int numSamples, float maxDist, bool& newBatch, U32 randomSeed)
{
	/*const float epsilon = 1e-3f;

    // Perform batching and setup output array.

    S32 lo, hi;
    if(!batching(irays.getSize(), numSamples, m_aoStartIdx, newBatch, lo, hi))
        return false;

    orays.resize((hi - lo) * numSamples);
    orays.setNeedClosestHit(false);

	Random rnd(randomSeed);
	const Scene::Triangle*  triangles   = (const Scene::Triangle*)scene.getTrianglePtr();

    // raygen
    for(int i=lo;i<hi;i++)
    {
        const Ray& iray = irays.getRayForSlot(i);
        const RayResult& irayres = irays.getResultForSlot(i);

        const float t = max(0.f,(irayres.t-epsilon));               // backtrack a little bit
        const Vec3f origin = iray.origin + t*iray.direction;

		// Lookup normal, flipping back-facing directions.

		int tri = irayres.id;
		Vec3f normal(1.0f, 0.0f, 0.0f);
		if (tri != -1)
			normal = triangles[tri].normal;
		if (dot(normal, iray.direction) > 0.0f)
			normal = -normal;

		// Construct perpendicular vectors.

		Vec3f na = abs(normal);
		F32 nm = max(max(na.x, na.y), na.z);
		Vec3f perp(normal.y, -normal.x, 0.0f); // assume y is largest
		if (nm == na.z)
			perp = Vec3f(0.0f, normal.z, -normal.y);
		else if (nm == na.x)
			perp = Vec3f(-normal.z, 0.0f, normal.x);

		perp = normalize(perp);
		Vec3f biperp = cross(normal, perp);

		// Pick random rotation angle.

		F32 angle = 2.0f * FW_PI * rnd.getF32(-1.0f, 1.0f);

		// Construct rotated tangent vectors.

		Vec3f t0 = perp * cosf(angle) + biperp * sinf(angle);
		Vec3f t1 = perp * -sinf(angle) + biperp * cosf(angle);

        for(int j=0;j<numSamples;j++)
        {
            // Base-2 Halton sequence for X.

			F32 x = 0.0f;
			F32 xadd = 1.0f;
			unsigned int hc2 = j + 1;
			while (hc2 != 0)
			{
				xadd *= 0.5f;
				if ((hc2 & 1) != 0)
					x += xadd;
				hc2 >>= 1;
			}

			// Base-3 Halton sequence for Y.

			F32 y = 0.0f;
			F32 yadd = 1.0f;
			int hc3 = j + 1;
			while (hc3 != 0)
			{
				yadd *= 1.0f / 3.0f;
				y += (F32)(hc3 % 3) * yadd;
				hc3 /= 3;
			}

			// Warp to a point on the unit hemisphere.

			F32 angle = 2.0f * FW_PI * y;
			F32 r = sqrtf(x);
			x = r * cosf(angle);
			y = r * sinf(angle);
			float z = sqrtf(1.0f - x * x - y * y);

			// Output ray.

			Ray oray;
            oray.origin     = origin;
            oray.direction  = normalize(x * t0 + y * t1 + z * normal);
            oray.tmin       = 0.0f;
            oray.tmax       = (tri == -1) ? -1.0f : maxDist;

            const S32 oindex = (i-lo)*numSamples+j;
            orays.setRay(oindex, oray);
        }
    }

    orays.setNeedClosestHit(false);*/
    return true;
}

bool RayGen::random (RayBuffer& orays, const AABB& bounds, int numRays, bool closestHit, bool PosDir, U32 randomSeed)
{
    bool temp = true;
    return random(orays,bounds,numRays,closestHit, temp, PosDir, randomSeed);
}

bool RayGen::random (RayBuffer& orays, const AABB& bounds, int numRays, bool closestHit, bool PosDir, bool& newBatch, U32 randomSeed)
{
    S32 lo,hi;
    if( !batching(numRays,1, m_randomStartIdx, newBatch, lo,hi) )
        return false;

    const S32 numOutputRays = (hi-lo);
    orays.resize(numOutputRays);
    Random rnd(randomSeed);

    for(int i=0;i<numRays;i++)
    {
        Vec3f a = rnd.getVec3f(0.0f, 1.0f);
        Vec3f b = rnd.getVec3f(0.0f, 1.0f);

        Ray oray;
        oray.origin    = bounds.min() + a*(bounds.max() - bounds.min());
        if(PosDir)  oray.direction = b.normalized() * (bounds.max() - bounds.min()).length();        // position, direction
        else        oray.direction = bounds.min() + b*(bounds.max() - bounds.min()) - oray.origin;  // position, position
        oray.tmin      = 0.f;
        oray.tmax      = 1.f;
        orays.setRay(i,oray);
    }

    orays.setNeedClosestHit(closestHit);
    return true;
}

bool RayGen::randomReflection (RayBuffer& orays, RayBuffer& irays, Scene& scene, int numSamples, float maxDist, bool& newBatch, U32 randomSeed)
{
    const float epsilon = 1e-4f;

    // batching
    S32 lo,hi;
    if( !batching(irays.getSize(),numSamples, m_randomStartIdx,newBatch, lo,hi) )
        return false;

    // allocate output array
    const S32 numOutputRays = (hi-lo)*numSamples;
    orays.resize(numOutputRays);
    Random rnd(randomSeed);

    // raygen
    const Vec3f* normals = (const Vec3f*)scene.getTriNormalBuffer().getPtr();
    for(int i=lo;i<hi;i++)
    {
        const Ray& iray = irays.getRayForSlot(i);
        const RayResult& irayres = irays.getResultForSlot(i);

        const float t = max(0.f,(irayres.t-epsilon));               // backtrack a little bit
        const Vec3f origin = iray.origin + t*iray.direction;
        Vec3f normal = irayres.hit() ? normals[irayres.id] : Vec3f(0.f);
        if(dot(normal,iray.direction) > 0.f)
            normal = -normal;

        for(int j=0;j<numSamples;j++)
        {
            Ray oray;

            if(irayres.hit())
            {
                oray.origin     = origin;

                do{
                    oray.direction.x = rnd.getF32();
                    oray.direction.y = rnd.getF32();
                    oray.direction.z = rnd.getF32();
                    oray.direction.normalize();
                } while(dot(oray.direction,normal)<0.f);

                oray.tmin       = 0.f;
                oray.tmax       = maxDist;
            }
            else
                oray.degenerate();

            const S32 oindex = (i-lo)*numSamples+j;
            orays.setRay(oindex, oray);
        }
    }

    orays.setNeedClosestHit(false);
    return true;
}

bool RayGen::batching(S32 numInputRays,S32 numSamples,S32& startIdx,bool& newBatch, S32& lo,S32& hi)
{
    if(newBatch)
    {
        newBatch = false;
        startIdx = 0;
    }

    if(startIdx == numInputRays)
        return false;   // finished

    // current index [lo,hi) in *input* array
    lo = startIdx;
    hi = min(numInputRays, lo+m_maxBatchSize/numSamples);

    // for the next round
    startIdx = hi;
    return true;        // continues
}

} //

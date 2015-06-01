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
#include "3d/Light.hpp"

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
    in.randomSeed			= (randomSeed != 0) ? Random(randomSeed).getU32() : 0;

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

Vec3f hemisphereDirection(float a, float b) {
	float azimuth = 2 * FW_PI * a;
	float elevation = FW_PI * 0.5 * b;
	return Vec3f(sin(azimuth) * cos(elevation), sin(elevation), cos(azimuth) * cos(elevation));
}

bool RayGen::primaryVPL(Buffer& lights, RayBuffer& orays, Scene* scene, int numLights, int maxBounces, float maxDist, U32 randomSeed) {

	// Lets generate this on CPU, won't be that many of the primary lights
	orays.resize(numLights);
    orays.setNeedClosestHit(true);

	lights.resize((numLights * (maxBounces + 2)) * sizeof(Light));

	Ray* outRayBuffer = (Ray *)orays.getRayBuffer().getMutablePtr();
	Light* outLightBuffer = (Light*)lights.getMutablePtr();

	Vec3i* emissiveTris = (Vec3i*) scene->getEmissiveTris().getPtr();
	const Vec3f* vertices = (Vec3f*) scene->getVtxPosBuffer().getPtr();
	const Vec3f* normals = (Vec3f*) scene->getVtxNormalBuffer().getPtr();
	const U32* matId = (U32*) scene->getMaterialIds().getPtr();
	const Vec4f* matInfo = (Vec4f*) scene->getMaterialInfo().getPtr();

	S64 emissiveTrisCount = scene->getEmissiveTris().getSize() / sizeof(Vec3i);

	if(emissiveTrisCount == 0) {
		printf("No emissive triangles found!\n");
		return true;
	}
	
	float emissiveTrisArea = 0;

	for(int i = 0; i < emissiveTrisCount; i++) {
		Vec3i triangle = emissiveTris[i];
		Vec3f v1 = vertices[triangle.y] - vertices[triangle.x];
		Vec3f v2 = vertices[triangle.z] - vertices[triangle.x];
		emissiveTrisArea += v1.cross(v2).length() * 0.5;
	}

	Random rnd(randomSeed);

	for(int i = 0; i < numLights; i++) {
		Vec3i triangle = emissiveTris[rnd.getU32(0, emissiveTrisCount)];
		
		float sqrtr1 = sqrt(Random::halton(2, i + 1)); 
		float r2 = Random::halton(3, i + 1);
		float u = 1 - sqrtr1;
		float v = sqrtr1 * (1 - r2);
		float w = r2 * sqrtr1;

		Vec3f v1 = vertices[triangle.y] - vertices[triangle.x];
		Vec3f v2 = vertices[triangle.z] - vertices[triangle.x];
		Vec3f normal = v1.cross(v2).normalized();
		Vec3f dir2Vec = normal.cross(v1).normalized();
		Vec3f hemiDirection = hemisphereDirection(Random::halton(3, i + 2), Random::halton(2, i + 2));
		outRayBuffer[i].direction = (normal * hemiDirection.y + v1.normalized() * hemiDirection.x + dir2Vec * hemiDirection.z).normalized();
		outRayBuffer[i].tmin = 0.0f;
		outRayBuffer[i].tmax = maxDist;

		Vec3f origin = vertices[triangle.x] * u + vertices[triangle.y] * v + vertices[triangle.z] * w + outRayBuffer[i].direction * 0.001;
		outRayBuffer[i].origin = origin;
		outLightBuffer[i].position = origin;
		/*
		* The following depnds on the intensity of the light. 
		* However the getEmissiveTris() function returns directly triplets of indices into vertex
		* array and not indices into triangle array so we cannot determine the triangle material.
		*/
		outLightBuffer[i].intensity = Vec3f(30,30,30);

	}

	return true;
   
}

bool RayGen::reflectedVPL(Buffer& lights, RayBuffer& rays, int numPrimaryLights, int iteration, Scene* scene, float maxDist) {

	const float epsilon = 1e-1f;

	Ray* rayBuffer = (Ray *)rays.getRayBuffer().getMutablePtr();
	RayResult* rayResults = (RayResult *)rays.getResultBuffer().getPtr();
	const Vec3i* indices = (Vec3i*) scene->getTriVtxIndexBuffer().getPtr();
	const Vec3f* normals = (Vec3f*) scene->getVtxNormalBuffer().getPtr();
	const Vec3f* vertices = (Vec3f*) scene->getVtxPosBuffer().getPtr();
	const Vec2f* texCoords = (Vec2f*) scene->getVtxTexCoordBuffer().getPtr();
	const Vec4f* atlasInfo = (Vec4f*) scene->getTextureAtlasInfo().getPtr();
	const U32* matId = (U32*) scene->getMaterialIds().getPtr();
	const Vec4f* matInfo = (Vec4f*) scene->getMaterialInfo().getPtr();
	const U32* triMaterialColor = (U32*) scene->getTriMaterialColorBuffer().getPtr();
	const Image* atlasTexture = scene->getTextureAtlas()->getAtlasTexture().getImage();

	Light* lightBuffer = ((Light*)lights.getMutablePtr()) + (iteration + 1) * numPrimaryLights;

	for(int i = 0; i < numPrimaryLights; i++) {
		int tri = rayResults[i].id;

		if(tri == -1) {
			lightBuffer[i].intensity = Vec3f(0,0,0);
			lightBuffer[i].position = Vec3f(0,0,0);
		} else {
			lightBuffer[i].position = rayBuffer[i].origin + rayBuffer[i].direction * (rayResults[i].t - epsilon);

			rayBuffer[i].origin = lightBuffer[i].position;

			float u = *((float*)&(rayResults[i].padA));
			float v = *((float*)&(rayResults[i].padB)); 
			float w = 1.0f - u - v;
			

			float tU = texCoords[indices[tri].x].x * u + texCoords[indices[tri].y].x * v + texCoords[indices[tri].z].x * w;
			float tV = texCoords[indices[tri].x].y * u + texCoords[indices[tri].y].y * v + texCoords[indices[tri].z].y * w;

			tU = tU - floorf(tU);
			tV = tV - floorf(tV);
		
			tU = tU * atlasInfo[tri].z + atlasInfo[tri].x;
			tV = tV * atlasInfo[tri].w + atlasInfo[tri].y;

			Vec3f texColor;

			if(matInfo[matId[tri]].w == 0.0f) {
				Vec4f diffuseColor = Vec4f::fromABGR(triMaterialColor[tri]);
				texColor = Vec3f(diffuseColor.x, diffuseColor.y, diffuseColor.z);
			} else {
				Vec4f diffuseColor = atlasTexture->getVec4fLinear(Vec2f(tU, tV));
				texColor = Vec3f(diffuseColor.x, diffuseColor.y, diffuseColor.z);
			}

			Vec3f normal = (normals[indices[tri].x] * u + normals[indices[tri].y] * v + normals[indices[tri].z] * w).normalized();
			Vec3f v1 = vertices[indices[tri].y] - vertices[indices[tri].x];
			Vec3f dir2Vec = normal.cross(v1).normalized();
			Vec3f hemiDirection = hemisphereDirection(Random::halton(2, numPrimaryLights + 1 - i), Random::halton(3, numPrimaryLights + 1 - i));

			lightBuffer[i].intensity = lightBuffer[i - numPrimaryLights].intensity * max(0.0f, dot(normal, -rayBuffer[i].direction)) * texColor;

			rayBuffer[i].direction = (normal * hemiDirection.y + v1.normalized() * hemiDirection.x + dir2Vec * hemiDirection.z).normalized();
			rayBuffer[i].tmin = 0.0f;
			rayBuffer[i].tmax = maxDist;

		}
	}

	rays.setNeedClosestHit(true);

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

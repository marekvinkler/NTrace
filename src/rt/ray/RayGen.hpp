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

/**
 * \file
 * \brief Definitions for the ray generator class.
 */

#pragma once
#include "base/Math.hpp"
#include "base/Array.hpp"
#include "gpu/CudaCompiler.hpp"
#include "ray/RayBuffer.hpp"
#include "ray/PixelTable.hpp"
#include "Scene.hpp"

namespace FW
{

/**
 * \brief Ray generator class. Generates rays for both the GPU and the CPU.
 */
class RayGen
{
public:

	/**
	 * \brief Constructor.
	 * \param[in] maxBatchSize Maximum number of secondary rays in one batch.
	 */
    RayGen(S32 maxBatchSize = 8*1024*1024);

    // true if batch continues
	/**
	 * \brief Generates primary rays on the GPU.
	 * \param[in,out] Generated rays.
	 * \param[in] origin Point from which rays will be cast.
	 * \param[in] nscreenToWorld Transformation view matrix.
	 * \param[in] w Horizontal resolution.
	 * \param[in] h Vertical resolution.
	 * \param[in] maxDist Maximum length of generated rays.
	 */
    void    primary   (RayBuffer& orays, const Vec3f& origin, const Mat4f& nscreenToWorld, S32 w,S32 h,float maxDist, U32 randomSeed);

	/**
	 * \brief Generates shadow rays on the GPU. Batches rays if necessary.
	 * \param[in,out] orays Generated rays.
	 * \param[in] irays Source rays.
	 * \param[in] numSamples Number of shadow rays for each source ray.
	 * \param[in] lightPos Position of a light source that casts these shadow rays.
	 * \param[in] lightRadius Radius of the light source.
	 * \param[in,out] newBatch Batching flag. 
	 * \param[in] randomSeed Random seed.
	 * \return True if batch continues.
	 */
    bool    shadow    (RayBuffer& orays, RayBuffer& irays, int numSamples, const Vec3f& lightPos, float lightRadius, bool& newBatch, U32 randomSeed=0);

	/**
	 * \brief Generates ao rays on the GPU. Batches rays if necessary.
	 * \param[in,out] orays Generated rays.
	 * \param[in] irays Source rays.
	 * \param[in] scene Source scene.
	 * \param[in] numSamples Number of ao rays for each source ray.
	 * \param[in] maxDist Maximum length of generated rays.
	 * \param[in,out] newBatch Batching flag.
	 * \[aram[in] randomSeed Random seed.
	 * \return True if batch contiunes.
	 */
    bool    ao        (RayBuffer& orays, RayBuffer& irays, Scene& scene, int numSamples, float maxDist, bool& newBatch, U32 randomSeed=0); // non-const because of Buffer transfers

	/**
	 * \brief Generates primary rays on the CPU.
	 * \param[in,out] Generated rays.
	 * \param[in] origin Point from which rays will be cast.
	 * \param[in] nscreenToWorld Transformation view matrix.
	 * \param[in] w Horizontal resolution.
	 * \param[in] h Vertical resolution.
	 * \param[in] maxDist Maximum length of generated rays.
	 */
	void    primaryCPU(RayBuffer& orays, const Vec3f& origin, const Mat4f& nscreenToWorld, S32 w,S32 h,float maxDist);

	/**
	 * \brief Generates shadow rays on the CPU. Batches rays if necessary.
	 * \param[in,out] orays Generated rays.
	 * \param[in] irays Source rays.
	 * \param[in] numSamples Number of shadow rays for each source ray.
	 * \param[in] lightPos Position of a light source that casts these shadow rays.
	 * \param[in] lightRadius Radius of the light source.
	 * \param[in,out] newBatch Batching flag. 
	 * \param[in] randomSeed Random seed.
	 * \return True if batch continues.
	 */
    bool    shadowCPU (RayBuffer& orays, RayBuffer& irays, int numSamples, const Vec3f& lightPos, float lightRadius, bool& newBatch, U32 randomSeed=0);

	/**
	 * \brief Generates ao rays on the CPU. Batches rays if necessary.
	 * \param[in,out] orays Generated rays.
	 * \param[in] irays Source rays.
	 * \param[in] scene Source scene.
	 * \param[in] numSamples Number of ao rays for each source ray.
	 * \param[in] maxDist Maximum length of generated rays.
	 * \param[in,out] newBatch Batching flag.
	 * \param[in] randomSeed Random seed.
	 * \return True if batch contiunes.
	 */
    bool    aoCPU     (RayBuffer& orays, RayBuffer& irays, Scene& scene, int numSamples, float maxDist, bool& newBatch, U32 randomSeed=0); // non-const because of Buffer transfers

     // these are hack for various tests
	/**
	 * \brief Generates random rays. Used for various tests.
	 * \param[in,out] orays Generated rays.
	 * \param[in] bounds Box in which rays will be generated.
	 * \param[in] numRays Number of generated rays.
	 * \param[in] closestHit Flag whether rays require closest hit.
	 * \param[in] PosDir If false, direction of generated rays will be absolute position in the scene.
	 * \param[in] randomSeed Random seed.
	 * \return True if batch continues.
	 */
    bool    random (RayBuffer& orays, const AABB& bounds, int numRays, bool closestHit, bool PosDir=false, U32 randomSeed=0);

	/**
	 * \brief Generates random rays.
	 * \param[in,out] orays Generated rays.
	 * \param[in] bounds Box in which rays will be generated.
	 * \param[in] numRays Number of generated rays.
	 * \param[in] closestHit Flag whether rays require closest hit.
	 * \param[in] PosDir If false, direction of generated rays will be absolute position in the scene.
	 * \param[in,out] newBatch Batching flag.
	 * \param[in] randomSeed Random seed.
	 * \return True if batch continues.
	 */
    bool    random (RayBuffer& orays, const AABB& bounds, int numRays, bool closestHit, bool PosDir, bool& newBatch, U32 randomSeed=0);

	/**
	 * \brief Generates random reflection rays.
	 * \param[in,out] orays Generated rays.
	 * \param[in] irays Source rays.
	 * \param[in] scene Source scene, necessary for normals.
	 * \param[in] numSamples Number of reflection rays for each source ray.
	 * \param[in] maxDist Maximum length of reflection rays.
	 * \param[in,out] newBatch Batching flag.
	 * \param[in] randomSeed Random seed.
	 * \return True if batch continues.
	 */
    bool    randomReflection (RayBuffer& orays, RayBuffer& irays, Scene& scene, int numSamples, float maxDist, bool& newBatch, U32 randomSeed=0);


	/**
	 * \brief Generates rays on the surface of light - the area is defined by base, two vectors and a normal.
	 * \param[in,out] orays Generated rays.
	 * \param[in] emitPlaneBase The base point of the emitting plane
	 */
	bool    primaryVPL(Buffer& lights, RayBuffer& orays, Vec3f& emitPlaneBase, Vec3f& emitPlaneV1, Vec3f& emitPlaneV2, Vec3f& emitPlaneNormal, int numLights, int maxBounces, float maxDist, U32 randomSeed=0);

	bool	reflectedVPL(Buffer& lights, RayBuffer& rays, int numPrimaryLights, int iteration, Scene* scene, float maxDist);

private:
	/**
	 * \brief Creates batches of rays to limit memory load.
	 * \param[in] numInputRays Number of input rays.
	 * \param[in] numSamples Number of samples for each input ray.
	 * \param[in,out] startIdx Index from where to start batching.
	 * \param[in,out] Flag whether current batch is a new batch.
	 * \param[in,out] lo Start of current batch.
	 * \param[out] hi End of current batch.
	 * \return True if batch is not complete.
	 */
    bool    batching(S32 numInputRays,S32 numSamples,S32& startIdx,bool& newBatch, S32& lo,S32& hi);

    S32             m_maxBatchSize;			//!< Maximum size of batch.
    CudaCompiler    m_compiler;				//!< CUDA compiler.
    PixelTable      m_pixelTable;			//!< Pixel table.

    S32             m_shadowStartIdx;		//!< Start index for shadow ray batching.
    S32             m_aoStartIdx;			//!< Start index for ao ray batching.
    S32             m_randomStartIdx;		//!< Start index for random ray batching.
};

} //

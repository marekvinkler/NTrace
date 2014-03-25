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

#include "cuda/CPURenderer.hpp"

using namespace FW;

//------------------------------------------------------------------------

CPURenderer::CPURenderer(void)
:   Renderer(Renderer::tBVH),
m_layout(BVHLayout_CPU)
{
    m_platform = Platform("CPU");
    m_platform.setLeafPreferences(1, 8);
}

//------------------------------------------------------------------------

CPURenderer::~CPURenderer(void)
{
}

//------------------------------------------------------------------------

void CPURenderer::setParams(const Params& params)
{
    m_params = params;
}

//------------------------------------------------------------------------

bool CPURenderer::nextBatch(void)
{
    FW_ASSERT(m_scene);

    // Clean up the previous batch.

    if (m_batchRays && !m_newBatch)
        m_batchStart += m_batchRays->getSize();
    //m_batchRays = NULL;

    // Generate new batch.

    U32 randomSeed = (m_enableRandom) ? m_random.getU32() : 0;
    switch (m_params.rayType)
    {
    case RayType_Primary:
        if (!m_newBatch)
            return false;
        m_newBatch = false;
        m_batchRays = &m_primaryRays;
        break;

    case RayType_AO:
        if (!m_raygen.aoCPU(m_secondaryRays, m_primaryRays, *m_scene, m_params.numSamples, m_params.aoRadius, m_newBatch, randomSeed))
            return false;
        m_batchRays = &m_secondaryRays;
        break;

    case RayType_Diffuse:
        if (!m_raygen.aoCPU(m_secondaryRays, m_primaryRays, *m_scene, m_params.numSamples, m_cameraFar, m_newBatch, randomSeed))
            return false;
        m_secondaryRays.setNeedClosestHit(true);
        m_batchRays = &m_secondaryRays;
        break;

    default:
        FW_ASSERT(false);
        return false;
    }

    // Sort rays.

    if (m_params.sortSecondary && !(m_params.rayType == RayType_Primary))
        m_batchRays->mortonSort();
    return true;
}

//------------------------------------------------------------------------

F32 CPURenderer::traceBatch(RayStats* stats)
{
    FW_ASSERT(m_batchRays);
	((CudaBVH*)m_accelStruct)->setTraceParams(&m_platform, m_scene);
	Timer timer(true);
	if(stats == NULL)
	{
		RayStats stats;
		if(m_buildParams.empty_boxes.getSize())
		{
			//m_bvh->trace(*m_batchRays, *m_emptybvh, &stats);
			((CudaBVH*)m_accelStruct)->trace(*m_batchRays, m_visibility, m_buildParams.empty_boxes, &stats);
		}
		else
		{
			((CudaBVH*)m_accelStruct)->trace(*m_batchRays, m_visibility, m_buildParams.twoTrees, &stats);
		}
	}
	else
	{
		if(m_buildParams.empty_boxes.getSize())
		{
			//m_bvh->trace(*m_batchRays, *m_emptybvh, stats);
			((CudaBVH*)m_accelStruct)->trace(*m_batchRays, m_visibility, m_buildParams.empty_boxes, ((CudaBVH*)m_accelStruct)->m_stats);
		}
		else
		{
			((CudaBVH*)m_accelStruct)->trace(*m_batchRays, m_visibility, m_buildParams.twoTrees, ((CudaBVH*)m_accelStruct)->m_stats);
		}
	}
	return timer.getElapsed(); // Trace time in seconds
}

//------------------------------------------------------------------------

void CPURenderer::updateResult(void)
{
    FW_ASSERT(m_scene && m_image && m_batchRays);

/*	if(m_params.rayType == RayType_PathTracing)
	{
		if(m_secondaryIndex == 1) // Camera ray
			m_shader.addEmittedLight(m_scene, &m_primaryRays);

		if((m_secondaryIndex % 2) == 0) // Shadow ray
			m_shader.evaluateBSDF(m_scene, &m_primaryRays, m_batchRays);
		return;
	}*/

	S32 numRaysPerPrimary = (m_params.rayType == RayType_Primary) ? 1 : m_params.numSamples;
	S32 firstPrimary = m_batchStart / numRaysPerPrimary;

	for(S32 i = 0; i < m_batchRays->getSize()/numRaysPerPrimary; i++)
	{
		int primaryID = m_primaryRays.getIDForSlot(firstPrimary + i);
		const S32* batchSlots = (const S32*)m_batchRays->getIDToSlotBuffer().getMutablePtr() + ((m_params.rayType == RayType_Primary) ? primaryID : i * numRaysPerPrimary);

		U32& pixel = ((U32*)m_image->getBuffer().getMutablePtr())[primaryID];
		Vec4f bgColor = Vec4f(0.2f, 0.4f, 0.8f, 1.0f);

		//if(m_firstPass)
		//	pixel = 0;

		// Accumulate color from each ray in the batch.

		Vec4f color = Vec4f(0.0f);
		F32 stat = 0.0f;
		for (int j = 0; j < numRaysPerPrimary; j++)
		{
			const RayResult& result = m_batchRays->getResultForSlot(batchSlots[j]);
			int tri = result.id;

			/*switch(m_params.colorMode)
			{
			case ColorMode_Shaded:*/
				if (tri == -1)
					color += (m_params.rayType == RayType_Primary) ? bgColor : Vec4f(1.0f);
				else
					//color += (m_params.rayType == RayType_Shadow || m_params.rayType == RayType_AO) ? Vec4f(0.0f) : Vec4f::fromABGR(m_scene->getTriangle(tri).shadedColor);
					color += (m_params.rayType == RayType_AO) ? Vec4f(0.0f) : ((Vec4f*)(m_scene->getTriShadedColorBuffer().getPtr()))[tri];
				/*break;

			case ColorMode_PseudoNodes:
				stat += (float)(result.padB & 0xFFFF);
				break;

			case ColorMode_PseudoTris:
				stat += (float)(result.padB >> 16);
				//break;

			case ColorMode_Distance:
				stat = max(stat, result.padA);
				break;

			case ColorMode_DepthMap:
				stat = max(stat, result.t);
				break;

			default:
				FW_ASSERT(0);
			}*/
		}

		const RayResult& primaryResult = m_primaryRays.getResultForSlot(firstPrimary + i);

		/*if(m_params.colorMode == ColorMode_Shaded)
		{*/
			color *= 1.0f / (F32)numRaysPerPrimary;

			// Diffuse: modulate with primary hit color.

			int tri = primaryResult.id;
			//if (m_params.rayType == RayType_Shadow)					color *= (tri == -1) ? bgColor : Vec4f::fromABGR(m_scene->getTriangle(tri).shadedColor);
			if (m_params.rayType == RayType_AO && tri == -1)		color = bgColor;
			if (m_params.rayType == RayType_Diffuse)				color *= (tri == -1) ? bgColor : ((Vec4f*)(m_scene->getTriMaterialColorBuffer().getPtr()))[tri];//Vec4f::fromABGR(m_scene->getTriangle(tri).materialColor);

			// Get color from the previous iterations and add the color for this iteration
			Vec4f pixelColor;
			//float coeficient = (m_params.rayType == RayType_Shadow) ? 1.0f/m_params.lights.getSize() : 1.0f;
			float coeficient = 1.0f;
			pixelColor.fromABGR(pixel);
			color = pixelColor + color*coeficient;
			//if(m_finalPass)
			//	color += Vec4f(0.1f, 0.1f, 0.1f, 1.0f); // +Ambient light
			color.w = 1.0f;

			// Write result.
			pixel = color.toABGR();
		/*}
		else
		{
			stat = *(float*)(&pixel) + stat; // pixel holds number of OPs accumulated so far

			if(m_finalPass)
			{
				switch(m_params.colorMode)
				{
				case ColorMode_PseudoNodes:
					color = getPseudoColor(stat + (m_params.rayType != 0 ? primaryResult.padB & 0xFFFF : 0.0f), 0.0f, 150.0f);
					break;

				case ColorMode_PseudoTris:
					color = getPseudoColor(stat + (m_params.rayType != 0 ? primaryResult.padB >> 16 : 0.0f), 0.0f, 75.0f);
					break;

				case ColorMode_Distance:
					//color = getDistanceColor(max(stat, primaryResult.padA), 0.0f, m_cameraFar);
					color = getDistanceColor(primaryResult.padA, 0.0f, m_cameraFar);
					break;

				case ColorMode_DepthMap:
					//color = getDistanceColor(max(stat, primaryResult.t), 0.0f, m_cameraFar);
					color = getDistanceColor(primaryResult.t, 0.0f, m_cameraFar);
					break;
				}

				// Write result.
				pixel = color.toABGR();
			}
			else
			{
				// Write partial result.
				pixel = *(U32*)(&stat);
			}
		}*/
	}

	// Write the bargraph
	//if(m_finalPass && m_params.colorMode != ColorMode_Shaded)
	{
		Vec4f color;
		Vec2i pos;
		for(pos.y = 0; pos.y < m_image->getSize().y; pos.y++)
		{
			/*switch(m_params.colorMode)
			{
			case ColorMode_PseudoNodes:
			case ColorMode_PseudoTris:
				color = getPseudoColor((F32)pos.y, 0.0f, (F32)m_image->getSize().y);
				break;
			case ColorMode_Distance:
			case ColorMode_DepthMap:*/
				color = getDistanceColor((F32)pos.y, 0.0f, (F32)m_image->getSize().y);
			/*	break;
			}*/

			for(pos.x = m_image->getSize().x - 10; pos.x < m_image->getSize().x; pos.x++)
			{
				if(pos.x < m_image->getSize().x - 8)
					m_image->setVec4f(pos, Vec4f(0.0f, 0.0f, 0.0f, 1.0f));
				else
					m_image->setVec4f(pos, color);
			}
		}
	}
}

//------------------------------------------------------------------------

int CPURenderer::getTotalNumRays(void)
{
	return 0;
   /* // Casting primary rays => no degenerates.

    if (m_params.rayType == RayType_Primary)
        return m_primaryRays.getSize();

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Set input and output.

    CountHitsInput& in = *(CountHitsInput*)module->getGlobal("c_CountHitsInput").getMutablePtr();
    in.numRays = m_primaryRays.getSize();
    in.rayResults = m_primaryRays.getResultBuffer().getCudaPtr();
    in.raysPerThread = 32;
    module->getGlobal("g_CountHitsOutput").clear();

    // Count primary ray hits.

    module->getKernel("countHitsKernel").launch(
        (in.numRays - 1) / in.raysPerThread + 1,
        Vec2i(CountHits_BlockWidth, CountHits_BlockHeight));

    int numHits = *(S32*)module->getGlobal("g_CountHitsOutput").getPtr();

    // numSecondary = secondaryPerPrimary * primaryHits

    return numHits * m_params.numSamples;*/
}

//------------------------------------------------------------------------

/*S32 CPURenderer::incrementNumRays(void)
{
	int numHits = 0;
    // Casting primary rays => no degenerates.

    if (m_params.rayType == RayType_Primary || (m_params.rayType == RayType_PathTracing && m_secondaryIndex == 0))
	{
        m_rayCount += m_primaryRays.getSize();
		return m_primaryRays.getSize();
	}

	for(S32 i=0;i<m_primaryRays.getSize();i++)
    {
        const RayResult& result = m_primaryRays.getResultForSlot(i);

		if(result.hit())
			numHits++;
    }

    // numSecondary = secondaryPerPrimary * primaryHits

	int samples = (m_params.rayType == RayType_PathTracing) ? 1 : m_params.numSamples;
    m_rayCount += numHits * samples;
	return numHits;
}

//------------------------------------------------------------------------

Buffer& CPURenderer::getVisibleTriangles(S32 triangleCount, bool setValue, S32 initValue)
{
	Buffer &vis = m_visibility;
	//S64 bitSize = (triangleCount + sizeof(U32) - 1) / sizeof(U32); // Round up to CPU machine word
	S64 bitSize = triangleCount*sizeof(S32);

	// Initialize the buffer if needed
	vis.resize(bitSize);
	
	if(setValue)
		vis.clear(initValue);

	// Return the buffer
	return vis;
}*/

//------------------------------------------------------------------------

Vec4f CPURenderer::getPseudoColor(F32 value, F32 minVal, F32 maxVal)
{
	Vec4f val(1.0f);

	if (value < minVal) value = minVal;
	if (value > maxVal) value = maxVal;

	float ratio = (value - minVal)/(maxVal - minVal);

	const float MAX_COLOR_VALUE = 0.98f;

	switch ((int)((ratio)*3.9999f)) {
	case 0:
		FW_ASSERT(ratio <= 0.25f);
		val.x = 0.0f; // red
		val.y = ratio * 4.0f * MAX_COLOR_VALUE; // green
		val.z = MAX_COLOR_VALUE; // blue
		break;
	case 1:
		FW_ASSERT( (ratio >= 0.25f) && (ratio <= 0.5f) );
		val.x = 0.0f; // red
		val.y = MAX_COLOR_VALUE; // green
		val.z = MAX_COLOR_VALUE * (1.0f - 4.0f*(ratio-0.25f)); // blue
		break;
	case 2:
		FW_ASSERT( (ratio >= 0.5f) && (ratio <= 0.75f) );
		val.x = (ratio-0.5f) * 4.0f * MAX_COLOR_VALUE; // red
		val.y = MAX_COLOR_VALUE; // green
		val.z = 0.0f; // blue
		break;
	case 3:
		FW_ASSERT( (ratio >= 0.75f) && (ratio <= 1.f) );
		val.x = MAX_COLOR_VALUE; // red
		val.y = MAX_COLOR_VALUE * (1.0f - 4.0f*(ratio-0.75f)); // green
		val.z = 0.0f; // blue
		break;
	default:
		FW_ASSERT(0);
		break;
	}

	/*if (value < minVal) value = minVal;
	if (value > maxVal) value = maxVal;

	float valueI = 1.0f - value;
	const float MAX_COLOR_VALUE = 0.999f;

	switch ((int)(value*4.0f)) {
	case 0:
		val.x = MAX_COLOR_VALUE; // red
		val.y = valueI*MAX_COLOR_VALUE; // green
		val.z = 0.f; // blue
		break;
	case 1:
		val.x = (1.0f - valueI)*MAX_COLOR_VALUE; // red
		val.y = MAX_COLOR_VALUE; // green
		val.z = 0.f; // blue
		break;
	case 2:
		val.x = 0.f; // red
		val.y = MAX_COLOR_VALUE; // green
		val.z = valueI*MAX_COLOR_VALUE; // blue
		break;
	case 3:
		val.x = 0.f; // red
		val.y = (1.0f - valueI)*MAX_COLOR_VALUE; // green
		val.z = MAX_COLOR_VALUE; // blue
		break;
	default:
		val.x = valueI * MAX_COLOR_VALUE; // red
		val.y = 0.f; // green
		val.z = MAX_COLOR_VALUE; // blue
		break;
	}*/

	return val;
}

//------------------------------------------------------------------------

Vec4f CPURenderer::getDistanceColor(F32 value, F32 minVal, F32 maxVal)
{
	Vec4f val(1.0f);

	if (value < minVal) value = minVal;
	if (value > maxVal) value = maxVal;

	float ratio = (value - minVal)/(maxVal - minVal);
	// The first color - corresponding to minimum
	const float minColorRed = 0.0f;
	const float minColorGreen = 0.0f;
	const float minColorBlue = 0.0f;
	// The second color - corresponding to maximum
	const float maxColorRed = 1.0f;
	const float maxColorGreen = 1.0f;
	const float maxColorBlue = 1.0f;

	val.x = maxColorRed * ratio + (1.0f-ratio)*minColorRed; // red
	val.y = maxColorGreen * ratio + (1.0f-ratio)*minColorGreen; // green
	val.z = maxColorBlue * ratio + (1.0f-ratio)*minColorBlue; // blue

	return val;
}

//------------------------------------------------------------------------

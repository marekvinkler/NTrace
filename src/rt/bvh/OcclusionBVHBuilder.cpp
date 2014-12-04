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

#include "bvh/OcclusionBVHBuilder.hpp"
#include "base/Sort.hpp"
#include "io/File.hpp"

using namespace FW;

// Selects how spatial splits should be used. Intended for testing only, not as independent build methods.
//#define BUILD_NSAH
//#define BUILD_SSAH
#define BUILD_OSAH
//#define BUILD_ASBVH
//#define BUILD_VSAH

//------------------------------------------------------------------------

OcclusionBVHBuilder::OcclusionBVHBuilder(BVH& bvh, const BVH::BuildParams& params)
:   SplitBVHBuilder (bvh, params), m_MaxVisibleDepth(48)
{
	// Other options
	// - Extend Array class so that it can operate on not owned arrays
	// - New Buffer class that will prevent reallocation from dirtying other device memories
	// - Array or Buffer operating on individual bits
	// - Use of vector<bool> class with bit-to-bit copy

	if(m_params.visibility != NULL)
	{
		Timer timer(true);
		m_visibility.set((S32*)m_params.visibility->getPtr(), m_bvh.getScene()->getNumTriangles());
		F32 transferTime = timer.getElapsed();
		printf("GPU-CPU visibility transfer time: %.2f (ms)\n", transferTime*1.0e3f);
	}
	else
	{
		m_visibility.reset(m_bvh.getScene()->getNumTriangles());
		memset(m_visibility.getPtr(), 0, m_visibility.getNumBytes());
	}
	
}

//------------------------------------------------------------------------

OcclusionBVHBuilder::~OcclusionBVHBuilder(void)
{
}

//------------------------------------------------------------------------

BVHNode* OcclusionBVHBuilder::run(void)
{
    // Initialize reference stack and determine root bounds.

	const Vec3i* tris = (Vec3i*)m_bvh.getScene()->getTriVtxIndexBuffer().getPtr();
	const Vec3f* verts = (Vec3f*)m_bvh.getScene()->getVtxPosBuffer().getPtr();

	// Find visible triangles
    NodeSpecOcl rootSpec;
    rootSpec.numRef = m_bvh.getScene()->getNumTriangles();
    m_refStack.reset(rootSpec.numRef);

	F32 log2scene = log((F32)rootSpec.numRef)/log(2.0f);
	m_MaxVisibleDepth = (S32)(log2scene/2.0f);
	printf("OSAH max depth: %d\n", m_MaxVisibleDepth);

	if(!m_params.twoTrees)
	{
		// Set the references - optionally only visible or invisible references could be set
		int last = 0;
		for (int i = 0; i < rootSpec.numRef; i++)
		{
#ifdef BUILD_VSAH
			if(m_visibility[i] == 0) // By setting this line, processed triangles are chosen
				continue;
#endif

			m_refStack[last].triIdx = i;
			for (int j = 0; j < 3; j++)
				m_refStack[last].bounds.grow(verts[tris[i][j]]);
			// Inflate the basic boxes so that box intersections are correct
			//m_refStack[last].bounds.min() -= BVH_EPSILON;
			//m_refStack[last].bounds.max() += BVH_EPSILON;
			rootSpec.bounds.grow(m_refStack[last].bounds);

			//if(m_refStack[i].numVisible)
			//	rootSpec.boundsVisible.grow(m_refStack[last].bounds);
		
			m_visibility[last] = m_visibility[i];
			rootSpec.numVisible += m_visibility[i];
			last++;
		}
		rootSpec.numRef = last;
		printf("Visible triangles: %d tris\n", rootSpec.numVisible);

		// Inflate the basic boxes so that box intersections are correct
		/*F32 EPSILON = (rootSpec.bounds.max()-rootSpec.bounds.min()).max() * 2e-5f;
		rootSpec.bounds.min().set(Vec3f(FW_F32_MAX, FW_F32_MAX, FW_F32_MAX));
		rootSpec.bounds.max().set(Vec3f(-FW_F32_MAX, -FW_F32_MAX, -FW_F32_MAX));
		for (int i = 0; i < rootSpec.numRef; i++)
		{
			m_refStack[i].bounds.grow(m_refStack[i].bounds.min()-EPSILON);
			m_refStack[i].bounds.grow(m_refStack[i].bounds.max()+EPSILON);
			rootSpec.bounds.grow(m_refStack[i].bounds);
		}*/

		// Initialize rest of the members.

		m_minOverlap = rootSpec.bounds.area() * m_params.splitAlpha;
		m_rightBounds.reset(max(rootSpec.numRef, (int)NumSpatialBins) - 1);
		//m_rightVisibleBounds.reset(max(rootSpec.numRef, (int)NumSpatialBins) - 1);
		m_numDuplicates = 0;
		m_progressTimer.start();

		// Build recursively.

		BVHNode* root = buildNode(rootSpec, 0, rootSpec.numRef-1, 0, 0.0f, 1.0f);
		m_bvh.getTriIndices().compact();

		// Done.

		if (m_params.enablePrints)
			printf("OcclusionBVHBuilder: progress %.0f%%, duplicates %.0f%%\n",
				100.0f, (F32)m_numDuplicates / (F32)m_bvh.getScene()->getNumTriangles() * 100.0f);
		return root;
	}
	else
	{
		NodeSpecOcl visibleSpec, invisibleSpec;

		// Set the references - separate visible and invisible
		int lastFound = 0;
		int lastInvisible = -1;
		for (int i = 0; i < rootSpec.numRef; i++)
		{
			if(m_refStack[i].triIdx == -1)
				m_refStack[i].triIdx = i;

			if(lastInvisible == -1 && m_visibility[i]) // Triangle is visible
			{
				int j;
				for(j = max(lastFound+1, i+1); j < rootSpec.numRef; j++) // Find next invisible
				{
					if(!m_visibility[j])
					{
						m_refStack[j].triIdx = j;
						sortSwap(this, i, j);
						lastFound = j;
						break;
					}
				}
				
				if(j >= rootSpec.numRef - 1)
					lastInvisible = m_visibility[i] ? i-1 : i;
			}

			for (int j = 0; j < 3; j++)
				m_refStack[i].bounds.grow(verts[tris[m_refStack[i].triIdx][j]]);
			// Inflate the basic boxes so that box intersections are correct
			//m_refStack[i].bounds.grow(m_refStack[i].bounds.min()-BVH_EPSILON);
			//m_refStack[i].bounds.grow(m_refStack[i].bounds.max()+BVH_EPSILON);

			if(m_visibility[i])
				visibleSpec.bounds.grow(m_refStack[i].bounds);
			else
				invisibleSpec.bounds.grow(m_refStack[i].bounds);
		
			visibleSpec.numVisible += m_visibility[i];
		}
		invisibleSpec.numRef = lastInvisible+1; // We want number of visible, not last index
		visibleSpec.numRef = rootSpec.numRef - invisibleSpec.numRef;

		// Build tree for visible nodes
		// Initialize rest of the members.

		m_minOverlap = visibleSpec.bounds.area() * m_params.splitAlpha;
		m_rightBounds.reset(max(visibleSpec.numRef, (int)NumSpatialBins) - 1);
		//m_rightVisibleBounds.reset(maxvisibleSpec.numRef, (int)NumSpatialBins) - 1);
		m_numDuplicates = 0;
		m_progressTimer.start();

		// Build recursively.

		BVHNode* visible;
		visible = SplitBVHBuilder::buildNode(visibleSpec, rootSpec.numRef - 1, 0.0f, 1.0f);

		// Done.

		if (m_params.enablePrints)
			printf("OcclusionBVHBuilder: progress %.0f%%, duplicates %.0f%%\n",
			100.0f, (F32)m_numDuplicates / (F32)visibleSpec.numRef * 100.0f);


		m_refStack.resize(invisibleSpec.numRef); // Forget the visible references, we have already build the tree for them


		// Build tree for invisible nodes
		// Initialize rest of the members.

		m_minOverlap = invisibleSpec.bounds.area() * m_params.splitAlpha;
		m_rightBounds.reset(max(invisibleSpec.numRef, (int)NumSpatialBins) - 1);
		//m_rightVisibleBounds.reset(max(invisibleSpec.numRef, (int)NumSpatialBins) - 1);
		m_numDuplicates = 0;
		m_progressTimer.start();

		// Build recursively.

		BVHNode* invisible;
		invisible = SplitBVHBuilder::buildNode(invisibleSpec, invisibleSpec.numRef-1, 0.0f, 1.0f);

		// Done.

		if (m_params.enablePrints)
			printf("OcclusionBVHBuilder: progress %.0f%%, duplicates %.0f%%\n",
				100.0f, (F32)m_numDuplicates / (F32)invisibleSpec.numRef * 100.0f);

		m_bvh.getTriIndices().compact();
		return new InnerNode(visibleSpec.bounds + invisibleSpec.bounds, visible, invisible, 0, SplitInfo::SAH, false);
	}
}

//------------------------------------------------------------------------

BVHNode* OcclusionBVHBuilder::buildNode(const NodeSpecOcl& spec, int start, int end, int level, F32 progressStart, F32 progressEnd)
{
	// Display progress.

    if (m_params.enablePrints && m_progressTimer.getElapsed() >= 1.0f)
    {
        printf("OcclusionBVHBuilder: progress %.0f%%, duplicates %.0f%%\r",
            progressStart * 100.0f, (F32)m_numDuplicates / (F32)m_bvh.getScene()->getNumTriangles() * 100.0f);
        m_progressTimer.start();
    }

    // Small enough or too deep => create leaf.
	
	if (level != 0 && spec.numRef <= m_platform.getMinLeafSize() || level >= MaxDepth) // Make sure we do not make the root a leaf -> GPU traversal will fail
		return createLeaf(spec);

    // Find split candidates.

    F32 area = spec.bounds.area();
    F32 leafSAH = area * m_platform.getTriangleCost(spec.numRef);
    F32 nodeSAH = area * m_platform.getNodeCost(2);
	
	bool osahChosen = false, osahTested = false;
	SplitInfo::SplitType splitType = SplitInfo::SAH;
	S32 axis = 0;

	// Occlusion information split

	ObjectSplitOcl object;
    SpatialSplitOcl spatial;

	// Choose which split types should be computed based on the desired construction method

#ifndef BUILD_ASBVH
	if(spec.numVisible != 0 && spec.numVisible != spec.numRef && level < m_MaxVisibleDepth)
	//if(spec.numVisible != 0 && spec.numVisible != spec.numRef && level < MaxVisibleDepth)
	{
		spatial = findSpatialOccludeSplit(spec, start, end, nodeSAH);
		osahTested = true;
		osahChosen = spatial.osahChosen;
	}
#endif
	
	if(!osahChosen)
	{
		object = findObjectSplit(spec, start, end, nodeSAH);

#if defined(BUILD_SSAH)
		if(level < MaxSpatialDepth)
#elif defined(BUILD_ASBVH)
		if(spec.numVisible != 0 && level < MaxSpatialDepth)
#elif defined(BUILD_OSAH)
		if(spec.numVisible != 0 && level < MaxSpatialDepth && level >= m_MaxVisibleDepth)
#endif

#if defined(BUILD_SSAH) || defined(BUILD_ASBVH) || defined(BUILD_OSAH)
		{
			spatial = SpatialSplitOcl();

			AABB overlap = object.leftBounds;
			overlap.intersect(object.rightBounds);
			if (overlap.area() >= m_minOverlap)
				spatial = findSpatialSplit(spec, start, end, nodeSAH);
		}
#endif
	}

    // Leaf SAH is the lowest => create leaf.

    F32 minSAH = min(leafSAH, object.sah, spatial.sah);
    if (level != 0 && minSAH == leafSAH && spec.numRef <= m_platform.getMaxLeafSize()) // Make sure we do not make the root a leaf -> GPU traversal will fail
		return createLeaf(spec);

    // Perform split.

    NodeSpecOcl left, right;
	if (osahChosen || minSAH == spatial.sah)
		performSpatialOccludeSplit(left, right, start, end, spatial);
	if (!left.numRef || !right.numRef)
	{
		if(osahChosen)
			object = findObjectSplit(spec, start, end, nodeSAH); // Fixes a bug when performSpatialOccludeSplit puts all references to one child and execution
																// falls back to this location but object split was not computed.
		performObjectSplit(left, right, spec, start, end, object);
		axis = object.sortDim;
	}
	else
	{
		splitType = osahChosen ? SplitInfo::OSAH : SplitInfo::SBVH;
		axis = spatial.dim;
	}

    // Create inner node.

    m_numDuplicates += left.numRef + right.numRef - spec.numRef;
    F32 progressMid = lerp(progressStart, progressEnd, (F32)right.numRef / (F32)(left.numRef + right.numRef));

	BVHNode* rightNode = buildNode(right, start+left.numRef, end, level + 1, progressStart, progressMid);
	BVHNode* leftNode = buildNode(left, start, start+left.numRef-1, level + 1, progressMid, progressEnd);

	// Swap the boxes so that the left child has more visible triangles
	if(right.numVisible > left.numVisible) // Will work fine with overlapping visible triangles? If not return to previous approach
		return new InnerNode(spec.bounds, rightNode, leftNode, axis, splitType, osahTested);
	else
		return new InnerNode(spec.bounds, leftNode, rightNode, axis, splitType, osahTested);
}

//------------------------------------------------------------------------

OcclusionBVHBuilder::ObjectSplitOcl OcclusionBVHBuilder::findObjectSplit(const NodeSpecOcl& spec, int start, int end, F32 nodeSAH)
{
    ObjectSplitOcl split;
	Reference* refPtr = m_refStack.getPtr(start);
	S32* visPtr = m_visibility.getPtr(start);
	
	S32 visibleLeft, visibleRight;

#ifdef ENABLE_GRAPHS
	BufferedOutputStream *buffer = NULL;
	File *file = NULL;
	char dimStr[] = {'x', 'y', 'z'};
#endif

    // Sort along each dimension.

    for (m_sortDim = 0; m_sortDim < 3; m_sortDim++)
    {
		sort(this, start, end+1, sortCompare, sortSwap);

#ifdef ENABLE_GRAPHS
		if(level == 0)
		{
			// Open file buffer
			CreateDirectory(m_params.logDirectory.getPtr(), NULL);
			String name = sprintf("%s\\cost_sah_%s%d_%c.log", m_params.logDirectory.getPtr(), m_params.buildName.getPtr(), m_params.cameraIdx, dimStr[m_sortDim]);
			file = new File(name, File::Create);
			buffer = new BufferedOutputStream(*file, 1024, true, true);
		}
#endif

		// Starting visibilities
		visibleLeft = 0;
		visibleRight = spec.numVisible;

        // Sweep right to left and determine bounds.

		AABB rightBounds;
		//AABB rightVisibleBounds;
        for (int i = end-start; i > 0; i--)
        {
			//if(refPtr[i].numVisible)
			//	rightVisibleBounds.grow(refPtr[i].bounds);
			//m_rightVisibleBounds[i - 1] = rightVisibleBounds;

            rightBounds.grow(refPtr[i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        AABB leftBounds;
		//AABB leftVisibleBounds;
        for (int i = 1; i < end-start+1; i++) // All own triangles have been tested
        {
			//if(refPtr[i-1].numVisible)
			//	leftVisibleBounds.grow(refPtr[i-1].bounds);

			visibleLeft += visPtr[i - 1];
			visibleRight -= visPtr[i - 1];

			leftBounds.grow(refPtr[i - 1].bounds);

			// Calculate SAH

			F32 sah = nodeSAH + leftBounds.area() * m_platform.getTriangleCost(i) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost((end-start+1) - i);

			if(sah < split.sah)
			{
				split.sah = sah;
				split.sortDim = m_sortDim;
				split.numLeft = i;
				split.leftBounds = leftBounds;
				split.leftVisible = visibleLeft;
				//split.leftVisibleBounds = leftVisibleBounds;
				split.rightBounds = m_rightBounds[i - 1];
				split.rightVisible = visibleRight;
				//split.rightVisibleBounds = m_rightVisibleBounds[j - 1];
			}

#ifdef ENABLE_GRAPHS
			// Print split data
			if(level == 0)
			{
				buffer->writef("%d\t%f\n", i, sah);
			}
#endif
        }

#ifdef ENABLE_GRAPHS
		// Free file buffer
		if(level == 0)
		{
			buffer->flush();
			delete file;
			delete buffer;
		}
#endif
	}
	
	return split;
}

//------------------------------------------------------------------------

OcclusionBVHBuilder::ObjectSplitOcl OcclusionBVHBuilder::findObjectOccludeSplit(const NodeSpecOcl& spec, int start, int end, F32 nodeSAH)
{
    ObjectSplitOcl split, osplit;
	Reference* refPtr = m_refStack.getPtr(start);
	S32* visPtr = m_visibility.getPtr(start);
	
	S32 visibleLeft, visibleRight;

#ifdef ENABLE_GRAPHS
	BufferedOutputStream *buffer = NULL;
	File *file = NULL;
	char dimStr[] = {'x', 'y', 'z'};
#endif

    // Sort along each dimension.

    for (m_sortDim = 0; m_sortDim < 3; m_sortDim++)
    {
		sort(this, start, end+1, sortCompare, sortSwap);

#ifdef ENABLE_GRAPHS
		if(level == 0)
		{
			// Open file buffer
			CreateDirectory(m_params.logDirectory.getPtr(), NULL);
			String name = sprintf("%s\\cost_%s%d_%c.log", m_params.logDirectory.getPtr(), m_params.buildName.getPtr(), m_params.cameraIdx, dimStr[m_sortDim]);
			file = new File(name, File::Create);
			buffer = new BufferedOutputStream(*file, 1024, true, true);
		}
#endif

		// Starting visibilities
		visibleLeft = 0;
		visibleRight = spec.numVisible;

        // Sweep right to left and determine bounds.

		AABB rightBounds;
		//AABB rightVisibleBounds;
        for (int i = end-start; i > 0; i--)
        {
			//if(refPtr[i].numVisible)
			//	rightVisibleBounds.grow(refPtr[i].bounds);
			//m_rightVisibleBounds[i - 1] = rightVisibleBounds;

            rightBounds.grow(refPtr[i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        AABB leftBounds;
		//AABB leftVisibleBounds;
        for (int i = 1; i < end-start+1; i++) // All own triangles have been tested
        {
			//if(refPtr[i-1].numVisible)
			//	leftVisibleBounds.grow(refPtr[i-1].bounds);

			// Calculate occlusion
			visibleLeft += visPtr[i - 1];
			visibleRight -= visPtr[i - 1];

			leftBounds.grow(refPtr[i - 1].bounds);

			// Calculate SAH

			F32 sah = nodeSAH + leftBounds.area() * m_platform.getTriangleCost(i) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost((end-start+1) - i);

			// Calculate OSAH
			F32 osah = FW_F32_MAX;

			if(spec.numVisible != 0 && spec.numVisible != spec.numRef)
			{
				F32 weight = m_params.osahWeight * (1.0f - (float)spec.numVisible/(float)(end-start+1));
				F32 probL = weight * (float)visibleLeft/(float)spec.numVisible + (1.0f - weight) * leftBounds.area()/spec.bounds.area();
				F32 probR = weight * (float)visibleRight/(float)spec.numVisible + (1.0f - weight) * m_rightBounds[i - 1].area()/spec.bounds.area();
				osah = nodeSAH + probL * m_platform.getTriangleCost(i) + probR * m_platform.getTriangleCost((end-start+1) - i);
			}

			// Update best splits
			if(osah < osplit.sah)
			{
				osplit.sah = osah;
				osplit.sortDim = m_sortDim;
				osplit.numLeft = i;
				osplit.leftBounds = leftBounds;
				osplit.leftVisible = visibleLeft;
				//osplit.leftVisibleBounds = leftVisibleBounds;
				osplit.rightBounds = m_rightBounds[i - 1];
				osplit.rightVisible = visibleRight;
				//osplit.rightVisibleBounds = m_rightVisibleBounds[i - 1];
				osplit.osahTested = true;
			}

			if(sah < split.sah)
			{
				split.sah = sah;
				split.sortDim = m_sortDim;
				split.numLeft = i;
				split.leftBounds = leftBounds;
				split.leftVisible = visibleLeft;
				//split.leftVisibleBounds = leftVisibleBounds;
				split.rightBounds = m_rightBounds[i - 1];
				split.rightVisible = visibleRight;
				//split.rightVisibleBounds = m_rightVisibleBounds[i - 1];
			}

#ifdef ENABLE_GRAPHS
			// Print split data
			if(level == 0)
			{
				if(osah < FW_F32_MAX)
					buffer->writef("%d\t%f\t%f\n", i, sah, osah);
				else
					buffer->writef("%d\t%f\t%f\n", i, sah, (float)2.0f*spec.numRef); // Should be always larger than any computed cost but small enough not to distort the graph
				//buffer->writef("%d\t%f\t?\n", i, sah);
			}
#endif
		}

#ifdef ENABLE_GRAPHS
		// Free file buffer
		if(level == 0)
		{
			buffer->flush();
			delete file;
			delete buffer;
		}
#endif
	}

	// Chose better of the two heuristics	
	S32 hidTris = osplit.leftVisible < osplit.rightVisible ? osplit.numLeft : spec.numRef - osplit.numLeft;
	S32 largerChild = FW::max(split.numLeft, spec.numRef - split.numLeft);

	if(osplit.sah < FW_F32_MAX && hidTris > largerChild)
	{
 		osplit.osahChosen = true;
		return osplit;
	}
	else
	{
		split.osahTested = osplit.osahTested;
		return split;
	}
}
//------------------------------------------------------------------------

void OcclusionBVHBuilder::performObjectSplit(NodeSpecOcl& left, NodeSpecOcl& right, const NodeSpecOcl& spec, int start, int end, const ObjectSplitOcl& split)
{
    m_sortDim = split.sortDim;
	sort(this, start, end+1, sortCompare, sortSwap);

    left.numRef = split.numLeft;
    left.bounds = split.leftBounds;
	left.numVisible = split.leftVisible;
	//left.boundsVisible = split.leftVisibleBounds;
    right.numRef = spec.numRef - split.numLeft;
    right.bounds = split.rightBounds;
	right.numVisible = split.rightVisible;
	//right.boundsVisible = split.rightVisibleBounds;
}

//------------------------------------------------------------------------

OcclusionBVHBuilder::SpatialSplitOcl OcclusionBVHBuilder::findSpatialSplit(const NodeSpecOcl& spec, int start, int end, F32 nodeSAH)
{
    // Initialize bins.

    Vec3f origin = spec.bounds.min();
    Vec3f binSize = (spec.bounds.max() - origin) * (1.0f / (F32)NumSpatialBins);
    Vec3f invBinSize = 1.0f / binSize;

	//size_t visibleLeft, visibleRight;

    for (int dim = 0; dim < 3; dim++)
    {
        for (int i = 0; i < NumSpatialBins; i++)
        {
            SpatialBinOcl& bin = m_bins[dim][i];
            bin.bounds = AABB();
            bin.enter = 0;
            bin.exit = 0;
			//bin.enterVisible = 0;
			//bin.exitVisible = 0;
        }
    }

    // Chop references into bins.

    for (int refIdx = start; refIdx < end+1; refIdx++)
    {
        const Reference& ref = m_refStack[refIdx];
        Vec3i firstBin = clamp(Vec3i((ref.bounds.min() - origin) * invBinSize), 0, NumSpatialBins - 1);
        Vec3i lastBin = clamp(Vec3i((ref.bounds.max() - origin) * invBinSize), firstBin, NumSpatialBins - 1);

        for (int dim = 0; dim < 3; dim++)
        {
            Reference currRef = ref;
            for (int i = firstBin[dim]; i < lastBin[dim]; i++)
            {
                Reference leftRef, rightRef;
                splitReference(leftRef, rightRef, currRef, dim, origin[dim] + binSize[dim] * (F32)(i + 1));
				if(leftRef.bounds.valid()) // May be invalid because the boxes are inflated by BVH_EPSILON
					m_bins[dim][i].bounds.grow(leftRef.bounds);
                currRef = rightRef;
            }
			if(currRef.bounds.valid()) // May be invalid because the boxes are inflated by BVH_EPSILON
				m_bins[dim][lastBin[dim]].bounds.grow(currRef.bounds);
            m_bins[dim][firstBin[dim]].enter++;
            m_bins[dim][lastBin[dim]].exit++;
			//if(ref.numVisible)
			//{
			//	m_bins[dim][firstBin[dim]].enterVisible++;
			//	m_bins[dim][lastBin[dim]].exitVisible++;
			//}
        }
    }

    // Select best split plane.

    SpatialSplitOcl split, osplit;
    for (int dim = 0; dim < 3; dim++)
    {
        // Sweep right to left and determine bounds.

        AABB rightBounds;
        for (int i = NumSpatialBins - 1; i > 0; i--)
        {
			if(m_bins[dim][i].bounds.valid())
				rightBounds.grow(m_bins[dim][i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        AABB leftBounds;
        int leftNum = 0;
        int rightNum = spec.numRef;

		// Starting visibilities
		//visibleLeft = 0;
		//visibleRight = spec.numVisible;

        for (int i = 1; i < NumSpatialBins; i++)
        {
			if(!m_bins[dim][i-1].bounds.valid())
				continue;

			leftBounds.grow(m_bins[dim][i - 1].bounds);
            leftNum += m_bins[dim][i - 1].enter;
            rightNum -= m_bins[dim][i - 1].exit;
			//visibleLeft += m_bins[dim][i - 1].enterVisible;
            //visibleRight -= m_bins[dim][i - 1].exitVisible;

			F32 sah = nodeSAH + leftBounds.area() * m_platform.getTriangleCost(leftNum) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost(rightNum);
			if (sah < split.sah)
            {
                split.sah = sah;
                split.dim = dim;
                split.pos = origin[dim] + binSize[dim] * (F32)i;
				split.leftNum = leftNum;
				split.rightNum = rightNum;
				//split.leftVisible = visibleLeft;
				//split.rightVisible = visibleRight;
            }
        }
    }

	return split;
}

//------------------------------------------------------------------------

OcclusionBVHBuilder::SpatialSplitOcl OcclusionBVHBuilder::findSpatialOccludeSplit(const NodeSpecOcl& spec, int start, int end, F32 nodeSAH)
{
    // Initialize bins.

    Vec3f origin = spec.bounds.min();
    Vec3f binSize = (spec.bounds.max() - origin) * (1.0f / (F32)NumSpatialBins);
    Vec3f invBinSize = 1.0f / binSize;

	S32 visibleLeft, visibleRight;

#ifdef ENABLE_GRAPHS
	BufferedOutputStream *buffer = NULL;
	File *file = NULL;
	char dimStr[] = {'x', 'y', 'z'};
#endif

    for (int dim = 0; dim < 3; dim++)
    {
        for (int i = 0; i < NumSpatialBins; i++)
        {
            SpatialBinOcl& bin = m_bins[dim][i];
            bin.bounds = AABB();
            bin.enter = 0;
            bin.exit = 0;
			bin.enterVisible = 0;
			bin.exitVisible = 0;
        }
    }

    // Chop references into bins.

    for (int refIdx = start; refIdx < end+1; refIdx++)
    {
        const Reference& ref = m_refStack[refIdx];
        Vec3i firstBin = clamp(Vec3i((ref.bounds.min() - origin) * invBinSize), 0, NumSpatialBins - 1);
        Vec3i lastBin = clamp(Vec3i((ref.bounds.max() - origin) * invBinSize), firstBin, NumSpatialBins - 1);

        for (int dim = 0; dim < 3; dim++)
        {
            Reference currRef = ref;
            for (int i = firstBin[dim]; i < lastBin[dim]; i++)
            {
                Reference leftRef, rightRef;
                splitReference(leftRef, rightRef, currRef, dim, origin[dim] + binSize[dim] * (F32)(i + 1));
				if(leftRef.bounds.valid()) // May be invalid because the boxes are inflated by BVH_EPSILON
					m_bins[dim][i].bounds.grow(leftRef.bounds);
                currRef = rightRef;
            }
			if(currRef.bounds.valid()) // May be invalid because the boxes are inflated by BVH_EPSILON
				m_bins[dim][lastBin[dim]].bounds.grow(currRef.bounds);
            m_bins[dim][firstBin[dim]].enter++;
            m_bins[dim][lastBin[dim]].exit++;
			if(m_visibility[refIdx])
			{
				m_bins[dim][firstBin[dim]].enterVisible++;
				m_bins[dim][lastBin[dim]].exitVisible++;
			}
        }
    }

    // Select best split plane.

    SpatialSplitOcl split, osplit;
    for (int dim = 0; dim < 3; dim++)
    {
        // Sweep right to left and determine bounds.

        AABB rightBounds;
        for (int i = NumSpatialBins - 1; i > 0; i--)
        {
			if(m_bins[dim][i].bounds.valid())
				rightBounds.grow(m_bins[dim][i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        AABB leftBounds;
        int leftNum = 0;
        int rightNum = spec.numRef;

		// Starting visibilities
		visibleLeft = 0;
		visibleRight = spec.numVisible;

#ifdef ENABLE_GRAPHS
		if(level == 0)
		{
			// Open file buffer
			CreateDirectory(m_params.logDirectory.getPtr(), NULL);
			String name = sprintf("%s\\cost_osah,ssah_%s%d_%c.log", m_params.logDirectory.getPtr(), m_params.buildName.getPtr(), m_params.cameraIdx, dimStr[dim]);
			file = new File(name, File::Create);
			buffer = new BufferedOutputStream(*file, 1024, true, true);
		}
#endif

        for (int i = 1; i < NumSpatialBins; i++)
        {
			if(!m_bins[dim][i-1].bounds.valid())
				continue;

			leftBounds.grow(m_bins[dim][i - 1].bounds);
            leftNum += m_bins[dim][i - 1].enter;
            rightNum -= m_bins[dim][i - 1].exit;
			visibleLeft += m_bins[dim][i - 1].enterVisible;
            visibleRight -= m_bins[dim][i - 1].exitVisible;

			F32 osah = FW_F32_MAX;
			//if(visibleLeft == 0 || visibleRight == 0)
			{
				//F32 weight = 0.9f * (1.0f - (float)spec.numVisible/(float)(end-start+1));
				F32 weight = m_params.osahWeight;
				F32 probL = weight * ((float)(visibleLeft)/(float)spec.numVisible) + (1.0f - weight) * leftBounds.area()/spec.bounds.area();
				F32 probR = weight * ((float)(visibleRight)/(float)spec.numVisible) + (1.0f - weight) * m_rightBounds[i - 1].area()/spec.bounds.area();
				osah = nodeSAH + probL * m_platform.getTriangleCost(leftNum) + probR * m_platform.getTriangleCost(rightNum);
			}
			if (osah < osplit.sah)
            {
                osplit.sah = osah;
                osplit.dim = dim;
                osplit.pos = origin[dim] + binSize[dim] * (F32)i;
				osplit.leftNum = leftNum;
				osplit.rightNum = rightNum;
				osplit.leftVisible = visibleLeft;
				osplit.rightVisible = visibleRight;
            }

			F32 sah = nodeSAH + leftBounds.area() * m_platform.getTriangleCost(leftNum) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost(rightNum);
			if (sah < split.sah)
            {
                split.sah = sah;
                split.dim = dim;
                split.pos = origin[dim] + binSize[dim] * (F32)i;
				split.leftNum = leftNum;
				split.rightNum = rightNum;
				split.leftVisible = visibleLeft;
				split.rightVisible = visibleRight;
            }

#ifdef ENABLE_GRAPHS
			// Print split data
			if(level == 0)
			{
				if(osah < FW_F32_MAX)
					buffer->writef("%d\t%f\t%f\n", i, sah, osah);
				else
					buffer->writef("%d\t%f\t%f\n", i, sah, (float)2.0f*spec.numRef); // Should be always larger than any computed cost but small enough not to distort the graph
				//buffer->writef("%d\t%f\t?\n", i, sah);
			}
#endif
		}

#ifdef ENABLE_GRAPHS
		// Free file buffer
		if(level == 0)
		{
			buffer->flush();
			delete file;
			delete buffer;
		}
#endif
    }

	S32 hidTris = osplit.leftVisible < osplit.rightVisible ? osplit.leftNum : osplit.rightNum;
	S32 smallerChild = FW::max(split.leftNum, split.rightNum);

	if(osplit.sah < FW_F32_MAX && hidTris > smallerChild)
	{
		osplit.osahChosen = true;
		return osplit;
	}
	else
	{
		split.osahChosen = false;
#ifndef BUILD_OSAH
		split.sah = FW_F32_MAX;
#endif
		return split;
	}
}

//------------------------------------------------------------------------

/*OcclusionBVHBuilder::SpatialSplitOcl OcclusionBVHBuilder::findVisibleSplit(const NodeSpecOcl& spec, int start, int end, F32 nodeSAH, int level)
{
    // Initialize bins.

	const int numBins = 3;
	F32 splitPos[3][numBins];

	size_t visibleLeft, visibleRight;

#ifdef ENABLE_GRAPHS
	BufferedOutputStream *buffer = NULL;
	File *file = NULL;
	char dimStr[] = {'x', 'y', 'z'};
#endif

    for (int dim = 0; dim < 3; dim++)
    {
        for (int i = 0; i < numBins; i++)
        {
            SpatialBinOcl& bin = m_bins[dim][i];
            bin.bounds = AABB();
            bin.enter = 0;
            bin.exit = 0;
			bin.enterVisible = 0;
			bin.exitVisible = 0;
        }

		splitPos[dim][1] = spec.boundsVisible.min()[dim];
		splitPos[dim][2] = spec.boundsVisible.max()[dim];
    }

    // Chop references into bins.

    for (int refIdx = start; refIdx < end+1; refIdx++)
    {
        const Reference& ref = m_refStack[refIdx];
		Vec3i firstBin, lastBin;
		for (int dim = 0; dim < 3; dim++)
		{
			firstBin[dim] = ref.bounds.min()[dim] < splitPos[dim][1] ? 0 : ref.bounds.min()[dim] <= splitPos[dim][2] ? 1 : 2;
			lastBin[dim] = ref.bounds.max()[dim] < splitPos[dim][1] ? 0 : ref.bounds.max()[dim] <= splitPos[dim][2] ? 1 : 2;
		}

        for (int dim = 0; dim < 3; dim++)
        {
            Reference currRef = ref;
            for (int i = firstBin[dim]; i < lastBin[dim]; i++)
            {
                Reference leftRef, rightRef;
                splitReference(leftRef, rightRef, currRef, dim, splitPos[dim][i + 1]);
				if(leftRef.bounds.valid()) // May be invalid because the boxes are inflated by BVH_EPSILON
					m_bins[dim][i].bounds.grow(leftRef.bounds);
                currRef = rightRef;
            }
			if(currRef.bounds.valid()) // May be invalid because the boxes are inflated by BVH_EPSILON
				m_bins[dim][lastBin[dim]].bounds.grow(currRef.bounds);
            m_bins[dim][firstBin[dim]].enter++;
            m_bins[dim][lastBin[dim]].exit++;
			if(ref.numVisible)
			{
				m_bins[dim][firstBin[dim]].enterVisible++;
				m_bins[dim][lastBin[dim]].exitVisible++;
			}
        }
    }

    // Select best split plane.

    SpatialSplitOcl split;
    for (int dim = 0; dim < 3; dim++)
    {
        // Sweep right to left and determine bounds.

        AABB rightBounds;
        for (int i = numBins - 1; i > 0; i--)
        {
			if(m_bins[dim][i].bounds.valid())
				rightBounds.grow(m_bins[dim][i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        AABB leftBounds;
        int leftNum = 0;
        int rightNum = spec.numRef;

		// Starting visibilities
		visibleLeft = 0;
		visibleRight = spec.numVisible;

#ifdef ENABLE_GRAPHS
		if(level == 0)
		{
			// Open file buffer
			CreateDirectory(m_params.logDirectory.getPtr(), NULL);
			String name = sprintf("%s\\cost_%s%d_%c.log", m_params.logDirectory.getPtr(), m_params.buildName.getPtr(), m_params.cameraIdx, dimStr[dim]);
			file = new File(name, File::Create);
			buffer = new BufferedOutputStream(*file, 1024, true, true);
		}
#endif

        for (int i = 1; i < numBins; i++)
        {
			if(!m_bins[dim][i-1].bounds.valid())
				continue;

			leftBounds.grow(m_bins[dim][i - 1].bounds);
            leftNum += m_bins[dim][i - 1].enter;
            rightNum -= m_bins[dim][i - 1].exit;
			visibleLeft += m_bins[dim][i - 1].enterVisible;
            visibleRight -= m_bins[dim][i - 1].exitVisible;

			F32 osah = FW_F32_MAX;
			if(visibleLeft == 0 || visibleRight == 0)
			{
				F32 weight = m_params.osahWeight * (1.0f - (float)spec.numVisible/(float)(end-start+1));
				F32 probL = weight * ((float)(visibleLeft)/(float)spec.numVisible) + (1.0f - weight) * leftBounds.area()/spec.bounds.area();
				F32 probR = weight * ((float)(visibleRight)/(float)spec.numVisible) + (1.0f - weight) * m_rightBounds[i - 1].area()/spec.bounds.area();
				osah = nodeSAH + probL * m_platform.getTriangleCost(leftNum) + probR * m_platform.getTriangleCost(rightNum);
			}

			if (osah < split.sah)
            {
                split.sah = osah;
                split.dim = dim;
                split.pos = splitPos[dim][i];
				split.leftArea = leftBounds.area();
				split.rightArea = m_rightBounds[i - 1].area();
				split.leftNum = leftNum;
				split.rightNum = rightNum;
				split.leftVisible = visibleLeft;
				split.rightVisible = visibleRight;
            }

#ifdef ENABLE_GRAPHS
			// Print split data
			if(level == 0)
			{
				if(osah < FW_F32_MAX)
					buffer->writef("%d\t?\t%f\n", i, osah);
				else
					buffer->writef("%d\t?\t%f\n", i, (float)2.0f*spec.numRef); // Should be always larger than any computed cost but small enough not to distort the graph
				//buffer->writef("%d\t%f\t?\n", i, sah);
			}
#endif
        }
		
#ifdef ENABLE_GRAPHS
		// Free file buffer
		if(level == 0)
		{
			buffer->flush();
			delete file;
			delete buffer;
		}
endif
    }

	//S32 invisChild = split.leftVisible < split.rightVisible ? split.leftNum : split.rightNum;
	//S32 visChild = split.leftVisible < split.rightVisible ? split.rightNum : split.leftNum;

	F32 visibleSAH, invisibleSAH;
	if(split.leftVisible < split.rightVisible)
	{
		visibleSAH = split.rightArea * m_platform.getTriangleCost(split.rightNum);
		invisibleSAH = nodeSAH + split.leftArea * m_platform.getTriangleCost(split.leftNum);
	}
	else
	{
		visibleSAH = nodeSAH + split.leftArea * m_platform.getTriangleCost(split.leftNum);
		invisibleSAH = split.rightArea * m_platform.getTriangleCost(split.rightNum);
	}

	//if(split.sah < FW_F32_MAX && invisChild > visChild)
	if(visibleSAH*2.0f < invisibleSAH)
	{
		return split;
	}
	else
	{
		split.dim = -1;
		return split;
	}
}*/

//------------------------------------------------------------------------

void OcclusionBVHBuilder::performSpatialOccludeSplit(NodeSpecOcl& left, NodeSpecOcl& right, int& start, int& end, const SpatialSplitOcl& split)
{
    // Categorize references and compute bounds.
    //
    // Left-hand side:      [leftStart, leftEnd[
    // Uncategorized/split: [leftEnd, rightStart[
    // Right-hand side:     [rightStart, refs.getSize()[

    Array<Reference>& refs = m_refStack;
    int leftStart = start;
    int leftEnd = leftStart;
    int rightStart = end+1;
    left.bounds = right.bounds = AABB();
	//left.boundsVisible = right.boundsVisible = AABB();

    for (int i = leftEnd; i < rightStart; i++)
    {
        // Entirely on the left-hand side?

        if (refs[i].bounds.max()[split.dim] <= split.pos)
        {
            left.bounds.grow(refs[i].bounds);
			//if(refs[i].numVisible)
			//	left.boundsVisible.grow(refs[i].bounds);
			left.numVisible += m_visibility[i];
            swap(refs[i], refs[leftEnd]);
			swap(m_visibility[i], m_visibility[leftEnd]);
			leftEnd++;
        }

        // Entirely on the right-hand side?

        else if (refs[i].bounds.min()[split.dim] >= split.pos)
        {
            right.bounds.grow(refs[i].bounds);
			//if(refs[i].numVisible)
			//	right.boundsVisible.grow(refs[i].bounds);
			right.numVisible += m_visibility[i];
			rightStart--;
            swap(refs[i], refs[rightStart]);
			swap(m_visibility[i], m_visibility[rightStart]);
			i--;
        }
    }

    // Duplicate or unsplit references intersecting both sides.

    while (leftEnd < rightStart)
    {
        // Split reference.

        Reference lref, rref;
        splitReference(lref, rref, refs[leftEnd], split.dim, split.pos);

        // Compute SAH for duplicate/unsplit candidates.

        AABB lub = left.bounds;  // Unsplit to left:     new left-hand bounds.
        AABB rub = right.bounds; // Unsplit to right:    new right-hand bounds.
        AABB ldb = left.bounds;  // Duplicate:           new left-hand bounds.
        AABB rdb = right.bounds; // Duplicate:           new right-hand bounds.
        lub.grow(refs[leftEnd].bounds);
        rub.grow(refs[leftEnd].bounds);
        ldb.grow(lref.bounds);
        rdb.grow(rref.bounds);

		if(!lref.bounds.valid()) // Unsplit to right
		{
			right.bounds = rub;
			//if(refs[leftEnd].numVisible)
			//	right.boundsVisible.grow(refs[leftEnd].bounds);
			right.numVisible += m_visibility[leftEnd];
			rightStart--;
            swap(refs[leftEnd], refs[rightStart]);
			swap(m_visibility[leftEnd], m_visibility[rightStart]);
			continue;
		}
		if(!rref.bounds.valid()) // Unsplit to left
		{
			left.bounds = lub;
			//if(refs[leftEnd].numVisible)
			//	left.boundsVisible.grow(refs[leftEnd].bounds);
			left.numVisible += m_visibility[leftEnd];
            leftEnd++;
			continue;
		}

        F32 lac = m_platform.getTriangleCost(leftEnd - leftStart);
        F32 rac = m_platform.getTriangleCost(end+1 - rightStart);
        F32 lbc = m_platform.getTriangleCost(leftEnd - leftStart + 1);
        F32 rbc = m_platform.getTriangleCost(end+1 - rightStart + 1);

        F32 unsplitLeftSAH = lub.area() * lbc + right.bounds.area() * rac;
        F32 unsplitRightSAH = left.bounds.area() * lac + rub.area() * rbc;
        F32 duplicateSAH = ldb.area() * lbc + rdb.area() * rbc;
        F32 minSAH = min(unsplitLeftSAH, unsplitRightSAH, duplicateSAH);

        // Unsplit to left?

		if (minSAH == unsplitLeftSAH && (!split.osahChosen || split.leftVisible))
        {
            left.bounds = lub;
			//if(refs[leftEnd].numVisible)
			//	left.boundsVisible.grow(refs[leftEnd].bounds);
			left.numVisible += m_visibility[leftEnd];
            leftEnd++;
        }

        // Unsplit to right?

		else if (minSAH == unsplitRightSAH && (!split.osahChosen || split.rightVisible))
        {
            right.bounds = rub;
			//if(refs[leftEnd].numVisible)
			//	right.boundsVisible.grow(refs[leftEnd].bounds);
			right.numVisible += m_visibility[leftEnd];
			rightStart--;
            swap(refs[leftEnd], refs[rightStart]);
			swap(m_visibility[leftEnd], m_visibility[rightStart]);
        }

        // Duplicate?

        else
        {
            left.bounds = ldb;
            right.bounds = rdb;
			left.numVisible += m_visibility[leftEnd];
			right.numVisible += m_visibility[leftEnd];
			S32 refVis = m_visibility[leftEnd]; // Must be saved because allocation in 'add' might change memory location of the reference
			//if(refs[leftEnd].numVisible)
			//{
			//	left.boundsVisible.grow(lref.bounds);
			//	right.boundsVisible.grow(rref.bounds);
			//}
            refs[leftEnd] = lref;
            refs.add(rref);
			m_visibility.add(refVis); // Add visibility for the new reference
			leftEnd++;
			end++;
			swap(refs[end], refs.getLast()); // If we weren't building from right to left this would cause a bug
			swap(m_visibility[end], m_visibility.getLast());
        }

		//if(left.numVisible && right.numVisible)
		//	fail("Visible triangles division!");
    }

    left.numRef = leftEnd - leftStart;
    right.numRef = end+1 - rightStart;
}

//------------------------------------------------------------------------

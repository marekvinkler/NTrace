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

#include "bvh/SAHBVHBuilder.hpp"
#include "base/Sort.hpp"

using namespace FW;

//------------------------------------------------------------------------

SAHBVHBuilder::SAHBVHBuilder(BVH& bvh, const BVH::BuildParams& params)
:   m_bvh           (bvh),
    m_platform      (bvh.getPlatform()),
    m_params        (params),
    m_sortDim       (-1)
{
}

//------------------------------------------------------------------------

SAHBVHBuilder::~SAHBVHBuilder(void)
{
}

//------------------------------------------------------------------------

BVHNode* SAHBVHBuilder::run(void)
{
    // Initialize reference stack and determine root bounds.

    const Vec3i* tris = (const Vec3i*)m_bvh.getScene()->getTriVtxIndexBuffer().getPtr();
    const Vec3f* verts = (const Vec3f*)m_bvh.getScene()->getVtxPosBuffer().getPtr();

    NodeSpec rootSpec;
	rootSpec.numRef = m_bvh.getScene()->getNumTriangles();
	m_refStack.reset(rootSpec.numRef);

	// Set the references
	for (int i = 0; i < rootSpec.numRef; i++)
	{
		m_refStack[i].triIdx = i;
		for (int j = 0; j < 3; j++)
			m_refStack[i].bounds.grow(verts[tris[i].get(j)]);
		// Inflate the basic boxes so that box intersections are correct
		m_refStack[i].bounds.min() -= BVH_EPSILON;
		m_refStack[i].bounds.max() += BVH_EPSILON;
		rootSpec.bounds.grow(m_refStack[i].bounds);
	}

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

	m_rightBounds.reset(rootSpec.numRef);
	m_numDuplicates = 0;
	m_progressTimer.start();

	// Build recursively.

	BVHNode* root;
	//root = buildNode(rootSpec, 0, 0.0f, 1.0f);
	root = buildNode(rootSpec, 0, rootSpec.numRef-1, 0, 0.0f, 1.0f);

	m_bvh.getTriIndices().compact();

	// Done.

	if (m_params.enablePrints)
		printf("SAHBVHBuilder: progress %.0f%%, duplicates %.0f%%\n",
			100.0f, (F32)m_numDuplicates / (F32)m_bvh.getScene()->getNumTriangles() * 100.0f);
	return root;
}

//------------------------------------------------------------------------

/** WARNING - new version might not work!!! **/
/*
int SAHBVHBuilder::sortCompare(void* data, int idxA, int idxB)
{
    const SAHBVHBuilder* ptr = (const SAHBVHBuilder*)data;
    int dim = ptr->m_sortDim;
    const Reference& ra = ptr->m_refStack[idxA];
    const Reference& rb = ptr->m_refStack[idxB];
    F32 ca = ra.bounds.min()[dim] + ra.bounds.max()[dim];
    F32 cb = rb.bounds.min()[dim] + rb.bounds.max()[dim];
    return (ca < cb) ? -1 : (ca > cb) ? 1 : (ra.triIdx < rb.triIdx) ? -1 : (ra.triIdx > rb.triIdx) ? 1 : 0;
}
*/

bool SAHBVHBuilder::sortCompare(void* data, int idxA, int idxB)
{
    const SAHBVHBuilder* ptr = (const SAHBVHBuilder*)data;
    int dim = ptr->m_sortDim;
    const Reference& ra = ptr->m_refStack[idxA];
    const Reference& rb = ptr->m_refStack[idxB];
    F32 ca = ra.bounds.min()[dim] + ra.bounds.max()[dim];
    F32 cb = rb.bounds.min()[dim] + rb.bounds.max()[dim];
    return (ca < cb) ? false : (ca > cb) ? true : (ra.triIdx < rb.triIdx) ? false : (ra.triIdx > rb.triIdx) ? true : false;
}

//------------------------------------------------------------------------

void SAHBVHBuilder::sortSwap(void* data, int idxA, int idxB)
{
    SAHBVHBuilder* ptr = (SAHBVHBuilder*)data;
    swap(ptr->m_refStack[idxA], ptr->m_refStack[idxB]);
}

//------------------------------------------------------------------------

BVHNode* SAHBVHBuilder::buildNode(const NodeSpec& spec, int level, F32 progressStart, F32 progressEnd)
{
    // Display progress.

    if (m_params.enablePrints && m_progressTimer.getElapsed() >= 1.0f)
    {
        printf("SAHBVHBuilder: progress %.0f%%, duplicates %.0f%%\r",
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

	SplitInfo::SplitType splitType = SplitInfo::SAH;
	S32 axis = 0;

	ObjectSplit object = findObjectSplit(spec, nodeSAH);

    // Leaf SAH is the lowest => create leaf.

    F32 minSAH = min(leafSAH, object.sah);
    if (level != 0 && minSAH == leafSAH && spec.numRef <= m_platform.getMaxLeafSize()) // Make sure we do not make the root a leaf -> GPU traversal will fail
		return createLeaf(spec);

    // Perform split.

    NodeSpec left, right;
    performObjectSplit(left, right, spec, object);
	axis = object.sortDim;

    // Create inner node.

    m_numDuplicates += left.numRef + right.numRef - spec.numRef;
    F32 progressMid = lerp(progressStart, progressEnd, (F32)right.numRef / (F32)(left.numRef + right.numRef));
    BVHNode* rightNode = buildNode(right, level + 1, progressStart, progressMid);
    BVHNode* leftNode = buildNode(left, level + 1, progressMid, progressEnd);

	return new InnerNode(spec.bounds, leftNode, rightNode, axis, splitType, false);
}

//------------------------------------------------------------------------

BVHNode* SAHBVHBuilder::buildNode(const NodeSpec& spec, int start, int end, int level, F32 progressStart, F32 progressEnd)
{
	// Display progress.

    if (m_params.enablePrints && m_progressTimer.getElapsed() >= 1.0f)
    {
        printf("SAHBVHBuilder: progress %.0f%%, duplicates %.0f%%\r",
            progressStart * 100.0f, (F32)m_numDuplicates / (F32)m_bvh.getScene()->getNumTriangles() * 100.0f);
        m_progressTimer.start();
    }

    // Small enough or too deep => create leaf.

    if (level != 0 && spec.numRef <= m_platform.getMinLeafSize() || level >= MaxDepth) // Make sure we do not make the root a leaf -> GPU traversal will fail
		return createLeaf(spec, start, end);

    // Find split candidates.

    F32 area = spec.bounds.area();
    F32 leafSAH = area * m_platform.getTriangleCost(spec.numRef);
    F32 nodeSAH = area * m_platform.getNodeCost(2);
	
	SplitInfo::SplitType splitType = SplitInfo::SAH;
	S32 axis = 0;

	ObjectSplit object = findObjectSplit(start, end, nodeSAH);

    // Leaf SAH is the lowest => create leaf.

    F32 minSAH = min(leafSAH, object.sah);
    if (level != 0 && minSAH == leafSAH && spec.numRef <= m_platform.getMaxLeafSize()) // Make sure we do not make the root a leaf -> GPU traversal will fail
		return createLeaf(spec, start, end);

    // Perform split.

    NodeSpec left, right;
	performObjectSplit(left, right, spec, start, end, object);
	axis = object.sortDim;

    // Create inner node.

    m_numDuplicates += left.numRef + right.numRef - spec.numRef;
    F32 progressMid = lerp(progressStart, progressEnd, (F32)right.numRef / (F32)(left.numRef + right.numRef));

	BVHNode* rightNode = buildNode(right, start+left.numRef, end, level + 1, progressStart, progressMid);
	BVHNode* leftNode = buildNode(left, start, start+left.numRef-1, level + 1, progressMid, progressEnd);

	return new InnerNode(spec.bounds, leftNode, rightNode, axis, splitType, false);
}

//------------------------------------------------------------------------

BVHNode* SAHBVHBuilder::createLeaf(const NodeSpec& spec)
{
    Array<S32>& tris = m_bvh.getTriIndices();
    for (int i = 0; i < spec.numRef; i++)
        tris.add(m_refStack.removeLast().triIdx);
    return new LeafNode(spec.bounds, tris.getSize() - spec.numRef, tris.getSize());
}

//------------------------------------------------------------------------

BVHNode* SAHBVHBuilder::createLeaf(const NodeSpec& spec, int start, int end)
{
    Array<S32>& tris = m_bvh.getTriIndices();
	for(int i = end; i >= start; i--)
		tris.add(m_refStack[i].triIdx);

	return new LeafNode(spec.bounds, tris.getSize() - spec.numRef, tris.getSize());
}

//------------------------------------------------------------------------

SAHBVHBuilder::ObjectSplit SAHBVHBuilder::findObjectSplit(const NodeSpec& spec, F32 nodeSAH)
{
    ObjectSplit split;
    const Reference* refPtr = m_refStack.getPtr(m_refStack.getSize() - spec.numRef);

    // Sort along each dimension.

    for (m_sortDim = 0; m_sortDim < 3; m_sortDim++)
    {
        sort(this, m_refStack.getSize() - spec.numRef, m_refStack.getSize(), sortCompare, sortSwap);

        // Sweep right to left and determine bounds.

        AABB rightBounds;
        for (int i = spec.numRef - 1; i > 0; i--)
        {
            rightBounds.grow(refPtr[i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        AABB leftBounds;
        for (int i = 1; i < spec.numRef; i++)
        {
            leftBounds.grow(refPtr[i - 1].bounds);
			F32 sah = nodeSAH + leftBounds.area() * m_platform.getTriangleCost(i) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost(spec.numRef - i);
            if (sah < split.sah)
            {
                split.sah = sah;
                split.sortDim = m_sortDim;
                split.numLeft = i;
                split.leftBounds = leftBounds;
                split.rightBounds = m_rightBounds[i - 1];
            }
        }
    }

    return split;
}

//------------------------------------------------------------------------

SAHBVHBuilder::ObjectSplit SAHBVHBuilder::findObjectSplit(int start, int end, F32 nodeSAH)
{
    ObjectSplit split;
	Reference* refPtr = m_refStack.getPtr(start);

    // Sort along each dimension.

    for (m_sortDim = 0; m_sortDim < 3; m_sortDim++)
    {
		sort(this, start, end+1, sortCompare, sortSwap);

        // Sweep right to left and determine bounds.

		AABB rightBounds;
        for (int i = end-start; i > 0; i--)
        {
            rightBounds.grow(refPtr[i].bounds);
            m_rightBounds[i - 1] = rightBounds;
        }

        // Sweep left to right and select lowest SAH.

        AABB leftBounds;
        for (int i = 1; i < end-start+1; i++)
        {
			leftBounds.grow(refPtr[i - 1].bounds);
			F32 sah = nodeSAH + leftBounds.area() * m_platform.getTriangleCost(i) + m_rightBounds[i - 1].area() * m_platform.getTriangleCost((end-start+1) - i);

			if(sah < split.sah)
			{
				split.sah = sah;
				split.sortDim = m_sortDim;
				split.numLeft = i;
				split.leftBounds = leftBounds;
				split.rightBounds = m_rightBounds[i - 1];
			}
        }
	}

	return split;
}

//------------------------------------------------------------------------

void SAHBVHBuilder::performObjectSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const ObjectSplit& split)
{
    m_sortDim = split.sortDim;
	sort(this, m_refStack.getSize() - spec.numRef, m_refStack.getSize(), sortCompare, sortSwap);

    left.numRef = split.numLeft;
    left.bounds = split.leftBounds;
    right.numRef = spec.numRef - split.numLeft;
    right.bounds = split.rightBounds;
}

//------------------------------------------------------------------------

void SAHBVHBuilder::performObjectSplit(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, int start, int end, const ObjectSplit& split)
{
    m_sortDim = split.sortDim;
	sort(this, start, end+1, sortCompare, sortSwap);

    left.numRef = split.numLeft;
    left.bounds = split.leftBounds;
    right.numRef = spec.numRef - split.numLeft;
    right.bounds = split.rightBounds;
}

//------------------------------------------------------------------------

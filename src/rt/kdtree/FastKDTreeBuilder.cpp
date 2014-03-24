/*
*  Copyright (c) 2013, Radek Stibora
*  All rights reserved.
*
*  Redistribution and use in source and binary forms, with or without
*  modification, are permitted provided that the following conditions are met:
*      * Redistributions of source code must retain the above copyright
*        notice, this list of conditions and the following disclaimer.
*      * Redistributions in binary form must reproduce the above copyright
*        notice, this list of conditions and the following disclaimer in the
*        documentation and/or other materials provided with the distribution.
*      * Neither the name of the <organization> nor the
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

#include "kdtree/FastKDTreeBuilder.hpp"

#include "base/Sort.hpp"

//#define DEBUG
#define MERGESORT

using namespace FW;

//------------------------------------------------------------------------

FastKDTreeBuilder::FastKDTreeBuilder(KDTree& kdtree, const KDTree::BuildParams& params)
:
	m_kdtree		(kdtree),
	m_platform		(kdtree.getPlatform()),
	m_params		(params),
	m_numDuplicates	(0),
	m_maxDepth		((S32)(1.2f * log2((FW::F32)kdtree.getScene()->getNumTriangles()) + 2.f)),
	m_maxFailSplits ((S32)(1.f + 0.2f * m_maxDepth))
	{}

//------------------------------------------------------------------------

KDTreeNode* FastKDTreeBuilder::run(void)
{
	// Initialize event and triangle index stack and determine root bounds.

	const Vec3i* tris = (const Vec3i*)m_kdtree.getScene()->getTriVtxIndexBuffer().getPtr();
	const Vec3f* verts = (const Vec3f*)m_kdtree.getScene()->getVtxPosBuffer().getPtr();

	NodeSpec rootSpec;

	m_measureTimer.unstart();
	m_measureTimer.start();

	m_evStack.reset();
	m_triStack.reset();
	m_mergeSortBuffer.reset();

	// Compute bounds and generate initial set of events.

	rootSpec.numTri = m_kdtree.getScene()->getNumTriangles();
	m_triData.resize(rootSpec.numTri);
	Event e;
	for (int i = 0; i < rootSpec.numTri; i++)
	{
		m_triStack.add(i);

		// Compute bounds.
		AABB triBounds;
		for (int j = 0; j < 3; j++)
			triBounds.grow(verts[tris[i][j]]);
		rootSpec.bounds.grow(triBounds);

		e.triIdx = i;

		for (int dim = 0; dim < 3; dim++)
		{
			e.dim = dim;

			if (triBounds.min()[dim] == triBounds.max()[dim])
			{
				e.pos = triBounds.min()[dim];
				e.type = Planar;
				m_evStack.add(e);
			}
			else
			{
				e.pos = triBounds.min()[dim];
				e.type = Start;
				m_evStack.add(e);

				e.pos = triBounds.max()[dim];
				e.type = End;
				m_evStack.add(e);
			}
		}
	}

#ifndef MERGESORT
	sort(&m_evStack, 0, m_evStack.getSize(),  eventSortCompare, eventSortSwap);
#else
	m_mergeSortBuffer.reserve(m_evStack.getSize());
	msort(m_evStack, 0, m_evStack.getSize());
#endif
	rootSpec.numEv = m_evStack.getSize();

	F32 initTime =  m_measureTimer.getElapsed();
	if (m_params.enablePrints)
	{
		std::printf("FastKDTreeBuilder: initialization done in %f seconds.\n", initTime);
	}

	m_progressTimer.start();

	// Build recursively.

	KDTreeNode* root = buildNode(rootSpec, 0, 0, 0.0f, 1.0f);
	m_kdtree.getTriIndices().compact();

	F32 totalTime = m_measureTimer.getElapsed();
	F32 buildTime = totalTime - initTime;

	if (m_params.enablePrints)
	{
		std::printf("FastKDTreeBuilder: progress %.0f%%, built in %f seconds. Total time: %f seconds\n", 100.0f, buildTime, totalTime);
	}

	return root;
}

//------------------------------------------------------------------------

KDTreeNode*	FastKDTreeBuilder::createLeaf(const NodeSpec& spec)
{
	Array<S32>& tris = m_kdtree.getTriIndices();

	int stackSize = m_triStack.getSize();
	for (int i = stackSize - spec.numTri; i < stackSize; i++)
	{
		tris.add(m_triStack.removeLast());
	}

	m_evStack.remove(m_evStack.getSize() - spec.numEv, m_evStack.getSize());

	return new KDTLeafNode(tris.getSize() - spec.numTri, tris.getSize());
}

//------------------------------------------------------------------------

KDTreeNode* FastKDTreeBuilder::buildNode(const NodeSpec& spec, int level, int forcedSplits, F32 progressStart, F32 progressEnd)
{
	// Display progress.

	if (m_params.enablePrints && m_progressTimer.getElapsed() >= 1.0f)
    {
        std::printf("FastKDTreeBuilder: progress %.0f%%\r", progressStart * 100.0f);
        m_progressTimer.start();
    }

	if (/*spec.numTri <= m_platform.getMaxLeafSize() ||*/ level == m_maxDepth)
		return createLeaf(spec);

	F32 nodePrice = m_platform.getTriangleCost(spec.numTri);
	Split split = findSplit(spec);

#ifdef DEBUG
	if (split.price < 0.f)
		FW_ASSERT(0);
#endif

	if (split.price / nodePrice > 0.9f)
		forcedSplits++;

	if (split.price == FW_F32_MAX || forcedSplits > m_maxFailSplits)
		return createLeaf(spec);

	NodeSpec left, right;
	performSplit(left, right, spec, split);

	F32 progressMid = lerp(progressStart, progressEnd, (F32)right.numTri / (F32)(left.numTri + right.numTri));
	KDTreeNode* rightNode = buildNode(right, level + 1, forcedSplits, progressStart, progressMid);
    KDTreeNode* leftNode = buildNode(left, level + 1, forcedSplits, progressMid, progressEnd);
    
	return new KDTInnerNode(split.pos, split.dim, leftNode, rightNode);
}

//------------------------------------------------------------------------

FastKDTreeBuilder::Split FastKDTreeBuilder::findSplit(const NodeSpec& spec) const
{
	//if (spec.bounds.min()[0] == spec.bounds.max()[0] ||
	//	spec.bounds.min()[1] == spec.bounds.max()[1] ||
	//	spec.bounds.min()[2] == spec.bounds.max()[2]   ) // ??
	//{
	//	Split dontSplit;
	//	dontSplit.price = FW_F32_MAX;
	//	return dontSplit;
	//}

	S32 nl [3]; S32 np [3]; S32 nr [3];


	for (int i = 0; i < 3; i++)
	{
		nl[i] = 0; np[i] = 0; nr[i] = spec.numTri;
	}

	Split bestSplit;
	for (int i = m_evStack.getSize() - spec.numEv; i < m_evStack.getSize();)
	{
		S32 numEnds = 0; S32 numPlanar = 0; S32 numStarts = 0;
		const F32 pos = m_evStack.get(i).pos;
		const S32 dim = m_evStack.get(i).dim;

		while(i < m_evStack.getSize() && m_evStack.get(i).dim == dim &&
			m_evStack.get(i).pos == pos && m_evStack.get(i).type == End)
		{
			numEnds++; i++;
		}

		while(i < m_evStack.getSize() && m_evStack.get(i).dim == dim &&
			m_evStack.get(i).pos == pos && m_evStack.get(i).type == Planar)
		{
			numPlanar++; i++;
		}

		while(i < m_evStack.getSize() && m_evStack.get(i).dim == dim &&
			m_evStack.get(i).pos == pos && m_evStack.get(i).type == Start)
		{
			numStarts++; i++;
		}

		np[dim] = numPlanar;
		nr[dim] -= numPlanar;
		nr[dim] -= numEnds;

		Split currentSplit;
		currentSplit.dim = dim;
		currentSplit.pos = pos;

		F32 costLeftSide = sahPrice(currentSplit, spec.bounds, nl[dim] + np[dim], nr[dim]);
		F32 costRightSide = sahPrice(currentSplit, spec.bounds, nl[dim], nr[dim] + np[dim]);
		if (costLeftSide < costRightSide)
		{
			currentSplit.price = costLeftSide;
			currentSplit.side = Left;
		}
		else
		{
			currentSplit.price = costRightSide;
			currentSplit.side = Right;
		}

#ifdef DEBUG
		if (currentSplit.pos < spec.bounds.min()[dim] || currentSplit.pos > spec.bounds.max()[dim])
			FW_ASSERT(0);
#endif

		if(currentSplit.price < bestSplit.price)
			bestSplit = currentSplit;

		nl[dim] += numStarts;
		nl[dim] += numPlanar;
		np[dim] = 0;
	}
	return bestSplit;
}

//------------------------------------------------------------------------

void FastKDTreeBuilder::performSplit (NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const Split& split)
{
	// Clasiffy triangles.

	for (int i = m_evStack.getSize() - spec.numEv; i < m_evStack.getSize(); i++)
	{
		const Event& currEv = m_evStack[i];

		if (currEv.type == End && currEv.dim == split.dim && currEv.pos <= split.pos)
		{
			m_triData[currEv.triIdx].side = LeftOnly;
		}
		else if (currEv.type == Start && currEv.dim == split.dim && currEv.pos >= split.pos)
		{
			m_triData[currEv.triIdx].side = RightOnly;
		}
		else if (currEv.type == Planar && currEv.dim == split.dim)
		{
			if (currEv.pos < split.pos || (currEv.pos == split.pos && split.side == Left))
			{
				m_triData[currEv.triIdx].side = LeftOnly;
			}
			else if (currEv.pos > split.pos || (currEv.pos == split.pos && split.side == Right))
			{
				m_triData[currEv.triIdx].side = RightOnly;
			}
		}
	}
	
	// Separate events related to triangles which do not straddle the split plane.

	for (int i = m_evStack.getSize() - spec.numEv; i < m_evStack.getSize(); i++)
	{
		if (m_triData[m_evStack[i].triIdx].side == LeftOnly)
		{
			m_eventsLO.add(m_evStack[i]);
		}
		else if (m_triData[m_evStack[i].triIdx].side == RightOnly)
		{
			m_eventsRO.add(m_evStack[i]);
		}
	}

	// Process straddling triangles. Also adjust triangle index stack.

	for (int i = m_triStack.getSize() - spec.numTri; i < m_triStack.getSize(); i++)
	{
		if (m_triData[m_triStack[i]].side == LeftOnly)
		{
			m_leftTriIdx.add(m_triStack[i]);
		}
		else if (m_triData[m_triStack[i]].side == RightOnly)
		{
			m_rightTriIdx.add(m_triStack[i]);
		}

		// Generate new events.

		if (m_triData[m_triStack[i]].side == Both)
		{
			AABB leftBounds; AABB rightBounds;
			splitBounds(leftBounds, rightBounds, m_triStack[i], split);

			leftBounds.intersect(spec.bounds);
			rightBounds.intersect(spec.bounds);
			//leftBounds.intersect(m_triData[m_triStack[i]].bounds);
			//rightBounds.intersect(m_triData[m_triStack[i]].bounds);

			AABB* bounds[] = {NULL, NULL};
			Array<Event>* events [] = {&m_eventsBL, &m_eventsBR};

			if (leftBounds.valid())
			{
				m_leftTriIdx.add(m_triStack[i]);
				bounds[0] = &leftBounds;
			}
			if (rightBounds.valid())
			{
				m_rightTriIdx.add(m_triStack[i]);
				bounds[1] = &rightBounds;
			}
			
			m_numDuplicates++;

			Event e;
			e.triIdx = m_triStack[i];

			for (int s = 0; s < 2; s++)
			{
				if (bounds[s] == NULL)
					continue;

				const Vec3f min = bounds[s]->min();
				const Vec3f max = bounds[s]->max();

				for (int dim = 0; dim < 3; dim++)
				{
					e.dim = dim;

					if(min[dim] == max[dim])
					{
						e.pos = min[dim];
						e.type = Planar;
						events[s]->add(e);
					}
					else
					{
						e.pos = min[dim];
						e.type = Start;
						events[s]->add(e);

						e.pos = max[dim];
						e.type = End;
						events[s]->add(e);
					}
				}
			}
		}

		m_triData[m_triStack[i]].side = Both;
	}

#ifndef MERGESORT
	sort(&m_eventsBL, 0, m_eventsBL.getSize(),  eventSortCompare, eventSortSwap);
	sort(&m_eventsBR, 0, m_eventsBR.getSize(),  eventSortCompare, eventSortSwap);
#else
	msort(m_eventsBL, 0, m_eventsBL.getSize());
	msort(m_eventsBR, 0, m_eventsBR.getSize());
#endif

	// Merge events.

	S32 stackTop = m_evStack.getSize() - spec.numEv;
	left.numEv = m_eventsLO.getSize() + m_eventsBL.getSize();
	right.numEv = m_eventsRO.getSize() + m_eventsBR.getSize();
	m_evStack.resize((m_evStack.getSize() - spec.numEv) + left.numEv + right.numEv);

	mergeEvents(stackTop, m_eventsLO, m_eventsBL);
	mergeEvents(stackTop, m_eventsRO, m_eventsBR);

	left.numTri = m_leftTriIdx.getSize();
	right.numTri = m_rightTriIdx.getSize();

	stackTop = m_triStack.getSize() - spec.numTri;
	m_triStack.resize(m_triStack.getSize() - spec.numTri);

	m_triStack.insert(stackTop, m_leftTriIdx);
	stackTop += left.numTri;
	m_triStack.insert(stackTop, m_rightTriIdx);

	// Split bounding box.

	left.bounds = spec.bounds;
	left.bounds.max()[split.dim] = split.pos;

	right.bounds = spec.bounds;
	right.bounds.min()[split.dim] = split.pos;

	m_eventsLO.clear();
	m_eventsRO.clear();
	m_eventsBL.clear();
	m_eventsBR.clear();
	m_leftTriIdx.clear();
	m_rightTriIdx.clear();
}

//------------------------------------------------------------------------

void FastKDTreeBuilder::mergeEvents (S32& stackTop, const Array<Event>& a, const Array<Event>& b)
{
	int idxA = 0; int idxB = 0;
	for (int i = 0; i < a.getSize() + b.getSize(); i++)
	{
		if (idxA == a.getSize())
		{
			m_evStack[stackTop++] = b.get(idxB++);
		}
		else if (idxB == b.getSize())
		{
			m_evStack[stackTop++] = a.get(idxA++);
		}
		else if (FastKDTreeBuilder::eventCompare(a.get(idxA), b.get(idxB)))
		{
			m_evStack[stackTop++] = a.get(idxA++);
		}
		else
		{
			m_evStack[stackTop++] = b.get(idxB++);
		}
	}
}

//------------------------------------------------------------------------

F32 FastKDTreeBuilder::sahPrice(const Split& split, const AABB& bounds, S32 nl, S32 nr) const
{
	AABB leftBounds = bounds;
	leftBounds.max()[split.dim] = split.pos;

	AABB rightBounds = bounds;
	rightBounds.min()[split.dim] = split.pos;

	//if (bounds.min()[split.dim] == bounds.max()[split.dim])
	//{
	//	return FW_F32_MAX;
	//}

	if (bounds.min()[0] == bounds.max()[0] ||
		bounds.min()[1] == bounds.max()[1] ||
		bounds.min()[2] == bounds.max()[2]   ) // ??
	{
		return FW_F32_MAX;
	}

	if ((split.pos == bounds.min()[split.dim] && nl == 0) || (split.pos == bounds.max()[split.dim] && nr == 0))
	{
		return FW_F32_MAX;
	}
	
	const F32 probabilityLeft = leftBounds.area() / bounds.area();
	const F32 probabilityRight = rightBounds.area() / bounds.area();

	F32 cost = probabilityLeft * m_platform.getTriangleCost(nl) + probabilityRight * m_platform.getTriangleCost(nr);
	if ((nl == 0 || nr == 0) && !(split.pos == bounds.min()[split.dim] || split.pos == bounds.max()[split.dim])) // No bonus for flat empty cells.
		cost *= 0.8f; // m_platform ??

	cost += m_platform.getNodeCost(1); // ??

	return cost;
}

//------------------------------------------------------------------------

void FastKDTreeBuilder::splitBounds (AABB& left, AABB& right, S32 triIdx, const Split& split) const
{
	const Vec3i* tris = (const Vec3i*)m_kdtree.getScene()->getTriVtxIndexBuffer().getPtr();
    const Vec3f* verts = (const Vec3f*)m_kdtree.getScene()->getVtxPosBuffer().getPtr();

	Vec3f vertices[] = {verts[tris[triIdx][0]], verts[tris[triIdx][1]], verts[tris[triIdx][2]]};

#ifdef DEBUG
	//if (split.pos > m_triData[triIdx].bounds.max()[split.dim] || split.pos < m_triData[triIdx].bounds.min()[split.dim])
	//{
	//	FW_ASSERT(0);
	//}
#endif

	int leftmostVertIdx = 0; int secondVertIdx = 0; int thirdVertIdx = 0;
	for (int i = 0; i < 3; i++)
	{
		if (vertices[i][split.dim] < vertices[leftmostVertIdx][split.dim])
		{
			leftmostVertIdx = i;
		}
	}

	if (vertices[(leftmostVertIdx+1) % 3][split.dim] < vertices[(leftmostVertIdx+2) % 3][split.dim])
	{	
		secondVertIdx = (leftmostVertIdx+1) % 3;
		thirdVertIdx = (leftmostVertIdx+2) % 3;
	}
	else
	{
		secondVertIdx = (leftmostVertIdx+2) % 3;
		thirdVertIdx = (leftmostVertIdx+1) % 3;
	}

	// Actual bbox construction here.

	left = AABB(); right = AABB();
	left.grow(vertices[leftmostVertIdx]);
	right.grow(vertices[thirdVertIdx]);

	if (vertices[secondVertIdx][split.dim] <= split.pos)
	{
		F32	firstToSplit = split.pos - vertices[leftmostVertIdx][split.dim];
		F32 secondToSplit = split.pos - vertices[secondVertIdx][split.dim];
		F32 firstToThird = vertices[thirdVertIdx][split.dim] - vertices[leftmostVertIdx][split.dim];
		F32 secondToThird = vertices[thirdVertIdx][split.dim] - vertices[secondVertIdx][split.dim];

		Vec3f edge1 = vertices[thirdVertIdx] - vertices[leftmostVertIdx];
		edge1 *= firstToSplit / firstToThird;

		Vec3f edge2 = vertices[thirdVertIdx] - vertices[secondVertIdx];
		edge2 *= secondToSplit / secondToThird;

		Vec3f newVert1 = vertices[leftmostVertIdx] + edge1;
		Vec3f newVert2 = vertices[secondVertIdx] + edge2;

		newVert1[split.dim] = split.pos; // ??
		newVert2[split.dim] = split.pos;

		left.grow(vertices[secondVertIdx]);
		left.grow(newVert1);
		left.grow(newVert2);
		right.grow(newVert1);
		right.grow(newVert2);

	}
	else if (vertices[secondVertIdx][split.dim] > split.pos)
	{
		F32	firstToSplit = split.pos - vertices[leftmostVertIdx][split.dim];
		F32 firstToSecond = vertices[secondVertIdx][split.dim] - vertices[leftmostVertIdx][split.dim];
		F32 firstToThird = vertices[thirdVertIdx][split.dim] - vertices[leftmostVertIdx][split.dim];

		Vec3f edge1 = vertices[secondVertIdx] - vertices[leftmostVertIdx];
		edge1 *= firstToSplit / firstToSecond;

		Vec3f edge2 = vertices[thirdVertIdx] - vertices[leftmostVertIdx];
		edge2 *= firstToSplit / firstToThird;

		Vec3f newVert1 = vertices[leftmostVertIdx] + edge1;
		Vec3f newVert2 = vertices[leftmostVertIdx] + edge2;

		newVert1[split.dim] = split.pos; // ??
		newVert2[split.dim] = split.pos;

		left.grow(newVert1);
		left.grow(newVert2);
		right.grow(newVert1);
		right.grow(newVert2);
		right.grow(vertices[secondVertIdx]);
	}
#ifdef DEBUG
	else
	{
		FW_ASSERT(0);
	}

	if(!left.valid())
		FW_ASSERT(0);
	if(!right.valid())
		FW_ASSERT(0);
#endif
}

//------------------------------------------------------------------------

bool FastKDTreeBuilder::eventSortCompare(void* data, int idxA, int idxB)
{
    const Array<Event>* ptr = (const Array<Event>*)data;
    const Event& ea = ptr->get(idxA);
    const Event& eb = ptr->get(idxB);
	return FastKDTreeBuilder::eventCompare(ea, eb);
}

//------------------------------------------------------------------------

void FastKDTreeBuilder::eventSortSwap(void* data, int idxA, int idxB)
{
    Array<Event>* ptr = (Array<Event>*)data;
    swap(ptr->get(idxA), ptr->get(idxB));
}

//------------------------------------------------------------------------

bool FastKDTreeBuilder::eventCompare (const Event& eventA, const Event& eventB)
{
	if (eventA.pos == eventB.pos)
	{
		if (eventA.dim == eventB.dim)
		{
			return (eventA.type < eventB.type);
		}
		else
		{
			return (eventA.dim < eventB.dim);
		}
	}
	else
	{
		return (eventA.pos < eventB.pos);
	}
}

//------------------------------------------------------------------------

void FastKDTreeBuilder::msort(Array<Event>& data, S32 l, S32 h)
{
	if ((h - l) <= 1)
		return;

	int middle = (h + l) / 2;

	msort(data, l, middle);
	msort(data, middle, h);
	mmerge(data, l, middle, h);
}

//------------------------------------------------------------------------

void FastKDTreeBuilder::mmerge(Array<Event>& data, S32 l, S32 m, S32 h)
{
	S32 idxL = l;
	S32 idxR = m;
	S32 idxB = 0;
	S32 pos = idxL;

	m_mergeSortBuffer.clear();

	while (pos < h)
	{
		if (m_mergeSortBuffer.getSize() - 1 < idxB)
		{
			if (idxL >= m)
			{
				idxR++;
				pos++;
			}
			else if (idxR >= h)
			{
				idxL++;
				pos++;
			}
			else if (eventCompare(data[idxL], data[idxR]))
			{
				idxL++;
				pos++;
			}
			else
			{
				m_mergeSortBuffer.add(data[pos]);
				data[pos] = data[idxR];
				idxR++;
				idxL++;
				pos++;
			}
		}
		else
		{
			if (idxR >= h)
			{
				data[pos] = m_mergeSortBuffer[idxB];
				idxB++;
				pos++;
			}
			else if (eventCompare(m_mergeSortBuffer[idxB], data[idxR]))
			{
				if (pos < m)
				{
					m_mergeSortBuffer.add(data[pos]);
				}
				data[pos] = m_mergeSortBuffer[idxB];
				idxB++;
				idxL++;
				pos++;
			}
			else
			{
				if (pos < m)
				{
					m_mergeSortBuffer.add(data[pos]);
				}
				data[pos] = data[idxR];
				idxR++;
				pos++;
			}
		}
	}

#ifdef DEBUG

	for (S32 i = l; i < h - 1; i++)
	{
		if (eventCompare(data[i+1], data[i]))
			FW_ASSERT(0);
	}

#endif
}
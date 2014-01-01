
#include "kdtree/FastKDTreeBuilder.hpp"

#include "base/Sort.hpp"

using namespace FW;

//------------------------------------------------------------------------

FastKDTreeBuilder::FastKDTreeBuilder(KDTree& kdtree, const KDTree::BuildParams& params)
:
	m_kdtree		(kdtree),
	m_platform		(kdtree.getPlatform()),
	m_params		(params),
	m_numDuplicates	(0)
	{}

//------------------------------------------------------------------------

KDTreeNode* FastKDTreeBuilder::run(void)
{
	// Initialize event stack and determine root bounds.

	const Vec3i* tris = (const Vec3i*)m_kdtree.getScene()->getTriVtxIndexBuffer().getPtr();
	const Vec3f* verts = (const Vec3f*)m_kdtree.getScene()->getVtxPosBuffer().getPtr();

	NodeSpec rootSpec;

	m_evStack.reset();
	rootSpec.numTri = m_kdtree.getScene()->getNumTriangles();
	m_triData.resize(rootSpec.numTri);
	Event e;
	for (int i = 0; i < rootSpec.numTri; i++)
	{
		// Compute bounds.
		for (int j =0; j < 3; j++)
			m_triData[i].bounds.grow(verts[tris[i][j]]);
		rootSpec.bounds.grow(m_triData[i].bounds);

		e.triIdx = i;

		for (int dim = 0; dim < 3; dim++)
		{
			e.dim = dim;

			if (m_triData[e.triIdx].bounds.min()[dim] == m_triData[e.triIdx].bounds.max()[dim])
			{
				e.pos = m_triData[e.triIdx].bounds.min()[dim];
				e.type = Planar;
				m_evStack.add(e);
			}
			else
			{
				e.pos = m_triData[e.triIdx].bounds.min()[dim];
				e.type = Start;
				m_evStack.add(e);

				e.pos = m_triData[e.triIdx].bounds.max()[dim];
				e.type = End;
				m_evStack.add(e);
			}
		}
	}

	sort(&m_evStack, 0, m_evStack.getSize(),  eventSortCompare, eventSortSwap); // ?? nlogn

	rootSpec.numEv = m_evStack.getSize();

	if (m_params.enablePrints)
        std::printf("FastKDTreeBuilder: initialization done\n");

	m_progressTimer.start();

	// Build recursively.

	KDTreeNode* root = buildNode(rootSpec, 0, 0.0f, 1.0f);
	m_kdtree.getTriIndices().compact();

	if (m_params.enablePrints)
		std::printf("FastKDTreeBuilder: progress %.0f%%\n", 100.0f);

	return root;
}

//------------------------------------------------------------------------

KDTreeNode*	FastKDTreeBuilder::createLeaf(const NodeSpec& spec)
{
	Array<S32>& tris = m_kdtree.getTriIndices();

	for (int i = 0; i < m_triData.getSize(); i++)
		m_triData[i].relevant = true;

	int size = m_evStack.getSize();
	for (int i = size-1; i >= size-spec.numEv; i--)
	{
		if (m_triData[m_evStack[i].triIdx].relevant == true)
		{
			tris.add(m_evStack[i].triIdx);
			m_triData[m_evStack[i].triIdx].relevant = false;
		}
		m_evStack.removeLast();
	}

	for (int i = 0; i < m_triData.getSize(); i++)
		m_triData[i].relevant = false;

	return new KDTLeafNode(tris.getSize() - spec.numTri, tris.getSize());
}

//------------------------------------------------------------------------

KDTreeNode* FastKDTreeBuilder::buildNode(NodeSpec spec, int level, F32 progressStart, F32 progressEnd)
{
	// Display progress.

	if (m_params.enablePrints && m_progressTimer.getElapsed() >= 1.0f)
    {
        std::printf("FastKDTreeBuilder: progress %.0f%%\r", progressStart * 100.0f);
        m_progressTimer.start();
    }

	// Too deep => create leaf.

	if (spec.numTri <= m_platform.getMaxLeafSize() || level >= MaxDepth)
		return createLeaf(spec);

	F32 nodeSah = m_platform.getTriangleCost(spec.numTri);
	Split split = findSplit(spec);
	if (split.sah < 0)
		FW_ASSERT(0);

	if (nodeSah <= split.sah)
		return createLeaf(spec);

	NodeSpec left, right;
	performSplit(left, right, spec, split);

	F32 progressMid = lerp(progressStart, progressEnd, (F32)right.numTri / (F32)(left.numTri + right.numTri));
	KDTreeNode* rightNode = buildNode(right, level + 1, progressStart, progressMid);
    KDTreeNode* leftNode = buildNode(left, level + 1, progressMid, progressEnd);
    
	return new KDTInnerNode(split.pos, split.dim, leftNode, rightNode);
}

//------------------------------------------------------------------------

FastKDTreeBuilder::Split FastKDTreeBuilder::findSplit(const NodeSpec& spec)
{
	if (spec.bounds.min()[0] == spec.bounds.max()[0] ||
		spec.bounds.min()[1] == spec.bounds.max()[1] ||
		spec.bounds.min()[2] == spec.bounds.max()[2]   ) // ??
	{
		Split dontSplit;
		dontSplit.sah = FW_F32_MAX;
		return dontSplit;
	}

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
			m_evStack.get(i).pos == pos && m_evStack.get(i).type == eventType::End)
		{
			numEnds++; i++;
		}

		while(i < m_evStack.getSize() && m_evStack.get(i).dim == dim &&
			m_evStack.get(i).pos == pos && m_evStack.get(i).type == eventType::Planar)
		{
			numPlanar++; i++;
		}

		while(i < m_evStack.getSize() && m_evStack.get(i).dim == dim &&
			m_evStack.get(i).pos == pos && m_evStack.get(i).type == eventType::Start)
		{
			numStarts++; i++;
		}

		np[dim] = numPlanar;
		nr[dim] -= numPlanar;
		nr[dim] -= numEnds;

		Split currentSplit;
		currentSplit.dim = dim;
		currentSplit.pos = pos;

		sah(currentSplit, spec.bounds, nl[dim], nr[dim], np[dim]);

		if (currentSplit.pos < spec.bounds.min()[dim] || currentSplit.pos > spec.bounds.max()[dim])
			currentSplit.sah = FW_F32_MAX;

		if(currentSplit.sah < bestSplit.sah)
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
			if (currEv.pos > split.pos || (currEv.pos == split.pos && split.side == Right))
			{
				m_triData[currEv.triIdx].side = RightOnly;
			}
		}
		m_triData[currEv.triIdx].relevant = true;
	}

	// Split events to left only and right only.
	Array<Event> eventsLO;
	Array<Event> eventsRO;

	for (int i = m_evStack.getSize() - spec.numEv; i < m_evStack.getSize(); i++)
	{
		if (m_triData[m_evStack[i].triIdx].side == LeftOnly)
		{
			eventsLO.add(m_evStack[i]);
		}
		else if (m_triData[m_evStack[i].triIdx].side == RightOnly)
		{
			eventsRO.add(m_evStack[i]);
		}
	}

	// Generate new events from triangles overalapping split plane.
	Array<Event> eventsBL;
	Array<Event> eventsBR;

	for (int i = 0; i < m_triData.getSize(); i++)
	{
		if (m_triData[i].side == Both && m_triData[i].relevant)
		{
			AABB leftBounds; AABB rightBounds;
			splitBounds(leftBounds, rightBounds, i, split);

			leftBounds.intersect(spec.bounds);
			rightBounds.intersect(spec.bounds);

			AABB* bounds[] = {NULL, NULL};

			if (leftBounds.valid())
				bounds[0] = &leftBounds;
			if (rightBounds.valid())
				bounds[1] = &rightBounds;
			
			m_numDuplicates++;

			Event e;
			e.triIdx = i;

			for (int s = 0; s < 2; s++)
			{
				if (bounds[s] == NULL)
					continue;

				Vec3f min = bounds[s]->min();
				Vec3f max = bounds[s]->max();

				for (int dim = 0; dim < 3; dim++)
				{
					e.dim = dim;

					if(min[dim] == max[dim])
					{
						e.pos = min[dim];
						e.type = eventType::Planar;
						if (s == 0)
							eventsBL.add(e);
						else
							eventsBR.add(e);
					}
					else
					{
						e.pos = min[dim];
						e.type = eventType::Start;
						if (s == 0)
							eventsBL.add(e);
						else
							eventsBR.add(e);

						e.pos = max[dim];
						e.type = eventType::End;
						if (s == 0)
							eventsBL.add(e);
						else
							eventsBR.add(e);
					}
				}
			}
		}
	}

	sort(&eventsBL, 0, eventsBL.getSize(),  eventSortCompare, eventSortSwap);
	sort(&eventsBR, 0, eventsBR.getSize(),  eventSortCompare, eventSortSwap);

	// Merge all four strains.
	Array<Event> eventsL = mergeEvents(eventsLO, eventsBL);
	eventsLO.reset();
	eventsBL.reset();
	left.numEv = eventsL.getSize();

	Array<Event> eventsR = mergeEvents(eventsRO, eventsBR);
	eventsRO.reset();
	eventsBR.reset();
	right.numEv = eventsR.getSize();

	S32 stackTop = m_evStack.getSize() - spec.numEv; // Discard this node's events.
	m_evStack.resize(m_evStack.getSize() - spec.numEv + eventsL.getSize() + eventsR.getSize());

	for (int i = 0; i < eventsL.getSize(); i++)
	{
		m_evStack[stackTop + i] = eventsL[i];
	}
	stackTop += eventsL.getSize();
	for (int i = 0; i < eventsR.getSize(); i++)
	{
		m_evStack[stackTop + i] = eventsR[i];
	}


	left.numTri = 0;
	right.numTri = 0;
	for (int i = 0; i < m_triData.getSize(); i++)
	{
		if (m_triData[i].relevant)
		{
			if (m_triData[i].side == Left)
				left.numTri++;
			else if (m_triData[i].side == Right)
				right.numTri++;
			else
			{
				left.numTri++;
				right.numTri++;
			}
		}
		
		m_triData[i].side = Both;
		m_triData[i].relevant = false;
	}

	Vec3f leftCut = spec.bounds.max();
	leftCut[split.dim] = split.pos;
	AABB leftBounds(spec.bounds.min(), leftCut);

	Vec3f rightCut = spec.bounds.min();
	rightCut[split.dim] = split.pos;
	AABB rightBounds(rightCut, spec.bounds.max());

	left.bounds = leftBounds;
	right.bounds = rightBounds;
}

//------------------------------------------------------------------------

Array<FastKDTreeBuilder::Event> FastKDTreeBuilder::mergeEvents (const Array<Event>& a, const Array<Event>& b) const
{
	Array<Event> merged;
	merged.reset(a.getSize() + b.getSize());

	if (a.getSize() == 0)
		return Array<Event>(b);

	if (b.getSize() == 0)
		return Array<Event>(a);

	int idxA = 0;
	int idxB = 0;
	for (int i = 0; i < merged.getSize(); i++)
	{
		if (idxA == a.getSize())
		{
			merged[i] = b.get(idxB);
			idxB++;
		}
		else if (idxB == b.getSize())
		{
			merged[i] = a.get(idxA);
			idxA++;
		}
		else if (FastKDTreeBuilder::eventCompare(a.get(idxA), b.get(idxB)))
		{
			merged[i] = a.get(idxA);
			idxA++;
		}
		else
		{
			merged[i] = b.get(idxB);
			idxB++;
		}
	}

	return merged;
}

//------------------------------------------------------------------------

void FastKDTreeBuilder::sah(Split& split, const AABB& bounds, S32 nl, S32 nr, S32 np)
{
	Vec3f leftCut = bounds.max();
	leftCut[split.dim] = split.pos;
	AABB leftBounds(bounds.min(), leftCut);

	Vec3f rightCut = bounds.min();
	rightCut[split.dim] = split.pos;
	AABB rightBounds(rightCut, bounds.max());
	
	F32 probabilityLeft = leftBounds.area() / bounds.area();
	F32 probabilityRight = rightBounds.area() / bounds.area();

	F32 costLeftSide = probabilityLeft * m_platform.getTriangleCost(nl + np) + probabilityRight * m_platform.getTriangleCost(nr);
	if (((nl + np) == 0 || nr == 0) && !(split.pos == bounds.min()[split.dim] || split.pos == bounds.max()[split.dim])) // No bonus for flat empty cells.
		costLeftSide *= 0.8; // m_platform ??

	F32 costRightSide = probabilityLeft * m_platform.getTriangleCost(nl) + probabilityRight * m_platform.getTriangleCost(np + nr);
	if ((nl == 0 || (np + nr) == 0) && !(split.pos == bounds.min()[split.dim] || split.pos == bounds.max()[split.dim])) // No bonus for flat empty cells.
		costRightSide *= 0.8; // m_platform ??

	if(costLeftSide < costRightSide)
	{
		split.side = Left;
		split.sah = costLeftSide;
	}
	else
	{
		split.side = Right;
		split.sah  = costRightSide;
	}

	split.sah += m_platform.getNodeCost(1); // ??
}

//------------------------------------------------------------------------

void FastKDTreeBuilder::splitBounds (AABB& left, AABB& right, S32 triIdx, const Split& split)
{
	const Vec3i* tris = (const Vec3i*)m_kdtree.getScene()->getTriVtxIndexBuffer().getPtr();
    const Vec3f* verts = (const Vec3f*)m_kdtree.getScene()->getVtxPosBuffer().getPtr();

	Vec3f vertices[] = {verts[tris[triIdx][0]], verts[tris[triIdx][1]], verts[tris[triIdx][2]]};

	if (split.pos > m_triData[triIdx].bounds.max()[split.dim] || split.pos < m_triData[triIdx].bounds.min()[split.dim])
	{
		FW_ASSERT(0);
	}

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

	if (vertices[secondVertIdx][split.dim] < split.pos)
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
	else if (vertices[secondVertIdx][split.dim] == split.pos && vertices[secondVertIdx][split.dim] != vertices[leftmostVertIdx][split.dim])
	{
		F32	firstToSplit = split.pos - vertices[leftmostVertIdx][split.dim];
		F32 firstToThird = vertices[thirdVertIdx][split.dim] - vertices[leftmostVertIdx][split.dim];

		Vec3f edge1 = vertices[thirdVertIdx] - vertices[leftmostVertIdx];
		edge1 *= firstToSplit / firstToThird;

		Vec3f newVert1 = vertices[leftmostVertIdx] + edge1;
		
		newVert1[split.dim] = split.pos;

		left.grow(newVert1);
		left.grow(vertices[secondVertIdx]);
		right.grow(newVert1);
		right.grow(vertices[secondVertIdx]);
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
	else
	{
		FW_ASSERT(0);
	}

	if(!left.valid())
		FW_ASSERT(0);
	if(!right.valid())
		FW_ASSERT(0);
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
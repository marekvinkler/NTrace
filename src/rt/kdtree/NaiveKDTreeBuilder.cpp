
#include "NaiveKDTreeBuilder.hpp"
#include "base/Sort.hpp"
#include <random>

namespace FW
{

NaiveKDTreeBuilder::NaiveKDTreeBuilder(KDTree& kdtree, const KDTree::BuildParams& params)
:	m_kdtree		(kdtree),
	m_platform		(kdtree.getPlatform()),
	m_params		(params),
	m_sortDim		(0),
	m_numDuplicates	(0)
{
}


KDTreeNode* NaiveKDTreeBuilder::run(void)
{
	const Vec3i* tris = (const Vec3i*)m_kdtree.getScene()->getTriVtxIndexBuffer().getPtr();
    const Vec3f* verts = (const Vec3f*)m_kdtree.getScene()->getVtxPosBuffer().getPtr();

	NodeSpec rootSpec;
    rootSpec.numRef = m_kdtree.getScene()->getNumTriangles();
    m_refStack.resize(rootSpec.numRef);

    for (int i = 0; i < rootSpec.numRef; i++)
    {
        m_refStack[i].triIdx = i;
        for (int j = 0; j < 3; j++)
            m_refStack[i].bounds.grow(verts[tris[i][j]]);
        rootSpec.bounds.grow(m_refStack[i].bounds);
    }	

	m_progressTimer.start();
	KDTreeNode* root = buildNode(rootSpec, 0, 0, 0.f, 1.f);
	m_kdtree.getTriIndices().compact();

    if (m_params.enablePrints)
        std::printf("\rNaiveKDTreeBuilder: progress %.0f%%\n", 100.0f);
    return root;
	
}


KDTreeNode* NaiveKDTreeBuilder::buildNode(NodeSpec spec, int level, S32 currentAxis, F32 progressStart, F32 progressEnd)
{
	if (m_params.enablePrints && m_progressTimer.getElapsed() >= 1.0f)
    {
        std::printf("NaiveKDTreeBuilder: progress %.0f%%\r", progressStart * 100.0f);
        m_progressTimer.start();
    }

	if(spec.numRef <= m_platform.getMaxLeafSize() || level >= MaxDepth)
		return createLeaf(spec);

	NodeSpec left, right;

	Split split = findMedianSplit(spec, currentAxis);
	performMedianSplit(left, right, spec, split);

	F32 progressMid = lerp(progressStart, progressEnd, (F32)right.numRef / (F32)(left.numRef + right.numRef));
	KDTreeNode* rightNode = buildNode(right, level + 1, nextCoordinate(currentAxis), progressStart, progressMid);
	KDTreeNode* leftNode = buildNode(left, level + 1, nextCoordinate(currentAxis), progressMid, progressEnd);
	return new KDTInnerNode(split.pos, split.dim, leftNode, rightNode);
}


KDTreeNode*	NaiveKDTreeBuilder::createLeaf(const NodeSpec& spec)
{
    Array<S32>& tris = m_kdtree.getTriIndices();
    for (int i = 0; i < spec.numRef; i++)
        tris.add(m_refStack.removeLast().triIdx);
    return new KDTLeafNode(tris.getSize() - spec.numRef, tris.getSize());
}


NaiveKDTreeBuilder::Split NaiveKDTreeBuilder::findMedianSplit (const NodeSpec& spec, S32 dim)
{
	Array<Reference>& refs = m_refStack;
	
	F32 splitPos = -1;

	if(m_params.builder == FW::KDTree::SpatialMedian)
	{
		F32 boundsMin = spec.bounds.min()[dim];
		F32 boundsMax = spec.bounds.max()[dim];

		splitPos = (boundsMin + boundsMax) / 2;
	}
	else if(m_params.builder == FW::KDTree::ObjectMedian) // ??
	{
		m_sortDim = dim;
		sort(this, refs.getSize() - spec.numRef, refs.getSize(), momCompare, momSwap);

		S32 medIdx = refs.getSize() - (int)(spec.numRef / 2);
		splitPos = (refs.get(medIdx).bounds.min()[dim] + refs.get(medIdx).bounds.max()[dim]) / 2;
	}
	else
		FW_ASSERT(0); // Should not occur.

	Split foundSplit;
	foundSplit.dim = dim;
	foundSplit.pos = splitPos;

	return foundSplit;
}


void NaiveKDTreeBuilder::performMedianSplit (NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const Split& split)
{
	Array<Reference>& refs = m_refStack;
	int leftStart = refs.getSize() - spec.numRef;
	int leftEnd = leftStart;
	int rightStart = refs.getSize();

    for (int i = leftEnd; i < rightStart; i++)
    {
        if (refs[i].bounds.max()[split.dim] <= split.pos)
        {
            swap(refs[i], refs[leftEnd++]);
        }

        else if (refs[i].bounds.min()[split.dim] >= split.pos)
        {
            swap(refs[i--], refs[--rightStart]);
        }
    }

	// Duplicate references intersecting both sides.


	for (int i = leftEnd; i < rightStart; i++)
	{
		refs.add(Reference(refs.get(i)));
		leftEnd++;
		m_numDuplicates++;
	}


	left.numRef = leftEnd - leftStart;
	right.numRef = refs.getSize() - rightStart;

	Vec3f leftCut = spec.bounds.max();
	leftCut[split.dim] = split.pos;

	Vec3f rightCut = spec.bounds.min();
	rightCut[split.dim] = split.pos;

	left.bounds = AABB(spec.bounds.min(), leftCut);
	right.bounds = AABB(rightCut, spec.bounds.max());
}


bool NaiveKDTreeBuilder::momCompare(void* data, int idxA, int idxB)
{
    const NaiveKDTreeBuilder* ptr = (const NaiveKDTreeBuilder*)data;
    int dim = ptr->m_sortDim;
    const Reference& ra = ptr->m_refStack[idxA];
    const Reference& rb = ptr->m_refStack[idxB];
    F32 ca = ra.bounds.min()[dim] + ra.bounds.max()[dim];
    F32 cb = rb.bounds.min()[dim] + rb.bounds.max()[dim];
    return (ca < cb || (ca == cb && ra.triIdx < rb.triIdx));
}

void NaiveKDTreeBuilder::momSwap(void* data, int idxA, int idxB)
{
    NaiveKDTreeBuilder* ptr = (NaiveKDTreeBuilder*)data;
    swap(ptr->m_refStack[idxA], ptr->m_refStack[idxB]);
}


}
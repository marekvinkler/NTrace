
#include "CudaKDTree.hpp"

using namespace FW;

//------------------------------------------------------------------------

CudaKDTree::CudaKDTree(const KDTree& kdtree)
{
	createNodeTriIdx(kdtree);
	createWoopTri(kdtree);
}

//------------------------------------------------------------------------

CudaKDTree::CudaKDTree(InputStream& in)
{
	Vec3f min, max;
	in >> min >> max >> m_nodes >> m_triWoop >> m_triIndex;
	m_bbox = AABB(min, max);
}

//------------------------------------------------------------------------

CudaKDTree::~CudaKDTree(void)
{
}

//------------------------------------------------------------------------

void CudaKDTree::serialize(OutputStream& out)
{
	out << m_bbox.min() << m_bbox.max() << m_nodes << m_triWoop << m_triIndex;
}

//------------------------------------------------------------------------

//CudaKDTree& CudaKDTree::operator=(CudaKDTree& other)
//{
//    if (&other != this)
//    {
//        m_nodes     = other.m_nodes;
//        m_triWoop   = other.m_triWoop;
//        m_triIndex  = other.m_triIndex;
//    }
//    return *this;
//}

//------------------------------------------------------------------------

void CudaKDTree::createWoopTri(const KDTree& kdtree)
{
	const int triangleCount = kdtree.getScene()->getNumTriangles();

	Vec4f woop[3];
	m_triWoop.resizeDiscard((triangleCount * 48 + 4096 - 1) & -4096);

    const Vec3i* triVtxIndex = (const Vec3i*)kdtree.getScene()->getTriVtxIndexBuffer().getPtr();
    const Vec3f* vtxPos = (const Vec3f*)kdtree.getScene()->getVtxPosBuffer().getPtr();

	for (int i = 0; i < triangleCount; i++)
	{
		//const Vec3i& inds = triVtxIndex[kdtree.getTriIndices()[i]];
		const Vec3i& inds = triVtxIndex[i];
		const Vec3f& v0 = vtxPos[inds.x];
		const Vec3f& v1 = vtxPos[inds.y];
		const Vec3f& v2 = vtxPos[inds.z];

		Mat4f mtx;
		mtx.setCol(0, Vec4f(v0 - v2, 0.0f));
		mtx.setCol(1, Vec4f(v1 - v2, 0.0f));
		mtx.setCol(2, Vec4f(cross(v0 - v2, v1 - v2), 0.0f));
		mtx.setCol(3, Vec4f(v2, 1.0f));
		mtx = invert(mtx);

		woop[0] = Vec4f(mtx(2,0), mtx(2,1), mtx(2,2), -mtx(2,3));
		woop[1] = mtx.getRow(0);
		woop[2] = mtx.getRow(1);

		memcpy(m_triWoop.getMutablePtr(i * 48), woop, 48);
	}
}

//------------------------------------------------------------------------

void CudaKDTree::createNodeTriIdx(const KDTree& kdtree)
{
	const Vec3i* tris = (const Vec3i*)kdtree.getScene()->getTriVtxIndexBuffer().getPtr(); // Necessary for bbox construction.
    const Vec3f* verts = (const Vec3f*)kdtree.getScene()->getVtxPosBuffer().getPtr();

	Array<Vec4i>		nodeData;
	Array<int>			triIndexData;

	const KDTreeNode* root = kdtree.getRoot();
	const int nodeCount = root->getSubtreeSize(KDTREE_STAT_NODE_COUNT);

	nodeData.resize(nodeCount);

	int nextNodeIdx = 0;
	int nextLeafIdx = 0; 
	Array<StackEntry> stack(StackEntry(root, nextNodeIdx++));

	while (stack.getSize())
	{
		StackEntry e = stack.removeLast(); // Pop stack. 'e' is always an inner node; leaves are not stored on the stack, they are processed immediately.

		const KDTreeNode* children[] = { e.node->getChildNode(0), e.node->getChildNode(1) };
		StackEntry childrenEntries[2];

		for (int c = 0; c < 2; c++) // Process both child nodes.
		{
			if(children[c]->isLeaf()) // Write to triIdx array.
			{
				childrenEntries[c] = StackEntry(children[c], nextLeafIdx);

				const Array<S32>& tidx = kdtree.getTriIndices();
				const KDTLeafNode* leaf = reinterpret_cast<const KDTLeafNode*>(childrenEntries[c].node);
				// Write leaf's triangle indexes followed by 0x8000000 and then increment nextLeafIdx to point to next free space in triIdx array.
				for(int i = leaf->m_lo; i < leaf->m_hi; i++)
				{
					triIndexData.add(tidx.get(i));
					for (int j = 0; j < 3; j++)
					{
						m_bbox.grow(verts[tris[tidx.get(i)][j]]);
					}
				}

				if (leaf->getNumTriangles() == 0)
				{
					childrenEntries[c].idx = ~KDTREE_EMPTYLEAF;
				}
				else
				{
					triIndexData.add(KDTREE_EMPTYLEAF);
					nextLeafIdx += leaf->m_hi - leaf->m_lo + 1;
				}
			}
			else
				childrenEntries[c] = stack.add(StackEntry(children[c], nextNodeIdx++)); // Add node on top of the node stack.
		}

		const KDTInnerNode* node = reinterpret_cast<const KDTInnerNode*>(e.node);

		S32 leftOffset = childrenEntries[0].encodeIdx();
		S32 rightOffset = childrenEntries[1].encodeIdx();
		F32 splitPos = node->m_pos;
		S32 flags = node->m_axis << 28;
		flags = flags & KDTREE_MASK;

		Vec4i data(leftOffset, rightOffset, floatToBits(splitPos), flags);
		nodeData.set(e.idx, data);
	}

	m_nodes.resizeDiscard(nodeData.getNumBytes());
    m_nodes.set(nodeData.getPtr(), nodeData.getNumBytes());

	m_triIndex.resizeDiscard(triIndexData.getNumBytes());
	m_triIndex.set(triIndexData.getPtr(), triIndexData.getNumBytes());
}

#include "CudaKDTree.hpp"

#define DFS 0
#define BFS 1
#define RND 2

#define METHOD DFS

#include "base/Random.hpp"
#include <queue>
#include <cmath>

using namespace FW;

//------------------------------------------------------------------------

CudaKDTree::CudaKDTree(const KDTree& kdtree)
{
	m_scene = kdtree.getScene();

#if METHOD == DFS
	createNodeTriIdx(kdtree);
#elif METHOD == RND
	createNodeTriIdx(kdtree);
	shuffle();
#else
	createNodeTriIdxBFS(kdtree);
#endif
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
	// Transfer data to CPU before serialization
	m_nodes.setOwner(Buffer::CPU, false);
	m_triWoop.setOwner(Buffer::CPU, false);
	m_triIndex.setOwner(Buffer::CPU, false);
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
	const int nodeCount = root->getSubtreeSize(KDTREE_STAT_INNER_COUNT);
	//const int nodeCount = root->getSubtreeSize(KDTREE_STAT_NODE_COUNT);

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

void CudaKDTree::shuffle()
{
	struct ParentInfo
	{
		S32 parentIdx;
		bool leftChild;

		ParentInfo(S32 idx, bool left) : parentIdx(idx), leftChild(left) {}
		ParentInfo() : parentIdx(-1) {}
	};

	int numOfNodes = m_nodes.getSize() / 16;
	int numSwaps = numOfNodes;

	Array<ParentInfo> parents;
	parents.reset(numOfNodes);

	Vec4i* ptr = (Vec4i*)m_nodes.getPtr();
	for (int i = 0; i < numOfNodes; i++)
	{
		S32 ptr1 = ptr[i].x;
		S32 ptr2 = ptr[i].y;

		if(ptr1 > 0)
			parents[ptr1] = ParentInfo(i, true);
		if(ptr2 > 0)
			parents[ptr2] = ParentInfo(i, false);
	}

	Random rnd;

	for (int i = 0; i < numSwaps; i++)
	{
		S32 idx1 = rnd.getU32(1, numOfNodes-1);
		S32 idx2 = rnd.getU32(1, numOfNodes-1);
		if(idx1 == idx2)
			continue;

		S32 chIdx1l = ptr[idx1].x;
		S32 chIdx1r = ptr[idx1].y;
		S32 chIdx2l = ptr[idx2].x;
		S32 chIdx2r = ptr[idx2].y;

		//swap nodes
		swap(ptr[idx1], ptr[idx2]);

		//swap parent info
		swap(parents[idx1], parents[idx2]);

		//fix parent info for children
		if(chIdx1l > 0 && parents[chIdx1l].parentIdx == idx1)
			parents[chIdx1l].parentIdx = idx2;
		if(chIdx1r > 0 && parents[chIdx1r].parentIdx == idx1)
			parents[chIdx1r].parentIdx = idx2;
		if(chIdx2l > 0 && parents[chIdx2l].parentIdx == idx2)
			parents[chIdx2l].parentIdx = idx1;
		if(chIdx2r > 0 && parents[chIdx2r].parentIdx == idx2)
			parents[chIdx2r].parentIdx = idx1;

		//??
		if(parents[idx1].parentIdx == idx1)
			parents[idx1].parentIdx = idx2;
		if(parents[idx2].parentIdx == idx2)
			parents[idx2].parentIdx = idx1;

		//fix child info for swapped nodes
		if(parents[idx1].leftChild)
			ptr[parents[idx1].parentIdx].x = idx1;
		else
			ptr[parents[idx1].parentIdx].y = idx1;

		if(parents[idx2].leftChild)
			ptr[parents[idx2].parentIdx].x = idx2;
		else
			ptr[parents[idx2].parentIdx].y = idx2;

		//sPtr[parents[idx1].parentIdx * 4 + (parents[idx1].leftChild ? 0 : 1)] = idx1 * 4;
		//sPtr[parents[idx2].parentIdx * 4 + (parents[idx2].leftChild ? 0 : 1)] = idx2 * 4;
	}
}

void CudaKDTree::createNodeTriIdxBFS(const KDTree& kdtree)
{
	const Vec3i* tris = (const Vec3i*)kdtree.getScene()->getTriVtxIndexBuffer().getPtr(); // Necessary for bbox construction.
    const Vec3f* verts = (const Vec3f*)kdtree.getScene()->getVtxPosBuffer().getPtr();

	Array<Vec4i>		nodeData;
	Array<int>			triIndexData;

	const KDTreeNode* root = kdtree.getRoot();
	const int nodeCount = root->getSubtreeSize(KDTREE_STAT_INNER_COUNT);
	//const int nodeCount = root->getSubtreeSize(KDTREE_STAT_NODE_COUNT);

	nodeData.resize(nodeCount);

	int nextNodeIdx = 0;
	int nextLeafIdx = 0; 
	std::queue<StackEntry> queue;
	queue.push(StackEntry(root, nextNodeIdx++));
	//Array<StackEntry> stack(StackEntry(root, nextNodeIdx++));

	while (!queue.empty())
	{
		StackEntry e = queue.front();
		queue.pop();// Pop stack. 'e' is always an inner node; leaves are not stored on the stack, they are processed immediately.

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
			{
				queue.push(StackEntry(children[c], nextNodeIdx++)); // Add node on top of the node stack.
				childrenEntries[c] = queue.back();
			}
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

void CudaKDTree::trace(RayBuffer& rays, Buffer& visibility)
{
	m_needClosestHit = rays.getNeedClosestHit();

	S32* visib = (S32*)visibility.getMutablePtr();

	for(S32 i=0;i<rays.getSize();i++)
	{
		Ray ray = rays.getRayForSlot(i);    // takes a local copy
		RayResult& result = rays.getMutableResultForSlot(i);

		result.clear();

		trace(0, ray, result);

		// Set visibility
		if(visibility.getSize() > 0 && result.hit())
			visib[result.id] = 1;
	}
}

void CudaKDTree::trace(S32 node, Ray& ray, RayResult& result)
{
	S32 stack[100];
	//F32 tStack[100];
	int stackIndex = 1;	
	const int TMIN = 0;
	const int TMAX = 1;

	Vec4i cell = ((Vec4i*)getNodeBuffer().getPtr())[node];

	Vec2f t = Intersect::RayBox(m_bbox, ray);

	if(!((t[TMIN]<=t[TMAX]) && (t[TMAX]>=ray.tmin) && (t[TMIN]<=ray.tmax)))
		return;

	/*			
	float origDim;
	float idirDim;
	float tmin;
	float tmax;

	float ooeps = FW::exp2(-80.0f); // Avoid div by zero.
	float idirx = 1.0f / (fabsf(ray.direction.x) > ooeps ? ray.direction.x : _copysign(ooeps, ray.direction.x)); //??
	float idiry = 1.0f / (fabsf(ray.direction.y) > ooeps ? ray.direction.y : _copysign(ooeps, ray.direction.y));
	float idirz = 1.0f / (fabsf(ray.direction.z) > ooeps ? ray.direction.z : _copysign(ooeps, ray.direction.z));
	float oodx = ray.origin.x * idirx; 
	float oody = ray.origin.y * idiry;
	float oodz = ray.origin.z * idirz;

	float clox = m_bbox.min()[0] * idirx - oodx;
	float chix = m_bbox.max()[0] * idirx - oodx;
	float cloy = m_bbox.min()[1] * idiry - oody;
	float chiy = m_bbox.max()[1] * idiry - oody;
	float cloz = m_bbox.min()[2] * idirz - oodz;
	float chiz = m_bbox.max()[2] * idirz - oodz;

	//float hitT = 1.f;

	tmin = FW::max(FW::min(clox, chix), FW::min(cloy, chiy), FW::min(cloz, chiz), tmin) - 1e-4f;
	tmax = FW::min(FW::max(clox, chix), FW::max(cloy, chiy), FW::max(cloz, chiz), hitT) + 1e-4f;
	*/



	while(stackIndex > 0)
	{
		for(;;)
		{
			if(node < 0)
			{
				bool end = intersectTriangles(node, ray, result);
				if(end)
					return;

				break;
			}
			else
			{
				Vec4i cell = ((Vec4i*)getNodeBuffer().getPtr())[node];
				Vec3f d = ray.direction;

				unsigned int type = cell.w & KDTREE_MASK;
				int dim = (type >> KDTREE_DIMPOS);
				float split = bitsToFloat(cell.z);

				Vec3f idir;
				float ooeps = exp2(-80.0f); // Avoid div by zero.
				idir.x = 1.0f / (fabsf(d.x) > ooeps ? d.x : _copysign(ooeps, d.x));
				idir.y = 1.0f / (fabsf(d.y) > ooeps ? d.y : _copysign(ooeps, d.y));
				idir.z = 1.0f / (fabsf(d.z) > ooeps ? d.z : _copysign(ooeps, d.z));


				float origDim;
				float idirDim;
				// Gather data for split plane intersection.
				switch(dim)	
				{
					case 0: origDim = ray.origin.x; idirDim = idir.x; break;
					case 1: origDim = ray.origin.y; idirDim = idir.y; break;
					case 2: origDim = ray.origin.z; idirDim = idir.z; break;
				}

				float tsplit = (split - origDim) * idirDim;

				bool nfd = ((*(unsigned int*)&idirDim) >> 31); // Choose based on the sign bit
				int first = nfd ? cell.y : cell.x;
				int second = nfd ? cell.x : cell.y;

				if(tsplit > t[TMAX])
				{
					node = first;
				}
				else if(tsplit < t[TMIN])
				{
					node = second;
				}
				else
				{
					node = first;
					stack[stackIndex++] = second;
				}
			}
		}
	//do
	//{
		stackIndex--;
		node = stack[stackIndex];
	//} while(tStack[stackIndex] > ray.tmax);
	}
}

bool CudaKDTree::intersectTriangles(S32 node, Ray& ray, RayResult& result)
{
	Buffer &nodes = getNodeBuffer();
	Buffer &tris = getTriIndexBuffer();
	S32* ptr = (S32*)tris.getPtr();

	node = ~node;

	// get node's triangles
	int lo, hi;
	lo = node;
	hi = lo;
	while(ptr[hi] != KDTREE_EMPTYLEAF)
		hi++;

	for(int i=lo; i<hi; i++)
	{
		S32 index = *(S32*)tris.getMutablePtr(i*4);

		const Vec3i& ind = ((Vec3i*)m_scene->getTriVtxIndexBuffer().getPtr())[index];
		const Vec3f& v0 = ((Vec3f*)m_scene->getVtxPosBuffer().getPtr())[ind.x];
		const Vec3f& v1 = ((Vec3f*)m_scene->getVtxPosBuffer().getPtr())[ind.y];
		const Vec3f& v2 = ((Vec3f*)m_scene->getVtxPosBuffer().getPtr())[ind.z];
		Vec3f bary = Intersect::RayTriangle(v0,v1,v2, ray);
		float t = bary[2];

		bool end = updateHit(ray, result, t, index);
		if(end)
			return true;
	}

	return false;
}

bool CudaKDTree::updateHit(Ray& ray, RayResult& result, float t, S32 index)
{
	if(t>ray.tmin && t<ray.tmax)
	{
		ray.tmax    = t;
		result.t    = t;
		result.id   = index;

		//if(!m_needClosestHit)
		//	return true;
	}

	return false;
}

#pragma once
#include <cstdio>
#include "Scene.hpp"
#include "KDTreeNode.hpp"
#include "ray/RayBuffer.hpp"

namespace FW
{

class KDTree
{
public:
	enum BuilderType
	{
		SpatialMedian,
		ObjectMedian,
		SAH
	};

	struct Stats
	{
		Stats()				{ clear(); }
		void clear()		{ memset(this, 0, sizeof(Stats)); }
		void print() const  { std::printf("Tree stats: [bfactor=2] %d nodes (%d+%d), ?.2f SAHCost, %.1f children/inner, %.1f tris/leaf, %.1f%% duplicates, %d empty leaves\n",numLeafNodes+numInnerNodes, numLeafNodes,numInnerNodes, 1.f*numChildNodes/max(numInnerNodes,1), 1.f*numTris/max(numLeafNodes,1), 1.f*percentDuplicates, numEmptyLeaves); }

        S32     numInnerNodes;
        S32     numLeafNodes;
        S32     numChildNodes;
        S32     numTris;
		S32		numEmptyLeaves;
		F32		percentDuplicates;
	};

	struct BuildParams
	{
		Stats*				stats;
		bool				enablePrints;
		//bool				spatialMedian;
		BuilderType			builder;				

		BuildParams(void)
		{
			stats			= nullptr;
			enablePrints	= true;
			builder			= SpatialMedian;
		}
	};

	KDTree				(Scene* scene, const Platform& platform, const BuildParams& params);
	~KDTree				(void)						{ if(m_root != nullptr) m_root->deleteSubtree(); }

	Scene*				getScene (void) const		{ return m_scene; }
	const Platform&		getPlatform (void) const	{ return m_platform; }
	KDTreeNode*			getRoot (void) const		{ return m_root; }

	Array<S32>&			getTriIndices (void)		{ return m_triIndices; }
	const Array<S32>&   getTriIndices (void) const	{ return m_triIndices; }

private:
	Scene*				m_scene;
	Platform			m_platform;

	KDTreeNode*			m_root;
	Array<S32>			m_triIndices;
};


}
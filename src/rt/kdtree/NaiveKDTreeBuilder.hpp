
#pragma once
#include "KDTree.hpp"
#include "base/Timer.hpp"

namespace FW
{

class NaiveKDTreeBuilder
{
private:
	enum
	{
		MaxDepth			= 18,
	};

	struct Reference
    {
        S32                 triIdx;
        AABB                bounds;

        Reference(void) : triIdx(-1) {}
    };

	struct NodeSpec
    {
        S32                 numRef;
		AABB                bounds;

        NodeSpec(void) : numRef(0) {}
    };

	struct Split
    {
        S32                 dim;
        F32                 pos;

        Split(void) :  dim(0), pos(0.0f) {}
    };

public:
								NaiveKDTreeBuilder		(KDTree& kdtree,const KDTree::BuildParams& params);
								~NaiveKDTreeBuilder		(void) {}

	KDTreeNode*					run						(void);

	S32							getNumDuplicates		(void)	{ return m_numDuplicates; }

private:
	static bool					momCompare				(void* data, int idxA, int idxB);
    static void					momSwap					(void* data, int idxA, int idxB);

	KDTreeNode*					buildNode				(NodeSpec spec, int level, S32 currentAxis, F32 progressStart, F32 progressEnd);
	KDTreeNode*					createLeaf				(const NodeSpec& spec);

	static S32					nextCoordinate			(S32 current) { return ((current + 1) % 3); }
	Split						findMedianSplit			(const NodeSpec&, S32 currentAxis);
	void						performMedianSplit		(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const Split& split);

private:
								NaiveKDTreeBuilder		(const NaiveKDTreeBuilder&); // forbidden
	NaiveKDTreeBuilder&			operator=				(const NaiveKDTreeBuilder&); // forbidden

private:
	KDTree&						m_kdtree;
	const Platform&				m_platform;
	const KDTree::BuildParams&	m_params;
	Array<Reference>			m_refStack;
	S32							m_sortDim;

	Timer						m_progressTimer;
	S32							m_numDuplicates;
};

}
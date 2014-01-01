
#pragma once
#include "kdtree/KDTree.hpp"
#include "base/Timer.hpp"

namespace FW
{

class FastKDTreeBuilder
{
private:
	enum
	{
		MaxDepth			= 10
	};

	enum eventType
	{
		End,
		Planar,
		Start
	};

	enum sahSide
	{
		Left,
		Right
	};

	enum splitSide
	{
		LeftOnly,
		RightOnly,
		Both
	};

	struct Event
	{
		S32					triIdx;

		F32					pos;
		S32					dim;
		eventType			type;

		Event() : triIdx(-1), pos(0.f), dim(-1) {}
	};

	struct NodeSpec
    {
        S32                 numEv;
		AABB                bounds;
		S32					numTri;

        NodeSpec(void) : numEv(0) {}
    };

	struct Split
	{
		F32					sah;
		S32					dim;
		F32					pos;
		sahSide				side;

		Split() : sah(FW_F32_MAX), dim(-1), pos(0.f) {}
	};

	struct TriData
	{
		AABB				bounds;
		splitSide			side;
		bool				relevant;

		TriData() : side(Both), relevant(false) {}
	};

public:

							FastKDTreeBuilder		(KDTree& kdtree, const KDTree::BuildParams& params);
							~FastKDTreeBuilder		(void)	{}

	KDTreeNode*				run						(void);

	S32						getNumDuplicates		(void)	{ return m_numDuplicates; }

private:
	KDTreeNode*				buildNode				(NodeSpec spec, int level, F32 progressStart, F32 progressEnd);
	KDTreeNode*				createLeaf				(const NodeSpec& spec);
	Split					findSplit				(const NodeSpec& spec);
	void					performSplit			(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const Split& split);

	static bool				eventSortCompare		(void* data, int idxA, int idxB);
	static void				eventSortSwap			(void* data, int idxA, int idxB);
	static bool				eventCompare			(const Event& eventA, const Event& eventB);

	void					splitBounds				(AABB& left, AABB& right, S32 triIdx, const Split& split); 

	Array<FastKDTreeBuilder::Event>		mergeEvents				(const Array<Event>& a, const Array<Event>& b) const;

	void					sah						(Split& split, const AABB& bounds, S32 nl, S32 nr, S32 np);



private:
	KDTree&						m_kdtree;
	const Platform&				m_platform;
	const KDTree::BuildParams&	m_params;

	Array<Event>				m_evStack;
	Array<TriData>				m_triData;

	Timer						m_progressTimer;
	S32							m_numDuplicates;
};

}
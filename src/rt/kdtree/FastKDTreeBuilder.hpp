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
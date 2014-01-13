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
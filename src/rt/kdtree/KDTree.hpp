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
#include <cstdio>
#include "Scene.hpp"
#include "KDTreeNode.hpp"
#include "ray/RayBuffer.hpp"
#include "acceleration\Acceleration.hpp"

namespace FW
{

class KDTree : public AccelerationStructure
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

	KDTreeNode*			getRoot (void) const		{ return m_root; }

	Array<S32>&			getTriIndices (void)		{ return m_triIndices; }
	const Array<S32>&   getTriIndices (void) const	{ return m_triIndices; }

private:
	KDTreeNode*			m_root;
	Array<S32>			m_triIndices;
};


}
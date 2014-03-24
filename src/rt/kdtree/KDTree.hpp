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

/**
 * \brief K-d tree acceleration structure class.
 */
class KDTree
{
public:
	/**
	 * \brief Supported k-d tree builder types.
	 */
	enum BuilderType
	{
		SpatialMedian,
		ObjectMedian,
		SAH
	};

	/**
	 * \brief Structure holding statistics about k-d tree.
	 */
	struct Stats
	{
		/**
		 * \brief Constructor.
		 */
		Stats()				{ clear(); }

		/**
		 * \brief Resets data to default values.
		 */
		void clear()		{ memset(this, 0, sizeof(Stats)); }

		/**
		 * \brief Prints statistics to stdout.
		 */
		void print() const  { std::printf("Tree stats: [bfactor=2] %d nodes (%d+%d), ?.2f SAHCost, %.1f children/inner, %.1f tris/leaf, %.1f%% duplicates, %d empty leaves\n",numLeafNodes+numInnerNodes, numLeafNodes,numInnerNodes, 1.f*numChildNodes/max(numInnerNodes,1), 1.f*numTris/max(numLeafNodes,1), 1.f*percentDuplicates, numEmptyLeaves); }

        S32     numInnerNodes;		//!< Number of inner nodes.
        S32     numLeafNodes;		//!< Number of leaf nodes.
        S32     numChildNodes;		//!< Number of child nodes
        S32     numTris;			//!< Triangle count of the source scene.
		S32		numEmptyLeaves;		//!< Number of empty leaves.
		F32		percentDuplicates;	//!< Percentage ratio of duplicated references to initial number of references (which is same as triangle count).
	};

	/**
	 * \brief Strucure holding build parameters.
	 */
	struct BuildParams
	{
		/**
		 * \brief Constructor.
		 */
		BuildParams(void)
		{
			stats			= nullptr;
			enablePrints	= true;
			builder			= SpatialMedian;
		}

		Stats*				stats;			//!< Statistics collected during build phase. Set to NULL if no stats should be collected.
		bool				enablePrints;	//!< Flag whether to print information during build phase.
		BuilderType			builder;		//!< Defines which builder type will be used to build the k-d tree.
	};

	/**
	 * \brief Constructor.
	 * \param[in] scene		Source scene.
	 * \param[in] platform	Platform settings.
	 * \param[in] params	Build parameters.
	 */
	KDTree				(Scene* scene, const Platform& platform, const BuildParams& params);

	/**
	 * \brief Destructor.
	 */
	~KDTree				(void)						{ if(m_root != nullptr) m_root->deleteSubtree(); }

	/**
	 * \brief Gets source scene of the k-d tree.
	 * \return Source scene.
	 */
	Scene*				getScene (void) const		{ return m_scene; }

	/**
	 * \brief Gets platform settings of the k-d tree.
	 * \brief Platform settings.
	 */
	const Platform&		getPlatform (void) const	{ return m_platform; }

	/**
	 * \brief Gets root node of the k-d tree.
	 * \return K-d tree's root node.
	 */
	KDTreeNode*			getRoot (void) const		{ return m_root; }

	/**
	 *  \brief Returns an array of triangle indices to which leaf nodes are pointig. These indices point to scene's triangle array.
	 *  \return Array of triangle indices.
	 */
	Array<S32>&			getTriIndices (void)		{ return m_triIndices; }

	/**
	 *  \brief Returns an array of triangle indices reffered to by the leaf nodes. These indices point to the scene's triangle array.
	 *  \return Array of triangle indices.
	 */
	const Array<S32>&   getTriIndices (void) const	{ return m_triIndices; }

private:
	Scene*				m_scene;		//!< Source scene.
	Platform			m_platform;		//!< Platform settings.

	KDTreeNode*			m_root;			//!< Root node.
	Array<S32>			m_triIndices;	//!< Indices pointing to the scene's triangle array.
};


}
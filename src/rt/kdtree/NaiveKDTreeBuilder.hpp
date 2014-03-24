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

/**
 * \brief Naive k-d tree builder class
 * \details Uses either spatial median or object median to determine position of the split. Axis of the split is selected in round-robin fashion.
 */
class NaiveKDTreeBuilder
{
private:
	/**
	 * \brief Maximum depth of the k-d tree.
	 */
	enum
	{
		MaxDepth			= 18,
	};

	/**
	 * \brief Internal structure holding triangle index and it's bounding box.
	 */
	struct Reference
    {
        S32                 triIdx; //!< Index of the referenced triangle.
        AABB                bounds;	//!< Triangle's bounding box.

		/**
		 * \brief Constructor.
		 */
        Reference(void) : triIdx(-1) {}
    };

	/**
	 * \brief Internal structure holding information about processed k-d tree node.
	 */
	struct NodeSpec
    {
        S32                 numRef;	//!< Number of triangles this node refferences.
		AABB                bounds;	//!< Node's bounding box.

		/**
		 * \brief Constructor.
		 */
        NodeSpec(void) : numRef(0) {}
    };

	/**
	 * \brief Internal structure holding information about node split.
	 */
	struct Split
    {
        S32                 dim; //!< Split axis.
        F32                 pos; //!< Split position.

		/**
		 * \brief Constructor.
		 */
        Split(void) :  dim(0), pos(0.0f) {}
    };

public:
	/**
	 * \brief Constructor.
	 * \param[in] kdtree K-d tree being built.
	 * \param[in] params Build parameters.
	 */
								NaiveKDTreeBuilder		(KDTree& kdtree, const KDTree::BuildParams& params);

	/**
	 * \brief Destructor.
	 */
								~NaiveKDTreeBuilder		(void) {}

	/**
	 * \brief Builds k-d tree.
	 * \return Root node of the built tree.
	 */
	KDTreeNode*					run						(void);

	/**
	 * \brief Returns number of duplicated references.
	 * \return Number of duplicated references.
	 */
	S32							getNumDuplicates		(void)	{ return m_numDuplicates; }

private:

	/**
	 * \brief Sort compare function. Compares references according to their bounding box center in m_sortDim dimension. See Sort.h in framework/base.
	 */
	static bool					sortCompare				(void* data, int idxA, int idxB);

	/**
	 * \brief Sort swap function. See Sort.h in framework/base.
	 */
    static void					sortSwap				(void* data, int idxA, int idxB);

	/**
	 * \brief Builds either an inner node or a leaf node, based on the node's specifications and level.
	 * \param[in] spec Specifications of the processed node.
	 * \param[in] level Node's level in the tree.
	 * \param[in] progressStart Percentage of the processed references before processing the current node.
	 * \param[in] progressEnd Percentage of the processed references after processing the current node.
	 * \return Built node.
	 */
	KDTreeNode*					buildNode				(NodeSpec spec, int level, F32 progressStart, F32 progressEnd);

	/** 
	 * \brief Builds leaf node.
	 * \param[in] spec Specifications of the processed node.
	 * \return Built leaf node.
	 */
	KDTreeNode*					createLeaf				(const NodeSpec& spec);

	/**
	 * \brief Finds either object or spatial median split.
	 * \param[in] spec Specifications of the processed node.
	 * \param[in] axis Split axis.
	 * \return Found split.
	 */
	Split						findSplit			(const NodeSpec& spec, S32 axis);

	/**
	 * \brief Splits the node according to a given split.
	 * \param[out] left Left part of the split node.
	 * \param[out] right Right part of the split node.
	 * \param[in] spec Specifications of the processed node.
	 * \param[in] split Split parameters.
	 */
	void						performSplit		(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const Split& split);

private:
								NaiveKDTreeBuilder		(const NaiveKDTreeBuilder&); // forbidden
	NaiveKDTreeBuilder&			operator=				(const NaiveKDTreeBuilder&); // forbidden

private:
	KDTree&						m_kdtree;				//!< K-d tree being built.
	const Platform&				m_platform;				//!< Platform settings.
	const KDTree::BuildParams&	m_params;				//!< Build parameters.
	Array<Reference>			m_refStack;				//!< Stack of references.
	S32							m_sortDim;				//!< Sort dimension. Used in sortCompare.

	Timer						m_progressTimer;		//!< Timer. Measures the time from the last print of the information to the console.
	S32							m_numDuplicates;		//!< Number of the duplicated references.
};

}
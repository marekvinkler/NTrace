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

/**
 * \brief Fast k-d tree builder class.
 * \details Implementation of k-d tree described in [Wald and Havran 2006]
 */
class FastKDTreeBuilder
{
private:

	/**
	 * \brief Event type.
	 */
	enum EventType
	{
		End,
		Planar,
		Start
	};

	/**
	 * \brief Indicates whether flat triangles lying in the split plane are in the left split or in the right split.
	 */
	enum SahSide
	{
		Left,
		Right
	};

	/**
	 * \brief Indicates whether a triangle is on the left or on the right of the split plane, or if it straddles the split plane.
	 */
	enum SplitSide
	{
		LeftOnly,
		RightOnly,
		Both
	};

	/**
	 * \brief Structure holding event data.
	 */
	struct Event
	{
		S32					triIdx;	//!< Index of associated triangle.

		F32					pos;	//!< Position of the event.
		S32					dim;	//!< Dimension of the event.
		EventType			type;	//!< Type of the event.

		/**
		 * \brief Constructor.
		 */
		Event() : triIdx(-1), pos(0.f), dim(-1) {}
	};

	/**
	 * \brief Structure holding information about processed k-d tree node.
	 */
	struct NodeSpec
    {
        S32                 numEv;	//!< Number of events that the node has saved on the stack.
		AABB                bounds;	//!< Bounding box of the node.
		S32					numTri; //!< Number of triangles this node refferences.

		/**
		 * \brief Constructor.
		 */
        NodeSpec(void) : numEv(0), numTri(0) {}
    };

	/**
	 * \brief Structure holding split data.
	 */
	struct Split
	{		
		F32					price;	//!< SAH price of the split.
		S32					dim;	//!< Dimension of the split.
		F32					pos;	//!< Position of the split.
		SahSide				side;	//!< Side of flat planes lying in the split plane.

		/**
		 * \brief Constructor.
		 */
		Split() : price(FW_F32_MAX), dim(-1), pos(0.f) {}
	};

	/**
	 * \brief Structure holding additional information about triangles.
	 */
	struct TriData
	{
		//AABB				bounds;		//!< Bounding box of the triangle.
		SplitSide			side;		//!< Indicates whether a triangle is on the left or on the right of the split plane, or if it straddles the split plane.

		/**
		 * \brief Constructor.
		 */
		TriData() : side(Both) {}
	};

public:
	/**
	 * \brief Constructor.
	 * \param[in] kdtree K-d tree being constructed.
	 * \param[in] params Build parameters.
	 */

							FastKDTreeBuilder		(KDTree& kdtree, const KDTree::BuildParams& params);
	/**
	 * \brief Destructor.
	 */
							~FastKDTreeBuilder		(void)	{}

	/**
	 * \brief Builds k-d tree.
	 * \return Root node of the built tree.
	 */
	KDTreeNode*				run						(void);

	/**
	 * \brief Returns total number of duplicated triangle references.
	 * \return Number of duplicated triangle references 
	 */
	S32						getNumDuplicates		(void)	{ return m_numDuplicates; }

private:
	/**
	 * \brief Builds either an inner node or a leaf node, based on the node's specifications, level and number of forced splits.
	 * \param[in] spec Specifications of the processed node.
	 * \param[in] level Node's level in the tree.
	 * \param[in] forcedSplits Number of forced splits in the current subtree. Forced split is a split which has a bad price but is performed nonetheless.
	 * \param[in] progressStart Percentage of the processed references before processing the current node.
	 * \param[in] progressEnd Percentage of the processed references after processing the current node.
	 */
	KDTreeNode*				buildNode				(const NodeSpec& spec, int level, int forcedSplits, F32 progressStart, F32 progressEnd);

	/** 
	 * \brief Builds leaf node.
	 * \param[in] spec Specifications of the processed node.
	 * \return Built leaf node.
	 */
	KDTreeNode*				createLeaf				(const NodeSpec& spec);

	/**
	 * \brief Finds best possible split given the node's specifications.
	 * \param[in] spec Node's specifications.
	 * \return Returns Found split.
	 */
	Split					findSplit				(const NodeSpec& spec) const;

	/**
	 * \brief Splits a node and creates a left and right child node.
	 * \param[out] left Left child node's specifications.
	 * \param[out] right Right child node's specifications.
	 * \param[in] spec Specifications the processed node.
	 *\ param[in] split Split according to which the node will be split.
	 */
	void					performSplit			(NodeSpec& left, NodeSpec& right, const NodeSpec& spec, const Split& split);

	/**
	 * \brief Sort compare function. Uses eventCompare as a comparator function. See Sort.h in framework/base.
	 */
	static bool				eventSortCompare		(void* data, int idxA, int idxB);

	/**
	 * \brief Sort swap function. See Sort.h in framework/base.
	 */
	static void				eventSortSwap			(void* data, int idxA, int idxB);

	/**
	 * \brief Compares two events, returns true if event A should go before event B.
	 * \param[in] eventA Event A.
	 * \param[in] eventB Event B.
	 * \retrun True if the two events are in the correct order, false otherwise.
	 */
	static bool				eventCompare			(const Event& eventA, const Event& eventB);

	/**
	 * \brief Generates tightest possible bounding boxes for two parts of a split triangle.
	 * \param[out] left Bounding box of the left part.
	 * \param[out] right Bounding box of the right part.
	 * \param[in] triIdx Index of a the triangle being split.
	 * \param[in] split Split that cuts the triangle.
	 */
	void					splitBounds				(AABB& left, AABB& right, S32 triIdx, const Split& split) const; 

	/**
	 * \brief Merges two arrays of events and inserts them intto the event stack.
	 * \param[out] Index of the event stack's top. Events will be placed into the stack from this position on and the index will be incremented accordingly.
	 * \param[in] a One of the two arrays to be merged.
	 * \param[in] b One of the two arrays to be merged.
	 */
	void					mergeEvents				(S32& stackTop, const Array<Event>& a, const Array<Event>& b);

	/**
	 * \brief Calculates SAH price of a split.
	 * \param[in] split Split whose price will be calculated.
	 * \param[in] bounds Bounding box which will be split.
	 * \param[in] nl Number of triangles on the left side of the split plane.
	 * \param[in] nr Number to triangles on the right side of the split plane.
	 * \return SAH price of the split.
	 */
	F32						sahPrice				(const Split& split, const AABB& bounds, S32 nl, S32 nr) const;

	void msort(Array<Event>& data, S32 l, S32 h);
	void mmerge(Array<Event>& data, S32 l, S32 middle, S32 h);



private:
	KDTree&						m_kdtree;			//!< K-d tree being built.
	const Platform&				m_platform;			//!< Platform settings.
	const KDTree::BuildParams&	m_params;			//!< Build parameters.
	const S32					m_maxDepth;			//!< Maximum depth of the tree.
	const S32					m_maxFailSplits;	//!< Maximum number of forced splits.

	Array<Event>				m_evStack;			//!< Event stack.
	Array<int>					m_triStack;			//!< Triangle index stack.
	Array<TriData>				m_triData;			//!< Additional triangle data.

	Timer						m_progressTimer;	//!< Timer.
	S32							m_numDuplicates;	//!< Total number of duplicated references.

	Timer						m_measureTimer;

	Array<Event>				m_eventsLO;			//!< Left only events.
	Array<Event>				m_eventsRO;			//!< Right only events.
	Array<Event>				m_eventsBL;			//!< Events for straddling triangles, left part.
	Array<Event>				m_eventsBR;			//!< Events for straddling triangles, right part.
	Array<S32>					m_leftTriIdx;		//!< Triangle indices for the left child.
	Array<S32>					m_rightTriIdx;		//!< Triangle indices for the right child.

	Array<Event>				m_mergeSortBuffer;	//!< Merge sort buffer.
};	

}
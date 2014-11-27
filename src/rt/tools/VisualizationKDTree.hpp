/* 
 *  Copyright (c) 2013, FI MUNI CZ
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
 *
 *  Authors:
 *  Radek Stibora
 *
 */

/*! \file
 *  \brief Definitions for the BVH visualization framework.
 */

#pragma once
#include "Visualization.hpp"


namespace FW
{
//------------------------------------------------------------------------

/**
 *  \brief Class for the BVH visualization.
 *  \details In the standard visualization current node's bounding box, its sibling's bounding box and its childrens' bounding boxes are visualized.
 */
class VisualizationKDTree : public Visualization
{
public:

	struct SplitInfo
	{
		F32		pos;
		S32		dim;

		SplitInfo(void) : pos(0), dim(-1) {}
		FW::String	getPos(void)		{ return String(pos); }
		FW::String	getAxisName(void)	{ if (dim == 0) return "X"; else if (dim == 1) return "Y"; else if (dim == 2) return "Z"; else return "NA"; }
	};

	/*!
	 *  \brief Constructor.
	 *  \param[in] bvh			CudaBVH to visualize.
	 *  \param[in] rays			Rays to visualize, pass NULL if no rays should be visualized.
	 *  \param[in] visibility	Array of triangle visibility flags.
	 */
    explicit    VisualizationKDTree    (CudaKDTree* kdtree, Scene* scene, const RayBuffer* rays = NULL, Buffer* visibility = NULL);
	/*!
	 *  \brief Destructor.
	 */
    ~VisualizationKDTree   (void);

	/*!
	 *  \brief Handles visualization events - key commands influencing the output.
	 *  \param[in] ev		Event to process.
	 *  \return				Returns true if the event has been processed.
	 */
	virtual bool handleEvent        (const Window::Event& ev); // must be before the listener that queries things

	/*!
	 *  \brief Sets the node to be visualized to be the parent of the currently visualized node.
	 */
    void        moveToParent        ();
	/*!
	 *  \brief Sets the node to be visualized to be the sibling of the currently visualized node.
	 */
	void        moveToSibling       ();
	/*!
	 *  \brief Sets the node to be visualized to be the left child of the currently visualized node.
	 */
	void        moveToLeft          ();
	/*!
	 *  \brief Sets the node to be visualized to be the right child of the currently visualized node.
	 */
	void        moveToRight         ();
	/*!
	 *  \brief Sets the node to be visualized to be the predecesor of the currently visualized node on the set path.
	 *  \details Unlike moveToParent() does not change the current path from root to the lowest discovered node. Only changes which node to visualize on this path.
	 */
	void        moveUp              ();
	/*!
	 *  \brief Sets the node to be visualized to be the succesor of the currently visualized node on the set path.
	 *  \details Unlike moveToLeftChild() or moveToRightChild does not change the current path from root to the lowest discovered node. Only changes which node to visualize on this path.
	 */
	void        moveDown            ();

	/*!
	 *  \brief The method used to draw the current state of visualization to the OpenGL context.
	 *  \param[in] gl		OpenGL context to draw into.
	 *  \param[in] camera	The camera for the current frame.
	 */
	void        draw                (GLContext* gl, CameraControls& camera);

private:
	/*!
	 *  \bried Internal structure for holding visualization information.
	 */
	struct NodeData
    {
		S32                 addr; //!< Address in the CudaBVH linear array structure.
		//S32					dim;
		//F32					pos;
        AABB                box; //!< The boxes bounding box stored as it's min and max value.

		/*!
		 *  \brief Constructor
		 */
        NodeData(void) : addr(0) {}
    };

	/*!
	 *  \brief Prepares the node at address idx for visualization.
	 *  \param[in] idx			Address of the node to make a current node.
	 */
	void        getFromIndex        (S32 idx);
	/*!
	 *  \brief Enlarges the current node's and it's siblings's node bounding box a little bit to supress z-fighting.
	 */
	void        growParentBox       ();
	/*!
	 *  \brief Draws the 4 nodes: current, sibling, left and right child or the VisualizationKDTree::m_boxes buffer.
	 *  \param[in] gl			The OpenGL context to draw into.
	 *  \param[in] onlyChildren	States whether to draw all the 4 boxes or just the 2 child boxes.
	 */
	void        drawNodes           (GLContext* gl, bool onlyChildren);
	/*!
	 *  \brief Draws one box into the OpenGL contex.
	 *  \param[in] gl			The OpenGL context to draw into.
	 *  \param[in] node			The node to draw.
	 *  \param[in] abgr			Color to draw the node.
	 */
	void        drawBox             (GLContext* gl, const NodeData &node, U32 abgr);
	/*!
	 *  \brief Draws all the rays in the VisualizationKDTree::m_rays buffer into the OpenGL contex.
	 *  \param[in] gl			The OpenGL context to draw into.
	 *  \param[in] abgr			Color to draw the rays.
	 */
	void        drawRays            (GLContext* gl, U32 abgr);
	/*!
	 *  \brief Draws all the primitives in the VisualizationKDTree::m_tris buffer into the OpenGL contex.
	 *  \param[in] gl			The OpenGL context to draw into.
	 */
	void        drawPrimitives      (GLContext* gl);
	/*!
	 *  \brief Draws the the path information into the OpenGL contex.
	 */
	void        drawPathInfo        (GLContext* gl);
	/*!
	 *  \brief Sets the color mapping for the drawn boxes.
	 */
	void        setColorMapping     ();
	/*!
	 *  \brief Traverses the entire BVH tree and fills the VisualizationKDTree::m_boxes and VisualizationKDTree::m_tris buffers.
	 *  \param[in] node			The root of the subtree to process.
	 */
	void        prepareTreeData     (NodeData node);
	/*!
	 *  \brief Converts a min,max representation of a box to a series of faces(quads) representation and adds it to the buffer.
	 *  \param[in] box			The box to convert.
	 *  \param[in] buffer		Array to add the new representation into.
	 */
	void        addBoxQuads				(const AABB &box, Array<Vec4f> &buffer);

	
	void		splitNode			(const NodeData& currNode, S32& leftAdd, S32& rightAdd, AABB& leftBox, AABB& rightBox, SplitInfo& split);

private:
	// Global and path data
    CudaKDTree*         m_kdtree;			//!< BVH to visualize.
    Array<NodeData>     m_nodeStack;	//!< Stack of the node addresses, holds the path from the BVH root to the current node.
	// Currently visible node data
	NodeData            m_node;			//!< Current node.
	NodeData            m_sibling;		//!< Sibling of the current node.
	NodeData            m_left;			//!< Left child of the current node.
	NodeData            m_right;		//!< Right child of the current node.
	SplitInfo			m_nodeSplit;
};

//------------------------------------------------------------------------
}

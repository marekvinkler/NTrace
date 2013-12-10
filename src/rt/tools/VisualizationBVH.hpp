/*
 *  Copyright 2009-2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file
 *  \brief Definitions for the BVH visualization framework.
 */

#pragma once
#include "gui/Window.hpp"
#include "cuda/CudaBVH.hpp"
#include "base/Array.hpp"
#include "3d/CameraControls.hpp"

namespace FW
{
//------------------------------------------------------------------------

/*!
 *  \brief Class for the BVH visualization.
 *  \details In the standard visualization current node's bounding box, its sibling's bounding box and its childrens' bounding boxes are visualized.
 */
class VisualizationBVH : public Window::Listener
{
public:
	/*!
	 *  \brief Constructor.
	 *  \param[in] bvh			CudaBVH to visualize.
	 *  \param[in] emptyBoxes	Array of empty boxes to be visualized.
	 *  \param[in] rays			Rays to visualize, pass NULL if no rays should be visualized.
	 *  \param[in] visibility	Array of triangle visibility flags.
	 */
    explicit    VisualizationBVH    (CudaBVH* bvh, const Array<AABB> &emptyBoxes, const RayBuffer* rays = NULL, Buffer* visibility = NULL);
	/*!
	 *  \brief Destructor.
	 */
                ~VisualizationBVH   (void);

	/*!
	 *  \brief Handles visualization events - key commands influencing the output.
	 *  \param[in] ev		Event to process.
	 *  \return				Returns true if the event has been processed.
	 */
	virtual bool handleEvent        (const Window::Event& ev); // must be before the listener that queries things

	/*!
	 *  \brief Return whether the visualization renders its output.
	 *  \return				Returns true if the visualization is rendering, false otherwise.
	 */
	bool        isVisible           () const                  { return m_visible; }
	/*!
	 *  \brief Sets whether the visualization should be rendered or not.
	 *  \param[in] visible	The visibility value.	
	 */
	void        setVisible          (bool visible)            { m_visible = visible; }

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
	 *  \brief Draws the 4 nodes: current, sibling, left and right child or the VisualizationBVH::m_boxes buffer.
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
	 *  \brief Draws all the rays in the VisualizationBVH::m_rays buffer into the OpenGL contex.
	 *  \param[in] gl			The OpenGL context to draw into.
	 *  \param[in] abgr			Color to draw the rays.
	 */
	void        drawRays            (GLContext* gl, U32 abgr);
	/*!
	 *  \brief Draws all the primitives in the VisualizationBVH::m_tris buffer into the OpenGL contex.
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
	 *  \brief Traverses the entire BVH tree and fills the VisualizationBVH::m_boxes and VisualizationBVH::m_tris buffers.
	 *  \param[in] node			The root of the subtree to process.
	 */
	void        prepareTreeData     (S32 node);
	/*!
	 *  \brief Converts a min,max representation of a box to a series of faces(quads) representation and adds it to the buffer.
	 *  \param[in] box			The box to convert.
	 *  \param[in] buffer		Array to add the new representation into.
	 */
	void        addBoxQuads				(const AABB &box, Array<Vec4f> &buffer);

private:
	// Global and path data
    CudaBVH*            m_bvh;			//!< BVH to visualize.
    Array<S32>          m_nodeStack;	//!< Stack of the node addresses, holds the path from the BVH root to the current node.
	Array<String>       m_splitPath;	//!< Text representation of the VisualizationBVH::m_nodeStack path.
	S32					m_osahSplits[3];//!< Counters of the number of OSAH splits in the subtree under the set node in the x, y and z dimensions.
	S32                 m_currentDepth; //!< Current node's depth information.
	Array<S32>          m_visibility;   //!< Visibility of individual triangles
	// Currently visible node data
	NodeData            m_node;			//!< Current node.
	U32                 m_nodeColor;	//!< Color of the current node.
	SplitInfo           m_nodeSplit;	//!< Information about the split in the current node.
	NodeData            m_sibling;		//!< Sibling of the current node.
	U32                 m_siblingColor; //!< Color of the sibling of the current node.
	NodeData            m_left;			//!< Left child of the current node.
	U32                 m_leftColor;	//!< Color of the left child of the current node.
	U32                 m_leftPrims;	//!< Number of primitives in the left child of the current node.
	NodeData            m_right;		//!< Right child of the current node.
	U32                 m_rightColor;	//!< Color of the right child of the current node.
	U32                 m_rightPrims;	//!< Number of primitives in the right child of the current node.
	// Buffered "special" data
	Buffer              m_rays;         //!< Buffer holding some rays as line segments.
	U32                 m_rayColor;		//!< Color of the ray line indices.
	Buffer              m_boxes;        //!< Buffer holding selected boxes as quad primitives.
	Buffer              m_emptyBoxes;   //!< Buffer holding empty boxes as quad primitives.
	Buffer              m_emptyColors;  //!< Buffer holding colors of empty boxes as quad primitives.
	Buffer              m_emptyLineColors;  //!< Buffer holding line colors of empty boxes as quad primitives.
	Buffer              m_invisTris;    //!< Buffer holding visible selected triangles.
	Buffer              m_visTris;      //!< Buffer holding invisible selected triangles.
	// Visualization setting
	bool                m_visible;      //!< Flag whether to show the BVH visualization.
	bool                m_showRays;		//!< Flag whether to show the ray segments.
	bool                m_splitColors;  //!< Flag whether to map left/right children colors based on the split type.
	bool                m_showChildren; //!< Flag whether to show children of the current node.
	bool                m_showAllOSAH;  //!< Flag whether to show all OSAH split nodes.
	bool                m_showEmpty;    //!< Flag whether to show empty nodes.
	bool                m_showCurrTris; //!< Flag whether to show triangles of the current node.
};

//------------------------------------------------------------------------
}

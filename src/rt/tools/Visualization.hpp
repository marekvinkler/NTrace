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
 *  Vilem Otte <vilem.otte@post.cz>
 *
 */

/*! \file
 *  \brief Visualisation framework abstract class above BVH/KDTree visualisations.
 */

#pragma once
#include "gui/Window.hpp"
#include "base/Array.hpp"
#include "3d/CameraControls.hpp"
#include "cuda/CudaBVH.hpp"
#include "cuda/CudaKDTree.hpp"

namespace FW
{
//------------------------------------------------------------------------

/*!
 *  \brief Class for the visualization.
 */
class Visualization : public Window::Listener
{
public:
	                   Visualization   (Scene* scene);
	virtual           ~Visualization   ();

	/*!
	 *  \brief The method used to draw the current state of visualization to the OpenGL context.
	 *  \param[in] gl		OpenGL context to draw into.
	 *  \param[in] camera	The camera for the current frame.
	 */
	virtual void       draw            (GLContext* gl, CameraControls& camera) = 0;

	/*!
	 *  \brief Return whether the visualization renders its output.
	 *  \return				Returns true if the visualization is rendering, false otherwise.
	 */
	bool               isVisible       () const                  { return m_visible; }

	/*!
	 *  \brief Sets whether the visualization should be rendered or not.
	 *  \param[in] visible	The visibility value.	
	 */
	void               setVisible      (bool visible)            { m_visible = visible; }

protected:	
	Array<String>       m_splitPath;	//!< Text representation of the VisualizationBVH::m_nodeStack path.
	S32					m_osahSplits[3];//!< Counters of the number of OSAH splits in the subtree under the set node in the x, y and z dimensions.
	S32                 m_currentDepth; //!< Current node's depth information.
	Array<S32>          m_visibility;   //!< Visibility of individual triangles
	// Currently visible node data
	//NodeData            m_node;			//!< Current node.
	U32                 m_nodeColor;	//!< Color of the current node.
	//NodeData            m_sibling;		//!< Sibling of the current node.
	U32                 m_siblingColor; //!< Color of the sibling of the current node.
	//NodeData            m_left;			//!< Left child of the current node.
	U32                 m_leftColor;	//!< Color of the left child of the current node.
	U32                 m_leftPrims;	//!< Number of primitives in the left child of the current node.
	//NodeData            m_right;		//!< Right child of the current node.
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
	bool                m_showCurrTris; //!< Flag whether to show triangles of the current node.};
	Scene*				m_scene;
};

}
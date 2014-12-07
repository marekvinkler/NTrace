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

#include "tools/VisualizationKDTree.hpp"

using namespace FW;

#define NO_NODE FW_S32_MIN
#define ROOT_ADDR 0

#define COLOR_NODE 0x14FFFFFF
#define COLOR_SIBLING 0x14000000
#define COLOR_LEFT 0x330000FF
#define COLOR_RIGHT 0x3300FF00
#define COLOR_RAY 0xFFFFFFFF
#define COLOR_TRI_INVIS 0xFF000000
#define COLOR_TRI_VIS 0xFFFFFFFF

#define COLOR_LEFT_SAH 0x330000FF
#define COLOR_RIGHT_SAH 0x3300FF00
#define COLOR_LEFT_SVBH 0x33FF00FF
#define COLOR_RIGHT_SVBH 0x33FFFF00
#define COLOR_LEFT_OSAH 0x3300FFFF
#define COLOR_RIGHT_OSAH 0x33FF0000

//------------------------------------------------------------------------

VisualizationKDTree::VisualizationKDTree(CudaKDTree* kdtree, Scene* scene, const RayBuffer* rays, Buffer* visibility)
:	Visualization(scene, rays, visibility),
	m_kdtree(kdtree)
{
	m_node.box = kdtree->getBBox(); // scene bounding box
	m_node.addr = 0;

	// Initialize m_node, m_sibling, m_left and m_right
	splitNode(m_node, m_left.addr, m_right.addr, m_left.box, m_right.box, m_nodeSplit);
	growParentBox();
	m_sibling.addr = NO_NODE;

	// Inititalize stacks
	m_nodeStack.add(m_node);
	m_splitPath.add(m_nodeSplit.getPos() + " " + m_nodeSplit.getAxisName() + ":");

	m_nodeColor = COLOR_NODE;
	m_siblingColor = COLOR_SIBLING;
	m_leftColor = COLOR_LEFT;
	m_rightColor = COLOR_RIGHT;
	m_rayColor = COLOR_RAY;
}

//------------------------------------------------------------------------

VisualizationKDTree::~VisualizationKDTree(void)
{
}

//------------------------------------------------------------------------

bool VisualizationKDTree::handleEvent(const Window::Event& ev)
{
	//FW_ASSERT(m_window == ev.window || ev.type == Window::EventType_AddListener);

	// Handle events.

	switch (ev.type)
	{
	/*case Window::EventType_AddListener:
		FW_ASSERT(!m_window);
		m_window = ev.window;
		repaint();
		return false;

	case Window::EventType_RemoveListener:
		repaint();
		m_window = NULL;
		return false;*/

	case Window::EventType_KeyDown:
		//if (ev.key == FW_KEY_V)                   setVisible(!m_visible);
		if(isVisible())
		{
			if (ev.key == FW_KEY_B)									{ m_splitColors = !m_splitColors; setColorMapping(); }
			if (ev.key == FW_KEY_J)									{ m_showChildren = !m_showChildren; }
			if (ev.key == FW_KEY_U && m_rays.getSize() > 0)			{ m_showRays = !m_showRays; }
			if (ev.key == FW_KEY_O)									{ m_showCurrTris = !m_showCurrTris; prepareTreeData(m_node); }
			if (ev.key == FW_KEY_L)									{ m_showAllOSAH = !m_showAllOSAH; prepareTreeData(m_node); }
			if (ev.key == FW_KEY_T)									{ moveToParent(); if(m_showCurrTris || m_showAllOSAH) prepareTreeData(m_node); }
			if (ev.key == FW_KEY_Y)									{ moveToSibling(); if(m_showCurrTris || m_showAllOSAH) prepareTreeData(m_node); }
			if (ev.key == FW_KEY_G)									{ moveToLeft(); if(m_showCurrTris || m_showAllOSAH) prepareTreeData(m_node); }
			if (ev.key == FW_KEY_H)									{ moveToRight(); if(m_showCurrTris || m_showAllOSAH) prepareTreeData(m_node); }
			if (ev.key == FW_KEY_I)									{ moveUp(); if(m_showCurrTris || m_showAllOSAH) prepareTreeData(m_node); }
			if (ev.key == FW_KEY_K)									{ moveDown(); if(m_showCurrTris || m_showAllOSAH) prepareTreeData(m_node); }
		}
		break;

	default:
		break;
	}

	return false;
}

//------------------------------------------------------------------------

void VisualizationKDTree::moveToParent()
{	
	moveUp();

	// Update path
	m_nodeStack[m_currentDepth] = m_node;
	if(!m_splitPath[m_currentDepth].endsWith(":"))
		m_splitPath[m_currentDepth] = m_splitPath[m_currentDepth].substring(0, m_splitPath[m_currentDepth].getLength()-1);
	m_nodeStack.resize(m_currentDepth+1);
	m_splitPath.resize(m_currentDepth+1);
}

//------------------------------------------------------------------------

void VisualizationKDTree::moveToSibling()
{
	if(m_sibling.addr == NO_NODE)
		return;

	NodeData temp;
	temp = m_node;
	m_node = m_sibling;
	m_sibling = temp;

	if(m_node.addr >= 0)//if(!m_kdtree->isLeaf(m_node.addr))
	{
		//m_bvh->getNode(m_node.addr, &m_nodeSplit, m_left.box, m_right.box, m_left.addr, m_right.addr);
		splitNode(m_node, m_left.addr, m_right.addr, m_left.box, m_right.box, m_nodeSplit);
	}
	// Set color mapping for visible nodes
	setColorMapping();

	// Update path
	m_nodeStack[m_currentDepth] = m_node;
	m_splitPath[m_currentDepth] = m_nodeSplit.getPos() + " " + m_nodeSplit.getAxisName() + ":";
	m_nodeStack.resize(m_currentDepth+1);
	m_splitPath.resize(m_currentDepth+1);
	// Flip child identifier from the previous split
	if(m_splitPath[m_currentDepth-1][m_splitPath[m_currentDepth-1].indexOf(':')+1] == 'L')
	{
		m_splitPath[m_currentDepth-1] = m_splitPath[m_currentDepth-1].substring(0, m_splitPath[m_currentDepth-1].getLength()-1);
		m_splitPath[m_currentDepth-1] = m_splitPath[m_currentDepth-1].append('R');
	}
	else
	{
		m_splitPath[m_currentDepth-1] = m_splitPath[m_currentDepth-1].substring(0, m_splitPath[m_currentDepth-1].getLength()-1);
		m_splitPath[m_currentDepth-1] = m_splitPath[m_currentDepth-1].append('L');
	}
}

//------------------------------------------------------------------------

void VisualizationKDTree::moveToLeft()
{
	if(m_left.addr < 0)
		return;

	m_node = m_left;
	m_sibling = m_right;

	if(m_node.addr >= 0)
	{
		splitNode(m_node, m_left.addr, m_right.addr, m_left.box, m_right.box, m_nodeSplit);
	}
	else
	{
		m_left.addr = NO_NODE;
		m_right.addr = NO_NODE;
	}

	growParentBox();
	// Set color mapping for visible nodes
	setColorMapping();

	// Update path
	m_splitPath[m_currentDepth] = m_splitPath[m_currentDepth].append('L');
	m_nodeStack.resize(m_currentDepth+1);
	m_splitPath.resize(m_currentDepth+1);
	// Add to the path
	m_nodeStack.add(m_node);
	m_splitPath.add(m_nodeSplit.getPos() + " " + m_nodeSplit.getAxisName() + ":");
	m_currentDepth++;
}

//------------------------------------------------------------------------

void VisualizationKDTree::moveToRight()
{
	if(m_right.addr < 0)
		return;

	m_node = m_right;
	m_sibling = m_left;

	if(m_node.addr >= 0)
	{
		splitNode(m_node, m_left.addr, m_right.addr, m_left.box, m_right.box, m_nodeSplit);
	}
	else
	{
		m_left.addr = NO_NODE;
		m_right.addr = NO_NODE;
	}

	growParentBox();
	// Set color mapping for visible nodes
	setColorMapping();

	// Update path
	m_splitPath[m_currentDepth] = m_splitPath[m_currentDepth].append('R');
	m_nodeStack.resize(m_currentDepth+1);
	m_splitPath.resize(m_currentDepth+1);
	// Add to the path
	m_nodeStack.add(m_node);
	m_splitPath.add(m_nodeSplit.getPos() + " " + m_nodeSplit.getAxisName() + ":");
	m_currentDepth++;
}

//------------------------------------------------------------------------

void VisualizationKDTree::moveUp()
{
	if(m_currentDepth > 0)
	{
		m_currentDepth--;
		getFromIndex(m_currentDepth);
	}
}

//------------------------------------------------------------------------

void VisualizationKDTree::moveDown()
{
	if(m_currentDepth < m_nodeStack.getSize()-1)
	{
		m_currentDepth++;
		getFromIndex(m_currentDepth);
	}
}

//------------------------------------------------------------------------

void VisualizationKDTree::draw(GLContext* gl, CameraControls& camera)
{
	FW_ASSERT(gl);

	if(!isVisible())
		return;

	Mat4f oldXform = gl->setVGXform(gl->xformFitToView(-1.0f, 2.0f) * camera.getWorldToClip());
    glPushAttrib(GL_ENABLE_BIT);
    //glEnable(GL_DEPTH_TEST);
	//glDepthFunc(GL_LESS);
	//glEnable(GL_CULL_FACE);
	//glFrontFace(GL_CW);

	// Draw primary rays
	drawRays(gl, m_rayColor);

	// Obsolete visualization, boxes hide each other
	// Draw back faces of bounding boxes - for correct depth
	/*glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
	//glCullFace(GL_FRONT);
	drawNodes(gl, true);
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);

	// Draw front faces of bounding boxes - for blending
	glDisable(GL_DEPTH_TEST);
	glDepthFunc(GL_LEQUAL);
	glCullFace(GL_BACK);
	drawNodes(gl, false);*/

	// New Visualization, all rays are visible
	// Render boxes and primitives without culling so everything is visible
	//glDisable(GL_DEPTH_TEST);
	//glDisable(GL_CULL_FACE);

	// Draw boxes
	drawNodes(gl, false);

	// Draw primitives
	drawPrimitives(gl);

	//glDisable(GL_DEPTH_TEST);
	glPopAttrib();
	gl->setVGXform(oldXform);

	// Draw path information
	drawPathInfo(gl);
}

//------------------------------------------------------------------------

void VisualizationKDTree::getFromIndex(S32 idx)
{
	if(idx < 0 || idx > m_nodeStack.getSize()-1)
		return;

	if(m_nodeStack[idx].addr >= 0)
	{
		splitNode(m_nodeStack[idx], m_left.addr, m_right.addr, m_left.box, m_right.box, m_nodeSplit);
	}
	else
	{
		m_left.addr = NO_NODE;
		m_right.addr = NO_NODE;
	}

	if(idx > 0) // Node is not the root, we can find its sibling
	{
		NodeData parent = m_nodeStack[idx-1];
		NodeData left, right;

		splitNode(parent, left.addr, right.addr, left.box, right.box, m_nodeSplit);

		if(left.addr == m_nodeStack[idx].addr) // Left child is the current node
		{
			m_node = left;
			m_sibling = right;
		}
		else // Right child is the current node
		{
			m_node = right;
			m_sibling = left;
		}
	}
	else // Node is the root, we need to compute the box and disable sibling
	{
		m_node.addr = m_nodeStack[idx].addr;
		m_node.box = m_left.box + m_right.box;
		m_sibling.addr = NO_NODE;
	}

	growParentBox();
	// Set color mapping for visible nodes
	setColorMapping();
}

//------------------------------------------------------------------------

void VisualizationKDTree::growParentBox()
{
	float incrTop = (m_node.box.max() - m_node.box.min()).length() / 50.0f;
	//float incrBottom = incrTop / 2.0f;
	m_node.box.grow(m_node.box.min()-incrTop/*-0.01f*/); // Grow the parent box to avoid z-fighting
	m_node.box.grow(m_node.box.max()+incrTop/*+0.01f*/); // Grow the parent box to avoid z-fighting

	if(m_sibling.addr != NO_NODE) // Root node has no sibling
	{
		m_sibling.box.grow(m_sibling.box.min()-incrTop/*+0.01f*/); // Grow the sibling box to avoid z-fighting
		m_sibling.box.grow(m_sibling.box.max()+incrTop/*-0.01f*/); // Grow the sibling box to avoid z-fighting
	}
}

//------------------------------------------------------------------------

void VisualizationKDTree::drawNodes(GLContext* gl, bool onlyChildren)
{	
	// Draw children of the current node
	if(m_showChildren)
	{
		drawBox(gl, m_left, m_leftColor);
		drawBox(gl, m_right, m_rightColor);
	}

	// Draw current node and its sibling
	if(!onlyChildren)
	{
		drawBox(gl, m_node, m_nodeColor);
		drawBox(gl, m_sibling, m_siblingColor);
	}

	// Draw visible child of the OSAH split
	if(m_showAllOSAH)
	{
		if(!onlyChildren)
		{
			//glDepthFunc(GL_ALWAYS);

			// Draw edges, use opaque ray color
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			if(m_showAllOSAH)
				gl->drawBuffer(m_boxes, GL_QUADS, 0, COLOR_RAY);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			//glDepthFunc(GL_LEQUAL);
		}
	}
}

//------------------------------------------------------------------------

void VisualizationKDTree::drawBox(GLContext* gl, const NodeData &node, U32 abgr)
{
	if(node.addr != NO_NODE)
	{
		// Draw filled faces
		gl->drawBox(node.box.min(), node.box.max(), abgr);
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		// Draw edges, use the same color but opaque
		gl->drawBox(node.box.min(), node.box.max(), abgr | 0xFF000000);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

//------------------------------------------------------------------------

void VisualizationKDTree::drawRays(GLContext* gl, U32 abgr)
{
	if(m_showRays)
	{
		glLineWidth(2.0f);
		gl->drawBuffer(m_rays, GL_LINES, 0, abgr);
		glLineWidth(1.0f);
	}
}

//------------------------------------------------------------------------

void VisualizationKDTree::drawPrimitives(GLContext* gl)
{
	if(m_showCurrTris)
	{
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
		gl->drawBuffer(m_invisTris, GL_TRIANGLES, 0, COLOR_TRI_INVIS);
		gl->drawBuffer(m_visTris, GL_TRIANGLES, 0, COLOR_TRI_VIS);
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
	}
}

//------------------------------------------------------------------------

void VisualizationKDTree::drawPathInfo(GLContext* gl)
{
	S32 fontSize = 16;
	Vec2f origin = Vec2f(8.0f, (F32)gl->getViewSize().y - 4.0f);
	Vec2f pos = origin;

	Mat4f oldXform = gl->setVGXform(gl->xformMatchPixels());
	gl->setFont("Arial", fontSize, GLContext::FontStyle_Bold);
	
	const size_t strLength = 200; // For unknown reason, for sprintf lower number is required
	char leftBox[strLength], rightBox[strLength];
	m_left.box.min().sprint(leftBox, strLength/2);
	strcat_s(leftBox, ", ");
	m_left.box.max().sprint(leftBox + strlen(leftBox), strLength/2-strlen(leftBox));
	m_right.box.min().sprint(rightBox, strLength/2);
	strcat_s(rightBox, ", ");
	m_right.box.max().sprint(rightBox + strlen(rightBox), strLength/2-strlen(rightBox));

	if(m_showCurrTris)
	{
		gl->drawLabel(sprintf("Left count: %u     Right count: %u",
		m_leftPrims, m_rightPrims), pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
		pos.y -= (F32)fontSize;
		gl->drawLabel(sprintf("Left area: %.2f (%s)",
		m_left.box.area(), leftBox), pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
		pos.y -= (F32)fontSize;
		gl->drawLabel(sprintf("Right area: %.2f (%s)",
		m_right.box.area(), rightBox), pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
	}
	else
	{
		gl->drawLabel(sprintf("Left area: %.2f (%s)",
		m_left.box.area(), leftBox), pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
		pos.y -= (F32)fontSize;
		gl->drawLabel(sprintf("Right area: %.2f (%s)",
		m_right.box.area(), rightBox), pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
	}
	pos.y -= (F32)fontSize;
	gl->drawLabel(sprintf("Current depth: %d     Path depth: %d",
		m_currentDepth, m_nodeStack.getSize()-1), pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
	pos.y -= (F32)fontSize;

	const float rightMargin = 100.0f;
	String header("Split path: ");
	gl->drawLabel(header, pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
	pos.x += gl->getStringSize(header).x;

	// Write path before the current node
	for(S32 i = 0; i < m_currentDepth; i++)
	{
		if(pos.x > (F32)gl->getViewSize().x - rightMargin)
		{
			pos.x = origin.x;
			pos.y -= (F32)fontSize;
		}

		String cur = m_splitPath[i] + "| ";
		gl->drawLabel(cur, pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
		pos.x += gl->getStringSize(cur).x;
	}

	// Write the current node
	if(pos.x > (F32)gl->getViewSize().x - rightMargin)
	{
		pos.x = origin.x;
		pos.y -= (F32)fontSize;
	}
	gl->drawLabel(m_splitPath[m_currentDepth], pos, Vec2f(0.0f, 1.0f), 0xFF0000FF);
	pos.x += gl->getStringSize(m_splitPath[m_currentDepth]).x;

	// Write path after the current node
	for(S32 i = m_currentDepth+1; i < m_splitPath.getSize(); i++)
	{
		if(pos.x > (F32)gl->getViewSize().x - rightMargin)
		{
			pos.x = origin.x;
			pos.y -= (F32)fontSize;
		}

		String cur = String("| ") + m_splitPath[i];
		gl->drawLabel(cur, pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
		pos.x += gl->getStringSize(cur).x;
	}

	gl->setVGXform(oldXform);
	gl->setDefaultFont();
}

//-----------------------------------------------------------------------

void VisualizationKDTree::setColorMapping()
{
	if(!m_splitColors)
	{
		m_leftColor = COLOR_LEFT;
		m_rightColor = COLOR_RIGHT;
		return;
	}
}

//-----------------------------------------------------------------------

void VisualizationKDTree::prepareTreeData(NodeData node)
{
	NodeData stack[100];
	int stackIndex = 1;
	Array<Vec4f> boxes;
	Array<Vec4f> verticesInvis;
	Array<Vec4f> verticesVis;
	Array<S32> indices;
	U32 prims = 0;
	S32 rightChild = 0;

	// Clear the osah split counts in current node
	memset(m_osahSplits, 0, sizeof(m_osahSplits));

	while(stackIndex > 0)
	{
		for(;;)
		{
			if(node.addr < 0)
			{
				//if(node.addr == KDTREE_EMPTYLEAF)
				//	break;

				if(m_showCurrTris) // Process triangle
				{
					indices.clear();

					S32 idx = ~node.addr;
					if (~idx != KDTREE_EMPTYLEAF)
					{
						while (((int*)m_kdtree->getTriIndexBuffer().getPtr())[idx] != KDTREE_EMPTYLEAF)
						{
							indices.add(((int*)m_kdtree->getTriIndexBuffer().getPtr())[idx]);
							idx++;
						}
					}
					//m_bvh->getTriangleIndices(node, indices);
					prims += indices.getSize();

					for(int i = 0; i < indices.getSize(); i++)
					{
						Array<Vec4f> *ptr = &verticesInvis;
						if(m_visibility[indices[i]])
							ptr = &verticesVis;

						const Vec3i& ind = ((const Vec3i*)m_scene->getTriVtxIndexBuffer().getPtr())[indices[i]];
						for(int j = 0; j < 3; j++)
						{
							const Vec3f& v = ((const Vec3f*)m_scene->getVtxPosBuffer().getPtr())[ind[j]];
							ptr->add(Vec4f(v, 1.0f));
						}
					}
				}

				break;
			}
			else
			{
				AABB child0, child1;
				S32 child0Addr, child1Addr;
				SplitInfo splitInfo;

				splitNode(node, child0Addr, child1Addr, child0, child1, m_nodeSplit);
				if(m_showCurrTris && rightChild == 0)
					rightChild = child1Addr;

				node.addr = child0Addr;
				stack[stackIndex++].addr = child1Addr;
			}
		}
		stackIndex--;
		node = stack[stackIndex];
		if(m_showCurrTris && node.addr == rightChild && stackIndex == 1)
		{
			m_leftPrims = prims;
			prims = 0;
		}
	}

	if(m_showAllOSAH)
	{
		m_boxes.resizeDiscard(boxes.getNumBytes());
		m_boxes.set(boxes.getPtr(), boxes.getNumBytes());
	}

	if(m_showCurrTris)
	{
		m_rightPrims = prims;
		m_invisTris.resizeDiscard(verticesInvis.getNumBytes());
		m_invisTris.set(verticesInvis.getPtr(), verticesInvis.getNumBytes());
		m_visTris.resizeDiscard(verticesVis.getNumBytes());
		m_visTris.set(verticesVis.getPtr(), verticesVis.getNumBytes());
	}
}

//------------------------------------------------------------------------

void VisualizationKDTree::splitNode(const NodeData& currNode, S32& leftAdd, S32& rightAdd, AABB& leftBox, AABB& rightBox, SplitInfo& split)
{
	leftAdd = ((Vec4i*)m_kdtree->getNodeBuffer().getPtr())[currNode.addr].x;
	rightAdd = ((Vec4i*)m_kdtree->getNodeBuffer().getPtr())[currNode.addr].y;

	float splitPos = *(float*)&(((Vec4i*)m_kdtree->getNodeBuffer().getPtr())[currNode.addr].z);
	unsigned int type = ((Vec4i*)m_kdtree->getNodeBuffer().getPtr())[currNode.addr].w & KDTREE_MASK;
	S32 dim = type >> KDTREE_DIMPOS;

	Vec3f leftCut = currNode.box.max();
	leftCut[dim] = splitPos;

	Vec3f rightCut = currNode.box.min();
	rightCut[dim] = splitPos;

	leftBox = AABB(currNode.box.min(), leftCut);
	rightBox = AABB(rightCut, currNode.box.max());

	split.dim = dim;
	split.pos = splitPos;
}
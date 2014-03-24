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

#include "tools/VisualizationBVH.hpp"
#include "ray/PixelTable.hpp"
#include "cuda\Renderer.hpp"

using namespace FW;

#define NO_NODE FW_S32_MIN
#define ROOT_ADDR 0

// Colors for the obsolete visualization
#define COLOR_NODE 0x33FFFFFF
#define COLOR_SIBLING 0x33000000
#define COLOR_LEFT 0x7F0000FF
#define COLOR_RIGHT 0x7F00FF00
#define COLOR_RAY 0xFFFFFFFF
#define COLOR_TRI_INVIS 0xFF000000
#define COLOR_TRI_VIS 0xFFFFFFFF

#define COLOR_LEFT_SAH 0x7F0000FF
#define COLOR_RIGHT_SAH 0x7F00FF00
#define COLOR_LEFT_SVBH 0x7FFF00FF
#define COLOR_RIGHT_SVBH 0x7FFFFF00
#define COLOR_LEFT_OSAH 0x7F00FFFF
#define COLOR_RIGHT_OSAH 0x7FFF0000*/

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

VisualizationBVH::VisualizationBVH(CudaKDTree* kdtree, Scene* scene, const Array<AABB> &emptyBoxes, const RayBuffer* rays, Buffer* visibility)
:	m_kdtree(kdtree),
	m_currentDepth(0),
	m_visible(false),
	m_showRays(false),
	m_showEmpty(false),
	m_splitColors(false),
	m_showChildren(true),
	m_showAllOSAH(false),
	m_showCurrTris(false),
	m_scene(scene)
{
	m_node.box = kdtree->getBBox(); // scene bounding box
	m_node.addr = 0;

	splitNode(m_node, m_left.addr, m_right.addr, m_left.box, m_right.box, m_nodeSplit);

	// Initialize m_node, m_sibling, m_left and m_right
	//m_bvh->getNode(m_node.addr, &m_nodeSplit, m_left.box, m_right.box, m_left.addr, m_right.addr);
	//m_node.box = m_left.box + m_right.box; // Compute the root box
	//growParentBox();
	m_sibling.addr = NO_NODE;

	// Inititalize stacks
	m_nodeStack.add(m_node);
	m_splitPath.add(m_nodeSplit.getPos() + " " + m_nodeSplit.getAxisName() + ":");

	// Clear the osah split counts in current node
	memset(m_osahSplits, 0, sizeof(m_osahSplits));

	// Initialize rays
	if(rays != NULL)
	{
		Array<Vec4f> lines;
		const int stride = 4096;

		//PixelTable pixelTable;
		//pixelTable.setSize(Vec2i(1024, 768));

		//for(S32 i = 0; i < 1024*32; i++)
		//{
		//	int index = pixelTable.getPixelToIndex().getMutablePtr()[i];
		//	Ray ray = rays->getRayForSlot(index);
		//	float t;
		//	if(rays->getResultForSlot(index).hit())
		//		//t = rays->getResultForSlot(i*stride).t;
		//		t = rays->getResultForSlot(index).padA;
		//	else
		//		t = ray.tmax;
		//	lines.add(Vec4f(ray.origin, 1.0f));
		//	lines.add(Vec4f(ray.origin + t*ray.direction, 1.0f));
		//}

		for(S32 i = 0; i < rays->getSize()/stride; i++)
		{
			Ray ray = rays->getRayForSlot(i*stride);
			float t;
			if(rays->getResultForSlot(i*stride).hit())
				//t = rays->getResultForSlot(i*stride).t;
				t = rays->getResultForSlot(i*stride).padA;
			else
				t = ray.tmax;
			lines.add(Vec4f(ray.origin, 1.0f));
			lines.add(Vec4f(ray.origin + t*ray.direction, 1.0f));
		}

		m_rays.resizeDiscard(lines.getNumBytes());
		m_rays.set(lines.getPtr(), lines.getNumBytes());
	}
	else
	{
		m_showRays = false;
	}

	// Initialize empty boxes
	Array<Vec4f> boxes, colors, lineColors, colorPalette;
	for(int i = 0; i < emptyBoxes.getSize(); i++)
		addBoxQuads(emptyBoxes[i], boxes);
	m_emptyBoxes.resizeDiscard(boxes.getNumBytes());
	m_emptyBoxes.set(boxes.getPtr(), boxes.getNumBytes());
	// Initialize colors for empty boxes
	/*for(int r = 1; r > -1; r--)
		for(int g = 1; g > -1; g--)
			for(int b = 1; b > -1; b--)*/
	for(int r = 0; r < 2; r++)
		for(int g = 0; g < 2; g++)
			for(int b = 0; b < 2; b++)
				if(r+g+b < 3)
					colorPalette.add(Vec4f((float)r, (float)g, (float)b, 0.33f));
	for(int i = 0; i < emptyBoxes.getSize(); i++)
		for(int j = 0; j < 6*4; j++) // Repeate for each vertex of the box 6 faces * 4 vertices per face
		{
			colors.add(colorPalette[i % colorPalette.getSize()]);
			lineColors.add(colorPalette[i % colorPalette.getSize()]);
			lineColors.getLast().w = 1.0f;
		}
	m_emptyColors.resizeDiscard(colors.getNumBytes());
	m_emptyColors.set(colors.getPtr(), colors.getNumBytes());
	m_emptyLineColors.resizeDiscard(lineColors.getNumBytes());
	m_emptyLineColors.set(lineColors.getPtr(), lineColors.getNumBytes());

	// Initialize visibility
	if(visibility != NULL)
	{
		m_visibility.set((S32*)visibility->getPtr(), scene->getNumTriangles());
	}
	else
	{
		m_visibility.reset(scene->getNumTriangles());
		memset(m_visibility.getPtr(), 0, m_visibility.getNumBytes());
	}

	m_nodeColor = COLOR_NODE;
	m_siblingColor = COLOR_SIBLING;
	m_leftColor = COLOR_LEFT;
	m_rightColor = COLOR_RIGHT;
	m_rayColor = COLOR_RAY;
}

//------------------------------------------------------------------------

VisualizationBVH::~VisualizationBVH(void)
{
}

//------------------------------------------------------------------------

bool VisualizationBVH::handleEvent(const Window::Event& ev)
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
			if (ev.key == FW_KEY_P && m_emptyBoxes.getSize() > 0)   { m_showEmpty = !m_showEmpty; }
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

void VisualizationBVH::moveToParent()
{	
	moveUp();

	// Update path
	m_nodeStack[m_currentDepth] = m_node;
	if(!m_splitPath[m_currentDepth].endsWith(":"))
		m_splitPath[m_currentDepth] = m_splitPath[m_currentDepth].substring(0, m_splitPath[m_currentDepth].getLength()-1);
	//(m_splitPath.get(m_currentDepth).[m_splitPath[m_currentDepth].indexOf(':')+1]) = ' ';
	m_nodeStack.resize(m_currentDepth+1);
	m_splitPath.resize(m_currentDepth+1);
}

//------------------------------------------------------------------------

void VisualizationBVH::moveToSibling()
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
	m_splitPath[m_currentDepth] = m_nodeSplit.getPos() + " " + m_nodeSplit.getAxisName() + ": ";
	m_nodeStack.resize(m_currentDepth+1);
	m_splitPath.resize(m_currentDepth+1);
	// Flip child identifier from the previous split
	if(m_splitPath[m_currentDepth-1][m_splitPath[m_currentDepth-1].indexOf(':')+1] == 'L')
	{
		m_splitPath[m_currentDepth-1] = m_splitPath[m_currentDepth-1].substring(0, m_splitPath[m_currentDepth-1].getLength()-1);
		m_splitPath[m_currentDepth-1] = m_splitPath[m_currentDepth-1].append('R');
	}
	//	m_splitPath[m_currentDepth-1][m_splitPath[m_currentDepth].indexOf(':')+1] = 'R';
	else
	{
	//	m_splitPath[m_currentDepth-1][m_splitPath[m_currentDepth].indexOf(':')+1] = 'L';
		m_splitPath[m_currentDepth-1] = m_splitPath[m_currentDepth-1].substring(0, m_splitPath[m_currentDepth-1].getLength()-1);
		m_splitPath[m_currentDepth-1] = m_splitPath[m_currentDepth-1].append('L');
	}
}

//------------------------------------------------------------------------

void VisualizationBVH::moveToLeft()
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

	//growParentBox();
	// Set color mapping for visible nodes
	setColorMapping();

	// Update path
	m_splitPath[m_currentDepth] = m_splitPath[m_currentDepth].append('L');
	m_nodeStack.resize(m_currentDepth+1);
	m_splitPath.resize(m_currentDepth+1);
	// Add to the path
	m_nodeStack.add(m_node);
	m_splitPath.add(m_nodeSplit.getPos() + " " + m_nodeSplit.getAxisName() + ": ");
	m_currentDepth++;
}

//------------------------------------------------------------------------

void VisualizationBVH::moveToRight()
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

	//growParentBox();
	// Set color mapping for visible nodes
	setColorMapping();

	// Update path
	m_splitPath[m_currentDepth] = m_splitPath[m_currentDepth].append('R');
	m_nodeStack.resize(m_currentDepth+1);
	m_splitPath.resize(m_currentDepth+1);
	// Add to the path
	m_nodeStack.add(m_node);
	m_splitPath.add(m_nodeSplit.getPos() + " " + m_nodeSplit.getAxisName() + ": ");
	m_currentDepth++;
}

//------------------------------------------------------------------------

void VisualizationBVH::moveUp()
{
	if(m_currentDepth > 0)
	{
		m_currentDepth--;
		getFromIndex(m_currentDepth);
	}
}

//------------------------------------------------------------------------

void VisualizationBVH::moveDown()
{
	if(m_currentDepth < m_nodeStack.getSize()-1)
	{
		m_currentDepth++;
		getFromIndex(m_currentDepth);
	}
}

//------------------------------------------------------------------------

void VisualizationBVH::draw(GLContext* gl, CameraControls& camera)
{
	FW_ASSERT(gl);

	if(!isVisible())
		return;

	Mat4f oldXform = gl->setVGXform(gl->xformFitToView(-1.0f, 2.0f) * camera.getWorldToClip());
    glPushAttrib(GL_ENABLE_BIT);
    glEnable(GL_DEPTH_TEST);
	glDepthFunc(GL_LESS);
	glEnable(GL_CULL_FACE);
	glFrontFace(GL_CW);

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
	glDisable(GL_DEPTH_TEST);
	glDisable(GL_CULL_FACE);

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

void VisualizationBVH::getFromIndex(S32 idx)
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

	//growParentBox();
	// Set color mapping for visible nodes
	setColorMapping();
}

//------------------------------------------------------------------------

void VisualizationBVH::growParentBox()
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

void VisualizationBVH::drawNodes(GLContext* gl, bool onlyChildren)
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
	if(m_showAllOSAH || m_showEmpty)
	{
		if(!onlyChildren)
		{
			// Draw filled faces
			if(m_showEmpty)
				gl->drawColorBuffer(m_emptyBoxes, m_emptyColors, GL_QUADS, 0);
			//glDepthFunc(GL_ALWAYS);

			// Draw edges, use opaque ray color
			glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
			if(m_showAllOSAH)
				gl->drawBuffer(m_boxes, GL_QUADS, 0, COLOR_RAY);
			if(m_showEmpty)
				gl->drawColorBuffer(m_emptyBoxes, m_emptyLineColors, GL_QUADS, 0);
			glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
			//glDepthFunc(GL_LEQUAL);
		}
	}
}

//------------------------------------------------------------------------

void VisualizationBVH::drawBox(GLContext* gl, const NodeData &node, U32 abgr)
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

void VisualizationBVH::drawRays(GLContext* gl, U32 abgr)
{
	if(m_showRays)
	{
		glLineWidth(2.0f);
		gl->drawBuffer(m_rays, GL_LINES, 0, abgr);
		glLineWidth(1.0f);
	}
}

//------------------------------------------------------------------------

void VisualizationBVH::drawPrimitives(GLContext* gl)
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

void VisualizationBVH::drawPathInfo(GLContext* gl)
{
	S32 fontSize = 16;
	Vec2f origin = Vec2f(8.0f, (F32)gl->getViewSize().y - 4.0f);
	Vec2f pos = origin;

	Mat4f oldXform = gl->setVGXform(gl->xformMatchPixels());
	gl->setFont("Arial", fontSize, GLContext::FontStyle_Bold);
	
	char leftBox[100], rightBox[100];
	memset(leftBox, '\0', 100);
	memset(rightBox, '\0', 100);
	//m_left.box.min().sprint(leftBox, 100);
	strcat_s(leftBox, ", ");
	//m_left.box.max().sprint(leftBox + strlen(leftBox), 100-strlen(leftBox));
	//m_right.box.min().sprint(rightBox, 100);
	strcat_s(rightBox, ", ");
	//m_right.box.max().sprint(rightBox + strlen(rightBox), 100-strlen(rightBox));

	if(m_showCurrTris)
	{
		gl->drawLabel(sprintf("Left count: %u     Right count: %u",
		m_leftPrims, m_rightPrims), pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
		pos.y -= (F32)fontSize;
		gl->drawLabel(sprintf("Left area: %.2f (%s)     Right area: %.2f (%s)",
		m_left.box.area(), leftBox, m_right.box.area(), rightBox), pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
	}
	else
	{
		gl->drawLabel(sprintf("Left area: %.2f (%s)     Right area: %.2f (%s)",
			m_left.box.area(), leftBox, m_right.box.area(), rightBox), pos, Vec2f(0.0f, 1.0f), 0xFFFFFFFF);
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

void VisualizationBVH::setColorMapping()
{
	if(!m_splitColors)
	{
		m_leftColor = COLOR_LEFT;
		m_rightColor = COLOR_RIGHT;
		return;
	}
	
	//if(m_nodeSplit.getType() == SplitInfo::SAH)
	//{
	//	m_leftColor = COLOR_LEFT_SAH;
	//	m_rightColor = COLOR_RIGHT_SAH;
	//}
	//else if(m_nodeSplit.getType() == SplitInfo::SBVH)
	//{
	//	m_leftColor = COLOR_LEFT_SVBH;
	//	m_rightColor = COLOR_RIGHT_SVBH;
	//}
	//else
	//{
	//	m_leftColor = COLOR_LEFT_OSAH;
	//	m_rightColor = COLOR_RIGHT_OSAH;
	//}
}

//-----------------------------------------------------------------------

void VisualizationBVH::prepareTreeData(NodeData node)
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

				if(true)/*m_showAllOSAH && splitInfo.getOSAHChosen()) */// Process split
				{
					//m_osahSplits[splitInfo.getAxis()]++;
					// Compute the box
					//AABB bbox = child0 + child1;
					//AABB bbox = child0; // child with more visible triangles in case of OSAH split
					//addBoxQuads(bbox, boxes);
					addBoxQuads(child0, boxes);
					addBoxQuads(child1, boxes);
				}

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

//-----------------------------------------------------------------------

void VisualizationBVH::addBoxQuads(const AABB &box, Array<Vec4f> &buffer)
{
	Vec3f min = box.min();
	Vec3f max = box.max();
	// Add buffer as 4 quads
	// Min x
	buffer.add(Vec4f(min.x, min.y, min.z, 1.0f));
	buffer.add(Vec4f(min.x, max.y, min.z, 1.0f));
	buffer.add(Vec4f(min.x, max.y, max.z, 1.0f));
	buffer.add(Vec4f(min.x, min.y, max.z, 1.0f));
	// Max 
	buffer.add(Vec4f(max.x, max.y, max.z, 1.0f));
	buffer.add(Vec4f(max.x, max.y, min.z, 1.0f));
	buffer.add(Vec4f(max.x, min.y, min.z, 1.0f));
	buffer.add(Vec4f(max.x, min.y, max.z, 1.0f));
	// Min y
	buffer.add(Vec4f(min.x, min.y, min.z, 1.0f));
	buffer.add(Vec4f(min.x, min.y, max.z, 1.0f));
	buffer.add(Vec4f(max.x, min.y, max.z, 1.0f));
	buffer.add(Vec4f(max.x, min.y, min.z, 1.0f));
	// Max y
	buffer.add(Vec4f(max.x, max.y, max.z, 1.0f));
	buffer.add(Vec4f(min.x, max.y, max.z, 1.0f));
	buffer.add(Vec4f(min.x, max.y, min.z, 1.0f));
	buffer.add(Vec4f(max.x, max.y, min.z, 1.0f));
	// Min z
	buffer.add(Vec4f(min.x, min.y, min.z, 1.0f));
	buffer.add(Vec4f(max.x, min.y, min.z, 1.0f));
	buffer.add(Vec4f(max.x, max.y, min.z, 1.0f));
	buffer.add(Vec4f(min.x, max.y, min.z, 1.0f));
	// Max z
	buffer.add(Vec4f(max.x, max.y, max.z, 1.0f));
	buffer.add(Vec4f(max.x, min.y, max.z, 1.0f));
	buffer.add(Vec4f(min.x, min.y, max.z, 1.0f));
	buffer.add(Vec4f(min.x, max.y, max.z, 1.0f));
}

//------------------------------------------------------------------------

void VisualizationBVH::splitNode(const NodeData& currNode, S32& leftAdd, S32& rightAdd, AABB& leftBox, AABB& rightBox, SplitInfo& split)
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
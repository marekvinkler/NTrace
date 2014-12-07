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
#include "Visualization.hpp"
//#include "ray/PixelTable.hpp"

namespace FW
{
//------------------------------------------------------------------------

Visualization::Visualization(Scene* scene, const RayBuffer* rays, Buffer* visibility) 
:	m_currentDepth(0),
	m_visible(false),
	m_showRays(false),
	m_splitColors(false),
	m_showChildren(true),
	m_showAllOSAH(false),
	m_showCurrTris(false),
	m_scene(scene)
{
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
				t = rays->getResultForSlot(i*stride).t;
				//t = (float)rays->getResultForSlot(i*stride).padA;
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

	// Initialize visibility
	if(visibility != NULL)
	{
		m_visibility.set((S32*)visibility->getPtr(), m_scene->getNumTriangles());
	}
	else
	{
		m_visibility.reset(m_scene->getNumTriangles());
		memset(m_visibility.getPtr(), 0, m_visibility.getNumBytes());
	}
}

//------------------------------------------------------------------------

Visualization::~Visualization()
{
}

//-----------------------------------------------------------------------

void Visualization::addBoxQuads(const AABB &box, Array<Vec4f> &buffer)
{
	Vec3f min = box.min();
	Vec3f max = box.max();
	// Add buffer as 4 quads
	// Min x
	buffer.add(Vec4f(min.x, min.y, min.z, 1.0f));
	buffer.add(Vec4f(min.x, max.y, min.z, 1.0f));
	buffer.add(Vec4f(min.x, max.y, max.z, 1.0f));
	buffer.add(Vec4f(min.x, min.y, max.z, 1.0f));
	// Max x
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

//-----------------------------------------------------------------------

}
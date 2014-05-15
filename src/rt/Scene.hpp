/*
 *  Copyright (c) 2009-2011, NVIDIA Corporation
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of NVIDIA Corporation nor the
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
#include "base/Math.hpp"
#include "3d/Mesh.hpp"
#include "3d/TextureAtlas.hpp"

namespace FW
{
//------------------------------------------------------------------------
	
class Scene
{
public:
                    Scene               (const MeshBase& mesh);
                    ~Scene              (void);

    int             getNumTriangles             (void) const    { return m_numTriangles; }
    int             getNumVertices              (void) const    { return m_numVertices; }
    int             getNumEmissive              (void) const    { return m_numEmissive; }

    Buffer&         getTriVtxIndexBuffer        (void)          { return m_triVtxIndex; }
    Buffer&         getTriNormalBuffer          (void)          { return m_triNormal; }
    Buffer&         getTriMaterialColorBuffer   (void)          { return m_triMaterialColor; }
    Buffer&         getTriShadedColorBuffer     (void)          { return m_triShadedColor; }
    Buffer&         getVtxPosBuffer             (void)          { return m_vtxPos; }
	Buffer&			getVtxNormalBuffer			(void)			{ return m_vtxNorm; }
	Buffer&			getVtxTexCoordBuffer		(void)			{ return m_vtxTC; }
	Buffer&			getTextureAtlasInfo			(void)			{ return m_atlasInfo; }
	Buffer&			getMaterialIds				(void)			{ return m_matId; }
	Buffer&			getEmissiveTris				(void)			{ return m_emissiveTris; }
	Buffer&			getMaterialInfo				(void)			{ return m_matInfo; }
	TextureAtlas*	getTextureAtlas				(void)			{ return m_texture; }

    U32             hash                        (void);
	
	void			getBBox(Vec3f& lo, Vec3f &hi) const	{lo = m_AABBMin; hi = m_AABBMax;};
private:
                    Scene               (const Scene&); // forbidden
    Scene&          operator=           (const Scene&); // forbidden

private:
    S32             m_numTriangles;
    S32             m_numVertices;
	S32				m_numEmissive;
    Buffer          m_triVtxIndex;			// Vec3i[numTriangles]
    Buffer          m_triNormal;			// Vec3f[numTriangles]
    Buffer          m_triMaterialColor;		// U32[numTriangles], ABGR
    Buffer          m_triShadedColor;		// U32[numTriangles], ABGR
    Buffer          m_vtxPos;				// Vec3f[numVertices]
	Buffer			m_vtxNorm;				// Vec3f[numVertices]
	Buffer			m_vtxTC;				// Vec2f[numVertices]
	Buffer			m_atlasInfo;			// Vec4f[numTriangles] - texture atlas information (xy contains offset, zw contains size)
	Buffer			m_matId;				// U32[numTriangles], material id
	Buffer			m_emissiveTris;			// Vec3i[m_numEmissive]
	Buffer			m_matInfo;				// Vec4f[numMaterials] - material information (emissivity, reflectivity, refractivity, texture?)
	Vec3f			m_AABBMin, m_AABBMax;	// BBox of the scene
	TextureAtlas*	m_texture;				// Texture atlas holding scene's textures
};

//------------------------------------------------------------------------
}

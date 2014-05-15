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

#include "Scene.hpp"

using namespace FW;

//------------------------------------------------------------------------

Scene::Scene(const MeshBase& mesh)
{
    Vec3f light = Vec3f(1.0f, 2.0f, 3.0f).normalized();
	
	mesh.getBBox(m_AABBMin, m_AABBMax);

    // Convert mesh and allocate buffers.

    Mesh<VertexPNT> meshP(mesh);
    m_numTriangles = meshP.numTriangles();
    m_numVertices = meshP.numVertices();
	m_numEmissive = 0;

    m_triVtxIndex.resizeDiscard(m_numTriangles * sizeof(Vec3i));
    m_triNormal.resizeDiscard(m_numTriangles * sizeof(Vec3f));
    m_triMaterialColor.resizeDiscard(m_numTriangles * sizeof(U32));
    m_triShadedColor.resizeDiscard(m_numTriangles * sizeof(U32));
    m_vtxPos.resizeDiscard(m_numVertices * sizeof(Vec3f));
	m_vtxTC.resizeDiscard(m_numVertices * sizeof(Vec2f));
	m_vtxNorm.resizeDiscard(m_numVertices * sizeof(Vec3f));
	m_atlasInfo.resizeDiscard(m_numTriangles * sizeof(Vec4f));
	m_matId.resizeDiscard(m_numTriangles * sizeof(U32));
	m_matInfo.resizeDiscard(meshP.numSubmeshes() * sizeof(Vec4f));

    Vec3i* triVtxIndex      = (Vec3i*)m_triVtxIndex.getMutablePtr();
    Vec3f* triNormal        = (Vec3f*)m_triNormal.getMutablePtr();
    U32* triMaterialColor   = (U32*)m_triMaterialColor.getMutablePtr();
    U32* triShadedColor     = (U32*)m_triShadedColor.getMutablePtr();
    Vec3f* vtxPos           = (Vec3f*)m_vtxPos.getMutablePtr();
	Vec2f* vtxTC			= (Vec2f*)m_vtxTC.getMutablePtr();
	Vec3f* vtxNorm			= (Vec3f*)m_vtxNorm.getMutablePtr();
	Vec4f* texInfo			= (Vec4f*)m_atlasInfo.getMutablePtr();
	U32* matId				= (U32*)m_matId.getMutablePtr();
	Vec4f* matInfo			= (Vec4f*)m_matInfo.getMutablePtr();

    // Copy vertices.

    const VertexPNT* v = meshP.getVertexPtr();
    for (int i = 0; i < m_numVertices; i++)
	{
        vtxPos[i] = v[i].p;
		vtxTC[i] = v[i].t;
		vtxNorm[i] = v[i].n;
	}

	// Create texture atlas
	m_texture = new TextureAtlas(ImageFormat::RGBA_Vec4f);
	
	// Fill texture atlas with ALL textures
    for (int submesh = 0; submesh < meshP.numSubmeshes(); submesh++)
    {
        const MeshBase::Material& material = meshP.material(submesh);
		if(material.textures[MeshBase::TextureType::TextureType_Diffuse].getImage() != NULL)
		{
			m_texture->addTexture(material.textures[MeshBase::TextureType::TextureType_Diffuse], 0);

			int val = 1;
			matInfo[submesh].w = 1.0f;
		}
		else
		{
			int val = 0;
			matInfo[submesh].w = 0.0f;
		}

		matInfo[submesh].x = material.emissivity;
		matInfo[submesh].y = material.reflectivity;
		matInfo[submesh].z = material.refractivity;
	}

    // Collapse submeshes to a single triangle list.
    for (int submesh = 0; submesh < meshP.numSubmeshes(); submesh++)
    {
        const Array<Vec3i>& indices = meshP.indices(submesh);
        const MeshBase::Material& material = meshP.material(submesh);
        U32 colorU32 = material.diffuse.toABGR();
        Vec3f colorVec3f = material.diffuse.getXYZ();

        for (int i = 0; i < indices.getSize(); i++)
        {
            const Vec3i& vi     = indices[i];
            Vec3f normal = normalize(cross(vtxPos[vi.y] - vtxPos[vi.x], vtxPos[vi.z] - vtxPos[vi.x]));

            *triVtxIndex++      = vi;
            *triNormal++        = normal;
            *triMaterialColor++ = colorU32;
            *triShadedColor++   = Vec4f(colorVec3f * (dot(normal, light) * 0.5f + 0.5f), 1.0f).toABGR();

			*matId++			= submesh;
			
			if(material.textures[MeshBase::TextureType::TextureType_Diffuse] != NULL)
			{
				*texInfo++			= Vec4f(m_texture->getTexturePosF(material.textures[MeshBase::TextureType::TextureType_Diffuse]), 
											m_texture->getTextureSizeF(material.textures[MeshBase::TextureType::TextureType_Diffuse]));
			}
			else
			{
				*texInfo++			= Vec4f(0.0f, 0.0f, 0.0f, 0.0f);
			}

			if(material.emissivity > 0.0f)
			{
				m_numEmissive++;
			}
        }
    }

	// Store emissive triangles
	U32 counter = 0;

	m_emissiveTris.resizeDiscard(m_numEmissive * sizeof(Vec3i));

	Vec3i* emissiveTris = (Vec3i*)m_emissiveTris.getMutablePtr();
	
    for (int submesh = 0; submesh < meshP.numSubmeshes(); submesh++)
    {
        const Array<Vec3i>& indices = meshP.indices(submesh);
        const MeshBase::Material& material = meshP.material(submesh);

        for (int i = 0; i < indices.getSize(); i++)
        {
			if(material.emissivity > 0.0f)
			{
				const Vec3i& vi     = indices[i];

				*emissiveTris++ = vi;
				counter++;
			}
		}
	}
}

//------------------------------------------------------------------------

Scene::~Scene(void)
{
}

//------------------------------------------------------------------------

U32 Scene::hash(void)
{
    return hashBits(
        hashBuffer(m_triVtxIndex.getPtr(), (int)m_triVtxIndex.getSize()),
        hashBuffer(m_triNormal.getPtr(), (int)m_triNormal.getSize()),
        hashBuffer(m_triMaterialColor.getPtr(), (int)m_triMaterialColor.getSize()),
        hashBuffer(m_triShadedColor.getPtr(), (int)m_triShadedColor.getSize()),
        hashBuffer(m_vtxPos.getPtr(), (int)m_vtxPos.getSize()));
}

//------------------------------------------------------------------------

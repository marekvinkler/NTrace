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
#include "cuda/InterpolationKernels.hpp"

using namespace FW;

//------------------------------------------------------------------------

Scene::Scene(const MeshBase& mesh)
{
    Vec3f light = Vec3f(1.0f, 2.0f, 3.0f).normalized();
	
	mesh.getBBox(m_AABBMin, m_AABBMax);
	m_recalculateBBox = false;

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

	// Copy animation data
	m_frames.reset();
	m_framerate = 0.f;

	for (int i = 0; i < mesh.getFrames().getSize(); i++)
	{
		Frame& frame = m_frames.add();
		frame.time = mesh.getFrames().get(i).time;
		frame.vertices = new Buffer;
		frame.vertices->resizeDiscard(m_numVertices * sizeof(Vec3f));

		Vec3f* pos = (Vec3f*)frame.vertices->getMutablePtr();
		const VertexPNT* vv = (const VertexPNT*)mesh.getFrames().get(i).vertices.getPtr();

		for (int vc = 0; vc < m_numVertices; vc++)
		{
			pos[vc] = vv[vc].p;
		}
	}

	if(m_frames.getSize() != 0)
	{
		m_framerate = 1.f;
		m_numRenderFrames = 0;
		setFrameRate(5.f);
	}

	m_compiler.setSourceFile("src/rt/cuda/animationInterpolation.cu");
	m_compiler.include("src/framework");
	m_compiler.addOptions("-use_fast_math");
}

//------------------------------------------------------------------------

Scene::~Scene(void)
{
	for (int i = 0; i < m_frames.getSize(); i++)
		delete m_frames[i].vertices;
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

void Scene::getBBox(Vec3f& lo, Vec3f &hi)
{
	if(m_recalculateBBox)
	{
		lo = Vec3f(+FW_F32_MAX);
		hi = Vec3f(-FW_F32_MAX);

		for (int i = 0; i < getNumVertices(); i++)
		{
			for (int d = 0; d < 3; d++)
			{
				m_AABBMin[d] = min(m_AABBMin[d], ((Vec3f*)m_vtxPos.getPtr())[i][d]);
				m_AABBMin[d] = max(m_AABBMin[d], ((Vec3f*)m_vtxPos.getPtr())[i][d]);
			}
		}
	}

	lo = m_AABBMin;
	hi = m_AABBMax;

	m_recalculateBBox = false;
}

//------------------------------------------------------------------------

F32 Scene::getAnimationLength() const
{
	return m_frames.getSize() != 0 ? m_frames.get(m_frames.getSize() - 1).time : 0.f;
}

//------------------------------------------------------------------------

void Scene::setTime(F32 newTime)
{
	m_recalculateBBox = true;
	int interpolateIdx = 0;

	while (interpolateIdx < m_frames.getSize() -1 && m_frames[interpolateIdx].time < newTime)
		interpolateIdx++;

	if (interpolateIdx > 1)
		m_frames.get(interpolateIdx - 2).vertices->free(FW::Buffer::Module::Cuda);

	if (interpolateIdx == m_frames.getSize())
	{
		m_frames.get(interpolateIdx - 1).vertices->free(FW::Buffer::Module::Cuda);
		return;
	}

	CudaModule* module = m_compiler.compile();
	InterpolationInput& in = *(InterpolationInput*)module->getGlobal("c_InterpolationInput").getMutablePtr();
	
	if(interpolateIdx == 0)
	{
		in.weight = 0.f;
		in.verticesB = (m_frames.get(interpolateIdx).vertices)->getCudaPtr();
		in.verticesA = (m_frames.get(interpolateIdx).vertices)->getCudaPtr();
	}
	else
	{
		in.weight = (newTime - m_frames.get(interpolateIdx - 1).time) / (m_frames.get(interpolateIdx).time - m_frames.get(interpolateIdx - 1).time);
		in.verticesA = (m_frames.get(interpolateIdx - 1).vertices)->getCudaPtr();
		in.verticesB = (m_frames.get(interpolateIdx).vertices)->getCudaPtr();
	}

	in.verticesIntrp = m_vtxPos.getMutableCudaPtr();
	in.vertCount = m_frames.get(interpolateIdx).vertices->getSize();

	module->getKernel("interpolateVertices").launch(m_frames.get(interpolateIdx).vertices->getSize());
}

//-----------------------------------------------------------------------

void Scene::setFrameRate(F32 newFramerate)
{
	if(m_frames.getSize() == 0)
		return;

	if(newFramerate < 0)
	{
		m_numRenderFrames = -(S32)(newFramerate);
		return;
	}

	const F32 conversionRate = m_framerate / newFramerate;

	for(int i = 0; i < m_frames.getSize(); i++)
	{
		m_frames.get(i).time *= conversionRate;
	}

	m_framerate = newFramerate;
}

//------------------------------------------------------------------------
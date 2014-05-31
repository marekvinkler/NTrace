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

/**
 * \brief Scene class declarations.
 */

#pragma once
#include "base/Math.hpp"
#include "3d/Mesh.hpp"
#include "3d/TextureAtlas.hpp"

namespace FW
{
//------------------------------------------------------------------------
	
/**
 * \brief Class holding 3d scene.
 */
class Scene
{
public:
	/**
	 * \brief Constructor.
	 * \param[in] mesh Source mesh.
	 */
                    Scene               (const MeshBase& mesh);

	/**
	 * \brief Destructor.
	 */
                    ~Scene              (void);

	/**
	 * \return Number of triangles in the scene.
	 */
    int             getNumTriangles             (void) const    { return m_numTriangles; }

	/**
	 * \return Number of vertices in the scene.
	 */
    int             getNumVertices              (void) const    { return m_numVertices; }

    int             getNumEmissive              (void) const    { return m_numEmissive; }

	/**
	 * \brief Returns buffer of triangle's vertex indieces.
	 * \details For each triangle, three vertex indices are saved in this buffer.
	 * \return Buffer of triangle's vertex indices.
	 */
    Buffer&         getTriVtxIndexBuffer        (void)          { return m_triVtxIndex; }

	/**
	 * \brief Returns triangle normal buffer.
	 * \details For each triangle there is a normal vector saved in this buffer.
	 * \return Triangle normal buffer.
	 */
    Buffer&         getTriNormalBuffer          (void)          { return m_triNormal; }

	/**
	 * \brief Returns material color buffer.
	 * \details Color of each triangle's material is saved in this buffer.
	 * \return Material color buffer.
	 */
    Buffer&         getTriMaterialColorBuffer   (void)          { return m_triMaterialColor; }

	/**
	 * \brief Returns shaded color buffer.
	 * \details Shaded color of each triangle is saved in this buffer.
	 * \return Shaded color buffer.
	 */
    Buffer&         getTriShadedColorBuffer     (void)          { return m_triShadedColor; }

	/**
	 * \brief Returns vertex position buffer.
	 * \details Position of each vertex is saved in this buffer.
	 * \return Vertex position buffer.
	 */
    Buffer&         getVtxPosBuffer             (void)          { return m_vtxPos; }

	/**
	 * \brief Returns vertex normal buffer.
	 * \details Normal vector of each vertex is saved in this buffer.
	 * \return Vertex normal buffer.
	 */
	Buffer&			getVtxNormalBuffer			(void)			{ return m_vtxNorm; }

	/**
	 * \brief Returns vertex texture coordinate buffer.
	 * \details Texture coordinates of each vertex is saved in this buffer.
	 * \return Vertex texture coordinates buffer.
	 */
	Buffer&			getVtxTexCoordBuffer		(void)			{ return m_vtxTC; }

	/**
	 * \brief Returns texture atlas information buffer.
	 * \details Buffer contains texture atlas information (xy contains offset, zw contains size).
	 * \return Vertex atlas information buffer.
	 */
	Buffer&			getTextureAtlasInfo			(void)			{ return m_atlasInfo; }

	/**
	 * \brief Returns material id buffer.
	 * \details Material id of each vertex is saved in this buffer.
	 * \return Material id buffer.
	 */
	Buffer&			getMaterialIds				(void)			{ return m_matId; }

	/**
	 * \brief Returns buffer of emissive triangles.
	 * \details For each emissive triangle there are three vertex indices saved in this buffer.
	 * \return Emissive triangle buffer.
	 */
	Buffer&			getEmissiveTris				(void)			{ return m_emissiveTris; }

	/**
	 * \brief Returns material info buffer.
	 * \details Buffer contains material information (emissivity, reflectivity, refractivity, texture) for each matId.
	 * \return Material info buffer.
	 */
	Buffer&			getMaterialInfo				(void)			{ return m_matInfo; }

	/**
	 * \brief Returns texture atlas holding scene's textures.
	 * \return Texture atlas.
	 */
	TextureAtlas*	getTextureAtlas				(void)			{ return m_texture; }

	/**
	 * \return Hash of the scene.
	 */
    U32             hash                        (void);
	
	/**
	 * \brief Gets scene AABB's minimum and maximum vector.
	 * \param[out] lo Minimum vector.
	 * \param[out] hi Maximum vector.
	 */
	void			getBBox(Vec3f& lo, Vec3f &hi) const	{lo = m_AABBMin; hi = m_AABBMax;};
private:
                    Scene               (const Scene&); // forbidden
    Scene&          operator=           (const Scene&); // forbidden

private:
    S32             m_numTriangles;			//!< Number of triangles.
    S32             m_numVertices;			//!< Number of vertices.
	S32				m_numEmissive;			//!< Number of emissive triangles.
    Buffer          m_triVtxIndex;			//!< Indices of triangle's vertices. Vec3i[numTriangles]
    Buffer          m_triNormal;			//!< Normal vector of each triangle. Vec3f[numTriangles]
    Buffer          m_triMaterialColor;		//!< Material color of each triangle. U32[numTriangles], ABGR
    Buffer          m_triShadedColor;		//!< Shaded color of each triangle.  U32[numTriangles], ABGR
    Buffer          m_vtxPos;				//!< Vertex positions. Vec3f[numVertices]
	Buffer			m_vtxNorm;				//!< Vertex normals. Vec3f[numVertices]
	Buffer			m_vtxTC;				//!< Vertex texture coordinates. Vec2f[numVertices]
	Buffer			m_atlasInfo;			//!< Texture atlas info. Vec4f[numTriangles] - texture atlas information (xy contains offset, zw contains size)
	Buffer			m_matId;				//!< Material ids U32[numTriangles], material id
	Buffer			m_emissiveTris;			//!< Emissive triangles. Vec3i[m_numEmissive]
	Buffer			m_matInfo;				//!< Material information. Vec4f[numMaterials] - material information (emissivity, reflectivity, refractivity, texture?)
	Vec3f			m_AABBMin, m_AABBMax;	//!< BBox of the scene
	TextureAtlas*	m_texture;				//!< Texture atlas holding scene's textures
};

//------------------------------------------------------------------------
}

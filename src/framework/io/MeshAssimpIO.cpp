#ifdef USE_ASSIMP

#include "MeshAssimpIO.hpp"
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>

using namespace FW;

//------------------------------------------------------------------------

struct ImportState
{
    Mesh<VertexPNT>*			mesh;

    //Array<Vec3f>				positions;

	S32							idxOffset;
	S32							submeshOffset;

	ImportState() : idxOffset(0), submeshOffset(0) {}

    //Array<Vec2f>            texCoords;
    //Array<Vec3f>            normals;

    //Hash<Vec3i, S32>        vertexHash;

    //Array<S32>              vertexTmp;
    //Array<Vec3i>            indexTmp;
};

//------------------------------------------------------------------------

Mat4f convertMatrix(const aiMatrix4x4& matrix)
{
	Mat4f converted;

	converted.m00 = matrix.a1;
	converted.m01 = matrix.a2;
	converted.m02 = matrix.a3;
	converted.m03 = matrix.a4;

	converted.m10 = matrix.b1;
	converted.m11 = matrix.b2;
	converted.m12 = matrix.b3;
	converted.m13 = matrix.b4;

	converted.m20 = matrix.c1;
	converted.m21 = matrix.c2;
	converted.m22 = matrix.c3;
	converted.m23 = matrix.c4;

	converted.m30 = matrix.d1;
	converted.m31 = matrix.d2;
	converted.m32 = matrix.d3;
	converted.m33 = matrix.d4;

	return converted;
}

//------------------------------------------------------------------------

void fillVerticesRecursive(ImportState& s, aiNode* node, aiMatrix4x4 transformation, const aiScene* aiscene)
{
	for (int m = 0; m < node->mNumMeshes; m++)
	{
		s.mesh->addSubmesh();
		MeshBase::Material mat;
		const aiMaterial* aimtl =  aiscene->mMaterials[aiscene->mMeshes[node->mMeshes[m]]->mMaterialIndex];

		aiColor3D aiDiffuse (0.f, 1.f, 0.f);
		aimtl->Get(AI_MATKEY_COLOR_DIFFUSE, aiDiffuse);
		mat.diffuse.x = aiDiffuse.r;
		mat.diffuse.y = aiDiffuse.g;
		mat.diffuse.z = aiDiffuse.b;
		mat.diffuse.w = 1.f;

		aiColor3D aiEmissive (0.f, 0.f, 0.f);
		aimtl->Get(AI_MATKEY_COLOR_EMISSIVE, aiEmissive);
		Vec3f em (aiEmissive.r, aiEmissive.g, aiEmissive.b);
		mat.emissivity = em.length();

		for(int i = 0; i < aimtl->GetTextureCount(aiTextureType_DIFFUSE); i++)
		{
			aiString path;
			aimtl->GetTexture(aiTextureType_DIFFUSE, 0, &path);
			Texture tex = Texture::import(path.C_Str());
			mat.textures[MeshBase::TextureType_Diffuse] = tex;
		}

		s.mesh->material(s.submeshOffset) = mat;

		for (int i = 0; i < aiscene->mMeshes[node->mMeshes[m]]->mNumFaces; i++)
		{
			if (aiscene->mMeshes[node->mMeshes[m]]->mFaces[i].mNumIndices != 3)
				continue;

			Vec3i tri;
			tri.x = aiscene->mMeshes[node->mMeshes[m]]->mFaces[i].mIndices[0] + s.idxOffset;
			tri.y = aiscene->mMeshes[node->mMeshes[m]]->mFaces[i].mIndices[1] + s.idxOffset;
			tri.z = aiscene->mMeshes[node->mMeshes[m]]->mFaces[i].mIndices[2] + s.idxOffset;
			s.mesh->mutableIndices(s.submeshOffset).add(tri);
		}

		for (int i = 0; i < aiscene->mMeshes[node->mMeshes[m]]->mNumVertices; i++)
		{
			Vec3f pos;
			Mat4f trans = convertMatrix(transformation);
			VertexPNT& v = s.mesh->addVertex();

			v.p.x = aiscene->mMeshes[node->mMeshes[m]]->mVertices[i].x;
			v.p.y = aiscene->mMeshes[node->mMeshes[m]]->mVertices[i].y;
			v.p.z = aiscene->mMeshes[node->mMeshes[m]]->mVertices[i].z;
			v.p = trans * v.p;

			if(aiscene->mMeshes[node->mMeshes[m]]->HasNormals())
			{
				v.n.x = aiscene->mMeshes[node->mMeshes[m]]->mNormals[i].x;
				v.n.y = aiscene->mMeshes[node->mMeshes[m]]->mNormals[i].y;
				v.n.z = aiscene->mMeshes[node->mMeshes[m]]->mNormals[i].z;
				v.n = trans * v.n;
			}

			if(aiscene->mMeshes[node->mMeshes[m]]->HasTextureCoords(0))
			{
				v.t.x = aiscene->mMeshes[node->mMeshes[m]]->mTextureCoords[0][i].x;
				v.t.y = aiscene->mMeshes[node->mMeshes[m]]->mTextureCoords[0][i].y;
			}
		}

		s.idxOffset += aiscene->mMeshes[node->mMeshes[m]]->mNumVertices;
		s.submeshOffset++;
	}

	for (int i = 0; i < node->mNumChildren; i++)
		fillVerticesRecursive(s, node->mChildren[i], node->mTransformation * transformation, aiscene);
}

Mesh<VertexPNT>* FW::importAssimpMesh(const Array<String>& files)
{
	Assimp::Importer importer;
	const aiScene* aiscene = NULL;

	ImportState s;
	s.mesh = new Mesh<VertexPNT>();
	s.mesh->clearVertices();

	for (int i = 0; i < files.getSize(); i++)
	{
		aiscene = importer.ReadFile(files.get(i).getPtr(), aiProcessPreset_TargetRealtime_Quality);
		s.mesh->setActiveFrame(i);
		s.mesh->setTime((F32)i);
		s.mesh->clear();
		fillVerticesRecursive(s, aiscene->mRootNode, aiMatrix4x4(), aiscene);
		s.mesh->compact();
		s.idxOffset = 0;
		s.submeshOffset = 0;
	}

	s.mesh->setActiveFrame(0);
	return s.mesh;
}

#endif
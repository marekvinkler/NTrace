#ifdef USE_ASSIMP

#pragma once
#include "base/Defs.hpp"
#include "3d/Mesh.hpp"

namespace FW
{

//Mesh<VertexPNT>*	importAssimpMesh	(const String& path);
Mesh<VertexPNT>*	importAssimpMesh	(const Array<String>& sequence);

// TODO: export?

}

#endif
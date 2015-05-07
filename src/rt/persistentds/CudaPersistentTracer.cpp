#include <sstream>
#include <iomanip>
#include "base/Random.hpp"
#include "CudaNoStructTracer.hpp"
#include "kernels/CudaTracerKernels.hpp"
#include "../../../../AppEnvironment.h"

#define TASK_SIZE 430000

using namespace FW;

#define BENCHMARK
#define TEST_TASKS
//#define CUTOFF_DEPTH // Reports only data at specific level - for debugging purpouses
//#define TRACE_L1

#ifndef TEST_TASKS
#include "kernels/thrustTest.hpp"
#endif

CudaNoStructTracer::CudaNoStructTracer(MiniMax::Scene& scene, F32 epsilon):
m_epsilon(epsilon)
{
	// init
	CudaModule::staticInit();
	//m_compiler.addOptions("-use_fast_math");
	m_compiler.addOptions("-use_fast_math -Xptxas -dlcm=cg");
	m_compiler.clearDefines();
	if (CudaModule::getComputeCapability() == 20 || CudaModule::getComputeCapability() == 21)
		m_compiler.define("FERMI");

	// convert from scene
	Vec3f light = Vec3f(1.0f, 2.0f, 3.0f).normalized();

	m_numTris           = scene.triangles.size();
	m_numVerts          = m_numTris * 3;
	m_numMaterials      = 1;
	m_numShadingNormals = m_numVerts;
	m_numTextureCoords  = m_numVerts;
	m_bbox = scene.box;
	
	m_tris.resizeDiscard(m_numTris * sizeof(SceneTriangle));
	m_triNormals.resizeDiscard(m_numTris * sizeof(Vec3f));
	//m_verts.resizeDiscard(m_numVerts * sizeof(Vec3f));
	m_materials.resizeDiscard(m_numMaterials * sizeof(Material));
	m_shadingNormals.resizeDiscard(m_numVerts * sizeof(Vec3f));
	m_shadedColor.resizeDiscard(m_numTris * sizeof(Vec3f));
	m_materialColor.resizeDiscard(m_numTris * sizeof(Vec3f));
	//m_textureCoords.resizeDiscard(m_numTextureCoords * sizeof(Vec2f));
	//m_trisBox.resizeDiscard(m_numTris * sizeof(CudaAABB));

	m_trisCompact.resizeDiscard(m_numTris * 3 * sizeof(Vec4f));
	m_trisIndex.resizeDiscard(m_numTris * sizeof(S32));

	SceneTriangle* tout  = (SceneTriangle*)m_tris.getMutablePtr();
	Vec3f*         nout = (Vec3f*)m_triNormals.getMutablePtr();
	//Vec3f*         vout  = (Vec3f*)m_verts.getMutablePtr();
	Material*      mout  = (Material*)m_materials.getMutablePtr();
	Vec3f*         snout = (Vec3f*)m_shadingNormals.getMutablePtr();
	U32*           scout = (U32*)m_shadedColor.getMutablePtr();
	U32*           mcout = (U32*)m_materialColor.getMutablePtr();
	//Vec2f*         uvout = (Vec2f*)m_textureCoords.getMutablePtr();
	//CudaAABB*      bout  = (CudaAABB*)m_trisBox.getMutablePtr();

	Vec4f* tcout  = (Vec4f*)m_trisCompact.getMutablePtr();
	S32*   tiout  = (S32*)m_trisIndex.getMutablePtr();

	// load vertices
	for (int i = 0; i < m_numTris; i++)
	{
		Triangle& tris = *scene.triangles[i];
		for(int j = 0; j < 3; j++)
		{			
			//vout[i*3+j]  = Vec3f(tris.vertices[j].x,tris.vertices[j].y,tris.vertices[j].z);
			snout[i*3+j] = Vec3f(tris.normals[j].x,tris.normals[j].y,tris.normals[j].z);
			//uvout[i*3+j] = Vec2f(tris.uvs[j].xx,tris.uvs[j].yy);

			*tcout = Vec4f(tris.vertices[j].x,tris.vertices[j].y,tris.vertices[j].z,0);
			tcout++;
		}

		/*Vec3f minV = min(vout[i*3+0], vout[i*3+1], vout[i*3+2]);
		bout[i].m_mn = make_float3(minV.x, minV.y, minV.z);
		Vec3f maxV = max(vout[i*3+0], vout[i*3+1], vout[i*3+2]);
		bout[i].m_mx = make_float3(maxV.x, maxV.y, maxV.z);*/
	}

	// default material
	Material m;
	m.diffuse      = Vec3f(1.0f,1.0f,1.0f);
	m.specular     = Vec3f(0.0f,0.0f,0.0f);
	m.type         = MeshBase::Material::MaterialType_Phong;
	m.texID        = -1; // no texture
	m.gloss_alpha  = Vec2f(0.0f, 0.f);
	mout[0] = m;

	unsigned int matid = 1;

	// load triangles
	Vec4f defaultColor(1.0f,1.0f,1.0f,1.0f);
	for(int i=0,j=0;i<m_numTris;i++,j+=3)
	{
		// triangle data
		tout->vertices = Vec3i(j,j+1,j+2);
		Triangle& tris = *scene.triangles[i];
		Vector3 normalVec = tris.GetNormal();
		tout->normal = Vec3f(normalVec.x,normalVec.y,normalVec.z);
		*nout = tout->normal;

		// material
		Material* mat;
		mat = &mout[0];
		matid = 0;
		
		Vec4f diffuseColor(mat->diffuse,1.0f);
		tout->materialColor = diffuseColor.toABGR();
		tout->shadedColor = Vec4f( diffuseColor.getXYZ() * (dot(tout->normal, light) * 0.5f + 0.5f), 1.0f).toABGR();
		tout->materialId = matid;

		*scout = tout->shadedColor;
		*mcout = tout->materialColor;
		scout++;
		mcout++;

		tout++;
		nout++;
	}

	m_sizeTask = 0.f;
	m_sizeSplit = 0.f;
	m_sizeADS = 0.f;
	m_sizeTri = 0.f;
	m_sizeTriIdx = 0.f;
	m_heap = 0.f;
}

F32 CudaNoStructTracer::traceBatch(RayBuffer& rays)
{
	m_kernelFile = "src/rt/kernels/persistent_nostruct.cu";
	m_compiler.setSourceFile(m_kernelFile);
	m_module = m_compiler.compile();
	failIfError();

	m_numRays = rays.getSize();

#ifdef TEST_TASKS
#ifdef DEBUG_PPS
	Random rand;
	m_numRays = rand.getU32(1, 1000000);
	m_numTris = rand.getU32(1, 1000000);
#endif
#endif

	// Set triangle index buffer
	S32* tiout = (S32*)m_trisIndex.getMutablePtr();
#ifdef DEBUG_PPS
	S32* ptout = (S32*)m_ppsTris.getMutablePtr();
	S32* cltout = (S32*)m_ppsTrisIndex.getMutablePtr();
	S32* stout = (S32*)m_sortTris.getMutablePtr();
#endif
	for(int i=0;i<m_numTris;i++)
	{
#ifndef DEBUG_PPS
		// indices 
		*tiout = i;
		tiout++;
#else
		int rnd = rand.getS32(-1, 2);
		//*ptout = rnd;
		*cltout = rnd;
		*stout = (rnd >= 1);
		//ptout++;
		cltout++;
		stout++;
#endif
	}

	m_raysIndex.resizeDiscard(sizeof(int)*m_numRays);
	rays.getResultBuffer().clear(-1); // Set as no hit
	int *rayIdx = (int*)m_raysIndex.getMutablePtr();
#ifdef DEBUG_PPS
	S32* prout = (S32*)m_ppsRays.getMutablePtr();
	S32* clrout = (S32*)m_ppsRaysIndex.getMutablePtr();
	S32* srout = (S32*)m_sortRays.getMutablePtr();
#endif
	for(int i = 0; i < m_numRays; i++)
	{
#ifndef DEBUG_PPS
		// Set rays index buffer
		*rayIdx = i;
		rayIdx++;

#ifdef TEST_TASKS
#if 0
		// CPU Clipping
		MiniMax::Ray mRay;
		memcpy(&mRay.origin, &ray.origin, sizeof(Vec3f));
		memcpy(&mRay.direction, &ray.direction, sizeof(Vec3f));
		// Clip the rays to the extent of scene box
		bool intersects = m_bbox.ComputeMinMaxT(mRay,
			ray.tmin, ray.tmax);

		// clip the origin of the rays 
		if (ray.tmin < 1e-3f)
			ray.tmin = 1e-3f;
#endif
#endif
#else
		int rnd = rand.getS32(-1, 3);
		//*prout = rnd;
		*clrout = rnd;
		*srout = (rnd >= 1);
		//prout++;
		clrout++;
		srout++;
#endif
	}

	// Start the timer
	m_timer.unstart();
	m_timer.start();

	//  Create the taskData
#ifdef TEST_TASKS
	m_taskData.resizeDiscard(TASK_SIZE * (sizeof(Task) + sizeof(int)));
	m_taskData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated

#if SPLIT_TYPE == 3
	m_splitData.resizeDiscard((S64)TASK_SIZE * (S64)sizeof(SplitInfo));
	m_splitData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#endif
#endif

	m_gpuTime = traceCudaRayBuffer(rays);
	m_cpuTime = m_timer.end();

	m_raysIndex.reset();
#ifdef DEBUG_PPS
	exit(0);
#endif

	return m_gpuTime;
}

F32 CudaNoStructTracer::buildBVH(bool sbvh)
{
#ifdef MALLOC_SCRATCHPAD
	// Set the memory limit according to triangle count
#ifndef BENCHMARK
	printf("Setting dynamic memory limit to %fMB\n", (float)(m_trisIndex.getSize()*5*3)/(float)(1024*1024));
#endif
	 cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, m_trisIndex.getSize()*5*3);
#endif

	// Compile the kernel
	if(!sbvh)
		m_kernelFile = "src/rt/kernels/persistent_bvh.cu";
	else
		m_kernelFile = "src/rt/kernels/persistent_sbvh.cu";

	m_compiler.setSourceFile(m_kernelFile);
	m_module = m_compiler.compile();
	failIfError();

#ifdef DEBUG_PPS
	Random rand;
	m_numTris = rand.getU32(1, 1000000);
#endif

	// Set triangle index buffer
	S32* tiout = (S32*)m_trisIndex.getMutablePtr();
#ifdef DEBUG_PPS
	S32* pout = (S32*)m_ppsTris.getMutablePtr();
	S32* clout = (S32*)m_ppsTrisIndex.getMutablePtr();
	S32* sout = (S32*)m_sortTris.getMutablePtr();
#endif
	for(int i=0;i<m_numTris;i++)
	{
#ifndef DEBUG_PPS
		// indices 
		*tiout = i;
		tiout++;
#else
		int rnd = rand.getU32(0, 2);
		//*pout = rnd;
		*clout = rnd;
		*sout = (rnd >= 1);
		//pout++;
		clout++;
		sout++;
#endif
	}

	// Start the timer
	m_timer.unstart();
	m_timer.start();

	// Create the taskData
	m_taskData.resizeDiscard(TASK_SIZE * (sizeof(TaskBVH) + sizeof(int)));
	m_taskData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
	//S64 bvhSize = ((m_numTris * sizeof(CudaBVHNode)) + 4096 - 1) & -4096;
	S64 bvhSize = ((m_numTris/2 * sizeof(CudaBVHNode)) + 4096 - 1) & -4096;
	m_bvhData.resizeDiscard(bvhSize);
	m_bvhData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
	//m_bvhData.clearRange32(0, 0, bvhSize); // Mark all tasks as 0 (important for debug)
#ifdef COMPACT_LAYOUT
	if(!sbvh)
	{
		m_trisCompactOut.resizeDiscard(m_numTris * (3+1) * sizeof(Vec4f));
		m_trisIndexOut.resizeDiscard(m_numTris * (3+1) * sizeof(S32));
	}
	else
	{
		m_trisCompactOut.resizeDiscard(m_numTris*2 * (3+1) * sizeof(Vec4f));
		m_trisIndexOut.resizeDiscard(m_numTris*2 * (3+1) * sizeof(S32));
	}
#endif

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
	m_splitData.resizeDiscard((S64)(TASK_SIZE+1) * (S64)sizeof(SplitArray));
	m_splitData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#endif

	m_gpuTime = buildCudaBVH();
	m_cpuTime = m_timer.end();

	// Resize to exact memory
	trimBVHBuffers();

#ifdef DEBUG_PPS
	exit(0);
#endif

	return m_gpuTime;
}

F32 CudaNoStructTracer::traceBatchBVH(RayBuffer& rays, RayStats* stats)
{
#ifdef TRACE_L1
	// Set compiler options
	m_compiler.clearOptions();
#endif

	m_compiler.setCachePath("cudacache"); // On the first compilation the cache path becames absolute which kills the second compilation
#ifdef COMPACT_LAYOUT
#ifdef WOOP_TRIANGLES
	String kernelName("src/rt/kernels/fermi_speculative_while_while");
#else
	String kernelName("src/rt/kernels/fermi_speculative_while_while_inter");
#endif
#ifdef TRACE_L1
	m_compiler.addOptions("-use_fast_math");
#endif
#else
	String kernelName("src/rt/kernels/fermi_persistent_speculative_while_while_inter");
#ifdef TRACE_L1
	m_compiler.addOptions("-use_fast_math -maxrregcount 40");
#endif
#endif
	if(stats != NULL)
	{
		kernelName += "_statistics";
	}
	kernelName += ".cu";
	m_compiler.setSourceFile(kernelName);

	m_module = m_compiler.compile();
	failIfError();

	CUfunction queryKernel = m_module->getKernel("queryConfig");
    if (!queryKernel)
        fail("Config query kernel not found!");

    // Initialize config with default values.
	KernelConfig& kernelConfig = *(KernelConfig*)m_module->getGlobal("g_config").getMutablePtr();
	kernelConfig.bvhLayout             = BVHLayout_Max;
	kernelConfig.blockWidth            = 0;
	kernelConfig.blockHeight           = 0;
	kernelConfig.usePersistentThreads  = 0;

    // Query config.

    m_module->launchKernel(queryKernel, 1, 1);
    kernelConfig = *(const KernelConfig*)m_module->getGlobal("g_config").getPtr();

	CUfunction kernel;
	if(stats != NULL)
		kernel = m_module->getKernel("trace_stats");
	else
		kernel = m_module->getKernel("trace");
	if (!kernel)
		fail("Trace kernel not found!");

	KernelInput& in = *(KernelInput*)m_module->getGlobal("c_in").getMutablePtr();
	// Start the timer
	m_timer.unstart();
	m_timer.start();

	CUdeviceptr nodePtr     = m_bvhData.getCudaPtr();
	Vec2i       nodeOfsA    = Vec2i(0, (S32)m_bvhData.getSize());

#ifdef COMPACT_LAYOUT
	CUdeviceptr triPtr      = m_trisCompactOut.getCudaPtr();
	Vec2i       triOfsA     = Vec2i(0, (S32)m_trisCompactOut.getSize());
	Buffer&     indexBuf    = m_trisIndexOut;
#else
	CUdeviceptr triPtr      = m_trisCompact.getCudaPtr();
	Vec2i       triOfsA     = Vec2i(0, (S32)m_trisCompact.getSize());
	Buffer&     indexBuf    = m_trisIndex;
#endif	

	// Set input.
	// The new version has it via parameters, not const memory
	in.numRays      = rays.getSize();
	in.anyHit       = (rays.getNeedClosestHit() == false);
	in.nodesA       = nodePtr + nodeOfsA.x;
	in.trisA        = triPtr + triOfsA.x;
	in.rays         = rays.getRayBuffer().getCudaPtr();
	in.results      = rays.getResultBuffer().getMutableCudaPtr();
	in.triIndices   = indexBuf.getCudaPtr();

	// Set texture references.
	m_module->setTexRef("t_rays", rays.getRayBuffer(), CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_nodesA", nodePtr + nodeOfsA.x, nodeOfsA.y, CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_trisA", triPtr + triOfsA.x, triOfsA.y, CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_triIndices", indexBuf, CU_AD_FORMAT_SIGNED_INT32, 1);

	// Determine block and grid sizes.
	int desiredWarps = (rays.getSize() + 31) / 32;
	if (kernelConfig.usePersistentThreads != 0)
	{
		*(S32*)m_module->getGlobal("g_warpCounter").getMutablePtr() = 0;
		desiredWarps = 720; // Tesla: 30 SMs * 24 warps, Fermi: 15 SMs * 48 warps
	}

	Vec2i blockSize(kernelConfig.blockWidth, kernelConfig.blockHeight);
	int blockWarps = (blockSize.x * blockSize.y + 31) / 32;
	Vec2i gridSize((desiredWarps + blockWarps - 1) / blockWarps, 1);

	if(stats != NULL)
	{
		m_module->getGlobal("g_NumNodes").clear();
		m_module->getGlobal("g_NumLeaves").clear();
		m_module->getGlobal("g_NumEmptyLeaves").clear();
		m_module->getGlobal("g_NumTris").clear();
		m_module->getGlobal("g_NumFailedTris").clear();
		m_module->getGlobal("g_NumHitTrisOutside").clear();
	}

	// Launch.
	F32 launchTime = m_module->launchKernelTimed(kernel, blockSize, gridSize);

	if(stats != NULL)
	{
		stats->numNodeTests += *(U32*)m_module->getGlobal("g_NumNodes").getPtr();
		stats->numLeavesVisited += *(U32*)m_module->getGlobal("g_NumLeaves").getPtr();
		stats->numEmptyLeavesVisited += *(U32*)m_module->getGlobal("g_NumEmptyLeaves").getPtr();
		stats->numTriangleTests += *(U32*)m_module->getGlobal("g_NumTris").getPtr();
		stats->numFailedTriangleTests += *(U32*)m_module->getGlobal("g_NumFailedTris").getPtr();
		stats->numSuccessTriangleTestsOutside += *(U32*)m_module->getGlobal("g_NumHitTrisOutside").getPtr();
		stats->numRays += rays.getSize();
	}

	m_gpuTime = launchTime;
	m_cpuTime = m_timer.end();

#ifdef TRACE_L1
	// reset options
	m_compiler.clearOptions();
	m_compiler.addOptions("-use_fast_math -Xptxas -dlcm=cg");
#endif

	return launchTime;
}

F32 CudaNoStructTracer::buildKdtree()
{
	// Compile the kernel
	m_kernelFile = "src/rt/kernels/persistent_kdtree.cu";
	m_compiler.setSourceFile(m_kernelFile);
	m_module = m_compiler.compile();
	failIfError();

	prepareDynamicMemory();

#ifdef DEBUG_PPS
	Random rand;
	m_numTris = rand.getU32(1, 1000000);
#endif

	// Set triangle index buffer
	S32* tiout = (S32*)m_trisIndex.getMutablePtr();
#ifdef DEBUG_PPS
	S32* pout = (S32*)m_ppsTris.getMutablePtr();
	S32* clout = (S32*)m_ppsTrisIndex.getMutablePtr();
	S32* sout = (S32*)m_sortTris.getMutablePtr();
#endif
	for(int i=0;i<m_numTris;i++)
	{
#ifndef DEBUG_PPS
		// indices 
		*tiout = i;
		tiout++;
#else
		int rnd = rand.getU32(0, 2);
		//*pout = rnd;
		*clout = rnd;
		*sout = (rnd >= 1);
		//pout++;
		clout++;
		sout++;
#endif
	}

	// Start the timer
	m_timer.unstart();
	m_timer.start();

	// Create the taskData
	m_taskData.resizeDiscard(TASK_SIZE * (sizeof(TaskBVH) + sizeof(int)));
	m_taskData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
//#if SPLIT_TYPE == 3
	m_splitData.resizeDiscard((S64)TASK_SIZE * (S64)sizeof(SplitInfoTri));
//#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
//	m_splitData.resizeDiscard((S64)(TASK_SIZE+1) * (S64)sizeof(SplitArray));
//#endif
	m_splitData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated

	// Node and triangle data
#ifndef INTERLEAVED_LAYOUT
	//S64 kdtreeSize = ((m_numTris*5 * sizeof(CudaKdtreeNode)) + 4096 - 1) & -4096;
	//S64 kdtreeSize = ((m_numTris*10 * sizeof(CudaKdtreeNode)) + 4096 - 1) & -4096;
	//S64 kdtreeSize = ((m_numTris*3 * sizeof(CudaKdtreeNode)) + 4096 - 1) & -4096;
	S64 kdtreeSize = ((m_numTris*20 * sizeof(CudaKdtreeNode)) + 4096 - 1) & -4096;
	m_bvhData.resizeDiscard(kdtreeSize);
	m_bvhData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
	//m_bvhData.clearRange32(0, 0, kdtreeSize); // Mark all tasks as 0 (important for debug)
#ifndef COMPACT_LAYOUT
	m_trisCompactOut.resizeDiscard(m_numTris*10 * 3 * sizeof(Vec4f));
	m_trisIndexOut.resizeDiscard(m_numTris*10 * 3 * sizeof(S32));
#else
#ifdef DUPLICATE_REFERENCES
	m_trisCompactOut.resizeDiscard(m_numTris*8 * (3+1) * sizeof(Vec4f));
	m_trisIndexOut.resizeDiscard(m_numTris*8 * (3+1) * sizeof(S32));
#else
	m_trisCompactOut.resizeDiscard(m_numTris * 3 * sizeof(Vec4f));
	//m_trisIndexOut.resizeDiscard(m_numTris*12 * (1+1) * sizeof(S32));
	m_trisIndexOut.resizeDiscard(m_numTris*20 * (1+1) * sizeof(S32));
	//m_trisIndexOut.resizeDiscard(m_numTris*6 * (1+1) * sizeof(S32));
#endif
#endif
#else
	// TODO: Rewrite this when double headed heap is realized
	S64 kdtreeSize = ((m_numTris*5 * sizeof(CudaKdtreeNode) + m_numTris*10 * 3 * (sizeof(Vec4f)+sizeof(S32))) + 4096 - 1) & -4096;
	//size_t f, t; cuMemGetInfo(&f, &t);
	//S64 kdtreeSize = f & -4096;
	m_bvhData.resizeDiscard(kdtreeSize);
	m_bvhData.clearRange32(0, 0, kdtreeSize); // Mark all tasks as 0 (important for debug)
#endif

	m_gpuTime = buildCudaKdtree();
	m_cpuTime = m_timer.end();

	// Resize to exact memory
	trimKdtreeBuffers();

#ifdef DEBUG_PPS
	exit(0);
#endif

	return m_gpuTime;
}

F32 CudaNoStructTracer::traceBatchKdtree(RayBuffer& rays, RayStats* stats)
{
#ifdef TRACE_L1
	// Set compiler options
	m_compiler.clearOptions();
#endif

	m_compiler.setCachePath("cudacache"); // On the first compilation the cache path becames absolute which kills the second compilation
#ifdef COMPACT_LAYOUT
#ifdef WOOP_TRIANGLES
#ifdef DUPLICATE_REFERENCES
	String kernelName("src/rt/kernels/fermi_kdtree_while_while_childPtr");
	//String kernelName("src/rt/kernels/fermi_kdtree_while_while_shortStack");
#else
	String kernelName("src/rt/kernels/fermi_kdtree_while_while_leafRef");
#endif
#else
#error Undefined kernel
#endif
#else
	String kernelName("src/rt/kernels/fermi_kdtree_while_while");
#endif
#ifdef TRACE_L1
	m_compiler.addOptions("-use_fast_math");
	//m_compiler.addOptions("-use_fast_math -maxrregcount 32");
#endif

	if(stats != NULL)
	{
		kernelName += "_statistics";
	}
	kernelName += ".cu";
	m_compiler.setSourceFile(kernelName);

	m_module = m_compiler.compile();
	failIfError();

	CUfunction queryKernel = m_module->getKernel("queryConfig");
    if (!queryKernel)
        fail("Config query kernel not found!");

    // Initialize config with default values.
	KernelConfig& kernelConfig = *(KernelConfig*)m_module->getGlobal("g_config").getMutablePtr();
	kernelConfig.bvhLayout             = BVHLayout_Max;
	kernelConfig.blockWidth            = 0;
	kernelConfig.blockHeight           = 0;
	kernelConfig.usePersistentThreads  = 0;

    // Query config.

    m_module->launchKernel(queryKernel, 1, 1);
    kernelConfig = *(const KernelConfig*)m_module->getGlobal("g_config").getPtr();

	CUfunction kernel;
	if(stats != NULL)
		kernel = m_module->getKernel("trace_stats");
	else
		kernel = m_module->getKernel("trace");
	if (!kernel)
		fail("Trace kernel not found!");

	KernelInput& in = *(KernelInput*)m_module->getGlobal("c_in").getMutablePtr();
	// Start the timer
	m_timer.unstart();
	m_timer.start();

	CUdeviceptr nodePtr     = m_bvhData.getCudaPtr();
	Vec2i       nodeOfsA    = Vec2i(0, (S32)m_bvhData.getSize());

#ifndef INTERLEAVED_LAYOUT
	CUdeviceptr triPtr      = m_trisCompactOut.getCudaPtr();
	Vec2i       triOfsA     = Vec2i(0, (S32)m_trisCompactOut.getSize());
	Buffer&     indexBuf    = m_trisIndexOut;
#else
	CUdeviceptr triPtr      = m_bvhData.getCudaPtr();
	Vec2i       triOfsA     = Vec2i(0, (S32)m_bvhData.getSize());
	Buffer&     indexBuf    = m_bvhData;
#endif

	// Set input.
	// The new version has it via parameters, not const memory
	in.numRays      = rays.getSize();
	in.anyHit       = (rays.getNeedClosestHit() == false);
	memcpy(&in.bmin, &m_bbox.min, sizeof(float3));
	memcpy(&in.bmax, &m_bbox.max, sizeof(float3));
	in.nodesA       = nodePtr + nodeOfsA.x;
	in.trisA        = triPtr + triOfsA.x;
	in.rays         = rays.getRayBuffer().getCudaPtr();
	in.results      = rays.getResultBuffer().getMutableCudaPtr();
	in.triIndices   = indexBuf.getCudaPtr();

	// Set texture references.
	m_module->setTexRef("t_rays", rays.getRayBuffer(), CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_nodesI", nodePtr + nodeOfsA.x, nodeOfsA.y, CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_trisA", triPtr + triOfsA.x, triOfsA.y, CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_triIndices", indexBuf, CU_AD_FORMAT_SIGNED_INT32, 1);

	// Determine block and grid sizes.
	int desiredWarps = (rays.getSize() + 31) / 32;
	if (kernelConfig.usePersistentThreads != 0)
	{
		*(S32*)m_module->getGlobal("g_warpCounter").getMutablePtr() = 0;
		desiredWarps = 720; // Tesla: 30 SMs * 24 warps, Fermi: 15 SMs * 48 warps
	}

	Vec2i blockSize(kernelConfig.blockWidth, kernelConfig.blockHeight);
	int blockWarps = (blockSize.x * blockSize.y + 31) / 32;
	Vec2i gridSize((desiredWarps + blockWarps - 1) / blockWarps, 1);

	if(stats != NULL)
	{
		m_module->getGlobal("g_NumNodes").clear();
		m_module->getGlobal("g_NumLeaves").clear();
		m_module->getGlobal("g_NumEmptyLeaves").clear();
		m_module->getGlobal("g_NumTris").clear();
		m_module->getGlobal("g_NumFailedTris").clear();
		m_module->getGlobal("g_NumHitTrisOutside").clear();
	}

	// Launch.
	F32 launchTime = m_module->launchKernelTimed(kernel, blockSize, gridSize);

	if(stats != NULL)
	{
		stats->numNodeTests += *(U32*)m_module->getGlobal("g_NumNodes").getPtr();
		stats->numLeavesVisited += *(U32*)m_module->getGlobal("g_NumLeaves").getPtr();
		stats->numEmptyLeavesVisited += *(U32*)m_module->getGlobal("g_NumEmptyLeaves").getPtr();
		stats->numTriangleTests += *(U32*)m_module->getGlobal("g_NumTris").getPtr();
		stats->numFailedTriangleTests += *(U32*)m_module->getGlobal("g_NumFailedTris").getPtr();
		stats->numSuccessTriangleTestsOutside += *(U32*)m_module->getGlobal("g_NumHitTrisOutside").getPtr();
		stats->numRays += rays.getSize();
	}

	m_gpuTime = launchTime;
	m_cpuTime = m_timer.end();

#ifdef TRACE_L1
	// reset options
	m_compiler.clearOptions();
	m_compiler.addOptions("-use_fast_math -Xptxas -dlcm=cg");
#endif

	return launchTime;
}

F32 CudaNoStructTracer::traceOnDemandBVH(RayBuffer& rays, bool rebuild, int numRays)
{
	m_numRays = numRays;

	if(rebuild)
	{
		// Compile the kernel
		m_kernelFile = "src/rt/kernels/persistent_ondemand.cu";
		m_compiler.setSourceFile(m_kernelFile);
		m_module = m_compiler.compile();
		failIfError();

		// Set triangle index buffer
		S32* tiout = (S32*)m_trisIndex.getMutablePtr();
		for(int i=0;i<m_numTris;i++)
		{
			// indices 
			*tiout = i;
			tiout++;
		}
	}

	// Start the timer
	m_timer.unstart();
	m_timer.start();

	if(rebuild)
	{
		// Create the taskData
		m_taskData.resizeDiscard(TASK_SIZE * (sizeof(TaskBVH) + sizeof(int)));
		m_taskData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
		S64 bvhSize = ((m_numTris * sizeof(CudaBVHNode)) + 4096 - 1) & -4096;
		//S64 bvhSize = ((m_numTris/2 * sizeof(CudaBVHNode)) + 4096 - 1) & -4096;
		m_bvhData.resizeDiscard(bvhSize);
		m_bvhData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated

#ifdef COMPACT_LAYOUT
		m_trisCompactOut.resizeDiscard(m_numTris * (3+1) * sizeof(Vec4f));
		m_trisIndexOut.resizeDiscard(m_numTris * (3+1) * sizeof(S32));
#endif

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
		m_splitData.resizeDiscard((S64)(TASK_SIZE+1) * (S64)sizeof(SplitArray));
		m_splitData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#endif
	}

	// Build + trace
	m_gpuTime = traceOnDemandBVHRayBuffer(rays, rebuild);
	m_cpuTime = m_timer.end();

	// Save sizes of buffer so that they can be printed
	if(rebuild)
		saveBufferSizes();

	return m_gpuTime;
}

F32 CudaNoStructTracer::traceOnDemandKdtree(RayBuffer& rays, bool rebuild, int numRays)
{
	m_numRays = numRays;

	if(rebuild)
	{
		// Compile the kernel
		m_kernelFile = "src/rt/kernels/persistent_ondemand_kdtree.cu";
		m_compiler.setSourceFile(m_kernelFile);
		m_module = m_compiler.compile();
		failIfError();

		prepareDynamicMemory();

		// Set triangle index buffer
		S32* tiout = (S32*)m_trisIndex.getMutablePtr();
		for(int i=0;i<m_numTris;i++)
		{
			// indices 
			*tiout = i;
			tiout++;
		}
	}

	// Start the timer
	m_timer.unstart();
	m_timer.start();

	if(rebuild)
	{
		// Create the taskData
		m_taskData.resizeDiscard(TASK_SIZE * (sizeof(TaskBVH) + sizeof(int)));
		m_taskData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
//#if SPLIT_TYPE == 3
		m_splitData.resizeDiscard((S64)TASK_SIZE * (S64)sizeof(SplitInfoTri));
//#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
		//	m_splitData.resizeDiscard((S64)(TASK_SIZE+1) * (S64)sizeof(SplitArray));
//#endif
		m_splitData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated

		// Node and triangle data
#ifndef INTERLEAVED_LAYOUT
		S64 kdtreeSize = ((m_numTris * sizeof(CudaKdtreeNode)) + 4096 - 1) & -4096;
		m_bvhData.resizeDiscard(kdtreeSize);
		m_bvhData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
		//m_bvhData.clearRange32(0, 0, kdtreeSize); // Mark all tasks as 0 (important for debug)
#ifndef COMPACT_LAYOUT
		m_trisCompactOut.resizeDiscard(m_numTris*10 * 3 * sizeof(Vec4f));
		m_trisIndexOut.resizeDiscard(m_numTris*10 * 3 * sizeof(S32));
#else
#ifdef DUPLICATE_REFERENCES
		m_trisCompactOut.resizeDiscard(m_numTris*8 * (3+1) * sizeof(Vec4f));
		m_trisIndexOut.resizeDiscard(m_numTris*8 * (3+1) * sizeof(S32));
#else
		m_trisCompactOut.resizeDiscard(m_numTris * 3 * sizeof(Vec4f));
		m_trisIndexOut.resizeDiscard(m_numTris*7 * (1+1) * sizeof(S32));
#endif
#endif
#else
		// TODO: Rewrite this when double headed heap is realized
		S64 kdtreeSize = ((m_numTris*5 * sizeof(CudaKdtreeNode) + m_numTris*10 * 3 * (sizeof(Vec4f)+sizeof(S32))) + 4096 - 1) & -4096;
		//size_t f, t; cuMemGetInfo(&f, &t);
		//S64 kdtreeSize = f & -4096;
		m_bvhData.resizeDiscard(kdtreeSize);
		m_bvhData.clearRange32(0, 0, kdtreeSize); // Mark all tasks as 0 (important for debug)
#endif
	}

	// Build + trace
	m_gpuTime = traceOnDemandKdtreeRayBuffer(rays, rebuild);
	m_cpuTime = m_timer.end();

	// Save sizes of buffer so that they can be printed
	if(rebuild)
		saveBufferSizes();

	return m_gpuTime;
}

void CudaNoStructTracer::traceOnDemandTrace(RayBuffer& rays, F32& GPUmegakernel, F32& CPUmegakernel, F32& GPUtravKernel, F32& CPUtravKernel, int& buildNodes, RayStats* stats)
{
	m_compiler.setCachePath("cudacache"); // On the first compilation the cache path becames absolute which kills the second compilation
	m_compiler.setSourceFile(m_kernelFile);
	m_module = m_compiler.compile();
	failIfError();

	CUfunction kernel;
	kernel = m_module->getKernel("build");
	if (!kernel)
		fail("Build kernel not found!");

	F32 tTrace, tTraceCPU;
#ifndef ONDEMAND_FULL_BUILD
#if 0
	// Needed for BVH since the data has been erased by module switch
	// Set BVH input.
	KernelInputBVH& inBVH = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
	inBVH.numTris		= m_numTris;
	inBVH.tris         = m_trisCompact.getCudaPtr();
	inBVH.trisIndex    = m_trisIndex.getMutableCudaPtr();
	//inBVH.trisBox      = m_trisBox.getCudaPtr();
	inBVH.ppsTrisBuf   = m_ppsTris.getMutableCudaPtr();
	inBVH.ppsTrisIndex = m_ppsTrisIndex.getMutableCudaPtr();
	inBVH.sortTris   = m_sortTris.getMutableCudaPtr();
#ifdef COMPACT_LAYOUT
	inBVH.trisOut      = m_trisCompactOut.getMutableCudaPtr();
	inBVH.trisIndexOut = m_trisIndexOut.getMutableCudaPtr();
#endif

	// Set traversal input
	CUdeviceptr nodePtr     = m_bvhData.getCudaPtr();
	CUdeviceptr triPtr      = m_trisCompact.getCudaPtr();
	Buffer&     indexBuf    = m_trisIndex;
	Vec2i       nodeOfsA    = Vec2i(0, (S32)m_bvhData.getSize());
	Vec2i       triOfsA     = Vec2i(0, (S32)m_trisCompact.getSize());

	KernelInput& in = *(KernelInput*)m_module->getGlobal("c_in").getMutablePtr();
	m_timer.start();
	in.numRays      = rays.getSize();
	in.anyHit       = (rays.getNeedClosestHit() == false);
	in.nodesA       = nodePtr + nodeOfsA.x;
	in.trisA        = triPtr + triOfsA.x;
	in.rays         = rays.getRayBuffer().getCudaPtr();
	in.results      = rays.getResultBuffer().getMutableCudaPtr();
	in.triIndices   = indexBuf.getCudaPtr();

	m_module->setTexRef("t_rays", rays.getRayBuffer(), CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_nodesA", m_bvhData, CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_trisA", m_trisCompact, CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_triIndices", m_trisIndex, CU_AD_FORMAT_SIGNED_INT32, 1);
#endif

	int numWarpsPerBlock = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numWarpsPerBlock");
	int numBlocksPerSM = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numBlockPerSM");
	Vec2i blockSize(WARP_SIZE, numWarpsPerBlock); // threadIdx.x must equal the thread lane in warp
	int gridSizeX = NUM_SM*numBlocksPerSM;
	Vec2i gridSize(gridSizeX, 1); // Number of SMs * Number of blocks?

	// Run the kernel as long as the traversal order does not change
	TaskStackBVH& tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getMutablePtr();
	int oldNodes;
	int i = 0;
	do
	{
		oldNodes = tasks.numNodes;
		
		m_timer.unstart();
		m_timer.start();
		// Launch trace until traversal path convergence.
#if 0
		// Needed for BVH since the task has been erased by module switch
		tasks.header     = (int*)m_taskData.getMutableCudaPtr();
		tasks.tasks      = (TaskBVH*)m_taskData.getMutableCudaPtr(TASK_SIZE * sizeof(int));
		tasks.nodeTop    = 2;
#endif

		tasks.warpCounter = rays.getSize();
		tasks.unfinished = -NUM_WARPS;
		tasks.launchFlag = 1;

		// Launch.
		tTrace = m_module->launchKernelTimed(kernel, blockSize, gridSize);
		tTraceCPU = m_timer.end();
		TaskStackBVH& tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getMutablePtr();
		buildNodes += tasks.numNodes - oldNodes;

		//printf("Verify run %d in %f (%d -> %d)\n", i, tTrace, oldNodes, tasks.numNodes);
		i++;
	} while(oldNodes != tasks.numNodes);

	// Launch just trace.
	/*tasks.warpCounter = rays.getSize();
	tasks.unfinished = -NUM_WARPS;
	tasks.launchFlag = 2;
	tTrace = m_module->launchKernelTimed(kernel, blockSize, gridSize);*/
	
	GPUmegakernel += tTrace; // Save the final traversal time inside the megakernel
	CPUmegakernel += tTraceCPU;
#endif

	//cout << "Verify trace in " << tTrace << "s" << "\n";

	/*TaskStackBVH& tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getMutablePtr();
	tasks.warpCounter = rays.getSize();
	tasks.unfinished = -NUM_WARPS;

	for(int i = 0; i < rays.getSize(); i++)
	{
		// Set ray result buffer
		RayResult& res = rays.getMutableResultForSlot(i);
		res.clear();

		// Set ray tmax
		Ray& ray = rays.getMutableRayForSlot(i);
		ray.tmax = defTmax;
	}
	rays.getResultBuffer().getMutableCudaPtr();
	float tTrace = m_module->launchKernelTimed(kernel, blockSize, gridSize);
	//cout << "Verify trace in " << tTrace << "s" << "\n";
	*/

	if(m_kernelFile.endsWith("kdtree.cu"))
		tTrace = traceBatchKdtree(rays, stats);
	else
		tTrace = traceBatchBVH(rays, stats);
	//tTrace = -1.f;
	tTraceCPU = getCPUTime();
	m_compiler.setCachePath("cudacache"); // On the first compilation the cache path becames absolute which kills the second compilation
	m_compiler.setSourceFile(m_kernelFile);
	m_module = m_compiler.compile();
	
	//printf("Verify kernel %d in %f\n", i, tTrace);

	GPUtravKernel += tTrace;
	CPUtravKernel += tTraceCPU;
}

F32 CudaNoStructTracer::test()
{
	float minTime = FLT_MAX;
	float sumTime = 0.f;
	const int numRepeats = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numRepeats");

//  Create the taskData
	m_taskData.resizeDiscard(TASK_SIZE * (sizeof(TaskBVH) + sizeof(int)));

	const int width = Environment::GetSingleton()->GetInt("Aila.width");
	const int height = Environment::GetSingleton()->GetInt("Aila.height");

	cout << "Prefix scan for problem size of " << width << "x" << height << " = " << width*height << "\n";

#ifdef TEST_TASKS
	cout << "Testing task pool" << "\n";
#else
	cout << "Testing thrust" << "\n";
#endif

	for(int i = 0; i < numRepeats; i++)
	{
//#ifdef TEST_TASKS
		float t = testSort(width*height);
		//float t = testSort(m_numTris);
/*#else
		float t = testThrustScan(width*height);
#endif*/

		printf("Run %d sort in %fs\n", i, t);
		minTime = min(t, minTime);
		sumTime += t;
	}

	printf("Minimum time from %d runs = %fs\n", numRepeats, minTime);
	printf("Average time from %d runs = %fs\n", numRepeats, sumTime/numRepeats);

	//size_t f, t; cuMemGetInfo(&f, &t); fprintf(stdout,"CUDA Memory allocated:%dMB free:%dMB\n",(t-f)/1048576,f/1048576);
	return minTime;
}

void CudaNoStructTracer::updateConstants()
{
	RtEnvironment& cudaEnv = *(RtEnvironment*)m_module->getGlobal("c_env").getMutablePtr();

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.maxDepth", cudaEnv.optMaxDepth);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.planeSelectionOverhead", cudaEnv.optPlaneSelectionOverhead);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.ci", cudaEnv.optCi);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.ct", cudaEnv.optCt);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.ctr", cudaEnv.optCtr);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.ctt", cudaEnv.optCtt);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.triangleBasedWeight", cudaEnv.optTriangleBasedWeight);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.rayBasedWeight", cudaEnv.optRayBasedWeight);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.axisAlignedWeight", cudaEnv.optAxisAlignedWeight);

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.cutOffDepth", cudaEnv.optCutOffDepth);
	m_cutOffDepth = cudaEnv.optCutOffDepth;

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.rayLimit", cudaEnv.rayLimit);

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.triLimit", cudaEnv.triLimit);
	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.triMaxLimit", cudaEnv.triMaxLimit);

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.popCount", cudaEnv.popCount);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.granularity", cudaEnv.granularity);

	Environment::GetSingleton()->GetFloatValue("SubdivisionRayCaster.failRq", cudaEnv.failRq);

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.failureCount", cudaEnv.failureCount);

	int siblingLimit;
	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.siblingLimit", siblingLimit);
	cudaEnv.siblingLimit = siblingLimit / WARP_SIZE;

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.childLimit", cudaEnv.childLimit);

	Environment::GetSingleton()->GetIntValue("SubdivisionRayCaster.subtreeLimit", cudaEnv.subtreeLimit);

	cudaEnv.subdivThreshold = (m_bbox.SurfaceArea() / (float)m_numRays) * ((float)cudaEnv.optCt/10.0f);

	cudaEnv.epsilon = m_epsilon;
	//cudaEnv.epsilon = 0.f;
}

//------------------------------------------------------------------------

int CudaNoStructTracer::warpSubtasks(int threads)
{
	//return (threads + WARP_SIZE - 1) / WARP_SIZE;
	return max((threads + WARP_SIZE - 1) / WARP_SIZE, 1); // Do not create empty tasks - at least on warp gets to clean this task
}

//------------------------------------------------------------------------

int CudaNoStructTracer::floatToOrderedInt(float floatVal)
{
	int intVal = *((int*)&floatVal);
	return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

/*unsigned int CudaNoStructTracer::floatToOrderedInt(float floatVal)
{
	unsigned int f = *((unsigned int*)&floatVal);
	unsigned int mask = -(int)(f >> 31) | 0x80000000;
	return f ^ mask;
}*/

//------------------------------------------------------------------------

void CudaNoStructTracer::allocateSnapshots(Buffer &snapData)
{
	// Prepare snapshot memory
#ifdef SNAPSHOT_POOL
	snapData.resizeDiscard(sizeof(PoolInfo)*SNAPSHOT_POOL);
	PoolInfo* &snapshots = *(PoolInfo**)m_module->getGlobal("g_snapshots").getMutablePtr();
	snapshots = (PoolInfo*)snapData.getMutableCudaPtr();
	snapData.clearRange32(0, 0, SNAPSHOT_POOL*sizeof(PoolInfo)); // Mark all tasks as empty (important for debug)
#endif
#ifdef SNAPSHOT_WARP
	snapData.resizeDiscard(sizeof(WarpInfo)*SNAPSHOT_WARP*NUM_WARPS);
	WarpInfo* &snapshots = *(WarpInfo**)m_module->getGlobal("g_snapshots").getMutablePtr();
	snapshots = (WarpInfo*)snapData.getMutableCudaPtr();
	snapData.clearRange32(0, 0, SNAPSHOT_WARP*NUM_WARPS*sizeof(WarpInfo)); // Mark all tasks as empty (important for debug)
#endif
}

//------------------------------------------------------------------------

void CudaNoStructTracer::printSnapshots(Buffer &snapData)
{
#ifdef SNAPSHOT_POOL
	PoolInfo* snapshots = (PoolInfo*)snapData.getPtr();

	if(snapshots[SNAPSHOT_POOL-1].pool != 0) // Full
		printf("\aSnapshot memory full!\n");

	long long int clockMin = snapshots[0].clockStart;
	long long int clockMax = 0;
	for(int i = 0; i < SNAPSHOT_POOL; i++)
	{
		if(snapshots[i].pool == 0)
		{
			clockMax = snapshots[i-1].clockEnd;
			break;
		}
	}

	ofstream snapfile("plots\\pool\\activity.dat");
	snapfile << "Snap#\tpool\t#tasks\t#active\t#chunks\tdepth\tclocks" << "\n";
	for(int i = 0; i < SNAPSHOT_POOL; i++)
	{
		if(snapshots[i].pool == 0)
			break;

		snapfile << i << "\t" << snapshots[i].pool << "\t" << snapshots[i].tasks << "\t" << snapshots[i].active << "\t" << snapshots[i].chunks << "\t" << snapshots[i].depth
			<< "\t" << snapshots[i].clockEnd - snapshots[i].clockStart << "\n";
	}
	snapfile.close();

	snapfile.open("plots\\pool\\activity_clockCor.dat");
	snapfile << "Snap#\tpool\t#tasks\t#active\t#chunks\tdepth\tclocks" << "\n";
	for(int i = 0; i < SNAPSHOT_POOL; i++)
	{
		if(snapshots[i].pool == 0)
			break;

		snapfile << (float)((long double)(snapshots[i].clockEnd - clockMin) / (long double)(clockMax - clockMin)) << "\t" << snapshots[i].pool << "\t" << snapshots[i].tasks << "\t"
			<< snapshots[i].active << "\t" << snapshots[i].chunks << "\t" << snapshots[i].depth	<< "\t" << snapshots[i].clockEnd - snapshots[i].clockStart << "\n";
	}

	snapfile.close();
#endif
#ifdef SNAPSHOT_WARP
	WarpInfo* snapshots = (WarpInfo*)snapData.getPtr();

	for(int w = 0; w < NUM_WARPS; w++)
	{
		if(snapshots[SNAPSHOT_WARP-1].reads != 0) // Full
			printf("\aSnapshot memory full for warp %d!\n", w);

		ostringstream filename;
		filename.fill('0');
		filename << "plots\\warps\\warp" << setw(3) << w << ".dat";
		//cout << filename.str() << "\n";
		ofstream snapfile(filename.str());

		snapfile << "Snap#\t#reads\t#rays\t#tris\ttype(leaf=8)\t#chunks\tpopCount\tdepth\tcDequeue\tcCompute\tstackTop\ttaskIdx" << "\n";
		for(int i = 0; i < SNAPSHOT_WARP; i++)
		{
			if(snapshots[i].reads == 0)
				break;

			if(snapshots[i].clockDequeue < snapshots[i].clockSearch || snapshots[i].clockFinished < snapshots[i].clockDequeue)
				cout << "Error timer for warp " << w << "\n";

			snapfile << i << "\t" << snapshots[i].reads << "\t" << snapshots[i].rays << "\t" << snapshots[i].tris << "\t" << snapshots[i].type << "\t"
				<< snapshots[i].chunks << "\t" << snapshots[i].popCount << "\t" << snapshots[i].depth << "\t" << (snapshots[i].clockDequeue - snapshots[i].clockSearch) << "\t"
				<< (snapshots[i].clockFinished - snapshots[i].clockDequeue) << "\t" << snapshots[i].stackTop << "\t" << snapshots[i].idx << "\n";
		}

		snapfile.close();
		snapshots += SNAPSHOT_WARP; // Next warp
	}
#endif
}

//------------------------------------------------------------------------

void CudaNoStructTracer::initPool(int numRays, Buffer* rayBuffer, Buffer* nodeBuffer)
{
	// Prepare the task data
	updateConstants();
#if PARALLELISM_TEST >= 0
	int& numActive = *(int*)m_module->getGlobal("g_numActive").getMutablePtr();
	numActive = 1;
#endif

#ifndef MALLOC_SCRATCHPAD
	// Set PPS buffers
	m_ppsTris.resizeDiscard(sizeof(int)*m_numTris);
	m_ppsTrisIndex.resizeDiscard(sizeof(int)*m_numTris);
	m_sortTris.resizeDiscard(sizeof(int)*m_numTris);

	if(numRays > 0)
	{
		m_ppsRays.resizeDiscard(sizeof(int)*numRays);
		m_ppsRaysIndex.resizeDiscard(sizeof(int)*numRays);
		m_sortRays.resizeDiscard(sizeof(int)*numRays);
	}
#endif

#if defined(SNAPSHOT_POOL) || defined(SNAPSHOT_WARP)
	// Prepare snapshot memory
	Buffer snapData;
	allocateSnapshots(snapData);
#endif

	// Set all headers empty
#ifdef TEST_TASKS
	m_taskData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#ifdef BENCHMARK
	m_taskData.clearRange32(0, TaskHeader_Empty, TASK_SIZE * sizeof(int)); // Mark all tasks as empty
#else
	m_taskData.clearRange32(0, TaskHeader_Empty, TASK_SIZE * (sizeof(int)+sizeof(Task))); // Mark all tasks as empty (important for debug)
#endif
#endif

	// Increase printf output size so that more can fit
	//cuCtxSetLimit(CU_LIMIT_PRINTF_FIFO_SIZE, 536870912);

	/*cuCtxSetCacheConfig(CU_FUNC_CACHE_PREFER_SHARED); // Driver does not seem to care and preffers L1
	cuFuncSetCacheConfig(kernel, CU_FUNC_CACHE_PREFER_SHARED);
	CUfunc_cache test;
	cuCtxGetCacheConfig(&test);
	if(test != CU_FUNC_CACHE_PREFER_SHARED)
		printf("Error\n");*/

	// Set texture references.
	if(rayBuffer != NULL)
	{
		m_module->setTexRef("t_rays", *rayBuffer, CU_AD_FORMAT_FLOAT, 4);
	}
	if(nodeBuffer != NULL)
	{
		m_module->setTexRef("t_nodesA", *nodeBuffer, CU_AD_FORMAT_FLOAT, 4);
	}
	m_module->setTexRef("t_trisA", m_trisCompact, CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_triIndices", m_trisIndex, CU_AD_FORMAT_SIGNED_INT32, 1);

/*#ifdef COMPACT_LAYOUT
	if(numRays == 0)
	{
		m_module->setTexRef("t_trisAOut", m_trisCompactOut, CU_AD_FORMAT_FLOAT, 4);
		m_module->setTexRef("t_triIndicesOut", m_trisIndexOut, CU_AD_FORMAT_SIGNED_INT32, 1);
	}
#endif*/
}

//------------------------------------------------------------------------

void CudaNoStructTracer::deinitPool(int numRays)
{
	m_ppsTris.reset();
	m_ppsTrisIndex.reset();
	m_sortTris.reset();

	if(numRays > 0)
	{
		m_ppsRays.reset();
		m_ppsRaysIndex.reset();
		m_sortRays.reset();
	}
}

//------------------------------------------------------------------------

void CudaNoStructTracer::printPoolHeader(TaskStackBase* tasks, int* header, int numWarps, FW::String state)
{
#if PARALLELISM_TEST >= 0
	numActive = *(int*)m_module->getGlobal("g_numActive").getPtr();
	printf("Active: %d\n", numActive);
#endif


#if defined(SNAPSHOT_POOL) || defined(SNAPSHOT_WARP)
	printSnapshots(snapData);
#endif

#ifdef DEBUG_INFO
	Debug << "\nPRINTING DEBUG_INFO STATISTICS" << "\n\n";
#else
	Debug << "\nPRINTING STATISTICS" << "\n\n";
#endif

	float4* debugData = (float4*)m_debug.getPtr();
	float minAll[4] = {MAX_FLOAT, MAX_FLOAT, MAX_FLOAT, MAX_FLOAT};
	float maxAll[4] = {0, 0, 0, 0};
	float sumAll[4] = {0, 0, 0, 0};
	int countDead = 0;
	Debug << "Warp No. cnt_task_queues Avg. #Reads Max #Reads #Restarts" << "\n";
	for(int i = 0; i < numWarps; i++)
	{
		Debug << "Warp " << i << ": (" << debugData[i].x << ", " << debugData[i].y << ", " << debugData[i].z << ", " << debugData[i].w << ")" << "\n";

		//fabs is because we do not care whether the warp stopped prematurely or not
		minAll[0] = min(fabs(debugData[i].x), minAll[0]);
		minAll[1] = min(fabs(debugData[i].y), minAll[1]);
		minAll[2] = min(fabs(debugData[i].z), minAll[2]);
		minAll[3] = min(fabs(debugData[i].w), minAll[3]);

		maxAll[0] = max(fabs(debugData[i].x), maxAll[0]);
		maxAll[1] = max(fabs(debugData[i].y), maxAll[1]);
		maxAll[2] = max(fabs(debugData[i].z), maxAll[2]);
		maxAll[3] = max(fabs(debugData[i].w), maxAll[3]);

		sumAll[0] += fabs(debugData[i].x);
		sumAll[1] += fabs(debugData[i].y);
		sumAll[2] += fabs(debugData[i].z);
		sumAll[3] += fabs(debugData[i].w);

		if(debugData[i].x < 0)
			countDead++;
	}
	Debug << "Dead=" << countDead << " / All=" << numWarps << " = " << (float)countDead/(float)numWarps << "\n";
	Debug << "Min: " << minAll[0] << ", " << minAll[1] << ", " << minAll[2] << ", " << minAll[3] << "\n";
	Debug << "Max: " << maxAll[0] << ", " << maxAll[1] << ", " << maxAll[2] << ", " << maxAll[3] << "\n";
	Debug << "Sum: " << sumAll[0] << ", " << sumAll[1] << ", " << sumAll[2] << ", " << sumAll[3] << "\n";
	Debug << "Avg: " << sumAll[0]/numWarps << ", " << sumAll[1]/numWarps << ", " << sumAll[2]/numWarps << ", " << sumAll[3]/numWarps << "\n\n" << "\n";
	Debug << "cnt_task_queues per object = " << sumAll[0]/(float)m_numTris << "\n";

	Debug << "Pool" << "\n";
	Debug << "Top = " << tasks->top << "; Bottom = " << tasks->bottom << "; Unfinished = " << tasks->unfinished << "; Size = " << tasks->sizePool << "; ";
	Debug << state.getPtr() << "\n";
	Debug << "ActiveTop = " << tasks->activeTop << "; Active = ";
	for(int i = 0; i < ACTIVE_MAX+1; i++)
		Debug << tasks->active[i] << " ";
	Debug << "\n" << "\n";
	Debug << "EmptyTop = " << tasks->emptyTop << "; EmptyBottom = " << tasks->emptyBottom  << "\nEmpty\n";
	for(int i = 0; i < EMPTY_MAX+1; i++)
	{
		if(i % 50 == 0)
			Debug << "\n";
		else
			Debug << " ";
		Debug << tasks->empty[i];
	}

	/*for(int i = 0; i < EMPTY_MAX+1; i++)
		Debug << tasks->empty[i] << " ";*/
	Debug << "\n" << "\n";

	int emptyItems = 0;
	int bellowEmpty = 0;
	Debug << "Header" << "\n";
	for(int i = 0; i < TASK_SIZE; i++)
	{
		if(i % 50 == 0)
			Debug << "\n";
		else
			Debug << " ";
		if(header[i] != TaskHeader_Empty)
		{
			Debug << header[i];
		}
		else
		{
			Debug << TaskHeader_Active;
			if(i < tasks->top)
				emptyItems++;
		}

		if(header[i] < TaskHeader_Empty)
			bellowEmpty++;
	}

	Debug << "\n\nEmptyItems = " << emptyItems << "\n";
	Debug << "BellowEmpty = " << bellowEmpty << "\n";
}

//------------------------------------------------------------------------

void CudaNoStructTracer::printPool(TaskStackBVH &tasks, int numWarps)
{
#ifdef LEAF_HISTOGRAM
	printf("Leaf histogram\n");
	unsigned int leafSum = 0;
	unsigned int triSum = 0;
	for(S32 i = 0; i <= Environment::GetSingleton()->GetInt("SubdivisionRayCaster.triLimit"); i++)
	{
		printf("%d: %d\n", i, tasks.leafHist[i]);
		leafSum += tasks.leafHist[i];
		triSum += i*tasks.leafHist[i];
	}
	printf("Leafs total %d, average leaf %.2f\n", leafSum, (float)triSum/(float)leafSum);
#endif

	int* header = (int*)m_taskData.getPtr();
	FW::String state = sprintf("BVH Top = %d; Tri Top = %d; Warp counter = %d; ", tasks.nodeTop, tasks.triTop, tasks.warpCounter);
#ifdef BVH_COUNT_NODES
	state.appendf("Number of inner nodes = %d; Number of leaves = %d; Sorted tris = %d; ", tasks.numNodes, tasks.numLeaves, tasks.numSortedTris);
#endif
	printPoolHeader(&tasks, header, numWarps, state);

	Debug << "\n\nTasks" << "\n";
	TaskBVH* task = (TaskBVH*)m_taskData.getPtr(TASK_SIZE*sizeof(int));
	int stackMax = 0;
	int maxDepth = 0;
	int syncCount = 0;
	int maxTaskId = -1;
	long double sumTris = 0;
	long double maxTris = 0;

	int sortTasks = 0;
	long double cntSortTris = 0;

	int subFailed = 0;

#ifdef DEBUG_INFO
	char terminatedNames[TerminatedBy_Max][255] = {
		"None", "Depth","TotalLimit","OverheadLimit","Cost","FailureCounter"
	};

	int terminatedBy[TerminatedBy_Max];
	memset(&terminatedBy,0,sizeof(int)*TerminatedBy_Max);
#endif

	for(int i = 0; i < TASK_SIZE; i++)
	{
		if(task[i].nodeIdx != TaskHeader_Empty || task[i].parentIdx != TaskHeader_Empty)
		{
#ifdef DEBUG_INFO
			_ASSERT(task[i].terminatedBy >= 0 && task[i].terminatedBy < TerminatedBy_Max);
			terminatedBy[ task[i].terminatedBy ]++;
#endif

			Debug << "Task " << i << "\n";
			Debug << "Header: " << header[i] << "\n";
			Debug << "Unfinished: " << task[i].unfinished << "\n";
			Debug << "Type: " << task[i].type << "\n";
			Debug << "TriStart: " << task[i].triStart << "\n";
			Debug << "TriLeft: " << task[i].triLeft << "\n";
			Debug << "TriRight: " << task[i].triRight << "\n";
			Debug << "TriEnd: " << task[i].triEnd << "\n";
			Debug << "ParentIdx: " << task[i].parentIdx << "\n";
			Debug << "NodeIdx: " << task[i].nodeIdx << "\n";
			Debug << "TaskID: " << task[i].taskID << "\n";
			Debug << "Split: (" << task[i].splitPlane.x << ", " << task[i].splitPlane.y << ", " << task[i].splitPlane.z << ", " << task[i].splitPlane.w << ")\n";
			Debug << "Box: (" << task[i].bbox.m_mn.x << ", " << task[i].bbox.m_mn.y << ", " << task[i].bbox.m_mn.z << ") - ("
				<< task[i].bbox.m_mx.x << ", " << task[i].bbox.m_mx.y << ", " << task[i].bbox.m_mx.z << ")\n";
			//Debug << "BoxLeft: (" << task[i].bboxLeft.m_mn.x << ", " << task[i].bboxLeft.m_mn.y << ", " << task[i].bboxLeft.m_mn.z << ") - ("
			//	<< task[i].bboxLeft.m_mx.x << ", " << task[i].bboxLeft.m_mx.y << ", " << task[i].bboxLeft.m_mx.z << ")\n";
			//Debug << "BoxRight: (" << task[i].bboxRight.m_mn.x << ", " << task[i].bboxRight.m_mn.y << ", " << task[i].bboxRight.m_mn.z << ") - ("
			//	<< task[i].bboxRight.m_mx.x << ", " << task[i].bboxRight.m_mx.y << ", " << task[i].bboxRight.m_mx.z << ")\n";
			Debug << "Axis: " << task[i].axis << "\n";
			Debug << "Depth: " << task[i].depth << "\n";
			Debug << "Step: " << task[i].step << "\n";
#ifdef DEBUG_INFO
			//Debug << "Step: " << task[i].step << "\n";
			//Debug << "Lock: " << task[i].lock << "\n";
#ifdef MALLOC_SCRATCHPAD
			Debug << "SubFailure: " << task[i].subFailureCounter << "\n";
#endif
			Debug << "GMEMSync: " << task[i].sync << "\n";
			Debug << "Parent: " << task[i].parent << "\n";
#endif

#ifdef DEBUG_INFO
			Debug << "TerminatedBy: " << task[i].terminatedBy << "\n";
#endif
			if(task[i].terminatedBy != TerminatedBy_None)
				Debug << "Triangles: " << task[i].triEnd - task[i].triStart << "\n";

			Debug << "\n";
			stackMax = i;

			if(header[i] > (int)0xFF800000) // Not waiting
			{
#ifdef CUTOFF_DEPTH
				if(task[i].depth == m_cutOffDepth)
				{
#endif
					long double tris = task[i].triEnd - task[i].triStart;
					if(task[i].terminatedBy != TerminatedBy_None)
					{
						if(tris > maxTris)
						{
							maxTris = tris;
							maxTaskId = i;
						}
						sumTris += tris;
					}
					sortTasks++;
					cntSortTris += tris;
#ifdef CUTOFF_DEPTH
				}
#endif

#ifdef DEBUG_INFO
				maxDepth = max(task[i].depth, maxDepth);
				syncCount += task[i].sync;
#endif
			}
		}
	}

	if(stackMax == TASK_SIZE-1)
		printf("\aIncomplete result!\n");
#ifdef CUTOFF_DEPTH
	Debug << "\n\nStatistics for cutoff depth " << m_cutOffDepth << "\n\n";
#else
	Debug << "\n\n";
#endif

#ifdef DEBUG_INFO
	Debug << "Avg naive task height (tris) = " << sumTris/(long double)sortTasks << "\n";
	Debug << "Max naive task height (tris) = " << maxTris << ", taskId: " << maxTaskId << "\n";
	Debug << "Cnt sorted operations = " << sortTasks << "\n";
	double cntTrisLog2Tris = (double(m_numTris) * (double)(logf(m_numTris)/logf(2.0f)));
	Debug << "Cnt sorted triangles = " << cntSortTris << "\n";	
	Debug << "Cnt sorted triangles/(N log N), N=#tris = " << cntSortTris/cntTrisLog2Tris << "\n";
	Debug << "\n";
	Debug << "Max task depth = " << maxDepth << "\n";
	Debug << "Cnt gmem synchronizations: " << syncCount << "\n";
	Debug << "Leafs failed to subdivide = " << subFailed << " (*3) => total useless tasks " << subFailed * 3 << "\n";
	Debug << "Terminated by:" << "\n";
	for(int i = 0; i < TerminatedBy_Max; i++)
	{
		Debug << terminatedNames[i] << ": " << terminatedBy[i] << "\n";
	}
#endif

	Debug << "max_queue_length = " << stackMax << "\n\n" << "\n";
}

//------------------------------------------------------------------------

void CudaNoStructTracer::printPool(TaskStack &tasks, int numWarps)
{
	tasks = *(TaskStack*)m_module->getGlobal("g_taskStack").getPtr();
	int* header = (int*)m_taskData.getPtr();
	printPoolHeader(&tasks, header, numWarps, FW::sprintf(""));

	Debug << "\n\nTasks" << "\n";
	Task* task = (Task*)m_taskData.getPtr(TASK_SIZE*sizeof(int));
	int stackMax = 0;
	int maxDepth = 0;
	int syncCount = 0;
	int maxTaskId = -1;
	int rayIssues = 0;
	int triIssues = 0;
	long double sumRays = 0;
	long double maxRays = 0;
	long double sumTris = 0;
	long double maxTris = 0;
	
	int isectTasks = 0;
	long double cntIsect = 0;
	long double maxIsect = 0;
	long double clippedIsect = 0;

	int sortTasks = 0;
	long double cntSortRays = 0;
	long double cntClippedRays = 0;
	long double cntSortTris = 0;

	int subFailed = 0;
	int failureCount = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.failureCount");

#ifdef DEBUG_INFO
	char terminatedNames[TerminatedBy_Max][255] = {
		"None", "Depth","TotalLimit","OverheadLimit","Cost","FailureCounter"
	};

	int terminatedBy[TerminatedBy_Max];
	memset(&terminatedBy,0,sizeof(int)*TerminatedBy_Max);
#endif

	for(int i = 0; i < TASK_SIZE; i++)
	{
		if(task[i].depend1 != TaskHeader_Empty || task[i].depend2 != TaskHeader_Empty)
		{
#ifdef DEBUG_INFO
			_ASSERT(task[i].terminatedBy >= 0 && task[i].terminatedBy < TerminatedBy_Max);
			terminatedBy[ task[i].terminatedBy ]++;
#endif

			Debug << "Task " << i << "\n";
			Debug << "Header: " << header[i] << "\n";
			Debug << "Unfinished: " << task[i].unfinished << "\n";
			Debug << "Type: " << task[i].type << "\n";
			Debug << "RayStart: " << task[i].rayStart << "\n";
			Debug << "RayEnd: " << task[i].rayEnd << "\n";
			if(task[i].type != TaskType_Intersect) // Splitted
			{
				Debug << "RayLeft: " << task[i].rayLeft << "\n";
				Debug << "RayRight: " << task[i].rayRight << "\n";
				Debug << "RayActive: " << task[i].rayActive << "\n";
			}
#ifdef CLIP_INTERSECT
			if(task[i].type == TaskType_Intersect)
			Debug << "RayActive: " << task[i].rayActive << "\n";
#endif
			Debug << "TriStart: " << task[i].triStart << "\n";
			Debug << "TriEnd: " << task[i].triEnd << "\n";
			if(task[i].type != TaskType_Intersect) // Splitted
			{
				//Debug << "BestOrder: " << task[i].bestOrder << "\n";
				Debug << "TriLeft: " << task[i].triLeft << "\n";
				Debug << "TriRight: " << task[i].triRight << "\n";
			}
			Debug << "Depend1: " << task[i].depend1 << "\n";
			Debug << "Depend2: " << task[i].depend2 << "\n";
			if(task[i].type != TaskType_Intersect) // Splitted
			{
				Debug << "Split: (" << task[i].splitPlane.x << ", " << task[i].splitPlane.y << ", " << task[i].splitPlane.z << ", " << task[i].splitPlane.w << ")\n";
			}
			Debug << "Box: (" << task[i].bbox.m_mn.x << ", " << task[i].bbox.m_mn.y << ", " << task[i].bbox.m_mn.z << ") - ("
				<< task[i].bbox.m_mx.x << ", " << task[i].bbox.m_mx.y << ", " << task[i].bbox.m_mx.z << ")\n";
			//Debug << "BoxLeft: (" << task[i].bboxLeft.m_mn.x << ", " << task[i].bboxLeft.m_mn.y << ", " << task[i].bboxLeft.m_mn.z << ") - ("
			//	<< task[i].bboxLeft.m_mx.x << ", " << task[i].bboxLeft.m_mx.y << ", " << task[i].bboxLeft.m_mx.z << ")\n";
			//Debug << "BoxMiddle (" << task[i].bboxMiddle.m_mn.x << ", " << task[i].bboxMiddle.m_mn.y << ", " << task[i].bboxMiddle.m_mn.z << ") - ("
			//	<< task[i].bboxMiddle.m_mx.x << ", " << task[i].bboxMiddle.m_mx.y << ", " << task[i].bboxMiddle.m_mx.z << ")\n";
			//Debug << "BoxRight: (" << task[i].bboxRight.m_mn.x << ", " << task[i].bboxRight.m_mn.y << ", " << task[i].bboxRight.m_mn.z << ") - ("
			//	<< task[i].bboxRight.m_mx.x << ", " << task[i].bboxRight.m_mx.y << ", " << task[i].bboxRight.m_mx.z << ")\n";
			Debug << "Depth: " << task[i].depth << "\n";
#ifdef DEBUG_INFO
			//Debug << "Step: " << task[i].step << "\n";
			//Debug << "Lock: " << task[i].lock << "\n";
			Debug << "SubFailure: " << task[i].subFailureCounter << "\n";
			Debug << "GMEMSync: " << task[i].sync << "\n";
			Debug << "TaskID: " << task[i].taskID << "\n";
			Debug << "Parent: " << task[i].parent << "\n";
#if AABB_TYPE < 3
			if(task[i].type == TaskType_AABB_Max)
#elif AABB_TYPE == 3
			if(task[i].type == TaskType_AABB)
#endif
			{
				Debug << "SubtaskIdx: " << task[i].subtaskIdx << "\n";
				Debug << "Clipped rays: " << task[i].rayEnd-task[i].rayActive << "\n";
			}
#endif

#ifdef CUTOFF_DEPTH
			if(task[i].depth == m_cutOffDepth)
#endif
			if(task[i].type == TaskType_Intersect)
			{
#ifdef CLIP_INTERSECT
				long double locRays = task[i].rayActive - task[i].rayStart;
#else
				long double locRays = task[i].rayEnd - task[i].rayStart;
#endif
				long double locTris = task[i].triEnd - task[i].triStart; 
				Debug << "Intersections: " << locRays * locTris << "\n";
				//if(locRays > 1000 || locTris > 1000 )
				{
					if( locRays < sqrt((double)locTris) )
						triIssues++;
					if( locTris < sqrt((double)locRays) )
						rayIssues++;
				}

				Debug << "ClippedIntersections: " << task[i].clippedRays * locTris << "\n";
				clippedIsect += task[i].clippedRays * locTris;
			}

#ifdef ONE_WARP_RUN
			//Debug << "Clock: " << task[i].clockEnd - task[i].clockStart << "\n";
			Debug << "Clock: " << task[i].clockEnd << "\n";
#endif
#ifdef DEBUG_INFO
			Debug << "TerminatedBy: " << task[i].terminatedBy << "\n";
#endif
			
			Debug << "\n";
			stackMax = i;

#ifdef CUTOFF_DEPTH
			if(task[i].depth == m_cutOffDepth)
			{
#endif

#ifdef CLIP_INTERSECT
			long double rays = task[i].rayActive - task[i].rayStart;
#else
			long double rays = task[i].rayEnd - task[i].rayStart;
#endif
			
			long double tris = task[i].triEnd - task[i].triStart;
			if(task[i].type == TaskType_Intersect)
			{
				isectTasks++;
				cntIsect += rays*tris;
				maxIsect = max<long double>(rays*tris, maxIsect);
				if(maxIsect==(rays*tris)) maxTaskId = i;
				sumRays += rays;
				maxRays = max<long double>(rays, maxRays);
				sumTris += tris;
				maxTris = max<long double>(tris, maxTris);
				if(task[i].subFailureCounter > failureCount)
					subFailed++;
			}
#if AABB_TYPE < 3
			if(task[i].type == TaskType_AABB_Max)
#elif AABB_TYPE == 3
			if(task[i].type == TaskType_AABB)
#endif
			{
				sortTasks++;
				cntSortRays += rays;
				cntClippedRays += task[i].rayEnd-task[i].rayActive;
				cntSortTris += tris;
			}
#ifdef CUTOFF_DEPTH
			}
#endif

#ifdef DEBUG_INFO
			maxDepth = max(task[i].depth, maxDepth);
			syncCount += task[i].sync;
#endif
		}
	}

	if(stackMax == TASK_SIZE-1)
		printf("\aIncomplete result!\n");
#ifdef CUTOFF_DEPTH
	Debug << "\n\nStatistics for cutoff depth " << m_cutOffDepth << "\n\n";
#else
	Debug << "\n\n";
#endif

#ifdef DEBUG_INFO
	Debug << "ray_obj_intersections per ray = " << cntIsect/m_numRays << "\n";
	Debug << "cnt_leaves = " << isectTasks << "\n";
	Debug << "cnt_leaves per obj = " << (float)isectTasks/(float)m_numTris << "\n";
	Debug << "ray_obj_intersections = " << cntIsect << "\n";
	Debug << "Useless ray_obj_intersections = " << clippedIsect << "\n";
	Debug << "Avg ray_obj_intersections per leaf = " << cntIsect/(long double)isectTasks << "\n";
	Debug << "Max ray_obj_intersections per leaf = " << maxIsect << ", taskId: " << maxTaskId << "\n";
	Debug << "reduction [%] = " << 100.0f * (cntIsect/((long double)m_numRays*(long double)m_numTris)) << "\n";
	Debug << "Avg naive task width (rays) = " << sumRays/(long double)isectTasks << "\n";
	Debug << "Max naive task width (rays) = " << maxRays << "\n";
	Debug << "Avg naive task height (tris) = " << sumTris/(long double)isectTasks << "\n";
	Debug << "Max naive task height (tris) = " << maxTris << "\n";
	Debug << "Cnt sorted operations = " << sortTasks << "\n";
	double cntTrisLog2Tris = (double(m_numTris) * (double)(logf(m_numTris)/logf(2.0f)));
	double cntRaysLog2Tris = (double(m_numRays) * (double)(logf(m_numTris)/logf(2.0f)));
	Debug << "Cnt sorted triangles = " << cntSortTris << "\n";	
	Debug << "Cnt sorted triangles/(N log N), N=#tris = " << cntSortTris/cntTrisLog2Tris << "\n";
	Debug << "Cnt sorted rays = " << cntSortRays << " BEFORE CLIPPING\n";
	Debug << "Cnt sorted rays/(log N)/R, N=#tris,R=#rays = " << cntSortRays/cntRaysLog2Tris << " BEFORE CLIPPING\n";
	Debug << "Cnt clipped rays = " << cntClippedRays << "\n";
	Debug << "\n";
	Debug << "Max task depth = " << maxDepth << "\n";
	Debug << "Cnt gmem synchronizations: " << syncCount << "\n";
	Debug << "Ray issues = " << rayIssues << ", tris issues = " << triIssues << "\n";
	Debug << "Leafs failed to subdivide = " << subFailed << " (*3) => total useless tasks " << subFailed * 3 << "\n";

	Debug << "Terminated by:" << "\n";
	for(int i = 0; i < TerminatedBy_Max; i++)
	{
		Debug << terminatedNames[i] << ": " << terminatedBy[i] << "\n";
	}
#endif

	Debug << "max_queue_length = " << stackMax << "\n\n" << "\n";
}

//------------------------------------------------------------------------

F32 CudaNoStructTracer::traceCudaRayBuffer(RayBuffer& rb)
{
	CUfunction kernel;
#ifdef TEST_TASKS
	kernel = m_module->getKernel("trace");
	if (!kernel)
		fail("Trace kernel not found!");
#endif

	// Prepare the task data
	initPool(m_numRays, &rb.getRayBuffer());

	// Set input.
	KernelInputNoStruct& in = *(KernelInputNoStruct*)m_module->getGlobal("c_ns_in").getMutablePtr();
	in.numRays      = m_numRays;
	in.numTris		= m_numTris;
	in.anyHit       = !rb.getNeedClosestHit();
	in.rays         = rb.getRayBuffer().getMutableCudaPtr();
	in.results      = rb.getResultBuffer().getMutableCudaPtr();
	in.tris         = m_trisCompact.getCudaPtr();
	in.trisIndex    = m_trisIndex.getMutableCudaPtr();
	in.raysIndex    = m_raysIndex.getMutableCudaPtr();
	in.ppsRaysBuf   = m_ppsRays.getMutableCudaPtr();
	in.ppsTrisBuf   = m_ppsTris.getMutableCudaPtr();
	in.ppsRaysIndex = m_ppsRaysIndex.getMutableCudaPtr();
	in.ppsTrisIndex = m_ppsTrisIndex.getMutableCudaPtr();
	in.sortRays   = m_sortRays.getMutableCudaPtr();
	in.sortTris   = m_sortTris.getMutableCudaPtr();


#ifndef TEST_TASKS
	kernel = m_module->getKernel("__naive");
	if (!kernel)
		fail("Trace kernel not found!");

	Vec2i blockSizeN(1024, 1);
	Vec2i gridSizeN((m_numRays+1023)/1024, 1);

	float tNaive = m_module->launchKernelTimed(kernel, blockSizeN, gridSizeN);

	printf("Verifying GPU trace\n");
	/*for(int i = 0; i < m_numRays; i++)
	{
		const RayResult& res = rb.getResultForSlot(i);
		Debug << "Ray " << i << "\tGPU naive: id=" << res.id << ", t=" << res.t << "\n";
	}*/

	return tNaive;
#endif


#if SPLIT_TYPE == 3
	m_splitData.clearRange(0, 0, sizeof(SplitInfo)); // Set first split to zeros
	// Prepare split stack
	SplitInfo* &splits = *(SplitInfo**)m_module->getGlobal("g_splitStack").getMutablePtr();
	splits = (SplitInfo*)m_splitData.getMutableCudaPtr();
#endif

	CudaAABB bbox;
	memcpy(&bbox.m_mn, &m_bbox.min, sizeof(float3));
	memcpy(&bbox.m_mx, &m_bbox.max, sizeof(float3));

	// Set parent task containing all the work
	Task all;
	all.rayStart     = 0;
	all.rayLeft      = 0;
	all.rayRight     = m_numRays;
	all.rayEnd       = m_numRays;
	all.triStart     = 0;
	all.triLeft      = 0;
	all.triRight     = m_numTris;
	all.triEnd       = m_numTris;
	all.bbox         = bbox;
	all.step         = 0;
	all.depend1      = DependType_Root;
	all.depend2      = DependType_None; // Only one task is dependent on this one - the unfinished counter
	all.lock         = LockType_Free;
	all.bestCost     = 1e38f;
	all.depth        = 0;
	all.subFailureCounter = 0;
	Vector3 size     = m_bbox.Diagonal();
	all.axis         = size.MajorAxis();
	all.terminatedBy = TerminatedBy_None;
#ifdef DEBUG_INFO
	all.sync         = 0;
	all.parent       = -1;
	all.taskID       = 0;
	all.clippedRays  = 0;
	all.clockStart   = 0;
	all.clockEnd     = 0;
#endif

#if SPLIT_TYPE == 0
#if SCAN_TYPE == 0
	all.type = TaskType_Sort_PPS1;
#elif SCAN_TYPE == 1
	all.type = TaskType_Sort_PPS1_Up;
#elif SCAN_TYPE == 2 ||  SCAN_TYPE == 3
	all.type = TaskType_Sort_SORT1;
#endif

	all.unfinished = warpSubtasks(m_numRays) + warpSubtasks(m_numTris);
	all.bestOrder = warpSubtasks(m_numRays);
	float pos = m_bbox.min[all.axis] + m_bbox.Size(all.axis)/2.0f;
	if(all.axis == 0)
		all.splitPlane = make_float4(1.f, 0.f, 0.f, -pos);
	else if(all.axis == 1)
		all.splitPlane = make_float4(0.f, 1.f, 0.f, -pos);
	else
		all.splitPlane = make_float4(0.f, 0.f, 1.f, -pos);
#else
	all.type = TaskType_Split;
#if SPLIT_TYPE == 1
	int evaluatedCandidates = (int)sqrtf(m_numRays) + (int)sqrtf(m_numTris);
	int numPlanes = 0.5f * (m_numRays + m_numTris)/evaluatedCandidates;
	all.unfinished = warpSubtasks(numPlanes); // This must be the same as in the GPU code
#elif SPLIT_TYPE == 2
	all.unfinished = 1;
#elif SPLIT_TYPE == 3
	all.type = TaskType_SplitParallel;
	int evaluatedRays = warpSubtasks((int)sqrtf(m_numRays));
	int evaluatedTris = warpSubtasks((int)sqrtf(m_numTris));
	all.unfinished = PLANE_COUNT*(evaluatedRays+evaluatedTris); // Each WARP_SIZE rays and tris add their result to one plane
#endif
#endif

#ifdef DEBUG_PPS
	all.type         = TaskType_Sort_PPS1_Up;
	int pRays = warpSubtasks(m_numRays);
	all.bestOrder = pRays;
	int pTris = warpSubtasks(m_numTris);
	all.unfinished   = pRays+pTris;
#endif

	all.origSize     = all.unfinished;

	m_taskData.setRange(TASK_SIZE * sizeof(int), &all, sizeof(Task)); // Set the first task

	// Set parent task header
	m_taskData.setRange(0, &all.unfinished, sizeof(int)); // Set the first task

	// Prepare the task stack
	TaskStack& tasks = *(TaskStack*)m_module->getGlobal("g_taskStack").getMutablePtr();
	tasks.header     = (int*)m_taskData.getMutableCudaPtr();
	tasks.tasks      = (Task*)m_taskData.getMutableCudaPtr(TASK_SIZE * sizeof(int));
	tasks.top        = 0;
	tasks.bottom     = 0;
	//memset(tasks.active, 0, sizeof(int)*(ACTIVE_MAX+1));
	memset(tasks.active, -1, sizeof(int)*(ACTIVE_MAX+1));
		tasks.active[0] = 0;
	//for(int i = 0; i < ACTIVE_MAX+1; i++)
	//	tasks.active[i] = i;
	tasks.activeTop = 1;
	//tasks.empty[0] = 0;
	//int j = 1;
	//for(int i = EMPTY_MAX; i > 0; i--, j++)
	//	tasks.empty[i] = j;
	memset(tasks.empty, 0, sizeof(int)*(EMPTY_MAX+1));
	tasks.emptyTop = 0;
	tasks.emptyBottom = 0;
	tasks.unfinished = -1; // We are waiting for one task to finish = task all
	tasks.sizePool = TASK_SIZE;
	tasks.sizeNodes = m_bvhData.getSize()/sizeof(CudaKdtreeNode);
	tasks.sizeTris = m_trisIndexOut.getSize()/sizeof(S32);

	// Determine block and grid sizes.
#ifdef ONE_WARP_RUN
	Vec2i blockSize(WARP_SIZE, 1); // threadIdx.x must equal the thread lane in warp
	Vec2i gridSize(1, 1); // Number of SMs * Number of blocks?
	int numWarps = 1;
#else
	int numWarpsPerBlock = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numWarpsPerBlock");
	int numBlocksPerSM = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numBlockPerSM");
	Vec2i blockSize(WARP_SIZE, numWarpsPerBlock); // threadIdx.x must equal the thread lane in warp
	int gridSizeX = NUM_SM*numBlocksPerSM;
	int numWarps = numWarpsPerBlock*gridSizeX;
	Vec2i gridSize(gridSizeX, 1); // Number of SMs * Number of blocks?

	if(gridSizeX*numWarpsPerBlock != NUM_WARPS)
		printf("\aNUM_WARPS constant does not match the launch parameters\n");
#endif

	m_debug.resizeDiscard(blockSize.y*gridSize.x*sizeof(float4));
	m_debug.clear();
	in.debug = m_debug.getMutableCudaPtr();

	// Launch.
	float tKernel = m_module->launchKernelTimed(kernel, blockSize, gridSize);

#ifndef BENCHMARK
	cuCtxSynchronize(); // Flushes printfs
#endif

#ifdef DEBUG_PPS
	ptout = (S32*)m_ppsTris.getPtr();
	stout = (S32*)m_sortTris.getPtr();

	prout = (S32*)m_ppsRays.getPtr();
	srout = (S32*)m_sortRays.getPtr();
	S32 sum = 0;
	S32 error = 0;
	int j = 0;
	for(int i=0;i<m_numTris;i++)
	{
		sum += *stout; // Here for inclusive scan
		if(*ptout != sum)
		{
			cout << "PPS error for item " << i << ", CPU=" << sum << ", GPU=" << *ptout << " for " << m_numTris << " triangles!" << "\n";
			error = 1;
			if(j == 10)
				break;
			j++;
		}
		if(*stout < -1 || *stout > 1)
		{
			cout << "\nWTF " << i << " of " << m_numTris << ": " << *stout << "!\n" << "\n";
			break;
		}
		//sum += *stout; // Here for exclusive scan
		ptout++;
		stout++;
	}

	sum = 0;
	for(int i=0;i<m_numRays;i++)
	{
		sum += *srout; // Here for inclusive scan
		if(*prout != sum)
		{
			cout << "PPS error for item " << i << ", CPU=" << sum << ", GPU=" << *prout << " for " << m_numRays << " rays!" << "\n";
			error = 1;
			if(j == 10)
				break;
			j++;
		}
		if(*srout < -1 || *srout > 2)
		{
			cout << "\nWTF " << i << " of " << m_numRays << ": " << *srout << "!\n" << "\n";
			break;
		}
		//sum += *srout; // Here for exclusive scan
		prout++;
		srout++;
	}

	if(!error)
		cout << "PPS correct for " << m_numTris << " triangles and " << m_numRays << " rays!" << "\n";
	return 0;
#endif

	// Set rays index buffer
	/*int* ind = (int*)m_raysIndex.getPtr();
	int count = 0;
	int mismatched = 0;

	// Validate if rays hit triangles
	printf("Verifying GPU trace\n");
	for(int i = 0; i < m_numRays; i++)
	{
		const RayResult& res = rb.getResultForSlot(i);
		//RayResult& res = rb.getMutableResultForSlot(i);
		//res.clear();
		Ray ray = rb.getRayForSlot(i);
		RayResult cpu;
		ray.tmax = 1e36;

		if(i % 10000 == 0)
			printf("rid: %d\n", i);
		traceCpuRay(ray, cpu, !rb.getNeedClosestHit());
		//traceCpuRay(ray, res, !rb.getNeedClosestHit());

		if(ind[i] != i)
			count++;

		if(res.id != cpu.id)
		{
			Debug << "Ray " << i << " CPU/GPU mismatch! Swapped: " << (ind[i] != i) << "\n"
			<< "\tCPU: id=" << cpu.id << ", t=" << cpu.t << "\n"
			<< "\tGPU: id=" << res.id << ", t=" << res.t << "\n";
			mismatched++;
		}

		//Debug << "Ray " << i << " CPU: id=" << cpu.id << ", t=" << cpu.t << "\n";
	}
	Debug << "Swaped " << count << "\n";
	Debug << "Mismatched " << mismatched << "\n";*/

	//Debug << "\nTraced in " << tKernel << "\n\n";

#ifndef BENCHMARK
	tasks = *(TaskStack*)m_module->getGlobal("g_taskStack").getPtr();
	printPool(tasks, numWarps);

	/*for(int i = 0; i < m_numRays; i++)
	{
		const RayResult& res = rb.getResultForSlot(i);
		if(res.id == -2)
		{
			printf("Error on ray %d! Value: (%d, %f, %f, %f)\n", i, res.id, res.t, res.u, res.v);
		}
		//Debug << "Ray " << i << "! Value: (" << res.id << ", " << res.t << ", " << res.u << ", " << res.v << ")" << "\n";
	}*/

	/*CUcontext ctx;
	cuCtxPopCurrent(&ctx);
	cuCtxDestroy(ctx);
	exit(0);*/
#endif

	return tKernel;
}

F32 CudaNoStructTracer::buildCudaBVH()
{
	CUfunction kernel;
	kernel = m_module->getKernel("build");
	if (!kernel)
		fail("Build kernel not found!");

#ifdef MALLOC_SCRATCHPAD
	KernelInputBVH& in = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
	in.numTris	    = m_numTris;
	in.tris         = m_trisCompact.getCudaPtr();
	in.trisIndex    = m_trisIndex.getMutableCudaPtr();
#ifdef COMPACT_LAYOUT
	in.trisOut      = m_trisCompactOut.getMutableCudaPtr();
	in.trisIndexOut = m_trisIndexOut.getMutableCudaPtr();
#endif
#endif

	// Prepare the task data
	initPool();

#ifndef MALLOC_SCRATCHPAD
	// Set input.
	KernelInputBVH& in = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
	in.numTris		= m_numTris;
	in.tris         = m_trisCompact.getCudaPtr();
	in.trisIndex    = m_trisIndex.getMutableCudaPtr();
	//in.trisBox      = m_trisBox.getCudaPtr();
	in.ppsTrisBuf   = m_ppsTris.getMutableCudaPtr();
	in.ppsTrisIndex = m_ppsTrisIndex.getMutableCudaPtr();
	in.sortTris     = m_sortTris.getMutableCudaPtr();
#ifdef COMPACT_LAYOUT
	in.trisOut      = m_trisCompactOut.getMutableCudaPtr();
	in.trisIndexOut = m_trisIndexOut.getMutableCudaPtr();
#endif
#else
	CUfunction kernelAlloc = m_module->getKernel("allocFreeableMemory", 2*sizeof(int));
	if (!kernelAlloc)
		fail("Memory allocation kernel not found!");

	int offset = 0;
	offset += m_module->setParami(kernelAlloc, offset, m_numTris);
	offset += m_module->setParami(kernelAlloc, offset, 0);
	F32 allocTime = m_module->launchKernelTimed(kernelAlloc, Vec2i(1,1), Vec2i(1, 1));

#ifndef BENCHMARK
	printf("Memory allocated in %f\n", allocTime);
#endif

	CUfunction kernelMemCpyIndex = m_module->getKernel("MemCpyIndex", sizeof(CUdeviceptr)+sizeof(int));
	if (!kernelMemCpyIndex)
		fail("Memory copy kernel not found!");

	int memSize = m_trisIndex.getSize()/sizeof(int);
	offset = 0;
	offset += m_module->setParamPtr(kernelMemCpyIndex, offset, m_trisIndex.getCudaPtr());
	offset += m_module->setParami(kernelMemCpyIndex, offset, memSize);
	F32 memcpyTime = m_module->launchKernelTimed(kernelMemCpyIndex, Vec2i(256,1), Vec2i((memSize-1+256)/256, 1));

#ifndef BENCHMARK
	printf("Triangle indices copied in %f\n", memcpyTime);
#endif
	in = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
#endif

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
	SplitRed split;
	for(int i = 0; i < 2; i++)
	{
		split.children[i].bbox.m_mn = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		split.children[i].bbox.m_mx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		split.children[i].cnt = 0;
	}

	SplitArray sArray;
	for(int i = 0; i < NUM_WARPS; i++)
	{
		for(int j = 0; j < PLANE_COUNT; j++)
			sArray.splits[i][j] = split;
	}
#else
	SplitRed split;
	for(int i = 0; i < 2; i++)
	{
		//split.children[i].bbox.m_mn = make_float3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
		//split.children[i].bbox.m_mx = make_float3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
		split.children[i].bbox.m_mn = make_int3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
		split.children[i].bbox.m_mx = make_int3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
		split.children[i].cnt = 0;
	}

	SplitArray sArray;
	for(int j = 0; j < PLANE_COUNT; j++)
		sArray.splits[j] = split;

	m_splitData.setRange(0, &sArray, sizeof(SplitArray)); // Set the first splits
#endif

	m_splitData.setRange(TASK_SIZE * sizeof(SplitArray), &sArray, sizeof(SplitArray)); // Set the last splits for copy
#endif

	CudaAABB bbox;
	memcpy(&bbox.m_mn, &m_bbox.min, sizeof(float3));
	memcpy(&bbox.m_mx, &m_bbox.max, sizeof(float3));

	// Set parent task containing all the work
	TaskBVH all;
	all.triStart     = 0;
	all.triLeft      = 0;
#ifndef MALLOC_SCRATCHPAD
	all.triRight     = m_numTris;
#else
	all.triRight     = 0;
#endif
	all.triEnd       = m_numTris;
	all.bbox         = bbox;
	all.step         = 0;
	all.lock         = LockType_Free;
	all.bestCost     = 1e38f;
	all.depth        = 0;
	all.dynamicMemory= 0;
#ifndef MALLOC_SCRATCHPAD
	all.triIdxCtr    = 0;
#endif
	all.parentIdx    = -1;
	all.nodeIdx      = 0;
	all.taskID       = 0;
	Vector3 size     = m_bbox.Diagonal();
	all.axis         = size.MajorAxis();
	all.terminatedBy = TerminatedBy_None;
#ifdef DEBUG_INFO
	all.sync         = 0;
	all.parent       = -1;
	all.clockStart   = 0;
	all.clockEnd     = 0;
#endif

#if SPLIT_TYPE == 0
#if SCAN_TYPE == 0
	all.type         = TaskType_Sort_PPS1;
#elif SCAN_TYPE == 1
	all.type         = TaskType_Sort_PPS1_Up;
#elif SCAN_TYPE == 2 ||  SCAN_TYPE == 3
	all.type         = TaskType_Sort_SORT1;
#endif
	all.unfinished   = warpSubtasks(m_numTris);
	float pos = m_bbox.min[all.axis] + m_bbox.Size(all.axis)/2.0f;
	if(all.axis == 0)
		all.splitPlane   = make_float4(1.f, 0.f, 0.f, -pos);
	else if(all.axis == 1)
		all.splitPlane   = make_float4(0.f, 1.f, 0.f, -pos);
	else
		all.splitPlane   = make_float4(0.f, 0.f, 1.f, -pos);
#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
	all.type         = TaskType_InitMemory;
	all.unfinished   = warpSubtasks(sizeof(SplitArray)/sizeof(int));
#else
	all.type         = TaskType_BinTriangles;
	all.unfinished   = (warpSubtasks(m_numTris)+BIN_MULTIPLIER-1)/BIN_MULTIPLIER;
	/*all.type         = TaskType_BuildObjectSAH;
	all.unfinished   = 1;*/
#endif
#endif
	all.origSize     = all.unfinished;

	m_taskData.setRange(TASK_SIZE * sizeof(int), &all, sizeof(TaskBVH)); // Set the first task

	// Set parent task header
	m_taskData.setRange(0, &all.unfinished, sizeof(int)); // Set the first task

	// Prepare the task stack
	TaskStackBVH& tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getMutablePtr();
	tasks.header     = (int*)m_taskData.getMutableCudaPtr();
	tasks.tasks      = (TaskBVH*)m_taskData.getMutableCudaPtr(TASK_SIZE * sizeof(int));
	tasks.nodeTop    = 1;
	tasks.triTop     = 0;
	tasks.top        = 0;
	tasks.bottom     = 0;
	//memset(tasks.active, 0, sizeof(int)*(ACTIVE_MAX+1));
	memset(tasks.active, -1, sizeof(int)*(ACTIVE_MAX+1));
	tasks.active[0] = 0;
	/*for(int i = 0; i < ACTIVE_MAX+1; i++)
	tasks.active[i] = i;*/
	tasks.activeTop = 1;
	//tasks.empty[0] = 0;
	//int j = 1;
	//for(int i = EMPTY_MAX; i > 0; i--, j++)
	//	tasks.empty[i] = j;
	memset(tasks.empty, 0, sizeof(int)*(EMPTY_MAX+1));
	tasks.emptyTop = 0;
	tasks.emptyBottom = 0;
	tasks.unfinished = -1; // We are waiting for one task to finish = task all
	tasks.numSortedTris = 0;
	tasks.numNodes = 0;
	tasks.numLeaves = 0;
	tasks.numEmptyLeaves = 0;
	tasks.sizePool = TASK_SIZE;
	tasks.sizeNodes = m_bvhData.getSize()/sizeof(CudaKdtreeNode);
	tasks.sizeTris = m_trisIndexOut.getSize()/sizeof(S32);
	memset(tasks.leafHist, 0, sizeof(tasks.leafHist));

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
	// Prepare split stack
	SplitArray* &splits = *(SplitArray**)m_module->getGlobal("g_redSplits").getMutablePtr();
	splits = (SplitArray*)m_splitData.getMutableCudaPtr();
#endif

	CudaBVHNode* &bvh = *(CudaBVHNode**)m_module->getGlobal("g_bvh").getMutablePtr();
	bvh = (CudaBVHNode*)m_bvhData.getMutableCudaPtr();

	// Determine block and grid sizes.
#ifdef ONE_WARP_RUN
	Vec2i blockSize(WARP_SIZE, 1); // threadIdx.x must equal the thread lane in warp
	Vec2i gridSize(1, 1); // Number of SMs * Number of blocks?
	int numWarps = 1;
#else
	int numWarpsPerBlock = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numWarpsPerBlock");
	int numBlocksPerSM = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numBlockPerSM");
	Vec2i blockSize(WARP_SIZE, numWarpsPerBlock); // threadIdx.x must equal the thread lane in warp
	int gridSizeX = NUM_SM*numBlocksPerSM;
	int numWarps = numWarpsPerBlock*gridSizeX;
	Vec2i gridSize(gridSizeX, 1); // Number of SMs * Number of blocks?

	if(gridSizeX*numWarpsPerBlock != NUM_WARPS)
		printf("\aNUM_WARPS constant does not match the launch parameters\n");
#endif

	m_debug.resizeDiscard(blockSize.y*gridSize.x*sizeof(float4));
	m_debug.clear();
	in.debug = m_debug.getMutableCudaPtr();

	// Launch.
	float tKernel = m_module->launchKernelTimed(kernel, blockSize, gridSize);

/*#ifdef MALLOC_SCRATCHPAD
	CUfunction kernelDealloc = m_module->getKernel("deallocFreeableMemory", 0);
	if (!kernelDealloc)
		fail("Memory allocation kernel not found!");

	F32 deallocTime = m_module->launchKernelTimed(kernelDealloc, Vec2i(1,1), Vec2i(1, 1));

	printf("Memory freed in %f\n", deallocTime);
#endif*/

#ifndef BENCHMARK
	cuCtxSynchronize(); // Flushes printfs
#endif

#ifdef DEBUG_PPS
	pout = (S32*)m_ppsTris.getPtr();
	sout = (S32*)m_sortTris.getPtr();
	S32 sum = 0;
	S32 error = 0;
	int j = 0;
	for(int i=0;i<m_numTris;i++)
	{
		sum += *sout; // Here for inclusive scan
		if(*pout != sum)
		{
			cout << "PPS error for item " << i << ", CPU=" << sum << ", GPU=" << *pout << " for " << m_numTris << " triangles!" << "\n";
			error = 1;
			if(j == 10)
				break;
			j++;
		}
		if(*sout != 0 && *sout != 1)
		{
			cout << "\nWTF " << i << " of " << m_numTris << ": " << *sout << "!\n" << "\n";
			break;
		}
		//sum += *sout; // Here for exclusive scan
		pout++;
		sout++;
	}

	if(!error)
		cout << "PPS correct for " << m_numTris << " triangles!" << "\n";
	return 0;
#endif

	tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();
	if(tasks.unfinished != 0 || tasks.top > tasks.sizePool || tasks.nodeTop > m_bvhData.getSize() / sizeof(CudaBVHNode) || tasks.triTop > m_trisIndexOut.getSize() / sizeof(S32)) // Something went fishy
		tKernel = 1e38f;
	//printf("%d (%d x %d) (%d x %d)\n", tasks.unfinished != 0, tasks.nodeTop, m_bvhData.getSize() / sizeof(CudaBVHNode), tasks.triTop, m_trisIndexOut.getSize() / sizeof(S32));

	//Debug << "\nBuild in " << tKernel << "\n\n";

#ifndef BENCHMARK
	printPool(tasks, numWarps);

	/*Debug << "\n\nBVH" << "\n";
	CudaBVHNode* nodes = (CudaBVHNode*)m_bvhData.getPtr();

	for(int i = 0; i < tasks.nodeTop; i++)
	{
		Debug << "Node " << i << "\n";
		Debug << "BoxLeft: (" << nodes[i].c0xy.x << ", " << nodes[i].c0xy.z << ", " << nodes[i].c01z.x << ") - ("
				<< nodes[i].c0xy.y << ", " << nodes[i].c0xy.w << ", " << nodes[i].c01z.y << ")\n";
		Debug << "BoxRight: (" << nodes[i].c1xy.x << ", " << nodes[i].c1xy.z << ", " << nodes[i].c01z.z << ") - ("
				<< nodes[i].c1xy.y << ", " << nodes[i].c1xy.w << ", " << nodes[i].c01z.w << ")\n";
		Debug << "Children: " << nodes[i].children.x << ", " << nodes[i].children.y << "\n\n";
	}*/

	// Free data
	deinitPool();
#endif

	return tKernel;
}

F32 CudaNoStructTracer::buildCudaKdtree()
{
	CUfunction kernel;
	kernel = m_module->getKernel("build");
	if (!kernel)
		fail("Build kernel not found!");

	KernelInputBVH& in = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
	in.numTris	    = m_numTris;
	in.tris         = m_trisCompact.getCudaPtr();
	in.trisIndex    = m_trisIndex.getMutableCudaPtr();

#ifndef INTERLEAVED_LAYOUT
	in.trisOut      = m_trisCompactOut.getMutableCudaPtr();
	in.trisIndexOut = m_trisIndexOut.getMutableCudaPtr();
#endif

	// Prepare the task data
	initPool();
	// Set the maximum depth for the current triangle count
	RtEnvironment& cudaEnv = *(RtEnvironment*)m_module->getGlobal("c_env").getMutablePtr();
	float k1 = Environment::GetSingleton()->GetFloat("SubdivisionRayCaster.depthK1");
	float k2 = Environment::GetSingleton()->GetFloat("SubdivisionRayCaster.depthK2");
	float f1 = Environment::GetSingleton()->GetFloat("SubdivisionRayCaster.failK1");
	float f2 = Environment::GetSingleton()->GetFloat("SubdivisionRayCaster.failK2");
	cudaEnv.optMaxDepth  = k1 * log2((F32)m_numTris) + k2;
	cudaEnv.failureCount = f1 * cudaEnv.optMaxDepth + f2;
#ifndef BENCHMARK
	printf("Maximum depth = %d\n", cudaEnv.optMaxDepth);
	printf("Failure count = %d\n", cudaEnv.failureCount);
#endif

	int baseOffset = setDynamicMemory();
	in = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();

#if SPLIT_TYPE == 3
	m_splitData.clearRange(0, 0, sizeof(SplitInfoTri)); // Set first split to zeros
#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
	SplitRed split;
	for(int i = 0; i < 2; i++)
	{
		split.children[i].bbox.m_mn = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		split.children[i].bbox.m_mx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		split.children[i].cnt = 0;
	}

	SplitArray sArray;
	for(int i = 0; i < NUM_WARPS; i++)
	{
		for(int j = 0; j < PLANE_COUNT; j++)
			sArray.splits[i][j] = split;
	}
#else
	//SplitRed split;
	//for(int i = 0; i < 2; i++)
	//{
	//	//split.children[i].bbox.m_mn = make_float3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
	//	//split.children[i].bbox.m_mx = make_float3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
	//	split.children[i].bbox.m_mn = make_int3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
	//	split.children[i].bbox.m_mx = make_int3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
	//	split.children[i].cnt = 0;
	//}

	//SplitArray sArray;
	//for(int j = 0; j < PLANE_COUNT; j++)
	//	sArray.splits[j] = split;

	//m_splitData.setRange(0, &sArray, sizeof(SplitArray)); // Set the first splits
	m_splitData.clearRange(0, 0, sizeof(SplitInfoTri)); // Set first split to zeros
#endif

	//m_splitData.setRange(TASK_SIZE * sizeof(SplitArray), &sArray, sizeof(SplitArray)); // Set the last splits for copy
	// Prepare split stack
	//SplitArray* &splits = *(SplitArray**)m_module->getGlobal("g_redSplits").getMutablePtr();
	//splits = (SplitArray*)m_splitData.getMutableCudaPtr();

	SplitInfoTri* &splits = *(SplitInfoTri**)m_module->getGlobal("g_splitStack").getMutablePtr();
	splits = (SplitInfoTri*)m_splitData.getMutableCudaPtr();
#endif

	CudaAABB bbox;
	memcpy(&bbox.m_mn, &m_bbox.min, sizeof(float3));
	memcpy(&bbox.m_mx, &m_bbox.max, sizeof(float3));
	/*bbox.m_mn.x -= m_epsilon;
	bbox.m_mn.y -= m_epsilon;
	bbox.m_mn.z -= m_epsilon;
	bbox.m_mx.x += m_epsilon;
	bbox.m_mx.y += m_epsilon;
	bbox.m_mx.z += m_epsilon;*/

	// Set parent task containing all the work
	TaskBVH all;
	all.triStart     = 0;
	all.triLeft      = 0;
	all.triRight     = 0;
	all.triEnd       = m_numTris;
	all.bbox         = bbox;
	all.step         = 0;
	all.lock         = LockType_Free;
	all.bestCost     = 1e38f;
	all.depth        = 0;
	all.dynamicMemory= baseOffset;
#ifdef MALLOC_SCRATCHPAD
	all.subFailureCounter = 0;
#endif
	all.parentIdx    = -1;
	all.nodeIdx      = 0;
	all.taskID       = 0;
	Vector3 size     = m_bbox.Diagonal();
	all.axis         = size.MajorAxis();
	all.terminatedBy = TerminatedBy_None;
#ifdef DEBUG_INFO
	all.sync         = 0;
	all.parent       = -1;
	all.clockStart   = 0;
	all.clockEnd     = 0;
#endif

#if SPLIT_TYPE == 0
#if SCAN_TYPE == 0
	all.type         = TaskType_Sort_PPS1;
#elif SCAN_TYPE == 1
	all.type         = TaskType_Sort_PPS1_Up;
#elif SCAN_TYPE == 2 ||  SCAN_TYPE == 3
	all.type         = TaskType_Sort_SORT1;
#endif
	all.unfinished   = warpSubtasks(m_numTris);
	float pos = m_bbox.min[all.axis] + m_bbox.Size(all.axis)/2.0f;
	if(all.axis == 0)
		all.splitPlane   = make_float4(1.f, 0.f, 0.f, -pos);
	else if(all.axis == 1)
		all.splitPlane   = make_float4(0.f, 1.f, 0.f, -pos);
	else
		all.splitPlane   = make_float4(0.f, 0.f, 1.f, -pos);
#elif SPLIT_TYPE == 1
	all.type = TaskType_Split;
#if 0 // SQRT candidates
	int evaluatedCandidates = (int)sqrtf(m_numTris);
	int evaluatedCandidates = 1;
	int numPlanes = 0.5f * m_numTris/evaluatedCandidates;
#elif 0 // Fixed candidates
	int numPlanes = 32768;
#else // All candidates
	int numPlanes = m_numTris*6; // Number of warp sized subtasks
#endif
	all.unfinished = warpSubtasks(numPlanes); // This must be the same as in the GPU code
#elif SPLIT_TYPE == 2
	all.type = TaskType_Split;
	all.unfinished = 1;
#elif SPLIT_TYPE == 3
	all.type = TaskType_SplitParallel;
	int evaluatedRays = warpSubtasks((int)sqrtf(m_numRays));
	int evaluatedTris = warpSubtasks((int)sqrtf(m_numTris));
	all.unfinished = PLANE_COUNT*(evaluatedRays+evaluatedTris); // Each WARP_SIZE rays and tris add their result to one plane
#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
	all.type         = TaskType_InitMemory;
	all.unfinished   = warpSubtasks(sizeof(SplitArray)/sizeof(int));
#else
	all.type         = TaskType_BinTriangles;
	all.unfinished   = (warpSubtasks(m_numTris)+BIN_MULTIPLIER-1)/BIN_MULTIPLIER;
	/*all.type         = TaskType_BuildObjectSAH;
	all.unfinished   = 1;*/
#endif
#endif
	all.origSize     = all.unfinished;

	m_taskData.setRange(TASK_SIZE * sizeof(int), &all, sizeof(TaskBVH)); // Set the first task

	// Set parent task header
	m_taskData.setRange(0, &all.unfinished, sizeof(int)); // Set the first task

	// Prepare the task stack
	TaskStackBVH& tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getMutablePtr();
	tasks.header     = (int*)m_taskData.getMutableCudaPtr();
	tasks.tasks      = (TaskBVH*)m_taskData.getMutableCudaPtr(TASK_SIZE * sizeof(int));
#ifndef INTERLEAVED_LAYOUT
	tasks.nodeTop    = 1;
#else
	tasks.nodeTop    = sizeof(CudaKdtreeNode);
#endif
	tasks.triTop     = 0;
	tasks.top        = 0;
	tasks.bottom     = 0;
	//memset(tasks.active, 0, sizeof(int)*(ACTIVE_MAX+1));
	memset(tasks.active, -1, sizeof(int)*(ACTIVE_MAX+1));
	tasks.active[0] = 0;
	/*for(int i = 0; i < ACTIVE_MAX+1; i++)
	tasks.active[i] = i;*/
	tasks.activeTop = 1;
	//tasks.empty[0] = 0;
	//int j = 1;
	//for(int i = EMPTY_MAX; i > 0; i--, j++)
	//	tasks.empty[i] = j;
	memset(tasks.empty, 0, sizeof(int)*(EMPTY_MAX+1));
	tasks.emptyTop = 0;
	tasks.emptyBottom = 0;
	tasks.unfinished = -1; // We are waiting for one task to finish = task all
	tasks.numSortedTris = 0;
	tasks.numNodes = 0;
	tasks.numEmptyLeaves = 0;
	tasks.numLeaves = 0;
	tasks.sizePool = TASK_SIZE;
	tasks.sizeNodes = m_bvhData.getSize()/sizeof(CudaKdtreeNode);
	tasks.sizeTris = m_trisIndexOut.getSize()/sizeof(S32);
	memset(tasks.leafHist, 0, sizeof(tasks.leafHist));

	CudaKdtreeNode* &kdtree = *(CudaKdtreeNode**)m_module->getGlobal("g_kdtree").getMutablePtr();
	kdtree = (CudaKdtreeNode*)m_bvhData.getMutableCudaPtr();

	// Determine block and grid sizes.
#ifdef ONE_WARP_RUN
	Vec2i blockSize(WARP_SIZE, 1); // threadIdx.x must equal the thread lane in warp
	Vec2i gridSize(1, 1); // Number of SMs * Number of blocks?
	int numWarps = 1;
#else
	int numWarpsPerBlock = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numWarpsPerBlock");
	int numBlocksPerSM = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numBlockPerSM");
	Vec2i blockSize(WARP_SIZE, numWarpsPerBlock); // threadIdx.x must equal the thread lane in warp
	int gridSizeX = NUM_SM*numBlocksPerSM;
	int numWarps = numWarpsPerBlock*gridSizeX;
	Vec2i gridSize(gridSizeX, 1); // Number of SMs * Number of blocks?

	if(gridSizeX*numWarpsPerBlock != NUM_WARPS)
		printf("\aNUM_WARPS constant does not match the launch parameters\n");
#endif

	m_debug.resizeDiscard(blockSize.y*gridSize.x*sizeof(float4));
	m_debug.clear();
	in.debug = m_debug.getMutableCudaPtr();

	// Launch.
	float tKernel = 0.f;
#ifndef DUPLICATE_REFERENCES
	tKernel += convertWoop();
#endif
	tKernel += m_module->launchKernelTimed(kernel, blockSize, gridSize);

/*#ifdef MALLOC_SCRATCHPAD
	CUfunction kernelDealloc = m_module->getKernel("deallocFreeableMemory", 0);
	if (!kernelDealloc)
		fail("Memory allocation kernel not found!");

	F32 deallocTime = m_module->launchKernelTimed(kernelDealloc, Vec2i(1,1), Vec2i(1, 1));

	printf("Memory freed in %f\n", deallocTime);
#endif*/

#ifndef BENCHMARK
	cuCtxSynchronize(); // Flushes printfs
#endif

#ifdef DEBUG_PPS
	pout = (S32*)m_ppsTris.getPtr();
	sout = (S32*)m_sortTris.getPtr();
	S32 sum = 0;
	S32 error = 0;
	int j = 0;
	for(int i=0;i<m_numTris;i++)
	{
		sum += *sout; // Here for inclusive scan
		if(*pout != sum)
		{
			cout << "PPS error for item " << i << ", CPU=" << sum << ", GPU=" << *pout << " for " << m_numTris << " triangles!" << "\n";
			error = 1;
			if(j == 10)
				break;
			j++;
		}
		if(*sout != 0 && *sout != 1)
		{
			cout << "\nWTF " << i << " of " << m_numTris << ": " << *sout << "!\n" << "\n";
			break;
		}
		//sum += *sout; // Here for exclusive scan
		pout++;
		sout++;
	}

	if(!error)
		cout << "PPS correct for " << m_numTris << " triangles!" << "\n";
	return 0;
#endif

	tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();
#ifndef INTERLEAVED_LAYOUT
	if(tasks.unfinished != 0 || tasks.top > tasks.sizePool || tasks.nodeTop > m_bvhData.getSize() / sizeof(CudaKdtreeNode) || tasks.triTop > m_trisIndexOut.getSize() / sizeof(S32)) // Something went fishy
#else
	if(tasks.unfinished != 0 || tasks.nodeTop > m_bvhData.getSize()) // Something went fishy
#endif
		tKernel = 1e38f;
	//printf("%d (%d x %d) (%d x %d)\n", tasks.unfinished != 0, tasks.nodeTop, m_bvhData.getSize() / sizeof(CudaKdtreeNode), tasks.triTop, m_trisIndexOut.getSize() / sizeof(S32));

	//Debug << "\nBuild in " << tKernel << "\n\n";

#ifndef BENCHMARK
	printPool(tasks, numWarps);

	/*Debug << "\n\nKdtree" << "\n";
	CudaBVHNode* nodes = (CudaKdtreeNode*)m_bvhData.getPtr();

	for(int i = 0; i < tasks.nodeTop; i++)
	{
		Debug << "Node " << i << "\n";
		Debug << "BoxLeft: (" << nodes[i].c0xy.x << ", " << nodes[i].c0xy.z << ", " << nodes[i].c01z.x << ") - ("
				<< nodes[i].c0xy.y << ", " << nodes[i].c0xy.w << ", " << nodes[i].c01z.y << ")\n";
		Debug << "BoxRight: (" << nodes[i].c1xy.x << ", " << nodes[i].c1xy.z << ", " << nodes[i].c01z.z << ") - ("
				<< nodes[i].c1xy.y << ", " << nodes[i].c1xy.w << ", " << nodes[i].c01z.w << ")\n";
		Debug << "Children: " << nodes[i].children.x << ", " << nodes[i].children.y << "\n\n";
	}*/

	// Free data
	deinitPool();
#endif

	return tKernel;
}

F32 CudaNoStructTracer::testSort(S32 arraySize)
{
	m_compiler.setSourceFile("src/rt/kernels/persistent_test.cu");
	m_module = m_compiler.compile();
	failIfError();

	CUfunction kernel;
	//kernel = m_module->getKernel("sort");
	//kernel = m_module->getKernel("testMemoryCamping");
	kernel = m_module->getKernel("testKeplerSort");
	if (!kernel)
		fail("Sort kernel not found!");

	// Prepare the task data
	initPool();

	// Set ppsTrisIndex
	/*S32* tid = (S32*)m_ppsTrisIndex.getMutablePtr();
	for(int i=0; i<arraySize/2; i++)
	{
		*tid = 0;
		tid++;
	}
	for(int i=arraySize/2; i<arraySize; i++)
	{
		*tid = 1;
		tid++;
	}*/
	
	m_trisIndex.resizeDiscard(sizeof(int)*arraySize);
	S32* tiout = (S32*)m_trisIndex.getMutablePtr();
	for(int i=0; i < arraySize; i++)
	{
		// indices 
		*tiout = (arraySize-1) - i;
		tiout++;
	}

	// Set input.
	KernelInputBVH& in = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
	in.numTris		= arraySize;
	in.trisIndex    = m_trisIndex.getMutableCudaPtr();
	in.ppsTrisBuf   = m_ppsTris.getMutableCudaPtr();
	in.ppsTrisIndex = m_ppsTrisIndex.getMutableCudaPtr();
	in.sortTris     = m_sortTris.getMutableCudaPtr();

	// Set parent task containing all the work
	TaskBVH all;
	all.triStart     = 0;
	all.triEnd       = arraySize;
	//all.bbox         = bbox;
	all.step         = 0;
	all.lock         = 0;
	all.bestCost     = 1e38f;
	all.depth        = 0;
	all.parentIdx    = -1;
	all.nodeIdx      = 0;
	all.taskID       = 0;
	all.pivot        = arraySize / 2;
#ifdef DEBUG_INFO
	all.sync         = 0;
	all.parent       = -1;
	all.clockStart   = 0;
	all.clockEnd     = 0;
#endif

	all.type         = TaskType_Sort_PPS1;
	all.unfinished   = warpSubtasks(arraySize);
	all.origSize     = all.unfinished;

	m_taskData.setRange(TASK_SIZE * sizeof(int), &all, sizeof(TaskBVH)); // Set the first task

	// Set parent task header
	m_taskData.setRange(0, &all.unfinished, sizeof(int)); // Set the first task

	// Prepare the task stack
	TaskStackBVH& tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getMutablePtr();
	tasks.header     = (int*)m_taskData.getMutableCudaPtr();
	tasks.tasks      = (TaskBVH*)m_taskData.getMutableCudaPtr(TASK_SIZE * sizeof(int));
	tasks.nodeTop     = 1;
	tasks.top        = 0;
	tasks.bottom     = 0;
	memset(tasks.active, 0, sizeof(int)*(ACTIVE_MAX+1));
	tasks.activeTop  = 1;
	//tasks.empty[0] = 0;
	//int j = 1;
	//for(int i = EMPTY_MAX; i > 0; i--, j++)
	//	tasks.empty[i] = j;
	tasks.emptyTop  = 0;
	tasks.emptyBottom  = 0;
	tasks.unfinished = -1; // We are waiting for one task to finish = task all
	tasks.sizePool = TASK_SIZE;
	tasks.sizeNodes = m_bvhData.getSize()/sizeof(CudaKdtreeNode);
	tasks.sizeTris = m_trisIndexOut.getSize()/sizeof(S32);

	// Determine block and grid sizes.
#ifdef ONE_WARP_RUN
	Vec2i blockSize(WARP_SIZE, 1); // threadIdx.x must equal the thread lane in warp
	Vec2i gridSize(1, 1); // Number of SMs * Number of blocks?
#else
	int numWarpsPerBlock = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numWarpsPerBlock");
	int numBlocksPerSM = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numBlockPerSM");
	Vec2i blockSize(WARP_SIZE, numWarpsPerBlock); // threadIdx.x must equal the thread lane in warp
	int gridSizeX = NUM_SM*numBlocksPerSM;
	Vec2i gridSize(gridSizeX, 1); // Number of SMs * Number of blocks?

	if(gridSizeX*numWarpsPerBlock != NUM_WARPS)
		printf("\aNUM_WARPS constant does not match the launch parameters\n");
#endif

	m_debug.resizeDiscard(blockSize.y*gridSize.x*sizeof(float4));
	m_debug.clear();
	in.debug = m_debug.getMutableCudaPtr();

	// Launch.
	float tKernel = m_module->launchKernelTimed(kernel, blockSize, gridSize, false, 0, false);

#ifndef BENCHMARK
	cuCtxSynchronize(); // Flushes printfs
#endif

	// Verify sort
	S32* tsort = (S32*)m_trisIndex.getPtr();
	for(int i=0; i < arraySize; i++)
	{
		if(*tsort != i)
		{
			printf("Sort error %d instead of %d\n", *tsort, i);
			break;
		}
		tsort++;
	}

	Debug << "\nSort in " << tKernel << "\n\n";

	tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();
	int* header = (int*)m_taskData.getPtr();
	printPoolHeader(&tasks, header, blockSize.y*gridSize.x, sprintf(""));

	Debug << "\n\nTasks" << "\n";
	TaskBVH* task = (TaskBVH*)m_taskData.getPtr(TASK_SIZE*sizeof(int));
	int stackMax = 0;
	int maxDepth = 0;
	int syncCount = 0;
	int maxTaskId = -1;
	long double sumTris = 0;
	long double maxTris = 0;

	int sortTasks = 0;
	long double cntSortTris = 0;

	int subFailed = 0;

	for(int i = 0; i < TASK_SIZE; i++)
	{
		if(task[i].nodeIdx != TaskHeader_Empty || task[i].parentIdx != TaskHeader_Empty)
		{
			Debug << "Task " << i << "\n";
			Debug << "Header: " << header[i] << "\n";
			Debug << "Unfinished: " << task[i].unfinished << "\n";
			Debug << "Type: " << task[i].type << "\n";
			Debug << "TriStart: " << task[i].triStart << "\n";
			Debug << "TriEnd: " << task[i].triEnd << "\n";
			Debug << "TriRight: " << task[i].triRight << "\n";
			Debug << "ParentIdx: " << task[i].parentIdx << "\n";
			Debug << "NodeIdx: " << task[i].nodeIdx << "\n";
			Debug << "TaskID: " << task[i].taskID << "\n";
			Debug << "Depth: " << task[i].depth << "\n";
#ifdef DEBUG_INFO
			//Debug << "Step: " << task[i].step << "\n";
			//Debug << "Lock: " << task[i].lock << "\n";
			//Debug << "SubFailure: " << task[i].subFailureCounter << "\n";
			Debug << "GMEMSync: " << task[i].sync << "\n";
			Debug << "Parent: " << task[i].parent << "\n";
#endif
			Debug << "Triangles: " << task[i].triEnd - task[i].triStart << "\n";
			Debug << "Pivot: " << task[i].pivot << "\n";
			
			Debug << "\n";
			stackMax = i;

#ifdef CUTOFF_DEPTH
			if(task[i].depth == m_cutOffDepth)
			{
#endif
			long double tris = task[i].triEnd - task[i].triStart;
			if(tris > maxTris)
			{
				maxTris = tris;
				maxTaskId = i;
			}
			sumTris += tris;
			sortTasks++;
			cntSortTris += tris;
#ifdef CUTOFF_DEPTH
			}
#endif

#ifdef DEBUG_INFO
			maxDepth = max(task[i].depth, maxDepth);
			syncCount += task[i].sync;
#endif
		}
	}

	if(stackMax == TASK_SIZE-1)
		printf("\aIncomplete result!\n");
#ifdef CUTOFF_DEPTH
	Debug << "\n\nStatistics for cutoff depth " << m_cutOffDepth << "\n\n";
#else
	Debug << "\n\n";
#endif

#ifdef DEBUG_INFO
	Debug << "Avg naive task height (tris) = " << sumTris/(long double)sortTasks << "\n";
	Debug << "Max naive task height (tris) = " << maxTris << ", taskId: " << maxTaskId << "\n";
	Debug << "Cnt sorted operations = " << sortTasks << "\n";
	double cntTrisLog2Tris = (double(arraySize) * (double)(logf(arraySize)/logf(2.0f)));
	Debug << "Cnt sorted triangles = " << cntSortTris << "\n";	
	Debug << "Cnt sorted triangles/(N log N), N=#tris = " << cntSortTris/cntTrisLog2Tris << "\n";
	Debug << "\n";
	Debug << "Max task depth = " << maxDepth << "\n";
	Debug << "Cnt gmem synchronizations: " << syncCount << "\n";
	Debug << "Leafs failed to subdivide = " << subFailed << " (*3) => total useless tasks " << subFailed * 3 << "\n";
#endif

	Debug << "max_queue_length = " << stackMax << "\n\n" << "\n";

	return tKernel;
}

F32 CudaNoStructTracer::traceOnDemandBVHRayBuffer(RayBuffer& rays, bool rebuild)
{
	CUfunction kernel;
	kernel = m_module->getKernel("build");
	if (!kernel)
		fail("Build kernel not found!");

	// Prepare the task data
	if(rebuild)
	{
		initPool(0, &rays.getRayBuffer(), &m_bvhData);
	}

	RtEnvironment& cudaEnv = *(RtEnvironment*)m_module->getGlobal("c_env").getMutablePtr();
	cudaEnv.subdivThreshold = (m_bbox.SurfaceArea() / (float)m_numRays) * ((float)cudaEnv.optCt/10.0f);

	// Set BVH input.
	KernelInputBVH& inBVH = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
	inBVH.numTris		= m_numTris;
	inBVH.tris         = m_trisCompact.getCudaPtr();
	inBVH.trisIndex    = m_trisIndex.getMutableCudaPtr();
	//inBVH.trisBox      = m_trisBox.getCudaPtr();
	inBVH.ppsTrisBuf   = m_ppsTris.getMutableCudaPtr();
	inBVH.ppsTrisIndex = m_ppsTrisIndex.getMutableCudaPtr();
	inBVH.sortTris   = m_sortTris.getMutableCudaPtr();
#ifdef COMPACT_LAYOUT
	inBVH.trisOut      = m_trisCompactOut.getMutableCudaPtr();
	inBVH.trisIndexOut = m_trisIndexOut.getMutableCudaPtr();
#endif

	// Set traversal input
	CUdeviceptr nodePtr     = m_bvhData.getCudaPtr();
	CUdeviceptr triPtr      = m_trisCompact.getCudaPtr();
	Buffer&     indexBuf    = m_trisIndex;
	Vec2i       nodeOfsA    = Vec2i(0, (S32)m_bvhData.getSize());
	Vec2i       triOfsA     = Vec2i(0, (S32)m_trisCompact.getSize());

	// Stop the timer for this copy as it is stopped in other algorithms as well
	m_timer.end();
	KernelInput& in = *(KernelInput*)m_module->getGlobal("c_in").getMutablePtr();
	m_timer.start();
	in.numRays      = rays.getSize();
	in.anyHit       = (rays.getNeedClosestHit() == false);
	in.nodesA       = nodePtr + nodeOfsA.x;
	in.trisA        = triPtr + triOfsA.x;
	in.rays         = rays.getRayBuffer().getCudaPtr();
	in.results      = rays.getResultBuffer().getMutableCudaPtr();
	in.triIndices   = indexBuf.getCudaPtr();
	
	if(rebuild)
	{
#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
		SplitRed split;
		for(int i = 0; i < 2; i++)
		{
			split.children[i].bbox.m_mn = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
			split.children[i].bbox.m_mx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			split.children[i].cnt = 0;
		}

		SplitArray sArray;
		for(int i = 0; i < NUM_WARPS; i++)
		{
			for(int j = 0; j < PLANE_COUNT; j++)
				sArray.splits[i][j] = split;
		}
#else
		SplitRed split;
		for(int i = 0; i < 2; i++)
		{
			//split.children[i].bbox.m_mn = make_float3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
			//split.children[i].bbox.m_mx = make_float3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
			split.children[i].bbox.m_mn = make_int3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
			split.children[i].bbox.m_mx = make_int3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
			split.children[i].cnt = 0;
		}

		SplitArray sArray;
		for(int j = 0; j < PLANE_COUNT; j++)
			sArray.splits[j] = split;

		m_splitData.setRange(0, &sArray, sizeof(SplitArray)); // Set the first splits
#endif

		m_splitData.setRange(TASK_SIZE * sizeof(SplitArray), &sArray, sizeof(SplitArray)); // Set the last splits for copy
#endif

		m_bvhData.clearRange32(0, UNBUILD_FLAG, sizeof(CudaBVHNode)); // Set the root as unbuild

		CudaAABB bbox;
		memcpy(&bbox.m_mn, &m_bbox.min, sizeof(float3));
		memcpy(&bbox.m_mx, &m_bbox.max, sizeof(float3));

		// Set parent task containing all the work
		TaskBVH all;
		all.triStart     = 0;
		all.triLeft      = 0;
		all.triRight     = m_numTris;
		all.triEnd       = m_numTris;
		all.bbox         = bbox;
		all.step         = 0;
		all.lock         = LockType_Free;
		all.bestCost     = 1e38f;
		all.depth        = 0;
#ifndef MALLOC_SCRATCHPAD
		all.triIdxCtr    = 0;
#endif
		all.parentIdx    = -1;
		all.nodeIdx      = 0;
		all.taskID       = 0;
		Vector3 size     = m_bbox.Diagonal();
		all.axis         = size.MajorAxis();
		all.pivot        = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.subtreeLimit");
		all.terminatedBy = TerminatedBy_None;
#ifdef DEBUG_INFO
		all.sync         = 0;
		all.parent       = -1;
		all.clockStart   = 0;
		all.clockEnd     = 0;
#endif
		all.cached       = LockType_None; // Mark this task as cached

#if SPLIT_TYPE == 0
#if SCAN_TYPE == 0
		all.type         = TaskType_Sort_PPS1;
#elif SCAN_TYPE == 1
		all.type         = TaskType_Sort_PPS1_Up;
#elif SCAN_TYPE == 2 ||  SCAN_TYPE == 3
		all.type         = TaskType_Sort_SORT1;
#endif
		all.unfinished   = warpSubtasks(m_numTris);
		float pos = m_bbox.min[all.axis] + m_bbox.Size(all.axis)/2.0f;
		if(all.axis == 0)
			all.splitPlane   = make_float4(1.f, 0.f, 0.f, -pos);
		else if(all.axis == 1)
			all.splitPlane   = make_float4(0.f, 1.f, 0.f, -pos);
		else
			all.splitPlane   = make_float4(0.f, 0.f, 1.f, -pos);
#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
		all.type         = TaskType_InitMemory;
		all.unfinished   = warpSubtasks(sizeof(SplitArray)/sizeof(int));
#else
		all.type         = TaskType_BinTriangles;
		all.unfinished   = (warpSubtasks(m_numTris)+BIN_MULTIPLIER-1)/BIN_MULTIPLIER;
#endif
#endif
		all.origSize     = all.unfinished;

		m_taskData.setRange(TASK_SIZE * sizeof(int), &all, sizeof(TaskBVH)); // Set the first task

		// Set parent task header
		m_taskData.setRange(0, &all.unfinished, sizeof(int)); // Set the first task
	}

	// Prepare the task stack
	TaskStackBVH& tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getMutablePtr();
	
	tasks.header     = (int*)m_taskData.getMutableCudaPtr();
	tasks.tasks      = (TaskBVH*)m_taskData.getMutableCudaPtr(TASK_SIZE * sizeof(int));
	tasks.launchFlag = 0;
	
	if(rebuild)
	{
		tasks.nodeTop     = 1;
		tasks.triTop     = 0;
		tasks.top        = 0;
		tasks.bottom     = 0;
		//memset(tasks.active, 0, sizeof(int)*(ACTIVE_MAX+1));
		memset(tasks.active, -1, sizeof(int)*(ACTIVE_MAX+1));
		tasks.active[0] = 0;
		/*for(int i = 0; i < ACTIVE_MAX+1; i++)
			tasks.active[i] = i;*/
		tasks.activeTop  = 1;
		//tasks.empty[0] = 0;
		//int j = 1;
		//for(int i = EMPTY_MAX; i > 0; i--, j++)
		//	tasks.empty[i] = j;
		memset(tasks.empty, 0, sizeof(int)*(EMPTY_MAX+1));
		tasks.emptyTop  = 0;
		tasks.emptyBottom  = 0;
		tasks.numSortedTris = 0;
		tasks.numNodes = 0;
		tasks.numLeaves = 0;
		tasks.numEmptyLeaves = 0;
		tasks.sizePool = TASK_SIZE;
		tasks.sizeNodes = m_bvhData.getSize()/sizeof(CudaKdtreeNode);
		tasks.sizeTris = m_trisIndexOut.getSize()/sizeof(S32);
		memset(tasks.leafHist, 0, sizeof(tasks.leafHist));
	}
	/*else
	{
		tasks.emptyTop  = 3;
	}*/

	tasks.warpCounter = rays.getSize();
	tasks.unfinished = -NUM_WARPS; // We are waiting for one task to finish = task all

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
	// Prepare split stack
	SplitArray* &splits = *(SplitArray**)m_module->getGlobal("g_redSplits").getMutablePtr();
	splits = (SplitArray*)m_splitData.getMutableCudaPtr();
#endif

	CudaBVHNode* &bvh = *(CudaBVHNode**)m_module->getGlobal("g_bvh").getMutablePtr();
	bvh = (CudaBVHNode*)m_bvhData.getMutableCudaPtr();

	// Determine block and grid sizes.
#ifdef ONE_WARP_RUN
	Vec2i blockSize(WARP_SIZE, 1); // threadIdx.x must equal the thread lane in warp
	Vec2i gridSize(1, 1); // Number of SMs * Number of blocks?
	int numWarps = 1;
#else
	int numWarpsPerBlock = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numWarpsPerBlock");
	int numBlocksPerSM = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numBlockPerSM");
	Vec2i blockSize(WARP_SIZE, numWarpsPerBlock); // threadIdx.x must equal the thread lane in warp
	int gridSizeX = NUM_SM*numBlocksPerSM;
	int numWarps = numWarpsPerBlock*gridSizeX;
	Vec2i gridSize(gridSizeX, 1); // Number of SMs * Number of blocks?

	if(gridSizeX*numWarpsPerBlock != NUM_WARPS)
		printf("\aNUM_WARPS constant does not match the launch parameters\n");
#endif

	m_debug.resizeDiscard(blockSize.y*gridSize.x*sizeof(float4));
	m_debug.clear();
	inBVH.debug = m_debug.getMutableCudaPtr();

	// Launch.
	//cuFuncSetSharedSize(kernel, 0); // Set shared memory to force some launch configurations
	float tKernel = m_module->launchKernelTimed(kernel, blockSize, gridSize);

#ifndef BENCHMARK
	cuCtxSynchronize(); // Flushes printfs
#endif

	tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();
	if(tasks.unfinished != 0 || tasks.top > tasks.sizePool || tasks.nodeTop > m_bvhData.getSize() / sizeof(CudaBVHNode) || tasks.triTop > m_trisIndexOut.getSize() / sizeof(S32)) // Something went fishy
		tKernel = 1e38f;

	//Debug << "\nBuild in " << tKernel << "\n\n";

#ifndef BENCHMARK
	printPool(tasks, numWarps);

	/*Debug << "\n\nBVH" << "\n";
	CudaBVHNode* nodes = (CudaBVHNode*)m_bvhData.getPtr();

	for(int i = 0; i < tasks.nodeTop; i++)
	{
		Debug << "Node " << i << "\n";
		Debug << "BoxLeft: (" << nodes[i].c0xy.x << ", " << nodes[i].c0xy.z << ", " << nodes[i].c01z.x << ") - ("
				<< nodes[i].c0xy.y << ", " << nodes[i].c0xy.w << ", " << nodes[i].c01z.y << ")\n";
		Debug << "BoxRight: (" << nodes[i].c1xy.x << ", " << nodes[i].c1xy.z << ", " << nodes[i].c01z.z << ") - ("
				<< nodes[i].c1xy.y << ", " << nodes[i].c1xy.w << ", " << nodes[i].c01z.w << ")\n";
		Debug << "Children: " << nodes[i].children.x << ", " << nodes[i].children.y << "\n\n";
	}*/
#endif

	return tKernel;
}

F32 CudaNoStructTracer::traceOnDemandKdtreeRayBuffer(RayBuffer& rays, bool rebuild)
{
	CUfunction kernel;
	kernel = m_module->getKernel("build");
	if (!kernel)
		fail("Build kernel not found!");

	// Prepare the task data
	if(rebuild)
	{
		initPool(0, &rays.getRayBuffer(), &m_bvhData);
	}

	RtEnvironment& cudaEnv = *(RtEnvironment*)m_module->getGlobal("c_env").getMutablePtr();
	cudaEnv.subdivThreshold = (m_bbox.SurfaceArea() / (float)m_numRays) * ((float)cudaEnv.optCt/10.0f);
	float k1 = Environment::GetSingleton()->GetFloat("SubdivisionRayCaster.depthK1");
	float k2 = Environment::GetSingleton()->GetFloat("SubdivisionRayCaster.depthK2");
	cudaEnv.optMaxDepth = k1 * log2((F32)m_numTris) + k2;
	//cudaEnv.failureCount = 0.2f*cudaEnv.optMaxDepth + 1.0f;
#ifndef BENCHMARK
	if(rebuild)
	{
		printf("Maximum depth = %d\n", cudaEnv.optMaxDepth);
		printf("Failure count = %d\n", cudaEnv.failureCount);
	}
#endif

	// Set BVH input.
	KernelInputBVH& inBVH = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
	inBVH.numTris		= m_numTris;
	inBVH.tris         = m_trisCompact.getCudaPtr();
	inBVH.trisIndex    = m_trisIndex.getMutableCudaPtr();
#ifndef INTERLEAVED_LAYOUT
	inBVH.trisOut      = m_trisCompactOut.getMutableCudaPtr();
	inBVH.trisIndexOut = m_trisIndexOut.getMutableCudaPtr();
#endif

	// Set traversal input
	CUdeviceptr nodePtr     = m_bvhData.getCudaPtr();
	Vec2i       nodeOfsA    = Vec2i(0, (S32)m_bvhData.getSize());
#ifndef INTERLEAVED_LAYOUT
	CUdeviceptr triPtr      = m_trisCompactOut.getCudaPtr();
	Vec2i       triOfsA     = Vec2i(0, (S32)m_trisCompactOut.getSize());
	Buffer&     indexBuf    = m_trisIndexOut;
#else
	CUdeviceptr triPtr      = m_bvhData.getCudaPtr();
	Vec2i       triOfsA     = Vec2i(0, (S32)m_bvhData.getSize());
	Buffer&     indexBuf    = m_bvhData;
#endif

	// Stop the timer for this copy as it is stopped in other algorithms as well
	m_timer.end();
	KernelInput& in = *(KernelInput*)m_module->getGlobal("c_in").getMutablePtr();
	m_timer.start();
	in.numRays      = rays.getSize();
	in.anyHit       = (rays.getNeedClosestHit() == false);
	memcpy(&in.bmin, &m_bbox.min, sizeof(float3));
	memcpy(&in.bmax, &m_bbox.max, sizeof(float3));
	in.nodesA       = nodePtr + nodeOfsA.x;
	in.trisA        = triPtr + triOfsA.x;
	in.rays         = rays.getRayBuffer().getCudaPtr();
	in.results      = rays.getResultBuffer().getMutableCudaPtr();
	in.triIndices   = indexBuf.getCudaPtr();

	// Set texture references.
	m_module->setTexRef("t_rays", rays.getRayBuffer(), CU_AD_FORMAT_FLOAT, 4);
	m_module->setTexRef("t_nodesI", nodePtr + nodeOfsA.x, nodeOfsA.y, CU_AD_FORMAT_FLOAT, 4);
	//m_module->setTexRef("t_trisA", triPtr + triOfsA.x, triOfsA.y, CU_AD_FORMAT_FLOAT, 4);
	//m_module->setTexRef("t_triIndices", indexBuf, CU_AD_FORMAT_SIGNED_INT32, 1);
	
	if(rebuild)
	{
		int baseOffset = setDynamicMemory();
		inBVH = *(KernelInputBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();

#if SPLIT_TYPE == 3
	m_splitData.clearRange(0, 0, sizeof(SplitInfoTri)); // Set first split to zeros
#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
		SplitRed split;
		for(int i = 0; i < 2; i++)
		{
			split.children[i].bbox.m_mn = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
			split.children[i].bbox.m_mx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
			split.children[i].cnt = 0;
		}

		SplitArray sArray;
		for(int i = 0; i < NUM_WARPS; i++)
		{
			for(int j = 0; j < PLANE_COUNT; j++)
				sArray.splits[i][j] = split;
		}
#else
		//SplitRed split;
		//for(int i = 0; i < 2; i++)
		//{
		//	//split.children[i].bbox.m_mn = make_float3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
		//	//split.children[i].bbox.m_mx = make_float3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
		//	split.children[i].bbox.m_mn = make_int3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
		//	split.children[i].bbox.m_mx = make_int3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
		//	split.children[i].cnt = 0;
		//}

		//SplitArray sArray;
		//for(int j = 0; j < PLANE_COUNT; j++)
		//	sArray.splits[j] = split;

		//m_splitData.setRange(0, &sArray, sizeof(SplitArray)); // Set the first splits
		m_splitData.clearRange(0, 0, sizeof(SplitInfoTri)); // Set first split to zeros
#endif

		//m_splitData.setRange(TASK_SIZE * sizeof(SplitArray), &sArray, sizeof(SplitArray)); // Set the last splits for copy
		// Prepare split stack
		//SplitArray* &splits = *(SplitArray**)m_module->getGlobal("g_redSplits").getMutablePtr();
		//splits = (SplitArray*)m_splitData.getMutableCudaPtr();

		SplitInfoTri* &splits = *(SplitInfoTri**)m_module->getGlobal("g_splitStack").getMutablePtr();
		splits = (SplitInfoTri*)m_splitData.getMutableCudaPtr();
#endif

		m_bvhData.clearRange32(0, UNBUILD_FLAG, sizeof(CudaKdtreeNode)); // Set the root as unbuild

		CudaAABB bbox;
		memcpy(&bbox.m_mn, &m_bbox.min, sizeof(float3));
		memcpy(&bbox.m_mx, &m_bbox.max, sizeof(float3));

		// Set parent task containing all the work
		TaskBVH all;
		all.triStart     = 0;
		all.triLeft      = 0;
		all.triRight     = 0;
		all.triEnd       = m_numTris;
		all.bbox         = bbox;
		all.step         = 0;
		all.lock         = LockType_Free;
		all.bestCost     = 1e38f;
		all.depth        = 0;
		all.dynamicMemory= baseOffset;
#ifdef MALLOC_SCRATCHPAD
		all.subFailureCounter = 0;
#endif
		all.parentIdx    = -1;
		all.nodeIdx      = 0;
		all.taskID       = 0;
		Vector3 size     = m_bbox.Diagonal();
		all.axis         = size.MajorAxis();
		all.pivot        = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.subtreeLimit");
		all.terminatedBy = TerminatedBy_None;
#ifdef DEBUG_INFO
		all.sync         = 0;
		all.parent       = -1;
		all.clockStart   = 0;
		all.clockEnd     = 0;
#endif
		all.cached       = LockType_None; // Mark this task as cached

#if SPLIT_TYPE == 0
#if SCAN_TYPE == 0
		all.type         = TaskType_Sort_PPS1;
#elif SCAN_TYPE == 1
		all.type         = TaskType_Sort_PPS1_Up;
#elif SCAN_TYPE == 2 ||  SCAN_TYPE == 3
		all.type         = TaskType_Sort_SORT1;
#endif
		all.unfinished   = warpSubtasks(m_numTris);
		float pos = m_bbox.min[all.axis] + m_bbox.Size(all.axis)/2.0f;
		if(all.axis == 0)
			all.splitPlane   = make_float4(1.f, 0.f, 0.f, -pos);
		else if(all.axis == 1)
			all.splitPlane   = make_float4(0.f, 1.f, 0.f, -pos);
		else
			all.splitPlane   = make_float4(0.f, 0.f, 1.f, -pos);
#elif SPLIT_TYPE == 1
	all.type = TaskType_Split;
#if 0 // SQRT candidates
	int evaluatedCandidates = (int)sqrtf(m_numTris);
	int evaluatedCandidates = 1;
	int numPlanes = 0.5f * m_numTris/evaluatedCandidates;
#elif 0 // Fixed candidates
	int numPlanes = 32768;
#else // All candidates
	int numPlanes = m_numTris*6; // Number of warp sized subtasks
#endif
	all.unfinished = warpSubtasks(numPlanes); // This must be the same as in the GPU code
#elif SPLIT_TYPE == 2
	all.type = TaskType_Split;
	all.unfinished = 1;
#elif SPLIT_TYPE == 3
	all.type = TaskType_SplitParallel;
	int evaluatedRays = warpSubtasks((int)sqrtf(m_numRays));
	int evaluatedTris = warpSubtasks((int)sqrtf(m_numTris));
	all.unfinished = PLANE_COUNT*(evaluatedRays+evaluatedTris); // Each WARP_SIZE rays and tris add their result to one plane
#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
		all.type         = TaskType_InitMemory;
		all.unfinished   = warpSubtasks(sizeof(SplitArray)/sizeof(int));
#else
		all.type         = TaskType_BinTriangles;
		all.unfinished   = (warpSubtasks(m_numTris)+BIN_MULTIPLIER-1)/BIN_MULTIPLIER;
#endif
#endif
		all.origSize     = all.unfinished;

		m_taskData.setRange(TASK_SIZE * sizeof(int), &all, sizeof(TaskBVH)); // Set the first task

		// Set parent task header
		m_taskData.setRange(0, &all.unfinished, sizeof(int)); // Set the first task
	}

	// Prepare the task stack
	TaskStackBVH& tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getMutablePtr();
	
	tasks.header     = (int*)m_taskData.getMutableCudaPtr();
	tasks.tasks      = (TaskBVH*)m_taskData.getMutableCudaPtr(TASK_SIZE * sizeof(int));
	tasks.launchFlag = 0;
	
	if(rebuild)
	{
#ifndef INTERLEAVED_LAYOUT
		tasks.nodeTop    = 1;
#else
		tasks.nodeTop    = sizeof(CudaKdtreeNode);
#endif
		tasks.triTop     = 0;
		tasks.top        = 0;
		tasks.bottom     = 0;
		//memset(tasks.active, 0, sizeof(int)*(ACTIVE_MAX+1));
		memset(tasks.active, -1, sizeof(int)*(ACTIVE_MAX+1));
		tasks.active[0] = 0;
		/*for(int i = 0; i < ACTIVE_MAX+1; i++)
			tasks.active[i] = i;*/
		tasks.activeTop  = 1;
		//tasks.empty[0] = 0;
		//int j = 1;
		//for(int i = EMPTY_MAX; i > 0; i--, j++)
		//	tasks.empty[i] = j;
		memset(tasks.empty, 0, sizeof(int)*(EMPTY_MAX+1));
		tasks.emptyTop  = 0;
		tasks.emptyBottom  = 0;
		tasks.numSortedTris = 0;
		tasks.numNodes = 0;
		tasks.numLeaves = 0;
		tasks.numEmptyLeaves = 0;
		tasks.sizePool = TASK_SIZE;
		tasks.sizeNodes = m_bvhData.getSize()/sizeof(CudaKdtreeNode);
		tasks.sizeTris = m_trisIndexOut.getSize()/sizeof(S32);
		memset(tasks.leafHist, 0, sizeof(tasks.leafHist));
	}
	/*else
	{
		tasks.emptyTop  = 3;
	}*/

	tasks.warpCounter = rays.getSize();
#ifndef ONDEMAND_FULL_BUILD
	tasks.unfinished = -NUM_WARPS; // We are waiting for all trace warps to finish
#else
	tasks.unfinished = -1; // We are waiting for one task to finish = task all
#endif

	CudaKdtreeNode* &kdtree = *(CudaKdtreeNode**)m_module->getGlobal("g_kdtree").getMutablePtr();
	kdtree = (CudaKdtreeNode*)m_bvhData.getMutableCudaPtr();

	// Determine block and grid sizes.
#ifdef ONE_WARP_RUN
	Vec2i blockSize(WARP_SIZE, 1); // threadIdx.x must equal the thread lane in warp
	Vec2i gridSize(1, 1); // Number of SMs * Number of blocks?
	int numWarps = 1;
#else
	int numWarpsPerBlock = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numWarpsPerBlock");
	int numBlocksPerSM = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numBlockPerSM");
	Vec2i blockSize(WARP_SIZE, numWarpsPerBlock); // threadIdx.x must equal the thread lane in warp
	int gridSizeX = NUM_SM*numBlocksPerSM;
	int numWarps = numWarpsPerBlock*gridSizeX;
	Vec2i gridSize(gridSizeX, 1); // Number of SMs * Number of blocks?

	if(gridSizeX*numWarpsPerBlock != NUM_WARPS)
		printf("\aNUM_WARPS constant does not match the launch parameters\n");
#endif

	m_debug.resizeDiscard(blockSize.y*gridSize.x*sizeof(float4));
	m_debug.clear();
	inBVH.debug = m_debug.getMutableCudaPtr();

	// Launch.
	//cuFuncSetSharedSize(kernel, 0); // Set shared memory to force some launch configurations
	float tKernel = 0.f;
#ifndef DUPLICATE_REFERENCES
	if(rebuild)
		tKernel += convertWoop();
#endif
	tKernel += m_module->launchKernelTimed(kernel, blockSize, gridSize);

/*#ifdef MALLOC_SCRATCHPAD
	CUfunction kernelDealloc = m_module->getKernel("deallocFreeableMemory", 0);
	if (!kernelDealloc)
	fail("Memory allocation kernel not found!");

	F32 deallocTime = m_module->launchKernelTimed(kernelDealloc, Vec2i(1,1), Vec2i(1, 1));

	printf("Memory freed in %f\n", deallocTime);
#endif*/

#ifndef BENCHMARK
	cuCtxSynchronize(); // Flushes printfs
#endif

	tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();
#ifndef INTERLEAVED_LAYOUT
	if(tasks.unfinished != 0 || tasks.top > tasks.sizePool || tasks.nodeTop > m_bvhData.getSize() / sizeof(CudaKdtreeNode) || tasks.triTop > m_trisIndexOut.getSize() / sizeof(S32)) // Something went fishy
#else
	if(tasks.unfinished != 0 || tasks.nodeTop > m_bvhData.getSize()) // Something went fishy
#endif
		tKernel = 1e38f;

	//Debug << "\nBuild in " << tKernel << "\n\n";

#ifndef BENCHMARK
	printPool(tasks, numWarps);

	/*Debug << "\n\nBVH" << "\n";
	CudaBVHNode* nodes = (CudaBVHNode*)m_bvhData.getPtr();

	for(int i = 0; i < tasks.nodeTop; i++)
	{
		Debug << "Node " << i << "\n";
		Debug << "BoxLeft: (" << nodes[i].c0xy.x << ", " << nodes[i].c0xy.z << ", " << nodes[i].c01z.x << ") - ("
				<< nodes[i].c0xy.y << ", " << nodes[i].c0xy.w << ", " << nodes[i].c01z.y << ")\n";
		Debug << "BoxRight: (" << nodes[i].c1xy.x << ", " << nodes[i].c1xy.z << ", " << nodes[i].c01z.z << ") - ("
				<< nodes[i].c1xy.y << ", " << nodes[i].c1xy.w << ", " << nodes[i].c01z.w << ")\n";
		Debug << "Children: " << nodes[i].children.x << ", " << nodes[i].children.y << "\n\n";
	}*/
#endif

	return tKernel;
}

F32 CudaNoStructTracer::traceCpuRayBuffer(RayBuffer& rb)
{
	const Ray* rays = (const Ray*)rb.getRayBuffer().getPtr();
	RayResult* results = (RayResult*)rb.getResultBuffer().getMutablePtr();
	for(int rid=0; rid < rb.getSize(); rid++)
	{
		if(rid % 10000 == 0) printf("rid: %d\n",rid);
		traceCpuRay(rays[rid], results[rid], !rb.getNeedClosestHit());
	}

	return 0;
}

void CudaNoStructTracer::traceCpuRay(const Ray& r, RayResult& result, bool anyHit)
{
	const Vec4f *t_trisA      = (Vec4f*)(m_trisCompact.getPtr());
	const S32   *t_trisIndices = (S32*)(m_trisIndex.getPtr());

	int   hitIndex;
	float hitT;
	float hitU;
	float hitV;
	float tmin;

	hitIndex          = -1;
	hitT              = r.tmax;
	hitU              = 0;
	hitV              = 0;
	tmin			  = 0;

	// naive over all triangles
	for (int triAddr = 0; triAddr < m_numTris * 3 ; triAddr += 3)
	{
		const Vec3f &v00 = t_trisA[triAddr + 0].getXYZ();
		const Vec3f &v11 = t_trisA[triAddr + 1].getXYZ();
		const Vec3f &v22 = t_trisA[triAddr + 2].getXYZ();

		Vec3f nrmN = cross(v11-v00,v22-v00);
		const float den = dot(nrmN,r.direction);

		if(den >= 0.0f)
			continue;

		const float deni = 1.0f / den;
		const Vec3f org0 = v00-r.origin;
		float t = dot(nrmN,org0)*deni;

		if (t > tmin && t < hitT)
		{
			const Vec3f crossProd = cross(r.direction,org0);
			const float v = dot(v00-v22,crossProd)*deni;
			if (v >= 0.0f && v <= 1.0f)
			{
				const float u = -dot(v00-v11,crossProd)*deni;
				if (u >= 0.0f && u + v <= 1.0f)
				{
					hitT = t;
					hitU = u;
					hitV = v;
					hitIndex = triAddr;
				}
			}
		}
	}

	if(hitIndex != -1)
		hitIndex = hitIndex / 3;

	result.id = hitIndex;
	result.t  = hitT;
	result.u  = hitU;
	result.v  = hitV;
}

void CudaNoStructTracer::saveBufferSizes(bool ads, bool aux)
{
	float MB = (float)(1024*1024);

	if(ads)
	{
		m_sizeADS    = m_bvhData.getSize()/MB;
#ifndef COMPACT_LAYOUT
		m_sizeTri    = m_trisCompact.getSize()/MB;
		m_sizeTriIdx = m_trisIndex.getSize()/MB;
#else
		m_sizeTri    = m_trisCompactOut.getSize()/MB;
		m_sizeTriIdx = m_trisIndexOut.getSize()/MB;
#endif
	}

	if(aux)
	{
		m_sizeTask   = m_taskData.getSize()/MB;
		m_sizeSplit  = m_splitData.getSize()/MB;
#ifdef MALLOC_SCRATCHPAD
#if !defined(ATOMIC_MALLOC) && !defined(SCATTER_ALLOC) && !defined(CIRCULAR_MALLOC)
		size_t heapSize;
		cuCtxGetLimit(&heapSize, CU_LIMIT_MALLOC_HEAP_SIZE);
		m_heap       = heapSize/MB; 
#else
		m_heap       = (m_mallocData.getSize()+m_mallocData2.getSize())/MB;
#endif
#else
		m_heap       = 0.f;
#endif
	}
}

void CudaNoStructTracer::prepareDynamicMemory()
{
	// Set the memory limit according to triangle count
	//U64 allocSize = (U64)m_trisIndex.getSize()*15ULL;
	//U64 allocSize = (U64)m_trisIndex.getSize()*20ULL;
	U64 allocSize = (U64)m_trisIndex.getSize()*150ULL;
	//U64 allocSize = (U64)m_trisIndex.getSize()*200ULL;

#if defined(SCATTER_ALLOC) || defined(FDG_ALLOC)
	U64 allocSize = max(allocSize, 8ULL*1024ULL*1024ULL);
#endif

#if !defined(ATOMIC_MALLOC) && !defined(SCATTER_ALLOC) && !defined(CIRCULAR_MALLOC)
	cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, allocSize);
#elif defined(ATOMIC_MALLOC) || defined(CIRCULAR_MALLOC)
	m_mallocData.resizeDiscard(allocSize);
#ifdef WITH_SCATTER_ALLOC
	m_mallocData2.resizeDiscard(allocSize);
#endif
#elif defined(SCATTER_ALLOC)
	m_mallocData.resizeDiscard(allocSize);
#endif

#if defined(SCATTER_ALLOC) || defined(WITH_SCATTER_ALLOC)
	// CUDA Driver API cannot deal with templates -> use C++ mangled name
	CUfunction initHeap = m_module->getKernel("_ZN8GPUTools8initHeapILj4096ELj8ELj16ELj2ELb0ELb1EEEvPNS_10DeviceHeapIXT_EXT0_EXT1_EXT2_EXT3_EXT4_EEEPvj", 2*sizeof(CUdeviceptr)+sizeof(int));
	if (!initHeap)
		fail("Scatter alloc initialization kernel not found!");

	int offset = 0;
	offset += m_module->setParamPtr(initHeap, offset, m_module->getGlobal("theHeap").getMutableCudaPtr());
#ifdef WITH_SCATTER_ALLOC
	offset += m_module->setParamPtr(initHeap, offset, m_mallocData2.getMutableCudaPtr());
#else
	offset += m_module->setParamPtr(initHeap, offset, m_mallocData.getMutableCudaPtr());
#endif
	offset += m_module->setParami(initHeap, offset, allocSize);
	F32 initTime = m_module->launchKernelTimed(initHeap, Vec2i(256,1), Vec2i(1, 1));

	printf("Scatter alloc initialized in %f\n", initTime);
#endif
}

int CudaNoStructTracer::setDynamicMemory()
{
	int baseOffset = 0;
#if !defined(ATOMIC_MALLOC) && !defined(CIRCULAR_MALLOC)
	CUfunction kernelAlloc = m_module->getKernel("allocFreeableMemory", 2*sizeof(int));
	if (!kernelAlloc)
		fail("Memory allocation kernel not found!");

	int offset = 0;
	offset += m_module->setParami(kernelAlloc, offset, m_numTris);
	offset += m_module->setParami(kernelAlloc, offset, 0);
	F32 allocTime = m_module->launchKernelTimed(kernelAlloc, Vec2i(1,1), Vec2i(1, 1));

#ifndef BENCHMARK
	printf("Memory allocated in %f\n", allocTime);
#endif
#else
	// Set the heapBase, heapOffset and heapSize
	char*& heapBase = *(char**)m_module->getGlobal("g_heapBase").getMutablePtr();
	heapBase = (char*)m_mallocData.getMutableCudaPtr();
	int& heapOffset = *(int*)m_module->getGlobal("g_heapOffset").getMutablePtr();
#if SCAN_TYPE < 2
	heapOffset = 4*m_numTris*sizeof(int);
#else
	heapOffset = m_numTris*sizeof(int);
#endif

	int& heapSize = *(int*)m_module->getGlobal("g_heapSize").getMutablePtr();
	heapSize = m_mallocData.getSize();

#if defined(CIRCULAR_MALLOC)
#ifndef DOUBLY_LINKED
	int headerSize = 2*sizeof(int);
#else
	int headerSize = 3*sizeof(int);
#endif
	heapOffset += headerSize;

	int& heapLock = *(int*)m_module->getGlobal("g_heapLock").getMutablePtr();
	heapLock = 0;

#ifndef DOUBLY_LINKED
	Vec2i first(LockType_Set, heapOffset); // Locked, allocated memory for parent
	m_mallocData.setRange(0, &first, sizeof(Vec2i)); // Set the first header
#else
	Vec3i first(LockType_Set, heapSize-headerSize, heapOffset); // Locked, allocated memory for parent
	m_mallocData.setRange(0, &first, sizeof(Vec3i)); // Set the first header
#endif

#ifdef GLOBAL_HEAP_LOCK
#ifndef DOUBLY_LINKED
	Vec2i second(LockType_Free, heapSize-headerSize); // Unlocked, next at the end of allocated memory
	m_mallocData.setRange(heapOffset, &second, sizeof(Vec2i)); // Set the second header
#else
	Vec3i second(LockType_Free, 0, heapSize-headerSize); // Unlocked, next at the end of allocated memory
	m_mallocData.setRange(heapOffset, &second, sizeof(Vec3i)); // Set the second header
#endif
#else
#if 0
	// Create regular chunks
	int numChunks = m_mallocData.getSize()/heapOffset;
	for(int i = 1; i < numChunks; i++)
	{
	#ifndef DOUBLY_LINKED
	Vec2i next(0, (i+1)*heapOffset); // Unlocked, next at the multiple of heapOffset
	m_mallocData.setRange(i*heapOffset, &next, sizeof(Vec2i)); // Set the next header
	#else
	Vec3i next(0, (i-1)*heapOffset, (i+1)*heapOffset); // Unlocked, next at the multiple of heapOffset
	m_mallocData.setRange(i*heapOffset, &next, sizeof(Vec3i)); // Set the next header
	#endif
	}

#else
	// Create hierarchical chunks
	int delta = ((int)(heapOffset)+headerSize+3) & -4;
	int prevOfs = 0;
	int ofs;
	int i = 0;
	int lvl = 2;
	for(ofs = heapOffset; true; ofs += delta, i++)
	{
		if(i == lvl) // New level in BFS order
		{
			delta = ((int)(delta * 0.8f)+headerSize+3) & -4;
			i = 0;
			lvl *= 2;
		}

		if(ofs+delta >= heapSize-2*headerSize) // We cannot make another chunk
			break;

#ifndef DOUBLY_LINKED
		Vec2i next(LockType_Free, ofs+delta); // Unlocked, next at the multiple of heapOffset
		m_mallocData.setRange(ofs, &next, sizeof(Vec2i)); // Set the next header
#else
		Vec3i next(LockType_Free, prevOfs, ofs+delta); // Unlocked, next at the multiple of heapOffset
		m_mallocData.setRange(ofs, &next, sizeof(Vec3i)); // Set the next header
#endif

		prevOfs = ofs;
	}
#endif

#ifndef DOUBLY_LINKED
	Vec2i last(LockType_Free, heapSize-headerSize); // Unlocked, next at the end of allocated memory
	m_mallocData.setRange(ofs, &last, sizeof(Vec2i)); // Set the last header
#else
	Vec3i last(LockType_Free, prevOfs, heapSize-headerSize); // Unlocked, next at the end of allocated memory
	m_mallocData.setRange(ofs, &last, sizeof(Vec3i)); // Set the last header
#endif
#endif

#ifndef DOUBLY_LINKED
	Vec2i tail(LockType_Set, 0); // Locked, next at the start of heap
	m_mallocData.setRange(heapSize-headerSize, &tail, sizeof(Vec2i)); // Set the last header
#else
	Vec3i tail(LockType_Set, ofs, 0); // Locked, next at the start of heap
	m_mallocData.setRange(heapSize-headerSize, &tail, sizeof(Vec3i)); // Set the last header
#endif

	// Offset of the memory allocation
	baseOffset = headerSize;

#ifdef WITH_SCATTER_ALLOC
	// With scatter alloc
	char*& heapBase2 = *(char**)m_module->getGlobal("g_heapBase2").getMutablePtr();
	heapBase2 = (char*)m_mallocData2.getMutableCudaPtr();
#endif
#endif

	int offset;
#endif

	CUfunction kernelMemCpyIndex = m_module->getKernel("MemCpyIndex", sizeof(CUdeviceptr)+sizeof(int));
	if (!kernelMemCpyIndex)
		fail("Memory copy kernel not found!");

	int memSize = m_trisIndex.getSize()/sizeof(int);
	offset = 0;
	offset += m_module->setParamPtr(kernelMemCpyIndex, offset, m_trisIndex.getCudaPtr());
	offset += m_module->setParami(kernelMemCpyIndex, offset, memSize);
	F32 memcpyTime = m_module->launchKernelTimed(kernelMemCpyIndex, Vec2i(256,1), Vec2i((memSize-1+256)/256, 1));

#ifndef BENCHMARK
	printf("Triangle indices copied in %f\n", memcpyTime);
#endif

#ifdef SCATTER_ALLOC
	CUdeviceptr& heap = *(CUdeviceptr*)m_module->getGlobal("g_heapBase").getMutablePtr();
	CUdeviceptr base = m_mallocData.getMutableCudaPtr();
	baseOffset = heap - base;
	heap = base;
	//if(heap != m_mallocData.getCudaPtr())
	//	printf("Wrong base address!\n");
#endif

	return baseOffset;
}


F32 CudaNoStructTracer::convertWoop()
{
	// Convert woop triangles
	CUfunction kernelCreateWoop = m_module->getKernel("createWoop", 2*sizeof(CUdeviceptr)+sizeof(int));
	if (!kernelCreateWoop)
		fail("Regular triangle to Woop triangle conversion kernel not found!");

	int offset = 0;
	offset += m_module->setParamPtr(kernelCreateWoop, offset, m_trisCompact.getCudaPtr());
	offset += m_module->setParamPtr(kernelCreateWoop, offset, m_trisCompactOut.getMutableCudaPtr());
	offset += m_module->setParami(kernelCreateWoop, offset, m_numTris);
	F32 woopTime = m_module->launchKernelTimed(kernelCreateWoop, Vec2i(256,1), Vec2i((m_numTris-1+256)/256, 1));

#ifndef BENCHMARK
	printf("Woop triangles created in %f\n", woopTime);
#endif

	return woopTime;
}

void CudaNoStructTracer::resetBuffers(bool resetADSBuffers)
{
	// Reset buffers so that reuse of space does not cause timing disturbs
	if(resetADSBuffers)
	{
		m_bvhData.reset();
		m_trisCompactOut.reset();
		m_trisIndexOut.reset();
	}

	m_mallocData.reset();
	m_mallocData2.reset();
	m_taskData.reset();
	m_splitData.reset();

	m_raysIndex.reset();

	m_ppsTris.reset();
	m_ppsTrisIndex.reset();
	m_sortTris.reset();
	m_ppsRays.reset();
	m_ppsRaysIndex.reset();
	m_sortRays.reset();
}

void CudaNoStructTracer::trimBVHBuffers()
{
	// Save sizes of auxiliary buffers so that they can be printed
	saveBufferSizes(false, true);
	// Free auxiliary buffers
	resetBuffers(false);

	// Resize to exact memory
	U32 nN, nL, eL, sT, bT, tT, sTr; 
	getStats(nN, nL, eL, sT, bT, tT, sTr);
#ifdef COMPACT_LAYOUT
	m_bvhData.resize(nN * sizeof(CudaBVHNode));
	m_trisCompactOut.resize(tT*3*sizeof(float4) + nL*sizeof(float4));
	m_trisIndexOut.resize(tT*3*sizeof(int) + nL*sizeof(int));
#else
	m_bvhData.resize((nN + nL) * sizeof(CudaBVHNode));
#endif

	// Save sizes of ads buffers so that they can be printed
	saveBufferSizes(true, false);
}

void CudaNoStructTracer::trimKdtreeBuffers()
{
	// Save sizes of auxiliary buffers so that they can be printed
	saveBufferSizes(false, true);
	// Free auxiliary buffers
	resetBuffers(false);

	// Resize to exact memory
	U32 nN, nL, eL, sT, nT, tT, sTr; 
	getStats(nN, nL, eL, sT, nT, tT, sTr);
#ifndef INTERLEAVED_LAYOUT
#ifndef COMPACT_LAYOUT
	getStats(nN, nL, eL, sT, nT, tT, sTr, false);
	m_bvhData.resize((nN + nL) * sizeof(CudaKdtreeNode));
	m_trisCompactOut.resize(tT*3*sizeof(float4));
	m_trisIndexOut.resize(tT*3*sizeof(int));
#else
#ifdef DUPLICATE_REFERENCES
	m_bvhData.resize(nN * sizeof(CudaKdtreeNode));
	m_trisCompactOut.resize(tT*3*sizeof(float4) + nL*sizeof(float4));
	m_trisIndexOut.resize(tT*3*sizeof(int) + nL*sizeof(int));
#else
	m_bvhData.resize(nN * sizeof(CudaKdtreeNode));
	m_trisIndexOut.resize(tT*sizeof(int) + nL*sizeof(int));
#endif
#endif
#else
	//m_bvhData.resize((nN + nL) * sizeof(CudaKdtreeNode) + tT*3*sizeof(float4) + tT*3*sizeof(int));
	m_bvhData.resize(nT);
#endif

	// Save sizes of ads buffers so that they can be printed
	saveBufferSizes(true, false);
}

void CudaNoStructTracer::getStats(U32& nodes, U32& leaves, U32& emptyLeaves, U32& stackTop, U32& nodeTop, U32& tris, U32& sortedTris, bool sub)
{
	TaskStackBVH tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();

#ifndef INTERLEAVED_LAYOUT
#ifndef BVH_COUNT_NODES
#ifndef COMPACT_LAYOUT
	nodes = tasks.nodeTop / 2;
	leaves = tasks.nodeTop - nodes;
#else
	nodes = tasks.nodeTop;
	leaves = tasks.triTop;
	emptyLeaves = 0;
#endif
#else // BVH_COUNT_NODES
	nodes = tasks.numNodes;
	leaves = tasks.numLeaves;
	emptyLeaves = tasks.numEmptyLeaves;
#endif // BVH_COUNT_NODES

#ifdef COMPACT_LAYOUT
	tris = tasks.triTop;
	if(sub)
		tris -= (leaves-emptyLeaves);
#ifdef DUPLICATE_REFERENCES
	tris /= 3;
#endif
#else
	if(sub)
	{
		tris = m_numTris;
	}
	else
	{
		tris = tasks.triTop;
		tris /= 3;
	}
#endif
#else
#ifndef BVH_COUNT_NODES
	nodes = tasks.nodeTop / 2;
	leaves = tasks.nodeTop - nodes;
	emptyLeaves = 0;
#else // BVH_COUNT_NODES
	nodes = tasks.numNodes;
	leaves = tasks.numLeaves;
	emptyLeaves = tasks.numEmptyLeaves;
#endif // BVH_COUNT_NODES

	tris = tasks.nodeTop - (nodes+leaves)*sizeof(CudaKdtreeNode); // Subtract node memory
	tris /= 3*sizeof(float4)+sizeof(int); // Only approximate because of padding
#endif

	nodeTop = tasks.nodeTop;
	sortedTris = tasks.numSortedTris;
	stackTop = tasks.top;
}

void CudaNoStructTracer::getSizes(F32& task, F32& split, F32& ads, F32& tri, F32& triIdx, F32& heap)
{
	task = m_sizeTask;
	split = m_sizeSplit;
	ads = m_sizeADS;
	tri = m_sizeTri;
	triIdx = m_sizeTriIdx;
	heap = m_heap;
}
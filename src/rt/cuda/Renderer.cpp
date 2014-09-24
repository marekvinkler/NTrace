#include "cuda/Renderer.hpp"
#include "cuda/RendererKernels.hpp"
#include "gui/Window.hpp"
#include "io/File.hpp"
#include "bvh/HLBVH/HLBVHBuilder.hpp"

using namespace FW;

//------------------------------------------------------------------------

Renderer::Renderer(AccelStructType as, Environment* env)
:   m_raygen            (1 << 20),

    m_window            (NULL),
    m_enableRandom      (false),

    m_mesh              (NULL),
    m_scene             (NULL),

	m_sampleImage		(NULL),
    m_image             (NULL),
    m_cameraFar         (0.0f),

    m_newBatch          (true),
    m_batchRays         (NULL),
    m_batchStart        (0),

	m_accelStruct		(NULL),

	m_asType			(as),
	m_vis				(NULL),
	m_showVis			(false)
{
	m_env = env;

	if (m_asType == tKDTree)
	{
		m_cudaTracer = new CudaKDTreeTracer();
	}
	else
	{
		m_cudaTracer = new CudaBVHTracer();
	}
	m_cudaTracer->setScene(NULL);

    m_compiler.setSourceFile("src/rt/cuda/RendererKernels.cu");
    m_compiler.addOptions("-use_fast_math");
    m_compiler.include("src/rt");
    m_compiler.include("src/framework");
    m_bvhCachePath = "bvhcache";

    m_platform = Platform("GPU");
    m_platform.setLeafPreferences(1, 1);
}

//------------------------------------------------------------------------

Renderer::~Renderer(void)
{
    setMesh(NULL);

	delete m_sampleImage;
    delete m_image;
	delete m_cudaTracer;
	delete m_accelStruct;
}

//------------------------------------------------------------------------

void Renderer::setMesh(MeshBase* mesh)
{
    // Same mesh => done.

    if (mesh == m_mesh)
        return;

    // Deinit scene and BVH.

    delete m_scene;
    m_scene = NULL;
	
    invalidateBVH();

    // Create scene.

    m_mesh = mesh;
    if (mesh)
	{
        m_scene = new Scene(*mesh);
		m_cudaTracer->setScene(m_scene);
	}
}

//------------------------------------------------------------------------

void Renderer::setParams(const Params& params)
{
    m_params = params;

	m_cudaTracer->setKernel(params.kernelName);
}

//------------------------------------------------------------------------

CudaAS* Renderer::getCudaBVH(void)
{
	string bvhBuilder;
	m_env->GetStringValue("BVHBuilder", bvhBuilder);

    // BVH is already valid => done.

    BVHLayout layout = m_cudaTracer->getDesiredBVHLayout();
    if (!m_mesh || (m_accelStruct && m_accelStruct->getLayout() == layout))
        return m_accelStruct;

    // Deinit.

    delete m_accelStruct;
    m_accelStruct = NULL;

    // Setup build parameters.

    BVH::Stats stats;
    m_buildParams.stats = &stats;

	// If we use HLBVH, use HLBVH

	if (bvhBuilder == "HLBVH")
	{
		HLBVHParams params;
		params.hlbvh = true;
		params.hlbvhBits = 4;
		params.leafSize = 8;
		params.epsilon = 0.001f;
		HLBVHBuilder* bvh = new HLBVHBuilder(m_scene, m_platform, params);
		return bvh;
	}

    // Determine cache file name.

    String cacheFileName = sprintf("%s/%08x.dat", m_bvhCachePath.getPtr(), hashBits(
        m_scene->hash(),
        m_platform.computeHash(),
        m_buildParams.computeHash(),
        layout,
		m_asType));

    // Cache file exists => import.

    if (!hasError())
    {
        File file(cacheFileName, File::Read);
        if (!hasError())
        {
            m_accelStruct = new CudaBVH(file);
            return m_accelStruct;
        }
        clearError();
    }

    // Display status.

    printf("\nBuilding BVH...\nThis will take a while.\n");
    if (m_window)
        m_window->showModalMessage("Building BVH...");

    // Build BVH.

    BVH bvh(m_scene, m_platform, m_buildParams, m_env);
    stats.print();
    m_accelStruct = new CudaBVH(bvh, layout);
    failIfError();

    // Write to cache.

    if (!hasError())
    {
        CreateDirectory(m_bvhCachePath.getPtr(), NULL);
        File file(cacheFileName, File::Create);
        m_accelStruct->serialize(file);
        clearError();
    }

    // Display status.

    printf("Done.\n\n");
    return m_accelStruct;
}

//------------------------------------------------------------------------

CudaAS*	Renderer::getCudaKDTree(void)
{
	BVHLayout layout = m_cudaTracer->getDesiredBVHLayout();
	if (!m_mesh || (m_accelStruct && m_accelStruct->getLayout() == layout))
		return m_accelStruct;

	String cacheFileName = sprintf("%s/%08x.dat", m_bvhCachePath.getPtr(), hashBits(
		m_scene->hash(),
		m_platform.computeHash(),
		m_buildParams.computeHash(),
		layout,
		m_asType));

	if(!hasError())
	{
		File file(cacheFileName, File::Read);
		if (!hasError())
		{
			m_accelStruct = new CudaKDTree(file);
			return m_accelStruct;
		}
		clearError();
	}

	delete m_accelStruct;
	m_accelStruct = NULL;

	printf("\nBuilding k-d tree...\nThis will take a while.\n");
    if (m_window)
        m_window->showModalMessage("Building k-d tree...");

	KDTree::BuildParams params;
	params.enablePrints = m_buildParams.enablePrints;
	params.stats = new KDTree::Stats();
	params.builder = KDTree::SAH;

	KDTree kdtree(m_scene, m_platform, params);
	m_accelStruct = new CudaKDTree(kdtree);

	failIfError();

    // Write to cache.

    //if (!hasError())
    //{
    //    CreateDirectory(m_bvhCachePath.getPtr(), NULL);
    //    File file(cacheFileName, File::Create);
    //    m_accelStruct->serialize(file);
    //    clearError();
    //}

	params.stats->print();

	return m_accelStruct;
}

//------------------------------------------------------------------------

F32 Renderer::renderFrame(GLContext* gl, const CameraControls& camera)
{
    F32 launchTime = 0.0f;
    beginFrame(gl, camera);
    while (nextBatch())
    {
        launchTime += traceBatch();
        updateResult();
    }
    displayResult(gl);
	CameraControls& c = const_cast<CameraControls&>(camera);
	if(m_showVis && m_vis != NULL) // If visualization is enabled
		m_vis->draw(gl, c);

    return launchTime;
}

//------------------------------------------------------------------------

void Renderer::beginFrame(GLContext* gl, const CameraControls& camera)
{
    FW_ASSERT(gl && m_mesh);

    // Setup BVH.
	if(m_asType == tBVH)
		m_cudaTracer->setBVH(getCudaBVH());
	else
		m_cudaTracer->setBVH(getCudaKDTree());

    // Setup result image.

    const Vec2i& size = gl->getViewSize();
    if (!m_image || m_image->getSize() != size)
    {
		delete m_sampleImage;
		m_sampleImage = new Image(size, ImageFormat::RGBA_Vec4f);
		m_sampleImage->getBuffer().setHints(Buffer::Hint_CudaGL);
		m_sampleImage->clear();
        delete m_image;
        m_image = new Image(size, ImageFormat::ABGR_8888);
        m_image->getBuffer().setHints(Buffer::Hint_CudaGL);
        m_image->clear();
    }

    // Generate primary rays.

    m_raygen.primary(m_primaryRays,
        camera.getPosition(),
        invert(gl->xformFitToView(-1.0f, 2.0f) * camera.getWorldToClip()),
        size.x, size.y,
        camera.getFar());

    // Secondary rays enabled => trace primary rays.

	if (m_params.rayType != RayType_Primary && m_params.rayType != RayType_Textured && m_params.rayType != RayType_PathTracing)
	{
		m_cudaTracer->traceBatch(m_primaryRays);
	}

    // Initialize state.

    m_cameraFar     = camera.getFar();
    m_newBatch      = true;
    m_batchRays     = NULL;
    m_batchStart    = 0;
}

//------------------------------------------------------------------------

bool Renderer::nextBatch(void)
{
    FW_ASSERT(m_scene);

    // Clean up the previous batch.

    if (m_batchRays)
        m_batchStart += m_batchRays->getSize();
    m_batchRays = NULL;

    // Generate new batch.

    U32 randomSeed = (m_enableRandom) ? m_random.getU32() : 0;
    switch (m_params.rayType)
    {
    case RayType_Primary:
	case RayType_Textured:
	case RayType_PathTracing:
        if (!m_newBatch)
            return false;
        m_newBatch = false;
        m_batchRays = &m_primaryRays;
        break;

    case RayType_AO:
        if (!m_raygen.ao(m_secondaryRays, m_primaryRays, *m_scene, m_params.numSamples, m_params.aoRadius, m_newBatch, randomSeed))
            return false;
        m_batchRays = &m_secondaryRays;
        break;

    case RayType_Diffuse:
        if (!m_raygen.ao(m_secondaryRays, m_primaryRays, *m_scene, m_params.numSamples, m_cameraFar, m_newBatch, randomSeed))
            return false;
        m_secondaryRays.setNeedClosestHit(true);
        m_batchRays = &m_secondaryRays;
        break;

    default:
        FW_ASSERT(false);
        return false;
    }

    // Sort rays.

    if (m_params.sortSecondary && (m_params.rayType != RayType_Primary || m_params.rayType != RayType_Textured || m_params.rayType != RayType_PathTracing))
        m_batchRays->mortonSort();
    return true;
}

//------------------------------------------------------------------------

F32 Renderer::traceBatch(void)
{
    FW_ASSERT(m_batchRays);

	return m_cudaTracer->traceBatch(*m_batchRays);
}

//------------------------------------------------------------------------

void Renderer::updateResult(void)
{
    FW_ASSERT(m_scene && m_image && m_batchRays);
	
	m_sampleCount += 1.0f;

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Setup input struct.

    ReconstructInput& in    = *(ReconstructInput*)module->getGlobal("c_ReconstructInput").getMutablePtr();
    in.numRaysPerPrimary    = (m_params.rayType == RayType_Primary || m_params.rayType == RayType_Textured || m_params.rayType == RayType_PathTracing) ? 1 : m_params.numSamples;
    in.firstPrimary         = m_batchStart / in.numRaysPerPrimary;
    in.numPrimary           = m_batchRays->getSize() / in.numRaysPerPrimary;
    in.isPrimary            = (m_params.rayType == RayType_Primary);
    in.isAO                 = (m_params.rayType == RayType_AO);
    in.isDiffuse            = (m_params.rayType == RayType_Diffuse);
	in.isTextured           = (m_params.rayType == RayType_Textured);
	in.isPathTraced         = (m_params.rayType == RayType_PathTracing);
    in.primarySlotToID      = m_primaryRays.getSlotToIDBuffer().getCudaPtr();
    in.primaryResults       = m_primaryRays.getResultBuffer().getCudaPtr();
    in.batchIDToSlot        = m_batchRays->getIDToSlotBuffer().getCudaPtr();
    in.batchResults         = m_batchRays->getResultBuffer().getCudaPtr();
    in.triMaterialColor     = m_scene->getTriMaterialColorBuffer().getCudaPtr();
    in.triShadedColor       = m_scene->getTriShadedColorBuffer().getCudaPtr();
    in.pixels               = m_image->getBuffer().getMutableCudaPtr();

	in.texCoords			= m_scene->getVtxTexCoordBuffer().getCudaPtr();
	in.normals				= m_scene->getVtxNormalBuffer().getCudaPtr();
	in.triVertIndex			= m_scene->getTriVtxIndexBuffer().getCudaPtr();
	in.atlasInfo			= m_scene->getTextureAtlasInfo().getCudaPtr();
	in.matId				= m_scene->getMaterialIds().getCudaPtr();
	in.matInfo				= m_scene->getMaterialInfo().getCudaPtr();
	in.outputColor			= m_sampleImage->getBuffer().getMutableCudaPtr();
	in.samplesCount			= m_sampleCount;

	module->setTexRef("t_textures", *m_scene->getTextureAtlas()->getAtlasTexture().getImage(), true, true, true, false);

    // Launch.

    module->getKernel("reconstructKernel").launch(in.numPrimary);
}

//------------------------------------------------------------------------

void Renderer::displayResult(GLContext* gl)
{
    FW_ASSERT(gl);
    Mat4f oldXform = gl->setVGXform(Mat4f());
    glPushAttrib(GL_ENABLE_BIT);
    glDisable(GL_DEPTH_TEST);
    gl->drawImage(*m_image, Vec2f(0.0f), 0.5f, false);
    gl->setVGXform(oldXform);
    glPopAttrib();
}

//------------------------------------------------------------------------

int Renderer::getTotalNumRays(void)
{
    // Casting primary rays => no degenerates.

    if (m_params.rayType == RayType_Primary || m_params.rayType == RayType_Textured || m_params.rayType == RayType_PathTracing)
        return m_primaryRays.getSize();

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Set input and output.

    CountHitsInput& in = *(CountHitsInput*)module->getGlobal("c_CountHitsInput").getMutablePtr();
    in.numRays = m_primaryRays.getSize();
    in.rayResults = m_primaryRays.getResultBuffer().getCudaPtr();
    in.raysPerThread = 32;
    module->getGlobal("g_CountHitsOutput").clear();

    // Count primary ray hits.

    module->getKernel("countHitsKernel").launch(
        (in.numRays - 1) / in.raysPerThread + 1,
        Vec2i(CountHits_BlockWidth, CountHits_BlockHeight));

    int numHits = *(S32*)module->getGlobal("g_CountHitsOutput").getPtr();

    // numSecondary = secondaryPerPrimary * primaryHits

    return numHits * m_params.numSamples;
}

//------------------------------------------------------------------------

F32 Renderer::calcNodeSAHCostKdtree(const Platform& platform, Buffer* nodes, Buffer* tris, S32 n, AABB bbox, S32 depth, S32& maxDepth, S32& sumDepth, S32& numNodes, S32& numLeaves, F32& nodeArea, F32 &weightedLeafArea, F32& test)
{
	numNodes++;
	if(depth > maxDepth)
		maxDepth = depth;
	sumDepth += depth;

	F32 pa = bbox.area();

	if((n & KDTREE_MASK) == KDTREE_EMPTYLEAF) // Empty leaf
	{
		return calcLeafSAHCostNum(platform, 0, numLeaves)*pa;
	}

	if(n >= 0) // Inner node
	{
		Vec4i cell = ((Vec4i*)nodes)[n];
		unsigned int type = cell.w & KDTREE_MASK;
		float split = *(float*)&cell.z;

		AABB bboxLeft = bbox;
		AABB bboxRight = bbox;
		
		switch((type >> KDTREE_DIMPOS))
		{
		case 0:
			bboxLeft.max().x = split;
			bboxRight.min().x = split;
			break;

		case 1:
			bboxLeft.max().y = split;
			bboxRight.min().y = split;
			break;

		case 2:
			bboxLeft.max().z = split;
			bboxRight.min().z = split;
			break;
		}

		S32 nl = cell.x;
		S32 nr = cell.y;

		nodeArea += pa;		

		weightedLeafArea += calcNodeSAHCostKdtree(platform, nodes, tris, nl, bboxLeft, depth+1, maxDepth, sumDepth, numNodes, numLeaves, nodeArea, weightedLeafArea, test);
		weightedLeafArea += calcNodeSAHCostKdtree(platform, nodes, tris, nr, bboxRight, depth+1, maxDepth, sumDepth, numNodes, numLeaves, nodeArea, weightedLeafArea, test);

		if(depth == 0)
		{
			F32 pa = bbox.area();
			return platform.getNodeCost(1) * nodeArea/pa + weightedLeafArea/pa;
		}

		return 0.f;
	}
	else // Leaf
	{
		return calcLeafSAHCostCompact(platform, tris, ~n, numLeaves)*pa;
	}
}

//------------------------------------------------------------------------

F32 Renderer::calcLeafSAHCostCompact(const Platform& platform, Buffer* triIdx, S32 n, S32& numLeaves)
{
	numLeaves++;

	int* idx = (int*)triIdx + n;

	S32 cnt = 0;
	while(idx[cnt] != 0x80000000)
		cnt++;

	return platform.getTriangleCost(cnt);
}

//------------------------------------------------------------------------

F32 Renderer::calcLeafSAHCostNum(const Platform& platform, S32 n, S32& numLeaves)
{
	numLeaves++;
	return platform.getTriangleCost(0);
}

//------------------------------------------------------------------------

void Renderer::startBVHVis(void)
{
	if(m_window == NULL)
		return;

	if(m_vis != NULL)
		endBVHVis();

	// Trace the primary rays so that we have correct rays to visualize
	// TODO: Enable secondary rays visualization too?
	RayStats stats;
	//m_as->setTraceParams(&m_platform, m_scene);
	//setTracerBVH(m_bvh);
	//traceBatch(m_primaryRays, &stats);
	//m_newBatch = true;
	//m_batchRays = NULL;
	//m_batchStart = 0;
	//m_sampleIndex = 0;
	//m_secondaryIndex = 0;

	//m_buildParams.visibility = &getVisibleTriangles(m_scene->getNumTriangles(), true); // Get visibility buffer
	//if(m_params.rayType != RayType_Primary)
	//{
	//	traceBatch();
	//}
	//while (nextBatch())
	//{
	//	traceBatch();
	//}

	m_vis = new VisualizationKDTree((CudaKDTree*)m_accelStruct, m_scene, Array<AABB>());
	m_vis->setVisible(true);
	m_window->addListener(m_vis);

	m_showVis = true;
}

//------------------------------------------------------------------------

void Renderer::endBVHVis(void)
{
	if(m_window != NULL)
		m_window->removeListener(m_vis);
	delete m_vis;
	m_vis = NULL;

	m_showVis = false;
}

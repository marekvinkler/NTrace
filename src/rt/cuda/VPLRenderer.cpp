#include "VPLRenderer.hpp"

#include "cuda/RendererKernels.hpp"

using namespace FW;

VPLrenderer::VPLrenderer()
	:	Renderer(),
		firstFrame(true)
{

}


F32 VPLrenderer::renderFrame(GLContext* gl, const CameraControls& camera) 
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

void VPLrenderer::beginFrame(GLContext* gl, const CameraControls& camera) 
{
	FW_ASSERT(gl && m_mesh);

	string ds;
	Environment::GetSingleton()->GetStringValue("Renderer.dataStructure", ds);

    // Setup BVH.
	if(ds == "BVH" || ds == "PersistentBVH")
		m_cudaTracer->setBVH(getCudaBVH(gl, camera));
	else if(ds == "KDTree")
		m_cudaTracer->setBVH(getCudaKDTree());
	else
		fail("Incorrect data structure type!");

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
	
    U32 randomSeed = (m_enableRandom) ? m_random.getU32() : 0;
    m_raygen.primary(m_primaryRays,
        camera.getPosition(),
        invert(gl->xformFitToView(-1.0f, 2.0f) * camera.getWorldToClip()),
        size.x, size.y,
        camera.getFar(), randomSeed);
	m_primaryRays.setNeedClosestHit(true);

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

bool VPLrenderer::nextBatch(void) 
{
	if (!m_newBatch)
        return false;
    m_newBatch = false;
    m_batchRays = &m_primaryRays;
	return true;
}

void VPLrenderer::updateResult(void) 
{
	FW_ASSERT(m_scene && m_image && m_batchRays);
	
	m_sampleCount += 1.0f;

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Setup input struct.

    VPLReconstructInput& in    = *(VPLReconstructInput*)module->getGlobal("c_VPLReconstructInput").getMutablePtr();
	in.numPrimary           = m_primaryRays.getSize();
    in.primarySlotToID      = m_primaryRays.getSlotToIDBuffer().getCudaPtr();
    in.primaryResults       = m_primaryRays.getResultBuffer().getCudaPtr();
    in.pixels               = m_image->getBuffer().getMutableCudaPtr();

	in.texCoords			= m_scene->getVtxTexCoordBuffer().getCudaPtr();
	in.normals				= m_scene->getVtxNormalBuffer().getCudaPtr();
	in.triVertIndex			= m_scene->getTriVtxIndexBuffer().getCudaPtr();
	in.vertices				= m_scene->getVtxPosBuffer().getCudaPtr();
	in.triShadedColor       = m_scene->getTriShadedColorBuffer().getCudaPtr();

	//module->setTexRef("t_textures", *m_scene->getTextureAtlas()->getAtlasTexture().getImage(), true, true, true, false);

    // Launch.

    module->getKernel("vplReconstructKernel").launch(in.numPrimary);
}
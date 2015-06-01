#include "VPLRenderer.hpp"

#include "cuda/RendererKernels.hpp"

#include "3d/Light.hpp"

using namespace FW;

VPLRenderer::VPLRenderer()
	:	Renderer(),
		m_firstFrame(true),
		m_material(Vec3f(0.7f, 0.7f, 0.7f)),
		m_meshChanged(false),
		m_shadowSamples(1)
{
	Environment::GetSingleton()->GetIntValue("VPL.primaryLights", m_lightCount);
	Environment::GetSingleton()->GetIntValue("VPL.maxLightBounces", m_lightBounces);
	
	m_lights.resize(m_lightCount * sizeof(Vec3f));
}

VPLRenderer::~VPLRenderer() {
	
}

void VPLRenderer::setMesh(MeshBase* mesh) {
	
	m_meshChanged = (mesh != m_mesh);

	Renderer::setMesh(mesh);

}

int VPLRenderer::getTotalNumRays(void)
{
	return m_primaryRays.getSize() * (m_lights.getSize() / sizeof(Light)) + m_primaryRays.getSize();
}

F32 VPLRenderer::renderFrame(GLContext* gl, const CameraControls& camera) 
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

void VPLRenderer::beginFrame(GLContext* gl, const CameraControls& camera) 
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

		m_pixels.resize(size.x * size.y * sizeof(Vec4f));
    }

	m_image->clear();
	m_pixels.clear();

	// Generate VPLs for the first time
	if(m_firstFrame || m_meshChanged) {

		m_lights.clear();

		Random rand(156);

		m_firstFrame = false;
		m_raygen.primaryVPL(m_lights, m_vplBuffer, m_scene, m_lightCount, m_lightBounces, 10000.0f, rand.getU32());

		for (int i = 0; i < m_lightCount; i++) {
			Light* lightBuffer = (Light*)m_lights.getPtr();
		}

		m_cudaTracer->traceBatch(m_vplBuffer);

		for(int i = 0; i <= m_lightBounces; i++) {
			m_raygen.reflectedVPL(m_lights, m_vplBuffer, m_lightCount, i, m_scene, 10000.0f, rand.getU32());
			if(i < m_lightBounces) {
				m_cudaTracer->traceBatch(m_vplBuffer);
			}
		}


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

	if (m_params.rayType != RayType_Textured && m_params.rayType != RayType_PathTracing)
	{
		m_cudaTracer->traceBatch(m_primaryRays);
	}

    // Initialize state.

    m_cameraFar     = camera.getFar();
    m_newBatch      = true;
    m_batchRays     = NULL;
    m_batchStart    = 0;
	m_currentLight	= 0;
}

bool VPLRenderer::nextBatch(void) 
{

	if (m_batchRays)
		m_batchStart += m_batchRays->getSize();
	m_batchRays = NULL;

	Light* lights = (Light*)m_lights.getPtr();

	if(m_params.rayType == RayType_VPL) {
		
		if(!m_raygen.shadow(m_shadowRays, m_primaryRays, m_shadowSamples,lights[m_currentLight].position , 0, m_newBatch)) {
			if(m_currentLight == (m_lightCount * (m_lightBounces + 2)) - 1){
				return false;
			} else {
				m_newBatch = true;
				m_batchStart = 0;
				m_currentLight++;
				return nextBatch();
			}
		} else {
			m_batchRays = &m_shadowRays;
		}
	} else {
		if (!m_newBatch)
	        return false;

		if(!m_raygen.shadow(m_shadowRays, m_primaryRays, m_shadowSamples, lights[0].position, 0, m_newBatch)) {
			return false;
		}

		m_batchRays = &m_shadowRays;
	}


	return true;
}

void VPLRenderer::updateResult(void) 
{
	FW_ASSERT(m_scene && m_image && m_batchRays);
	
	m_sampleCount += 1.0f;

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Setup input struct.
    VPLReconstructInput& in    = *(VPLReconstructInput*)module->getGlobal("c_VPLReconstructInput").getMutablePtr();
	in.currentLight			= m_currentLight;
	in.lights				= m_lights.getCudaPtr();
	in.lightCount			= m_lightCount * (m_lightBounces + 2);
	in.firstPrimary			= m_batchStart;
	in.numPrimary           = m_primaryRays.getSize();
    in.primarySlotToID      = m_primaryRays.getSlotToIDBuffer().getCudaPtr();
    in.primaryResults       = m_primaryRays.getResultBuffer().getCudaPtr();
	in.primaryRays			= m_primaryRays.getRayBuffer().getCudaPtr();
	in.pixels				= m_pixels.getMutableCudaPtr();
	in.shadowSamples		= m_shadowSamples;
	in.shadowResults		= m_shadowRays.getResultBuffer().getCudaPtr();
	in.shadowIdToSlot		= m_shadowRays.getIDToSlotBuffer().getCudaPtr();

	in.texCoords			= m_scene->getVtxTexCoordBuffer().getCudaPtr();
	in.normals				= m_scene->getVtxNormalBuffer().getCudaPtr();
	in.triVertIndex			= m_scene->getTriVtxIndexBuffer().getCudaPtr();
	in.vertices				= m_scene->getVtxPosBuffer().getCudaPtr();
	in.atlasInfo			= m_scene->getTextureAtlasInfo().getCudaPtr();
	in.matId				= m_scene->getMaterialIds().getCudaPtr();
	in.matInfo				= m_scene->getMaterialInfo().getCudaPtr();
	in.triMaterialColor		= m_scene->getTriMaterialColorBuffer().getCudaPtr();

	module->setTexRef("t_textures", *m_scene->getTextureAtlas()->getAtlasTexture().getImage(), true, true, true, false);

    // Launch.
	if(m_params.rayType == RayType_VPL) {
		in.shadow = true;
		module->getKernel("vplReconstructKernel").launch(m_shadowRays.getSize());
		if(m_currentLight == (m_lightCount * (m_lightBounces + 2)) - 1) {
			module->getKernel("vplNormalizeKernel").setParams(CudaKernel::Param(m_pixels.getCudaPtr()), CudaKernel::Param(m_image->getBuffer().getMutableCudaPtr()), CudaKernel::Param(m_lightCount * (m_lightBounces + 2))).launch(in.numPrimary);
		}

	} else {
		in.shadow = false;
		module->getKernel("vplReconstructKernel").launch(in.numPrimary);
		module->getKernel("vplNormalizeKernel").setParams(CudaKernel::Param(m_pixels.getCudaPtr()), CudaKernel::Param(m_image->getBuffer().getMutableCudaPtr()), 1).launch(in.numPrimary);
	}

}
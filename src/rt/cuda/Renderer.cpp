#include "cuda/Renderer.hpp"
#include "cuda/RendererKernels.hpp"
#include "gui/Window.hpp"
#include "io/File.hpp"

using namespace FW;

Renderer::Renderer(void)
:   m_raygen            (1 << 20),

    m_window            (NULL),
    m_enableRandom      (false),

    m_mesh              (NULL),
    m_scene             (NULL),
    m_bvh               (NULL),

    m_image             (NULL),
    m_cameraFar         (0.0f),

    m_newBatch          (true),
    m_batchRays         (NULL),
    m_batchStart        (0)
{
    m_bvhCachePath = "bvhcache";
}	

//------------------------------------------------------------------------

Renderer::~Renderer(void)
{
    setMesh(NULL);
    delete m_image;
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
        m_scene = new Scene(*mesh);
}

//------------------------------------------------------------------------

void Renderer::setParams(const Params& params)
{
    m_params = params;
}

//------------------------------------------------------------------------

CudaBVH* Renderer::getCudaBVH(void)
{
    // BVH is already valid => done.

    BVHLayout layout = getLayout();
    if (!m_mesh || (m_bvh && m_bvh->getLayout() == layout))
        return m_bvh;

    // Deinit.

    delete m_bvh;
    m_bvh = NULL;

    // Setup build parameters.

    BVH::Stats stats;
    m_buildParams.stats = &stats;

    // Determine cache file name.

    String cacheFileName = sprintf("%s/%08x.dat", m_bvhCachePath.getPtr(), hashBits(
        m_scene->hash(),
        m_platform.computeHash(),
        m_buildParams.computeHash(),
        layout));

    // Cache file exists => import.

    if (!hasError())
    {
        File file(cacheFileName, File::Read);
        if (!hasError())
        {
            m_bvh = new CudaBVH(file);
            return m_bvh;
        }
        clearError();
    }

    // Display status.

    printf("\nBuilding BVH...\nThis will take a while.\n");
    if (m_window)
        m_window->showModalMessage("Building BVH...");

    // Build BVH.

    BVH bvh(m_scene, m_platform, m_buildParams);
    stats.print();
    m_bvh = new CudaBVH(bvh, layout);
    failIfError();

    // Write to cache.

    if (!hasError())
    {
        CreateDirectory(m_bvhCachePath.getPtr(), NULL);
        File file(cacheFileName, File::Create);
        m_bvh->serialize(file);
        clearError();
    }

    // Display status.

    printf("Done.\n\n");
    return m_bvh;
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
    return launchTime;
}

//------------------------------------------------------------------------

void Renderer::beginFrame(GLContext* gl, const CameraControls& camera)
{
    FW_ASSERT(gl && m_mesh);

    // Setup BVH.

    //m_tracer.setBVH(getCudaBVH());
	setTracerBVH(getCudaBVH());

    // Setup result image.

    const Vec2i& size = gl->getViewSize();
    if (!m_image || m_image->getSize() != size)
    {
        delete m_image;
        m_image = new Image(size, ImageFormat::ABGR_8888);
        m_image->getBuffer().setHints(Buffer::Hint_CudaGL);
        m_image->clear();
    }

    // Generate primary rays.

	/** TODO **/
    m_raygen.primaryCPU(m_primaryRays,
        camera.getPosition(),
        invert(gl->xformFitToView(-1.0f, 2.0f) * camera.getWorldToClip()),
        size.x, size.y,
        camera.getFar());

    // Secondary rays enabled => trace primary rays.

    if (m_params.rayType != RayType_Primary)
        //m_tracer.traceBatch(m_primaryRays);
	{
		traceBatch();
	}

    // Initialize state.

    m_cameraFar     = camera.getFar();
    m_newBatch      = true;
    m_batchRays     = NULL;
    m_batchStart    = 0;
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
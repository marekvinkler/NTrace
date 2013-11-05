#include "cuda/CudaRenderer.hpp"
#include "cuda/RendererKernels.hpp"
#include "gui/Window.hpp"
#include "io/File.hpp"

using namespace FW;

//------------------------------------------------------------------------

CudaRenderer::CudaRenderer(void)
:   Renderer()
{
    m_compiler.setSourceFile("src/rt/cuda/RendererKernels.cu");
    m_compiler.addOptions("-use_fast_math");
    m_compiler.include("src/rt");
    m_compiler.include("src/framework");

    m_platform = Platform("GPU");
    m_platform.setLeafPreferences(1, 8);
}

//------------------------------------------------------------------------

CudaRenderer::~CudaRenderer(void)
{
}

//------------------------------------------------------------------------

void CudaRenderer::setParams(const Params& params)
{
    m_params = params;
    m_tracer.setKernel(params.kernelName);
}

//------------------------------------------------------------------------

bool CudaRenderer::nextBatch()
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

    if (m_params.sortSecondary && m_params.rayType != RayType_Primary)
        m_batchRays->mortonSort();
    return true;
}

//------------------------------------------------------------------------

F32 CudaRenderer::traceBatch(RayStats* stats)
{
    FW_ASSERT(m_batchRays);
    return m_tracer.traceBatch(*m_batchRays);
}

//------------------------------------------------------------------------

void CudaRenderer::updateResult(void)
{
    FW_ASSERT(m_scene && m_image && m_batchRays);

    // Compile kernel.

    CudaModule* module = m_compiler.compile();

    // Setup input struct.

    ReconstructInput& in    = *(ReconstructInput*)module->getGlobal("c_ReconstructInput").getMutablePtr();
    in.numRaysPerPrimary    = (m_params.rayType == RayType_Primary) ? 1 : m_params.numSamples;
    in.firstPrimary         = m_batchStart / in.numRaysPerPrimary;
    in.numPrimary           = m_batchRays->getSize() / in.numRaysPerPrimary;
    in.isPrimary            = (m_params.rayType == RayType_Primary);
    in.isAO                 = (m_params.rayType == RayType_AO);
    in.isDiffuse            = (m_params.rayType == RayType_Diffuse);
    in.primarySlotToID      = m_primaryRays.getSlotToIDBuffer().getCudaPtr();
    in.primaryResults       = m_primaryRays.getResultBuffer().getCudaPtr();
    in.batchIDToSlot        = m_batchRays->getIDToSlotBuffer().getCudaPtr();
    in.batchResults         = m_batchRays->getResultBuffer().getCudaPtr();
    in.triMaterialColor     = m_scene->getTriMaterialColorBuffer().getCudaPtr();
    in.triShadedColor       = m_scene->getTriShadedColorBuffer().getCudaPtr();
    in.pixels               = m_image->getBuffer().getMutableCudaPtr();

    // Launch.

    module->getKernel("reconstructKernel").launch(in.numPrimary);
}

//------------------------------------------------------------------------

int CudaRenderer::getTotalNumRays(void)
{
    // Casting primary rays => no degenerates.

    if (m_params.rayType == RayType_Primary)
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

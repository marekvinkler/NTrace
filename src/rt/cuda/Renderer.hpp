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

#pragma once
#include "3d/CameraControls.hpp"
#include "base/Random.hpp"
#include "cuda/CudaBVHTracer.hpp"
#include "ray/RayGen.hpp"
#include "cuda/CudaPersistentBVHTracer.hpp"
#include "cuda/CudaKDTreeTracer.hpp"
#include "tools/visualizationKDTree.hpp"
#include "tools/visualizationBVH.hpp"
#include "Environment.h"


namespace FW
{
//------------------------------------------------------------------------

class Renderer
{
public:
    enum RayType
    {
        RayType_Primary = 0,
        RayType_AO,
        RayType_Diffuse,
		RayType_Textured,
		RayType_PathTracing,
		RayType_VPL,

        RayType_Max
    };

    struct Params
    {
        String          kernelName;
        RayType         rayType;
        F32             aoRadius;
        S32             numSamples;
        bool            sortSecondary;

        Params(void)
        {
            kernelName      = "";
            rayType         = RayType_Primary;
            aoRadius        = 1.0f;
            numSamples      = 32;
            sortSecondary   = false;
        }
    };

public:
						Renderer			();
    virtual             ~Renderer           (void);

    void                setMesh             (MeshBase* mesh);
    void                setBuildParams      (const BVH::BuildParams& params) { invalidateBVH(); m_buildParams = params; }
    void                invalidateBVH       (void)                  { delete m_accelStruct; m_accelStruct = NULL; }

    void                setParams           (const Params& params);
    void                setMessageWindow    (Window* window)        { m_window = window; m_compiler.setMessageWindow(window); m_cudaTracer->setMessageWindow(window); }
    void                setEnableRandom     (bool enable)           { m_enableRandom = enable; }

    CudaVirtualTracer&  getCudaTracer       (void)                  { return *m_cudaTracer; }
    Scene*              getScene            (void) const            { return m_scene; }
	CudaAS*             getCudaBVH          (GLContext* gl = NULL, const CameraControls& camera = CameraControls());

    virtual F32         renderFrame         (GLContext* gl, const CameraControls& camera); // returns total launch time

    virtual void        beginFrame          (GLContext* gl, const CameraControls& camera);
    virtual bool        nextBatch           (void);
    F32                 traceBatch          (void); // returns launch time
    virtual void        updateResult        (void); // for current batch
    void                displayResult       (GLContext* gl);

    int                 getTotalNumRays     (void); // for selected ray type, excluding degenerates

	F32					calcNodeSAHCostKdtree(const Platform& platform, Buffer* nodes, Buffer* tri,  S32 n, AABB bbox, S32 depth, S32& maxDepth, S32& sumDepth, S32& numNodes, S32& numLeaves, F32& nodeArea, F32 &weightedLeafArea, F32& test);
	F32					calcLeafSAHCostCompact(const Platform& platform, Buffer* triIdx, S32 n, S32& numLeaves);
	F32					calcLeafSAHCostNum	(const Platform& platform, S32 n, S32& numLeaves);

	CudaAS*				getCudaKDTree		(void);

	void                startBVHVis         (void);
	void                endBVHVis           (void);
	void                toggleBVHVis        (void)					{ m_showVis = !m_showVis; m_showVis ? startBVHVis() : endBVHVis(); }

	void				resetSampling		(void)					{ m_sampleCount = 0.0f; }
	void                getVisibleTriangles (RayBuffer* rayBuffer);

protected:
                        Renderer            (const Renderer&); // forbidden
    Renderer&           operator=           (const Renderer&); // forbidden

protected:
    CudaCompiler        m_compiler;
    String              m_cachePath;
    Platform            m_platform;
    BVH::BuildParams    m_buildParams;
    RayGen              m_raygen;
    Random              m_random;

    Params              m_params;
    Window*             m_window;
    bool                m_enableRandom;

    MeshBase*           m_mesh;
    Scene*              m_scene;

    Image*              m_image;
	Image*				m_sampleImage;
	float				m_sampleCount;
    F32                 m_cameraFar;
    RayBuffer           m_primaryRays;
	RayBuffer           m_shadowRays;
    RayBuffer           m_secondaryRays;
	Buffer              m_triangleVisibility;	//!< Visibility buffer

    bool                m_newBatch;
    RayBuffer*          m_batchRays;
    S32                 m_batchStart;

	CudaAS*				m_accelStruct;
	CudaVirtualTracer*	m_cudaTracer;

	Visualization*		m_vis;
	bool				m_showVis;
};

//------------------------------------------------------------------------
}

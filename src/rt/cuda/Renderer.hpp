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
#include "cuda/CudaTracer.hpp"
#include "ray/RayGen.hpp"

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
                        Renderer            (void);
    virtual             ~Renderer           (void);

    void                setMesh             (MeshBase* mesh);
    virtual void        setBuildParams      (const BVH::BuildParams& params) { invalidateBVH(); m_buildParams = params; }
    void                invalidateBVH       (void)                  { delete m_bvh; m_bvh = NULL; }

    virtual void        setParams           (const Params& params);
    virtual void        setMessageWindow    (Window* window)        { m_window = window; }
    void                setEnableRandom     (bool enable)           { m_enableRandom = enable; }
	virtual BVHLayout	getLayout           (void) = 0;		

    Scene*              getScene            (void) const            { return m_scene; }
    CudaBVH*            getCudaBVH          (void);

    F32                 renderFrame         (GLContext* gl, const CameraControls& camera); // returns total launch time

    void                beginFrame          (GLContext* gl, const CameraControls& camera);
    virtual bool        nextBatch           (void) = 0;
    virtual F32         traceBatch          (RayStats* stats = NULL) = 0; // returns launch time
    virtual void        updateResult        (void) = 0; // for current batch
    void                displayResult       (GLContext* gl);

    virtual int         getTotalNumRays     (void) = 0; // for selected ray type, excluding degenerates
	
protected:
	virtual void        setTracerBVH        (CudaBVH* bvh)          { m_bvh = bvh; }

private:
                        Renderer            (const Renderer&); // forbidden
    Renderer&           operator=           (const Renderer&); // forbidden

protected:
    String              m_bvhCachePath;
    Platform            m_platform;
    BVH::BuildParams    m_buildParams;
    RayGen              m_raygen;
    Random              m_random;

    Params              m_params;
    Window*             m_window;
    bool                m_enableRandom;

    MeshBase*           m_mesh;
    Scene*              m_scene;
    CudaBVH*            m_bvh;

    Image*              m_image;
    F32                 m_cameraFar;
    RayBuffer           m_primaryRays;
    RayBuffer           m_secondaryRays;

    bool                m_newBatch;
    RayBuffer*          m_batchRays;
    S32                 m_batchStart;
};

//------------------------------------------------------------------------
}

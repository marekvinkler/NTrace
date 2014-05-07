/*
 *  Copyright 2009-2010 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

#pragma once
#include "Renderer.hpp"

namespace FW
{
//------------------------------------------------------------------------

class CPURenderer : public Renderer
{
public:
	                    CPURenderer         (Environment* env);
	virtual             ~CPURenderer        (void);

	
    virtual void        setParams           (const Params& params);
    virtual void        setMessageWindow    (Window* window)        { m_window = window; }
	virtual BVHLayout	getLayout           (void)                  { return m_layout; }
	
    virtual bool        nextBatch           (void);
    virtual F32         traceBatch          (RayStats* stats = NULL); // returns launch time
    virtual void        updateResult        (void); // for current batch

    virtual int         getTotalNumRays     (void); // for selected ray type, excluding degenerates

	Vec4f				getPseudoColor(F32 value, F32 minVal, F32 maxVal);
	Vec4f				getDistanceColor(F32 value, F32 minVal, F32 maxVal);
	
private:
                        CPURenderer         (const CPURenderer&); // forbidden
    CPURenderer&        operator=           (const CPURenderer&); // forbidden
	
private:
    BVHLayout           m_layout;
	Buffer              m_visibility;

/*public:
                        CPURenderer         (void);
    virtual             ~CPURenderer        (void);

	virtual BVHLayout   getLayout           (void)                  { return m_layout; }
	void                setLayout           (BVHLayout layout)      { m_layout = layout; }

    virtual bool        nextBatch           (void);
    virtual F32         traceBatch          (RayStats* stats = NULL); // returns launch time
    virtual void        updateResult        (void); // for current batch

    virtual S32         incrementNumRays    (void); // for selected ray type, excluding degenerates

protected:
	virtual Buffer&     getVisibleTriangles (S32 triangleCount, bool setValue, S32 initValue = 0); // gets the bit array of triangle visibility. If sizes do not match it is initialized

private:
                        CPURenderer         (const CPURenderer&); // forbidden
    CPURenderer&        operator=			(const CPURenderer&); // forbidden

	Vec4f               getPseudoColor      (F32 value, F32 minVal, F32 maxVal);
	Vec4f               getDistanceColor    (F32 value, F32 minVal, F32 maxVal);

private:
    BVHLayout           m_layout;
	Buffer              m_visibility;*/
};

//------------------------------------------------------------------------
}

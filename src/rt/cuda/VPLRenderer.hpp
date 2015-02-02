#pragma once

#include "Renderer.hpp"

namespace FW
{

class VPLRenderer : public Renderer 
{

public:

						VPLRenderer();
	virtual				~VPLRenderer();

	virtual F32         renderFrame         (GLContext* gl, const CameraControls& camera); // returns total launch time

    virtual void        beginFrame          (GLContext* gl, const CameraControls& camera);
    virtual bool        nextBatch           (void);
    virtual void        updateResult        (void); // for current batch

protected:
	bool                m_firstFrame;


private:
	Buffer				m_lights;
	RayBuffer			m_vplBuffer;
	int					m_lightCount;
	int					m_lightBounces;
	int					m_currentLight;
	Vec3f				m_material;

};

}
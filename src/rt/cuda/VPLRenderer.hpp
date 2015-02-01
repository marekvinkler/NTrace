#pragma once

#include "Renderer.hpp"

namespace FW
{

class VPLRenderer : public Renderer 
{

public:

						VPLRenderer();

	virtual F32         renderFrame         (GLContext* gl, const CameraControls& camera); // returns total launch time

    virtual void        beginFrame          (GLContext* gl, const CameraControls& camera);
    virtual bool        nextBatch           (void);
    virtual void        updateResult        (void); // for current batch

protected:
	bool                firstFrame;

};

}
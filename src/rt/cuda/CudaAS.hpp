
/**
* \file
* \brief Definitions for acceleration structure interface.
*/

#pragma once
#include "gpu/Buffer.hpp"
#include "ray/RayBuffer.hpp"
#include "io/Stream.hpp"
#include "kernels/CudaTracerKernels.hpp"

namespace FW
{

/**
* \brief Interface for acceleration structure.
* \details Acceleration structures used in rendering muse immplement this interface.
*/
class CudaAS
{
public:
	/**
	* \brief Destructor.
	*/
	virtual								~CudaAS				(void) {}

	/**
	* \brief Returns node buffer.
	* \return Node buffer.
	*/
	virtual Buffer&						getNodeBuffer       (void) = 0;

	/**
	* \brief Returns buffer of woopified triangles.
	* \return Buffer of woopified triangles.
	*/
	virtual Buffer&						getTriWoopBuffer    (void) = 0;

	/**
	* \brief Returns buffer of triangle indexes.
	* \return Buffer of triangle indexes.
	*/
	virtual Buffer&						getTriIndexBuffer   (void) = 0;

	/**
	* \brief Returns layout of buffers.
	* \return Layout of buffers.
	*/
	virtual BVHLayout					getLayout           (void) const = 0;

	//virtual Vec2i						getNodeSubArray		(int idx) const = 0;
    //virtual Vec2i						getTriWoopSubArray	(int idx) const = 0;

    //virtual CudaAS&						operator=			(CudaAS& other) = 0;

	/**
	* \brief Writes buffers to output stream.
	* \param[in] out	Output stream to write to.
	*/
    virtual void						serialize           (OutputStream& out) = 0;

	virtual void						trace				(RayBuffer& rays, Buffer& visibility) = 0;
};

}



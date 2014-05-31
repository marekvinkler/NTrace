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
#include "gpu/Buffer.hpp"
#include "Util.hpp"

namespace FW
{

/**
 * \brief Ray buffer class. Stores rays.
 */
class RayBuffer
{
public:
	/**
	 * \brief Constructor.
	 * \param[in] n Size of the buffer (number of rays).
	 * \param[in] closestHit Flag whether closest hit is required.
	 */
                        RayBuffer               (S32 n=0, bool closestHit=true)     : m_size(0), m_needClosestHit(closestHit) { resize(n); }

	/**
	 * \brief Gets size of the buffer (number of rays).
	 * \return Size of the buffer.
	 */
    S32                 getSize                 () const                            { return m_size; }

	/**
	 * \brief Resizes the buffer.
	 * \param[in] n New size (number of rays).
	 */
    void                resize                  (S32 n);

	/**
	 * \brief Assigns ray to a slot. Id is same as slot.
	 * \param[in] slot Slot for the ray.
	 * \param[in] ray Ray to be assigned.
	 */
    void                setRay                  (S32 slot, const Ray& ray)          { setRay(slot, ray, slot); }

	/**
	 * \brief Assigns ray to a slot and to an id.
	 * \param[in] slot Slot for the ray.
	 * \param[in] ray Ray to be assigned.
	 * \param[in] id Id of the ray.
	 */
    void                setRay                  (S32 slot, const Ray& ray, S32 id);

	/**
	 * \brief Assigns ray result to a slot.
	 * \param[in] slot Slot for the ray.
	 * \param[in] r Ray result to be assigned.
	 */
    void                setResult               (S32 slot, const RayResult& r)      { getMutableResultForSlot(slot) = r; }

//  const Ray&          operator[]              (S32 slot) const                    { return getRayForSlot(slot); }

	/**
	 * \brief Gets a ray assigned to a given slot.
	 * \param[in] slot Slot where the ray is saved.
	 * \return Ray from the given slot.
	 */
    const Ray&          getRayForSlot           (S32 slot) const                    { FW_ASSERT(slot >= 0 && slot < m_size); return ((const Ray*)m_rays.getPtr())[slot]; }

	/**
	 * \brief Gets a ray with a given id.
	 * \param[in] id Id of the ray.
	 * \return Ray with the given id.
	 */
    const Ray&          getRayForID             (S32 id) const                      { return getRayForSlot(getSlotForID(id)); }

	/**
	 * \brief Gets a ray result assigned to a given slot.
	 * \param[in] slot Slot where the ray result is saved.
	 * \return Ray result from the given slot.
	 */
    const RayResult&    getResultForSlot        (S32 slot) const                    { FW_ASSERT(slot >= 0 && slot < m_size); return ((const RayResult*)m_results.getPtr())[slot]; }

	/**
	 * \brief Gets a mutable ray assigned to a given slot.
	 * \param[in] slot Slot where the mutable ray is saved.
	 * \return Mutable ray from the given slot.
	 */
          RayResult&    getMutableResultForSlot (S32 slot)                          { FW_ASSERT(slot >= 0 && slot < m_size); return ((RayResult*)m_results.getMutablePtr())[slot]; }

	/**
	 * \brief Gets a ray result with a given id.
	 * \param[in] id Id of the ray.
	 * \return Ray result with the given id.
	 */
    const RayResult&    getResultForID          (S32 id) const                      { return getResultForSlot(getSlotForID(id)); }

	/**
	 * \brief Gets a mutable ray result with a given id.
	 * \param[in] id Id of the mutable ray result.
	 * \return Mutable ray result with the given id.
	 */
          RayResult&    getMutableResultForID   (S32 id)                            { return getMutableResultForSlot(getSlotForID(id)); }

	/**
	 * \brief Gets a ray slot for a given id.
	 * \param id Id of the slot.
	 * \return Slot with the given id.
	 */
    S32                 getSlotForID            (S32 id) const                      { FW_ASSERT(id >= 0 && id < m_size); return ((const S32*)m_IDToSlot.getPtr())[id]; }

	/** 
	 * \brief Gets an id for a given ray slot.
	 * \param[in] slot Slot with the desired id.
	 * return Id of the given slot.
	 */
    S32                 getIDForSlot            (S32 slot) const                    { FW_ASSERT(slot >= 0 && slot < m_size); return ((const S32*)m_slotToID.getPtr())[slot]; }

	/**
	 * \brief Sets whether the closet hit is needed.
	 * \param[in] c Closest hit flag.
	 */
    void                setNeedClosestHit       (bool c)                            { m_needClosestHit = c; }

	/**
	 * \brief Returns whether the closest hit is needed.
	 * \return Closest hit flag.
	 */
    bool                getNeedClosestHit       () const                            { return m_needClosestHit; }

	/**
	 * \brief Performs morton sort.
	 */
    void                mortonSort              ();

	/**
	 * \brief Shuffles rays in the buffer.
	 * \param randomSeed Random seed.
	 */
    void                randomSort              (U32 randomSeed=0);

	/**
	 * \brief Gets ray buffer.
	 * \return Ray buffer.
	 */
    Buffer&             getRayBuffer            ()                                  { return m_rays; }

	/**
	 * \brief Gets ray result buffer.
	 * \return Ray result buffer.
	 */
    Buffer&             getResultBuffer         ()                                  { return m_results; }

	/**
	* \brief Gets buffer mapping ids to slots.
	* \return Buffer mapping ids to slots.
	*/
    Buffer&             getIDToSlotBuffer       ()                                  { return m_IDToSlot; }

	/**
	* \brief Gets buffer slots to ids.
	* \return Buffer mapping slots to ids.
	*/
    Buffer&             getSlotToIDBuffer       ()                                  { return m_slotToID; }

private:
    S32                 m_size;					//!< Size of the buffer (number of rays).
    mutable Buffer      m_rays;					//!< Ray buffer. (Ray)
    mutable Buffer      m_results;				//!< Ray result buffer. (RayResult)
    mutable Buffer      m_IDToSlot;				//!< Buffer mapping ids to slots. (S32)
    mutable Buffer      m_slotToID;				//!< Buffer mapping slots to ids. (S32)

    bool                m_needClosestHit;		//!< Flag whether closest hit is needed.
};

} //
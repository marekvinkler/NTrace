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

/**
 * \file
 * \brief Declarations for platform settings.
 */

#pragma once
#include "base/String.hpp"
#include "base/Hash.hpp"

namespace FW
{

class LeafNode;
class BVHNode;

/**
 * \brief Class holding various SAH and batch processing parameters.
 */
class Platform
{
public:

	/**
	 * \brief Constructor.
	 */
    Platform()                                                                                                          { m_name=String("Default"); m_SAHNodeCost = 1.f; m_SAHTriangleCost = 1.f; m_nodeBatchSize = 1; m_triBatchSize = 1; m_minLeafSize=1; m_maxLeafSize=0x7FFFFFF; }

	/**
	 * \brief Constructor.
	 * \param[in] name Name of the platform settings.
	 * \param[in] nodeCost Cost of a single node.
	 * \param[in] triCost Cost of a single triangle.
	 * \param[in] nodeBatchSize Size of a node batch.
	 * \param[in] triBatchSize Size of a triangle batch.
	 */
	Platform(const String& name, float nodeCost=1.f, float triCost=1.f, S32 nodeBatchSize=1, S32 triBatchSize=1) { m_name=name; m_SAHNodeCost = nodeCost; m_SAHTriangleCost = triCost; m_nodeBatchSize = nodeBatchSize; m_triBatchSize = triBatchSize; m_minLeafSize=1; m_maxLeafSize=0x7FFFFFF; }

	/**
	 * \return Name of the platform settings.
	 */
    const String&   getName() const                     { return m_name; }

    // SAH weights
	/**
	 * \return SAH cost of a single triangle.
	 */
    float getSAHTriangleCost() const                    { return m_SAHTriangleCost; }

	/**
	 * \return SAH cost of a single node.
	 */
    float getSAHNodeCost() const                        { return m_SAHNodeCost; }

    // SAH costs, raw and batched
	/**
	 * \brief Calculates cost of a single node.
	 * \param[in] numChildNodes Number of child nodes.
	 * \param[in] numTrix Number of triangles.
	 * \return SAH cost of a node.
	 */
    float getCost(int numChildNodes, int numTris) const	{ return getNodeCost(numChildNodes) + getTriangleCost(numTris); }

	/**
	 * \brief Calcuates cost of a given number of triangles rounded to the batch size.
	 * \param[in] n Number of triangles.
	 * \return Cost of triangles.
	 */
    float getTriangleCost(S32 n) const                  { return roundToTriangleBatchSize(n) * m_SAHTriangleCost; }

	/**
	 * \brief Calculates cost of a given number of nodes rounded to the batch size.
	 * \param[in] n Number of nodes.
	 * \return Cost of nodes.
	 */
    float getNodeCost(S32 n) const                      { return roundToNodeBatchSize(n) * m_SAHNodeCost; }

    // batch processing (how many ops at the price of one)
	/**
	 * \return Size of the triangle batch.
	 */
    S32   getTriangleBatchSize() const                  { return m_triBatchSize; }

	/**
	 * \return Size of the node batch.
	 */
    S32   getNodeBatchSize() const                      { return m_nodeBatchSize; }

	/**
	 * \brief Sets triangle batch size to a given value.
	 * \param[in] triBatchSize New triangle batch size.
	 */
    void  setTriangleBatchSize(S32 triBatchSize)        { m_triBatchSize = triBatchSize; }

	/**
	 * \brief Sets node batch size to a given value.
	 * \param[in] nodeBatchSize New node batch size.
	 */
    void  setNodeBatchSize(S32 nodeBatchSize)           { m_nodeBatchSize = nodeBatchSize; }

	/**
	 * \brief Rounds given value up to the nearest triangle batch size multiple.
	 * \param[in] n Value to be rounded.
	 * \return Rounded value.
	 */
    S32   roundToTriangleBatchSize(S32 n) const         { return ((n+m_triBatchSize-1)/m_triBatchSize)*m_triBatchSize; }

	/**
	 * \brief Rounds given value up to the nearest node batch size multiple.
	 * \param[in] n Value to be rounded.
	 * \return Rounded value.
	 */
    S32   roundToNodeBatchSize(S32 n) const             { return ((n+m_nodeBatchSize-1)/m_nodeBatchSize)*m_nodeBatchSize; }

    // leaf preferences
	/**
	 * \brief Sets leaf size preferences (desired number of triangles in one leaf node).
	 * \param[in] minSize Minimum leaf size.
	 * \param[in] maxSize Maximum leaf size.
	 */
    void  setLeafPreferences(S32 minSize, S32 maxSize)	{ m_minLeafSize=minSize; m_maxLeafSize=maxSize; }

	/**
	* \return Minimum leaf size.
	*/
    S32   getMinLeafSize() const                        { return m_minLeafSize; }

	/**
	 * \return Maximum leaf size.
	 */
    S32   getMaxLeafSize() const                        { return m_maxLeafSize; }

	/**
	 * \return Hash of the platform settings.
	 */
    U32   computeHash() const                           { return hashBits(hash<String>(m_name), floatToBits(m_SAHNodeCost), floatToBits(m_SAHTriangleCost), hashBits(m_triBatchSize, m_nodeBatchSize, m_minLeafSize, m_maxLeafSize)); }

private:
    String  m_name;										//!< Name.
    float   m_SAHNodeCost;								//!< Node SAH cost.
    float   m_SAHTriangleCost;							//!< Triangle SAH cost.
    S32     m_triBatchSize;								//!< Triangle batch size.
    S32     m_nodeBatchSize;							//!< Node batch size.
    S32     m_minLeafSize;								//!< Minimum leaf size.
    S32     m_maxLeafSize;								//!< Maximum leaf size.
};

} //
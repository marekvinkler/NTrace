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

/** \file
 *  \brief Declarations for the BVH acceleration structure.
 */

#pragma once
#include "Scene.hpp"
#include "bvh/BVHNode.hpp"
#include "ray/RayBuffer.hpp"
#include "acceleration\Acceleration.hpp"
#include "Environment.h"

namespace FW
{

/**
 * \brief Structure holding ray statistics. Also provides print to the console. These statistics are used in a CPU trace method provided by this class.
 */
struct RayStats
{
	/**
	 * \brief Constructor.
	 */
    RayStats()          { clear(); }

	/** 
	 * \brief Resets the statistics to the default values.
	 */
    void clear()        { memset(this,0,sizeof(RayStats)); }

	/**
	 * \brief Prints ray statistics to the console.
	 */
    void print() const  { if(numRays>0) printf("Ray stats: (%s) %d rays, %.1f tris/ray, %.1f nodes/ray (cost=%.2f) %.2f treelets/ray\n", platform.getName().getPtr(), numRays, 1.f*numTriangleTests/numRays, 1.f*numNodeTests/numRays, (platform.getSAHTriangleCost()*numTriangleTests/numRays + platform.getSAHNodeCost()*numNodeTests/numRays), 1.f*numTreelets/numRays ); }

    S32         numRays;				//!< Total number of rays.
    S32         numTriangleTests;		//!< Total number of ray-triangle tests.
    S32         numNodeTests;			//!< Total number of ray-node tests.
    S32         numTreelets;			//!< Total number of traversal steps.
    Platform    platform;				//!< Platform settings of the BVH. Set by whoever sets the stats.

};

/**
 *  \brief BVH acceleration structure class.
 *  \details Holds a BVH acceleration structure and also provides method for its CPU traversal.
 */
class BVH : public AccelerationStructure
{
public:

	/**
	 * \brief Sturcture for holding statistics about the BVH.
	 */
    struct Stats
    {
		/**
		* \brief Constructor.
		*/
        Stats()             { clear(); }

		/** 
		* \brief Resets the statistics to the default values.
		*/
        void clear()        { memset(this, 0, sizeof(Stats)); }

		/**
		* \brief Prints the statistics to the console.
		*/
        void print() const  { printf("Tree stats: [bfactor=%d] %d nodes (%d+%d), %.2f SAHCost, %.1f children/inner, %.1f tris/leaf\n", branchingFactor,numLeafNodes+numInnerNodes, numLeafNodes,numInnerNodes, SAHCost, 1.f*numChildNodes/max(numInnerNodes,1), 1.f*numTris/max(numLeafNodes,1)); }

        F32     SAHCost;				//!< Total sah cost of the BVH.
        S32     branchingFactor;		//!< Number of children nodes per one parent node.
        S32     numInnerNodes;			//!< Total number of inner nodes.
        S32     numLeafNodes;			//!< Total number of leaf nodes.
        S32     numChildNodes;			//!< Total number of children nodes.
        S32     numTris;				//!< Total number of triangles.
    };

	/**
	 * \brief Stucture holding the BVH build parameters.
	 */
    struct BuildParams
    {
        Stats*					stats;			//!< Statistics. If NULL, no statistics are gathered.
        bool					enablePrints;	//!< Flag whether to enable prints about build progress.
        F32						splitAlpha;     //!< Spatial split area threshold.
		F32                     osahWeight;     //!< Weighting factor for OSAH construction.
		String	                accelerator;    //!< The name of the acceleration data structure method for ray tracing.
		Array<AABB>				empty_boxes;	//!< Information about boxes with no triangles inside.
		Buffer*                 visibility;		//!< Visibility buffer for the CPU renderer.
		String					logDirectory;	//!< Directory where the log file will be saved.
		String					buildName;		//!< Build name.
		int						cameraIdx;		//!< Camera index.
		bool                    twoTrees;       //!< Flag whether to build BVH from two separate trees.

		/**
		 * \brief Constructor.
		 */
        BuildParams(void)
        {
            stats           = NULL;
            enablePrints    = true;
            splitAlpha      = 1.0e-5f;
			osahWeight      = 0.9f;
			//camera			= NULL;
			cameraIdx       = 0;
			twoTrees        = false;
			visibility		= NULL;
        }

		/**
		 * \brief Computes hash of the build parameters.
		 * \return Hashed build parameters.
		 */
        U32 computeHash(void) const
        {
            return hashBits(floatToBits(splitAlpha));
        }
    };

public:

	/**
	 *  \brief Constructor.
	 *  \param[in] scene		Source scene for the BVH.
	 *  \param[in] platform		Platform settings.
	 *  \param[in] params		Build parameters.
	 */
                        BVH                     (Scene* scene, const Platform& platform, const BuildParams& params, Environment* env);

	/**
	 *  \brief Destructor.
	 */
                        ~BVH                    (void)                  { if(m_root) m_root->deleteSubtree(); }

	/**
	 *  \brief Returns root node of the BVH.
	 *  \return Root node of the BVH.
	 */
    BVHNode*            getRoot                 (void) const            { return m_root; }

	/**
	 *  \brief CPU traversal.
	 *  \param[out] rays		Buffer of rays that will be traced.
	 *  \param[out] stats		Ray statistics collected during the traversal. Leave blank if no stats should be collected.
	 */
    void                trace                   (RayBuffer& rays, RayStats* stats = NULL) const;

	/**
	 *  \brief Returns an array of triangle indices to which leaf nodes are pointig. These indices point to scene's triangle array.
	 *  \return Buffer of triangle indices.
	 */
    Array<S32>&         getTriIndices           (void)                  { return m_triIndices; }

	/**
	 *  \brief Returns an array of triangle indices to which leaf nodes are pointig. These indices point to scene's triangle array.
	 *  \return Buffer of triangle indices.
	 */
    const Array<S32>&   getTriIndices           (void) const            { return m_triIndices; }

private:

	/**
	 *  \brief Recursively traverses the BVH.
	 *  \param[in] node				Root node of the traversal.
	 *  \param[in] rays				Ray that will traverse the BVH.
	 *  \param[out] result			Result of the traversal.
	 *	\param[in] needClosestHit	Wheter the ray needs the closest hit or first hit is sufficient.
	 *	\param[out] stats			Ray statistics collected during the traversal.
	 */
    void                traceRecursive          (BVHNode* node, Ray& ray, RayResult& result, bool needClosestHit, RayStats* stats) const;
	
    BVHNode*            m_root;				//!< Root node.
    Array<S32>          m_triIndices;		//!< Array of indices pointing to the scene triangle array.
	Environment*		m_env;				//!< Environment settings.
};

}

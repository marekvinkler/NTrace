/* 
 *  Copyright (c) 2013, Faculty of Informatics, Masaryk University
 *  All rights reserved.
 *  
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions are met:
 *      * Redistributions of source code must retain the above copyright
 *        notice, this list of conditions and the following disclaimer.
 *      * Redistributions in binary form must reproduce the above copyright
 *        notice, this list of conditions and the following disclaimer in the
 *        documentation and/or other materials provided with the distribution.
 *      * Neither the name of the <organization> nor the
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
 *
 *  Authors:
 *  Tomas Kopal, 1996
 *  Vilem Otte <vilem.otte@post.cz>
 *
 */

/*! \file
 *  \brief Environment variables class for this app.
 */

#include "AppEnvironment.h"

void AppEnvironment::RegisterOptions()
{
	/*************************************************************************/
	/*    Renderer                                                           */
	/*************************************************************************/

	RegisterOption("Renderer.dataStructure", optString, "renderer_ds=");
	RegisterOption("Renderer.builder", optString, "renderer.builder=");

	/*************************************************************************/
	/*    Raygen                                                             */
	/*************************************************************************/
	RegisterOption("Raygen.random", optBool, "raygen_random=", "false");

	/*************************************************************************/
	/*    Persistent BVH                                                     */
	/*************************************************************************/
	RegisterOption("SubdivisionRayCaster.numWarpsPerBlock", optInt, "persistent_numWarpsPerBlock=", "24");
	RegisterOption("SubdivisionRayCaster.numBlockPerSM", optInt, "persistent_numBlockPerSM=", "2");

	RegisterOption("SubdivisionRayCaster.triangleBasedWeight", optInt, "persistent_triangleBasedWeight=", "0");
	RegisterOption("SubdivisionRayCaster.rayBasedWeight", optInt, "persistent_rayBasedWeight=", "0");
	RegisterOption("SubdivisionRayCaster.axisAlignedWeight", optInt, "persistent_axisAlignedWeight=", "1");
	RegisterOption("SubdivisionRayCaster.planeSelectionOverhead", optFloat, "persistent_planeSelectionOverhead=", "0.5");
	
	RegisterOption("SubdivisionRayCaster.rayLimit", optInt, "persistent_rayLimit=", "32");
	RegisterOption("SubdivisionRayCaster.triLimit", optInt, "persistent_triLimit=", "2");
	RegisterOption("SubdivisionRayCaster.triMaxLimit", optInt, "persistent_triMaxLimit=", "16");
	RegisterOption("SubdivisionRayCaster.maxDepth", optInt, "persistent_maxDepth=", "50");
	RegisterOption("SubdivisionRayCaster.depthK1", optFloat, "persistent_depthK1=", "1.2");
	RegisterOption("SubdivisionRayCaster.depthK2", optFloat, "persistent_depthK2=", "2.0");
	RegisterOption("SubdivisionRayCaster.failRq", optFloat, "persistent_failRq=", "0.9");
	RegisterOption("SubdivisionRayCaster.failK1", optFloat, "persistent_failK1=", "0.26");
	RegisterOption("SubdivisionRayCaster.failK2", optFloat, "persistent_failK2=", "1.0");
	
	RegisterOption("SubdivisionRayCaster.failureCount", optInt, "persistent_failureCount=", "0");
	
	RegisterOption("SubdivisionRayCaster.ci", optInt, "persistent_ci=", "1");
	RegisterOption("SubdivisionRayCaster.ct", optInt, "persistent_ct=", "1");
	RegisterOption("SubdivisionRayCaster.ctr", optInt, "persistent_ctr=", "1");
	RegisterOption("SubdivisionRayCaster.ctt", optInt, "persistent_ctt=", "1");
	
	RegisterOption("SubdivisionRayCaster.siblingLimit", optInt, "persistent_siblingLimit=", "0");
	RegisterOption("SubdivisionRayCaster.childLimit", optInt, "persistent_childLimit=", "0");
	RegisterOption("SubdivisionRayCaster.subtreeLimit", optInt, "persistent_subtreeLimit=", "0");

	RegisterOption("SubdivisionRayCaster.popCount", optInt, "persistent_popCount=", "14");
	RegisterOption("SubdivisionRayCaster.granularity", optFloat, "persistent_granularity=", "50.0");
	
	RegisterOption("SubdivisionRayCaster.nodeRatio", optInt, "persistent_nodeRatio=", "5");
	RegisterOption("SubdivisionRayCaster.triRatio", optInt, "persistent_triRatio=", "3");
	RegisterOption("SubdivisionRayCaster.idxRatio", optInt, "persistent_idxRatio=", "12");
	
	RegisterOption("SubdivisionRayCaster.log", optString, "persistent_log=", "ntrace.log");
	RegisterOption("SubdivisionRayCaster.sumTimes", optBool, "persistent_sumTimes=", "true");
	RegisterOption("SubdivisionRayCaster.cutOffDepth", optInt, "persistent_cutOffDepth=", "30");
	RegisterOption("SubdivisionRayCaster.numRepeats", optInt, "persistent_numRepeats=", "1");
}


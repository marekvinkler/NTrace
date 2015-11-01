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
	/*    App							                                     */
	/*************************************************************************/
	RegisterOption("App.benchmark", optBool, "app_benchmark=", "true");
	RegisterOption("App.log", optString, "app_log=", "ntrace.log");
	RegisterOption("App.stats", optString, "app_stats=", "stats.log");
	RegisterOption("App.frameWidth", optInt, "app_frame_width=", "1024");
	RegisterOption("App.frameHeight", optInt, "app_frame_height=", "768");

	/*************************************************************************/
	/*    Benchmark						                                     */
	/*************************************************************************/
	RegisterOption("Benchmark.scene", optString, "benchmark_scene=");
	RegisterOption("Benchmark.camera", optString, "benchmark_camera=");
	RegisterOption("Benchmark.kernel", optString, "benchmark_kernel=");
	RegisterOption("Benchmark.warmupRepeats", optInt, "benchmark_warmup=", "1");
	RegisterOption("Benchmark.measureRepeats", optInt, "benchmark_measure=", "5");
	RegisterOption("Benchmark.screenshot", optBool, "benchmark_scr=", "false");
	RegisterOption("Benchmark.screenshotName", optString, "benchmark_scrname=", "screenshot_kernel=%d_rt=%s_cam=%d.png");
	
	/*************************************************************************/
	/*    Renderer                                                           */
	/*************************************************************************/
	RegisterOption("Renderer.dataStructure", optString, "renderer_ds=");
	RegisterOption("Renderer.builder", optString, "renderer_builder=");
	RegisterOption("Renderer.rayType", optString, "renderer_raytype=");
	RegisterOption("Renderer.samples", optInt, "renderer_samples=", "8");
	RegisterOption("Renderer.sortRays", optBool, "renderer_sortrays=", "true");
	RegisterOption("Renderer.cacheDataStructure", optBool, "renderer_cache_ds=", "true");

	/*************************************************************************/
	/*    VPL                                                                */
	/*************************************************************************/
	RegisterOption("VPL.primaryLights", optInt, "primary_lights=", "20");
	RegisterOption("VPL.maxLightBounces", optInt, "max_light_bounces=", "10");


	/*************************************************************************/
	/*    Raygen                                                             */
	/*************************************************************************/
	RegisterOption("Raygen.random", optBool, "raygen_random=", "false");
	RegisterOption("Raygen.aoRadius", optFloat, "raygen_aoradius=", "5.0");

	/*************************************************************************/
	/*    SBVH	                                                             */
	/*************************************************************************/
	RegisterOption("SBVH.alpha", optFloat, "sbvh_alpha=", "1.0e-5");

	/*************************************************************************/
	/*    Persistent Data Structure                                          */
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

	/*************************************************************************/
	/*    Persistent BVH                                                     */
	/*************************************************************************/
	RegisterOption("PersistentBVH.triLimit", optInt, "persistent_bvh_triLimit=", "2");
	RegisterOption("PersistentBVH.triMaxLimit", optInt, "persistent_bvh_triMaxLimit=", "16");
	RegisterOption("PersistentBVH.maxDepth", optInt, "persistent_bvh_maxDepth=", "50");
	
	RegisterOption("PersistentBVH.ci", optInt, "persistent_bvh_ci=", "1");
	RegisterOption("PersistentBVH.ct", optInt, "persistent_bvh_ct=", "1");

	RegisterOption("PersistentBVH.childLimit", optInt, "persistent_bvh_childLimit=", "0");
	
	RegisterOption("PersistentBVH.popCount", optInt, "persistent_bvh_popCount=", "14");
	RegisterOption("PersistentBVH.granularity", optFloat, "persistent_bvh_granularity=", "50.0");

	/*************************************************************************/
	/*    Persistent BVH                                                     */
	/*************************************************************************/
	RegisterOption("PersistentSBVH.heapMultiplicator", optFloat, "persistent_sbvh_heap_mult=", "20.0");
	RegisterOption("PersistentSBVH.nodeRatio", optInt, "persistent_sbvh_nodeRatio=", "5");
	RegisterOption("PersistentSBVH.triRatio", optInt, "persistent_sbvh_triRatio=", "3");
	RegisterOption("PersistentSBVH.idxRatio", optInt, "persistent_sbvh_idxRatio=", "12");

	/*************************************************************************/
	/*    Persistent KDTree                                                  */
	/*************************************************************************/
	RegisterOption("PersistentKDTree.triLimit", optInt, "persistent_kdtree_triLimit=", "2");
	RegisterOption("PersistentKDTree.triMaxLimit", optInt, "persistent_kdtree_triMaxLimit=", "16");
	RegisterOption("PersistentKDTree.maxDepth", optInt, "persistent_kdtree_maxDepth=", "50");
	RegisterOption("PersistentKDTree.depthK1", optFloat, "persistent_kdtree_depthK1=", "1.2");
	RegisterOption("PersistentKDTree.depthK2", optFloat, "persistent_kdtree_depthK2=", "2.0");
	RegisterOption("PersistentKDTree.failRq", optFloat, "persistent_kdtree_failRq=", "0.9");
	RegisterOption("PersistentKDTree.failK1", optFloat, "persistent_kdtree_failK1=", "0.26");
	RegisterOption("PersistentKDTree.failK2", optFloat, "persistent_kdtree_failK2=", "1.0");
	
	RegisterOption("PersistentKDTree.failureCount", optInt, "persistent_kdtree_failureCount=", "0");
	
	RegisterOption("PersistentKDTree.ci", optInt, "persistent_kdtree_ci=", "1");
	RegisterOption("PersistentKDTree.ct", optInt, "persistent_kdtree_ct=", "1");

	RegisterOption("PersistentKDTree.childLimit", optInt, "persistent_kdtree_childLimit=", "0");
	
	RegisterOption("PersistentKDTree.popCount", optInt, "persistent_kdtree_popCount=", "14");
	RegisterOption("PersistentKDTree.granularity", optFloat, "persistent_kdtree_granularity=", "50.0");

	RegisterOption("PersistentKDTree.nodeRatio", optInt, "persistent_kdtree_nodeRatio=", "5");
	RegisterOption("PersistentKDTree.triRatio", optInt, "persistent_kdtree_triRatio=", "3");
	RegisterOption("PersistentKDTree.idxRatio", optInt, "persistent_kdtree_idxRatio=", "12");
	RegisterOption("PersistentKDTree.heapMultiplicator", optFloat, "persistent_kdtree_heap_mult=", "20.0");
}


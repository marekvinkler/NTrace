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

#include "bvh/HLBVH/HLBVHBuilder.hpp"
#include "gpu/CudaCompiler.hpp"
#include "base/Math.hpp"
#include "radixSort.hpp"

#include "bvh/HLBVH/emitTreeKernel.cuh"

#include <cuda_runtime_api.h>

using namespace FW;
#define BENCHMARK
const float MB = (float)(1024*1024);

//------------------------------------------------------------------------

HLBVHBuilder::HLBVHBuilder(Scene* scene, const Platform& platform, HLBVHParams params)
	: CudaBVH(BVHLayout_Compact), m_scene(scene),m_platform(platform),m_params(params)
{
	//m_params.epsilon = 0.f;
	//cudaDeviceSetLimit(cudaLimitPrintfFifoSize, 1024*1024*500);

	m_sizeTask = 0;
	m_sizeSplit = 0;
	m_sizeADS = 0;
	m_sizeTri = 0;
	m_sizeTriIdx = 0;

	if (!params.hlbvh || params.hlbvhBits == 10)
		buildLBVH();
	else
		buildHLBVH();
}

//------------------------------------------------------------------------

HLBVHBuilder::~HLBVHBuilder(void)
{
}

//------------------------------------------------------------------------

void HLBVHBuilder::getStats(U32& nodes, U32& leaves, U32& nodeTop)
{
	nodes = m_nodes;
	leaves = m_leafs;
	nodeTop = m_nodes;
}

//------------------------------------------------------------------------

void HLBVHBuilder::calcMortonAndSort(Buffer &triMorton, Buffer &triIdx)
{
	CudaKernel kernelMorton = module->getKernel("calcMorton");

	// scene AABB
	Vec3f sceneMin, sceneMax;
	m_scene->getBBox(sceneMin, sceneMax);

	////////////// morton codes
	const float k2 = 1024.0f; // 2^n (n = 10)
	//const U32 k2 = (1 << n);
	//const U32 k2 = pow(2,n);
	Vec3f step = (sceneMax - sceneMin) / k2;
		
	module->setParami(kernelMorton, 0, triCnt);
	module->setParamf(kernelMorton, 4, sceneMin.x);
	module->setParamf(kernelMorton, 8, sceneMin.y);
	module->setParamf(kernelMorton, 12, sceneMin.z);
	module->setParamf(kernelMorton, 16, step.x);
	module->setParamf(kernelMorton, 20, step.y);
	module->setParamf(kernelMorton, 24, step.z);
	F32 cudaTime = module->launchKernelTimed(kernelMorton, Vec2i(BLOCK_SIZE,1), Vec2i((triCnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
	cudaTotalTime += cudaTime;
#ifndef BENCHMARK
	printf("? Morton codes: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! Morton codes: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
#endif

	////////////// radix sort
	cudaTime = radixSortCuda(triMorton.getMutableCudaPtr(), triIdx.getMutableCudaPtr(), triCnt);
	cudaTotalTime += cudaTime;
#ifndef BENCHMARK
	printf("? Radix sort: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! Radix sort: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());		
#endif
}

void HLBVHBuilder::createClustersC(Buffer &triMorton, S32 d, Buffer &clusters)
{
	CudaKernel kernelClusterAABB = module->getKernel("clusterAABB");

	//////////// compute clusters 
	clusters.resize((triCnt+1) * sizeof(U32));

	F32 cudaTime = createClusters(triMorton.getMutableCudaPtr(), triCnt, d, clusters.getMutableCudaPtr(), cluster_cnt);	
	cudaTotalTime += cudaTime;
	
	clusters.resize((cluster_cnt+1) * sizeof(U32));
	*(CUdeviceptr*)module->getGlobal("g_clsStart").getMutablePtr() = clusters.getCudaPtr();
	
#ifndef BENCHMARK
	printf("Clusters: %d\n", cluster_cnt);
	printf("? Cluster create: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! Cluster create: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
#endif

	////////////// clusters AABB
	cluster_bb.resize(cluster_cnt*sizeof(AABB));
	cluster_bin_id.resize(cluster_cnt*sizeof(S32)*3);
	cluster_split_id.resize(cluster_cnt*sizeof(S32));

	*(CUdeviceptr*)module->getGlobal("g_clsAABB").getMutablePtr() = cluster_bb.getMutableCudaPtr(); // cluster AABB
	*(CUdeviceptr*)module->getGlobal("g_clsBinId").getMutablePtr() = cluster_bin_id.getMutableCudaPtr(); // cluster bin ID
	*(CUdeviceptr*)module->getGlobal("g_clsSplitId").getMutablePtr() = cluster_split_id.getMutableCudaPtr(); // cluster node ID

	cluster_split_id.clear();

#if CLUSTER_AABB == 3
	CudaKernel kernelInitBins = module->getKernel("initClusterAABB");

	module->setParami(kernelInitBins, 0, cluster_cnt);
	cudaTime = module->launchKernelTimed(kernelInitBins, Vec2i(BLOCK_SIZE,1), Vec2i((cluster_cnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
#endif
	
	module->setParami(kernelClusterAABB, 0, cluster_cnt);
	module->setParami(kernelClusterAABB, sizeof(S32), triCnt);
#if CLUSTER_AABB == 0
	cudaTime = module->launchKernelTimed(kernelClusterAABB, Vec2i(BLOCK_SIZE,1), Vec2i((cluster_cnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
#elif CLUSTER_AABB == 1
	int warpsPerBlock = BLOCK_SIZE/WARP_SIZE;
	cudaTime = module->launchKernelTimed(kernelClusterAABB, Vec2i(WARP_SIZE,warpsPerBlock), Vec2i((cluster_cnt-1+warpsPerBlock)/warpsPerBlock, 1));
#elif CLUSTER_AABB == 2
	cudaTime = module->launchKernelTimed(kernelClusterAABB, Vec2i(BLOCK_SIZE,1), Vec2i(cluster_cnt, 1));
#elif CLUSTER_AABB == 3
	cudaTime += module->launchKernelTimed(kernelClusterAABB, Vec2i(BLOCK_SIZE,1), Vec2i(NUM_BLOCKS, 1));
#endif
	cudaTotalTime += cudaTime;
#ifndef BENCHMARK
	printf("? Cluster AABB: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! Cluster AABB: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
#endif
}

void HLBVHBuilder::buildTopLevel(Buffer *ooq, U32 &nodeWritten, U32 &nodeCreated, Buffer &clusters)
{
	CudaKernel kernelSAHInitBins = module->getKernel("initBins");

	CudaKernel kernelSAHFillBins = module->getKernel("fillBins");

	CudaKernel kernelSAHSplit = module->getKernel("findSplit");

	CudaKernel kernelSAHDistribute = module->getKernel("distribute");

	// scene AABB
	Vec3f sceneMin, sceneMax;
	m_scene->getBBox(sceneMin, sceneMax);

	////////////// top-level SAH
	U32 sahCreated = 1;
	S64 bufferSize = 2*cluster_cnt;
	
	Buffer qs0_bb, qs0_cls, qs0_id, qs0_plane, qs0_child;
	Buffer qs1_bb, qs1_cls, qs1_id, qs1_plane, qs1_child;
	Buffer *qsi_bb, *qsi_cls, *qsi_id, *qsi_plane, *qsi_child;
	Buffer *qso_bb, *qso_cls, *qso_id, *qso_plane, *qso_child;
	Buffer *qst_bb, *qst_cls, *qst_id, *qst_plane, *qst_child;

	qso_bb = &qs1_bb;
	qso_cls = &qs1_cls;
	qso_id = &qs1_id;
	qso_plane = &qs1_plane;
	qso_child = &qs1_child;

	qsi_bb = &qs0_bb;
	qsi_cls = &qs0_cls;
	qsi_id = &qs0_id;
	qsi_plane = &qs0_plane;
	qsi_child = &qs0_child;

	// resize input queue
	qsi_bb->resize(bufferSize * sizeof(AABB));
	qsi_cls->resize(bufferSize * sizeof(S32));
	qsi_id->resize(bufferSize * sizeof(S32));
	qsi_plane->resize(bufferSize * sizeof(S32));
	qsi_child->resize(bufferSize * sizeof(S32));

	// resize output queue
	qso_bb->resizeDiscard(bufferSize * sizeof(AABB));
	qso_cls->resizeDiscard(bufferSize * sizeof(S32));
	qso_id->resizeDiscard(bufferSize * sizeof(S32));
	qso_plane->resizeDiscard(bufferSize * sizeof(S32));
	qso_child->resizeDiscard(bufferSize * sizeof(S32));

	m_sizeTask += (qs0_bb.getSize() + qs1_bb.getSize()
		+ qs0_cls.getSize() + qs1_cls.getSize()
		+ qs0_id.getSize() + qs1_id.getSize()
		+ qs0_plane.getSize() + qs1_plane.getSize()
		+ qs0_child.getSize() + qs1_child.getSize()) / MB;

	// insert first split task
	*(S32*)qsi_id->getMutablePtr() = 0;
	*(S32*)qsi_cls->getMutablePtr() = cluster_cnt;
	*(S32*)qsi_child->getMutablePtr() = -1;

	memcpy((void*)qsi_bb->getMutablePtr(), &sceneMin, sizeof(sceneMin));
	memcpy((void*)qsi_bb->getMutablePtr(sizeof(sceneMin)), &sceneMax, sizeof(sceneMax));

	*(CUdeviceptr*)module->getGlobal("g_ooq").getMutablePtr() = ooq->getMutableCudaPtr();		
	module->getGlobal("g_oofs").clear();

	// bins	
	Buffer bin_bb, bin_cnt;
	// init bins
	bin_bb.resizeDiscard(sizeof(AABB)*BIN_CNT*3*bufferSize);
	bin_cnt.resizeDiscard(sizeof(S32)*BIN_CNT*3*bufferSize); //x,y,z * BIN_CNT

	// Save sizes
	m_sizeSplit = (bin_bb.getSize() + bin_cnt.getSize()) / MB;

	*(CUdeviceptr*)module->getGlobal("g_binAABB").getMutablePtr() = bin_bb.getMutableCudaPtr();
	*(CUdeviceptr*)module->getGlobal("g_binCnt").getMutablePtr() = bin_cnt.getMutableCudaPtr();

	U32 sahLvl = 0;
	U32 sahTerminated = 0;
	U32 oldTerminated = 0;
	U32 sahWritten = 1;

	//Array<U32> lvlNodes;
	//lvlNodes.add(sahCreated - sahSingles);	
	
#ifndef BENCHMARK
	printf("! top-level SAH prepare: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
#endif

	F32 cudaTime = 0.0f;
	while (sahCreated > 0) {
		// fill task split input queue 
		*(CUdeviceptr*)module->getGlobal("g_qsiAABB").getMutablePtr() = qsi_bb->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsiCnt").getMutablePtr() = qsi_cls->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsiId").getMutablePtr() = qsi_id->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsiPlane").getMutablePtr() = qsi_plane->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsiChildId").getMutablePtr() = qsi_child->getMutableCudaPtr();
		
		// fill task split output queue 
		*(CUdeviceptr*)module->getGlobal("g_qsoAABB").getMutablePtr() = qso_bb->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsoCnt").getMutablePtr() = qso_cls->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsoId").getMutablePtr() = qso_id->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsoPlane").getMutablePtr() = qso_plane->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsoChildId").getMutablePtr() = qso_child->getMutableCudaPtr();

		module->setParami(kernelSAHInitBins, 0, sahCreated*BIN_CNT*3);
		cudaTime += module->launchKernelTimed(kernelSAHInitBins, Vec2i(BLOCK_SIZE,1), Vec2i((sahCreated*BIN_CNT*3-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
		
		// fill bins
		module->setParami(kernelSAHFillBins, 0, cluster_cnt);
		cudaTime += module->launchKernelTimed(kernelSAHFillBins, Vec2i(BLOCK_SIZE,1), Vec2i((cluster_cnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
		//cuCtxSynchronize(); // Flushes printfs
		//exit(0);
		
		// find SAH split
		module->getGlobal("g_sahCreated").clear();

		module->setParami(kernelSAHSplit, 0, sahCreated);
		module->setParami(kernelSAHSplit, 4, sahWritten);
		cudaTime += module->launchKernelTimed(kernelSAHSplit, Vec2i(BLOCK_SIZE,1), Vec2i((sahCreated-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
				
		// cluster distribution		
		module->setParami(kernelSAHDistribute, 0, cluster_cnt);
		module->setParami(kernelSAHDistribute, 4, sahWritten);
		cudaTime += module->launchKernelTimed(kernelSAHDistribute, Vec2i(BLOCK_SIZE,1), Vec2i((cluster_cnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
		
		sahTerminated = *(U32*)module->getGlobal("g_oofs").getPtr(); // terminated total
		sahCreated = *(U32*)module->getGlobal("g_sahCreated").getPtr();  // created + terminated - leafs
		
		S32 terminated = sahTerminated - oldTerminated;
		oldTerminated = sahTerminated;
		
		if (sahCreated != 0) // old sahCreated
			lvlNodes.add(sahCreated); // - new sahSingles		
		
		sahWritten += sahCreated;
		sahCreated -= terminated;
		
		//printf("%2d: nodes %d, written %d[%d], ", sahLvl, sahCreated - sahSingles + terminated, sahCreated - sahSingles + terminated, sahWritten);
		//printf("created %d, terminated %d, leafs %d, offset %d[%d]\n", sahCreated, terminated, *(U32*)(module->getGlobal("g_leafsPtr").getPtr(0)),
		//																sahCreated - sahSingles + terminated, sahWritten);
		
		qst_bb = qsi_bb; qst_cls = qsi_cls; qst_id = qsi_id; qst_plane = qsi_plane; qst_child = qsi_child;
		qsi_bb = qso_bb; qsi_cls = qso_cls; qsi_id = qso_id; qsi_plane = qso_plane; qsi_child = qso_child;
		qso_bb = qst_bb; qso_cls = qst_cls; qso_id = qst_id; qso_plane = qst_plane; qso_child = qst_child;
		
		sahLvl++;
		//cout << "SAHLevel " << sahLvl << " nodes " << sahCreated << "\n";
		//getchar();
	}	
	cudaTotalTime += cudaTime;

#ifndef BENCHMARK
	//printf("SAH: written %d, leafs %d\n", sahWritten, sahTerminated);
	//getNodeBuffer().resize(sahWritten*64);
	printf("? top-level SAH: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! top-level SAH: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
#endif

	nodeWritten = sahWritten;
	nodeCreated = *(S32*)module->getGlobal("g_oofs").getPtr();
}

void HLBVHBuilder::buildBottomLevel(Buffer *q_in, Buffer *q_out, U32 &nodeWritten, U32 &nodeCreated, U32 bOfs, U32 n_bits)
{
	CudaKernel kernel = module->getKernel("emitTreeKernel");

	//////////////////////////////////////////////////////////////////////////
	// LBVH
	//////////////////////////////////////////////////////////////////////////

	Buffer *q_tmp;
	S32 bit_ofs = bOfs;

#ifndef BENCHMARK
	printf("! LBVH prepare: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
	//printf("Building LBVH... in %d, ofs %d\n", nodeCreated, nodeWritten);	
#endif

	F32 cudaTime = 0.0f;
	S32 level = 0;
	while((level < (n_bits-bit_ofs)) && nodeCreated > 0) {
		*(CUdeviceptr*)module->getGlobal("g_inQueueMem").getMutablePtr() = q_in->getCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_outQueueMem").getMutablePtr() = q_out->getMutableCudaPtr();

		module->getGlobal("g_inQueuePtr").clear();
		module->getGlobal("g_outQueuePtr").clear();

		Vec2i blockSize(BLOCK_SIZE, 1);
		Vec2i gridSize((nodeCreated - 1 + BLOCK_SIZE)/BLOCK_SIZE, 1);
		module->setParami(kernel, 0, n_bits - (level+1 + bit_ofs));
		module->setParami(kernel, 4, nodeCreated);
		module->setParami(kernel, 8, nodeWritten);
		cudaTime += module->launchKernelTimed(kernel, blockSize, gridSize);
		
		nodeCreated = *(U32*)(module->getGlobal("g_outQueuePtr").getPtr(0));
		lvlNodes.add(nodeCreated);

		nodeWritten += nodeCreated;
		if (lvlNodes.getLast() == 0)
			lvlNodes.removeLast();		
		
		q_tmp = q_in;
		q_in = q_out;
		q_out = q_tmp;

		level++;
		//printf("lvl %d: created %d, written %d, leafs %d\n", level, nodeCreated, nodeWritten, *(S32*)module->getGlobal("g_leafsPtr").getPtr());
		//getchar();
	}
	cudaTotalTime += cudaTime;	
	
#ifndef BENCHMARK
	//printf("LBVH: written %d, leafs %d [%d]\n", nodeWritten, leafs, nodeWritten-sahWritten);
	printf("? bottom-level LBVH: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! bottom-level LBVH: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
	//printf("? LBVH build done: %f\n", cudaTime);	
	//printf("! LBVH build done: %f [%f], levels %d, nodes %d, leafs %d\n", m_progressTimer.end(), m_progressTimer.getTotal(), level, nodeWritten, leafs);
	
	//printf("Last level: %d of %d\n", level-1, n_bits-1);			
#endif
	
	m_progressTimer.end();
	U32 leafs = *(U32*)(module->getGlobal("g_leafsPtr").getPtr(0));
	//printf("Resizing node buffer %d -> %d [%d]\n", getNodeBuffer().getSize(), nodeWritten*64, getNodeBuffer().getSize() - nodeWritten*64);
	getNodeBuffer().resize(nodeWritten*64);
	//printf("Resizing woop buffer %d -> %d [%d]\n", getTriWoopBuffer().getSize(), triCnt*4*4*3+leafs*4*4, getTriWoopBuffer().getSize() - (triCnt*4*4*3+leafs*4*4));
	getTriWoopBuffer().resize(triCnt*4*4*3+leafs*4*4);
	//printf("Resizing index buffer %d -> %d [%d]\n", getTriIndexBuffer().getSize(), triCnt*4*3+leafs*4, getTriIndexBuffer().getSize() - (triCnt*4*3+leafs*4));
	getTriIndexBuffer().resize(triCnt*4*3+leafs*4);

	m_sizeADS = getNodeBuffer().getSize() / MB;
	m_sizeTri = getTriWoopBuffer().getSize() / MB;
	m_sizeTriIdx = getTriIndexBuffer().getSize() / MB;
	m_progressTimer.start();

#ifdef LEAF_HISTOGRAM
	U32 *histogram = (U32*)(module->getGlobal("g_leafHist").getPtr());
	printf("Leaf histogram\n");
	U32 leafSum = 0;
	U32 triSum = 0;
	for(S32 i = 0; i <= m_params.leafSize; i++)
	{
		printf("%d: %d\n", i, histogram[i]);
		leafSum += histogram[i];
		triSum += i*histogram[i];
	}
	printf("Leafs total %d, average leaf %.2f\n", leafSum, (float)triSum/(float)leafSum);
#endif
}

void HLBVHBuilder::calcAABB(U32 nodeWritten)
{
	CudaKernel kernelAABB = module->getKernel("calcAABB");

	*(CUdeviceptr*)module->getGlobal("g_outNodes").getMutablePtr() = getNodeBuffer().getMutableCudaPtr();	
	
#ifdef MEASURE_STATS
	module->getGlobal("g_ga").clear();
	module->getGlobal("g_gb").clear();
	module->getGlobal("g_gc").clear();
	*(S32*)module->getGlobal("g_gd").getMutablePtr() = -1;
#endif

#ifndef BENCHMARK
	printf("! Refit nodes: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
#endif
	
	F32 cudaTime = 0.0f;
	//S32 aa = 0, bb = 0;
	for (S32 lvl = lvlNodes.getSize()-1; lvl >= 0; lvl--) { //    > 0    dont recalculated top level AABBs? already done in SAH?
		nodeWritten -= lvlNodes[lvl];
		
		Vec2i blockSize(BLOCK_SIZE, 1);
		Vec2i gridSize((lvlNodes[lvl] - 1 + BLOCK_SIZE)/BLOCK_SIZE, 1);		
		
		module->setParami(kernelAABB, 0, nodeWritten);
		module->setParami(kernelAABB, 4, lvlNodes[lvl]);
		//printf("IN: %d %d\n", nodeWritten, lvlNodes[lvl]);
		cudaTime += module->launchKernelTimed(kernelAABB, blockSize, gridSize);		
		//module->launchKernel(kernelAABB, blockSize, gridSize);
		
		//printf("Level %d: time %f [nodes %d - %d] start %d, cnt %d", lvl, cudaTime, nodeWritten, nodeWritten+lvlNodes[lvl], nodeWritten, lvlNodes[lvl]);
		//printf(", nodes %d, leafs %d\n", *(S32*)module->getGlobal("g_ga").getPtr() - aa, *(S32*)module->getGlobal("g_gb").getPtr() - bb);
		//aa = *(S32*)module->getGlobal("g_ga").getPtr();
		//bb = *(S32*)module->getGlobal("g_gb").getPtr();
		//getchar();
	}	
	cudaTotalTime += cudaTime;
#ifndef BENCHMARK
	printf("? calcAABB GPU: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! calcAABB GPU: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
#endif
	//printf("calcAABB GPU: nodes %d, leafs %d, tris %d, biggest leaf: %d\n", *(S32*)module->getGlobal("g_ga").getPtr(),
														  //*(S32*)module->getGlobal("g_gb").getPtr(),
														  //*(S32*)module->getGlobal("g_gc").getPtr(),
														  //*(S32*)module->getGlobal("g_gd").getPtr());
}

void HLBVHBuilder::buildLBVH(void)
{
	F32 cudaTime = 0.0f;
	cudaTotalTime = 0.0f;

	// morton codes of order n => resulting in 3n bit grid
	S32 n = 10;
	S32 n_bits = 3 * n;	

	// compile CUDA kernels
	CudaCompiler m_compiler;
	m_compiler.addOptions("-use_fast_math -Xptxas=\"-v\"");
	m_compiler.setSourceFile("src/rt/bvh/HLBVH/emitTreeKernel.cu");
	m_compiler.clearDefines();

	if (CudaModule::getComputeCapability() == 20 || CudaModule::getComputeCapability() == 21)
		m_compiler.define("FERMI");

	module = m_compiler.compile();
	failIfError();

	// Set leaf size and scene epsilon
	*(int*)module->getGlobal("c_leafSize").getMutablePtr() = m_params.leafSize;
	*(float*)module->getGlobal("c_epsilon").getMutablePtr() = m_params.epsilon;

#ifdef LEAF_HISTOGRAM
	module->getGlobal("g_leafHist").clear();
#endif

	CudaKernel kernelWoop = module->getKernel("calcWoopKernel");

	m_progressTimer.unstart();
#ifndef BENCHMARK
	printf("HLBVHBuilder LBVH: Build start\n");
#endif
	m_progressTimer.start();
	
	triCnt = m_scene->getNumTriangles();

	// upload scene triangles and vertices
	*(CUdeviceptr*)module->getGlobal("g_tris").getMutablePtr() = m_scene->getTriangleBuffer().getCudaPtr();
	*(CUdeviceptr*)module->getGlobal("g_verts").getMutablePtr() = m_scene->getVertexBuffer().getCudaPtr();
#ifndef BENCHMARK
	printf("! Upload tris and verts: %f [total %f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
#endif

	// morton
	Buffer triMorton, triIdx;
	triMorton.resize(triCnt * sizeof(U32));
	triIdx.resize(triCnt * sizeof(S32));
	
	*(CUdeviceptr*)module->getGlobal("g_inTriMem").getMutablePtr() = triMorton.getMutableCudaPtr();
	*(CUdeviceptr*)module->getGlobal("g_inTriIdxMem").getMutablePtr() = triIdx.getMutableCudaPtr();
#ifndef BENCHMARK
	printf("! Alloc morton and index: %f [total %f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
#endif
			
	calcMortonAndSort(triMorton, triIdx);

	m_sizeTask += triMorton.getSize() / MB;
	
	////////////// woop data
#ifdef WOOP_TRIANGLES
	Buffer inWoop;
	inWoop.resizeDiscard(triCnt*3*sizeof(Vec4i));
	*(CUdeviceptr*)module->getGlobal("g_inWoopMem").getMutablePtr() = inWoop.getMutableCudaPtr();
	
	module->setParami(kernelWoop, 0, triCnt);
	cudaTime = module->launchKernelTimed(kernelWoop, Vec2i(BLOCK_SIZE,1), Vec2i((triCnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
	cudaTotalTime += cudaTime;
#ifndef BENCHMARK
	printf("? Woop data: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! Woop data: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
#endif

	m_sizeTask += inWoop.getSize() / MB;
#endif

	// alloc out woop and idx buffers
#ifdef COMPACT_LAYOUT
#ifdef WOOP_TRIANGLES
	getTriWoopBuffer().resizeDiscard(triCnt*(3+1)*sizeof(Vec4i));  // just to be sure
	*(CUdeviceptr*)module->getGlobal("g_outWoopMem").getMutablePtr() = getTriWoopBuffer().getMutableCudaPtr();
#else
	getTriBuffer().resizeDiscard(triCnt*(3+1)*sizeof(Vec4i));  // just to be sure
	*(CUdeviceptr*)module->getGlobal("g_outWoopMem").getMutablePtr() = getTriBuffer().getMutableCudaPtr();
#endif
	getTriIndexBuffer().resizeDiscard(triCnt*(3+1)*sizeof(S32));
#else
#ifdef WOOP_TRIANGLES
	getTriWoopBuffer().resizeDiscard(triCnt*3*sizeof(Vec4i));  // just to be sure
#else
	getTriBuffer().resizeDiscard(triCnt*3*sizeof(Vec4i));  // just to be sure
	*(CUdeviceptr*)module->getGlobal("g_outWoopMem").getMutablePtr() = getTriBuffer().getMutableCudaPtr();
#endif
	getTriIndexBuffer().resizeDiscard(triCnt*3*sizeof(S32));
#endif
	*(CUdeviceptr*)module->getGlobal("g_outIdxMem").getMutablePtr() = getTriIndexBuffer().getMutableCudaPtr();		

	module->getGlobal("g_leafsPtr").clear();		

	lvlNodes.clear();
	lvlNodes.add(1); // there is always 1 top root node

	U32 nodeWritten = 1;
	U32 nodeCreated = 1;

	// insert top node
	Buffer q0,q1;
	initMemory(q0, q1, min(2, m_params.leafSize));
	((S32*)q0.getMutablePtr())[0] = 0;
	((S32*)q0.getMutablePtr())[1] = 0;
	((S32*)q0.getMutablePtr())[2] = triCnt;

	m_sizeTask += (q0.getSize() + q1.getSize()) / MB;

	buildBottomLevel(&q0, &q1, nodeWritten, nodeCreated, 0, n_bits);
		
	calcAABB(nodeWritten);
	
	m_gpuTime = cudaTotalTime;
	m_progressTimer.end();
	m_cpuTime = m_progressTimer.getTotal();
#ifndef BENCHMARK
	printf("? Build finished: %f\n", m_gpuTime);
	printf("! Build finished: %f\n", m_cpuTime);
#endif
	
	F32* root = (F32*)getNodeBuffer().getPtr();
	U32 leafs = *(U32*)(module->getGlobal("g_leafsPtr").getPtr(0));
	/*printf("=== BVH stats: nodes %d, leafs %d\n", nodeWritten, leafs);
	printf("=== AABB: (%.1f %.1f %.1f) - (%.1f %.1f %.1f)\n",   min(root[0],root[4]),
																min(root[2],root[6]),
																min(root[8],root[10]),
																max(root[1],root[5]),	
																max(root[3],root[7]),	
																max(root[9],root[11]));*/
	
	/*Debug << "BVH Top = " << nodeWritten << " => number of inner nodes (number of tasks) = " << nodeWritten << " + number of leaves = " << leafs << "\n";
	Debug << "Sorted tris = " << triCnt << "\n\n";*/
	m_nodes = nodeWritten;
	m_leafs = leafs;
}

void HLBVHBuilder::buildHLBVH(void) 
{
	F32 cudaTime = 0.0f;
	cudaTotalTime = 0.0f;

	// morton codes of order n => resulting in 3n bit grid
	S32 n = 10;
	// sorting into coarse 3m bit grid according to m bit morton code
	S32 m = (n - m_params.hlbvhBits);
	// then sorting rest of primitives in each grid cell according to remaining 3(n-m) bits
	S32 d = 3 * (n - m); // top level SAH (clusters)

	S32 n_bits = 3 * n;	

	// compile CUDA kernels
	CudaCompiler m_compiler;
	m_compiler.addOptions("-use_fast_math -Xptxas=\"-v\"");
	m_compiler.setSourceFile("src/rt/bvh/HLBVH/emitTreeKernel.cu");
	m_compiler.clearDefines();

	if (CudaModule::getComputeCapability() == 20 || CudaModule::getComputeCapability() == 21)
		m_compiler.define("FERMI");

	module = m_compiler.compile();
	failIfError();

	// Set leaf size and scene epsilon
	*(int*)module->getGlobal("c_leafSize").getMutablePtr() = m_params.leafSize;
	*(float*)module->getGlobal("c_epsilon").getMutablePtr() = m_params.epsilon;

#ifdef LEAF_HISTOGRAM
	module->getGlobal("g_leafHist").clear();
#endif

	CudaKernel kernelWoop = module->getKernel("calcWoopKernel");

	m_progressTimer.unstart();
#ifndef BENCHMARK
	printf("HLBVHBuilder HLBVH: Build start\n");
#endif
	m_progressTimer.start();
	
	triCnt = m_scene->getNumTriangles();

	// upload scene triangles and vertices
	*(CUdeviceptr*)module->getGlobal("g_tris").getMutablePtr() = m_scene->getTriangleBuffer().getCudaPtr();
	*(CUdeviceptr*)module->getGlobal("g_verts").getMutablePtr() = m_scene->getVertexBuffer().getCudaPtr();
#ifndef BENCHMARK
	printf("! Upload tris and verts: %f [total %f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
#endif

	// morton
	Buffer triMorton, triIdx;
	triMorton.resize(triCnt * sizeof(U32));
	triIdx.resize(triCnt * sizeof(S32));
	
	*(CUdeviceptr*)module->getGlobal("g_inTriMem").getMutablePtr() = triMorton.getMutableCudaPtr();
	*(CUdeviceptr*)module->getGlobal("g_inTriIdxMem").getMutablePtr() = triIdx.getMutableCudaPtr();
#ifndef BENCHMARK
	printf("! Alloc morton and index: %f [total %f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
#endif
			
	calcMortonAndSort(triMorton, triIdx);

	m_sizeTask += triMorton.getSize() / MB;

	Buffer clusters;
	createClustersC(triMorton, d, clusters);

	m_sizeTask += clusters.getSize() / MB;
	m_sizeTask += cluster_bb.getSize() / MB;
	m_sizeTask += cluster_bin_id.getSize() / MB;
	m_sizeTask += cluster_split_id.getSize() / MB;
	
	////////////// woop data
#ifdef WOOP_TRIANGLES
	Buffer inWoop;
	inWoop.resizeDiscard(triCnt*3*sizeof(Vec4i));
	*(CUdeviceptr*)module->getGlobal("g_inWoopMem").getMutablePtr() = inWoop.getMutableCudaPtr();
	
	module->setParami(kernelWoop, 0, triCnt);
	cudaTime = module->launchKernelTimed(kernelWoop, Vec2i(BLOCK_SIZE,1), Vec2i((triCnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
	cudaTotalTime += cudaTime;
#ifndef BENCHMARK
	printf("? Woop data: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! Woop data: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
#endif

	m_sizeTask += inWoop.getSize() / MB;
#endif

	// alloc out woop and idx buffers
#ifdef COMPACT_LAYOUT
#ifdef WOOP_TRIANGLES
	getTriWoopBuffer().resizeDiscard(triCnt*(3+1)*sizeof(Vec4i));  // just to be sure
	*(CUdeviceptr*)module->getGlobal("g_outWoopMem").getMutablePtr() = getTriWoopBuffer().getMutableCudaPtr();
#else
	getTriBuffer().resizeDiscard(triCnt*(3+1)*sizeof(Vec4i));  // just to be sure
	*(CUdeviceptr*)module->getGlobal("g_outWoopMem").getMutablePtr() = getTriBuffer().getMutableCudaPtr();
#endif
	getTriIndexBuffer().resizeDiscard(triCnt*(3+1)*sizeof(S32));
#else
#ifdef WOOP_TRIANGLES
	getTriWoopBuffer().resizeDiscard(triCnt*3*sizeof(Vec4i));  // just to be sure
#else
	getTriBuffer().resizeDiscard(triCnt*3*sizeof(Vec4i));  // just to be sure
	*(CUdeviceptr*)module->getGlobal("g_outWoopMem").getMutablePtr() = getTriBuffer().getMutableCudaPtr();
#endif
	getTriIndexBuffer().resizeDiscard(triCnt*3*sizeof(S32));
#endif
	*(CUdeviceptr*)module->getGlobal("g_outIdxMem").getMutablePtr() = getTriIndexBuffer().getMutableCudaPtr();		

	module->getGlobal("g_leafsPtr").clear();		

	lvlNodes.clear();
	lvlNodes.add(1); // there is always 1 top root node

	U32 nodeWritten, nodeCreated;
	Buffer q0, q1;
	initMemory(q0, q1, (d == 0) ? 1 : min(2, m_params.leafSize));

	m_sizeTask += (q0.getSize() + q1.getSize()) / MB;
	
	buildTopLevel(&q0, nodeWritten, nodeCreated, clusters);
	if(d != 0)
		buildBottomLevel(&q0, &q1, nodeWritten, nodeCreated, 3*m, n_bits);
	calcAABB(nodeWritten);

	//q0.free(Buffer::Module::CPU);

	//clusters.clear();
	//cluster_bb.clear();
	//cluster_bin_id.clear();
	//cluster_split_id.clear();
	
	m_gpuTime = cudaTotalTime;
	m_progressTimer.end();
	m_cpuTime = m_progressTimer.getTotal();

#ifndef BENCHMARK
	printf("? Build finished: %f\n", m_gpuTime);
	printf("! Build finished: %f\n", m_cpuTime);
#endif
	
	F32* root = (F32*)getNodeBuffer().getPtr();
	U32 leafs = *(U32*)(module->getGlobal("g_leafsPtr").getPtr());
	/*printf("=== BVH stats: nodes %d, leafs %d\n", nodeWritten, leafs);
	printf("=== AABB: (%.1f %.1f %.1f) - (%.1f %.1f %.1f)\n",   min(root[0],root[4]),
																min(root[2],root[6]),
																min(root[8],root[10]),
																max(root[1],root[5]),	
																max(root[3],root[7]),	
																max(root[9],root[11]));	*/
	
	/*Debug << "BVH Top = " << nodeWritten << " => number of inner nodes (number of tasks) = " << nodeWritten << " + number of leaves = " << leafs << "\n";
	Debug << "Sorted tris = " << triCnt << "\n\n";*/
	m_nodes = nodeWritten;
	m_leafs = leafs;
}
/*
void HLBVHBuilder::build(void)
{
	F32 cudaTime = 0.0f;
	F32 cudaTotalTime = 0.0f;

	// morton codes of order n => resulting in 3n bit grid
	S32 n = 10;
	// sorting into coarse 3m bit grid according to m bit morton code
	S32 m = 5;
	// then sorting rest of primitives in each grid cell according to remaining 3(n-m) bits
	S32 d = 3 * (n - m); // top level SAH (clusters)

	S32 n_bits = 3 * n;	

	CudaCompiler m_compiler;
	m_compiler.addOptions("-use_fast_math -Xptxas=\"-v\"");
	m_compiler.setSourceFile("src/rt/bvh/HLBVH/emitTreeKernel.cu");
	m_compiler.clearDefines();

	if (CudaModule::getComputeCapability() == 20 || CudaModule::getComputeCapability() == 21)
		m_compiler.define("FERMI");

	CudaModule* module = m_compiler.compile();
	failIfError();
	CudaKernel kernel = module->getKernel("emitTreeKernel", 12);
	if (!kernel)
		fail("Tree generation kernel not found!");

	CudaKernel kernelWoop = module->getKernel("calcWoopKernel", 4);
	if (!kernelWoop)
		fail("Woop calculation kernel not found!");		

	CudaKernel kernelMorton = module->getKernel("calcMorton", 28);
	if (!kernelMorton)
		fail("Morton calculation kernel not found!");

	CudaKernel kernelAABB = module->getKernel("calcAABB", 8);
	if (!kernelAABB)
		fail("AABB calculation kernel not found!");

	CudaKernel kernelSAHInitBins = module->getKernel("initBins", 4);
	if (!kernelSAHInitBins)
		fail("Bin initialization kernel not found!");

	CudaKernel kernelSAHFillBins = module->getKernel("fillBins", 4);
	if (!kernelSAHFillBins)
		fail("Bin filling kernel not found!");

	CudaKernel kernelSAHSplit = module->getKernel("findSplit", 8);
	if (!kernelSAHSplit)
		fail("SAH calculation kernel not found!");

	CudaKernel kernelSAHDistribute = module->getKernel("distribute", 8);
	if (!kernelSAHDistribute)
		fail("Cluster distribution kernel not found!");
		
	CudaKernel kernelClusterAABB = module->getKernel("clusterAABB", 4);
	if (!kernelClusterAABB)
		fail("ClusterAABB kernel not found!");

	m_progressTimer.unstart();
	printf("HLBVHBuilder: Build start\n");
	m_progressTimer.start();
	
	const S32 triCnt = m_scene->getNumTriangles();

	// upload scene triangles and vertices
	*(CUdeviceptr*)module->getGlobal("g_tris").getMutablePtr() = m_scene->getTriangleBuffer().getCudaPtr();
	*(CUdeviceptr*)module->getGlobal("g_verts").getMutablePtr() = m_scene->getVertexBuffer().getCudaPtr();
	printf("! Upload tris and verts: %f [total %f]\n", m_progressTimer.end(), m_progressTimer.getTotal());

	// scene AABB
	Vec3f sceneMin, sceneMax;
	m_scene->getBBox(sceneMin, sceneMax);

	Buffer triMorton, triIdx;
	triMorton.resize(triCnt * sizeof(U32));
	triIdx.resize(triCnt * sizeof(S32));
	
	*(CUdeviceptr*)module->getGlobal("g_inTriMem").getMutablePtr() = triMorton.getMutableCudaPtr();
	*(CUdeviceptr*)module->getGlobal("g_inTriIdxMem").getMutablePtr() = triIdx.getMutableCudaPtr();
	printf("! Alloc morton and index: %f [total %f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
			
	////////////// morton codes
	const float k2 = 1024.0f; // 2^n (n = 10)
	//const U32 k2 = (1 << n);
	//const U32 k2 = pow(2,n);
	Vec3f step = (sceneMax - sceneMin) / k2;	
		
	module->setParami(kernelMorton, 0, triCnt);
	module->setParamf(kernelMorton, 4, sceneMin.x);
	module->setParamf(kernelMorton, 8, sceneMin.y);
	module->setParamf(kernelMorton, 12, sceneMin.z);
	module->setParamf(kernelMorton, 16, step.x);
	module->setParamf(kernelMorton, 20, step.y);
	module->setParamf(kernelMorton, 24, step.z);
	cudaTime = module->launchKernelTimed(kernelMorton, Vec2i(BLOCK_SIZE,1), Vec2i((triCnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
	cudaTotalTime += cudaTime;
	printf("? Morton codes: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! Morton codes: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
		
	////////////// radix sort
	cudaTime = radixSortCuda(triMorton.getMutableCudaPtr(), triIdx.getMutableCudaPtr(), triCnt);
	cudaTotalTime += cudaTime;
	printf("? Radix sort: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! Radix sort: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
	
	//////////// compute clusters 
	S32 cluster_cnt;
	Buffer clusters;
	clusters.resize((triCnt+1) * 4);

	cudaTime = createClusters(triMorton.getMutableCudaPtr(), triCnt, d, clusters.getMutableCudaPtr(), cluster_cnt);	
	cudaTotalTime += cudaTime;
	
	clusters.resize((cluster_cnt+1) * 4);
	*(CUdeviceptr*)module->getGlobal("g_clsStart").getMutablePtr() = clusters.getCudaPtr();
	
	printf("Clusters: %d\n", cluster_cnt);
	printf("? Cluster create: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! Cluster create: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());

	//clusters.resize((cluster_cnt+1) * 4);
	//*(S32*)clusters.getMutablePtr(cluster_cnt*4) = triCnt;		

	////////////// clusters AABB
	Buffer cluster_bb;
	Buffer cluster_bin_id;
	Buffer cluster_split_id;

	cluster_bb.resize(cluster_cnt*sizeof(AABB));
	cluster_bin_id.resize(cluster_cnt*sizeof(S32)*3);
	cluster_split_id.resize(cluster_cnt*sizeof(S32));

	memset((void*)cluster_split_id.getPtr(), 0, cluster_split_id.getSize());	

	*(CUdeviceptr*)module->getGlobal("g_clsAABB").getMutablePtr() = cluster_bb.getMutableCudaPtr(); // cluster AABB
	*(CUdeviceptr*)module->getGlobal("g_clsBinId").getMutablePtr() = cluster_bin_id.getMutableCudaPtr(); // cluster bin ID
	*(CUdeviceptr*)module->getGlobal("g_clsSplitId").getMutablePtr() = cluster_split_id.getMutableCudaPtr(); // cluster node ID
	
	module->setParami(kernelClusterAABB, 0, cluster_cnt);
	cudaTime = module->launchKernelTimed(kernelClusterAABB, Vec2i(BLOCK_SIZE,1), Vec2i((cluster_cnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));		
	cudaTotalTime += cudaTime;
	printf("? Cluster AABB: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! Cluster AABB: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
	
	////////////// woop data
	Buffer inWoop;
	inWoop.resizeDiscard(triCnt*3*sizeof(Vec4i));
	*(CUdeviceptr*)module->getGlobal("g_inWoopMem").getMutablePtr() = inWoop.getMutableCudaPtr();
	
	module->setParami(kernelWoop, 0, triCnt);
	cudaTime = module->launchKernelTimed(kernelWoop, Vec2i(BLOCK_SIZE,1), Vec2i((triCnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
	cudaTotalTime += cudaTime;
	printf("? Woop data: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! Woop data: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	

	// alloc out woop and idx buffers
	getTriWoopBuffer().resizeDiscard(triCnt*(3+1)*sizeof(Vec4i));  // just to be sure
	getTriIndexBuffer().resizeDiscard(triCnt*(3+1)*sizeof(S32));
	*(CUdeviceptr*)module->getGlobal("g_outWoopMem").getMutablePtr() = getTriWoopBuffer().getMutableCudaPtr();
	*(CUdeviceptr*)module->getGlobal("g_outIdxMem").getMutablePtr() = getTriIndexBuffer().getMutableCudaPtr();		

	module->getGlobal("g_leafsPtr").clear();		
	
	////////////// top-level SAH
	//U32 sahWritten = 0;
	U32 sahCreated = 1;
	
	Buffer qs0_bb, qs0_cls, qs0_id, qs0_plane, qs0_child;
	Buffer qs1_bb, qs1_cls, qs1_id, qs1_plane, qs1_child;
	Buffer *qsi_bb, *qsi_cls, *qsi_id, *qsi_plane, *qsi_child;
	Buffer *qso_bb, *qso_cls, *qso_id, *qso_plane, *qso_child;
	Buffer *qst_bb, *qst_cls, *qst_id, *qst_plane, *qst_child;

	qso_bb = &qs1_bb;
	qso_cls = &qs1_cls;
	qso_id = &qs1_id;
	qso_plane = &qs1_plane;
	qso_child = &qs1_child;

	qsi_bb = &qs0_bb;
	qsi_cls = &qs0_cls;
	qsi_id = &qs0_id;
	qsi_plane = &qs0_plane;
	qsi_child = &qs0_child;

	qsi_bb->resize(sahCreated * sizeof(AABB));
	qsi_cls->resize(sahCreated * sizeof(S32));
	qsi_id->resize(sahCreated * sizeof(S32));
	qsi_plane->resize(sahCreated * sizeof(S32));
	qsi_child->resize(sahCreated * sizeof(S32));

	// insert first split task
	*(S32*)qsi_id->getPtr() = 0;
	*(S32*)qsi_cls->getPtr() = cluster_cnt;
	*(S32*)qsi_child->getPtr() = -1;

	memcpy((void*)qsi_bb->getPtr(), &sceneMin, sizeof(sceneMin));
	memcpy((void*)qsi_bb->getPtr(sizeof(sceneMin)), &sceneMax, sizeof(sceneMax));

	getNodeBuffer().resize(2*cluster_cnt*64); // more?
	// maybe log2(cluster_cnt)?	
	
	Buffer ooq;
	ooq.resize(sizeof(S32)*3 * cluster_cnt*2); // log2 ??
	module->getGlobal("g_oofs").clear();

	// bins	
	Buffer bin_bb, bin_cnt;
	U32 sahLvl = 0;
	U32 sahTerminated = 0;
	U32 oldTerminated = 0;
	U32 sahWritten = 1;
	U32 sahSingles = 0;
	//U32 sahOffset = 1;	

	Array<U32> lvlNodes;
	lvlNodes.add(sahCreated - sahSingles);	
	
	printf("! top-level SAH prepare: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
	cudaTime = 0.0f;
	while (sahCreated > 0) {
		// fill task split input queue 
		*(CUdeviceptr*)module->getGlobal("g_qsiAABB").getMutablePtr() = qsi_bb->getCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsiCnt").getMutablePtr() = qsi_cls->getCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsiId").getMutablePtr() = qsi_id->getCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsiPlane").getMutablePtr() = qsi_plane->getCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsiChildId").getMutablePtr() = qsi_child->getCudaPtr();

		// resize output queue
		qso_bb->resizeDiscard(2*sahCreated * sizeof(AABB));
		qso_cls->resizeDiscard(2*sahCreated * sizeof(S32));
		qso_id->resizeDiscard(2*sahCreated * sizeof(S32));
		qso_plane->resizeDiscard(2*sahCreated * sizeof(S32));
		qso_child->resizeDiscard(2*sahCreated * sizeof(S32));		
		
		// fill task split output queue 
		*(CUdeviceptr*)module->getGlobal("g_qsoAABB").getMutablePtr() = qso_bb->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsoCnt").getMutablePtr() = qso_cls->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsoId").getMutablePtr() = qso_id->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsoPlane").getMutablePtr() = qso_plane->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsoChildId").getMutablePtr() = qso_child->getMutableCudaPtr();
		
		*(CUdeviceptr*)module->getGlobal("g_clsBinId").getMutablePtr() = cluster_bin_id.getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_clsSplitId").getMutablePtr() = cluster_split_id.getMutableCudaPtr();

		// init bins
		bin_bb.resizeDiscard(sizeof(AABB)*BIN_CNT*3*sahCreated);
		bin_cnt.resizeDiscard(sizeof(S32)*BIN_CNT*3*sahCreated); //x,y,z * BIN_CNT

		*(CUdeviceptr*)module->getGlobal("g_binAABB").getMutablePtr() = bin_bb.getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_binCnt").getMutablePtr() = bin_cnt.getMutableCudaPtr();
		module->setParami(kernelSAHInitBins, 0, sahCreated);
		cudaTime += module->launchKernelTimed(kernelSAHInitBins, Vec2i(BLOCK_SIZE,1), Vec2i((sahCreated-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
		
		// fill bins
		*(CUdeviceptr*)module->getGlobal("g_binAABB").getMutablePtr() = bin_bb.getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_binCnt").getMutablePtr() = bin_cnt.getMutableCudaPtr();
		module->setParami(kernelSAHFillBins, 0, cluster_cnt);
		cudaTime += module->launchKernelTimed(kernelSAHFillBins, Vec2i(BLOCK_SIZE,1), Vec2i((cluster_cnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
		
		// find SAH split		
		*(CUdeviceptr*)module->getGlobal("g_ooq").getMutablePtr() = ooq.getMutableCudaPtr();
		
		*(CUdeviceptr*)module->getGlobal("g_qsiCnt").getMutablePtr() = qsi_cls->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsiPlane").getMutablePtr() = qsi_plane->getMutableCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsiChildId").getMutablePtr() = qsi_child->getMutableCudaPtr();
		module->getGlobal("g_sahCreated").clear();
		*(CUdeviceptr*)module->getGlobal("g_outNodes").getMutablePtr() = getNodeBuffer().getMutableCudaPtr();

		module->setParami(kernelSAHSplit, 0, sahCreated);
		module->setParami(kernelSAHSplit, 4, sahWritten);
		cudaTime += module->launchKernelTimed(kernelSAHSplit, Vec2i(BLOCK_SIZE,1), Vec2i((sahCreated-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
				
		// cluster distribution
		*(CUdeviceptr*)module->getGlobal("g_qsiPlane").getMutablePtr() = qsi_plane->getCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_qsiChildId").getMutablePtr() = qsi_child->getCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_ooq").getMutablePtr() = ooq.getMutableCudaPtr();		
		
		module->setParami(kernelSAHDistribute, 0, cluster_cnt);
		module->setParami(kernelSAHDistribute, 4, sahWritten);
		cudaTime += module->launchKernelTimed(kernelSAHDistribute, Vec2i(BLOCK_SIZE,1), Vec2i((cluster_cnt-1+BLOCK_SIZE)/BLOCK_SIZE, 1));
		
		sahTerminated = *(U32*)module->getGlobal("g_oofs").getPtr(); // terminated total
		sahCreated = *(U32*)module->getGlobal("g_sahCreated").getPtr(0);  // created + terminated - leafs
		
		S32 terminated = sahTerminated - oldTerminated;
		oldTerminated = sahTerminated;
		
		if (sahCreated - sahSingles != 0) // old sahCreated
			lvlNodes.add(sahCreated - sahSingles); // - new sahSingles		
		
		sahWritten += sahCreated - sahSingles;
		sahCreated -= terminated;
		
		//printf("%2d: nodes %d, written %d[%d], ", sahLvl, sahCreated - sahSingles + terminated, sahCreated - sahSingles + terminated, sahWritten);
		//printf("created %d, terminated %d, leafs %d, offset %d[%d]\n", sahCreated, terminated, *(U32*)(module->getGlobal("g_leafsPtr").getPtr(0)),
			//															sahCreated - sahSingles + terminated, sahOffset);
		
		qst_bb = qsi_bb; qst_cls = qsi_cls; qst_id = qsi_id; qst_plane = qsi_plane; qst_child = qsi_child;
		qsi_bb = qso_bb; qsi_cls = qso_cls; qsi_id = qso_id; qsi_plane = qso_plane; qsi_child = qso_child;
		qso_bb = qst_bb; qso_cls = qst_cls; qso_id = qst_id; qso_plane = qst_plane; qso_child = qst_child;
		
		sahLvl++;
		//getchar();
	}	
	cudaTotalTime += cudaTime;
	//printf("SAH: written %d, leafs %d\n", sahWritten, sahTerminated);
	getNodeBuffer().resize(sahWritten*64);	
	printf("? top-level SAH: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! top-level SAH: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
	
	//////////////////////////////////////////////////////////////////////////
	// LBVH
	//////////////////////////////////////////////////////////////////////////

	Buffer q0,q1;
	Buffer *q_in = &q0;
	Buffer *q_out = &q1;
	Buffer *q_tmp;

	U32 nodeWritten = 1;
	U32 nodeCreated = 1;
	S32 bit_ofs = 0;

	// insert top node
	//q_in->resize(3*sizeof(S32) * nodeCreated);
	//((S32*)q_in->getMutablePtr())[0] = 0;
	//((S32*)q_in->getMutablePtr())[1] = 0;
	//((S32*)q_in->getMutablePtr())[2] = triCnt;

	// insert nodes from SAH
	
	//S32 ooq_size = *(S32*)module->getGlobal("g_oofs").getPtr();
	//q_in->set(ooq);	
	q_in = &ooq;
	nodeWritten = sahWritten;
	nodeCreated = *(S32*)module->getGlobal("g_oofs").getPtr();
	//getNodeBuffer().set(sahNodes); // not needed already used getNodeBuffer() insteaf of sahNodes
	bit_ofs = 3*m;

	//Array<S32> lvlNodes;
	//lvlNodes.add(nodeWritten);

	printf("! LBVH prepare: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
	//printf("Building LBVH... in %d, ofs %d\n", nodeCreated, nodeWritten);	
	cudaTime = 0.0f;
	U32 level = 0;
	do {
		if (nodeCreated == 0)
			break;
		getNodeBuffer().resize((nodeWritten + nodeCreated) * 64);
		q_out->resizeDiscard(3*sizeof(S32) * nodeCreated*2);		
				
		*(CUdeviceptr*)module->getGlobal("g_outNodes").getMutablePtr() = getNodeBuffer().getMutableCudaPtr();
		
		*(CUdeviceptr*)module->getGlobal("g_inQueueMem").getMutablePtr() = q_in->getCudaPtr();
		*(CUdeviceptr*)module->getGlobal("g_outQueueMem").getMutablePtr() = q_out->getMutableCudaPtr();
		
		module->getGlobal("g_inQueuePtr").clear();
		module->getGlobal("g_outQueuePtr").clear();

		Vec2i blockSize(BLOCK_SIZE, 1);
		Vec2i gridSize((nodeCreated - 1 + BLOCK_SIZE)/BLOCK_SIZE, 1);
		module->setParami(kernel, 0, n_bits - (level+1 + bit_ofs));
		module->setParami(kernel, 4, nodeCreated);
		module->setParami(kernel, 8, nodeWritten);
		cudaTime += module->launchKernelTimed(kernel, blockSize, gridSize);
		
		nodeCreated = *(U32*)(module->getGlobal("g_outQueuePtr").getPtr(0));
		lvlNodes.add(*(U32*)(module->getGlobal("g_outQueuePtr").getPtr(4)));		

		nodeWritten += lvlNodes.getLast();
		if (lvlNodes.getLast() == 0)
			lvlNodes.removeLast();		
		
		q_tmp = q_in;
		q_in = q_out;
		q_out = q_tmp;

		//printf("lvl %d: created %d, written %d, singles %d, leafs %d\n", level, nodeCreated, lvlNodes.getLast(), *(S32*)module->getGlobal("g_s").getPtr(),
			//*(S32*)module->getGlobal("g_l").getPtr());
		//getchar();
	} while((++level < (n_bits-bit_ofs)) && nodeCreated > 0);
	cudaTotalTime += cudaTime;	
	U32 leafs = *(U32*)(module->getGlobal("g_leafsPtr").getPtr(0));
	//printf("LBVH: written %d, leafs %d [%d]\n", nodeWritten, leafs, nodeWritten-sahWritten);
	printf("? bottom-level LBVH: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! bottom-level LBVH: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
	//printf("? LBVH build done: %f\n", cudaTime);	
	//printf("! LBVH build done: %f [%f], levels %d, nodes %d, leafs %d\n", m_progressTimer.end(), m_progressTimer.getTotal(), level, nodeWritten, leafs);
	
	//printf("Last level: %d of %d\n", level-1, n_bits-1);			
	
	//printf("Resizing node buffer %d -> %d [%d]\n", getNodeBuffer().getSize(), nodeWritten*64, getNodeBuffer().getSize() - nodeWritten*64);
	getNodeBuffer().resize(nodeWritten*64);
	//printf("Resizing woop buffer %d -> %d [%d]\n", getTriWoopBuffer().getSize(), triCnt*4*4*3+leafs*4*4, getTriWoopBuffer().getSize() - (triCnt*4*4*3+leafs*4*4));
	getTriWoopBuffer().resize(triCnt*4*4*3+leafs*4*4);
	//printf("Resizing index buffer %d -> %d [%d]\n", getTriIndexBuffer().getSize(), triCnt*4*3+leafs*4, getTriIndexBuffer().getSize() - (triCnt*4*3+leafs*4));
	getTriIndexBuffer().resize(triCnt*4*3+leafs*4);		
		
	//printf("! Resizing done: %f [total %f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
	//printf("Computing AABB...\n");
	S32 oldWritten = nodeWritten;
	*(CUdeviceptr*)module->getGlobal("g_outNodes").getMutablePtr() = getNodeBuffer().getMutableCudaPtr();	
	
	module->getGlobal("g_ga").clear();
	module->getGlobal("g_gb").clear();
	module->getGlobal("g_gc").clear();
	*(S32*)module->getGlobal("g_gd").getMutablePtr() = -1;
	printf("! Refit nodes: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
	
	cudaTime = 0.0f;
	//S32 aa = 0, bb = 0;
	for (S32 lvl = lvlNodes.getSize()-1; lvl >= 0; lvl--) { //    > 0    dont recalculated top level AABBs? already done in SAH?
		nodeWritten -= lvlNodes[lvl];
		
		Vec2i blockSize(BLOCK_SIZE, 1);
		Vec2i gridSize((lvlNodes[lvl] - 1 + BLOCK_SIZE)/BLOCK_SIZE, 1);		
		
		module->setParami(kernelAABB, 0, nodeWritten);
		module->setParami(kernelAABB, 4, lvlNodes[lvl]);
		//printf("IN: %d %d\n", nodeWritten, lvlNodes[lvl]);
		cudaTime += module->launchKernelTimed(kernelAABB, blockSize, gridSize);		
		//module->launchKernel(kernelAABB, blockSize, gridSize);
		
		//printf("Level %d: time %f [nodes %d - %d] start %d, cnt %d", lvl, cudaTime, nodeWritten, nodeWritten+lvlNodes[lvl], nodeWritten, lvlNodes[lvl]);
		//printf(", nodes %d, leafs %d\n", *(S32*)module->getGlobal("g_ga").getPtr() - aa, *(S32*)module->getGlobal("g_gb").getPtr() - bb);
		//aa = *(S32*)module->getGlobal("g_ga").getPtr();
		//bb = *(S32*)module->getGlobal("g_gb").getPtr();
		//getchar();
	}	
	cudaTotalTime += cudaTime;
	nodeWritten = oldWritten;
	printf("? calcAABB GPU: %f [%f]\n", cudaTime, cudaTotalTime);
	printf("! calcAABB GPU: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());	
	//printf("? calcAABB GPU done: %f\n", cudaTime);
	//printf("! calcAABB GPU done: %f [%f]\n", m_progressTimer.end(), m_progressTimer.getTotal());
	//printf("calcAABB GPU: nodes %d, leafs %d, tris %d, biggest leaf: %d\n", *(S32*)module->getGlobal("g_ga").getPtr(),
														  //*(S32*)module->getGlobal("g_gb").getPtr(),
														  //*(S32*)module->getGlobal("g_gc").getPtr(),
														  //*(S32*)module->getGlobal("g_gd").getPtr());
	//getchar();

	printf("? Build finished: %f\n", cudaTotalTime);
	m_progressTimer.end();
	printf("! Build finished: %f\n", m_progressTimer.getTotal());
	
	F32* root = (F32*)getNodeBuffer().getPtr();
	printf("=== BVH stats: nodes %d, leafs %d\n", nodeWritten, leafs);
	printf("=== AABB: (%.1f %.1f %.1f) - (%.1f %.1f %.1f)\n",   min(root[0],root[4]),
																min(root[2],root[6]),
																min(root[8],root[10]),
																max(root[1],root[5]),	
																max(root[3],root[7]),	
																max(root[9],root[11]));	
}
*/
F32 HLBVHBuilder::calcSAHGPU() {
	CudaCompiler m_compiler;
	m_compiler.addOptions("-use_fast_math -Xptxas=\"-v\"");
	m_compiler.setSourceFile("src/rt/bvh/HLBVH/emitTreeKernel.cu");
	m_compiler.clearDefines();

	if (CudaModule::getComputeCapability() == 20 || CudaModule::getComputeCapability() == 21)
		m_compiler.define("FERMI");

	CudaModule* module = m_compiler.compile();
	failIfError();
	CudaKernel kernel = module->getKernel("calcSAH");

	*(CUdeviceptr*)module->getGlobal("g_outWoopMem").getMutablePtr() = getTriWoopBuffer().getCudaPtr();
	module->getGlobal("g_sahCost").clear();
	kernel.launch(1, 1);

	return *(F32*)module->getGlobal("g_sahCost").getPtr();
}

void HLBVHBuilder::initMemory(Buffer& q_in, Buffer& q_out, int leafSize) {
	S64 size = 2*(triCnt/leafSize);
	getNodeBuffer().resize(size * 64);
	*(CUdeviceptr*)module->getGlobal("g_outNodes").getMutablePtr() = getNodeBuffer().getMutableCudaPtr();
	
	q_in.resize(3*sizeof(S32) * size);
	q_out.resizeDiscard(3*sizeof(S32) * size);
}

void HLBVHBuilder::getSizes(F32& task, F32& split, F32& ads, F32& tri, F32& triIdx)
{
	task = m_sizeTask;
	split = m_sizeSplit;
	ads = m_sizeADS;
	tri = m_sizeTri;
	triIdx = m_sizeTriIdx;
}
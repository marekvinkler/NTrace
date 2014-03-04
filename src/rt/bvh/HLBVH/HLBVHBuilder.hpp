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
#include "cuda/CudaBVH.hpp"
#include "base/Timer.hpp"

namespace FW
{

struct HLBVHParams
{
	bool hlbvh;
	S32 hlbvhBits;
	S32 leafSize;
	F32 epsilon;
};

//------------------------------------------------------------------------

class HLBVHBuilder : public CudaBVH
{
public:
							 HLBVHBuilder				(Scene* scene, const Platform& platform, HLBVHParams params);
    virtual                 ~HLBVHBuilder			    (void);

	float                   getCPUTime                  () { return m_cpuTime; }
	float                   getGPUTime                  () { return m_gpuTime; }
	void                    getStats                    (U32& nodes, U32& leaves, U32& nodeTop);
	void                    getSizes                    (F32& task, F32& split, F32& ads, F32& tri, F32& triIdx);

private:
                            HLBVHBuilder				(const HLBVHBuilder&); // forbidden
    HLBVHBuilder&           operator=					(const HLBVHBuilder&); // forbidden

	Scene*					m_scene;
    Platform				m_platform;
    BVHNode*				m_root;
	float                   m_cpuTime;
	float                   m_gpuTime;
	U32 m_leafs;
	U32 m_nodes;

	F32    m_sizeTask;
	F32    m_sizeSplit;
	F32    m_sizeADS;
	F32    m_sizeTri;
	F32    m_sizeTriIdx;

	S32 cluster_cnt;
	//Buffer clusters;
	Buffer cluster_bb;
	Buffer cluster_bin_id;
	Buffer cluster_split_id;

	F32 cudaTotalTime;
	CudaModule* module;
	S32 triCnt;
	Array<U32> lvlNodes;

protected:
	void build();
	void buildLBVH();
	void buildHLBVH();

	void calcMortonAndSort(Buffer &triMorton, Buffer &triIdx);
	void createClustersC(Buffer &triMorton, S32 d, Buffer &clusters);
	void calcAABB(U32 nodeWritten);

	void buildTopLevel(Buffer *ooq, U32 &nodeWritten, U32 &nodeCreated, Buffer &clusters);
	void buildBottomLevel(Buffer *q_in, Buffer *q_out, U32 &nodeWritten, U32 &nodeCreated, U32 bOfs, U32 n_bits);	
	
	F32 calcSAHGPU();
	F32 calcSAHCPU(S32 n);
	S32 calcLeafs(S32 n);
	void initMemory(Buffer& q_in, Buffer& q_out, int leafSize = 1);

    Timer                   m_progressTimer;
	HLBVHParams             m_params;
};

//------------------------------------------------------------------------
}

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

#include "bvh/BVH.hpp"
#include "bvh/SplitBVHBuilder.hpp"

using namespace FW;

BVH::BVH(Scene* scene, const Platform& platform, const BuildParams& params)
{
    FW_ASSERT(scene);
    m_scene = scene;
    m_platform = platform;

    if (params.enablePrints)
        printf("BVH builder: %d tris, %d vertices\n", scene->getNumTriangles(), scene->getNumVertices());

    m_root = SplitBVHBuilder(*this, params).run();

    if (params.enablePrints)
        printf("BVH: Scene bounds: (%.1f,%.1f,%.1f) - (%.1f,%.1f,%.1f)\n", m_root->m_bounds.min().x, m_root->m_bounds.min().y, m_root->m_bounds.min().z,
                                                                           m_root->m_bounds.max().x, m_root->m_bounds.max().y, m_root->m_bounds.max().z);

    float sah = 0.f;
    m_root->computeSubtreeProbabilities(m_platform, 1.f, sah);
    if (params.enablePrints)
        printf("top-down sah: %.2f\n", sah);

    if(params.stats)
    {
        params.stats->SAHCost           = sah;
        params.stats->branchingFactor   = 2;
        params.stats->numLeafNodes      = m_root->getSubtreeSize(BVH_STAT_LEAF_COUNT);
        params.stats->numInnerNodes     = m_root->getSubtreeSize(BVH_STAT_INNER_COUNT);
        params.stats->numTris           = m_root->getSubtreeSize(BVH_STAT_TRIANGLE_COUNT);
        params.stats->numChildNodes     = m_root->getSubtreeSize(BVH_STAT_CHILDNODE_COUNT);
    }
}

static S32 currentTreelet;
static Set<S32> uniqueTreelets;

void BVH::trace(RayBuffer& rays, RayStats* stats) const
{
    for(S32 i=0;i<rays.getSize();i++)
    {
        Ray ray = rays.getRayForSlot(i);    // takes a local copy
        RayResult& result = rays.getMutableResultForSlot(i);

        result.clear();

        currentTreelet = -2;
        uniqueTreelets.clear();

        if(stats)
        {
            stats->platform = m_platform;
            stats->numRays++;
        }

        traceRecursive(m_root, ray,result,rays.getNeedClosestHit(), stats);
    }
}

void BVH::traceRecursive(BVHNode* node, Ray& ray, RayResult& result,bool needClosestHit, RayStats* stats) const
{
    if(currentTreelet != node->m_treelet)
    {
        if(stats)
        {
//          if(!uniqueTreelets.contains(node->m_treelet))   // count unique treelets (comment this line to count all)
                stats->numTreelets++;
        }
        currentTreelet = node->m_treelet;
    }

    if(node->isLeaf())
    {
        const LeafNode* leaf = reinterpret_cast<const LeafNode*>(node);
        const Vec3i* triVtxIndex = (const Vec3i*)m_scene->getTriVtxIndexBuffer().getPtr();
        const Vec3f* vtxPos = (const Vec3f*)m_scene->getVtxPosBuffer().getPtr();

        if(stats)
            stats->numTriangleTests += m_platform.roundToTriangleBatchSize( leaf->getNumTriangles() );

        for(int i=leaf->m_lo; i<leaf->m_hi; i++)
        {
            S32 index = m_triIndices[i];
            const Vec3i& ind = triVtxIndex[index];
            const Vec3f& v0 = vtxPos[ind.x];
            const Vec3f& v1 = vtxPos[ind.y];
            const Vec3f& v2 = vtxPos[ind.z];
            Vec3f bary = Intersect::RayTriangle(v0,v1,v2, ray);
            float t = bary[2];

            if(t>ray.tmin && t<ray.tmax)
            {
                ray.tmax    = t;
                result.t    = t;
                result.id   = index;

                if(!needClosestHit)
                    return;
            }
        }
    }
    else
    {
        if(stats)
            stats->numNodeTests += m_platform.roundToNodeBatchSize( node->getNumChildNodes() );

        const int TMIN = 0;
        const int TMAX = 1;
        const InnerNode* inner = reinterpret_cast<const InnerNode*>(node);
        BVHNode* child0 = inner->m_children[0];
        BVHNode* child1 = inner->m_children[1];
        Vec2f tspan0 = Intersect::RayBox(child0->m_bounds, ray);
        Vec2f tspan1 = Intersect::RayBox(child1->m_bounds, ray);
        bool intersect0 = (tspan0[TMIN]<=tspan0[TMAX]) && (tspan0[TMAX]>=ray.tmin) && (tspan0[TMIN]<=ray.tmax);
        bool intersect1 = (tspan1[TMIN]<=tspan1[TMAX]) && (tspan1[TMAX]>=ray.tmin) && (tspan1[TMIN]<=ray.tmax);

        if(intersect0 && intersect1)
        if(tspan0[TMIN] > tspan1[TMIN])
        {
            swap(tspan0,tspan1);
            swap(child0,child1);
        }

        if(intersect0)
            traceRecursive(child0,ray,result,needClosestHit,stats);

        if(result.hit() && !needClosestHit)
            return;

//      if(tspan1[TMIN] <= ray.tmax)    // this test helps only about 1-2%
        if(intersect1)
            traceRecursive(child1,ray,result,needClosestHit,stats);
    }
}

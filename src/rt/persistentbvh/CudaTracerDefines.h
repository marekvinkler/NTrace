#define WARP_SIZE 32
#define LOG_WARP_SIZE 5
#define PLANE_COUNT 32

#define SCAN_TYPE 2 // 0 Means our naive scan
// 1 Means Harris
// 2 Means atomic partitioning
// 3 Means atomic partitioning with single atomic per warp

#define ISECT_TYPE 1 // 0 means parallelize by rays
// 1 means parallelize by intersection pairs

// BEST VALUES
// MEDIAN + SAH: 0
// NoStruct: 0
// Ondemand 3
//
#define ENQUEUE_TYPE 3 // 0: add from top to 0, then from top up
// 1: add from bottom up, then on active like 0
// 2: from top up
// 3: enqueue from cached items, then from top up

// BEST VALUES
// MEDIAN + SAH: 4
// NoStruct: 4
// Ondemand 5
//
#define DEQUEUE_TYPE 4 // 0: from top to 0, then again
// 1: both top to 0 and from 0 to top, then again
// 2: from modulo based index in the array to 0, then again
// 3: checks last active then same as 0
// 4: all warps participate in finding the active task
// 5: all warps participate in finding the active task + cache
// 6: all warps participate in finding the active task + repetitive cache

// BEST VALUES
// MEDIAN: 0
// SAH: 5
// NoStruct: 3
//
#define SPLIT_TYPE 5 // 0 Means median
// 1 Means cost model with many planes
// 2 Means cost model with 32 planes
// 3 Means cost model with 32 planes and parallel computation of ray-plane and tri-plane positions
// 4 Means cost model with PLANE_COUNT planes and parallel computation of SAH heuristic, round-robin change of bin axis (used only for bvh construction)
// 5 Means cost model with PLANE_COUNT planes and parallel computation of SAH heuristic, binning in all axes (used only for bvh construction)
// 6 Means cost model with PLANE_COUNT planes and parallel computation of SAH heuristic, triangle planes near leaves (used only for bvh construction)

#define AABB_TYPE 3 // 0 Means 1 level in a step
// 1 Means 5 levels in a step
// 2 Means 6 levels in a step
// 3 Means 5 levels in a step, min and max in a single phase

#define BINNING_TYPE 2 // 0 Means 32 planes, each warp processes 32 triangles with each thread processing one plane. All threads reduce their data to their own global memory.
// 1 Means parallel reduction over 32 triangles, each warp processes the 32 planes in sequence. All warps reduce their data to their own global memory.
// 2 Means naive atomics

#define CACHE_TYPE 1 // 0 Means overwrite cache
// 1 Means invalidating cache

#define BIN_MULTIPLIER 1 // Measures how much warpsized chunks is processed by a single warp at once in binning
#define POP_MULTIPLIER 2 // Measures how much warpsized chunks should be written to the pool and how much processed by a single warp

//#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
//#define OBJECT_SAH 
//#endif
#define SAH_TERMINATION
#define COMPUTE_MEDIAN_BOUNDS
#define SHUFFLE_RED_SCAN

#define TRIANGLE_CLIPPING 0 // 0 Means no clipping
							// 1 Means clipping during partition
							// 2 Means clipping during binning
							// 3 Means clipping during both
							// 4 Means exact triangleBox overlap test during partition
							// 5 Means exact triangleBox overlap test during binning
							// 6 Means exact triangleBox overlap test during both

#define NUM_SM 15 // Number of SMs on the device
#define NUM_WARPS_PER_BLOCK 4
#define NUM_BLOCKS_PER_SM 4
#define NUM_WARPS (NUM_SM*NUM_BLOCKS_PER_SM*NUM_WARPS_PER_BLOCK) // Number of warps on the device when 4 blocks with 128 threads are launched per SM
#define NUM_THREADS (NUM_WARPS_PER_BLOCK*WARP_SIZE)
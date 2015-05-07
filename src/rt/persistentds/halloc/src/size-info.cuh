/** @file size-infos.cuh implementation of some stuff related to size
		information */

#pragma once

/** information on sizes */
static __device__ size_info_t size_infos_g[MAX_NSIZES];
//static __constant__ size_info_t size_infos_g[MAX_NSIZES];

/** same data, but in different memory */
// __device__ size_info_t size_infos_dg[MAX_NSIZES];

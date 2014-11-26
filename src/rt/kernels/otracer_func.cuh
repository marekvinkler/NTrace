#ifndef __OTRACER_FUNC__CUH__
#define __OTRACER_FUNC__CUH__

#include <helper_math.h>

/**
 * Sample 2D texture
 *
 * @param bary Barycentric triangle coordinates on given pixel
 * @param tc Texture coordinates interpolated on given pixel
 * @param texAtlasInfo Texture atlas info for triangle to which given pixel belongs
 * @param tex Texture atlas image
 * @return Sampled texture at given pixel
 *
 */
extern "C" __device__ inline float4 sample2D(float3 bary, float2 tc, float4 texAtlasInfo, texture<float4, 2> tex)
{
	// Grab fract of texture coordinates
	tc.x = tc.x - floorf(tc.x);
	tc.y = tc.y - floorf(tc.y);
	
	// Calculate physical texture coordinates from atlas info
	tc.x = tc.x * texAtlasInfo.z + texAtlasInfo.x;
	tc.y = tc.y * texAtlasInfo.w + texAtlasInfo.y;
	
	// Return tex2D (Free bilinear filtering)
	return tex2D(tex, tc.x, tc.y);
}

/**
 * Interpolate attribute using barycentric coordinates
 *
 * @param bary Barycentric coordinates
 * @param attribA Attribute vertex 1
 * @param attribB Attribute vertex 2
 * @param attribC Attribute vertex 3
 * @return Interpolated attribute at given barycentric coordinates
 *
 */
extern "C" __device__ inline float interpolateAttribute1f(float3 bary, float attribA, float attribB, float attribC)
{
	return attribA * bary.x + attribB * bary.y + attribC * bary.z;
}

/**
 * Interpolate attribute using barycentric coordinates
 *
 * @param bary Barycentric coordinates
 * @param attribA Attribute vertex 1
 * @param attribB Attribute vertex 2
 * @param attribC Attribute vertex 3
 * @return Interpolated attribute at given barycentric coordinates
 *
 */
extern "C" __device__ inline float2 interpolateAttribute2f(float3 bary, float2 attribA, float2 attribB, float2 attribC)
{
	return make_float2(
			attribA.x * bary.x + attribB.x * bary.y + attribC.x * bary.z,
			attribA.y * bary.x + attribB.y * bary.y + attribC.y * bary.z
		);
}

/**
 * Interpolate attribute using barycentric coordinates
 *
 * @param bary Barycentric coordinates
 * @param attribA Attribute vertex 1
 * @param attribB Attribute vertex 2
 * @param attribC Attribute vertex 3
 * @return Interpolated attribute at given barycentric coordinates
 *
 */
extern "C" __device__ inline float3 interpolateAttribute3f(float3 bary, float3 attribA, float3 attribB, float3 attribC)
{
	return make_float3(
			attribA.x * bary.x + attribB.x * bary.y + attribC.x * bary.z,
			attribA.y * bary.x + attribB.y * bary.y + attribC.y * bary.z,
			attribA.z * bary.x + attribB.z * bary.y + attribC.z * bary.z
		);
}

/**
 * Interpolate attribute using barycentric coordinates
 *
 * @param bary Barycentric coordinates
 * @param attribA Attribute vertex 1
 * @param attribB Attribute vertex 2
 * @param attribC Attribute vertex 3
 * @return Interpolated attribute at given barycentric coordinates
 *
 */
extern "C" __device__ inline float4 interpolateAttribute4f(float3 bary, float4 attribA, float4 attribB, float4 attribC)
{
	return make_float4(
			attribA.x * bary.x + attribB.x * bary.y + attribC.x * bary.z,
			attribA.y * bary.x + attribB.y * bary.y + attribC.y * bary.z,
			attribA.z * bary.x + attribB.z * bary.y + attribC.z * bary.z,
			attribA.w * bary.x + attribB.w * bary.y + attribC.w * bary.z
		);
}

#endif
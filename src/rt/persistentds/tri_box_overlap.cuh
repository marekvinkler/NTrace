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

/********************************************************/

/* AABB-triangle overlap test code                      */

/* by Tomas Akenine-Möller                              */

/* Function: int triBoxOverlap(float boxcenter[3],      */

/*          float boxhalfsize[3],float triverts[3][3]); */

/* History:                                             */

/*   2001-03-05: released the code in its first version */

/*   2001-06-18: changed the order of the tests, faster */

/*                                                      */

/* Acknowledgement: Many thanks to Pierre Terdiman for  */

/* suggestions and discussions on how to optimize code. */

/* Thanks to David Hunt for finding a ">="-bug!         */

/********************************************************/

/*
    Triangle-AABB intersection test based on the code of Tomas Akenine-Möller.
	Rewritten to CUDA by Marek Vinkler.
*/

#include <math_constants.h>

//#define TRI_EPS 1e-8f
#define TRI_EPS 0.f

__device__ __forceinline__ int planeBoxOverlap(const float3& normal, const float3& vert, const float3& maxbox)
{
	float3 vmin,vmax;
	
	// X min-max
	if(normal.x > 0.0f)
	{
		vmin.x = -maxbox.x - vert.x;
		vmax.x =  maxbox.x - vert.x;
	}
	else
	{
		vmin.x =  maxbox.x - vert.x;
		vmax.x = -maxbox.x - vert.x;
	}
	// Y min-max
	if(normal.y > 0.0f)
	{
		vmin.y = -maxbox.y - vert.y;
		vmax.y =  maxbox.y - vert.y;
	}
	else
	{
		vmin.y =  maxbox.y - vert.y;
		vmax.y = -maxbox.y - vert.y;
	}
	// Z min-max
	if(normal.z > 0.0f)
	{
		vmin.z = -maxbox.z - vert.z;
		vmax.z =  maxbox.z - vert.z;
	}
	else
	{
		vmin.z =  maxbox.z - vert.z;
		vmax.z = -maxbox.z - vert.z;
	}

	if(dot(normal, vmin) > 0.0f)
		return 0;
	
	if(dot(normal, vmax) >= 0.0f)
		return 1;
	
	return 0;
}

__device__ __forceinline__ int planeBoxOverlap(const float3& normal, const float3& vert, const float3& minbox, const float3& maxbox)
{
	float3 vmin, vmax;

	// X min-max
	if(normal.x > 0.0f)
	{
		vmin.x = minbox.x - TRI_EPS - vert.x;
		vmax.x = maxbox.x + TRI_EPS - vert.x;
	}
	else
	{
		vmin.x = maxbox.x + TRI_EPS - vert.x;
		vmax.x = minbox.x - TRI_EPS - vert.x;
	}
	// Y min-max
	if(normal.y > 0.0f)
	{
		vmin.y = minbox.y - TRI_EPS - vert.y;
		vmax.y = maxbox.y + TRI_EPS - vert.y;
	}
	else
	{
		vmin.y = maxbox.y + TRI_EPS - vert.y;
		vmax.y = minbox.y - TRI_EPS - vert.y;
	}
	// Z min-max
	if(normal.z > 0.0f)
	{
		vmin.z = minbox.z - TRI_EPS - vert.z;
		vmax.z = maxbox.z + TRI_EPS - vert.z;
	}
	else
	{
		vmin.z = maxbox.z + TRI_EPS - vert.z;
		vmax.z = minbox.z - TRI_EPS - vert.z;
	}

	if(dot(normal, vmin) > 0.0f)
		return 0;
	
	if(dot(normal, vmax) >= 0.0f)
		return 1;
	
	return 0;
}

#if 1
/*======================== X-tests ========================*/

#define AXISTEST_X01(a, b, fa, fb)							\
	p0 = a*v0.y - b*v0.z;									\
	p2 = a*v2.y - b*v2.z;									\
	if(p0<p2) {mn=p0; mx=p2;} else {mn=p2; mx=p0;}		\
	rad = fa * boxhalfsize.y + fb * boxhalfsize.z + TRI_EPS;			\
	if(mn>rad || mx<-rad) return 0;


#define AXISTEST_X2(a, b, fa, fb)							\
	p0 = a*v0.y - b*v0.z;									\
	p1 = a*v1.y - b*v1.z;									\
	if(p0<p1) {mn=p0; mx=p1;} else {mn=p1; mx=p0;}		\
	rad = fa * boxhalfsize.y + fb * boxhalfsize.z + TRI_EPS;			\
	if(mn>rad || mx<-rad) return 0;


/*======================== Y-tests ========================*/

#define AXISTEST_Y02(a, b, fa, fb)							\
	p0 = -a*v0.x + b*v0.z;									\
	p2 = -a*v2.x + b*v2.z;									\
	if(p0<p2) {mn=p0; mx=p2;} else {mn=p2; mx=p0;}		\
	rad = fa * boxhalfsize.x + fb * boxhalfsize.z + TRI_EPS;			\
	if(mn>rad || mx<-rad) return 0;


#define AXISTEST_Y1(a, b, fa, fb)							\
	p0 = -a*v0.x + b*v0.z;									\
	p1 = -a*v1.x + b*v1.z;									\
	if(p0<p1) {mn=p0; mx=p1;} else {mn=p1; mx=p0;}		\
	rad = fa * boxhalfsize.x + fb * boxhalfsize.z + TRI_EPS;			\
	if(mn>rad || mx<-rad) return 0;


/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)							\
	p1 = a*v1.x - b*v1.y;									\
	p2 = a*v2.x - b*v2.y;									\
	if(p2<p1) {mn=p2; mx=p1;} else {mn=p1; mx=p2;}		\
	rad = fa * boxhalfsize.x + fb * boxhalfsize.y + TRI_EPS;			\
	if(mn>rad || mx<-rad) return 0;


#define AXISTEST_Z0(a, b, fa, fb)							\
	p0 = a*v0.x - b*v0.y;									\
	p1 = a*v1.x - b*v1.y;									\
	if(p0<p1) {mn=p0; mx=p1;} else {mn=p1; mx=p0;}		\
	rad = fa * boxhalfsize.x + fb * boxhalfsize.y + TRI_EPS;			\
	if(mn>rad || mx<-rad) return 0;

#else

/*======================== X-tests ========================*/

#if 0
#define AXISTEST_X01(a, b, fa, fb)							\
	p0 = a*vert0.y - b*vert0.z;								\
	p2 = a*vert2.y - b*vert2.z;								\
	if(p0<p2) {mn=p0; mx=p2;} else {mn=p2; mx=p0;}		\
	rad0 = a * bMin.y - b * bMin.z;			\
	rad2 = a * bMax.y - b * bMax.z;			\
	if(rad0<rad2) {mnR=rad0; mxR=rad2;} else {mnR=rad2; mxR=rad0;}		\
	if(mn>mxR || mx<mnR) return 0;

#else
#define AXISTEST_X01(a, b, fa, fb)							\
	p0 = a*vr0.y - b*vr0.z;								\
	p2 = a*vr2.y - b*vr2.z;								\
	if(p0<p2) {mn=p0; mx=p2;} else {mn=p2; mx=p0;}		\
	rad = a * boxfullsize.y - b * boxfullsize.z;			\
	if(rad<0) {mnR=rad; mxR=0;} else {mnR=0; mxR=rad;}		\
	if(mn>mxR || mx<mnR) return 0;
#endif

#define AXISTEST_X2(a, b, fa, fb)							\
	p0 = a*v0.y - b*v0.z;									\
	p1 = a*v1.y - b*v1.z;									\
	if(p0<p1) {mn=p0; mx=p1;} else {mn=p1; mx=p0;}		\
	rad = fa * boxhalfsize.y + fb * boxhalfsize.z;			\
	if(mn>rad || mx<-rad) return 0;


/*======================== Y-tests ========================*/

#define AXISTEST_Y02(a, b, fa, fb)							\
	p0 = -a*v0.x + b*v0.z;									\
	p2 = -a*v2.x + b*v2.z;									\
	if(p0<p2) {mn=p0; mx=p2;} else {mn=p2; mx=p0;}		\
	rad = fa * boxhalfsize.x + fb * boxhalfsize.z;			\
	if(mn>rad || mx<-rad) return 0;


#define AXISTEST_Y1(a, b, fa, fb)							\
	p0 = -a*v0.x + b*v0.z;									\
	p1 = -a*v1.x + b*v1.z;									\
	if(p0<p1) {mn=p0; mx=p1;} else {mn=p1; mx=p0;}		\
	rad = fa * boxhalfsize.x + fb * boxhalfsize.z;			\
	if(mn>rad || mx<-rad) return 0;


/*======================== Z-tests ========================*/

#define AXISTEST_Z12(a, b, fa, fb)							\
	p1 = a*v1.x - b*v1.y;									\
	p2 = a*v2.x - b*v2.y;									\
	if(p2<p1) {mn=p2; mx=p1;} else {mn=p1; mx=p2;}		\
	rad = fa * boxhalfsize.x + fb * boxhalfsize.y;			\
	if(mn>rad || mx<-rad) return 0;


#define AXISTEST_Z0(a, b, fa, fb)							\
	p0 = a*v0.x - b*v0.y;									\
	p1 = a*v1.x - b*v1.y;									\
	if(p0<p1) {mn=p0; mx=p1;} else {mn=p1; mx=p0;}		\
	rad = fa * boxhalfsize.x + fb * boxhalfsize.y;			\
	if(mn>rad || mx<-rad) return 0;
#endif


__device__ __forceinline__ int triBoxOverlap(const float3& boxcenter, const float3& boxhalfsize, const float3& vert0, const float3& vert1, const float3& vert2, const float3& bMin, const float3& bMax)
{
	/*	use separating axis theorem to test overlap between triangle and box */
	/*	need to test for overlap in these directions: */	
	/*	1) the {x,y,z}-directions (actually, since we use the AABB of the triangle */	
	/*	   we do not even need to test these) */	
	/*	2) normal of the triangle */	
	/*	3) crossproduct(edge from tri, {x,y,z}-directin) */
	/*	   this gives 3x3=9 more tests */
	
	float3 v0, v1, v2, vr0, vr1, vr2;
	float mn, mx, mnR, mxR, p0, p1, p2, rad, rad0, rad1, rad2, fex, fey, fez;
	float3 normal, e0, e1, e2, boxfullsize;

	boxfullsize = bMax - bMin;


	/* This is the fastest branch on Sun */	
	/* move everything so that the boxcenter is in (0,0,0) */	
	
	v0 = vert0 - boxcenter;
	v1 = vert1 - boxcenter;
	v2 = vert2 - boxcenter;

	vr0 = vert0 - bMin;
	vr1 = vert1 - bMin;
	vr2 = vert2 - bMin;
	
	/* compute triangle edges */
	
	e0 = v1 - v0;	/* tri edge 0 */
	e1 = v2 - v1;	/* tri edge 1 */
	e2 = v0 - v2;	/* tri edge 2 */
	
	
	/* Bullet 3: */
	
	/*  test the 9 tests first (this was faster) */
	
	fex = fabsf(e0.x);
	fey = fabsf(e0.y);
	fez = fabsf(e0.z);
	AXISTEST_X01(e0.z, e0.y, fez, fey);
	AXISTEST_Y02(e0.z, e0.x, fez, fex);
	AXISTEST_Z12(e0.y, e0.x, fey, fex);	
	
	
	fex = fabsf(e1.x);
	fey = fabsf(e1.y);
	fez = fabsf(e1.z);
	AXISTEST_X01(e1.z, e1.y, fez, fey);
	AXISTEST_Y02(e1.z, e1.x, fez, fex);
	AXISTEST_Z0(e1.y, e1.x, fey, fex);
	
	
	fex = fabsf(e2.x);
	fey = fabsf(e2.y);
	fez = fabsf(e2.z);
	AXISTEST_X2(e2.z, e2.y, fez, fey);
	AXISTEST_Y1(e2.z, e2.x, fez, fex);
	AXISTEST_Z12(e2.y, e2.x, fey, fex);
	
	
	/* Bullet 1: */
	
	/*  first test overlap in the {x,y,z}-directions */
	/*  find min, max of the triangle each direction, and test for overlap in */
	/*  that direction -- this is equivalent to testing a minimal AABB around */
	/*  the triangle against the AABB */
	
	
#if 0
	/* test in X-direction */
	
	mn = fminf(fminf(v0.x, v1.x), v2.x);
	mx = fmaxf(fmaxf(v0.x, v1.x), v2.x);
	if(mn > boxhalfsize.x || mx < -boxhalfsize.x) return 0;

	/* test in Y-direction */
	
	mn = fminf(fminf(v0.y, v1.y), v2.y);
	mx = fmaxf(fmaxf(v0.y, v1.y), v2.y);
	if(mn > boxhalfsize.y || mx < -boxhalfsize.y) return 0;

	/* test in Z-direction */

	mn = fminf(fminf(v0.z, v1.z), v2.z);
	mx = fmaxf(fmaxf(v0.z, v1.z), v2.z);
	if(mn > boxhalfsize.z || mx < -boxhalfsize.z) return 0;
#else

	/* test in X-direction */

	mn = fminf(fminf(vert0.x, vert1.x), vert2.x);
	mx = fmaxf(fmaxf(vert0.x, vert1.x), vert2.x);
	if(mn > bMax.x + TRI_EPS || mx < bMin.x - TRI_EPS) return 0;

	/* test in Y-direction */
	
	mn = fminf(fminf(vert0.y, vert1.y), vert2.y);
	mx = fmaxf(fmaxf(vert0.y, vert1.y), vert2.y);
	if(mn > bMax.y + TRI_EPS || mx < bMin.y - TRI_EPS) return 0;
	
	/* test in Z-direction */

	mn = fminf(fminf(vert0.z, vert1.z), vert2.z);
	mx = fmaxf(fmaxf(vert0.z, vert1.z), vert2.z);
	if(mn > bMax.z + TRI_EPS || mx < bMin.z - TRI_EPS) return 0;
#endif
	
	
	/* Bullet 2: */
	
	/*  test if the box intersects the plane of the triangle */
	/*  compute plane equation of triangle: normal*x+d=0 */
	
	normal = cross(e0, e1);
#if 0
	if(!planeBoxOverlap(normal, v0, boxhalfsize)) return 0;
#else
	if(!planeBoxOverlap(normal, vert0, bMin, bMax)) return 0;
#endif
	
	
	return 1;	/* box and triangle overlaps */
}
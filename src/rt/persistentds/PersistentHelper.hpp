#pragma once

#include "persistentds/CudaTracerDefines.h"

namespace FW
{

//------------------------------------------------------------------------

// Alignment to multiply of S
template<typename T, int S>
inline T align(T a)
{
	 return (a+S-1) & -S;
}

//------------------------------------------------------------------------

inline int warpSubtasks(int threads)
{
	//return (threads + WARP_SIZE - 1) / WARP_SIZE;
	return max((threads + WARP_SIZE - 1) / WARP_SIZE, 1); // Do not create empty tasks - at least on warp gets to clean this task
}

//------------------------------------------------------------------------

inline int floatToOrderedInt(float floatVal)
{
	int intVal = *((int*)&floatVal);
	return (intVal >= 0) ? intVal : intVal ^ 0x7FFFFFFF;
}

//------------------------------------------------------------------------

}
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

/*
    Test of various dynamic allocation methods.
*/

#include "Allocators.hpp"
#include "warp_common.cu"

//------------------------------------------------------------------------
// CudaMalloc
//------------------------------------------------------------------------

__device__ __forceinline__ void* mallocCudaMalloc(uint allocSize)
{
	uint allocMem = align<uint, ALIGN>(allocSize);
	void* alloc = malloc(allocMem);
	return alloc;
}

//------------------------------------------------------------------------

__device__ __forceinline__ void freeCudaMalloc(void* ptr)
{
	free(ptr);
}

//------------------------------------------------------------------------
// AtomicMalloc
//------------------------------------------------------------------------

__device__ __forceinline__ void* mallocAtomicMalloc(uint allocSize)
{
	uint allocMem = align<uint, ALIGN>(allocSize);
	uint offset = atomicAdd(&g_heapOffset, allocMem);
#ifdef CHECK_OUT_OF_MEMORY
	if(offset + allocMem > c_alloc.heapSize)
		return NULL;
#endif
	return g_heapBase + offset;
}

//------------------------------------------------------------------------
// AtomicMallocCircular
//------------------------------------------------------------------------

__device__ __forceinline__ void* mallocAtomicMallocCircular(uint allocSize)
{
	// Cyclic allocator, needs to know heap size
	//int warpIdx = blockDim.y*blockIdx.x + threadIdx.y; // Warp ID
	uint allocMem = align<uint, ALIGN>(allocSize);
	uint offset = atomicAdd(&g_heapOffset, allocMem);
	uint newOffset = offset + allocMem;
	while(newOffset > c_alloc.heapSize) // Try allocating from beginning
	{
		newOffset = atomicCAS(&g_heapOffset, newOffset, allocMem); // Wrap allocation using CAS
		//printf("Wrap allocation try %d!\n", warpIdx);
		if(newOffset == offset + allocMem) // This thread succeeded in wrapping the allocator
		{
			//printf("Wrap allocation %d!\n", warpIdx);
			return g_heapBase; // The allocation is from the beginning to allocSize
		}
		else if(newOffset + allocMem <= c_alloc.heapSize) // Retry add allocation
		{
			//printf("Add allocation retry %d!\n", warpIdx);
			offset = atomicAdd(&g_heapOffset, allocMem);
			newOffset = offset + allocMem;
		}
		else
		{
			//printf("Wrap allocation retry %d!\n", warpIdx);
			offset = newOffset - allocMem; // Retry wrap allocation
		}
	}
	return g_heapBase + offset;
}

//------------------------------------------------------------------------
// CircularMalloc
//------------------------------------------------------------------------

__device__ __forceinline__ void* mallocCircularMalloc(uint allocSize)
{
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	// Lock the heap
	while(atomicCAS(&g_heapLock, AllocatorLockType_Free, AllocatorLockType_Set) != 0)
		;
#endif

	uint offset = getMemory(&g_heapOffset);
	uint allocMem = align<uint, ALIGN>(allocSize+CIRCULAR_MALLOC_HEADER_SIZE);

	// Find an empty chunk that is large enough
	uint lock, next, size;
#ifdef CIRCULAR_MALLOC_CHECK_DEADLOCK
	int count = 0;
#endif

	while(true)
	{
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		lock = getMemory((uint*)(g_heapBase+offset));
#else
#ifdef CIRCULAR_MALLOC_PRELOCK
		// Each item is atomically locked
		lock = atomicCAS((uint*)(g_heapBase+offset), AllocatorLockType_Free, AllocatorLockType_Set);
#endif
#endif
		next = getMemory((uint*)(g_heapBase+offset+CIRCULAR_MALLOC_NEXT_OFS));
		size = next - offset; // Chunk size (including the header)

#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		if(lock == AllocatorLockType_Free && allocMem <= size)
			break;
#else
#ifdef CIRCULAR_MALLOC_PRELOCK
		// Each item is atomically locked
		if(lock == AllocatorLockType_Free)
		{
			if(allocMem <= size) // If the payload can be fitted into the current item
				break;
			
			// Set the item free if it is too small
			setMemory((uint*)(g_heapBase+offset), lock); // Set free
			//setMemory(&g_heapOffset, offset); // Does this make a difference?
		}
#else
		// Only viable chunks are atomically locked - BUG: The chunk may be destroyed before we move to next chunk (may lead to incorrect next)
		if(allocMem <= size)
		{
			if(atomicCAS((uint*)(g_heapBase+offset), AllocatorLockType_Free, AllocatorLockType_Set) == AllocatorLockType_Free)
			{
				if(allocMem <= size)
					break;

				setMemory((uint*)(g_heapBase+offset), AllocatorLockType_Free); // Set free
				setMemory(&g_heapOffset, offset);
			}
		}

#endif
#endif

#ifdef CIRCULAR_MALLOC_CHECK_DEADLOCK
		if(count > CIRCULAR_MALLOC_WAIT_COUNT)
			return NULL;
		count++;
#endif

		offset = next;
		/*int newOffset = atomicCAS(&g_heapOffset, offset, next); // Move the offset for others
		//if(newOffset != offset && newOffset != next)
		//	printf("Jump from %d past %d to %d\n", offset, next, newOffset);
		if(newOffset == offset)
			offset = next;
		else
			offset = newOffset;*/
	}

#ifdef COUNT_STEPS_ALLOC
	maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], count);
	sumSteps[threadIdx.y] += count;
	numSteps[threadIdx.y]++;
#endif

	//if(lock != 0 || offset < 0 || offset > c_alloc.heapSize || size < 0 || size > c_alloc.heapSize || next < 0 || next > c_alloc.heapSize || size < allocSize+2*sizeof(int))
	//	printf("HeapOffset %d nodeIdx %d offset %d next %d size %d, allocSize %d\n", g_heapOffset, s_task[threadIdx.y].nodeIdx, offset, next, size, allocSize);

	//printf("HeapOffset %d nodeIdx %d offset %d next %d size %d, allocSize %d\n", g_heapOffset, s_task[threadIdx.y].nodeIdx, offset, next, size, allocSize);

	// Insert new header, splitting the chunk

	uint newNext = next;
	// Create new header if there is room for it and the fragmentation would be too high
	if(allocMem+CIRCULAR_MALLOC_HEADER_SIZE <= size && allocMem*c_alloc.maxFrag <= size)
	{
		// Update next header
		newNext = offset + allocMem;
#if 0
		setMemory((uint*)(g_heapBase+newNext), AllocatorLockType_Free); // Set free

#ifdef CIRCULAR_MALLOC_DOUBLY_LINKED
		setMemory((uint*)(g_heapBase+newNext+CIRCULAR_MALLOC_PREV_OFS), offset);
#endif
		setMemory((uint*)(g_heapBase+newNext+CIRCULAR_MALLOC_NEXT_OFS), next);

#else
#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
		setMemory((uint2*)(g_heapBase+newNext), make_uint2(AllocatorLockType_Free, next));
#else
		setMemory((uint4*)(g_heapBase+newNext), make_uint4(AllocatorLockType_Free, offset, next, 0));
#endif
#endif

#ifndef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		__threadfence(); // Make sure next is visible before linking the task
#endif

		setMemory((uint*)(g_heapBase+offset+CIRCULAR_MALLOC_NEXT_OFS), newNext);
#ifdef CIRCULAR_MALLOC_DOUBLY_LINKED
		setMemory((uint*)(g_heapBase+next+CIRCULAR_MALLOC_PREV_OFS), newNext);
#endif
	}

	//atomicCAS(&g_heapOffset, offset, newNext); // Move the offset for others
	//atomicCAS(&g_heapOffset, offset, next); // Move the offset for others

//#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	setMemory(&g_heapOffset, newNext);
/*#else
	if(g_heapOffset < offset + allocSize + 2*sizeof(int)) // If we had not wrap around
		atomicMax(&g_heapOffset, newNext);
	else
		setMemory(&g_heapOffset, newNext);
#endif*/

	//printf("heapOffset %d\n", g_heapOffset);

#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	// Unlock the heap
	setMemory(&g_heapLock, AllocatorLockType_Free);
#endif

	return g_heapBase + offset + CIRCULAR_MALLOC_HEADER_SIZE; // The allocated memory after the header
}

//------------------------------------------------------------------------

__device__ __forceinline__ void freeCircularMalloc(void* ptr)
{
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	// Lock the heap
	while(atomicCAS(&g_heapLock, AllocatorLockType_Free, AllocatorLockType_Set) != AllocatorLockType_Free)
		;
#endif

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
#ifdef CIRCULAR_MALLOC_CONNECT_CHUNKS
	// Find out whether the next chunk is free
	uint next = getMemory((uint*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE+CIRCULAR_MALLOC_NEXT_OFS));

	// Connect the two chunks
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	if(getMemory((uint*)(g_heapBase+next)) == AllocatorLockType_Free)
#else
	if(atomicCAS((uint*)(g_heapBase+next), AllocatorLockType_Free, AllocatorLockType_Set) == AllocatorLockType_Free)
#endif
	{		
		next = getMemory((uint*)(g_heapBase+next+CIRCULAR_MALLOC_NEXT_OFS));
		setMemory(&g_heapOffset, next);
#if 0
		setMemory((uint*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE+CIRCULAR_MALLOC_NEXT_OFS), next);
		
		// This fence can be omited (but may result in temporarily smaller free chunk)
#ifndef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		__threadfence(); // Make sure the link is set before unlocking the task
#endif

#else
		setMemory((uint2*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE), make_uint2(AllocatorLockType_Free, next));
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		// Unlock the heap
		setMemory(&g_heapLock, AllocatorLockType_Free);
#endif
		return;
#endif
	}
#endif

	// Set the chunk free
	setMemory((uint*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE), AllocatorLockType_Free);

#else // CIRCULAR_MALLOC_DOUBLY_LINKED
	uint head = ((char*)ptr-g_heapBase)-CIRCULAR_MALLOC_HEADER_SIZE;

#ifdef CIRCULAR_MALLOC_CONNECT_CHUNKS
	uint next = getMemory((uint*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE+CIRCULAR_MALLOC_NEXT_OFS));
	//int nextVal;
	uint prev = getMemory((uint*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE+CIRCULAR_MALLOC_PREV_OFS));
	bool nextMerged = false;
	bool prevMerged = false;

	// Connect the next chunks
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	if(getMemory((uint*)(g_heapBase+next)) == AllocatorLockType_Free)
#else
	if(atomicCAS((uint*)(g_heapBase+next), AllocatorLockType_Free, AllocatorLockType_Set) == AllocatorLockType_Free)
#endif
	{
		next = getMemory((uint*)(g_heapBase+next+CIRCULAR_MALLOC_NEXT_OFS));
		setMemory(&g_heapOffset, next);
		nextMerged = true;
	}

	// Connect the prev chunk
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	if(getMemory((uint*)(g_heapBase+prev)) == AllocatorLockType_Free)
#else
	if(atomicCAS((uint*)(g_heapBase+prev), AllocatorLockType_Free, AllocatorLockType_Set) == AllocatorLockType_Free)
#endif
	{
		head = prev;
		//prev = getMemory((uint*)(g_heapBase+prev+sizeof(int)));
		prevMerged = true;
	}
	
	
	if(nextMerged || prevMerged)
	{
		setMemory((uint*)(g_heapBase+head+CIRCULAR_MALLOC_NEXT_OFS), next); // This segments next
		setMemory((uint*)(g_heapBase+next+CIRCULAR_MALLOC_PREV_OFS), head); // Next segments prev

		// This fence can be omited (but may result in temporarily smaller free chunk)
#ifndef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		__threadfence(); // Make sure the link is set before unlocking the task
#endif
	}
#endif

	// Set the chunk free
	setMemory((uint*)(g_heapBase+head), AllocatorLockType_Free);
#endif

#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	// Unlock the heap
	setMemory(&g_heapLock, AllocatorLockType_Free);
#endif
}

//------------------------------------------------------------------------
// CircularMallocFused
//------------------------------------------------------------------------

const uint flagMask = 0x80000000u;

__device__ __forceinline__ void* mallocCircularMallocFused(uint allocSize)
{
	uint offset = getMemory(&g_heapOffset);
	uint headerSize = sizeof(uint);
	uint allocMem = align<uint, ALIGN>(allocSize+headerSize);

	// Find an empty chunk that is large enough
	uint header, next, size;
#ifdef CIRCULAR_MALLOC_CHECK_DEADLOCK
	int count = 0;
#endif

	while(true)
	{
		// Each item is atomically locked
		header = atomicOr((uint*)(g_heapBase+offset), flagMask);

		next = header & (~flagMask);
		size = next - offset; // Chunk size (including the header)

		// Each item is atomically locked
		if((header & flagMask) == 0)
		{
			if(allocMem <= size) // If the payload can be fitted into the current item
				break;
			
			// Set the item free if it is too small
			setMemory((uint*)(g_heapBase+offset), next); // Set free
		}

#ifdef CIRCULAR_MALLOC_CHECK_DEADLOCK
		if(count > CIRCULAR_MALLOC_WAIT_COUNT)
			return NULL;
		count++;
#endif

		offset = next;
	}

	// Insert new header, splitting the chunk

	uint newNext = next;
	// Create new header if there is room for it and the fragmentation would be too high
	if(allocMem+headerSize <= size && allocMem*c_alloc.maxFrag <= size)
	{
		// Update next header
		newNext = offset + allocMem;
		setMemory((uint*)(g_heapBase+newNext), next);  // Set free

		__threadfence(); // Make sure next is visible before linking the task

		setMemory((uint*)(g_heapBase+offset), newNext | flagMask);
	}

	setMemory(&g_heapOffset, newNext);

	return g_heapBase + offset + headerSize; // The allocated memory after the header
}

//------------------------------------------------------------------------

__device__ __forceinline__ void freeCircularMallocFused(void* ptr)
{
	// This chunk header
	uint* offset = (uint*)((char*)ptr-sizeof(uint));
	// Find out whether the next chunk is free
	uint next = getMemory(offset) & (~flagMask);
	uint newNext = next;

#ifdef CIRCULAR_MALLOC_CONNECT_CHUNKS
	// Connect the two chunks
	uint header = atomicOr((uint*)(g_heapBase+next), flagMask);
	if((header & flagMask) == 0)
	{
		newNext = header; // New next is the next stored in header
		setMemory(&g_heapOffset, newNext);
	}
#endif

	// Set the chunk free
	setMemory(offset, newNext);
}

//------------------------------------------------------------------------
// CircularMultiMalloc
//------------------------------------------------------------------------

__device__ __forceinline__ void* mallocCircularMultiMalloc(uint allocSize)
{
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	// Lock the heap
	while(atomicCAS(&g_heapLock, AllocatorLockType_Free, AllocatorLockType_Set) != 0)
		;
#endif

	uint smid = GPUTools::smid();
	uint offset = getMemory(&g_heapMultiOffset[smid]);
	uint allocMem = align<uint, ALIGN>(allocSize+CIRCULAR_MALLOC_HEADER_SIZE);

	// Find an empty chunk that is large enough
	uint lock, next, size;
#ifdef CIRCULAR_MALLOC_CHECK_DEADLOCK
	int count = 0;
#endif

	while(true)
	{
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		lock = getMemory((uint*)(g_heapBase+offset));
#else
#ifdef CIRCULAR_MALLOC_PRELOCK
		// Each item is atomically locked
		lock = atomicCAS((uint*)(g_heapBase+offset), AllocatorLockType_Free, AllocatorLockType_Set);
#endif
#endif
		next = getMemory((uint*)(g_heapBase+offset+CIRCULAR_MALLOC_NEXT_OFS));
		size = next - offset; // Chunk size (including the header)

#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		if(lock == AllocatorLockType_Free && allocMem <= size)
			break;
#else
#ifdef CIRCULAR_MALLOC_PRELOCK
		// Each item is atomically locked
		if(lock == AllocatorLockType_Free)
		{
			if(allocMem <= size) // If the payload can be fitted into the current item
				break;
			
			// Set the item free if it is too small
			setMemory((uint*)(g_heapBase+offset), lock); // Set free
			//setMemory(&g_heapMultiOffset[smid], offset); // Does this make a difference?
		}
#else
		// Only viable chunks are atomically locked - BUG: The chunk may be destroyed before we move to next chunk (may lead to incorrect next)
		if(allocMem <= size)
		{
			if(atomicCAS((uint*)(g_heapBase+offset), AllocatorLockType_Free, AllocatorLockType_Set) == AllocatorLockType_Free)
			{
				if(allocMem <= size)
					break;

				setMemory((uint*)(g_heapBase+offset), AllocatorLockType_Free); // Set free
				setMemory(&g_heapMultiOffset[smid], offset);
			}
		}

#endif
#endif

#ifdef CIRCULAR_MALLOC_CHECK_DEADLOCK
		if(count > CIRCULAR_MALLOC_WAIT_COUNT)
			return NULL;
		count++;
#endif

		offset = next;
		/*int newOffset = atomicCAS(&g_heapMultiOffset[smid], offset, next); // Move the offset for others
		//if(newOffset != offset && newOffset != next)
		//	printf("Jump from %d past %d to %d\n", offset, next, newOffset);
		if(newOffset == offset)
			offset = next;
		else
			offset = newOffset;*/
	}

#ifdef COUNT_STEPS_ALLOC
	maxSteps[threadIdx.y] = max(maxSteps[threadIdx.y], count);
	sumSteps[threadIdx.y] += count;
	numSteps[threadIdx.y]++;
#endif

	//if(lock != 0 || offset < 0 || offset > c_alloc.heapSize || size < 0 || size > c_alloc.heapSize || next < 0 || next > c_alloc.heapSize || size < allocSize+2*sizeof(int))
	//	printf("HeapOffset %d nodeIdx %d offset %d next %d size %d, allocSize %d\n", g_heapMultiOffset[smid], s_task[threadIdx.y].nodeIdx, offset, next, size, allocSize);

	//printf("HeapOffset %d nodeIdx %d offset %d next %d size %d, allocSize %d\n", g_heapMultiOffset[smid], s_task[threadIdx.y].nodeIdx, offset, next, size, allocSize);

	// Insert new header, splitting the chunk

	uint newNext = next;
	// Create new header if there is room for it and the fragmentation would be too high
	if(allocMem+CIRCULAR_MALLOC_HEADER_SIZE <= size && allocMem*c_alloc.maxFrag <= size)
	{
		// Update next header
		newNext = offset + allocMem;
#if 0
		setMemory((uint*)(g_heapBase+newNext), AllocatorLockType_Free); // Set free

#ifdef CIRCULAR_MALLOC_DOUBLY_LINKED
		setMemory((uint*)(g_heapBase+newNext+CIRCULAR_MALLOC_PREV_OFS), offset);
#endif
		setMemory((uint*)(g_heapBase+newNext+CIRCULAR_MALLOC_NEXT_OFS), next);

#else
#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
		setMemory((uint2*)(g_heapBase+newNext), make_uint2(AllocatorLockType_Free, next));
#else
		setMemory((uint4*)(g_heapBase+newNext), make_uint4(AllocatorLockType_Free, offset, next, 0));
#endif
#endif

#ifndef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		__threadfence(); // Make sure next is visible before linking the task
#endif

		setMemory((uint*)(g_heapBase+offset+CIRCULAR_MALLOC_NEXT_OFS), newNext);
#ifdef CIRCULAR_MALLOC_DOUBLY_LINKED
		setMemory((uint*)(g_heapBase+next+CIRCULAR_MALLOC_PREV_OFS), newNext);
#endif
	}

	//atomicCAS(&g_heapMultiOffset[smid], offset, newNext); // Move the offset for others
	//atomicCAS(&g_heapMultiOffset[smid], offset, next); // Move the offset for others

//#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	setMemory(&g_heapMultiOffset[smid], newNext);
/*#else
	if(g_heapMultiOffset[smid] < offset + allocSize + 2*sizeof(int)) // If we had not wrap around
		atomicMax(&g_heapMultiOffset[smid], newNext);
	else
		setMemory(&g_heapMultiOffset[smid], newNext);
#endif*/

	//printf("heapOffset %d\n", g_heapMultiOffset[smid]);

#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	// Unlock the heap
	setMemory(&g_heapLock, AllocatorLockType_Free);
#endif

	return g_heapBase + offset + CIRCULAR_MALLOC_HEADER_SIZE; // The allocated memory after the header
}

//------------------------------------------------------------------------

__device__ __forceinline__ void freeCircularMultiMalloc(void* ptr)
{
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	// Lock the heap
	while(atomicCAS(&g_heapLock, AllocatorLockType_Free, AllocatorLockType_Set) != AllocatorLockType_Free)
		;
#endif

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
#ifdef CIRCULAR_MALLOC_CONNECT_CHUNKS
	// Find out whether the next chunk is free
	uint next = getMemory((uint*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE+CIRCULAR_MALLOC_NEXT_OFS));

	// Connect the two chunks
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	if(getMemory((uint*)(g_heapBase+next)) == AllocatorLockType_Free)
#else
	if(atomicCAS((uint*)(g_heapBase+next), AllocatorLockType_Free, AllocatorLockType_Set) == AllocatorLockType_Free)
#endif
	{		
		uint newNext = getMemory((uint*)(g_heapBase+next+CIRCULAR_MALLOC_NEXT_OFS));
		uint smid = GPUTools::smid();
		for(int i = 0; i < g_numSM; i++)
		{
			if(g_heapMultiOffset[i] == next)
				setMemory(&g_heapMultiOffset[i], newNext);
		}
		next = newNext;
		setMemory(&g_heapMultiOffset[smid], next);
#if 0
		setMemory((uint*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE+CIRCULAR_MALLOC_NEXT_OFS), next);
		
		// This fence can be omited (but may result in temporarily smaller free chunk)
#ifndef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		__threadfence(); // Make sure the link is set before unlocking the task
#endif

#else
		setMemory((uint2*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE), make_uint2(AllocatorLockType_Free, next));
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		// Unlock the heap
		setMemory(&g_heapLock, AllocatorLockType_Free);
#endif
		return;
#endif
	}
#endif

	// Set the chunk free
	setMemory((uint*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE), AllocatorLockType_Free);

#else // CIRCULAR_MALLOC_DOUBLY_LINKED
	uint head = ((char*)ptr-g_heapBase)-CIRCULAR_MALLOC_HEADER_SIZE;

#ifdef CIRCULAR_MALLOC_CONNECT_CHUNKS
	uint next = getMemory((uint*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE+CIRCULAR_MALLOC_NEXT_OFS));
	//int nextVal;
	uint prev = getMemory((uint*)((char*)ptr-CIRCULAR_MALLOC_HEADER_SIZE+CIRCULAR_MALLOC_PREV_OFS));
	bool nextMerged = false;
	bool prevMerged = false;

	// Connect the next chunks
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	if(getMemory((uint*)(g_heapBase+next)) == AllocatorLockType_Free)
#else
	if(atomicCAS((uint*)(g_heapBase+next), AllocatorLockType_Free, AllocatorLockType_Set) == AllocatorLockType_Free)
#endif
	{
		uint newNext = getMemory((uint*)(g_heapBase+next+CIRCULAR_MALLOC_NEXT_OFS));
		uint smid = GPUTools::smid();
		for(int i = 0; i < g_numSM; i++)
		{
			if(g_heapMultiOffset[i] == next)
				setMemory(&g_heapMultiOffset[i], newNext);
		}
		next = newNext;
		setMemory(&g_heapMultiOffset[smid], next);
		nextMerged = true;
	}

	// Connect the prev chunk
#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	if(getMemory((uint*)(g_heapBase+prev)) == AllocatorLockType_Free)
#else
	if(atomicCAS((uint*)(g_heapBase+prev), AllocatorLockType_Free, AllocatorLockType_Set) == AllocatorLockType_Free)
#endif
	{
		head = prev;
		//prev = getMemory((uint*)(g_heapBase+prev+sizeof(int)));
		prevMerged = true;
	}
	
	
	if(nextMerged || prevMerged)
	{
		setMemory((uint*)(g_heapBase+head+CIRCULAR_MALLOC_NEXT_OFS), next); // This segments next
		setMemory((uint*)(g_heapBase+next+CIRCULAR_MALLOC_PREV_OFS), head); // Next segments prev

		// This fence can be omited (but may result in temporarily smaller free chunk)
#ifndef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
		__threadfence(); // Make sure the link is set before unlocking the task
#endif
	}
#endif

	// Set the chunk free
	setMemory((uint*)(g_heapBase+head), AllocatorLockType_Free);
#endif

#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	// Unlock the heap
	setMemory(&g_heapLock, AllocatorLockType_Free);
#endif
}

//------------------------------------------------------------------------
// CircularMultiMallocFused
//------------------------------------------------------------------------

__device__ __forceinline__ void* mallocCircularMultiMallocFused(uint allocSize)
{
	uint smid = GPUTools::smid();
	uint offset = getMemory(&g_heapMultiOffset[smid]);
	uint headerSize = sizeof(uint);
	uint allocMem = align<uint, ALIGN>(allocSize+headerSize);

	// Find an empty chunk that is large enough
	uint header, next, size;
#ifdef CIRCULAR_MALLOC_CHECK_DEADLOCK
	int count = 0;
#endif

	while(true)
	{
		// Each item is atomically locked
		header = atomicOr((uint*)(g_heapBase+offset), flagMask);

		next = header & (~flagMask);
		size = next - offset; // Chunk size (including the header)

		// Each item is atomically locked
		if((header & flagMask) == 0)
		{
			if(allocMem <= size) // If the payload can be fitted into the current item
				break;
			
			// Set the item free if it is too small
			setMemory((uint*)(g_heapBase+offset), next); // Set free
		}

#ifdef CIRCULAR_MALLOC_CHECK_DEADLOCK
		if(count > CIRCULAR_MALLOC_WAIT_COUNT)
			return NULL;
		count++;
#endif

		offset = next;
	}

	// Insert new header, splitting the chunk

	uint newNext = next;
	// Create new header if there is room for it and the fragmentation would be too high
	if(allocMem+headerSize <= size && allocMem*c_alloc.maxFrag <= size)
	{
		// Update next header
		newNext = offset + allocMem;
		setMemory((uint*)(g_heapBase+newNext), next);  // Set free

		__threadfence(); // Make sure next is visible before linking the task

		setMemory((uint*)(g_heapBase+offset), newNext | flagMask);
	}

	setMemory(&g_heapMultiOffset[smid], newNext);

	return g_heapBase + offset + headerSize; // The allocated memory after the header
}

//------------------------------------------------------------------------

__device__ __forceinline__ void freeCircularMultiMallocFused(void* ptr)
{
	// This chunk header
	uint* offset = (uint*)((char*)ptr-sizeof(uint));
	// Find out whether the next chunk is free
	uint next = getMemory(offset) & (~flagMask);
	uint newNext = next;

#ifdef CIRCULAR_MALLOC_CONNECT_CHUNKS
	// Connect the two chunks
	uint header = atomicOr((uint*)(g_heapBase+next), flagMask);
	if((header & flagMask) == 0)
	{
		newNext = header; // New next is the next stored in header
		uint smid = GPUTools::smid();
		for(int i = 0; i < g_numSM; i++)
		{
			if(g_heapMultiOffset[i] == next)
				setMemory(&g_heapMultiOffset[i], newNext);
		}
		setMemory(&g_heapMultiOffset[smid], newNext);
	}
#endif

	// Set the chunk free
	setMemory(offset, newNext);
}

//------------------------------------------------------------------------
// ScatterAlloc
//------------------------------------------------------------------------

__device__ __forceinline__ void* mallocScatterAlloc(uint allocSize)
{
	return theHeap.alloc(allocSize);
}

//------------------------------------------------------------------------

__device__ __forceinline__ void freeScatterAlloc(void* ptr)
{
	theHeap.dealloc(ptr);
}

//------------------------------------------------------------------------
// FDGMalloc
//------------------------------------------------------------------------

__device__ __forceinline__ void* mallocFDGMalloc(FDG::Warp* warp, uint allocSize)
{
	return warp->alloc(allocSize);
}

//------------------------------------------------------------------------

__device__ __forceinline__ void freeFDGMalloc(FDG::Warp* warp)
{
	warp->end();
}

//------------------------------------------------------------------------
// CircularMalloc - Allocator using a circular linked list of chunks
//------------------------------------------------------------------------

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
__device__ __forceinline__ void setHeader(uint tid, uint lastTid, uint ofs, uint nextOfs, uint lastMark)
{
	uint2 mark = make_uint2(AllocatorLockType_Free, nextOfs);
#if ALIGN >= 8
	*(uint2*)(g_heapBase+ofs) = mark; // Set the next header
#else
	uint* markPtr = (uint*)(g_heapBase+ofs);
	*(markPtr+0) = mark.x; // Set the next header
	*(markPtr+1) = mark.y;
#endif

	if(tid == lastTid)
	{
		uint2 tail = make_uint2(AllocatorLockType_Set, 0); // Locked, next at the start of heap
#if ALIGN >= 8
		*(uint2*)(g_heapBase+lastMark) = tail; // Set the last header
#else
		uint* tailPtr = (uint*)(g_heapBase+lastMark);
		*(tailPtr+0) = tail.x; // Set the next header
		*(tailPtr+1) = tail.y;
#endif
	}
}

#else
__device__ __forceinline__ void setHeader(uint tid, uint lastTid, uint ofs, uint prevOfs, uint nextOfs, uint lastMark)
{
	uint4 mark = make_uint4(AllocatorLockType_Free, prevOfs, nextOfs, 0);
#if ALIGN >= 16
	*(uint4*)(g_heapBase+ofs) = mark; // Set the next header
#else
	uint* markPtr = (uint*)(g_heapBase+ofs);
	*(markPtr+0) = mark.x; // Set the next header
	*(markPtr+1) = mark.y;
	*(markPtr+2) = mark.z;
#endif

	if(tid == lastTid)
	{
		uint4 tail = make_uint4(AllocatorLockType_Set, ofs, 0, 0); // Locked, next at the start of heap
#if ALIGN >= 16
		*(uint4*)(g_heapBase+lastMark) = tail; // Set the last header
#else
		uint* tailPtr = (uint*)(g_heapBase+lastMark);
		*(tailPtr+0) = tail.x; // Set the next header
		*(tailPtr+1) = tail.y;
		*(tailPtr+2) = tail.z;
#endif
	}
}
#endif

//------------------------------------------------------------------------

extern "C" __global__ void CircularMallocPrepare1(uint numChunks)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint lastMark = c_alloc.heapSize-CIRCULAR_MALLOC_HEADER_SIZE;

	uint chunkSize = align<uint, ALIGN>((CIRCULAR_MALLOC_HEADER_SIZE + c_alloc.payload)*c_alloc.chunkRatio);
#ifdef CIRCULAR_MALLOC_DOUBLY_LINKED
	uint prevOfs = (tid == 0) ? lastMark : (tid-1)*chunkSize; // Previous at the multiple of chunk
#endif
	uint ofs     = tid*chunkSize;
	uint nextOfs = (tid == lastTid) ? lastMark : (tid+1)*chunkSize; // Next at the multiple of chunk

	//printf("Tid %d Ofs %u Chunk size %u prevOfs %u nextOfs %u\n", tid, ofs, chunkSize, prevOfs, nextOfs);

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
	setHeader(tid, lastTid, ofs, nextOfs, lastMark);
#else
	setHeader(tid, lastTid, ofs, prevOfs, nextOfs, lastMark);
#endif
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMallocPrepare2(uint numChunks)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint lastMark = c_alloc.heapSize-CIRCULAR_MALLOC_HEADER_SIZE;

	uint minChunkSize = align<uint, ALIGN>((CIRCULAR_MALLOC_HEADER_SIZE + c_alloc.payload)*c_alloc.chunkRatio);
	uint chunkSize = (1 << tid)*minChunkSize;
	//uint prevOfs = (tid == 0) ? lastMark : (1 << (tid-1))*minChunkSize - minChunkSize; // Previous at geometric sequence previous minus the first term
	uint ofs     = chunkSize - minChunkSize; // Current at geometric sequence previous minus the first term
#ifdef CIRCULAR_MALLOC_DOUBLY_LINKED
	uint prevOfs = (tid == 0) ? lastMark : ofs - (chunkSize>>1);
#endif
	//uint nextOfs = (tid == lastTid) ? lastMark : (1 << (tid+1))*minChunkSize - minChunkSize; // Next at geometric sequence next minus the first term
	uint nextOfs = (tid == lastTid) ? lastMark : ofs + chunkSize;
	
	//printf("Tid %d Ofs %u Chunk size %u prevOfs %u nextOfs %u\n", tid, ofs, chunkSize, prevOfs, nextOfs);

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
	setHeader(tid, lastTid, ofs, nextOfs, lastMark);
#else
	setHeader(tid, lastTid, ofs, prevOfs, nextOfs, lastMark);
#endif
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMallocPrepare3(uint numChunks, uint rootChunk)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint lastMark = c_alloc.heapSize-CIRCULAR_MALLOC_HEADER_SIZE;

	uint minChunkSize = align<uint, ALIGN>((CIRCULAR_MALLOC_HEADER_SIZE + c_alloc.payload)*c_alloc.chunkRatio);
	// POSSIBLE BUG: Rounding errors may cause misaligned memory accesses - solved by double at the cost of significant slowdown
	uint depth = log2f((float)(tid+1)); // Depth of the thread's node.
	uint chunkSize = rootChunk >> depth; // Chunks size corresponding to the level
	uint lvlTid = tid - ((1 << depth) - 1);
	uint ofs     = depth*rootChunk + lvlTid*chunkSize; // Current at geometric sequence previous minus the first term
#ifdef CIRCULAR_MALLOC_DOUBLY_LINKED
	uint chunkPrev = (lvlTid == 0) ? (chunkSize<<1) : chunkSize; // The previous chunk is either the same size of twice the size
	uint prevOfs = (tid == 0) ? lastMark : ofs - chunkPrev;
#endif
	uint nextOfs = (tid == lastTid) ? lastMark : ofs + chunkSize;
	
	//printf("Tid %d Ofs %u Chunk size %u prevOfs %u nextOfs %u\n", tid, ofs, chunkSize, prevOfs, nextOfs);

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
	setHeader(tid, lastTid, ofs, nextOfs, lastMark);
#else
	setHeader(tid, lastTid, ofs, prevOfs, nextOfs, lastMark);
#endif
}

//------------------------------------------------------------------------
// CircularMallocFused - Allocator using a circular linked list of chunks with fused header and next pointer
//------------------------------------------------------------------------

__device__ __forceinline__ void setHeaderFused(uint tid, uint lastTid, uint ofs, uint nextOfs, uint lastMark)
{
	uint* markPtr = (uint*)(g_heapBase+ofs);
	*markPtr = nextOfs; // Set the next header

	if(tid == lastTid)
	{
		uint* tailPtr = (uint*)(g_heapBase+lastMark);
		*tailPtr = flagMask; // Set the next header
	}
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMallocFusedPrepare1(uint numChunks)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint headerSize = sizeof(uint);
	uint lastMark   = c_alloc.heapSize-headerSize;

	uint chunkSize = align<uint, ALIGN>((headerSize + c_alloc.payload)*c_alloc.chunkRatio);
	uint ofs       = tid*chunkSize;
	uint nextOfs   = (tid == lastTid) ? lastMark : (tid+1)*chunkSize; // Next at the multiple of chunk

	//printf("Tid %d Ofs %u Chunk size %u nextOfs %u\n", tid, ofs, chunkSize, nextOfs);

	setHeaderFused(tid, lastTid, ofs, nextOfs, lastMark);
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMallocFusedPrepare2(uint numChunks)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint headerSize = sizeof(uint);
	uint lastMark   = c_alloc.heapSize-headerSize;

	uint minChunkSize = align<uint, ALIGN>((headerSize + c_alloc.payload)*c_alloc.chunkRatio);
	uint chunkSize    = (1 << tid)*minChunkSize;
	uint ofs          = chunkSize - minChunkSize; // Current at geometric sequence previous minus the first term
	//uint nextOfs     = (tid == lastTid) ? lastMark : (1 << (tid+1))*minChunkSize - minChunkSize; // Next at geometric sequence next minus the first term
	uint nextOfs      = (tid == lastTid) ? lastMark : ofs + chunkSize;
	
	//printf("Tid %d Ofs %u Chunk size %u nextOfs %u\n", tid, ofs, chunkSize, nextOfs);

	setHeaderFused(tid, lastTid, ofs, nextOfs, lastMark);
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMallocFusedPrepare3(uint numChunks, uint rootChunk)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint headerSize = sizeof(uint);
	uint lastMark   = c_alloc.heapSize-headerSize;

	uint minChunkSize = align<uint, ALIGN>((headerSize + c_alloc.payload)*c_alloc.chunkRatio);
	uint depth        = log2((float)(tid+1)); // Depth of the thread's node
	uint chunkSize    = rootChunk >> depth; // Chunks size corresponding to the level
	uint lvlTid       = tid - ((1 << depth) - 1);
	uint ofs          = depth*rootChunk + lvlTid*chunkSize; // Current at geometric sequence previous minus the first term
	uint nextOfs      = (tid == lastTid) ? lastMark : ofs + chunkSize;
	
	//printf("Tid %d Ofs %u Chunk size %u nextOfs %u\n", tid, ofs, chunkSize, nextOfs);

	setHeaderFused(tid, lastTid, ofs, nextOfs, lastMark);
}

//------------------------------------------------------------------------
// CircularMultiMalloc - Allocator using a circular linked list of chunks
//------------------------------------------------------------------------

extern "C" __global__ void CircularMultiMallocPrepare1(uint numChunks)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint lastMark = c_alloc.heapSize-CIRCULAR_MALLOC_HEADER_SIZE;

	uint chunkSize = align<uint, ALIGN>((CIRCULAR_MALLOC_HEADER_SIZE + c_alloc.payload)*c_alloc.chunkRatio);
#ifdef CIRCULAR_MALLOC_DOUBLY_LINKED
	uint prevOfs = (tid == 0) ? lastMark : (tid-1)*chunkSize; // Previous at the multiple of chunk
#endif
	uint ofs     = tid*chunkSize;
	uint nextOfs = (tid == lastTid) ? lastMark : (tid+1)*chunkSize; // Next at the multiple of chunk

	//printf("Tid %d Ofs %u Chunk size %u prevOfs %u nextOfs %u\n", tid, ofs, chunkSize, prevOfs, nextOfs);

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
	setHeader(tid, lastTid, ofs, nextOfs, lastMark);
#else
	setHeader(tid, lastTid, ofs, prevOfs, nextOfs, lastMark);
#endif

	// Set the heap offsets
	uint chunksPerSM = ceil((float)numChunks/(float)g_numSM);
	if(tid % chunksPerSM == 0)
		g_heapMultiOffset[tid / chunksPerSM] = ofs;
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMultiMallocPrepare2(uint numChunks)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint lastMark = c_alloc.heapSize-CIRCULAR_MALLOC_HEADER_SIZE;

	uint minChunkSize = align<uint, ALIGN>((CIRCULAR_MALLOC_HEADER_SIZE + c_alloc.payload)*c_alloc.chunkRatio);
	uint chunkSize = (1 << tid)*minChunkSize;
	//uint prevOfs = (tid == 0) ? lastMark : (1 << (tid-1))*minChunkSize - minChunkSize; // Previous at geometric sequence previous minus the first term
	uint ofs     = chunkSize - minChunkSize; // Current at geometric sequence previous minus the first term
#ifdef CIRCULAR_MALLOC_DOUBLY_LINKED
	uint prevOfs = (tid == 0) ? lastMark : ofs - (chunkSize>>1);
#endif
	//uint nextOfs = (tid == lastTid) ? lastMark : (1 << (tid+1))*minChunkSize - minChunkSize; // Next at geometric sequence next minus the first term
	uint nextOfs = (tid == lastTid) ? lastMark : ofs + chunkSize;
	
	//printf("Tid %d Ofs %u Chunk size %u prevOfs %u nextOfs %u\n", tid, ofs, chunkSize, prevOfs, nextOfs);

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
	setHeader(tid, lastTid, ofs, nextOfs, lastMark);
#else
	setHeader(tid, lastTid, ofs, prevOfs, nextOfs, lastMark);
#endif
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMultiMallocPrepare3(uint numChunks, uint rootChunk)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint chunksPerSM = numChunks / g_numSM;
	uint sid = tid % chunksPerSM;
	uint scnt = tid / chunksPerSM;
	uint subdepth = log2f(chunksPerSM+1); // Depth of the thread's node.

	uint lastMark = c_alloc.heapSize-CIRCULAR_MALLOC_HEADER_SIZE;

	uint minChunkSize = align<uint, ALIGN>((CIRCULAR_MALLOC_HEADER_SIZE + c_alloc.payload)*c_alloc.chunkRatio);
	// POSSIBLE BUG: Rounding errors may cause misaligned memory accesses - solved by double at the cost of significant slowdown
	uint depth = log2f((float)(sid+1)); // Depth of the thread's node.
	uint chunkSize = rootChunk >> depth; // Chunks size corresponding to the level
	uint lvlTid = sid - ((1 << depth) - 1);
	uint ofs    = scnt*rootChunk*subdepth + depth*rootChunk + lvlTid*chunkSize; // Previous subtrees plus current at geometric sequence previous minus the first term
#ifdef CIRCULAR_MALLOC_DOUBLY_LINKED
	uint chunkPrev = (lvlTid == 0) ? (chunkSize<<1) : chunkSize; // The previous chunk is either the same size of twice the size
	uint prevOfs = (tid == 0) ? lastMark : (sid == 0) ? (scnt-1)*rootChunk*subdepth : ofs - chunkPrev;
#endif
	uint nextOfs = (tid == lastTid) ? lastMark : ofs + chunkSize;
	
	//printf("Tid %d Ofs %u Chunk size %u prevOfs %u nextOfs %u\n", tid, ofs, chunkSize, prevOfs, nextOfs);

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
	setHeader(tid, lastTid, ofs, nextOfs, lastMark);
#else
	setHeader(tid, lastTid, ofs, prevOfs, nextOfs, lastMark);
#endif

	// Set the heap offsets
	if(sid == 0 && tid != lastTid)
		g_heapMultiOffset[scnt] = ofs;
}

//------------------------------------------------------------------------
// CircularMultiMallocFused - Allocator using a circular linked list of chunks with fused header and next pointer
//------------------------------------------------------------------------

extern "C" __global__ void CircularMultiMallocFusedPrepare1(uint numChunks)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint headerSize = sizeof(uint);
	uint lastMark   = c_alloc.heapSize-headerSize;

	uint chunkSize = align<uint, ALIGN>((headerSize + c_alloc.payload)*c_alloc.chunkRatio);
	uint ofs       = tid*chunkSize;
	uint nextOfs   = (tid == lastTid) ? lastMark : (tid+1)*chunkSize; // Next at the multiple of chunk

	//printf("Tid %d Ofs %u Chunk size %u nextOfs %u\n", tid, ofs, chunkSize, nextOfs);

	setHeaderFused(tid, lastTid, ofs, nextOfs, lastMark);

	// Set the heap offsets
	uint chunksPerSM = ceil((float)numChunks/(float)g_numSM);
	if(tid % chunksPerSM == 0)
		g_heapMultiOffset[tid / chunksPerSM] = ofs;
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMultiMallocFusedPrepare2(uint numChunks)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint headerSize = sizeof(uint);
	uint lastMark   = c_alloc.heapSize-headerSize;

	uint minChunkSize = align<uint, ALIGN>((headerSize + c_alloc.payload)*c_alloc.chunkRatio);
	uint chunkSize    = (1 << tid)*minChunkSize;
	uint ofs          = chunkSize - minChunkSize; // Current at geometric sequence previous minus the first term
	//uint nextOfs     = (tid == lastTid) ? lastMark : (1 << (tid+1))*minChunkSize - minChunkSize; // Next at geometric sequence next minus the first term
	uint nextOfs      = (tid == lastTid) ? lastMark : ofs + chunkSize;
	
	//printf("Tid %d Ofs %u Chunk size %u nextOfs %u\n", tid, ofs, chunkSize, nextOfs);

	setHeaderFused(tid, lastTid, ofs, nextOfs, lastMark);
}

//------------------------------------------------------------------------

extern "C" __global__ void CircularMultiMallocFusedPrepare3(uint numChunks, uint rootChunk)
{
	uint tid = threadIdx.x + blockDim.x * (blockIdx.x + gridDim.x * blockIdx.y);
	//int lastTid = blockDim.x-1 + blockDim.x * ((gridDim.x-1) + gridDim.x * (gridDim.y-1));
	uint lastTid = numChunks-1;

	if(tid >= numChunks)
		return;

	uint chunksPerSM = numChunks / g_numSM;
	uint sid = tid % chunksPerSM;
	uint scnt = tid / chunksPerSM;
	uint subdepth = log2f(chunksPerSM+1); // Depth of the thread's node.

	uint headerSize = sizeof(uint);
	uint lastMark   = c_alloc.heapSize-headerSize;

	uint minChunkSize = align<uint, ALIGN>((headerSize + c_alloc.payload)*c_alloc.chunkRatio);
	uint depth        = log2f((float)(sid+1)); // Depth of the thread's node.
	uint chunkSize    = rootChunk >> depth; // Chunks size corresponding to the level
	uint lvlTid       = sid - ((1 << depth) - 1);
	uint ofs          = scnt*rootChunk*subdepth + depth*rootChunk + lvlTid*chunkSize; // Previous subtrees plus current at geometric sequence previous minus the first term
	uint nextOfs      = (tid == lastTid) ? lastMark : ofs + chunkSize;
	
	//printf("Tid %d Ofs %u Chunk size %u nextOfs %u\n", tid, ofs, chunkSize, nextOfs);

	setHeaderFused(tid, lastTid, ofs, nextOfs, lastMark);

	// Set the heap offsets
	if(sid == 0 && tid != lastTid)
		g_heapMultiOffset[scnt] = ofs;
}
#include "base/Random.hpp"
#include "persistentds/CudaPersistentSBVHBuilder.hpp"
#include "persistentds/PersistentHelper.hpp"
#include "AppEnvironment.h"

using namespace FW;

#define TASK_SIZE 150000
#define BENCHMARK

// Allocator settings
#define CIRCULAR_MALLOC_PREALLOC 3
#define CPU 0
#define GPU 1
#define CIRCULAR_MALLOC_INIT GPU
#define CIRCULAR_MALLOC_PREALLOC_SPECIAL

#include <cuda_profiler_api.h>


//------------------------------------------------------------------------

CudaPersistentSBVHBuilder::CudaPersistentSBVHBuilder(Scene& scene, F32 epsilon) : CudaBVH(BVHLayout_Compact), m_epsilon(epsilon), m_numTris(scene.getNumTriangles()), m_trisCompact(scene.getTriCompactBuffer())
{
	// debug
	size_t printfBuffer;
	cudaDeviceGetLimit(&printfBuffer, cudaLimitPrintfFifoSize);
	cudaDeviceSetLimit(cudaLimitPrintfFifoSize, printfBuffer * 10);
	cudaDeviceGetLimit(&printfBuffer, cudaLimitPrintfFifoSize);
	// init
	CudaModule::staticInit();
	//m_compiler.addOptions("-use_fast_math");
	m_compiler.addOptions("-use_fast_math -Xptxas -dlcm=cg");

	//m_trisCompactIndex.resizeDiscard(m_numTris * sizeof(S32));
	scene.getBBox(m_bboxMin, m_bboxMax);

	m_sizeTask = 0.f;
	m_sizeSplit = 0.f;
	m_sizeADS = 0.f;
	m_sizeTri = 0.f;
	m_sizeTriIdx = 0.f;
	m_heap = 0.f;

#ifndef BENCHMARK
	Debug.open("persistent_sbvh_debug.log");
#endif
}

//------------------------------------------------------------------------

CudaPersistentSBVHBuilder::~CudaPersistentSBVHBuilder()
{
#ifndef BENCHMARK
	Debug.close();
#endif
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::prepareDynamicMemory()
{
	// Set the memory limit according to triangle count
	U64 allocSize = (U64)((m_trisCompact.getSize() / 12 / sizeof(int) * sizeof(Reference)) * Environment::GetSingleton()->GetFloat("PersistentSBVH.heapMultiplicator"));

#if (MALLOC_TYPE == SCATTER_ALLOC) || (MALLOC_TYPE == FDG_MALLOC)
	allocSize = max(allocSize, 8ULL*1024ULL*1024ULL);
#elif (MALLOC_TYPE == HALLOC)
	// Memory pool size must be larger than 256MB, otherwise allocation always fails
	// May be possibly tweaked by changing halloc_opts_t.sb_sz_sh
	allocSize = max(allocSize, 256ULL*1024ULL*1024ULL);
#endif

#if (MALLOC_TYPE == CUDA_MALLOC) || (MALLOC_TYPE == FDG_MALLOC)
	CudaModule::checkError("cuCtxSetLimit", cuCtxSetLimit(CU_LIMIT_MALLOC_HEAP_SIZE, allocSize));
	size_t val;
	cuCtxGetLimit(&val, CU_LIMIT_MALLOC_HEAP_SIZE);
	printf("HEAP SIZE: %ull\n",val);
#elif (MALLOC_TYPE == ATOMIC_MALLOC) || (MALLOC_TYPE == ATOMIC_MALLOC_CIRCULAR) || (MALLOC_TYPE == CIRCULAR_MALLOC) || (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED) \
	|| (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC_FUSED) || (MALLOC_TYPE == SCATTER_ALLOC)
	m_mallocData.resizeDiscard(allocSize);
#ifdef CIRCULAR_MALLOC_WITH_SCATTER_ALLOC
	m_mallocData2.resizeDiscard(allocSize);
#endif
#endif

#if (MALLOC_TYPE == SCATTER_ALLOC) || defined(CIRCULAR_MALLOC_WITH_SCATTER_ALLOC)
	// CUDA Driver API cannot deal with templates -> use C++ mangled name
	CudaKernel initHeap = m_module->getKernel("_ZN8GPUTools8initHeapILj" STR(SCATTER_ALLOC_PAGESIZE) "ELj" STR(SCATTER_ALLOC_ACCESSBLOCKS)
		"ELj" STR(SCATTER_ALLOC_REGIONSIZE) "ELj" STR(SCATTER_ALLOC_WASTEFACTOR) "ELb" STR(SCATTER_ALLOC_COALESCING) "ELb" STR(SCATTER_ALLOC_RESETPAGES)
		"EEEvPNS_10DeviceHeapIXT_EXT0_EXT1_EXT2_EXT3_EXT4_EEEPvj");

	FW_ASSERT(allocSize < MAXUINT32);

	initHeap.setParams(
		m_module->getGlobal("theHeap").getMutableCudaPtr(),
#ifdef CIRCULAR_MALLOC_WITH_SCATTER_ALLOC
		m_mallocData2.getMutableCudaPtr(),
#else
		m_mallocData.getMutableCudaPtr(),
#endif
		(U32)allocSize);

#if 0
	F32 initTime = initHeap.launchTimed(1, Vec2i(256, 1));
#else
	unsigned int numregions = ((unsigned long long)m_mallocData.getSize())/( ((unsigned long long)SCATTER_ALLOC_REGIONSIZE)*(3*sizeof(unsigned int)+SCATTER_ALLOC_PAGESIZE)+sizeof(unsigned int));
    unsigned int numpages = numregions*SCATTER_ALLOC_REGIONSIZE;
	F32 initTime = initHeap.launchTimed(numpages, Vec2i(256, 1));
#endif

	printf("Scatter alloc initialized in %f\n", initTime);

#elif (MALLOC_TYPE == HALLOC)
	// Set the memory limit
	halloc_opts_t opts = halloc_opts_t((size_t)allocSize);

	// TODO: initialize all devices
	// get total device memory (in bytes) & total number of superblocks
	uint64 dev_memory;
	cudaDeviceProp dev_prop;
	int dev;
	cucheck(cudaGetDevice(&dev));
	cucheck(cudaGetDeviceProperties(&dev_prop, dev));
	dev_memory = dev_prop.totalGlobalMem;
	uint sb_sz = 1 << opts.sb_sz_sh;

	// set cache configuration
	cucheck(cudaDeviceSetCacheConfig(cudaFuncCachePreferL1));

	// limit memory available to 3/4 of device memory
	opts.memory = min((uint64)opts.memory, 3ull * dev_memory / 4ull);

	// split memory between halloc and CUDA allocator
	uint64 halloc_memory = opts.halloc_fraction * opts.memory;
	uint64 cuda_memory = opts.memory - halloc_memory;
	cucheck(cudaDeviceSetLimit(cudaLimitMallocHeapSize, cuda_memory));
	cuset(cuda_mem_g, uint64, cuda_memory);
	cuset(total_mem_g, uint64, halloc_memory + cuda_memory);

	// set the number of slabs
	//uint nsbs = dev_memory / sb_sz;
	uint nsbs = halloc_memory / sb_sz;
	cuset(nsbs_g, uint, nsbs);
	cuset(sb_sz_g, uint, sb_sz);
	cuset(sb_sz_sh_g, uint, opts.sb_sz_sh);

	// allocate a fixed number of superblocks, copy them to device
	uint nsbs_alloc = (uint)min((uint64)nsbs, (uint64)halloc_memory / sb_sz);
	size_t sbs_sz = MAX_NSBS * sizeof(superblock_t);
	size_t sb_ptrs_sz = MAX_NSBS * sizeof(void *);
	superblock_t *sbs = (superblock_t *)malloc(sbs_sz);
	void **sb_ptrs = (void **)malloc(sb_ptrs_sz);
	memset(sbs, 0, sbs_sz);
	memset(sb_ptrs, 0, sb_ptrs_sz);
	uint *sb_counters = (uint *)malloc(MAX_NSBS * sizeof(uint));
	memset(sbs, 0xff, MAX_NSBS * sizeof(uint));
	char *base_addr = (char *)~0ull;
	/********************************************/
	halloc_base = (char *)0;
	/********************************************/
	for(uint isb = 0; isb < nsbs_alloc; isb++) {
		sb_counters[isb] = sb_counter_val(0, false, SZ_NONE, SZ_NONE);
		sbs[isb].size_id = SZ_NONE;
		sbs[isb].chunk_id = SZ_NONE;
		sbs[isb].is_head = 0;
		//sbs[isb].flags = 0;
		sbs[isb].chunk_sz = 0;
		//sbs[isb].chunk_id = SZ_NONE;
		//sbs[isb].state = SB_FREE;
		//sbs[isb].mutex = 0;
		cucheck(cudaMalloc(&sbs[isb].ptr, sb_sz));
		sb_ptrs[isb] = sbs[isb].ptr;
		base_addr = (char *)min((uint64)base_addr, (uint64)sbs[isb].ptr);
		/********************************************/
		//halloc_base = (char *)max((uint64)base_addr, (uint64)sbs[isb].ptr);
		/********************************************/
	}
	/********************************************/
		halloc_base = base_addr;
	/********************************************/
	//cuset_arr(sbs_g, (superblock_t (*)[MAX_NSBS])&sbs);
	cuset_arr(sbs_g, (superblock_t (*)[MAX_NSBS])sbs);
	cuset_arr(sb_counters_g, (uint (*)[MAX_NSBS])sb_counters);
	cuset_arr(sb_ptrs_g, (void* (*)[MAX_NSBS])sb_ptrs);
	// also mark free superblocks in the set
	sbset_t free_sbs;
	memset(free_sbs, 0, sizeof(free_sbs));
	for(uint isb = 0; isb < nsbs_alloc; isb++) {
		uint iword = isb / WORD_SZ, ibit = isb % WORD_SZ;
		free_sbs[iword] |= 1 << ibit;
	}
	free_sbs[SB_SET_SZ - 1] = nsbs_alloc;
	cuset_arr(free_sbs_g, &free_sbs);
	base_addr = (char *)((uint64)base_addr / sb_sz * sb_sz);
	if((uint64)base_addr < dev_memory)
		base_addr = 0;
	else
		base_addr -= dev_memory;
	cuset(base_addr_g, void *, base_addr);

	// allocate block bits and zero them out
	void *bit_blocks, *alloc_sizes;
	uint nsb_bit_words = sb_sz / (BLOCK_STEP * WORD_SZ),
		nsb_alloc_words = sb_sz / (BLOCK_STEP * 4);
	// TODO: move numbers into constants
	uint nsb_bit_words_sh = opts.sb_sz_sh - (4 + 5);
	cuset(nsb_bit_words_g, uint, nsb_bit_words);
	cuset(nsb_bit_words_sh_g, uint, nsb_bit_words_sh);
	cuset(nsb_alloc_words_g, uint, nsb_alloc_words);
	size_t bit_blocks_sz = nsb_bit_words * nsbs * sizeof(uint), 
		alloc_sizes_sz = nsb_alloc_words * nsbs * sizeof(uint);
	cucheck(cudaMalloc(&bit_blocks, bit_blocks_sz));
	cucheck(cudaMemset(bit_blocks, 0, bit_blocks_sz));
	cuset(block_bits_g, uint *, (uint *)bit_blocks);
	cucheck(cudaMalloc(&alloc_sizes, alloc_sizes_sz));
	cucheck(cudaMemset(alloc_sizes, 0, alloc_sizes_sz));
	cuset(alloc_sizes_g, uint *, (uint *)alloc_sizes);

	// set sizes info
	//uint nsizes = (MAX_BLOCK_SZ - MIN_BLOCK_SZ) / BLOCK_STEP + 1;
	uint nsizes = 2 * NUNITS;
	cuset(nsizes_g, uint, nsizes);
	size_info_t size_infos[MAX_NSIZES];
	memset(size_infos, 0, MAX_NSIZES * sizeof(size_info_t));
	for(uint isize = 0; isize < nsizes; isize++) {
		uint iunit = isize / 2, unit = 1 << (iunit + 3);
		size_info_t *size_info = &size_infos[isize];
		//size_info->block_sz = isize % 2 ? 3 * unit : 2 * unit;
		uint block_sz = isize % 2 ? 3 * unit : 2 * unit;
		uint nblocks = sb_sz / block_sz;
		// round #blocks to a multiple of THREAD_MOD
		uint tmod = tmod_by_size(isize);
		nblocks = nblocks / tmod * tmod;
		//nblocks = nblocks / THREAD_MOD * THREAD_MOD;
		size_info->chunk_id = isize % 2 + (isize < nsizes / 2 ? 0 : 2);
		uint chunk_sz = (size_info->chunk_id % 2 ? 3 : 2) * 
			(size_info->chunk_id / 2 ? 128 : 8);
		size_info->chunk_sz = chunk_val(chunk_sz);
		size_info->nchunks_in_block = block_sz / chunk_sz;
		size_info->nchunks = nblocks * size_info->nchunks_in_block;
		// TODO: use a better hash step
		size_info->hash_step = size_info->nchunks_in_block *
		 	max_prime_below(nblocks / 256 + nblocks / 64, nblocks);
		//size_info->hash_step = size_info->nchunks_in_block * 17;
		// printf("block = %d, step = %d, nchunks = %d, nchunks/block = %d\n", 
		// 			 block_sz, size_info->hash_step, size_info->nchunks, 
		// 			 size_info->nchunks_in_block);
		size_info->roomy_threshold = opts.roomy_fraction * size_info->nchunks;
		size_info->busy_threshold = opts.busy_fraction * size_info->nchunks;
		size_info->sparse_threshold = opts.sparse_fraction * size_info->nchunks;
	}  // for(each size)
	cuset_arr(size_infos_g, &size_infos);

	// set grid info
	uint64 sb_grid[2 * MAX_NSBS];
	for(uint icell = 0; icell < 2 * MAX_NSBS; icell++) 
		sb_grid[icell] = grid_cell_init();
	for(uint isb = 0; isb < nsbs_alloc; isb++)
		grid_add_sb(sb_grid, base_addr, isb, sbs[isb].ptr, sb_sz);
	cuset_arr(sb_grid_g, &sb_grid);
	
	// zero out sets (but have some of the free set)
	//fprintf(stderr, "started cuda-memsetting\n");
	//cuvar_memset(unallocated_sbs_g, 0, sizeof(unallocated_sbs_g));
	cuvar_memset(busy_sbs_g, 0, sizeof(roomy_sbs_g));
	cuvar_memset(roomy_sbs_g, 0, sizeof(roomy_sbs_g));
	cuvar_memset(sparse_sbs_g, 0, sizeof(sparse_sbs_g));
	//cuvar_memset(roomy_sbs_g, 0, (MAX_NSIZES * SB_SET_SZ * sizeof(uint)));
	cuvar_memset(head_sbs_g, ~0, sizeof(head_sbs_g));
	cuvar_memset(cached_sbs_g, ~0, sizeof(head_sbs_g));
	cuvar_memset(head_locks_g, 0, sizeof(head_locks_g));
	cuvar_memset(sb_locks_g, 0, sizeof(sb_locks_g));
	//cuvar_memset(counters_g, 1, sizeof(counters_g));
	cuvar_memset(counters_g, 11, sizeof(counters_g));
	//fprintf(stderr, "finished cuda-memsetting\n");
	cucheck(cudaStreamSynchronize(0));

	// free all temporary data structures
	free(sbs);
	free(sb_counters);

#endif

#ifdef CIRCULAR_MALLOC_CHECK_INTERNAL_FRAGMENTATION
	int numWarpsPerBlock = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numWarpsPerBlock");
	int numBlocksPerSM = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numBlockPerSM");

	// Prepare memory for internal fragmentation data
	m_interFragSum.resizeDiscard(numWarpsPerBlock*WARP_SIZE * NUM_SM*numBlocksPerSM * sizeof(float));
	m_interFragSum.clear(0);
	float*& interFragSum = *(float**)m_module->getGlobal("g_interFragSum").getMutablePtr();
	interFragSum = (float*)m_interFragSum.getMutableCudaPtr();
#endif
}

//------------------------------------------------------------------------

int CudaPersistentSBVHBuilder::setDynamicMemory()
{
	int baseOffset = 0;
	int heapSize = int(m_mallocData.getSize());
	
	AllocInfo& allocInfo = *(AllocInfo*)m_module->getGlobal("c_alloc").getMutablePtr();
	allocInfo.heapSize = heapSize;

	// Prepare allocations for the methods supporting direct allocation
#if (MALLOC_TYPE == ATOMIC_MALLOC) || (MALLOC_TYPE == ATOMIC_MALLOC_CIRCULAR)
	// Set the heapBase, heapOffset and heapSize
	char*& heapBase = *(char**)m_module->getGlobal("g_heapBase").getMutablePtr();
	heapBase = (char*)m_mallocData.getMutableCudaPtr();
	int& heapOffset = *(int*)m_module->getGlobal("g_heapOffset").getMutablePtr();

#if SCAN_TYPE < 2
	heapOffset = align<U32, ALIGN>(4*m_numTris*sizeof(int));
#else
	heapOffset = align<U32, ALIGN>(m_numTris*sizeof(Reference));
#endif

#elif (MALLOC_TYPE == CIRCULAR_MALLOC) || (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC_FUSED)
	// Set the heapBase, heapOffset and heapSize
	char*& heapBase = *(char**)m_module->getGlobal("g_heapBase").getMutablePtr();
	heapBase = (char*)m_mallocData.getMutableCudaPtr();
	U32& heapOffset = *(U32*)m_module->getGlobal("g_heapOffset").getMutablePtr();
	heapOffset = 0;

	// Init the heapMultiOffset
	CUdevice device;
	int m_numSM;
	CudaModule::checkError("cuCtxGetDevice", cuCtxGetDevice(&device));
	CudaModule::checkError("cuDeviceGetAttribute", cuDeviceGetAttribute(&m_numSM, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));
	U32& numSM = *(U32*)m_module->getGlobal("g_numSM").getMutablePtr();
	numSM = m_numSM;

	m_multiOffset.resizeDiscard(m_numSM * sizeof(unsigned int*));
	unsigned int*& heapMultiOffset = *(unsigned int**)m_module->getGlobal("g_heapMultiOffset").getMutablePtr();

	U32 rootSize = 0;
#if SCAN_TYPE < 2
	rootSize = 4*m_numTris*sizeof(int);
#else
	rootSize = m_numTris*sizeof(Reference);
#endif

	U32& heapLock = *(U32*)m_module->getGlobal("g_heapLock").getMutablePtr();
	heapLock = 0;

	allocInfo.payload = rootSize;
	allocInfo.maxFrag = 2;
	allocInfo.chunkRatio = 1;

#ifdef CIRCULAR_MALLOC_PREALLOC_SPECIAL
	heapMultiOffset = (unsigned int*)m_multiOffset.getMutablePtr();

#if (MALLOC_TYPE == CIRCULAR_MALLOC) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC)
	U32 headerSize = CIRCULAR_MALLOC_HEADER_SIZE;
#elif (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC_FUSED)
	U32 headerSize = sizeof(U32);
#endif

	heapOffset = align<U32, ALIGN>(rootSize + headerSize);
	U32 prevOfs = 0;

	setCircularMallocHeader(true, 0, heapSize-headerSize, heapOffset); // Locked, allocated memory for parent

#ifdef CIRCULAR_MALLOC_GLOBAL_HEAP_LOCK
	setCircularMallocHeader(false, heapOffset, 0, heapSize-headerSize); // Unlocked, next at the end of allocated memory
#else
#if (CIRCULAR_MALLOC_PREALLOC == 1)
	// Create regular chunks
	U32 numChunks = m_mallocData.getSize()/heapOffset;
	U32 chunksPerSM = ceil((F32)numChunks/(F32)m_numSM);
	U32 ofs = heapOffset;
	for(U32 i = 1; i < numChunks; i++)
	{
		setCircularMallocHeader(false, ofs, (i-1)*heapOffset, (i+1)*heapOffset); // Unlocked, next at the multiple of heapOffset

		// Set the heap offsets
		if((i-1) % chunksPerSM == 0)
			heapMultiOffset[(i-1) / chunksPerSM] = ofs;

		ofs = (i+1)*heapOffset;
	}

#elif (CIRCULAR_MALLOC_PREALLOC == 3)
	for(int i = 0; i < m_numSM; i++)
		heapMultiOffset[i] = heapOffset;

#if (MALLOC_TYPE == CIRCULAR_MALLOC) || (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED)
	// Create hierarchical chunks
	//int delta = align<U32, ALIGN>(heapOffset+headerSize);
	U32 delta = heapOffset;
	U32 ofs;
	U32 i = 0;
	U32 lvl = 2;
	for(ofs = heapOffset; true; ofs += delta, i++)
	{
		if(i == lvl) // New level in BFS order
		{
			//heapMultiOffset[(int)log2((float)lvl)] = ofs; // Not very clever solution, we do not know which SM will allocate memory
			delta = align<U32, ALIGN>(U32((delta * 0.8f)+headerSize));
			i = 0;
			lvl *= 2;
		}

		if(ofs+delta >= heapSize-2*headerSize) // We cannot make another chunk
			break;

		setCircularMallocHeader(false, ofs, prevOfs, ofs+delta); // Unlocked, next at ofs+delta

		prevOfs = ofs;
	}

#elif (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC_FUSED)
	// Create hierarchical chunks
	F32 heapTotalMem = heapSize-2*headerSize;
	F32 heapMem = heapTotalMem/(F32)m_numSM;

	U32 ofs = heapOffset;

	for(int r = 0; r < m_numSM; r++)
	{
		// Set the heap offsets
		heapMultiOffset[r] = ofs;
		U32 localOfs = 0;

		//int delta = align<U32, ALIGN>(heapOffset+headerSize);
		U32 delta = heapOffset;
		U32 i = 0;
		U32 lvl = 2;
		for(; true; ofs += delta, localOfs += delta, i++)
		{
			if(i == lvl) // New level in BFS order
			{
				delta = align<U32, ALIGN>((delta * 0.8f)+headerSize);
				i = 0;
				lvl *= 2;
			}

			if(localOfs+delta >= heapMem) // We cannot make another chunk
				break;

			setCircularMallocHeader(false, ofs, prevOfs, ofs+delta); // Unlocked, next at ofs+delta

			prevOfs = ofs;
		}

		if(ofs+delta >= heapTotalMem) // We cannot make another chunk
			break;
	}
#endif

#else
#error Unsupported special CPU initialization method!
#endif

	setCircularMallocHeader(false, ofs, prevOfs, heapSize-headerSize); // Unlocked, next at the end of allocated memory
#endif

	setCircularMallocHeader(true, heapSize-headerSize, ofs, 0); // Locked, next at the start of heap

	heapMultiOffset = (unsigned int*)m_multiOffset.getMutableCudaPtr();

#else
	F32 initTime = 0.f;

#if (MALLOC_TYPE == CIRCULAR_MALLOC) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC)
	U32 headerSize = CIRCULAR_MALLOC_HEADER_SIZE;
	String method("CircularMalloc");
#elif (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED) || (MALLOC_TYPE == CIRCULAR_MULTI_MALLOC_FUSED)
	U32 headerSize = sizeof(U32);
	String method("CircularMallocFused");
#endif

	// Set the chunk size
	U32 numChunks = 0;
	U32 chunkSize = align<U32, ALIGN>(U32((headerSize + allocInfo.payload)*allocInfo.chunkRatio));

#if (CIRCULAR_MALLOC_INIT == CPU)
	// Prepare the buffer on the CPU
	//m_mallocData.getMutablePtr();
	// Offset of the division
	U32 ofs = 0;
	U32 prevOfs = 0;

#if (CIRCULAR_MALLOC_PREALLOC == 0)
	ofs = 0;
	prevOfs = heapSize-headerSize;

#elif (CIRCULAR_MALLOC_PREALLOC == 1)
	// Create regular chunks
	prevOfs = heapSize-headerSize;
	ofs = 0;

	numChunks = (m_mallocData.getSize()-(chunkSize+headerSize))/chunkSize;
	for(int i = 0; i < numChunks; i++)
	{
#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
		Vec2u next(AllocatorLockType_Free, (i+1)*chunkSize); // Unlocked, next at the multiple of chunkSize
		m_mallocData.setRange(i*chunkSize, &next, sizeof(Vec2u)); // Set the next header
#else
		Vec4u next(AllocatorLockType_Free, prevOfs, (i+1)*chunkSize, 0); // Unlocked, next at the multiple of chunkSize
		m_mallocData.setRange(i*chunkSize, &next, sizeof(Vec4u)); // Set the next header
#endif

		prevOfs = ofs;
		ofs += chunkSize;
	}

#elif (CIRCULAR_MALLOC_PREALLOC == 2)
	// Create exponential chunks
	prevOfs = heapSize-headerSize;
#if 1
	for(ofs = 0; ofs+chunkSize < heapSize-2*headerSize && ofs+chunkSize > ofs;)
#else
	U32 minChunkSize = chunkSize;
	U32 expChunks = log2((float)(heapSize-2*headerSize)/(float)chunkSize) - 0.5f; // Temporary
	chunkSize = (1 << expChunks) * minChunkSize;
	for(ofs = 0; ofs+chunkSize < heapSize-2*headerSize && ofs+chunkSize > ofs && chunkSize >= minChunkSize;)
#endif
	{
#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
		Vec2u next(AllocatorLockType_Free, ofs+chunkSize); // Unlocked, next at the multiple of chunkSize
		m_mallocData.setRange(ofs, &next, sizeof(Vec2u)); // Set the next header
#else
		Vec4u next(AllocatorLockType_Free, prevOfs, ofs+chunkSize, 0); // Unlocked, next at the multiple of chunkSize
		m_mallocData.setRange(ofs, &next, sizeof(Vec4u)); // Set the next header
#endif
		//printf("Ofs %u Chunk size %u\n", ofs, chunkSize);
		//printf("Ofs %u Chunk size %u. Ofs + Chunk size %u, Heap %u\n", ofs, chunkSize, ofs + chunkSize, heapSize-2*headerSize);
		prevOfs = ofs;
		ofs += chunkSize;
#if 1
		chunkSize = align<U32, ALIGN>(chunkSize*2);
#else
		chunkSize = align<U32, ALIGN>(chunkSize/2);
#endif
		numChunks++;
	}

#elif (CIRCULAR_MALLOC_PREALLOC == 3)
	// Create hierarchical chunks
	U32 minChunkSize = chunkSize;
	F32 treeMem = minChunkSize;
	U32 i = 1;
	for(; treeMem < heapSize-2*headerSize; i++)
	{
		treeMem = ((float)(i+1))*((float)(1 << i))*((float)minChunkSize);
	}

	chunkSize = (1 << (i-2))*minChunkSize;

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
	Vec2u first(AllocatorLockType_Free, chunkSize); // Locked, allocated memory for parent
	m_mallocData.setRange(0, &first, sizeof(Vec2u)); // Set the first header
#else
	Vec4u first(AllocatorLockType_Free, heapSize-headerSize, chunkSize, 0); // Locked, allocated memory for parent
	m_mallocData.setRange(0, &first, sizeof(Vec4u)); // Set the first header
#endif
	numChunks++;

	//printf("Ofs %u Chunk size %u\n", ofs, chunkSize);

	i = 0;
	U32 lvl = 1;
	for(ofs = chunkSize; true; ofs += chunkSize, i++)
	{
		if(i == 0 || i == lvl) // New level in BFS order
		{
			chunkSize = align<U32, ALIGN>(chunkSize/2);
			i = 0;
			lvl *= 2;
		}

		if(ofs+chunkSize >= heapSize-2*headerSize || chunkSize < minChunkSize) // We cannot make another chunk
			break;

		//printf("Ofs %u Chunk size %u\n", ofs, chunkSize);

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
		Vec2u next(AllocatorLockType_Free, ofs+chunkSize); // Unlocked, next at the multiple of chunkSize
		m_mallocData.setRange(ofs, &next, sizeof(Vec2u)); // Set the next header
#else
		Vec4u next(AllocatorLockType_Free, prevOfs, ofs+chunkSize, 0); // Unlocked, next at the multiple of chunkSize
		m_mallocData.setRange(ofs, &next, sizeof(Vec4u)); // Set the next header
#endif

		prevOfs = ofs;
		numChunks++;
	}

#else
#error Unsupported CPU initialization method!
#endif

	//printf("Ofs %u Chunk size %u\n", ofs, (heapSize-headerSize)-ofs);
	numChunks++;

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
	Vec2u last(AllocatorLockType_Free, heapSize-headerSize); // Unlocked, next at the end of allocated memory
	m_mallocData.setRange(ofs, &last, sizeof(Vec2u)); // Set the last header
#else
	Vec4u last(AllocatorLockType_Free, prevOfs, heapSize-headerSize, 0); // Unlocked, next at the end of allocated memory
	m_mallocData.setRange(ofs, &last, sizeof(Vec4u)); // Set the last header
#endif

#ifndef CIRCULAR_MALLOC_DOUBLY_LINKED
	Vec2u tail(AllocatorLockType_Set, 0); // Locked, next at the start of heap
	m_mallocData.setRange(heapSize-headerSize, &tail, sizeof(Vec2u)); // Set the last header
#else
	Vec4u tail(AllocatorLockType_Set, ofs, 0, 0); // Locked, next at the start of heap
	m_mallocData.setRange(heapSize-headerSize, &tail, sizeof(Vec4u)); // Set the last header
#endif

	// Transfer the buffer to the GPU
	//m_mallocData.getMutableCudaPtr();

#elif (CIRCULAR_MALLOC_INIT == GPU)
#if (CIRCULAR_MALLOC_PREALLOC == 1)
	// Create regular chunks
	CudaKernel initHeap = m_module->getKernel(method+"Prepare1");

	numChunks = (m_mallocData.getSize()-headerSize)/chunkSize;
	kernel.setParams(numChunks);
	initTime = initHeap.launchTimed(numChunks, Vec2i(256, 1));

#ifndef BENCHMARK
	if(m_firstRun)
		printf("Grid dimensions tpb %d, gridDim.x %d\n", tpb, gridSize.x);
#endif

#elif (CIRCULAR_MALLOC_PREALLOC == 2)
	// Create exponential chunks
	CudaKernel initHeap = m_module->getKernel(method+"Prepare2");

	numChunks = ceil(log2((float)(heapSize-2*headerSize)/(float)chunkSize));
	kernel.setParams(numChunks);
	initTime = initHeap.launchTimed(numChunks, Vec2i(256, 1));

#elif (CIRCULAR_MALLOC_PREALLOC == 3)
	// Create hierarchical chunks
	U32 minChunkSize = chunkSize;
	F32 treeMem = minChunkSize;
	U32 i = 1;
	for(; treeMem < heapSize-2*headerSize; i++)
	{
		treeMem = ((float)(i+1))*((float)(1 << i))*((float)minChunkSize);
	}

	chunkSize = (1 << (i-2))*minChunkSize;

	CudaKernel initHeap = m_module->getKernel(method+"Prepare3");

	numChunks = (1 << (i-1)); // Number of nodes of the tree + 1 for the rest
	kernel.setParams(numChunks, chunkSize);
	initTime = initHeap.launchTimed(numChunks, Vec2i(256, 1));
#else
#error Unsupported GPU initialization method!
#endif

#ifndef BENCHMARK
	printf("Init heap executed for heap size %lld, headerSize %d, chunkSize %u, numChunks %u\n", m_mallocData.getSize(), headerSize, chunkSize, numChunks);
#endif
#endif
#endif

	// Offset of the memory allocation
	baseOffset = headerSize;


#ifdef CIRCULAR_MALLOC_WITH_SCATTER_ALLOC
	// With scatter alloc
	char*& heapBase2 = *(char**)m_module->getGlobal("g_heapBase2").getMutablePtr();
	heapBase2 = (char*)m_mallocData2.getMutableCudaPtr();
#endif

#endif // ATOMIC_MALLOC || ATOMIC_MALLOC_CIRCULAR || CIRCULAR_MALLOC || CIRCULAR_MALLOC_FUSED

// Allocate first chunk of memory for the methods than do not support direct allocation
#if (MALLOC_TYPE == CUDA_MALLOC) || (MALLOC_TYPE == SCATTER_ALLOC) || (MALLOC_TYPE == FDG_MALLOC) || (MALLOC_TYPE == HALLOC) || (!defined(CIRCULAR_MALLOC_PREALLOC_SPECIAL) && ((MALLOC_TYPE == CIRCULAR_MALLOC) || (MALLOC_TYPE == CIRCULAR_MALLOC_FUSED)))
	CudaKernel kernelAlloc = m_module->getKernel("allocFreeableMemory");

	int size = m_numTris * sizeof(Reference) / sizeof(int);
	kernelAlloc.setParams(
		//48,
		size,
		0);

	F32 allocTime = kernelAlloc.launchTimed(1, 1);

#ifndef BENCHMARK
	printf("Memory allocated in %f\n", allocTime);
#endif
#endif

	// Copy the triangle index data into the first allocation
	CudaKernel kernelCopy = m_module->getKernel("MemCpyIndex");

	int memSize = int(m_refs.getSize()/sizeof(Reference) * (sizeof(Reference)/sizeof(int)));
	kernelCopy.setParams(
		m_refs.getCudaPtr(),
		baseOffset,
		memSize);

	
#ifndef BENCHMARK
	F32 memcpyTime = kernelCopy.launchTimed(memSize, Vec2i(256, 1));
	printf("Triangle indices copied in %f\n", memcpyTime);
#else
	kernelCopy.launchTimed(memSize, Vec2i(256, 1));
#endif

#if (MALLOC_TYPE == SCATTER_ALLOC)
	// Compute the base offset as the difference between the first allocation and the heap start
	char*& heapBase = *(char**)m_module->getGlobal("g_heapBase").getMutablePtr();
	char* base = (char*)m_mallocData.getMutableCudaPtr();
	baseOffset = heapBase - base;
	heapBase = base;

#elif (MALLOC_TYPE == HALLOC)
	// Set the heapBase
	char*& heapBase = *(char**)m_module->getGlobal("g_heapBase").getMutablePtr();
	char*& heapBase2 = *(char**)m_module->getGlobal("g_heapBase2").getMutablePtr();
	baseOffset = heapBase - heapBase2;
	heapBase = heapBase2;
	heapBase2 = halloc_base;
#endif

	return baseOffset;
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::setCircularMallocHeader(bool set, U32 ofs, U32 prevOfs, U32 nextOfs)
{
	AllocatorLockType type;
	if(set)
		type = AllocatorLockType_Set;
	else
		type = AllocatorLockType_Free;

	Vec2u header(type, nextOfs);
	m_mallocData.setRange(ofs, &header, sizeof(Vec2u));
}

//------------------------------------------------------------------------

F32 CudaPersistentSBVHBuilder::build()
{
	Vec4f * verts = (Vec4f*)m_trisCompact.getPtr();

	// Compile the kernel
	m_compiler.setSourceFile("src/rt/kernels/persistent_sbvh.cu");
	m_compiler.clearDefines();
	m_module = m_compiler.compile();
	failIfError();

	prepareDynamicMemory();

#ifdef DEBUG_PPS
	Random rand;
	m_numTris = rand.getU32(1, 1000000);
#endif

	// Prepare references
	FW::Array<Reference> refs;
	refs.reserve(m_numTris);

	// Set triangle index buffer
	//S32* tiout = (S32*)m_trisCompactIndex.getMutablePtr();
#ifdef DEBUG_PPS
	S32* pout = (S32*)m_ppsTris.getMutablePtr();
	S32* clout = (S32*)m_ppsTrisIndex.getMutablePtr();
	S32* sout = (S32*)m_sortTris.getMutablePtr();
#endif
	for(int i=0;i<m_numTris;i++)
	{
		Reference ref;
		ref.idx = i;
		for(int j=0;j<3;j++)
			ref.bbox.grow(verts[3*i+j].getXYZ());
		refs.add(ref);

#ifndef DEBUG_PPS
		// indices 
		//*tiout = i;
		//tiout++;
#else
		int rnd = rand.getU32(0, 2);
		//*pout = rnd;
		*clout = rnd;
		*sout = (rnd >= 1);
		//pout++;
		clout++;
		sout++;
#endif
	}

	m_refs.resizeDiscard(m_numTris * sizeof(Reference));
	m_refs.set(refs);

	// Start the timer
	m_timer.unstart();
	m_timer.start();

	// Create the taskData
	m_taskData.resizeDiscard(TASK_SIZE * (sizeof(TaskBVH) + sizeof(int)));
	m_taskData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
	m_splitData.resizeDiscard((S64)(TASK_SIZE+1) * (S64)sizeof(SplitArray));
	m_splitData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#endif

	// Node and triangle data
	S64 bvhSize = align<S64, 4096>(m_numTris * Environment::GetSingleton()->GetFloat("PersistentSBVH.nodeRatio") * sizeof(CudaBVHNode));
	m_nodes.resizeDiscard(bvhSize);
	m_nodes.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
	//m_nodes.clearRange32(0, 0, bvhSize); // Mark all tasks as 0 (important for debug)
#ifdef COMPACT_LAYOUT
	m_triWoop.resizeDiscard(m_numTris *(S64)(Environment::GetSingleton()->GetFloat("PersistentSBVH.triRatio")) * (3+1) * sizeof(Vec4f));
	m_triIndex.resizeDiscard(m_numTris *(S64)(Environment::GetSingleton()->GetFloat("PersistentSBVH.idxRatio")) *  (3+1) * sizeof(S32));
#endif

	m_gpuTime = buildCuda();
	m_cpuTime = m_timer.end();

	// Resize to exact memory
	trimBuffers();

#ifdef DEBUG_PPS
	exit(0);
#endif

	return m_gpuTime;
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::updateConstants()
{
	RtEnvironment& cudaEnv = *(RtEnvironment*)m_module->getGlobal("c_env").getMutablePtr();

	Environment::GetSingleton()->GetIntValue("PersistentBVH.maxDepth", cudaEnv.optMaxDepth);

	Environment::GetSingleton()->GetFloatValue("PersistentBVH.ci", cudaEnv.optCi);

	Environment::GetSingleton()->GetFloatValue("PersistentBVH.ct", cudaEnv.optCt);

	Environment::GetSingleton()->GetIntValue("PersistentBVH.triLimit", cudaEnv.triLimit);

	Environment::GetSingleton()->GetIntValue("PersistentBVH.triMaxLimit", cudaEnv.triMaxLimit);

	Environment::GetSingleton()->GetIntValue("PersistentBVH.popCount", cudaEnv.popCount);

	Environment::GetSingleton()->GetFloatValue("PersistentBVH.granularity", cudaEnv.granularity);
	
	cudaEnv.epsilon = m_epsilon;
	//cudaEnv.epsilon = 0.f;
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::initPool(Buffer* nodeBuffer)
{
	// Prepare the task data
	updateConstants();

	// Set PPS buffers
	m_ppsTris.resizeDiscard(sizeof(int)*m_numTris);
	m_ppsTrisIndex.resizeDiscard(sizeof(int)*m_numTris);
	m_sortTris.resizeDiscard(sizeof(Reference)*m_numTris);

#if defined(SNAPSHOT_POOL) || defined(SNAPSHOT_WARP)
	// Prepare snapshot memory
	allocateSnapshots(m_module, m_snapData);
#endif

	// Set all headers empty
	m_taskData.setOwner(Buffer::Cuda, true); // Make CUDA the owner so that CPU memory is never allocated
#ifdef BENCHMARK
	m_taskData.clearRange32(0, TaskHeader_Empty, TASK_SIZE * sizeof(int)); // Mark all tasks as empty
#else
	m_taskData.clearRange32(0, TaskHeader_Empty, TASK_SIZE * (sizeof(int)+sizeof(TaskBVH))); // Mark all tasks as empty (important for debug)
#endif

	// Set texture references.
	if(nodeBuffer != NULL)
	{
		m_module->setTexRef("t_nodesA", *nodeBuffer, CU_AD_FORMAT_FLOAT, 4);
	}
	m_module->setTexRef("t_trisA", m_trisCompact, CU_AD_FORMAT_FLOAT, 4);
	//m_module->setTexRef("t_triIndices", m_trisCompactIndex, CU_AD_FORMAT_SIGNED_INT32, 1);
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::deinitPool()
{
	m_ppsTris.reset();
	m_ppsTrisIndex.reset();
	m_sortTris.reset();
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::printPoolHeader(TaskStackBase* tasks, int* header, int numWarps, FW::String state)
{
#if defined(SNAPSHOT_POOL) || defined(SNAPSHOT_WARP)
	printSnapshots(m_snapData);
#endif

#ifdef DEBUG_INFO
	Debug << "\nPRINTING DEBUG_INFO STATISTICS" << "\n\n";
#else
	Debug << "\nPRINTING STATISTICS" << "\n\n";
#endif

	float4* debugData = (float4*)m_debug.getPtr();
	float minAll[4] = {FLT_MAX, FLT_MAX, FLT_MAX, FLT_MAX};
	float maxAll[4] = {0, 0, 0, 0};
	float sumAll[4] = {0, 0, 0, 0};
	int countDead = 0;
	Debug << "Warp No. cnt_task_queues Avg. #Reads Max #Reads #Restarts" << "\n";
	for(int i = 0; i < numWarps; i++)
	{
		Debug << "Warp " << i << ": (" << debugData[i].x << ", " << debugData[i].y << ", " << debugData[i].z << ", " << debugData[i].w << ")" << "\n";

		//fabs is because we do not care whether the warp stopped prematurely or not
		minAll[0] = min(fabs(debugData[i].x), minAll[0]);
		minAll[1] = min(fabs(debugData[i].y), minAll[1]);
		minAll[2] = min(fabs(debugData[i].z), minAll[2]);
		minAll[3] = min(fabs(debugData[i].w), minAll[3]);

		maxAll[0] = max(fabs(debugData[i].x), maxAll[0]);
		maxAll[1] = max(fabs(debugData[i].y), maxAll[1]);
		maxAll[2] = max(fabs(debugData[i].z), maxAll[2]);
		maxAll[3] = max(fabs(debugData[i].w), maxAll[3]);

		sumAll[0] += fabs(debugData[i].x);
		sumAll[1] += fabs(debugData[i].y);
		sumAll[2] += fabs(debugData[i].z);
		sumAll[3] += fabs(debugData[i].w);

		if(debugData[i].x < 0)
			countDead++;
	}
	Debug << "Dead=" << countDead << " / All=" << numWarps << " = " << (float)countDead/(float)numWarps << "\n";
	Debug << "Min: " << minAll[0] << ", " << minAll[1] << ", " << minAll[2] << ", " << minAll[3] << "\n";
	Debug << "Max: " << maxAll[0] << ", " << maxAll[1] << ", " << maxAll[2] << ", " << maxAll[3] << "\n";
	Debug << "Sum: " << sumAll[0] << ", " << sumAll[1] << ", " << sumAll[2] << ", " << sumAll[3] << "\n";
	Debug << "Avg: " << sumAll[0]/numWarps << ", " << sumAll[1]/numWarps << ", " << sumAll[2]/numWarps << ", " << sumAll[3]/numWarps << "\n\n" << "\n";
	Debug << "cnt_task_queues per object = " << sumAll[0]/(float)m_numTris << "\n";

	Debug << "Pool" << "\n";
	Debug << "Top = " << tasks->top << "; Bottom = " << tasks->bottom << "; Unfinished = " << tasks->unfinished << "; Size = " << tasks->sizePool << "; ";
	Debug << state.getPtr() << "\n";
	Debug << "ActiveTop = " << tasks->activeTop << "; Active = ";
	for(int i = 0; i < ACTIVE_MAX+1; i++)
		Debug << tasks->active[i] << " ";
	Debug << "\n" << "\n";
	Debug << "EmptyTop = " << tasks->emptyTop << "; EmptyBottom = " << tasks->emptyBottom  << "\nEmpty\n";
	for(int i = 0; i < EMPTY_MAX+1; i++)
	{
		if(i % 50 == 0)
			Debug << "\n";
		else
			Debug << " ";
		Debug << tasks->empty[i];
	}

	Debug << "\n" << "\n";

	int emptyItems = 0;
	int bellowEmpty = 0;
	Debug << "Header" << "\n";
	for(int i = 0; i < TASK_SIZE; i++)
	{
		if(i % 50 == 0)
			Debug << "\n";
		else
			Debug << " ";
		if(header[i] != TaskHeader_Empty)
		{
			Debug << header[i];
		}
		else
		{
			Debug << TaskHeader_Active;
			if(i < tasks->top)
				emptyItems++;
		}

		if(header[i] < TaskHeader_Empty)
			bellowEmpty++;
	}

	Debug << "\n\nEmptyItems = " << emptyItems << "\n";
	Debug << "BellowEmpty = " << bellowEmpty << "\n";
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::printPool(TaskStackBVH &tasks, int numWarps)
{
#ifdef LEAF_HISTOGRAM
	printf("Leaf histogram\n");
	unsigned int leafSum = 0;
	unsigned int triSum = 0;
	for(S32 i = 0; i <= Environment::GetSingleton()->GetInt("PersistentBVH.triLimit"); i++)
	{
		printf("%d: %d\n", i, tasks.leafHist[i]);
		leafSum += tasks.leafHist[i];
		triSum += i*tasks.leafHist[i];
	}
	printf("Leafs total %d, average leaf %.2f\n", leafSum, (float)triSum/(float)leafSum);
#endif

	int* header = (int*)m_taskData.getPtr();
	FW::String state = sprintf("BVH Top = %d; Tri Top = %d; Warp counter = %d; ", tasks.nodeTop, tasks.triTop, tasks.warpCounter);
#ifdef COUNT_NODES
	state.appendf("Number of inner nodes = %d; Number of leaves = %d; Sorted tris = %d; ", tasks.numNodes, tasks.numLeaves, tasks.numSortedTris);
#endif
	printPoolHeader(&tasks, header, numWarps, state);

	Debug << "\n\nTasks" << "\n";
	TaskBVH* task = (TaskBVH*)m_taskData.getPtr(TASK_SIZE*sizeof(int));
	int stackMax = 0;
	int maxTaskId = -1;
	long double sumTris = 0;
	long double maxTris = 0;

	int sortTasks = 0;
	long double cntSortTris = 0;

	

#ifdef DEBUG_INFO
	int maxDepth = 0;
	int syncCount = 0;
	int subFailed = 0;
	char terminatedNames[TerminatedBy_Max][255] = {
		"None", "Depth","TotalLimit","OverheadLimit","Cost","FailureCounter"
	};

	int terminatedBy[TerminatedBy_Max];
	memset(&terminatedBy,0,sizeof(int)*TerminatedBy_Max);
#endif

	for(int i = 0; i < TASK_SIZE; i++)
	{
		if(task[i].nodeIdx != TaskHeader_Empty || task[i].parentIdx != TaskHeader_Empty)
		{
#ifdef DEBUG_INFO
			_ASSERT(task[i].terminatedBy >= 0 && task[i].terminatedBy < TerminatedBy_Max);
			terminatedBy[ task[i].terminatedBy ]++;
#endif

			Debug << "Task " << i << "\n";
			Debug << "Header: " << header[i] << "\n";
			Debug << "Unfinished: " << task[i].unfinished << "\n";
			Debug << "Type: " << task[i].type << "\n";
			Debug << "TriStart: " << task[i].triStart << "\n";
			Debug << "TriLeft: " << task[i].triLeft << "\n";
			Debug << "TriRight: " << task[i].triRight << "\n";
			Debug << "TriEnd: " << task[i].triEnd << "\n";
			Debug << "ParentIdx: " << task[i].parentIdx << "\n";
			Debug << "NodeIdx: " << task[i].nodeIdx << "\n";
			Debug << "TaskID: " << task[i].taskID << "\n";
			Debug << "Split: (" << task[i].splitPlane.x << ", " << task[i].splitPlane.y << ", " << task[i].splitPlane.z << ", " << task[i].splitPlane.w << ")\n";
			Debug << "Box: (" << task[i].bbox.m_mn.x << ", " << task[i].bbox.m_mn.y << ", " << task[i].bbox.m_mn.z << ") - ("
				<< task[i].bbox.m_mx.x << ", " << task[i].bbox.m_mx.y << ", " << task[i].bbox.m_mx.z << ")\n";
			Debug << "BoxLeft: (" << task[i].bboxLeft.m_mn.x << ", " << task[i].bboxLeft.m_mn.y << ", " << task[i].bboxLeft.m_mn.z << ") - ("
				<< task[i].bboxLeft.m_mx.x << ", " << task[i].bboxLeft.m_mx.y << ", " << task[i].bboxLeft.m_mx.z << ")\n";
			Debug << "BoxRight: (" << task[i].bboxRight.m_mn.x << ", " << task[i].bboxRight.m_mn.y << ", " << task[i].bboxRight.m_mn.z << ") - ("
				<< task[i].bboxRight.m_mx.x << ", " << task[i].bboxRight.m_mx.y << ", " << task[i].bboxRight.m_mx.z << ")\n";
			Debug << "Axis: " << task[i].axis << "\n";
			Debug << "Depth: " << task[i].depth << "\n";
			Debug << "Step: " << task[i].step << "\n";
#ifdef DEBUG_INFO
			//Debug << "Step: " << task[i].step << "\n";
			//Debug << "Lock: " << task[i].lock << "\n";
			Debug << "GMEMSync: " << task[i].sync << "\n";
			Debug << "Parent: " << task[i].parent << "\n";
#endif

#ifdef DEBUG_INFO
			Debug << "TerminatedBy: " << task[i].terminatedBy << "\n";
#endif
			if(task[i].terminatedBy != TerminatedBy_None)
				Debug << "Triangles: " << task[i].triEnd - task[i].triStart << "\n";

			Debug << "\n";
			stackMax = i;

			if(header[i] > (int)0xFF800000) // Not waiting
			{
				long double tris = task[i].triEnd - task[i].triStart;
				if(task[i].terminatedBy != TerminatedBy_None)
				{
					if(tris > maxTris)
					{
						maxTris = tris;
						maxTaskId = i;
					}
					sumTris += tris;
				}
				sortTasks++;
				cntSortTris += tris;

#ifdef DEBUG_INFO
				maxDepth = max(task[i].depth, maxDepth);
				syncCount += task[i].sync;
#endif
			}
		}
	}

	if(stackMax == TASK_SIZE-1)
		printf("\aIncomplete result!\n");
	Debug << "\n\n";

#ifdef DEBUG_INFO
	Debug << "Avg naive task height (tris) = " << sumTris/(long double)sortTasks << "\n";
	Debug << "Max naive task height (tris) = " << maxTris << ", taskId: " << maxTaskId << "\n";
	Debug << "Cnt sorted operations = " << sortTasks << "\n";
	double cntTrisLog2Tris = (double(m_numTris) * (double)(logf(m_numTris)/logf(2.0f)));
	Debug << "Cnt sorted triangles = " << cntSortTris << "\n";	
	Debug << "Cnt sorted triangles/(N log N), N=#tris = " << cntSortTris/cntTrisLog2Tris << "\n";
	Debug << "\n";
	Debug << "Max task depth = " << maxDepth << "\n";
	Debug << "Cnt gmem synchronizations: " << syncCount << "\n";
	Debug << "Leafs failed to subdivide = " << subFailed << " (*3) => total useless tasks " << subFailed * 3 << "\n";
	Debug << "Terminated by:" << "\n";
	for(int i = 0; i < TerminatedBy_Max; i++)
	{
		Debug << terminatedNames[i] << ": " << terminatedBy[i] << "\n";
	}
#endif

	Debug << "max_queue_length = " << stackMax << "\n\n" << "\n";
}

//------------------------------------------------------------------------

F32 CudaPersistentSBVHBuilder::buildCuda()
{
	CudaKernel kernel;
	kernel = m_module->getKernel("build");

	// Prepare the task data
	initPool();

	// Set input.
	KernelInputSBVH& in = *(KernelInputSBVH*)m_module->getGlobal("c_bvh_in").getMutablePtr();
	in.numTris		= m_numTris;
	in.tris         = m_trisCompact.getCudaPtr();
	in.refs			= m_refs.getMutableCudaPtr();
	//in.trisBox      = m_trisBox.getCudaPtr();
	in.ppsTrisBuf   = m_ppsTris.getMutableCudaPtr();
	in.ppsTrisIndex = m_ppsTrisIndex.getMutableCudaPtr();
	in.sortRefs     = m_sortTris.getMutableCudaPtr();
#ifdef COMPACT_LAYOUT
	in.trisOut      = m_triWoop.getMutableCudaPtr();
	in.trisIndexOut = m_triIndex.getMutableCudaPtr();
#endif
	int baseOffset = setDynamicMemory();

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
	SplitRed split;
	for(int i = 0; i < 2; i++)
	{
		split.children[i].bbox.m_mn = make_float3(FLT_MAX, FLT_MAX, FLT_MAX);
		split.children[i].bbox.m_mx = make_float3(-FLT_MAX, -FLT_MAX, -FLT_MAX);
		split.children[i].cnt = 0;
	}

	SplitArray sArray;
	for(int i = 0; i < NUM_WARPS; i++)
	{
		for(int j = 0; j < PLANE_COUNT; j++)
			sArray.splits[i][j] = split;
	}
#else
	SplitRed split;
	for(int i = 0; i < 2; i++)
	{
		//split.children[i].bbox.m_mn = make_float3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
		//split.children[i].bbox.m_mx = make_float3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
		split.children[i].bbox.m_mn = make_int3(floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX), floatToOrderedInt(FLT_MAX));
		split.children[i].bbox.m_mx = make_int3(floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX), floatToOrderedInt(-FLT_MAX));
		split.children[i].cnt = 0;
	}

	SplitArray sArray;
	for(int j = 0; j < PLANE_COUNT; j++)
	{
		sArray.splits[j] = split;
		sArray.spatialSplits[j] = split;
	}

	m_splitData.setRange(0, &sArray, sizeof(SplitArray)); // Set the first splits
#endif

	m_splitData.setRange(TASK_SIZE * sizeof(SplitArray), &sArray, sizeof(SplitArray)); // Set the last splits for copy
#endif

	CudaAABB bbox;
	memcpy(&bbox.m_mn, &m_bboxMin, sizeof(float3));
	memcpy(&bbox.m_mx, &m_bboxMax, sizeof(float3));

	// Set parent task containing all the work
	TaskBVH all;
	all.triStart     = 0;
	all.triLeft      = 0;
	//all.triRight     = m_numTris;
	all.triRight	 = 0;
	all.triEnd       = m_numTris;
	all.bbox         = bbox;
	all.step         = 0;
	all.lock         = LockType_Free;
	all.bestCost     = 1e38f;
	all.depth        = 0;
	all.dynamicMemory= baseOffset;
	all.triIdxCtr    = 0;
	all.parentIdx    = -1;
	all.nodeIdx      = 0;
	all.taskID       = 0;
	Vec3f size       = m_bboxMax - m_bboxMin;
	all.axis         = size.x > size.y ? (size.x > size.z ? 0 : 2) : (size.y > size.z ? 1 : 2);
	all.terminatedBy = TerminatedBy_None;
#ifdef DEBUG_INFO
	all.sync         = 0;
	all.parent       = -1;
	all.clockStart   = 0;
	all.clockEnd     = 0;
#endif

#if SPLIT_TYPE == 0
#if SCAN_TYPE == 0
	all.type         = TaskType_Sort_PPS1;
#elif SCAN_TYPE == 1
	all.type         = TaskType_Sort_PPS1_Up;
#elif SCAN_TYPE == 2 ||  SCAN_TYPE == 3
	all.type         = TaskType_Sort_SORT1;
#endif
	all.unfinished   = warpSubtasks(m_numTris);
	float pos = m_bbox.min[all.axis] + m_bbox.Size(all.axis)/2.0f;
	if(all.axis == 0)
		all.splitPlane   = make_float4(1.f, 0.f, 0.f, -pos);
	else if(all.axis == 1)
		all.splitPlane   = make_float4(0.f, 1.f, 0.f, -pos);
	else
		all.splitPlane   = make_float4(0.f, 0.f, 1.f, -pos);
#elif SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
#if BINNING_TYPE == 0 || BINNING_TYPE == 1
	all.type         = TaskType_InitMemory;
	all.unfinished   = warpSubtasks(sizeof(SplitArray)/sizeof(int));
#else
	all.type         = TaskType_BinTriangles;
	all.unfinished   = (warpSubtasks(m_numTris)+BIN_MULTIPLIER-1)/BIN_MULTIPLIER;
	/*all.type         = TaskType_BuildObjectSAH;
	all.unfinished   = 1;*/
#endif
#endif
	all.origSize     = all.unfinished;

	m_taskData.setRange(TASK_SIZE * sizeof(int), &all, sizeof(TaskBVH)); // Set the first task

	// Set parent task header
	m_taskData.setRange(0, &all.unfinished, sizeof(int)); // Set the first task

	// Prepare the task stack
	TaskStackBVH& tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getMutablePtr();
	tasks.header     = (int*)m_taskData.getMutableCudaPtr();
	tasks.tasks      = (TaskBVH*)m_taskData.getMutableCudaPtr(TASK_SIZE * sizeof(int));
	tasks.nodeTop    = 1;
	tasks.triTop     = 0;
	tasks.top        = 0;
	tasks.bottom     = 0;
	//memset(tasks.active, 0, sizeof(int)*(ACTIVE_MAX+1));
	memset(tasks.active, -1, sizeof(int)*(ACTIVE_MAX+1));
	tasks.active[0] = 0;
	/*for(int i = 0; i < ACTIVE_MAX+1; i++)
	tasks.active[i] = i;*/
	tasks.activeTop = 1;
	//tasks.empty[0] = 0;
	//int j = 1;
	//for(int i = EMPTY_MAX; i > 0; i--, j++)
	//	tasks.empty[i] = j;
	memset(tasks.empty, 0, sizeof(int)*(EMPTY_MAX+1));
	tasks.emptyTop = 0;
	tasks.emptyBottom = 0;
	tasks.unfinished = -1; // We are waiting for one task to finish = task all
	tasks.numSortedTris = 0;
	tasks.numNodes = 0;
	tasks.numLeaves = 0;
	tasks.sizePool = TASK_SIZE;
	tasks.sizeNodes = int(m_nodes.getSize()/sizeof(CudaBVHNode));
	tasks.sizeTris = int(m_triIndex.getSize()/sizeof(S32));
	memset(tasks.leafHist, 0, sizeof(tasks.leafHist));

#if SPLIT_TYPE >= 4 && SPLIT_TYPE <= 6
	// Prepare split stack
	SplitArray* &splits = *(SplitArray**)m_module->getGlobal("g_redSplits").getMutablePtr();
	splits = (SplitArray*)m_splitData.getMutableCudaPtr();
#endif

	CudaBVHNode* &bvh = *(CudaBVHNode**)m_module->getGlobal("g_bvh").getMutablePtr();
	bvh = (CudaBVHNode*)m_nodes.getMutableCudaPtr();

	// Determine block and grid sizes.
#ifdef ONE_WARP_RUN
	Vec2i blockSize(WARP_SIZE, 1); // threadIdx.x must equal the thread lane in warp
	Vec2i gridSize(1, 1); // Number of SMs * Number of blocks?
	int numWarps = 1;
#else
	int numWarpsPerBlock = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numWarpsPerBlock");
	int numBlocksPerSM = Environment::GetSingleton()->GetInt("SubdivisionRayCaster.numBlockPerSM");
	Vec2i blockSize(WARP_SIZE, numWarpsPerBlock); // threadIdx.x must equal the thread lane in warp
	int gridSizeX = NUM_SM*numBlocksPerSM;
	Vec2i gridSize(gridSizeX, 1); // Number of SMs * Number of blocks?

	if(gridSizeX*numWarpsPerBlock != NUM_WARPS)
		printf("\aNUM_WARPS constant does not match the launch parameters\n");
#endif

	m_debug.resizeDiscard(blockSize.y*gridSize.x*sizeof(float4));
	m_debug.clear();
	in.debug = m_debug.getMutableCudaPtr();

	// Launch.
	kernel.setGridExact(blockSize, gridSize);
	//cudaProfilerStart();
	float tKernel = kernel.launchTimed();
	//cudaProfilerStop();

#ifndef BENCHMARK
	cuCtxSynchronize(); // Flushes printfs
#endif

#ifdef DEBUG_PPS
	pout = (S32*)m_ppsTris.getPtr();
	sout = (S32*)m_sortTris.getPtr();
	S32 sum = 0;
	S32 error = 0;
	int j = 0;
	for(int i=0;i<m_numTris;i++)
	{
		sum += *sout; // Here for inclusive scan
		if(*pout != sum)
		{
			cout << "PPS error for item " << i << ", CPU=" << sum << ", GPU=" << *pout << " for " << m_numTris << " triangles!" << "\n";
			error = 1;
			if(j == 10)
				break;
			j++;
		}
		if(*sout != 0 && *sout != 1)
		{
			cout << "\nWTF " << i << " of " << m_numTris << ": " << *sout << "!\n" << "\n";
			break;
		}
		//sum += *sout; // Here for exclusive scan
		pout++;
		sout++;
	}

	if(!error)
		cout << "PPS correct for " << m_numTris << " triangles!" << "\n";
	return 0;
#endif

	tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();
	if(tasks.unfinished != 0 || tasks.top >= tasks.sizePool || tasks.nodeTop >= m_nodes.getSize() / sizeof(CudaBVHNode) || tasks.triTop >= m_triIndex.getSize() / sizeof(S32)) // Something went fishy
	{
		tKernel = 1e38f;
		fail("%d (%d x %d) (%d x %d) (%d x %d)\n", tasks.unfinished != 0, tasks.top, tasks.sizePool, tasks.nodeTop, m_nodes.getSize() / sizeof(CudaBVHNode), tasks.triTop, m_triIndex.getSize() / sizeof(S32));
	}

	//Debug << "\nBuild in " << tKernel << "\n\n";

#ifndef BENCHMARK
	int numWarps = numWarpsPerBlock*gridSizeX;
	printPool(tasks, numWarps);

	/*Debug << "\n\nBVH" << "\n";
	CudaBVHNode* nodes = (CudaBVHNode*)m_nodes.getPtr();

	for(int i = 0; i < tasks.nodeTop; i++)
	{
		Debug << "Node " << i << "\n";
		Debug << "BoxLeft: (" << nodes[i].c0xy.x << ", " << nodes[i].c0xy.z << ", " << nodes[i].c01z.x << ") - ("
				<< nodes[i].c0xy.y << ", " << nodes[i].c0xy.w << ", " << nodes[i].c01z.y << ")\n";
		Debug << "BoxRight: (" << nodes[i].c1xy.x << ", " << nodes[i].c1xy.z << ", " << nodes[i].c01z.z << ") - ("
				<< nodes[i].c1xy.y << ", " << nodes[i].c1xy.w << ", " << nodes[i].c01z.w << ")\n";
		Debug << "Children: " << nodes[i].children.x << ", " << nodes[i].children.y << "\n\n";
	}*/

	// Free data
	deinitPool();
#endif

	return tKernel;
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::saveBufferSizes(bool ads, bool aux)
{
	float MB = (float)(1024*1024);

	if(ads)
	{
		m_sizeADS    = m_nodes.getSize()/MB;
#ifndef COMPACT_LAYOUT
		m_sizeTri    = m_trisCompact.getSize()/MB;
		m_sizeTriIdx = m_trisCompactIndex.getSize()/MB;
#else
		m_sizeTri    = m_triWoop.getSize()/MB;
		m_sizeTriIdx = m_triIndex.getSize()/MB;
#endif
	}

	if(aux)
	{
		m_sizeTask   = m_taskData.getSize()/MB;
		m_sizeSplit  = m_splitData.getSize()/MB;
		m_heap       = 0.f;
	}
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::resetBuffers(bool resetADSBuffers)
{
	// Reset buffers so that reuse of space does not cause timing disturbs
	if(resetADSBuffers)
	{
		m_nodes.reset();
		m_triWoop.reset();
		m_triIndex.reset();
	}

	m_taskData.reset();
	m_splitData.reset();

	m_ppsTris.reset();
	m_ppsTrisIndex.reset();
	m_sortTris.reset();
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::trimBuffers()
{
	// Save sizes of auxiliary buffers so that they can be printed
	saveBufferSizes(false, true);
	// Free auxiliary buffers
	resetBuffers(false);

	// Resize to exact memory
	U32 nN, nL, sT, bT, tT, sTr; 
	getStats(nN, nL, sT, bT, tT, sTr);
#ifdef COMPACT_LAYOUT
	m_nodes.resize(nN * sizeof(CudaBVHNode));
	m_triWoop.resize(tT*3*sizeof(float4) + nL*sizeof(float4));
	m_triIndex.resize(tT*3*sizeof(int) + nL*sizeof(int));
#else
	m_nodes.resize((nN + nL) * sizeof(CudaBVHNode));
#endif

	// Save sizes of ads buffers so that they can be printed
	saveBufferSizes(true, false);
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::getStats(U32& nodes, U32& leaves, U32& stackTop, U32& nodeTop, U32& tris, U32& sortedTris, bool sub)
{
	TaskStackBVH tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();

#ifndef COUNT_NODES
#ifndef COMPACT_LAYOUT
	nodes = tasks.nodeTop / 2;
	leaves = tasks.nodeTop - nodes;
#else
	nodes = tasks.nodeTop;
	leaves = tasks.triTop;
#endif
#else // COUNT_NODES
	nodes = tasks.numNodes;
	leaves = tasks.numLeaves;
#endif // COUNT_NODES

#ifdef COMPACT_LAYOUT
	tris = tasks.triTop;
	if(sub)
		tris -= leaves;
#ifdef DUPLICATE_REFERENCES
	tris /= 3;
#endif
#else
	if(sub)
	{
		tris = m_numTris;
	}
	else
	{
		tris = tasks.triTop;
		tris /= 3;
	}
#endif

	nodeTop = tasks.nodeTop;
	sortedTris = tasks.numSortedTris;
	stackTop = tasks.top;
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::getSizes(F32& task, F32& split, F32& ads, F32& tri, F32& triIdx, F32& heap)
{
	task = m_sizeTask;
	split = m_sizeSplit;
	ads = m_sizeADS;
	tri = m_sizeTri;
	triIdx = m_sizeTriIdx;
	heap = m_heap;
}

//------------------------------------------------------------------------

void CudaPersistentSBVHBuilder::getAllocStats(U32& numAllocs, F32& allocSum, F32& allocSumSquare)
{
	TaskStackBVH tasks = *(TaskStackBVH*)m_module->getGlobal("g_taskStackBVH").getPtr();

#ifdef COUNT_NODES
	numAllocs = tasks.numAllocations;
	allocSum = tasks.allocSum;
	allocSumSquare = tasks.allocSumSquare;
#endif
}
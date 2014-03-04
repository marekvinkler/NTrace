#include <thrust/sort.h>
#include <thrust/unique.h>
#include <thrust/iterator/discard_iterator.h>
#include "radixSort.hpp"

//#include <thrust/detail/device/cuda/detail/b40c/radixsort_api.h>
//#include <b40c/radix_sort/enactor.cuh>
//#include <b40c/util/multiple_buffering.cuh>

//#include <radixsort_single_grid.cu>
//#include <radixsort_early_exit.cu>
//#include <radixsort_early_exit.cu>
//#include <radixsort_multi_cta.cu>
//#include <cuda/cutil.h>						// Utilities for commandline parsing

//using namespace b40c;

//using namespace FW;

float radixSortCuda(CUdeviceptr keys, CUdeviceptr values, int n) {
	cudaEvent_t start = NULL, stop = NULL;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	
	//cudaEventRecord(start, NULL);

	thrust::device_ptr<unsigned int> d_keys((unsigned int*)keys);
	thrust::device_ptr<int> d_values((int*)values);
	
	cudaEventRecord(start, NULL);
  
	thrust::sort_by_key(d_keys, d_keys + n, d_values);

	//cudaDeviceSynchronize();
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);

	float time;
	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	return time * 1.0e-3f;
};

struct pred : public thrust::binary_function<unsigned int, unsigned int, bool> {
	int d;
	pred(int dd) {
		d = dd;
	}

	__host__ __device__
	bool operator()(const unsigned int a, const unsigned int b) {
		return ((a >> d) == (b >> d));
	}
};

struct predFalse : public thrust::binary_function<unsigned int, unsigned int, bool> {
	__host__ __device__
	bool operator()(const unsigned int a, const unsigned int b) {
		return false;
	}
};

struct pred2 {
	__device__
	bool operator()(const unsigned int a, const unsigned int b) {
		return (b - a == 1);
	}
};

float createClusters(CUdeviceptr values, int n, int d, CUdeviceptr out, int &out_cnt) {
	cudaEvent_t start = NULL, stop = NULL;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);	

	thrust::device_ptr<unsigned int> d_values((unsigned int*)values);
	thrust::device_ptr<unsigned int> d_out((unsigned int*)out), d_out_cnt;

	thrust::pair<thrust::discard_iterator<>, thrust::device_ptr<unsigned int>> d_ret;
	
	cudaEventRecord(start, NULL);

	if(d != 0)
	{
		pred bin(d);
		d_ret = thrust::unique_by_key_copy(d_values, d_values + n, thrust::make_counting_iterator(0), thrust::make_discard_iterator(), d_out, bin);	
	}
	else
	{
		predFalse bin;
		d_ret = thrust::unique_by_key_copy(d_values, d_values + n, thrust::make_counting_iterator(0), thrust::make_discard_iterator(), d_out, bin);	
	}
	
	//pred2 bin2;
	//d_out_cnt = thrust::unique(d_out, d_ret.second, bin2);
	
	//cudaDeviceSynchronize();
	cudaEventRecord(stop, NULL);
	cudaEventSynchronize(stop);
	
	float time;
	cudaEventElapsedTime(&time, start, stop);

	cudaEventDestroy(start);
	cudaEventDestroy(stop);

	//out_cnt = d_out_cnt - d_out;
			
	out_cnt = d_ret.second - d_out;
	
	thrust::host_vector<unsigned int> h_last(1);
	h_last[0] = n;
	d_out[out_cnt] = h_last[0];
	//d_out[out_cnt] = n;
	
	return time * 1.0e-3f;
}
/*
#define LOWER_BITS 30
float radixSortBack40(CUdeviceptr keys, CUdeviceptr values, int n) {
	thrust::detail::device::cuda::detail::b40c_thrust::RadixSortStorage<unsigned int, unsigned int> device_storage((unsigned int*)keys, (unsigned int*)values);			
 
 	thrust::detail::device::cuda::detail::b40c_thrust::RadixSortingEnactor sorter<unsigned int, unsigned int>(n);
 	thrust::detail::device::cuda::detail::b40c_thrust::sorter.EnactSort(device_storage);
 
	// Re-acquire pointer to sorted keys and values, free unused/temp storage 
 	//d_int_keys = device_storage.d_keys;
 	//d_double_values = device_storage.d_values;
 	device_storage.CleanupTempStorage();


	//b40c::DeviceInit(args);

	// Create a scan enactor
//	b40c::radix_sort::Enactor enactor;
	//b40c::util::DoubleBuffer<unsigned int, unsigned int> sort_storage((unsigned int*)keys, (unsigned int*)values);

	//printf("selector = %d\n", sort_storage.selector);

	//cudaMemcpy(sort_storage.d_keys[sort_storage.selector], keys, sizeof(unsigned int) * n, cudaMemcpyDeviceToDevice);
	//cudaMemcpy(sort_storage.d_values[sort_storage.selector], values, sizeof(unsigned int) * n, cudaMemcpyDeviceToDevice);

	//enactor.Sort(sort_storage, n);
	//enactor.Sort<b40c::radix_sort::SMALL_SIZE>(sort_storage, n);
	//enactor.Sort<0, LOWER_BITS, b40c::radix_sort::SMALL_SIZE>(sort_storage, n);


	/*
	// Allocate device storage   
	MultiCtaRadixSortStorage<unsigned int, unsigned int> device_storage(n);	
	//cudaMalloc((void**) &device_storage.d_keys[0], sizeof(unsigned int) * n);
	//cudaMalloc((void**) &device_storage.d_values[0], sizeof(unsigned int) * n);

	// Create sorting enactor
	EarlyExitRadixSortingEnactor<unsigned int, unsigned int> sorting_enactor;

	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	device_storage.d_keys[0] = (unsigned int*)keys;
	device_storage.d_values[0] = (unsigned int*)values;

	sorting_enactor.EnactSort(device_storage);

	// Perform the timed number of sorting iterations
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	// Start cuda timing record
	cudaEventRecord(start, 0);
	
	// Call the sorting API routine
	sorting_enactor.EnactSort(device_storage);

	// End cuda timing record
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float time = cudaEventElapsedTime(&time, start, stop);
	return time * 1.0e-3f;

	// Display timing information
	/*
    // Copy out keys 
    cudaMemcpy(
    	h_keys, 
    	device_storage.d_keys[device_storage.selector], 
    	sizeof(K) * num_elements, 
    	cudaMemcpyDeviceToHost);
    
    // Free allocated memory
    if (device_storage.d_keys[0]) cudaFree(device_storage.d_keys[0]);
    if (device_storage.d_keys[1]) cudaFree(device_storage.d_keys[1]);
    if (device_storage.d_values[0]) cudaFree(device_storage.d_values[0]);
    if (device_storage.d_values[1]) cudaFree(device_storage.d_values[1]);

    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	*/
//}



/*
template <typename K, typename V, int LOWER_KEY_BITS> 
void SmallProblemTimedSort(
	unsigned int num_elements, 
	K *h_keys,
	K *h_reference_keys,
	unsigned int iterations)
{
	printf("Single-kernel, small-problem key-value sort, %d iterations, %d elements", iterations, num_elements);
	
	// Allocate device storage   
	MultiCtaRadixSortStorage<K, V> device_storage(num_elements);	
	cudaMalloc((void**) &device_storage.d_keys[0], sizeof(K) * num_elements);
	cudaMalloc((void**) &device_storage.d_values[0], sizeof(V) * num_elements);

	// Create sorting enactor
	SingleGridRadixSortingEnactor<K, V> sorting_enactor;

	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	cudaMemcpy(
		device_storage.d_keys[0], 
		h_keys, 
		sizeof(K) * num_elements, 
		cudaMemcpyHostToDevice);											// copy keys
	sorting_enactor.template EnactSort<LOWER_KEY_BITS>(device_storage);		// sort

	// Perform the timed number of sorting iterations

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < iterations; i++) {

		RADIXSORT_DEBUG = (i == 0);

		// Move a fresh copy of the problem into device storage
		cudaMemcpy(
			device_storage.d_keys[0], 
			h_keys, 
			sizeof(K) * num_elements, 
			cudaMemcpyHostToDevice);										// copy keys

		// Start cuda timing record
		cudaEventRecord(start_event, 0);

		// Call the sorting API routine
		sorting_enactor.template EnactSort<LOWER_KEY_BITS>(device_storage);	// sort

		// End cuda timing record
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&duration, start_event, stop_event);
		elapsed += (double) duration;		
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0; 
    printf(", %f GPU ms, %f x10^9 elts/sec\n", 
		avg_runtime,
		throughput);
	
    // Copy out keys 
    cudaMemcpy(
    	h_keys, 
    	device_storage.d_keys[device_storage.selector], 
    	sizeof(K) * num_elements, 
    	cudaMemcpyDeviceToHost);
    
    // Free allocated memory
    if (device_storage.d_keys[0]) cudaFree(device_storage.d_keys[0]);
    if (device_storage.d_keys[1]) cudaFree(device_storage.d_keys[1]);
    if (device_storage.d_values[0]) cudaFree(device_storage.d_values[0]);
    if (device_storage.d_values[1]) cudaFree(device_storage.d_values[1]);

    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	
	// Display sorted key data
	if (g_verbose) {
		printf("\n\nKeys:\n");
		for (int i = 0; i < num_elements; i++) {	
			PrintValue<K>(h_keys[i]);
			printf(", ");
		}
		printf("\n\n");
	}	
	
    // Verify solution
	VerifySort<K>(h_keys, h_reference_keys, num_elements, true);
	printf("\n");
	fflush(stdout);
}
*/
/*
template <typename K, typename V, int LOWER_KEY_BITS> 
void LargeProblemTimedSort(
	unsigned int num_elements, 
	K *h_keys,
	K *h_reference_keys,
	unsigned int iterations)
{
	printf("Early-exit key-value sort, %d iterations, %d elements", iterations, num_elements);
	
	// Allocate device storage   
	MultiCtaRadixSortStorage<K, V> device_storage(num_elements);	
	cudaMalloc((void**) &device_storage.d_keys[0], sizeof(K) * num_elements);
	cudaMalloc((void**) &device_storage.d_values[0], sizeof(V) * num_elements);

	// Create sorting enactor
	EarlyExitRadixSortingEnactor<K, V> sorting_enactor;

	// Perform a single sorting iteration to allocate memory, prime code caches, etc.
	cudaMemcpy(
		device_storage.d_keys[0], 
		h_keys, 
		sizeof(K) * num_elements, 
		cudaMemcpyHostToDevice);		// copy keys
	sorting_enactor.EnactSort(device_storage);

	// Perform the timed number of sorting iterations

	cudaEvent_t start_event, stop_event;
	cudaEventCreate(&start_event);
	cudaEventCreate(&stop_event);

	double elapsed = 0;
	float duration = 0;
	for (int i = 0; i < iterations; i++) {

		RADIXSORT_DEBUG = (i == 0);

		// Move a fresh copy of the problem into device storage
		cudaMemcpy(
			device_storage.d_keys[0], 
			h_keys, 
			sizeof(K) * num_elements, 
			cudaMemcpyHostToDevice);		// copy keys

		// Start cuda timing record
		cudaEventRecord(start_event, 0);

		// Call the sorting API routine
		sorting_enactor.EnactSort(device_storage);

		// End cuda timing record
		cudaEventRecord(stop_event, 0);
		cudaEventSynchronize(stop_event);
		cudaEventElapsedTime(&duration, start_event, stop_event);
		elapsed += (double) duration;		
	}

	// Display timing information
	double avg_runtime = elapsed / iterations;
	double throughput = ((double) num_elements) / avg_runtime / 1000.0 / 1000.0; 
    printf(", %f GPU ms, %f x10^9 elts/sec\n", 
		avg_runtime,
		throughput);
	
    // Copy out keys 
    cudaMemcpy(
    	h_keys, 
    	device_storage.d_keys[device_storage.selector], 
    	sizeof(K) * num_elements, 
    	cudaMemcpyDeviceToHost);
    
    // Free allocated memory
    if (device_storage.d_keys[0]) cudaFree(device_storage.d_keys[0]);
    if (device_storage.d_keys[1]) cudaFree(device_storage.d_keys[1]);
    if (device_storage.d_values[0]) cudaFree(device_storage.d_values[0]);
    if (device_storage.d_values[1]) cudaFree(device_storage.d_values[1]);

    // Clean up events
	cudaEventDestroy(start_event);
	cudaEventDestroy(stop_event);
	
	// Display sorted key data
	if (g_verbose) {
		printf("\n\nKeys:\n");
		for (int i = 0; i < num_elements; i++) {	
			PrintValue<K>(h_keys[i]);
			printf(", ");
		}
		printf("\n\n");
	}	
	
    // Verify solution
	VerifySort<K>(h_keys, h_reference_keys, num_elements, true);
	printf("\n");
	fflush(stdout);
}

*/
/**
 * Creates an example sorting problem whose keys is a vector of the specified 
 * number of elements having the specfied number of valid bits, and then 
 * dispatches the problem to the GPU for the given number of iterations, 
 * displaying runtime information.
 *
 * @param[in] 		iterations  
 * 		Number of times to invoke the GPU sorting primitive
 * @param[in] 		num_elements 
 * 		Size in elements of the vector to sort
 * @param[in]		use_small_problem_enactor
 */
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"
#define THREADS_PER_BLOCK 512 
extern float toBW(int bytes, float sec);


/* Helper function to round up to a power of 2. 
 */
static inline int nextPow2(int n)
{
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}
// perform first parallel scan from input
/*__global__ void
first_upward_scan(int* input, int* result, int upper_bound) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    size_t j = index * 2;
    if (j >= (size_t) upper_bound) return;
    result[j] = input[j];
    if (j + 1 >= upper_bound) return;
    result[j + 1] = input[j] + input[j + 1];
}*/

__global__ void
first_upward_scan(int* input, int* result, int upper_bound) {
    size_t even_index = 2*(blockIdx.x * blockDim.x + threadIdx.x);
    size_t odd_index = even_index + 1;

    if (even_index >= (size_t) upper_bound) return;
    if (odd_index >= (size_t) upper_bound) {
        result[even_index] = input[even_index];
        return;
    }

    int x = input[even_index];
    int y = input[odd_index];
    result[even_index] = x;
    result[odd_index] = x + y;
}

// perform parallel scan in place
__global__ void
upward_scan(int* result, int two_d, int two_dplus1, int upper_bound) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t) upper_bound / two_dplus1) return;
    size_t j = index * two_dplus1;
    result[j + two_dplus1 - 1] += result[j + two_d - 1];
}

__global__ void
reset_last(int* result, int N) {
    result[N - 1] = 0;
}

__global__ void
downward_scan(int* result, int two_d, int two_dplus1, int upper_bound) {
    size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (size_t) upper_bound / two_dplus1) return;
    size_t j = index * two_dplus1;
    int t = result[j + two_d - 1];
    result[j + two_d - 1] = result[j + two_dplus1 - 1];
    result[j + two_dplus1 - 1] += t;
}

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel segmented scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep input
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.

    // change N to the rounded length
    int rounded_length = nextPow2(N);

    // perform first upward sweep that also copies values
    const int num_threads_required = (N + 1) / 2;
    const int first_sweep_blocks = (num_threads_required + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    first_upward_scan<<<first_sweep_blocks, THREADS_PER_BLOCK>>>(input, result, N);
    cudaDeviceSynchronize();

    // upward sweep
    for (int two_d = 2; two_d < rounded_length / 2; two_d *= 2) {
        const int two_dplus1 = 2 * two_d;
        const int num_threads_required = rounded_length / two_dplus1;
        const int num_blocks = (num_threads_required + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        upward_scan<<<num_blocks, THREADS_PER_BLOCK>>>(result, two_d, two_dplus1, rounded_length);
        cudaDeviceSynchronize();
    }

    reset_last<<<1, 1>>>(result, rounded_length);
    cudaDeviceSynchronize();

    // downward sweep
    for (int two_d = rounded_length/2; two_d >= 1; two_d /= 2) {
        const int two_dplus1 = 2 * two_d;
        const int num_threads_required = rounded_length / two_dplus1;
        const int num_blocks = (num_threads_required + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        downward_scan<<<num_blocks, THREADS_PER_BLOCK>>>(result, two_d, two_dplus1, rounded_length);
        cudaDeviceSynchronize();
    }
}


/* This function is a wrapper around the code you will write - it copies the
 * input to the GPU and times the invocation of the exclusive_scan() function
 * above. You should not modify it.
 */
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input; 
    // We round the array sizes up to a power of 2, but elements after
    // the end of the original input are left uninitialized and not checked
    // for correctness. 
    // You may have an easier time in your implementation if you assume the 
    // array's length is a power of 2, but this will result in extra work on
    // non-power-of-2 inputs.
    int rounded_length = nextPow2(end - inarray);
    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);
    cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    // For convenience, both the input and output vectors on the device are
    // initialized to the input values. This means that you are free to simply
    // implement an in-place scan on the result vector if you wish.
    // If you do this, you will need to keep that fact in mind when calling
    // exclusive_scan from find_repeats.
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, end - inarray, device_result);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    return overallDuration;
}

/* Wrapper around the Thrust library's exclusive scan function
 * As above, copies the input onto the GPU and times only the execution
 * of the scan itself.
 * You are not expected to produce competitive performance to the
 * Thrust version.
 */
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);
    
    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int),
               cudaMemcpyDeviceToHost);
    thrust::device_free(d_input);
    thrust::device_free(d_output);
    double overallDuration = endTime - startTime;
    return overallDuration;
}

int find_repeats(int *device_input, int length, int *device_output) {
    /* Finds all pairs of adjacent repeated elements in the list, storing the
     * indices of the first element of each pair (in order) into device_result.
     * Returns the number of pairs found.
     * Your task is to implement this function. You will probably want to
     * make use of one or more calls to exclusive_scan(), as well as
     * additional CUDA kernel launches.
     * Note: As in the scan code, we ensure that allocated arrays are a power
     * of 2 in size, so you can use your exclusive_scan function with them if 
     * it requires that. However, you must ensure that the results of
     * find_repeats are correct given the original length.
     */    
    return 0;
}

/* Timing wrapper around find_repeats. You should not modify this function.
 */
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);
    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), 
               cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();
    
    int result = find_repeats(device_input, length, device_output);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    *output_length = result;

    cudaMemcpy(output, device_output, length * sizeof(int),
               cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    return endTime - startTime;
}

void printCudaInfo()
{
    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n"); 
}

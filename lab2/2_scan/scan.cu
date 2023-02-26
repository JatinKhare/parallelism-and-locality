#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include "CycleTimer.h"

//#define DEBUG

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

__global__ void
upsweep_kernel(int* x, int twod, int twod1, int N, int rounded_length) {
    // calculate the thread index
    int index = blockIdx.x * blockDim.x * twod1 + threadIdx.x * twod1;
    // if in the provided bound
    if (index < N){
       x[index+twod1-1] += x[index+twod-1];
    }
    __syncthreads();
    if(index == 0)
        x[rounded_length] = 0;
}

__global__ void downsweep_kernel(int* x, int twod, int twod1, int N) {
    int index = blockIdx.x * blockDim.x * twod1 + threadIdx.x * twod1;
    __syncthreads();
    if (index < N) {
        int t = x[index+twod-1];   
        x[index+twod-1] = x[index+twod1-1];
        x[index+twod1-1] += t; 
    }
}

void exclusive_scan(int* device_start, int length, int* device_result)
{
    /* Fill in this function with your exclusive scan implementation.
     * You are passed the locations of the input and output in device memory,
     * but this is host code -- you will need to declare one or more CUDA 
     * kernels (with the __global__ decorator) in order to actually run code
     * in parallel on the GPU.
     * Note you are given the real length of the array, but may assume that
     * both the input and the output arrays are sized to accommodate the next
     * power of 2 larger than the input.
     */
    int threadsPerBlock = 128; //this had the best performance

    int rounded_length = nextPow2(length);
    int blocksPerGrid;

    //Upsweep Phase
    for (int twod = 1; twod < rounded_length; twod*=2){
        int twod1 = twod*2;
        blocksPerGrid = ((std::ceil((float)length/twod1)) + threadsPerBlock - 1) / threadsPerBlock;
        upsweep_kernel<<<blocksPerGrid, threadsPerBlock>>>(device_result, twod, twod1, length, rounded_length-1);
        //cudaDeviceSynchronize();
    }
    //Upsweep Phase
    for (int twod = rounded_length/2; twod >= 1; twod /= 2){
        int twod1 = twod*2;
        blocksPerGrid = (std::ceil(rounded_length/twod1) + threadsPerBlock - 1) / threadsPerBlock;
        downsweep_kernel<<<blocksPerGrid, threadsPerBlock, 2*threadsPerBlock*sizeof(int)>>>(device_result, twod, twod1, length); 
        //cudaDeviceSynchronize();
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

    exclusive_scan(device_input, (end - inarray) , device_result);

    // Wait for any work left over to be completed.
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    
    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int),
               cudaMemcpyDeviceToHost);
    
    #ifdef DEBUG
        int limit = 100;
        fprintf(stderr,"i = ["); 
        for(int i = 0; i< limit; i++){
        if(i!=limit-1)
            fprintf(stderr,"%d, ",inarray[i]); 
        else
            fprintf(stderr,"%d]",inarray[i]); 
        }
        fprintf(stderr,"\n"); 

        fprintf(stderr,"A = ["); 
        for(int i = 0; i< limit; i++){
        if(i!=limit-1)
            fprintf(stderr,"%d, ",resultarray[i]); 
        else
            fprintf(stderr,"%d]",resultarray[i]); 
        }
        fprintf(stderr,"\n"); 
    #endif
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
__global__ void find_index_kernel(int* A, int length, int* indices) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length-1) {
        if (A[index] == A[index+1]) {
            indices[index] = 1;
        } else {
            indices[index] = 0;
        }
    }
}
/*__global__ void find_repeats_kernel(int* input, int n, int* prefix_sum, int* output) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
       if(thread == 0){            
        for(int index = 0; index <n-1;index++){
                     if (input[index] == input[index+1]) {
                        int location = prefix_sum[index];
                        output[location] = index;
                     }
            }
        }
}*/
__global__ void find_repeats_kernel(int* input, int length, int* prefix_sum, int* output) {
    int thread = blockIdx.x * blockDim.x + threadIdx.x;
        if(thread < length-1){
                     if (input[thread] == input[thread+1]) {
                        int location = prefix_sum[thread];
                        output[location] = thread;
                     }
            }
}
__global__ void count_kernel(int *input, int length, int *count){
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < length-1) {
        if (input[index] == input[index+1]) {
           atomicAdd(count, 1);
        }
    } 
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

    int threadsPerBlock = 1024;
    int threadBlocks = (length + threadsPerBlock - 1) / threadsPerBlock;
    int rounded_length = nextPow2(length);
    int count = 0;
    int *device_indices, *prefix_sum, *count_device;

    cudaMalloc((void **)&count_device, sizeof(int));
    cudaMalloc((void **)&device_indices, rounded_length * sizeof(int));
    cudaMalloc((void **)&prefix_sum, rounded_length * sizeof(int));
    cudaMemcpy(count_device, &count, sizeof(int), cudaMemcpyHostToDevice);
    // seperate kernel to find the total count. 
    count_kernel<<<threadBlocks, threadsPerBlock>>>(device_input, length, count_device);
    cudaMemcpy(&count, count_device, sizeof(int), cudaMemcpyDeviceToHost);

    //kernel to calculate the index array
    find_index_kernel<<<threadBlocks, threadsPerBlock>>>(device_input, length, device_indices);
    //finding the exclusive sum of the indices array
    exclusive_scan(device_indices, length, device_indices);
    find_repeats_kernel<<<threadBlocks, threadsPerBlock>>>(device_input, length, device_indices, device_output);

    return count;
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

    #ifdef DEBUG
        int limit = 16;
        fprintf(stderr,"i = ["); 
        for(int i = 0; i< limit; i++){
        if(i!=limit-1)
            fprintf(stderr,"%d, ",input[i]); 
        else
            fprintf(stderr,"%d]",input[i]); 
        }
        fprintf(stderr,"\n"); 

        fprintf(stderr,"A = ["); 
        for(int i = 0; i< limit; i++){
        if(i!=limit-1)
            fprintf(stderr,"%d, ",output[i]); 
        else
            fprintf(stderr,"%d]",output[i]); 
        }
        fprintf(stderr,"\n"); 
    #endif
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

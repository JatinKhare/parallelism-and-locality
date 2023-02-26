#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "saxpy.h"
__global__ void
saxpy_kernel(int N, float alpha, float* x, float* y, float* result) {

    // compute overall index from position of thread in current block,
    // and given the block we are in
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < N){
       result[index] = alpha * x[index] + y[index];
    }
}

static inline
int getBlocks(long working_set_size, int threadsPerBlock) {
  // TODO: implement and use this interface if necessary  
    return (working_set_size + threadsPerBlock - 1)/threadsPerBlock;
}

void 
getArrays(int size, float **xarray, float **yarray, float **resultarray) {
  // TODO: implement and use this interface if necessary  
    cudaMallocHost((void**)xarray, size*sizeof(float));
    cudaMallocHost((void**)yarray, size*sizeof(float));
    cudaMallocHost((void**)resultarray, size*sizeof(float));
    //*xarray = (float*)malloc(size*sizeof(float));
    //*yarray = (float*)malloc(size*sizeof(float));
    //*resultarray = (float*)malloc(size*sizeof(float)); 
}

void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
  // TODO: implement and use this interface if necessary  
  cudaFreeHost(xarray);
  cudaFreeHost(yarray);
  cudaFreeHost(resultarray);
}

void saxpyCuda(long total_elems, float alpha, float* xarray, float* yarray, float* resultarray, int partitions) {

    const int threadsPerBlock = 512; // change this if necessary

    float *device_x;
    float *device_y;
    float *device_result;

    // Allocate device memory buffers on the GPU
    cudaMalloc(&device_x, total_elems * sizeof(float));
    cudaMalloc(&device_y, total_elems * sizeof(float));
    cudaMalloc(&device_result, total_elems * sizeof(float));

    // Create CUDA streams
    double startTime = CycleTimer::currentSeconds();
    int StreamCount = partitions;
    cudaStream_t* streams = new cudaStream_t[StreamCount];

    for (int i = 0; i < StreamCount; i++) {
        cudaStreamCreate(&streams[i]);
    }

    int partitionSize = getBlocks(total_elems, partitions); 

    for (int i = 0; i < partitions; i++) {

        // Compute the size of the partition
        int start = i * partitionSize;
        int length = (i != partitions-1) ? partitionSize : total_elems - start;
        //Copy HtoD asynchronously
        cudaMemcpyAsync(device_x + start, xarray + start, length * sizeof(float), cudaMemcpyHostToDevice, streams[i%StreamCount]);
        cudaMemcpyAsync(device_y + start, yarray + start, length * sizeof(float), cudaMemcpyHostToDevice, streams[i%StreamCount]);

        int threadBlocks = getBlocks(length, threadsPerBlock);        
        // Run saxpy_kernel on the GPU with the appropriate stream
        saxpy_kernel<<<threadBlocks, threadsPerBlock, 0, streams[i%StreamCount]>>>(length, alpha, device_x + start, device_y + start, device_result + start);
        // Copy result from GPU with the appropriate stream
        cudaMemcpyAsync(resultarray + start, device_result + start, length * sizeof(float), cudaMemcpyDeviceToHost, streams[i%StreamCount]);
    }
    /*for (int i = 0; i < StreamCount; i++) {
        cudaStreamSynchronize(streams[i]);
    }*/
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg   += overallDuration;
    // Free memory buffers on the GPU and destroy streams
    cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);
    for (int i = 0; i < partitions; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}


void
printCudaInfo() {

    // for fun, just print out some stats on the machine

    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
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

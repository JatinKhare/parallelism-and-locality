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

    if (index < N)
       result[index] = alpha * x[index] + y[index];
}

static inline
int getBlocks(long working_set_size, int threadsPerBlock) {
    return (working_set_size + threadsPerBlock - 1)/threadsPerBlock;
  // TODO: implement and use this interface if necessary  
}

void 
getArrays(int size, float **xarray, float **yarray, float **resultarray) {
  // TODO: implement and use this interface if necessary  
}

void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
  // TODO: implement and use this interface if necessary  
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
    cudaStream_t* streams = new cudaStream_t[partitions];
    for (int i = 0; i < partitions; i++) {
        cudaStreamCreate(&streams[i]);
    }

    double startCopyH2Dtime, endCopyH2Dtime, startCopyD2Htime, endCopyD2Htime, startGPUTime, endGPUTime, timeKernel;
    // Start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();
    startCopyH2Dtime = CycleTimer::currentSeconds();  
    startGPUTime = CycleTimer::currentSeconds();

    for (int i = 0; i < partitions; i++) {
        // Compute the size of the partition
        int partition_size = total_elems / partitions;
        int partition_offset = i * partition_size;

        // Copy input arrays to the GPU using cudaMemcpyAsync with the appropriate stream
        //startCopyH2Dtime = CycleTimer::currentSeconds(); 
        cudaMemcpyAsync(&device_x[partition_offset], &xarray[partition_offset], partition_size * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        cudaMemcpyAsync(&device_y[partition_offset], &yarray[partition_offset], partition_size * sizeof(float), cudaMemcpyHostToDevice, streams[i]);
        //endCopyH2Dtime = CycleTimer::currentSeconds(); 
        //timeCopyH2DAvg += (endCopyH2Dtime - startCopyH2Dtime);
        // Compute number of blocks and threads per block
        int num_blocks = (partition_size + threadsPerBlock - 1) / threadsPerBlock;
        // Run saxpy_kernel on the GPU with the appropriate stream
        saxpy_kernel<<<num_blocks, threadsPerBlock, 0, streams[i]>>>(partition_size, alpha, &device_x[partition_offset], &device_y[partition_offset], &device_result[partition_offset]);

 
        // Copy result from GPU using cudaMemcpyAsync with the appropriate stream
        //startCopyD2Htime = CycleTimer::currentSeconds();
        cudaMemcpyAsync(&resultarray[partition_offset], &device_result[partition_offset], partition_size * sizeof(float), cudaMemcpyDeviceToHost, streams[i]);
        //endCopyD2Htime = CycleTimer::currentSeconds();
        //timeCopyD2HAvg += (endCopyD2Htime - startCopyD2Htime);
    }
    endCopyH2Dtime = CycleTimer::currentSeconds(); 
    timeCopyH2DAvg += (endCopyH2Dtime - startCopyH2Dtime);
    // Synchronize with all streams to ensure all operations have completed
    for (int i = 0; i < partitions; i++) {
        cudaStreamSynchronize(streams[i]);
    }
    endGPUTime = CycleTimer::currentSeconds();
    timeKernel += (endGPUTime - startGPUTime);
    // End timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg   += overallDuration;
    timeKernelAvg += timeKernel;
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

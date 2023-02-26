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
  // TODO: implement and use this interface if necessary  
  return (working_set_size + threadsPerBlock - 1)/threadsPerBlock;
}

void 
getArrays(int size, float **xarray, float **yarray, float **resultarray) {
  // TODO: implement and use this interface if necessary  
    cudaMallocManaged(xarray, size*sizeof(float));
    cudaMallocManaged(yarray, size*sizeof(float));
    cudaMallocManaged(resultarray, size*sizeof(float));
}

void 
freeArrays(float *xarray, float *yarray, float *resultarray) {
  // TODO: implement and use this interface if necessary
    cudaFree(xarray);
    cudaFree(yarray);
    cudaFree(resultarray);

}

void
saxpyCuda(long total_elems, float alpha, float* xarray, float* yarray, float* resultarray, int partitions) {
    const int threadsPerBlock = 512; // change this if necessary
    //float *device_x;
    //float *device_y;
    //float *device_result;
    // TODO: do we need to allocate device memory buffers on the GPU here? [ANSWER: NO]
    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();
    // TODO: do we need copy here? [ANSWER: NO]
    // TODO: insert time here to begin timing only the kernel
    double startGPUTime = CycleTimer::currentSeconds();
    // compute number of blocks and threads per block
    int threadBlocks = getBlocks(total_elems, threadsPerBlock); 
    // run saxpy_kernel on the GPU

    saxpy_kernel<<<threadBlocks, threadsPerBlock>>>(total_elems, alpha, xarray, yarray, resultarray);
    cudaDeviceSynchronize();

    double endGPUTime = CycleTimer::currentSeconds();
    double timeKernel = endGPUTime - startGPUTime; 

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    //
    // TODO: copy result from GPU using cudaMemcpy
    //
    // What would be copy time when we use UVM? [ANSWER: NO EXPLICIT COPYING IN UVM]
    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg   += overallDuration;
    timeKernelAvg += timeKernel;

    //
    // TODO free device memory if you allocate some device memory earlier in this function.
    //
    /*cudaFree(device_x);
    cudaFree(device_y);
    cudaFree(device_result);*/
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

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

void
saxpyCuda(long total_elems, float alpha, float* xarray, float* yarray, float* resultarray, int partitions) {

    const int threadsPerBlock = 512; // change this if necessary

    float *device_x;
    float *device_y;
    float *device_result;
    //
    // TODO: do we need to allocate device memory buffers on the GPU here?
    //
    cudaMallocManaged(&device_x, total_elems*sizeof(float));
    cudaMallocManaged(&device_y, total_elems*sizeof(float));
    cudaMallocManaged(&device_result, total_elems*sizeof(float));
    
    // start timing after allocation of device memory.
    double startTime = CycleTimer::currentSeconds();
    //
    // TODO: do we need copy here?
    //
    double startCopyO2Ntime = CycleTimer::currentSeconds();
    for(int i = 0; i< total_elems; i++){
      device_x[i] = xarray[i];
      device_y[i] = yarray[i];
    }
    double endCopyO2Ntime = CycleTimer::currentSeconds();
    //
    // TODO: insert time here to begin timing only the kernel
    //
    // compute number of blocks and threads per block
    int threadBlocks = getBlocks(total_elems, threadsPerBlock); 

    double startGPUTime = CycleTimer::currentSeconds();
    // run saxpy_kernel on the GPU
    saxpy_kernel<<<threadBlocks, threadsPerBlock>>>(total_elems, alpha, device_x, device_y, device_result);
    double endGPUTime = CycleTimer::currentSeconds();
    double timeKernel = endGPUTime - startGPUTime; 
    //
    // TODO: insert timer here to time only the kernel.  Since the
    // kernel will run asynchronously with the calling CPU thread, you
    // need to call cudaDeviceSynchronize() before your timer to
    // ensure the kernel running on the GPU has completed.  (Otherwise
    // you will incorrectly observe that almost no time elapses!)
    //
    // cudaDeviceSynchronize();
    cudaDeviceSynchronize();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
    
    //
    // TODO: copy result from GPU using cudaMemcpy
    //

    double startCopyN2Otime = CycleTimer::currentSeconds();
    for(int i = 0; i< total_elems; i++){
      resultarray[i] = device_result[i];
     }
    double endCopyN2Otime = CycleTimer::currentSeconds();

    // What would be copy time when we use UVM?

    double endTime = CycleTimer::currentSeconds();
    double overallDuration = endTime - startTime;
    totalTimeAvg   += overallDuration;
    timeKernelAvg += timeKernel;
    timeCopyH2DAvg += endCopyO2Ntime - startCopyO2Ntime;
    timeCopyD2HAvg += endCopyN2Otime - startCopyN2Otime;
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

#include <string>
#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "cudaRenderer.h"
#include "image.h"
#include "noise.h"
#include "sceneLoader.h"
#include "util.h"

#define WARP_SIZE 32

////////////////////////////////////////////////////////////////////////////////////////
// Putting all the cuda kernels here
///////////////////////////////////////////////////////////////////////////////////////

struct GlobalConstants {

    SceneName sceneName;

    int numCircles;
    float* position;
    float* velocity;
    float* color;
    float* radius;

    int imageWidth;
    int imageHeight;
    float* imageData;
};

// Global variable that is in scope, but read-only, for all cuda
// kernels.  The __constant__ modifier designates this variable will
// be stored in special "constant" memory on the GPU. (we didn't talk
// about this type of memory in class, but constant memory is a fast
// place to put read-only variables).
__constant__ GlobalConstants cuConstRendererParams;

// read-only lookup tables used to quickly compute noise (needed by
// advanceAnimation for the snowflake scene)
__constant__ int    cuConstNoiseYPermutationTable[256];
__constant__ int    cuConstNoiseXPermutationTable[256];
__constant__ float  cuConstNoise1DValueTable[256];

// color ramp table needed for the color ramp lookup shader
#define COLOR_MAP_SIZE 5
__constant__ float  cuConstColorRamp[COLOR_MAP_SIZE][3];


// including parts of the CUDA code from external files to keep this
// file simpler and to seperate code that should not be modified
#include "noiseCuda.cu_inl"
#include "lookupColor.cu_inl"
#include "circleBoxTest.cu_inl"


// kernelClearImageSnowflake -- (CUDA device code)
//
// Clear the image, setting the image to the white-gray gradation that
// is used in the snowflake image
__global__ void kernelClearImageSnowflake() {
    
    //printf("Block = %d %d\n", blockIdx.x, blockIdx.y);   
     
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float shade = .4f + .45f * static_cast<float>(height-imageY) / height;
    float4 value = make_float4(shade, shade, shade, 1.f);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelClearImage --  (CUDA device code)
//
// Clear the image, setting all pixels to the specified color rgba
__global__ void kernelClearImage(float r, float g, float b, float a) {
    
    //printf("Block = %d %d\n", blockIdx.x, blockIdx.y);  
    int imageX = blockIdx.x * blockDim.x + threadIdx.x;
    int imageY = blockIdx.y * blockDim.y + threadIdx.y;

    int width = cuConstRendererParams.imageWidth;
    int height = cuConstRendererParams.imageHeight;

    if (imageX >= width || imageY >= height)
        return;

    int offset = 4 * (imageY * width + imageX);
    float4 value = make_float4(r, g, b, a);

    // write to global memory: As an optimization, I use a float4
    // store, that results in more efficient code than if I coded this
    // up as four seperate fp32 stores.
    *(float4*)(&cuConstRendererParams.imageData[offset]) = value;
}

// kernelAdvanceFireWorks
//
// Update the position of the fireworks (if circle is firework)
__global__ void kernelAdvanceFireWorks() {
    const float dt = 1.f / 60.f;
    const float pi = 3.14159;
    const float maxDist = 0.25f;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;
    float* radius = cuConstRendererParams.radius;

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    if (0 <= index && index < NUM_FIREWORKS) { // firework center; no update
        return;
    }

    // determine the fire-work center/spark indices
    int fIdx = (index - NUM_FIREWORKS) / NUM_SPARKS;
    int sfIdx = (index - NUM_FIREWORKS) % NUM_SPARKS;

    int index3i = 3 * fIdx;
    int sIdx = NUM_FIREWORKS + fIdx * NUM_SPARKS + sfIdx;
    int index3j = 3 * sIdx;

    float cx = position[index3i];
    float cy = position[index3i+1];

    // update position
    position[index3j] += velocity[index3j] * dt;
    position[index3j+1] += velocity[index3j+1] * dt;

    // fire-work sparks
    float sx = position[index3j];
    float sy = position[index3j+1];

    // compute vector from firework-spark
    float cxsx = sx - cx;
    float cysy = sy - cy;

    // compute distance from fire-work
    float dist = sqrt(cxsx * cxsx + cysy * cysy);
    if (dist > maxDist) { // restore to starting position
        // random starting position on fire-work's rim
        float angle = (sfIdx * 2 * pi)/NUM_SPARKS;
        float sinA = sin(angle);
        float cosA = cos(angle);
        float x = cosA * radius[fIdx];
        float y = sinA * radius[fIdx];

        position[index3j] = position[index3i] + x;
        position[index3j+1] = position[index3i+1] + y;
        position[index3j+2] = 0.0f;

        // travel scaled unit length
        velocity[index3j] = cosA/5.0;
        velocity[index3j+1] = sinA/5.0;
        velocity[index3j+2] = 0.0f;
    }
}

// kernelAdvanceHypnosis
//
// Update the radius/color of the circles
__global__ void kernelAdvanceHypnosis() {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= cuConstRendererParams.numCircles)
        return;

    float* radius = cuConstRendererParams.radius;

    float cutOff = 0.5f;
    // place circle back in center after reaching threshold radisus
    if (radius[index] > cutOff) {
        radius[index] = 0.02f;
    } else {
        radius[index] += 0.01f;
    }
}


// kernelAdvanceBouncingBalls
//
// Update the positino of the balls
__global__ void kernelAdvanceBouncingBalls() {
    const float dt = 1.f / 60.f;
    const float kGravity = -2.8f; // sorry Newton
    const float kDragCoeff = -0.8f;
    const float epsilon = 0.001f;

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    float* velocity = cuConstRendererParams.velocity;
    float* position = cuConstRendererParams.position;

    int index3 = 3 * index;
    // reverse velocity if center position < 0
    float oldVelocity = velocity[index3+1];
    float oldPosition = position[index3+1];

    if (oldVelocity == 0.f && oldPosition == 0.f) { // stop-condition
        return;
    }

    if (position[index3+1] < 0 && oldVelocity < 0.f) { // bounce ball
        velocity[index3+1] *= kDragCoeff;
    }

    // update velocity: v = u + at (only along y-axis)
    velocity[index3+1] += kGravity * dt;

    // update positions (only along y-axis)
    position[index3+1] += velocity[index3+1] * dt;

    if (fabsf(velocity[index3+1] - oldVelocity) < epsilon
        && oldPosition < 0.0f
        && fabsf(position[index3+1]-oldPosition) < epsilon) { // stop ball
        velocity[index3+1] = 0.f;
        position[index3+1] = 0.f;
    }
}

// kernelAdvanceSnowflake -- (CUDA device code)
//
// move the snowflake animation forward one time step.  Updates circle
// positions and velocities.  Note how the position of the snowflake
// is reset if it moves off the left, right, or bottom of the screen.
__global__ void kernelAdvanceSnowflake() {

    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index >= cuConstRendererParams.numCircles)
        return;

    const float dt = 1.f / 60.f;
    const float kGravity = -1.8f; // sorry Newton
    const float kDragCoeff = 2.f;

    int index3 = 3 * index;

    float* positionPtr = &cuConstRendererParams.position[index3];
    float* velocityPtr = &cuConstRendererParams.velocity[index3];

    // loads from global memory
    float3 position = *((float3*)positionPtr);
    float3 velocity = *((float3*)velocityPtr);

    // hack to make farther circles move more slowly, giving the
    // illusion of parallax
    float forceScaling = fmin(fmax(1.f - position.z, .1f), 1.f); // clamp

    // add some noise to the motion to make the snow flutter
    float3 noiseInput;
    noiseInput.x = 10.f * position.x;
    noiseInput.y = 10.f * position.y;
    noiseInput.z = 255.f * position.z;
    float2 noiseForce = cudaVec2CellNoise(noiseInput, index);
    noiseForce.x *= 7.5f;
    noiseForce.y *= 5.f;

    // drag
    float2 dragForce;
    dragForce.x = -1.f * kDragCoeff * velocity.x;
    dragForce.y = -1.f * kDragCoeff * velocity.y;

    // update positions
    position.x += velocity.x * dt;
    position.y += velocity.y * dt;

    // update velocities
    velocity.x += forceScaling * (noiseForce.x + dragForce.y) * dt;
    velocity.y += forceScaling * (kGravity + noiseForce.y + dragForce.y) * dt;

    float radius = cuConstRendererParams.radius[index];

    // if the snowflake has moved off the left, right or bottom of
    // the screen, place it back at the top and give it a
    // pseudorandom x position and velocity.
    if ( (position.y + radius < 0.f) ||
         (position.x + radius) < -0.f ||
         (position.x - radius) > 1.f)
    {
        noiseInput.x = 255.f * position.x;
        noiseInput.y = 255.f * position.y;
        noiseInput.z = 255.f * position.z;
        noiseForce = cudaVec2CellNoise(noiseInput, index);

        position.x = .5f + .5f * noiseForce.x;
        position.y = 1.35f + radius;

        // restart from 0 vertical velocity.  Choose a
        // pseudo-random horizontal velocity.
        velocity.x = 2.f * noiseForce.y;
        velocity.y = 0.f;
    }

    // store updated positions and velocities to global memory
    *((float3*)positionPtr) = position;
    *((float3*)velocityPtr) = velocity;
}


// Warp ID and lane ID for 2D threads
__device__ __inline__ int lane_id(void) { 
    int tid = threadIdx.x + threadIdx.y*blockDim.x;
    return tid%WARP_SIZE; }



// Warp scan for 2D threads
__device__ __inline__ int warp_scan(int val){
  int x = val;

  #pragma unroll
  for(int offset = 1; offset < WARP_SIZE; offset <<= 1){

    int y = __shfl_up_sync(0xffffffff, x, offset);

    if(lane_id() >= offset)
      x += y;
  }

  return x - val;
}

// Block exclusive scan for 2D threads
__device__ __inline__ int block_scan(int* count, int x){

  __shared__ int sdata[WARP_SIZE];

  // A. Exclusive scan within each warp
  int warpPrefix = warp_scan(x);
  unsigned int ind = threadIdx.x + threadIdx.y * blockDim.x;
  unsigned int warpIdx = ind / WARP_SIZE;

  // B. Store in shared memory
  if(lane_id() == WARP_SIZE - 1)
    sdata[warpIdx] = warpPrefix + x;

  __syncthreads();

  // C. One warp scans in shared memory
  if(warpIdx == 0){
    sdata[ind] = warp_scan(sdata[ind]);
  }

  __syncthreads();

  // D. Each thread calculates its final value
  int thread_out_element = warpPrefix + sdata[warpIdx];
  int output = thread_out_element + *count;
  
  __syncthreads();

  if(ind == (blockDim.x * blockDim.y - 1))
    *count += (thread_out_element + x);

  return output;
}


// kernelRenderCircles -- (CUDA device code)
//
// Each thread renders a pixel for non-snow images
__global__ void kernelRenderCircles() {
    //printf("Block = %d %d\n", blockIdx.x, blockIdx.y);  
    unsigned int pixelX = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int pixelY = blockIdx.y*blockDim.y + threadIdx.y;
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    int tid =  threadIdx.y * blockDim.x + threadIdx.x; //Thread ID

    //Find region dimensions
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    int L = blockIdx.x * blockDim.x;
    int R =  L + blockDim.x;
    int B = blockIdx.y * blockDim.y;
    int T = B + blockDim.y;
    float boxL = (L) * invWidth;
    float boxR = (R) * invWidth;
    float boxB = (B) * invHeight;
    float boxT = (T) * invHeight;
    
    float alpha = .5f;
    float oneMinusAlpha =  1 - alpha;
    
    float4* imgPtr = (float4 *)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    register float4 existingColor = *imgPtr;
    register float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                    invHeight * (static_cast<float>(pixelY) + 0.5f));
    int num_threads = blockDim.x*blockDim.y; 
    int numCircles = cuConstRendererParams.numCircles;
    int circlesPerIteration = 1000; // Only these many circles will be handled per iteration for rendering
    float diffX1, diffY1, pixelDist1, diffX2, diffY2, pixelDist2;
    int probable_circle_index[32];
    // Shared memory variables
    __shared__ int scanned_circle_in_box[1024];
    __shared__ int num_circles_shared[1024];
    __shared__ int circles_to_render[1000]; // Maximum of 1000 circles to be rendered in this iteration
                                            // This was chosen based on available memory constraints
    __shared__ int s_circle_index[1000], s_index3[1000];
    __shared__ float3 s_p[1000];
    __shared__ float s_rad[1000];
    __shared__ float3 s_rgb[1000];
    __shared__ float s_maxDist[1000];
    __shared__ int count;
    // Other variables
    int num_circles_to_render;
    int index3;
    float3 p;
    float rad;            
    int start_index;
    int local_num_circles;
    int num_circles_to_check;
    int circleIndexStart;
    int num_circles;
    // We assume that the circles handled per iteration is more than or equal to number of threads
    for(int it = 0; it < numCircles; it = it + circlesPerIteration) {
        // Here we split the circles to render in this iteration amongst all threads to find if the circles
        // are within the current block of pixels
        local_num_circles = (it + circlesPerIteration - 1 <= numCircles - 1)? circlesPerIteration : (numCircles - it) ;
        num_circles_to_check = (local_num_circles + num_threads - 1)/(num_threads);
        circleIndexStart = tid * num_circles_to_check + it; 
        num_circles_to_check = (tid == num_threads-1) ? (local_num_circles - circleIndexStart) : num_circles_to_check;
        if(tid == 0) count = 0;
        num_circles = 0;
        #pragma unroll
        for (int circleToCheck = circleIndexStart; circleToCheck<circleIndexStart+num_circles_to_check; circleToCheck++) {
            // Each thread checks a few circles per iteration
            index3 = 3 * circleToCheck;
            p = *(float3*)(&cuConstRendererParams.position[index3]);
            rad = cuConstRendererParams.radius[circleToCheck];    
            if(circleInBox(p.x, p.y, rad, boxL, boxR, boxT, boxB)) {
                probable_circle_index[num_circles++] = circleToCheck;
            }
        }
        num_circles_shared[tid] = num_circles;
        __syncthreads();  
        // The probable circle count are run through an exclusive scan to obtain the position of the shared
        // memory array into which the circle details are stored by each individial thread
        scanned_circle_in_box[tid] = block_scan(&count, num_circles_shared[tid]);
        __syncthreads();
        num_circles_to_render = num_circles_shared[num_threads-1] + scanned_circle_in_box[num_threads-1];
        #pragma unroll
        for(int i=0; i<num_circles; i++) {
            // Store the circle info in the corresponding location in shared memory
            if (i == 0) start_index = scanned_circle_in_box[tid];
            circles_to_render[start_index] = probable_circle_index[i];
            s_circle_index[start_index] = circles_to_render[start_index];
            s_index3[start_index] = 3 * s_circle_index[start_index];
            s_p[start_index] = *(float3 *)(&cuConstRendererParams.position[s_index3[start_index]]);
            s_rad[start_index] = cuConstRendererParams.radius[s_circle_index[start_index]];
            s_rgb[start_index] = *(float3*)&(cuConstRendererParams.color[s_index3[start_index]]);
            s_maxDist[start_index] = s_rad[start_index] * s_rad[start_index];
            start_index++;
        }
        __syncthreads();
        // Rendering loop - Here we unroll the loop twice to improve performance
        for(int circleToCheck = 0; circleToCheck < num_circles_to_render; circleToCheck = circleToCheck+2) {
            // Each thread works on a single pixel. 
            // Amongst the found circles, it iterates through each circle and render if it lies within pixel
            diffX1 = s_p[circleToCheck].x - pixelCenterNorm.x;
            diffY1 = s_p[circleToCheck].y - pixelCenterNorm.y;
            pixelDist1 = diffX1 * diffX1 + diffY1 * diffY1;
            if (pixelDist1 <= s_maxDist[circleToCheck]) {
                // If pixel contributes to this circle
                // This is the atomic region. It has been changed to local memory write
                existingColor.x = alpha * s_rgb[circleToCheck].x + oneMinusAlpha * existingColor.x;
                existingColor.y = alpha * s_rgb[circleToCheck].y + oneMinusAlpha * existingColor.y;
                existingColor.z = alpha * s_rgb[circleToCheck].z + oneMinusAlpha * existingColor.z;
                existingColor.w = alpha + existingColor.w;       
            }
            if(circleToCheck+1 < num_circles_to_render) {
                diffX2 = s_p[circleToCheck+1].x - pixelCenterNorm.x;
                diffY2 = s_p[circleToCheck+1].y - pixelCenterNorm.y;
                pixelDist2 = diffX2 * diffX2 + diffY2 * diffY2;
                if (pixelDist2 <= s_maxDist[circleToCheck+1]) {
                    // If pixel contributes to this circle
                    // This is the atomic region. It has been changed to local memory write
                    existingColor.x = alpha * s_rgb[circleToCheck+1].x + oneMinusAlpha * existingColor.x;
                    existingColor.y = alpha * s_rgb[circleToCheck+1].y + oneMinusAlpha * existingColor.y;
                    existingColor.z = alpha * s_rgb[circleToCheck+1].z + oneMinusAlpha * existingColor.z;
                    existingColor.w = alpha + existingColor.w;       
                }
            }
        }
    }
    // Store the colour for the pixel in the global memory location
    *imgPtr = existingColor;
}



 
////////////////////////////////////////////////////////////////////////////////////////

// kernelRenderCirclesSnow -- (CUDA device code)
//
// Each thread renders a pixel for snowy images
__global__ void kernelRenderCirclesSnow() {
    unsigned int pixelX = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int pixelY = blockIdx.y*blockDim.y + threadIdx.y;
    short imageWidth = cuConstRendererParams.imageWidth;
    short imageHeight = cuConstRendererParams.imageHeight;

    int tid =  threadIdx.y * blockDim.x + threadIdx.x; //Thread ID

    //Find region dimensions
    float invWidth = 1.f / imageWidth;
    float invHeight = 1.f / imageHeight;
    int L = blockIdx.x * blockDim.x;
    int R =  L + blockDim.x;
    int B = blockIdx.y * blockDim.y;
    int T = B + blockDim.y;
    float boxL = (L) * invWidth;
    float boxR = (R) * invWidth;
    float boxB = (B) * invHeight;
    float boxT = (T) * invHeight;
    
    float alpha = .5f;
    float oneMinusAlpha =  1 - alpha;
    
    float4* imgPtr = (float4 *)(&cuConstRendererParams.imageData[4 * (pixelY * imageWidth + pixelX)]);
    register float4 existingColor = *imgPtr;
    register float2 pixelCenterNorm = make_float2(invWidth * (static_cast<float>(pixelX) + 0.5f),
                                                    invHeight * (static_cast<float>(pixelY) + 0.5f));
    int num_threads = blockDim.x*blockDim.y; 
    int numCircles = cuConstRendererParams.numCircles;
    int circlesPerIteration = 1000; // Only these many circles will be handled per iteration for rendering
    float diffX1, diffY1, pixelDist1, diffX2, diffY2, pixelDist2;
    int probable_circle_index[32];
    // Shared memory variables
    __shared__ int scanned_circle_in_box[1024];
    __shared__ int num_circles_shared[1024];
    __shared__ int circles_to_render[1000]; // Maximum of 1000 circles to be rendered in this iteration
                                            // This was chosen based on available memory constraints
    __shared__ int s_circle_index[1000], s_index3[1000];
    __shared__ float3 s_p[1000];
    __shared__ float s_rad[1000];
    __shared__ float3 s_rgb[1000];
    __shared__ float s_maxDist[1000];
    __shared__ int count;
    // Other variables
    float normPixelDist;
    float maxAlpha;
    const float kCircleMaxAlpha = .5f;
    const float falloffScale = 4.f;
    int num_circles_to_render;
    int index3;
    float3 p;
    float rad;            
    int start_index;
    int local_num_circles;
    int num_circles_to_check;
    int circleIndexStart;
    int num_circles;
    // We assume that the circles handled per iteration is more than or equal to number of threads
    for(int it = 0; it < numCircles; it = it + circlesPerIteration) {
        // Here we split the circles to render in this iteration amongst all threads to find if the circles
        // are within the current block of pixels
        local_num_circles = (it + circlesPerIteration - 1 <= numCircles - 1)? circlesPerIteration : (numCircles - it) ;
        num_circles_to_check = (local_num_circles + num_threads - 1)/(num_threads);
        circleIndexStart = tid * num_circles_to_check + it; 
        num_circles_to_check = (tid == num_threads-1) ? (local_num_circles - circleIndexStart) : num_circles_to_check;
        if(tid == 0) count = 0;
        num_circles = 0;
        #pragma unroll
        for (int circleToCheck = circleIndexStart; circleToCheck<circleIndexStart+num_circles_to_check; circleToCheck++) {
            // Each thread checks a few circles per iteration
            index3 = 3 * circleToCheck;
            p = *(float3*)(&cuConstRendererParams.position[index3]);
            rad = cuConstRendererParams.radius[circleToCheck];    
            if(circleInBox(p.x, p.y, rad, boxL, boxR, boxT, boxB)) {
                probable_circle_index[num_circles++] = circleToCheck;
            }
        }
        num_circles_shared[tid] = num_circles;
        __syncthreads();  
        // The probable circle count are run through an exclusive scan to obtain the position of the shared
        // memory array into which the circle details are stored by each individial thread
        scanned_circle_in_box[tid] = block_scan(&count, num_circles_shared[tid]);
        __syncthreads();
        num_circles_to_render = num_circles_shared[num_threads-1] + scanned_circle_in_box[num_threads-1];
        #pragma unroll
        for(int i=0; i<num_circles; i++) {
            // Store the circle info in the corresponding location in shared memory
            if (i == 0) start_index = scanned_circle_in_box[tid];
            circles_to_render[start_index] = probable_circle_index[i];
            s_circle_index[start_index] = circles_to_render[start_index];
            s_index3[start_index] = 3 * s_circle_index[start_index];
            s_p[start_index] = *(float3 *)(&cuConstRendererParams.position[s_index3[start_index]]);
            s_rad[start_index] = cuConstRendererParams.radius[s_circle_index[start_index]];
            s_rgb[start_index] = *(float3*)&(cuConstRendererParams.color[s_index3[start_index]]);
            s_maxDist[start_index] = s_rad[start_index] * s_rad[start_index];
            start_index++;
        }
        __syncthreads();
        // Rendering loop
        //#pragma unroll
        for(int circleToCheck = 0; circleToCheck < num_circles_to_render; circleToCheck = circleToCheck+2) {
            // Each thread works on a single pixel. 
            // Amongst the found circles, it iterates through each circle and render if it lies within pixel
            diffX1 = s_p[circleToCheck].x - pixelCenterNorm.x;
            diffY1 = s_p[circleToCheck].y - pixelCenterNorm.y;
            pixelDist1 = diffX1 * diffX1 + diffY1 * diffY1;
            if (pixelDist1 <= s_maxDist[circleToCheck]) {
                // If pixel contributes to this circle
                normPixelDist = sqrt(pixelDist1) / s_rad[circleToCheck];
                s_rgb[circleToCheck] = lookupColor(normPixelDist);

                maxAlpha = .6f + .4f * (1.f-s_p[circleToCheck].z);
                maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
                alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
                oneMinusAlpha = 1.f - alpha;
                // This is the atomic region. It has been changed to local memory write
                existingColor.x = alpha * s_rgb[circleToCheck].x + oneMinusAlpha * existingColor.x;
                existingColor.y = alpha * s_rgb[circleToCheck].y + oneMinusAlpha * existingColor.y;
                existingColor.z = alpha * s_rgb[circleToCheck].z + oneMinusAlpha * existingColor.z;
                existingColor.w = alpha + existingColor.w;     
            }
            if(circleToCheck+1 < num_circles_to_render) {
                diffX2 = s_p[circleToCheck+1].x - pixelCenterNorm.x;
                diffY2 = s_p[circleToCheck+1].y - pixelCenterNorm.y;
                pixelDist2 = diffX2 * diffX2 + diffY2 * diffY2;
                if (pixelDist2 <= s_maxDist[circleToCheck+1]) {
                    normPixelDist = sqrt(pixelDist2) / s_rad[circleToCheck+1];
                    s_rgb[circleToCheck+1] = lookupColor(normPixelDist);

                    maxAlpha = .6f + .4f * (1.f-s_p[circleToCheck+1].z);
                    maxAlpha = kCircleMaxAlpha * fmaxf(fminf(maxAlpha, 1.f), 0.f); // kCircleMaxAlpha * clamped value
                    alpha = maxAlpha * exp(-1.f * falloffScale * normPixelDist * normPixelDist);
                    oneMinusAlpha = 1.f - alpha;
                    // This is the atomic region. It has been changed to local memory write
                    existingColor.x = alpha * s_rgb[circleToCheck+1].x + oneMinusAlpha * existingColor.x;
                    existingColor.y = alpha * s_rgb[circleToCheck+1].y + oneMinusAlpha * existingColor.y;
                    existingColor.z = alpha * s_rgb[circleToCheck+1].z + oneMinusAlpha * existingColor.z;
                    existingColor.w = alpha + existingColor.w;      
                }
            }
        }
    }
    // Store the colour for the pixel in the global memory location
    *imgPtr = existingColor;
}

CudaRenderer::CudaRenderer() {
    image = NULL;

    numCircles = 0;
    position = NULL;
    velocity = NULL;
    color = NULL;
    radius = NULL;

    cudaDevicePosition = NULL;
    cudaDeviceVelocity = NULL;
    cudaDeviceColor = NULL;
    cudaDeviceRadius = NULL;
    cudaDeviceImageData = NULL;
}

CudaRenderer::~CudaRenderer() {

    if (image) {
        delete image;
    }

    if (position) {
        delete [] position;
        delete [] velocity;
        delete [] color;
        delete [] radius;
    }

    if (cudaDevicePosition) {
        cudaFree(cudaDevicePosition);
        cudaFree(cudaDeviceVelocity);
        cudaFree(cudaDeviceColor);
        cudaFree(cudaDeviceRadius);
        cudaFree(cudaDeviceImageData);
    }
}

const Image*
CudaRenderer::getImage() {

    // need to copy contents of the rendered image from device memory
    // before we expose the Image object to the caller

    printf("Copying image data from device\n");

    cudaMemcpy(image->data,
               cudaDeviceImageData,
               sizeof(float) * 4 * image->width * image->height,
               cudaMemcpyDeviceToHost);

    return image;
}

void
CudaRenderer::loadScene(SceneName scene) {
    sceneName = scene;
    loadCircleScene(sceneName, numCircles, position, velocity, color, radius);
}

void
CudaRenderer::setup() {

    int deviceCount = 0;
    std::string name;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Initializing CUDA for CudaRenderer\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++) {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        name = deviceProps.name;

        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n", static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");

    // By this time the scene should be loaded.  Now copy all the key
    // data structures into device memory so they are accessible to
    // CUDA kernels
    //
    // See the CUDA Programmer's Guide for descriptions of
    // cudaMalloc and cudaMemcpy

    cudaMalloc(&cudaDevicePosition, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceVelocity, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceColor, sizeof(float) * 3 * numCircles);
    cudaMalloc(&cudaDeviceRadius, sizeof(float) * numCircles);
    cudaMalloc(&cudaDeviceImageData, sizeof(float) * 4 * image->width * image->height);

    cudaMemcpy(cudaDevicePosition, position, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceVelocity, velocity, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceColor, color, sizeof(float) * 3 * numCircles, cudaMemcpyHostToDevice);
    cudaMemcpy(cudaDeviceRadius, radius, sizeof(float) * numCircles, cudaMemcpyHostToDevice);

    // Initialize parameters in constant memory.  We didn't talk about
    // constant memory in class, but the use of read-only constant
    // memory here is an optimization over just sticking these values
    // in device global memory.  NVIDIA GPUs have a few special tricks
    // for optimizing access to constant memory.  Using global memory
    // here would have worked just as well.  See the Programmer's
    // Guide for more information about constant memory.

    GlobalConstants params;
    params.sceneName = sceneName;
    params.numCircles = numCircles;
    params.imageWidth = image->width;
    params.imageHeight = image->height;
    params.position = cudaDevicePosition;
    params.velocity = cudaDeviceVelocity;
    params.color = cudaDeviceColor;
    params.radius = cudaDeviceRadius;
    params.imageData = cudaDeviceImageData;

    cudaMemcpyToSymbol(cuConstRendererParams, &params, sizeof(GlobalConstants));

    // also need to copy over the noise lookup tables, so we can
    // implement noise on the GPU
    int* permX;
    int* permY;
    float* value1D;
    getNoiseTables(&permX, &permY, &value1D);
    cudaMemcpyToSymbol(cuConstNoiseXPermutationTable, permX, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoiseYPermutationTable, permY, sizeof(int) * 256);
    cudaMemcpyToSymbol(cuConstNoise1DValueTable, value1D, sizeof(float) * 256);

    // last, copy over the color table that's used by the shading
    // function for circles in the snowflake demo

    float lookupTable[COLOR_MAP_SIZE][3] = {
        {1.f, 1.f, 1.f},
        {1.f, 1.f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, .9f, 1.f},
        {.8f, 0.8f, 1.f},
    };

    cudaMemcpyToSymbol(cuConstColorRamp, lookupTable, sizeof(float) * 3 * COLOR_MAP_SIZE);

}

// allocOutputImage --
//
// Allocate buffer the renderer will render into.  Check status of
// image first to avoid memory leak.
void
CudaRenderer::allocOutputImage(int width, int height) {

    if (image)
        delete image;
    image = new Image(width, height);
}

// clearImage --
//
// Clear's the renderer's target image.  The state of the image after
// the clear depends on the scene being rendered.
void
CudaRenderer::clearImage() {

    // 256 threads per block is a healthy number
    dim3 blockDim(32, 32, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);

    if (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME) {
        kernelClearImageSnowflake<<<gridDim, blockDim>>>();
    } else {
        kernelClearImage<<<gridDim, blockDim>>>(1.f, 1.f, 1.f, 1.f);
    }
    cudaDeviceSynchronize();
}

// advanceAnimation --
//
// Advance the simulation one time step.  Updates all circle positions
// and velocities
void
CudaRenderer::advanceAnimation() {
     // 256 threads per block is a healthy number
    dim3 blockDim(1024, 1);
    dim3 gridDim((numCircles + blockDim.x - 1) / blockDim.x);

    // only the snowflake scene has animation
    if (sceneName == SNOWFLAKES) {
        kernelAdvanceSnowflake<<<gridDim, blockDim>>>();
    } else if (sceneName == BOUNCING_BALLS) {
        kernelAdvanceBouncingBalls<<<gridDim, blockDim>>>();
    } else if (sceneName == HYPNOSIS) {
        kernelAdvanceHypnosis<<<gridDim, blockDim>>>();
    } else if (sceneName == FIREWORKS) {
        kernelAdvanceFireWorks<<<gridDim, blockDim>>>();
    }
    cudaDeviceSynchronize();
}

void
CudaRenderer::render() {
    int dimx, dimy;
    if (numCircles < 100) {
        // For scenes with lesser number of circles, we use lesser threads as it gives better performance
        dimx = 16;
        dimy = 16;
    } else {
        dimx = 32;
        dimy = 32;
    } 
    dim3 blockDim(dimx, dimy, 1);
    dim3 gridDim(
        (image->width + blockDim.x - 1) / blockDim.x,
        (image->height + blockDim.y - 1) / blockDim.y);
    bool snowScene = (sceneName == SNOWFLAKES || sceneName == SNOWFLAKES_SINGLE_FRAME);
    // This condition has been moved here from the innermost loop inside shadePixel to increase performance
    // The image is divided into regions of 16 x 16 or 32 x 32 and each region is handled by a kernel
    if(snowScene) 
        kernelRenderCirclesSnow<<<gridDim, blockDim>>>();
    else
        kernelRenderCircles<<<gridDim, blockDim>>>();
    cudaDeviceSynchronize();
}

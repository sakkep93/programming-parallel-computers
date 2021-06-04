#include "cp.h"
#include <cuda_runtime.h>
#include "cudacheck.h"

static inline int divup(int a, int b) { return (a + b - 1)/b; }
static inline int roundup(int a, int b) { return divup(a, b) * b;}

__global__ void normalizekernel(int ny, int nx, float* data, float* ntdata){
  int y = blockIdx.x;
  if(y>=ny) return;

  // mean
  float s=0.0;
  for(int x=0; x<nx; ++x){
    float v = data[x+y*nx];
    s += v;
  }
  float m = s / (float) nx;

  // rootsquaresum
  float rs = 0.0;
  for(int x=0; x<nx; ++x){
    float v = data[x+y*nx];
    rs += ((v-m)*(v-m));
  }
  float r = sqrt(rs);

  // store
  for(int x=0; x<nx; ++x){
    float v = ( (data[x+y*nx]) - m ) / r;
    ntdata[y+x*ny] = v;
  }
  
}

__global__ void matmulkernel(int ny, int nx, float* ntdata, float* r){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if(i>=ny || j>=ny) return;
  float s = 0.0;
  if(i<=j){
    for(int k=0; k<nx; ++k){
      float x = ntdata[ny*k+j];
      float y = ntdata[ny*k+i];
      s += (x * y);
    }
  }
  r[j+i*ny] = s;
}

void correlate(int ny, int nx, const float* data, float* result) {

  int tmpsize = ny*nx*sizeof(float);
  int ressize = ny*ny*sizeof(float);

  // Allocate GPU memory for input data shape: (ny, nx) size: ny*nx
  float* dGPU = NULL;
  CHECK(cudaMalloc((void**)&dGPU, tmpsize));

  // Allocate GPU memory for normalized & transposed data shape: (nx, ny) size: ny*nx
  float* ntGPU = NULL;
  CHECK(cudaMalloc((void**)&ntGPU, tmpsize));

  // Allocate GPU memory for result data shape: (ny, ny) size: ny*ny
  float* rGPU = NULL;
  CHECK(cudaMalloc((void**)&rGPU, ressize));

  // Copy input data to GPU
  CHECK(cudaMemcpy(dGPU, data, tmpsize, cudaMemcpyHostToDevice));

  int nBlocks = roundup(ny, 64);
  // Run normalization & transpose kernel
  {
    normalizekernel<<<nBlocks, 1>>>(ny, nx, dGPU, ntGPU);
    CHECK(cudaGetLastError());
  }

  // Run kernel (matmul)
  {
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    matmulkernel<<<dimGrid, dimBlock>>>(ny, nx, ntGPU, rGPU);
    CHECK(cudaGetLastError());
  }

  // Copy data back to CPU & release memory
  CHECK(cudaMemcpy(result, rGPU, ressize, cudaMemcpyDeviceToHost));

  // Free
  CHECK(cudaFree(dGPU)); CHECK(cudaFree(ntGPU)); CHECK(cudaFree(rGPU));
}


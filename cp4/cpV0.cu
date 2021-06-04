#include "cp.h"
#include <cuda_runtime.h>
#include "cudacheck.h"

static inline int divup(int a, int b) { return (a + b - 1)/b; }

float mean(const float* row, int nx){
  float s=0.0, m;
  for (int i=0; i<nx; ++i){ s = s + row[i]; }
  m = s / (float) nx;
  return m;
}

float rootsquaresum(const float* row, int nx, float m){
  float s = 0.0, r;
  for(int i=0; i<nx; ++i){
    float v = row[i];
    s = s + ((v-m)*(v-m));
  }
  r = sqrt(s);
  return r;
}

void normalize(int ny, int nx, const float* data, float* ndata){
  for(int j=0; j<ny; ++j){
    float m = mean(&data[j*nx], nx);
    float rss = rootsquaresum(&data[j*nx], nx, m);
    for(int i=0; i<nx; ++i){
      ndata[nx*j+i] = (data[j*nx+i] - m) / rss;
    }
  }
}

__global__ void matmulkernel(int ny, int nx, float* nd, float* r){
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  int j = threadIdx.y + blockIdx.y * blockDim.y;
  if(i>=ny || j>=ny) return;
  float s = 0.0;
  if(i<=j){
    for(int k=0; k<nx; ++k){
      float x = nd[i*nx+k];
      float y = nd[j*nx+k];
      s += (x * y);
    }
  }
  r[j+i*ny] = s;
}

void correlate(int ny, int nx, const float* data, float* result) {
  
  // normalization done in CPU
  float* ndata = new float[ny*nx];
  normalize(ny, nx, data, ndata);

  // Allocate memory & copy data to GPU
  float* nGPU = NULL;
  CHECK(cudaMalloc((void**)&nGPU, ny*nx*sizeof(float)));
  float* rGPU = NULL;
  CHECK(cudaMalloc((void**)&rGPU, ny*ny*sizeof(float)));
  CHECK(cudaMemcpy(nGPU, ndata, ny*nx*sizeof(float), cudaMemcpyHostToDevice));

  // Run kernel (matmul)
  dim3 dimBlock(16, 16);
  dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
  matmulkernel<<<dimGrid, dimBlock>>>(ny, nx, nGPU, rGPU);
  CHECK(cudaGetLastError());

  // Copy data back to CPU & release memory
  CHECK(cudaMemcpy(result, rGPU, ny*ny*sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(nGPU));
  CHECK(cudaFree(rGPU));
  std::free(ndata);
}

#include "cp.h"
#include <cuda_runtime.h>
#include "cudacheck.h"
#include <vector>

static inline int divup(int a, int b) { return (a + b - 1)/b; }
static inline int roundup(int a, int b) { return divup(a, b) * b;}

float mean(const float* row, int nx){
  float s=0.0;
  for(int x=0; x<nx; ++x){
    s += row[x];
  }

  return s / (float) nx;
}

float rootsquaresum(const float* row, int nx, float m){
  float rs=0.0;

  for(int x=0; x<nx; ++x){
    float v = row[x];
    rs += ((v-m)*(v-m));
  }
  return sqrt(rs);
}

std::vector<float> normalize(int ny, int nx, const float* data){
  std::vector<float> ndata(ny*nx);
  for(int y=0; y<ny; ++y){
    
    float m = mean(&data[y*nx], nx);
    float rs = rootsquaresum(&data[y*nx], nx, m);
    
    for(int x=0; x<nx; ++x){
      float v = data[x+nx*y];
      ndata[x + y*nx] = (v-m) / rs;
    }
  }
  return ndata;
}


__global__ void tpadkernel(int ny, int nx, int nny, int nnx, float* ndata, float* ntpdata){
  int xa = threadIdx.x;
  int y = blockIdx.y;
 
  for(int xb=0; xb<nnx; xb += 64){
    int x = xb+xa;
    float v = (y<ny && x<nx) ? ndata[x + nx*y]: 0.0;
    ntpdata[nny*x + y] = v;
  }
}


__global__ void matmul64kernel(int ny, int nx, int nny, float* ntpdata, float* r){
  int ia = threadIdx.x;
  int ja = threadIdx.y;
  int ic = blockIdx.x;
  int jc = blockIdx.y;

  if(jc < ic) return;

  __shared__ float xx[4][64];
  __shared__ float yy[4][64];

  float s[8][8];
  for(int ib=0; ib<8; ++ib){
    for(int jb=0; jb<8; ++jb){
      s[ib][jb] = 0.0;
    }
  }

  for(int ks=0; ks<nx; ks+=4){
    int ija = ja*8 + ia;
    int i = ic*64 + ija;
    int j = jc*64 + ija;

    for(int f=0; f<4; ++f){
      int k = ks + f;
      xx[f][ija] = ntpdata[nny*k+i];
      yy[f][ija] = ntpdata[nny*k+j];
    }

    __syncthreads();

    #pragma unroll
    for(int f=0; f<4; ++f){
      float y[8];
      for(int jb=0; jb<8; ++jb){
        y[jb] = yy[f][jb*8 + ja];
      }
      for(int ib=0; ib<8; ++ib){
        float x = xx[f][ib*8 + ia];
        for(int jb=0; jb<8; ++jb){
          s[ib][jb] += x * y[jb];
        }
      }
    }
    __syncthreads();
  }

  for(int ib=0; ib<8; ++ib){
    for(int jb=0; jb<8; ++jb){
      int i = ic*64 + ib*8 + ia;
      int j = jc*64 + jb*8 + ja;
      if(i<ny && j<ny){
        r[ny*i + j] = s[ib][jb];
      }
    }
  }
}


void correlate(int ny, int nx, const float* data, float* result) {

  // normalize on CPU
  std::vector<float> ndata = normalize(ny, nx, data);

  // padded
  int nny = roundup(ny, 64);
  int nnx = roundup(nx, 64);

  // Allocate GPU memory for normalized data
  float* nGPU = NULL;
  CHECK(cudaMalloc((void**)&nGPU, ny*nx*sizeof(float)));

  // Allocate GPU memory for padded & transposed ndata
  float* ntpGPU = NULL;
  CHECK(cudaMalloc((void**)&ntpGPU, nny*nnx*sizeof(float)));

  // Allocate GPU memory for result
  float* rGPU = NULL;
  CHECK(cudaMalloc((void**)&rGPU, ny*ny*sizeof(float)));

  // Copy normalized data to GPU
  CHECK(cudaMemcpy(nGPU, ndata.data(), ny*nx*sizeof(float), cudaMemcpyHostToDevice));

  // Run padding & transpose kernel
  {
    dim3 dimBlock(64, 1);
    dim3 dimGrid(1, nny);
    tpadkernel<<<dimGrid, dimBlock>>>(ny, nx, nny, nnx, nGPU, ntpGPU);
    CHECK(cudaGetLastError());
  }

  // Run matmul 64 kernel
  {
    dim3 dimBlock(8,8);
    dim3 dimGrid(nny/64, nny/64);
    matmul64kernel<<<dimGrid, dimBlock>>>(ny, nx, nny, ntpGPU, rGPU);
    CHECK(cudaGetLastError());
  }

  CHECK(cudaMemcpy(result, rGPU, ny*ny*sizeof(float), cudaMemcpyDeviceToHost));
  CHECK(cudaFree(nGPU)); CHECK(cudaFree(ntpGPU)); CHECK(cudaFree(rGPU));
  
}

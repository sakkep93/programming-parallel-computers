#include "is.h"
#include <cuda_runtime.h>
#include "cudacheck.h"
#include <vector>

static inline int divup(int a, int b) { return (a + b - 1)/b; }

std::vector<float> precomputeSums(int ny, int nx, int pny, int pnx, const float* data){

  // Allocate memory from heap and initialize with zeros
  std::vector<float> sums(pnx*pny, 0.f);

  // Calculate rect c sums having upper-left corner at 0,0 and only 1st color component
  for(int y=0; y<ny; ++y){
    for(int x=0; x<nx; ++x){
      sums[(x+1) + pnx*(y+1)] = data[3 * (x+nx*y)]
                              + sums[(x+1) + pnx*y]
                              + sums[x + pnx*(y+1)]
                              - sums[x + pnx*y];
    }
  }
  return sums;
}

__global__ void rectDimHkernel(int ny, int nx, int pSize, int pny, int pnx, const float* sums, float* rectdims){

  int width = threadIdx.x + blockIdx.x * blockDim.x;
  int height = threadIdx.y + blockIdx.y * blockDim.y;

  if( !(0 < width && width <= nx) || !(0 < height && height <=ny) ) return;

  int xSize = height * width;
  int ySize = pSize - xSize;

  float xInv = 1.0f / (float) xSize;
  float yInv = ySize == 0 ? 0.f : 1.0f / (float) ySize;

  float vPc = sums[pnx*pny-1];
  float H = 0.f;

  for(int y0=0; y0<=ny-height; ++y0){

    int y1 = y0 + height;
    for(int x0=0; x0<=nx-width; ++x0){
      int x1 = x0 + width;

      float s1 = sums[y1*pnx + x1];
      float s2 = sums[y1*pnx + x0];
      float s3 = sums[y0*pnx + x1];
      float s4 = sums[y0*pnx + x0];

      float vXc = s1 - s2 - s3 + s4;
      float vYc = vPc - vXc;

      float h = vXc * vXc * xInv + vYc * vYc * yInv;

      if(h > H) H = h;
    }
  }

  rectdims[height*pnx + width] = H;
}

struct Rectangle{ int width; int height; int size; };

Rectangle findOptimalRectangle(int ny, int nx, int pnx, const float* rectdims){

  float H = 0.f;
  int width = 0, height = 0;

  for(int h=1; h<=ny; ++h){
    for(int w=1; w<=nx; ++w){

      float hu = rectdims[h*pnx+w];
      if (hu > H){
        H = hu;
        width = w;
        height = h;
      }
    }
  }

  Rectangle rect = {width, height, width*height};
  return rect;
}

struct SegmentResult{ int y0; int x0; int y1; int x1; float outer[3]; float inner[3]; };

SegmentResult findOptimalSegment(int ny, int nx, int pny, int pnx, Rectangle* rect, const float* sums){

  // image
  int pSize = nx*ny;
  float vPc = sums[pnx*pny-1];

  // rectangle
  int height = rect->height;
  int width = rect->width;
  int xSize = rect->size;

  // background
  int ySize = pSize - xSize;

  // size inverses
  float xInv = 1.0f / (float) xSize;
  float yInv = ySize == 0 ? 0.f : 1.0f / (float) ySize;

  float H = 0.f;
  float rC = 0.f, bC = 0.f;
  int bx0 = 0, bx1 = 0, by0 = 0, by1 = 0;

  for(int y0=0; y0<=ny-height; ++y0){
    for(int x0=0; x0<=nx-width; ++x0){
      int y1 = y0 + height;
      int x1 = x0 + width;

      float s1 = sums[y1*pnx + x1];
      float s2 = sums[y1*pnx + x0];
      float s3 = sums[y0*pnx + x1];
      float s4 = sums[y0*pnx + x0];

      float vXc = s1 - s2 - s3 + s4;
      float vYc = vPc - vXc;
      float h = vXc * vXc * xInv + vYc * vYc * yInv;

      if(h > H){
        H = h;
        rC = vXc;
        bC = vYc;
        bx0 = x0;
        bx1 = x1;
        by0 = y0;
        by1 = y1;
      }
    }
  }

  rC *= xInv;
  bC *= yInv;

  SegmentResult sr = { by0, bx0, by1, bx1, { bC, bC, bC }, {rC, rC, rC } };
  return sr;
}

Result segment(int ny, int nx, const float* data){

  // Paddings for sums
  int pnx = nx+1, pny = ny+1;

  // Precompute rectangle color sums on CPU
  std::vector<float> sums = precomputeSums(ny, nx, pny, pnx, data);

  // Allocate GPU memory for padded sums
  float* sGPU = NULL;
  CHECK(cudaMalloc((void**)&sGPU, pnx*pny*sizeof(float)));

  // Allocate GPU memory for result holding H utility values for different rectangle sizes
  float* hGPU = NULL;
  CHECK(cudaMalloc((void**)&hGPU, pnx*pny*sizeof(float)));

  // Copy sums data to GPU
  CHECK(cudaMemcpy(sGPU, sums.data(), pnx*pny*sizeof(float), cudaMemcpyHostToDevice));

  // Run rectangle size H kernel
  {
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(nx, dimBlock.x), divup(ny, dimBlock.y));
    rectDimHkernel<<<dimGrid, dimBlock>>>(ny, nx, nx*ny, pny, pnx, sGPU, hGPU);
    CHECK(cudaGetLastError());
  }

  // Copy to CPU from GPU
  std::vector<float> rectdims(pnx*pny);
  CHECK(cudaMemcpy(rectdims.data(), hGPU, pnx*pny*sizeof(float), cudaMemcpyDeviceToHost));

  // Find best sized rectangle
  Rectangle rect = findOptimalRectangle(ny, nx, pnx, rectdims.data());

  // Find coordinates and color distribution with best Rectangle
  SegmentResult sr = findOptimalSegment(ny, nx, pny, pnx, &rect, sums.data());
  
  Result result {
    sr.y0,
    sr.x0,
    sr.y1,
    sr.x1,
    { sr.outer[0], sr.outer[1], sr.outer[2] },
    { sr.inner[0], sr.inner[1], sr.inner[2] }
  };

  // Free GPU memory
  CHECK(cudaFree(sGPU)); CHECK(cudaFree(hGPU));

  return result;
}


#include "is.h"
#include "vector.h"
#include <x86intrin.h>
#include <cmath>
#include <vector>

static inline float8_t max8(float8_t x, float8_t y){ return x > y ? x : y;}

static inline float hmax8(float8_t vv){
  float v = 0.f;
  for(int i=0; i<8; ++i) v = std::max(vv[i], v);
  return v;
}

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

struct RectDim{ int width; int height; int size; };

RectDim findOptimalRectDim(int ny, int nx, int pny, int pnx, const float* sums){
  // Find best width and heigth of rectangle that maximizes h function

  // Global constants
  int pSize = nx*ny;
  float vPc = sums[pnx*pny-1];
  float8_t vvPc = {vPc, vPc, vPc, vPc, vPc, vPc, vPc, vPc};

  // Global best values
  float bH = 0.f;
  int bWidth = 0, bHeight = 0;

  #pragma omp parallel
  {
    // Thread specific best values
    float tH = 0.f;
    float8_t vtH = float8_0;

    #pragma omp for schedule(static, 1)
    for(int h=1; h<=ny; ++h){
      for(int w=1; w<=nx; ++w){

        int xSize = h * w;
        int ySize = pSize - xSize;

        float xInv = 1.0f / (float) xSize;
        float yInv = ySize == 0 ? 0 : 1.0f / (float) ySize;

        for(int y0=0; y0<=ny-h; ++y0){

          int y1 = y0 + h;
          int x0Iters = pnx - w;
          int batch8 = x0Iters / 8;

          // Calculate 8 wide H utility with vector operations
          for(int i=0; i<batch8; ++i){
            int x0 = 8*i;
            int x1 = x0 + w;

            float8_t s1 = _mm256_loadu_ps(sums + y1*pnx + x1);
            float8_t s2 = _mm256_loadu_ps(sums + y1*pnx + x0);
            float8_t s3 = _mm256_loadu_ps(sums + y0*pnx + x1);
            float8_t s4 = _mm256_loadu_ps(sums + y0*pnx + x0);

            float8_t vvXc = s1 - s2 - s3 + s4;
            float8_t vvYc = vvPc - vvXc;

             // H utility function right param
            vtH = max8(vtH, vvXc * vvXc * xInv + vvYc * vvYc * yInv);
          }

          // Calculate remaining with scalar operations
          for(int x0=8*batch8; x0<x0Iters; ++x0){
            int x1 = x0 + w;

            float s1 = sums[y1*pnx + x1];
            float s2 = sums[y1*pnx + x0];
            float s3 = sums[y0*pnx + x1];
            float s4 = sums[y0*pnx + x0];

            float vXc = s1 - s2 - s3 + s4;
            float vYc = vPc - vXc;

            // H utility function
            float H = (vXc * vXc) * xInv + vYc * vYc * yInv;

            // Update local best
            if(H > tH) tH = H;
          }
        }

        // Update local best from vector operations
        float H = hmax8(vtH);
        if(H>tH) tH = H;

        // Update global best rectangle dims
        #pragma omp critical
        {
          if(tH > bH){ bH = tH; bWidth = w; bHeight = h; }
        }
      }
    }
  }
  RectDim rectDim = {bWidth, bHeight, bWidth*bHeight};
  return rectDim;
}

struct SegmentResult { int y0; int x0; int y1; int x1; float outer[3]; float inner[3]; };

SegmentResult findResult(int ny, int nx, int pny, int pnx, RectDim* rectDim, const float* sums){

    // image
    int pSize = ny*nx;
    float vPc = sums[pnx*pny-1];

    // best rectangle
    int height = rectDim->height;
    int width = rectDim->width;
    int rSize = rectDim->size;
    float rInv = 1.0f / (float) rSize;

    // background
    int bSize = pSize - rSize;
    float bInv = bSize == 0 ? 0 : 1.0f / (float) bSize;

    // Find coordinates and color distribution
    float bH = 0.f, rC = 0.f, bC = 0.f;
    int bx0=0, bx1=0, by0=0, by1=0;

    for(int y0=0; y0<=ny-height; ++y0){
      for(int x0=0; x0<=nx-width; ++x0){
        int y1 = y0 + height;
        int x1 = x0 + width;

        float s1 = sums[y1*pnx + x1];
        float s2 = sums[y1*pnx + x0];
        float s3 = sums[y0*pnx + x1];
        float s4 = sums[y0*pnx + x0];

        float vRc = s1 - s2 - s3 + s4;
        float vBc = vPc - vRc;

        float H = vRc * vRc * rInv + vBc * vBc * bInv;

        if(H > bH){
          bH = H;
          rC = vRc * rInv;
          bC = vBc * bInv;
          bx0 = x0;
          bx1 = x1;
          by0 = y0;
          by1 = y1;
        }
      }
    }

    SegmentResult sr = { by0, bx0, by1, bx1, {bC, bC, bC}, {rC, rC, rC} };
    return sr;
}

Result segment(int ny, int nx, const float* data){

  // Paddings
  int pnx = nx+1, pny = ny+1;

  // Precompute rectangle color sums
  std::vector<float> sums = precomputeSums(ny, nx, pny, pnx, data);

  // Find optimal rectangle dimensions
  RectDim Rd = findOptimalRectDim(ny, nx, pny, pnx, sums.data());

  // Find coordinates and color distribution with best rectDim
  SegmentResult Sr = findResult(ny, nx, pny, pnx, &Rd, sums.data());

  Result result {
    Sr.y0,
    Sr.x0,
    Sr.y1,
    Sr.x1,
    { Sr.outer[0], Sr.outer[1], Sr.outer[2] },
    { Sr.inner[0], Sr.inner[1], Sr.inner[2] }
  };

  return result;
}

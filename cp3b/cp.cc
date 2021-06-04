#include "cp.h"
#include <cmath>
#include "vector.h"
#include <x86intrin.h>
#include <vector>

static inline float8_t swap4(float8_t x) { return _mm256_permute2f128_ps(x, x, 0b00000001); }
static inline float8_t swap2(float8_t x) { return _mm256_permute_ps(x, 0b01001110); }
static inline float8_t swap1(float8_t x) { return _mm256_permute_ps(x, 0b10110001); }

float mean(const float* row, int nx){
  float s=0.0, m;
  for(int i=0; i<nx; ++i){
    s += row[i];
  }
  m = s / (float) nx;
  return m;
}

float rootsquaresum(const float* row, int nx, float m){
  float s=0.0, r;
  for(int i=0; i<nx; ++i){
    float v = row[i];
    s += ((v-m)*(v-m));
  }
  r = sqrt(s);
  return r;
}

void correlate(int ny, int nx, const float* data, float* result) {

  // normalize
  std::vector<float> ndata(ny*nx);
  #pragma omp parallel for schedule(static, 1)
  for(int i=0; i<ny; ++i){
    float m = mean(&data[i*nx], nx);
    float rss = rootsquaresum(&data[i*nx], nx, m);
    for(int j=0; j<nx; ++j){
      ndata[j+i*nx] = (data[j+i*nx] - m)/rss;
    }

  }
  int na = (ny+8-1)/8;
  float8_t* vx = float8_alloc(nx*na);

  #pragma omp parallel for schedule(static, 1)
  for(int ja=0; ja<na; ++ja){
    for(int i=0; i<nx; ++i){
      for(int jb=0; jb<8; ++jb){
        int j=ja*8+jb;
        vx[nx*ja+i][jb] = j < ny ? ndata[nx*j+i] : 0.0;
      }
    }
  }

  #pragma omp parallel for schedule(static, 1)
  for(int ia=0; ia<na; ++ia){
    for(int ja=ia; ja<na; ++ja){
      float8_t ss000 = float8_0;
      float8_t ss001 = float8_0;
      float8_t ss010 = float8_0;
      float8_t ss011 = float8_0;
      float8_t ss100 = float8_0;
      float8_t ss101 = float8_0;
      float8_t ss110 = float8_0;
      float8_t ss111 = float8_0;

      for(int k=0; k<nx; ++k){
        constexpr int PF = 20;
        __builtin_prefetch(&vx[nx*ia + k + PF]);
        __builtin_prefetch(&vx[nx*ja + k + PF]);
        float8_t a000 = vx[nx*ia+k];
        float8_t b000 = vx[nx*ja+k];
        float8_t a100 = swap4(a000);
        float8_t a010 = swap2(a000);
        float8_t a110 = swap2(a100);
        float8_t b001 = swap1(b000);

        ss000 = ss000 + (a000*b000);
        ss001 = ss001 + (a000*b001);
        ss010 = ss010 + (a010*b000);
        ss011 = ss011 + (a010*b001);
        ss100 = ss100 + (a100*b000);
        ss101 = ss101 + (a100*b001);
        ss110 = ss110 + (a110*b000);
        ss111 = ss111 + (a110*b001);
      }
      float8_t ss[8] = {ss000, ss001, ss010, ss011, ss100, ss101, ss110, ss111};
      for(int kb=1; kb<8; kb+=2){
        ss[kb] = swap1(ss[kb]);
      }
      for(int jb=0; jb<8; ++jb){
        for(int ib=0; ib<8; ++ib){
          int i=ib+ia*8;
          int j=jb+ja*8;
          if(j<ny && i<ny && i<=j){
            result[ny*i+j] = ss[ib^jb][jb];
          }
        }
      }
    }
  }
  std::free(vx);
}

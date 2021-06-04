#include "cp.h"
#include <cmath>
#include "vector.h"
#include <vector>
#include <x86intrin.h>

static inline double4_t swap2(double4_t x) { return _mm256_permute2f128_pd(x, x, 0b00000001); }
static inline double4_t swap1(double4_t x) { return _mm256_permute_pd(x, 0b00000101); }

double mean(const float* row, int nx){
  double s=0.0, m;
  for (int i=0; i<nx; ++i){
    s = s + (double) row[i];
  }
  m = s / (double) nx;
  return m;
}

double rootsquaresum(const float* row, int nx, double m){
  double s = 0.0, r;
  for(int i=0; i<nx; ++i){
    double v = (double) row[i];
    s = s + ((v-m)*(v-m));
  }
  r = sqrt(s);
  return r;
}

void correlate(int ny, int nx, const float* data, float* result) {

  // normalize
  std::vector<double> ndata(ny*nx);
  #pragma omp parallel for schedule(static, 1)
  for(int j=0; j<ny; ++j){
    double m = mean(&data[j*nx], nx);
    double rss = rootsquaresum(&data[j*nx], nx, m);
    for(int i=0; i<nx; ++i){
      ndata[i+j*nx] = (((double) data[i+j*nx])-m)/rss;
    }
  }

  //vectorize
  int na = (ny+4-1)/4;
  double4_t* vdata = double4_alloc(nx*na);
  #pragma omp parallel for schedule(static, 1)
  for(int ja=0; ja<na; ++ja){
    for(int i=0; i<nx; ++i){
      for(int jb=0; jb<4; ++jb){
        int j=ja*4+jb;
        vdata[nx*ja+i][jb] = j < ny ? ndata[nx*j+i] : 0.0;
      }
    }
  }
  // matrix multiplication
  #pragma omp parallel for schedule(static, 1)
  for(int ia=0; ia<na; ++ia){
    for(int ja=ia; ja<na; ++ja){
      double4_t ss00 = double4_0;
      double4_t ss01 = double4_0;
      double4_t ss10 = double4_0;
      double4_t ss11 = double4_0;

      for(int k=0; k<nx; ++k){
        constexpr int PF = 20;
        __builtin_prefetch(&vdata[nx*ia + k + PF]);
        __builtin_prefetch(&vdata[nx*ja + k + PF]);
        double4_t a00 = vdata[nx*ia+k];
        double4_t b00 = vdata[nx*ja+k];
        double4_t a10 = swap2(a00);
        double4_t b01 = swap1(b00);

        ss00 = ss00 + (a00*b00);
        ss01 = ss01 + (a00*b01);
        ss10 = ss10 + (a10*b00);
        ss11 = ss11 + (a10*b01);
      }

      double4_t ss[4] = {ss00, ss01, ss10, ss11};
      for(int kb=1; kb<4; kb+=2){
        ss[kb] = swap1(ss[kb]);
      }
      for(int jb=0; jb<4; ++jb){
        for(int ib=0; ib<4; ++ib){
          int i=ib+ia*4;
          int j=jb+ja*4;
          if(j<ny && i<ny && i<=j){
            result[ny*i+j] = ss[ib^jb][jb];
          }
        }
      }
    }
  }
  free(vdata);
}


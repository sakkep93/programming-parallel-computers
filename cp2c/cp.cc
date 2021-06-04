#include "cp.h"
#include <cmath>
#include <vector>
#include "vector.h"

static inline double hsum4(double4_t ss) {
    double s = 0.0;
    for (int i = 0; i < 4; ++i) {
        s += ss[i];
    }
    return s;
}

double calc_rm(const float* row, int nx){
  double s = 0.0, m;
  for (int j=0; j < nx; j++){
    s += (double) row[j];
  }
  m = s / (double) nx;
  return m;
}

double calc_rrss(const float* row, int nx, double m){
  double square_sum = 0.0, s;
  for(int j=0; j < nx; j++){
    double v = (double) row[j];
    square_sum += ((v-m)*(v-m));
  }
  s = sqrt(square_sum);
  return s;
}

void correlate(int ny, int nx, const float* data, float* result) {
  // elemnts per vector
  int nxb = 4;

  // vectors per row
  int nxa = (nx + nxb - 1) / nxb;

  double4_t* vector_ndata = double4_alloc(ny*nxa);

  // normalize
  for(int j = 0; j < ny; ++j){

    double m = calc_rm(&data[j*nx], nx);
    double rss = calc_rrss(&data[j*nx], nx, m);

    for(int ka=0; ka < nxa; ++ka){
      for(int kb=0; kb < nxb; ++kb){
        int i = ka * nxb + kb;
        vector_ndata[nxa*j + ka][kb] = i < nx ? ((double) data[nx*j + i] - m)/rss : 0.0;
      }
    }
  }

  // matrix multiplication
  for(int i=0; i < ny; i++){
    for(int j=i; j< ny; j++){

      double4_t ss = double4_0;
      for(int ka=0; ka < nxa; ++ka){
        double4_t vx = vector_ndata[nxa*i + ka];
        double4_t vy = vector_ndata[nxa*j + ka];

        ss += (vx * vy);
      }

      result[j+i*ny] = hsum4(ss);
    }
  }

  free(vector_ndata);
}


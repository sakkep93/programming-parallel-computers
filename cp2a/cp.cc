#include "cp.h"
#include <cmath>
#include <vector>

double calculate_m(const float* row, int nx){
  double sum = 0.0, m;
  for(int j=0; j < nx; j++){
    sum += (double) row[j];
  }
  m = sum / (double) nx;
  return m;
}

double calculate_rss(const float* row, int nx, double m){
  double square_sum = 0.0, s;
  for(int j=0; j < nx; j++){
    double v = (double) row[j];
    square_sum += ((v-m)*(v-m));
  }
  s = sqrt(square_sum);
  return s;
}

void correlate(int ny, int nx, const float* data, float* result) {
  // make row lengths of normalized matrix to be a multiple of 4 (padding)
  int nxb = 4;
  int nxa = (nx + nxb - 1) / nxb;
  int nxab = nxa*nxb;

  // reserve heap memory for normalized data
  // instead of reserving ny*nx from heap reserve ny*nxab (increased row length)
  std::vector<double>n_data(ny*nxab, 0.0);

  // normalize
  for(int i=0; i<ny; ++i){
    double m = calculate_m(&data[i*nx], nx);
    double rss = calculate_rss(&data[i*nx], nx, m);
    for(int j=0; j<nx; ++j){
      double rv = (double) data[j+i*nx];
      double v = (rv-m)/rss;
      n_data[j+i*nxab] = v;
    }
  }

  // matrix multiply ndata * ntdata
  for(int i=0; i < ny; i++){
    for(int j=i; j<ny; j++){
      double s[nxb] = {0.0};
      for(int ka=0; ka<nxa; ++ka){
        for(int kb=0; kb < nxb; ++kb){
          double x = n_data[nxab*i + ka * nxb + kb];
          double y = n_data[nxab*j + ka * nxb + kb];
          s[kb] += x*y;
        }
      }
      double Y_ij = 0.0;
      for (int kb=0; kb<nxb; ++kb){
        Y_ij += s[kb];
      }
      result[j+i*ny] = Y_ij;
    }
  }
}


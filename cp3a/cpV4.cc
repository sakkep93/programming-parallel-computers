#include "cp.h"
#include <cmath>
#include "vector.h"

static inline double hsum4(double4_t ss){
  double s = 0.0;
  for(int i=0; i<4; i++){ s += ss[i];}
  return s;
}

double mean(const float* row, int nx){
  double s=0.0, m;
  for (int i=0; i<nx; ++i){ s = s + (double) row[i];}
  m = s / (double) nx;
  return m;
}

double root_square_sum(const float* row, int nx, double m){
  double s = 0.0, r;
  for(int i=0; i<nx; ++i){
    double v = (double) row[i];
    s = s + ((v-m)*(v-m));
  }
  r = sqrt(s);
  return r;
}

void correlate(int ny, int nx, const float* data, float* result) {
  constexpr int nb = 4;
  int na = (nx+nb-1)/nb;
  constexpr int nd = 4;
  int nc = (ny+nd-1)/nd;
  int ncd = nc*nd;
  double4_t* vX = double4_alloc(ncd*na);

  #pragma omp parallel for
  for (int j=0; j < ny; j++){
    double m = mean(&data[j*nx], nx);
    double rss = root_square_sum(&data[j*nx], nx, m);

    for(int ka=0; ka<na; ++ka){
      for(int kb=0; kb<nb; ++kb){
        int i = ka * nb + kb;
        double v = i < nx ? ((double) data[nx*j+i] - m) / rss : 0.0;
        vX[na*j+ka][kb] = v;
      }
    }
  }
  #pragma omp parallel for
  for(int j=ny; j < ncd; ++j){
    for(int ka=0; ka<na; ++ka){
      for(int kb=0; kb<nb; ++kb){
        vX[na*j + ka][kb] = 0.0;
      }
    }
  }

  #pragma omp parallel for schedule(static, 1)
  for(int ic=0; ic<nc; ++ic){
    for(int jc=ic; jc<nc; ++jc){
      double4_t ss[nd][nd];
      for(int id=0; id<nd; ++id){
        for(int jd=0; jd<nd; ++jd){
          ss[id][jd] = double4_0;
        }
      }

      for(int ka=0; ka<na; ++ka){
        double4_t y0 = vX[na*(jc*nd+0)+ka];
        double4_t y1 = vX[na*(jc*nd+1)+ka];
        double4_t y2 = vX[na*(jc*nd+2)+ka];
        double4_t y3 = vX[na*(jc*nd+3)+ka];
        double4_t x0 = vX[na*(ic*nd+0)+ka];
        double4_t x1 = vX[na*(ic*nd+1)+ka];
        double4_t x2 = vX[na*(ic*nd+2)+ka];
        double4_t x3 = vX[na*(ic*nd+3)+ka];
        ss[0][0] = ss[0][0] + (x0 * y0);
        ss[0][1] = ss[0][1] + (x0 * y1);
        ss[0][2] = ss[0][2] + (x0 * y2);
        ss[0][3] = ss[0][3] + (x0 * y3);
        ss[1][0] = ss[1][0] + (x1 * y0);
        ss[1][1] = ss[1][1] + (x1 * y1);
        ss[1][2] = ss[1][2] + (x1 * y2);
        ss[1][3] = ss[1][3] + (x1 * y3);
        ss[2][0] = ss[2][0] + (x2 * y0);
        ss[2][1] = ss[2][1] + (x2 * y1);
        ss[2][2] = ss[2][2] + (x2 * y2);
        ss[2][3] = ss[2][3] + (x2 * y3);
        ss[3][0] = ss[3][0] + (x3 * y0);
        ss[3][1] = ss[3][1] + (x3 * y1);
        ss[3][2] = ss[3][2] + (x3 * y2);
        ss[3][3] = ss[3][3] + (x3 * y3);
      }
      for(int id=0; id<nd; ++id){
        for(int jd=0; jd<nd; ++jd){
          int i = ic*nd + id;
          int j = jc*nd + jd;
          if(i<ny && j<ny && i<=j){
            result[j+i*ny] = hsum4(ss[id][jd]);
          }
        }
      }
    }
  }
  free(vX);
}


#include "mf.h"
#include <vector>
#include <algorithm>

void mf(int ny, int nx, int hy, int hx, const float* in, float* out){

  #pragma omp parallel for
  for(int y=0; y<ny; y++){
    for(int x=0; x<nx; x++){

      // window boundaries
      int xmin = (x-hx) > 0 ? (x-hx) : 0;
      int ymin = (y-hy) > 0 ? (y-hy) : 0;
      int xmax = (x+hx+1) > nx ? nx : (x+hx+1);
      int ymax = (y+hy+1) > ny ? ny : (y+hy+1);

      // window size
      int ws = (xmax-xmin)*(ymax-ymin);

      std::vector<float> w(ws);

      // window fill
      int z = 0;
      #pragma omp parallel for
      for(int i=xmin; i<xmax; ++i){
        for(int j=ymin; j<ymax; ++j){
          w[z++] = in[i+nx*j];
        }
      }

      // find median
      float m, m1, m2;
      if(ws % 2 == 0){
        std::nth_element(w.begin(), w.begin() + ws/2, w.begin()+ws);
        m1 = w[ws/2];
        std::nth_element(w.begin(), w.begin() + ws/2 - 1, w.begin()+ws);
        m2 = w[ws/2 - 1];
        m = (m1+m2) / 2.0;
      }
      else{
        std::nth_element(w.begin(), w.begin() + ws/2, w.begin()+ws);
        m = w[ws/2];
      }
      out[x + nx*y] = m;
    }
  }
}


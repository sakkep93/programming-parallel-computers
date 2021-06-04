#include "is.h"
#include "vector.h"

double4_t* vectorize(int ny, int nx, const float* data){
  double4_t* vdata = double4_alloc(ny*nx);

  #pragma omp parallel for schedule(static, 1)
  for(int y=0; y<ny; ++y){
    for(int x=0; x<nx; ++x){
      int idx = x + nx*y;
      int loc = 3*idx;
      double r = data[loc];
      double g = data[1+loc];
      double b = data[2+loc];
      double4_t c = {r, g, b};
      vdata[idx] = c;
    }
  }
  return vdata;
}


double4_t* precompute_S(int ny, int nx, int pny, int pnx, double4_t* vdata){

  // reserve memory for color component sums S
  double4_t* S = double4_alloc(pnx*pny);

  // initialize S with zeros
  #pragma omp parallel for schedule(static, 1)
  for(int y=0; y<pny; ++y){
    for(int x=0; x<pnx; ++x){
      S[x+pnx*y] = double4_0;
    }
  }

  // calculate vRc for every x, y and c (calculate rectangle color sums where upper left corner is fixed 0,0
  for(int y=0; y<ny; ++y){
    for(int x=0; x<nx; ++x){
      S[(x+1) + pnx*(y+1)] = vdata[x + nx*y] + S[(x+1) + pnx*y] + S[x + pnx*(y+1)] - S[x + pnx*y];
    }
  }
  return S;
}

static inline double scalarize(double4_t c){ return c[0] + (c[1] + c[2]);}

static inline double hfunc(double4_t vXc, double4_t vYc, double Xdiv, double Ydiv){
  return scalarize((vXc * vXc * Xdiv) + (vYc * vYc * Ydiv));
}

struct Optimal{
  int x0;
  int y0;
  int x1;
  int y1;
  double4_t rectC;
  double4_t bgC;
  double _rectSize;
  double _bgSize;
};

Optimal findOptimal(int ny, int nx, int pny, int pnx, double4_t* S){
  // find rectangle coordinates x0, x1, y0, y1 that maximize hfunc
  double imgSize = nx*ny;
  double4_t vPc = S[pnx*pny-1], bvXc = double4_0, bvYc = double4_0;
  double _bXsize = 1, _bYsize = 1, bH = 0.0;
  int by0=0, bx0=0, by1=1, bx1=1;
  #pragma omp parallel
  {
    // thread specific
    double tH = -1, _tXsize=1, _tYsize=1;
    double4_t tvXc = double4_0, tvYc = double4_0;
    int tx0 = 0, ty0 = 0, tx1 = 1, ty1 = 1;
    #pragma omp for schedule(static, 1)
    for (int h=1; h<=ny; h++) {
      for (int w=1; w<=nx; w++) {
        double Xsize = (double)h * (double)w;
        double Ysize = imgSize - Xsize;
        double Xdiv = 1.0 / Xsize;
        double Ydiv = 1.0 / Ysize;

        for (int y0=0; y0<=ny-h; y0++) {
          for (int x0=0; x0<=nx-w; x0++) {
            int y1 = y0 + h;
            int x1 = x0 + w;
            double4_t vXc = S[y1*pnx + x1] - S[y1*pnx + x0] - S[y0*pnx + x1] + S[y0*pnx + x0];
            double4_t vYc = vPc - vXc;
            double H = hfunc(vXc, vYc, Xdiv, Ydiv);

            if (H > tH) {
              tH = H;
              tx0 = x0;
              ty0 = y0;
              tx1 = x1;
              ty1 = y1;
              tvXc = vXc;
              tvYc = vYc;
              _tXsize = Xdiv;
              _tYsize = Ydiv;
            }
          }
        }
      }
    }
    #pragma omp critical
    {
      if (tH > bH) {
        bH = tH;
        bx0 = tx0;
        by0 = ty0;
        bx1 = tx1;
        by1 = ty1;
        bvXc=tvXc;
        bvYc=tvYc;
        _bXsize=_tXsize;
        _bYsize=_tYsize;
      }
    }
  }

  Optimal optimal = {bx0, by0, bx1, by1, bvXc, bvYc, _bXsize, _bYsize};
  return optimal;
}

Result segment(int ny, int nx, const float* data) {
  double4_t* vdata = vectorize(ny, nx, data);

  int pnx = nx+1, pny=ny+1;
  double4_t* S = precompute_S(ny, nx, pny, pnx, vdata);

  Optimal optimal = findOptimal(ny, nx, pny, pnx, S);

  double4_t inner = optimal.rectC * optimal._rectSize;
  double4_t outer = optimal.bgC * optimal._bgSize;

  free(vdata);
  free(S);
  Result result {
    optimal.y0,
    optimal.x0,
    optimal.y1,
    optimal.x1,
    { (float) outer[0], (float) outer[1], (float) outer[2] },
    { (float) inner[0], (float) inner[1], (float) inner[2] }
  };
  return result;
}

#include "mf.h"
#include <vector>
#include <algorithm>
void mf(int ny, int nx, int hy, int hx, const float* in, float* out) {
    int xl, yl, xr, yr, as;
    float m, m1, m2;

    for(auto x=nx; x--;){
        for(auto y=ny; y--;){
            xl = (x-hx) > 0 ? (x-hx) : 0;
            yl = (y-hy) > 0 ? (y-hy) : 0;
            xr = (x+hx+1) > nx ? nx : (x+hx+1);
            yr = (y+hy+1) > ny ? ny : (y+hy+1);
            std::vector<float> a;
            for(auto i=xl; i < xr; i++){for(auto j=yl; j < yr; j++){a.push_back(in[i+nx*j]);}}
            as = (xr - xl) * (yr - yl);
            if(as % 2 == 0){
                std::nth_element(a.begin(), a.begin() + as/2, a.end());
                m1 = a[as/2];
                std::nth_element(a.begin(), a.begin() + as/2 - 1, a.end());
                m2 = a[as/2 - 1];
                m = (m1+m2) / 2.0;
            }else{
                std::nth_element(a.begin(), a.begin() + as/2, a.end());
                m = a[as/2];
            }
            out[x+nx*y] = m;
        }
    }
}


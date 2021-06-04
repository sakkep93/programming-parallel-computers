#include "so.h"
#include <algorithm>
#include <omp.h>
#include <cmath>

void psort(int n, data_t* data) {
  int p = (2 << ((int)(std::log2(omp_get_max_threads())))) /2;
  if(n<p) {
    std::sort(data, data+n);
    return ;
  }
  int psize = n/p;
  int fixed = p*psize;
  #pragma omp parallel num_threads(p)
  {
    int t = omp_get_thread_num();
    std::sort(data+t*psize, data+(t+1)*psize);
    if(t==p-1) std::sort(data+fixed, data+n);
  }

  while(p>1){
    #pragma omp parallel for schedule(static, 1)
    for(int t=0; t<p; t+=2){
      int s = t*psize;
      int e = s+(psize*2);
      std::inplace_merge(data+s, data+(s+psize), data+e);
      if(t==p-2) std::inplace_merge(data+s, data+e, data+n);
    }
    p = (p+1)/2;
    psize = psize * 2;
  }
}


#include "so.h"
#include <algorithm>
#include <omp.h>

data_t* mid3(data_t* first, data_t* mid, data_t* last){
  data_t f = *first;
  data_t m = *mid;
  data_t l = *last;
  if((f < m && m < l) || (l < m && m < f)) return mid;
  if((m < f && f < l) || (l < f && f < m)) return first;
  return last;
}

void quicksort(data_t* start, data_t* end, int nthreads){

  if(nthreads == 1){ std::sort(start, end); return;}
  if(start==end) return;

  data_t p = *mid3(start, (start+(end-start)/2), end-1);
  data_t* lt = std::partition(start, end, [p](data_t em) -> bool {return em < p;});
  data_t* gte = std::partition(lt, end, [p](data_t em) -> bool {return !(p < em);});

  #pragma omp task
  quicksort(start, lt, nthreads-1);

  #pragma omp task
  quicksort(gte, end, nthreads-1);
}

void psort(int n, data_t* data) {

  //int nthreads = (2 << ((int)(std::log2(omp_get_max_threads()))));
  int nthreads = omp_get_max_threads();
  #pragma omp parallel num_threads(nthreads)
  #pragma omp single
  {
    quicksort(data, data+n, nthreads);
  }
}

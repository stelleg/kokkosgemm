#include <Kokkos_Core.hpp>
#include <cstdio>
#include <typeinfo>

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL); 
  printf("Initializing...");
  Kokkos::initialize(argc, argv);
  int n = argc > 1 ? atoi(argv[1]) : 1024;

  float* A = new float[n*n];
  float* B = new float[n*n];
  float* C = new float[n*n];

  float alpha = (float)rand() / (float)RAND_MAX;
  float beta = (float)rand() / (float)RAND_MAX;

  for(size_t i = 0; i < n; ++i) {
      A[i] = (float)rand() / (float)RAND_MAX;
      for(size_t j = 0; j < n ; j++)
        B[n*i+j] = (float)rand() / (float)RAND_MAX;
      C[i] = 0;
  }
  printf("Done.\n");

  printf("Computing (%d x %d x %d) gemm ...", n, n, n);
  Kokkos::parallel_for(
      n, KOKKOS_LAMBDA(const int row) {
        for(size_t col = 0; col < n; col++){
          float sum = 0;
          for(size_t i = 0; i < n; i++)
            sum += A[row*n+i] * B[i*n+col];
          C[row*n+col] = alpha * sum + beta * C[row*n+col];
        }
      });
  printf("Done.\n");
  Kokkos::finalize();
}

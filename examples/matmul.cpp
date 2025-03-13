#include<iostream>
#include "hip/hip_runtime.h"


#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  16
#define THREADS_PER_BLOCK_Y  16
#define THREADS_PER_BLOCK_Z  1


__global__ void vectoradd2D(const float* a, const float* b, float* c, int width, int height) {

  
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  float prod = 0.0;

  for(int i=0; i< WIDTH; i++) {
    r = x + i;
    c = 
    prod += a[r]*b[c];
  //  sum = a[i] + b[i];
  }
    c[i] = prod;
}

using namespace std;

int main() {
  
  float* h_A;
  float* h_B;
  float* h_C;

  float* d_A;
  float* d_B;
  float* d_C;

  int i;
  int errors;

  h_A = new float[NUM * sizeof(float)];
  h_B = new float[NUM * sizeof(float)]; 
  h_C = new float[NUM * sizeof(float)];

  // initialize the input data
  for (i = 0; i < NUM; i++) {
    h_A[i] = (float)i;
    h_B[i] = (float)i*100.0f;
  }
  
  hipMalloc((void**)&d_A, NUM * sizeof(float));
  hipMalloc((void**)&d_B, NUM * sizeof(float));
  hipMalloc((void**)&d_C, NUM * sizeof(float));
  
  hipMemcpy(d_A, h_A, NUM*sizeof(float), hipMemcpyHostToDevice);
  hipMemcpy(d_B, h_B, NUM*sizeof(float), hipMemcpyHostToDevice);

  dim3 blockDim = dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y,1);
  dim3 gridDim  = dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y,1);

  vectoradd2D<<<gridDim,blockDim>>>(d_A ,d_B ,d_C ,WIDTH ,HEIGHT);


  hipMemcpy(h_C, d_C, NUM*sizeof(float), hipMemcpyDeviceToHost);

  // verify the results
  errors = 0;
  for (i = 0; i < NUM; i++) {
    if (h_C[i] != (h_A[i] + h_B[i])) {
      errors++;
    }
  }
  if (errors!=0) {
    printf("FAILED: %d errors\n",errors);
  } else {
      printf ("PASSED!\n");
  }

  hipFree(d_A);
  hipFree(d_B);
  hipFree(d_C);

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;


  return errors;
}

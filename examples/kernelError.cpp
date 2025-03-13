#include <iostream>
#include "hip/hip_runtime.h"
#include "hipCheck.h"


#define WIDTH     1024
#define HEIGHT    1024

#define NUM       (WIDTH*HEIGHT)

#define THREADS_PER_BLOCK_X  1024
#define THREADS_PER_BLOCK_Y  1024
#define THREADS_PER_BLOCK_Z  1

__global__ void vectoradd2D(const float* a, const float* b, float* c, int width, int height) {

  
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;

  int i = y * width + x;
  if ( i < (width * height)) {
   c[i] = a[i] + b[i];
  }
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
  
  HIP_CHECK(hipMalloc((void**)&d_A, NUM * sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&d_B, NUM * sizeof(float)));
  HIP_CHECK(hipMalloc((void**)&d_C, NUM * sizeof(float)));
  
  HIP_CHECK(hipMemcpy(d_A, h_A, NUM*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_B, h_B, NUM*sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_C, 0, NUM*sizeof(float)));

  dim3 blockDim = dim3(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y,1);
  dim3 gridDim  = dim3(WIDTH/THREADS_PER_BLOCK_X, HEIGHT/THREADS_PER_BLOCK_Y,1);

  vectoradd2D<<<gridDim, blockDim>>>(d_A ,d_B ,d_C ,WIDTH ,HEIGHT);
  hipError_t err = hipGetLastError();
  if (err != hipSuccess) {
    cout << "Kernel launch error: " << hipGetErrorString(err) << endl;
  }


  // Read the result back
  HIP_CHECK(hipMemcpy(h_C, d_C, NUM*sizeof(float), hipMemcpyDeviceToHost));

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

  HIP_CHECK(hipFree(d_A));
  HIP_CHECK(hipFree(d_B));
  HIP_CHECK(hipFree(d_C));

  delete[] h_A;
  delete[] h_B;
  delete[] h_C;


  return errors;
}

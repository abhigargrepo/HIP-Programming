/******************************************************************************
Program to calculate Matrix Multiplication of two Matrices.
******************************************************************************/

#include "hip/hip_runtime.h"
#include <stdio.h>

#define HEIGHT_X 3
#define WIDTH_X  4 // Also HEIGHT_Y
#define WIDTH_Y  3

__global__ void matrix_mul(float *X, float *Y, float *Z) {

  float sum = 0.0f;
  // find Row and Column corresponding to a data element for each thread
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;

  // calculate dot product of Row of First Matrix
  //  and Column of Second Matrix
  if (row < HEIGHT_X && col < WIDTH_Y) {
      for (int i = 0; i < WIDTH_X; ++i) {
        float m = X[row * WIDTH_X + i];
        float n = Y[col + WIDTH_Y * i];
        sum += m * n;
      }
  }

  // store dot product at corresponding positon in resultant Matrix
  Z[row * WIDTH_Y + col] = sum;
}

int main() {
  int i, j;

  float *X_h, *Y_h, *Z_h;
  float *X_d, *Y_d, *Z_d;

  // open a file for outputting the result
  FILE *f;
  f = fopen("multiply.txt", "w");

  size_t sizeX = sizeof(float) * HEIGHT_X * WIDTH_X;
  size_t sizeY = sizeof(float) * WIDTH_X * WIDTH_Y;
  size_t sizeZ = sizeof(float) * HEIGHT_X * WIDTH_Y;

  // allocate host side memory
  X_h = (float *)malloc(sizeX);
  Y_h = (float *)malloc(sizeY);
  Z_h = (float *)malloc(sizeZ);

  for (i = 0; i < WIDTH_X; i++) {
    for (j = 0; j < HEIGHT_X; j++) {
      X_h[j * WIDTH_X + i] = 5.0*i; 
    }
  }

  for (i = 0; i < WIDTH_Y; i++) {
    for (j = 0; j < WIDTH_X; j++) {
      Y_h[j * WIDTH_Y + i] = 4.0*j;
    }
  }
  // allocate device memory
  hipMalloc(&X_d, sizeX);
  hipMalloc(&Y_d, sizeY);
  hipMalloc(&Z_d, sizeZ);

  // copy value from host to device
  hipMemcpy(X_d, X_h, sizeX, hipMemcpyHostToDevice);
  hipMemcpy(Y_d, Y_h, sizeY, hipMemcpyHostToDevice);
  printf("\nAfter HostToDevice Memcpy\n%s\n",
         hipGetErrorString(hipGetLastError()));

  // calculate execution configuration
  dim3 blockDim(16, 16);        // each block contains 16 * 16 (=256) threads
  dim3 gridDim((WIDTH_Y + blockDim.x - 1) / blockDim.x, (HEIGHT_X + blockDim.y - 1) / blockDim.y);
  
  // GPU timer code
  float time;
  hipEvent_t start, stop;
  hipEventCreate(&start);
  hipEventCreate(&stop);
  hipEventRecord(start, 0);

  matrix_mul<<<gridDim, blockDim>>>(X_d, Y_d, Z_d);

  hipEventRecord(stop, 0);
  hipEventSynchronize(stop);
  hipEventElapsedTime(&time, start, stop); // time taken in kernel call calculated
  hipEventDestroy(start);
  hipEventDestroy(stop);

  // copy back results
  hipMemcpy(Z_h, Z_d, sizeZ, hipMemcpyDeviceToHost);
  printf("\nAfter DeviceToHost Memcpy\n%s\n", hipGetErrorString(hipGetLastError()));
  fprintf(f, "Array A was---\n");
  for (i = 0; i < HEIGHT_X; i++) {
    for (j = 0; j < WIDTH_X; j++)
      fprintf(f, "%f ", X_h[i * WIDTH_X + j]);
    fprintf(f, "\n");
  }

  fprintf(f, "\nArray B was---\n");
  for (i = 0; i < WIDTH_X; i++) {
    for (j = 0; j < WIDTH_Y; j++)
      fprintf(f, "%f ", Y_h[i * WIDTH_Y + j]);
      fprintf(f, "\n");
  }

  fprintf(f, "\nMultiplication of A and B gives C----\n");
  for (i = 0; i < HEIGHT_X; i++)
  {
    for (j = 0; j < WIDTH_Y; j++)
    fprintf(f, "%f ", Z_h[i * WIDTH_Y + j]);
    fprintf(f, "\n");
  }
    
    printf("\nYou can see output in multiply.txt file in project directory");
    printf("\n\nTime taken is %f (ms)\n", time);
    fprintf(f, "\n\nTime taken is %f (ms)\n", time);
    fclose(f);

    hipFree(X_d);
    hipFree(Y_d);
    hipFree(Z_d);
    free(X_h);
    free(Y_h);
    free(Z_h);

    return 0;
  }

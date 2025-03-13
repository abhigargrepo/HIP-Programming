#include <iostream>             
#include "hip/hip_runtime.h"   
#include "hipCheck.h"          

using namespace std;

int main() {

  float *h_A   = new float[256];
  float *h_B   = new float[256];
  float *device;

  HIP_CHECK(hipMalloc((void **)&device, 256*sizeof(float)));
  
  // Initialize h_A

  for (int i=0; i<256; i++)
      h_A[i] = 2*i;

  // Send data to device
  HIP_CHECK(hipMemcpy(device, h_A, 256*sizeof(float), hipMemcpyHostToDevice));

  // Read data back

  HIP_CHECK(hipMemcpy(h_B, device, 256*sizeof(float), hipMemcpyDeviceToHost));

  // Verify the data
  for (int i=0; i<256; i++)  {
    if (h_A[i] != h_B[i]){
      cout << "failed " << endl;
      exit(1);     
    }
  }

  cout << "passed" << endl;

  delete[] h_A;
  delete[] h_B;
  HIP_CHECK(hipFree(device));

}

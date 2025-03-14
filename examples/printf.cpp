#include <hip/hip_runtime.h>
#include <iostream>


// HIP Kernel for vector addition
__global__ void vectorAdd() {
    printf("I am %d\n",(threadIdx.x + blockIdx.x*blockDim.x));
}

int main() {

    // Launch HIP Kernel
    vectorAdd<<<1, 128>>>();
    return 0;
}

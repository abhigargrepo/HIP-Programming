#define HIP_CHECK(err) do {      \
  hipError_t e = err;            \
  if(e != hipSuccess) {          \
    printf("HIP failure: %s\n",  \
    hipGetErrorString(e));       \
    return -1;                   \
  }                              \
} while(0)

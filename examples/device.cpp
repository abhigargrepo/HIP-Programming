# include <iostream>
# include <hip/hip_runtime.h>
# include "hipCheck.h"

using namespace std;

int main() {
    
   hipDeviceProp_t devProp;
   int count;

   HIP_CHECK(hipGetDeviceCount(&count));             
   HIP_CHECK(hipGetDeviceProperties(&devProp, 4));

   cout << " Device count " << count << endl;
   cout << " agent prop name " << devProp.name << endl;        
   cout << " Total global memory " << devProp.totalGlobalMem << " Bytes" << endl;                                                        
   cout << " Clock Rate " << devProp.clockRate << " KHz" << endl;        
   cout << " Number of CU " << devProp.multiProcessorCount << endl;
   cout << " Max Block Size " << devProp.maxThreadsPerBlock << endl;
   cout << " hip Device prop succeeded " << endl ;               
}

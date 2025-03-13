# include <iostream>
# include <hip/hip_runtime.h>
# include "hipCheck.h"

using namespace std;

int main() {
    
   hipDeviceProp_t devProp;
   int count;

   HIP_CHECK(hipGetDeviceCount(&count));             
   HIP_CHECK(hipGetDeviceProperties(&devProp, 0));

   cout << " Device count " << count << endl;
   cout << " System minor " << devProp.minor << endl;          
   cout << " System major " << devProp.major << endl;          
   cout << " agent prop name " << devProp.name << endl;        
   cout << " Total global memory " << devProp.totalGlobalMem << endl;                                                        
   cout << " Clock Rate " << devProp.clockRate << endl;        
   cout << " Number of CU " << devProp.multiProcessorCount << endl;
   cout << "hip Device prop succeeded " << endl ;               
}

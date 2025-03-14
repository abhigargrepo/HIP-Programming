int main() {

int *hA, *hB, *hC;    // Host Pointers
int *dA, *dB, *dC;    // Device Pointers

hA = new int[NUM_ELEMENTS]; 
hB = new int[NUM_ELEMENTS]; 
hC = new int[NUM_ELEMENTS]; 


// Allocating Memory on Device 

InitVectors(hA, hB);//Initialize hA and hB

// Transfer Data from Host to Device

// Launch HIP Kernel to perform operation

// Transfer Data from Device to Host

delete[] hA;
delete[] hB;
delete[] hC;

// Freeing Device Memory

return 0;
}

# HIP-Programming
It contains HIP programming examples and exercises.

# Steps to compile the code

# Compile for native architecture
hipcc  <test_name>.cpp –o <test_name>.o

# Compile for specific architecture
hipcc --offload-arch=gfx942 <test_name>.cpp –o <test_name>.o

# Run
./<test_name>.o


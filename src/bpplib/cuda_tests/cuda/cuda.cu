#include <cuda_runtime.h>
#include "device_launch_parameters.h"


__global__ void kernel_copy(const unsigned *__restrict__ in, unsigned *__restrict__ out, unsigned count)
{
	unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < count)
		out[idx] = in[idx];
}


extern "C"
void run_kernel_copy(const unsigned *in, unsigned *out, unsigned count)
{
	unsigned blockSize = 1024;
	unsigned blocks = (count + blockSize - 1) / blockSize;
	kernel_copy<<<blocks, blockSize>>>(in, out, count);
}


#include <cuda/cuda.hpp>

#include "../cuda_test.hpp"

#include <random>
#include <iostream>
#include <cstdint>


extern "C"
void run_kernel_copy(const unsigned *in, unsigned *out, unsigned count);


/**
 * \brief Tests the system/stopwatch.hpp using TBB features and tick counter.
 */
class BPPCudaBufferTest : public BPPCudaTest
{
private:

public:
	BPPCudaBufferTest() : BPPCudaTest("buffers") {}

	virtual bool run() const
	{
		// Generate random data ...
		std::default_random_engine rdGen;
		std::uniform_int_distribution<std::uint32_t> rdDist;
		std::vector<std::uint32_t> data(1024*1024);
		for (std::size_t i = 0; i < data.size(); ++i) {
			data[i] = rdDist(rdGen);
		}
		std::size_t devices = bpp::CudaDevice::count();
		for (std::size_t device = 0; device < devices; ++device) {
			std::cout << "Running tests on device #" << device << std::endl;
			CUCH(cudaSetDevice((int)device));
			std::size_t errors = 0;

			/*
			 * Read, write, and copy ...
			 */
			bpp::CudaBuffer<std::uint32_t> cudaBuf1(data.size());
			bpp::CudaBuffer<std::uint32_t> cudaBuf2(data.size());
			bpp::HostBuffer<std::uint32_t> hostBuf(data.size());
			cudaBuf1.write(data);

			run_kernel_copy(*cudaBuf1, *cudaBuf2, (unsigned)data.size());
			CUCH(cudaDeviceSynchronize());

			cudaBuf2.read(hostBuf);

			for (std::size_t i = 0; i < data.size(); ++i) {
				if (data[i] != hostBuf[i])
					++errors;
			}
			if (errors > 0) {
				std::cout << "Total " << errors << " mismatches found in " << data.size() << " data items." << std::endl;
				return false;
			}


			/*
			 * Read/write with offset ...
			 */
			cudaBuf2.write(data, 4096, 65536);
			std::vector<std::uint32_t> tmp(1024);
			cudaBuf2.read(tmp, 1024, 65536 + 1024);
			for (std::size_t i = 0; i < 1024; ++i) {
				if (data[i+1024] != tmp[i])
					++errors;
			}
			if (errors > 0) {
				std::cout << "Total " << errors << " mismatches found in 1024 items (testing offset)." << std::endl;
				return false;
			}

			/*
			 * Memory Fill
			 */
			bpp::CudaBuffer<unsigned char> cudaBuf3(1024);
			cudaBuf3.realloc(4096);
			cudaBuf3.memset(42);
			std::vector<unsigned char> tmp2(cudaBuf3.size());
			cudaBuf3.read(tmp2);
			for (std::size_t i = 0; i < tmp2.size(); ++i) {
				if (tmp2[i] != 42)
					++errors;
			}
			if (errors > 0) {
				std::cout << "Total " << errors << " found in " << tmp2.size() << " char fill." << std::endl;
				return false;
			}
		}

		return true;
	}
};


BPPCudaBufferTest _cudaBufferTest;

#include "../test.hpp"
#include <system/memory_pool.hpp>

#include <iostream>
#include <cstdint>

/**
 * \brief Tests the system/memory_pool.hpp buffer allocation and disposal features.
 */
class BPPSystemMemoryPoolTest : public BPPLibTest
{
public:
	BPPSystemMemoryPoolTest() : BPPLibTest("system/memory_pool/basic") {}

	virtual bool run() const
	{
		bpp::MemoryPool pool(512 * 1024 * sizeof(size_t));
		
		bpp::Buffer<size_t> buf1 = pool.allocateBuffer<size_t>(64 * 1024);
		if (buf1.count() != 64 * 1024) {
			std::cout << "Invalid buffer size " << buf1.count() << ", 65536 expected." << std::endl;
			return false;
		}

		pool.allocateBuffer<size_t>(128 * 1024);
		bpp::Buffer<size_t> buf3 = pool.allocateBuffer<size_t>(256 * 1024);

		try {
			pool.allocateBuffer<size_t>(64 * 1024 + 1);
			std::cout << "Last allocation did not fail due to insufficient memory as expected." << std::endl;
			return false;
		}
		catch (bpp::RuntimeError&) {
			// Nothing to do, exception was expected.
		}

		pool.disposeOf(buf3);
		try {
			pool.disposeOf(buf1);
			std::cout << "Invocation of disposeOf(buf1) should have failed due to wrond dispose order." << std::endl;
			return false;
		}
		catch (bpp::RuntimeError&) {
			// Nothing to do, exception was expected.
		}

		pool.disposeAll();
		try {
			pool.disposeOf(buf1);
			std::cout << "Invocation of disposeOf(buf1) should have failed, since the buf1 is already disposed." << std::endl;
			return false;
		}
		catch (bpp::RuntimeError&) {
			// Nothing to do, exception was expected.
		}
		return true;
	}
};


BPPSystemMemoryPoolTest _systemMemoryPoolTest;

#define _CRT_SECURE_NO_WARNINGS

#include "../test.hpp"
#include <system/file_pool.hpp>

#include <iostream>


/**
 * \brief Tests the system/file_pool.hpp file allocation and disposal.
 */
class BPPSystemFilePoolTest : public BPPLibTest
{
public:
	BPPSystemFilePoolTest() : BPPLibTest("system/file_pool/basic") {}

	virtual bool run() const
	{
		bpp::FilePool pool(".", ".tmp.file_pool_", 3);
		if (pool.newFile().opened() != true) {
			std::cout << "The new file is not opened even if requested so." << std::endl;
			return false;
		}
		if (pool.newFile(false).opened() != false) {
			std::cout << "The new file is opened even if requested so." << std::endl;
			return false;
		}
		pool.newFile().close();
		
		if (pool.count() != 3) {
			std::cout << "Three files were allocated, yet the pool reports " << pool.count() << " files." << std::endl;
			return false;
		}

		pool[0].write<char>("abcd", 4);
		pool.clearAll();
		return true;
	}
};


BPPSystemFilePoolTest _systemFilePoolTest;

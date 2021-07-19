#include "../test.hpp"
#include <system/info.hpp>

#include <iostream>


/**
 * \brief Tests the system/info.hpp feature acquiring the amount of total system memory.
 */
class BPPSystemInfoTotalMemoryTest : public BPPLibTest
{
public:
	BPPSystemInfoTotalMemoryTest() : BPPLibTest("system/info/total_memory") {}

	virtual bool run() const
	{
		size_t totalMem = bpp::SysInfo::getTotalMemory();
		std::cout << "System reports " << totalMem << " bytes of memory." << std::endl;
		return (totalMem >= 1024*1024) && (totalMem <= 0x1000000000000L);
	}
};


BPPSystemInfoTotalMemoryTest _systemInfoTest;

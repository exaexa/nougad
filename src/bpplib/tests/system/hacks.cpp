#include "../test.hpp"
#include <system/hacks.hpp>

#include <iostream>


/**
 * \brief Tests the system/info.hpp feature acquiring the amount of total system memory.
 */
class BPPSystemHacksTrashCPUCachesTest : public BPPLibTest
{
public:
	BPPSystemHacksTrashCPUCachesTest() : BPPLibTest("system/hacks/trash_cpu_caches") {}

	virtual bool run() const
	{
		bpp::trash_cpu_caches();
		return true;
	}
};


BPPSystemHacksTrashCPUCachesTest _systemHacksTrashCPUCachesTest;

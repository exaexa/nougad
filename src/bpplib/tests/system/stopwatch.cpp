#include "../test.hpp"
#include <system/stopwatch.hpp>

#ifdef USE_TBB

#include <tbb/tick_count.h>
#include <tbb/tbb_thread.h>

#include <algorithm>
#include <iostream>
#include <cmath>


/**
 * \brief Tests the system/stopwatch.hpp using TBB features and tick counter.
 */
class BPPSystemStopwatchTBBTest : public BPPLibTest
{
private:
	bool testInterval(double sec) const
	{
		sec = std::fabs(sec);

		bpp::Stopwatch stopwatch(false);
		tbb::tick_count start = tbb::tick_count::now();
		stopwatch.start();
		
		tbb::this_tbb_thread::sleep(tbb::tick_count::interval_t(sec));

		tbb::tick_count end = tbb::tick_count::now();
		stopwatch.stop();

		double diff = std::fabs((end-start).seconds() - stopwatch.getSeconds());
		double tolerance = std::max<double>(0.001, sec/20.0);
		if (diff > tolerance) {
			std::cout << "Measuring sleep(" << sec << ") failed. TBB measured " << (end-start).seconds() << "s, while Stopwatch measured "
				<< stopwatch.getSeconds() << "s. The tolerance was " << tolerance << "s." << std::endl;
			return false;
		}

		return true;
	}


public:
	BPPSystemStopwatchTBBTest() : BPPLibTest("system/stopwatch/tbb-verification") {}

	virtual bool run() const
	{
		double tests[] = { 0.0, 0.1, 1.0 };
		size_t testsCount = sizeof(tests) / sizeof(double);

		for (size_t i = 0; i < testsCount; ++i)
			if (!testInterval(tests[i]))
				return false;

		return true;
	}
};


BPPSystemStopwatchTBBTest _systemStopwatchTBBTest;

#endif
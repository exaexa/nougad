#ifdef USE_TBB

#include "../test.hpp"
#include <para/thread_pool_old.hpp>
#include <math/random.hpp>

#include <vector>
#include <iostream>


/**
 * \brief Tests the system/info.hpp feature acquiring the amount of total system memory.
 */
class BPPParaThreadPoolOldSumTest : public BPPLibTest
{
private:
	template<class T>
	class TestThreadTask : public bpp::ThreadTask<T>
	{
	private:
		const std::vector<T> &mData;
		size_t mOffset, mCount;
		T mResult;

	protected:
		virtual void execute(T &context)
		{
			for (size_t i = 0; i < mCount; ++i)
				mResult += mData[mOffset + i];
			context += mResult;
		}

	public:
		TestThreadTask(const std::vector<T> &data, size_t offset, size_t count)
			: mData(data), mOffset(offset), mCount(count), mResult(0)
		{}

		T getResult() const { return mResult; }
	};


public:
	BPPParaThreadPoolSumTest() : BPPLibTest("para/thread_pool_old/sum") {}

	virtual bool run() const
	{
		size_t correctSum = 0;
		std::vector<size_t> data(10000);
		for (size_t i = 0; i < 10000; ++i) {
			data[i] = bpp::Random<size_t>::next(1, 1000);
			correctSum += data[i];
		}

		bpp::ThreadPool<size_t> threadPool(4);
		for (size_t i = 0; i < 10; ++i) {
			TestThreadTask<size_t> *task = new TestThreadTask<size_t>(data, i*1000, 1000);
			threadPool.enqueue(*task);
		}

		size_t res = 0, done = 0;
		while (!threadPool.empty()) {
			bpp::ThreadTask<size_t> &task = threadPool.getCompletedTask();
			TestThreadTask<size_t> *testTask = dynamic_cast<TestThreadTask<size_t>*>(&task);
			
			if (testTask == nullptr)
				throw (bpp::RuntimeError() << "Invalid task type returned from the thread pool.");

			if (testTask->failed())
				throw (bpp::RuntimeError() << "Testing task failed: " << testTask->getErrorMessage());

			res += testTask->getResult();
			delete testTask;
			++done;
		}

		if (done != 10) {
			std::cout << "Total 10 tasks were spawned, but " << done << " tasks were yielded from the thread pool." << std::endl;
			return false;
		}
		if (res != correctSum) {
			std::cout << "Tasks yielded wrong result " << res << ", " << correctSum << " was expected." << std::endl;
			return false;
		}

		res = 0;
		std::cout << "Accumulated contexts:";
		for (size_t i = 0; i < threadPool.workerCount(); ++i) {
			std::cout << " " << threadPool.getContext(i);
			res += threadPool.getContext(i);
		}
		std::cout << std::endl;

		if (res != correctSum) {
			std::cout << "Accumulated contexts hold a wrong result " << res << ", " << correctSum << " was expected." << std::endl;
			return false;
		}

		return true;
	}
};


BPPParaThreadPoolSumTest _paraThreadPoolSumTest;

#endif

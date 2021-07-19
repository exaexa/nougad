#include "../test.hpp"
#include <para/thread_pool.hpp>
#include <math/random.hpp>
#include <misc/ptr_fix.hpp>

#include <vector>
#include <iostream>


/**
 * \brief Tests the system/info.hpp feature acquiring the amount of total system memory.
 */
class BPPParaThreadPoolSumTest : public BPPLibTest
{
private:
	class TestTask : public bpp::ThreadPool::Task
	{
	private:
		const std::vector<std::size_t> &mData;
		std::size_t mOffset, mCount;
		std::size_t mResult;

	protected:
		virtual void execute(bpp::ThreadPool &pool)
		{
			mResult = 0;
			for (size_t i = 0; i < mCount; ++i)
				mResult += mData[mOffset + i];
		}

	public:
		TestTask(const std::vector<std::size_t> &data, std::size_t offset, std::size_t count)
			: mData(data), mOffset(offset), mCount(count), mResult(0)
		{}

		std::size_t getResult() const { return mResult; }
	};


public:
	BPPParaThreadPoolSumTest() : BPPLibTest("para/thread_pool/sum") {}

	virtual bool run() const
	{
		std::size_t correctSum = 0;
		std::vector<size_t> data(10000);
		for (size_t i = 0; i < 10000; ++i) {
			data[i] = bpp::Random<size_t>::next(1, 1000);
			correctSum += data[i];
		}

		std::cout << "Initializing thread pool and sending tasks ..." << std::endl;
		bpp::ThreadPool threadPool(4, 8);
		std::vector< std::shared_ptr<TestTask> > tasks(10);
		for (std::size_t i = 0; i < tasks.size(); ++i) {
			tasks[i] = bpp::make_shared<TestTask>(data, i*1000, 1000);
			threadPool.addTask(tasks[i]);
		}

		std::cout << "Finalizing tasks ..." << std::endl;
		threadPool.finalize();

		std::size_t res = 0;
		for (std::size_t i = 0; i < tasks.size(); ++i) {
			res += tasks[i]->getResult();
		}

		std::cout << "Terminating thread pool ..." << std::endl;
		threadPool.terminate();

		if (res != correctSum) {
			std::cerr << "Computed sum is " << res << ", but " << correctSum << " was expected.";
			return false;
		}
		else
			return true;
	}
};


BPPParaThreadPoolSumTest _paraThreadPoolSumTest;





/**
 * \brief Checks whether exceptions in tasks are propagated correctly.
 * \note This test is not executed by default, since it aborts the whole application.
 */
class BPPParaThreadPoolExceptionsTest : public BPPLibTest
{
private:
	class TestTask : public bpp::ThreadPool::Task
	{
	protected:
		std::size_t mId;

		virtual void execute(bpp::ThreadPool &pool)
		{
			throw (bpp::RuntimeError() << "Testing exception from task #" << mId);
		}

	public:
		TestTask(std::size_t id) : mId(id) {}
	};


public:
	BPPParaThreadPoolExceptionsTest() : BPPLibTest("para/thread_pool/exceptions") {}

	virtual bool run() const
	{
		try {
			std::cout << "Initializing thread pool and sending exceptional tasks ..." << std::endl;
			bpp::ThreadPool threadPool(4, 8);
			std::vector< std::shared_ptr<TestTask> > tasks(5);
			for (std::size_t i = 0; i < tasks.size(); ++i) {
				tasks[i] = bpp::make_shared<TestTask>(i);
				threadPool.addTask(tasks[i]);
			}

			std::cout << "Finalizing ..." << std::endl;
			threadPool.finalize();
		}
		catch (std::exception &e) {
			std::cout << "Exception caught: " << e.what() << std::endl;
		}
		return true;
	}
};


//BPPParaThreadPoolExceptionsTest _paraThreadPoolExceptionsTest;

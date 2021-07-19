/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 25.9.2015
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 *
 * New version of thread pool based on C++11 threads and sync. primitives.
 */
#ifndef BPPLIB_PARA_THREAD_POOL_HPP
#define BPPLIB_PARA_THREAD_POOL_HPP

#include <para/blocking_queue.hpp>
#include <misc/exception.hpp>


#include <thread>
#include <vector>
#include <memory>
#include <atomic>
#include <sstream>
#include <string>
#include <exception>


namespace bpp
{
/**
 * \brief Specific exception that is thrown when internal error in thread pool or thread task occurs.
 */
class ThreadException : public RuntimeError
{
public:
	ThreadException() : RuntimeError() {}
	ThreadException(const char *msg) : RuntimeError(msg) {}
	ThreadException(const std::string &msg) : RuntimeError(msg) {}
	virtual ~ThreadException() throw() {}


	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	ThreadException& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};




/**
 * \brief Simple thread pool which executes tasks.
 * \tparam CTX Type of context of each thread (user-initialized thread-local object).
 */
template<typename CTX = void*>
class ThreadPoolWithContext
{
public:
	typedef CTX context_t;

	/**
	 * \brief Base class for all tasks. Defines execution interface.
	 */
	class Task
	{
	public:
		virtual ~Task() {}
		virtual void execute(bpp::ThreadPoolWithContext<context_t> &pool) {}
		virtual void execute(bpp::ThreadPoolWithContext<context_t> &pool, context_t &context)
		{
			// The default implementation is to execute the first method which does not use context.
			// This is a hack for typedefed bpp::ThreadPool tasks, which act as if they do not have any context.
			this->execute(pool);
		}
	};


private:
	std::atomic<bool> mTerminating;		///< Flag indicating that shutdown procedure has been started.
	std::atomic<std::size_t> mPending;	///< Number of pending tasks (both in queue and executed by threads).
	
	/*
	 * \brief Global task queue.
	 */
	bpp::BlockingQueue< std::shared_ptr<Task> > mTaskQueue;
	std::vector<std::thread> mThreads;
	std::vector<context_t> mThreadContexts;

	std::mutex mFinalizeMutex;
	std::condition_variable mFinalizeCV;



	void createThreads(std::size_t threads)
	{
		if (threads == 0)
			throw ThreadException("Thread pool cannot be empty.");

		// Start the threads.
		mThreads.reserve(threads);
		for (std::size_t i = 0; i < threads; ++i) {
			mThreads.push_back(std::thread(&ThreadPoolWithContext<context_t>::threadWorker, this, i));
		}
	}


	void threadWorker(std::size_t threadIdx)
	{
		while (true) {
			// Get (wait for) another task from the queue.
			std::shared_ptr<Task> task = mTaskQueue.pop();
			if (mTerminating.load())
				break;
			
			task->execute(*this, mThreadContexts[threadIdx]);

			// Reduce number of pending operations.
			std::size_t pending;
			{
				std::lock_guard<std::mutex> lock(mFinalizeMutex);
				pending = mPending.fetch_sub(1);
			}
			if (pending == 1)		// this was the last pending operation
				mFinalizeCV.notify_all();
		}

		// Make sure that there is something in the queue, so that other threads are woken and may shut down.
		mTaskQueue.tryPush(std::shared_ptr<Task>(new Task));
	}


public:
	/*
	 * \brief Creates a thread pool of given number of threads.
	 * \param threads Number of threads being spawned (must be > 0).
	 * \param queueCapacity Optional parameter -- length of the task queue.
	 *		If zero, the task queue length is equal to number of threads.
	 */
	ThreadPoolWithContext(std::size_t threads = std::thread::hardware_concurrency(), std::size_t queueCapacity = 0, context_t context = context_t())
		: mTerminating(false), mPending(0), mTaskQueue((queueCapacity > 0) ? queueCapacity : threads)
	{
		createThreads(threads);

		// Initialize all thread contexts with a copy of given context.
		mThreadContexts.reserve(threads);
		for (std::size_t i = 0; i < threads; ++i)
			mThreadContexts.push_back(context);
	}


	/*
	 * \brief Creates a thread pool of given number of threads.
	 * \param threads Number of threads being spawned (must be > 0).
	 * \param queueCapacity Optional parameter -- length of the task queue.
	 *		If zero, the task queue length is equal to number of threads.
	 */
	ThreadPoolWithContext(std::vector<context_t> &contexts, std::size_t queueCapacity = 0)
		: mTerminating(false), mPending(0), mTaskQueue((queueCapacity > 0) ? queueCapacity : contexts.size())
	{
		createThreads(contexts.size());

		// Initialize all thread contexts by moving values from given context vector.
		mThreadContexts.resize(contexts.size());
		for (std::size_t i = 0; i < contexts.size(); ++i)
			mThreadContexts[i] = std::move(contexts[i]);
		contexts.clear();
	}


	/*
	 * \brief Destructor also attempts to shut down the pool.
	 */
	~ThreadPoolWithContext()
	{
		try {
			terminate();
		}
		catch (std::exception&) {}
	}


	std::size_t threads() const
	{
		return mThreads.size();
	}


	/**
	 * \brief Get number of pending tasks.
	 */
	std::size_t pending()
	{
		return mPending.load();
	}


	/*
	 * \brief Add another task to be processed in the pool.
	 * \param task Shared pointer to a task that is being enqueued.
	 */
	void addTask(std::shared_ptr<Task> task)
	{
		if (mTerminating.load())
			throw ThreadException("The thread pool is already terminating. No more tasks can be added.");
		++mPending;
		mTaskQueue.push(task);
	}


	/*
	 * \brief Attempts to add a task to the pool, fails if the queue is currently full.
	 * \param task Shared pointer to a task that is being enqueued.
	 * \return True if the task was successfully added, false if the queue is full.
	 */
	bool tryAddTask(std::shared_ptr<Task> task)
	{
		if (mTerminating.load())
			throw ThreadException("The thread pool is already terminating. No more tasks can be added.");
		if (mTaskQueue.tryPush(task)) {
			++mPending;
			return true;
		}
		else
			return false;
	}


	/*
	 * \brief Waits until all tasks pushed to the queue are completed.
	 */
	void finalize()
	{
		if (mTerminating.load())
			throw ThreadException("The thread pool is already terminating. Cannot wait for finalization.");

		std::unique_lock<std::mutex> lock(mFinalizeMutex);
		while (mPending.load() != 0) {
			mFinalizeCV.wait(lock);
			if (mTerminating.load())
				throw ThreadException("The thread pool is being terminated.");
		}
	}


	/*
	 * \brief Clears the queue and shuts down the threads.
	 *		The operation blocks until the threads are joined.
	 */
	void terminate()
	{
		if (mTerminating.exchange(true))	// set the terminating flag to true
			return;							// if it already was true, a shutdown operation is under way
		
		// Notify waiters on finalization.
		mFinalizeCV.notify_all();

		// Clear pending tasks and insert one empty for each thread so they get woken.
		mTaskQueue.clear();
		for (std::size_t i = 0; i < mThreads.size(); ++i)
			mTaskQueue.tryPush(std::shared_ptr<Task>(new Task));	// failures are ignored

		// Wait for all threads to finish.
		for (auto &th : mThreads)
			th.join();
		mThreads.clear();
	}


	/**
	 * \brief Return context of selected thread.
	 * \param tid Index of a thread whose context is returned.
	 * \warn This method is not thread safe, but it may be used
	 *		to access contexts safely when no tasks are being processed.
	 */
	CTX& getThreadContext(std::size_t tid)
	{
		return mThreadContexts[tid];
	}
};



/*
 * Simple thread pool with void context ...
 */
typedef ThreadPoolWithContext<> ThreadPool;

}

#endif

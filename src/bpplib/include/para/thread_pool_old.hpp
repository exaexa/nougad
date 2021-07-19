/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 1.7.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 *
 * This version will probably be deprecated
 */
#ifndef BPPLIB_PARA_THREAD_POOL_OLD_HPP
#define BPPLIB_PARA_THREAD_POOL_OLD_HPP
#ifdef USE_TBB

#include <misc/exception.hpp>

#include <tbb/concurrent_queue.h>
#include <tbb/tbb_thread.h>

#include <vector>
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
 * \brief A special empty class representing void context.
 */
class ThreadEmptyContext {};


// Forward Declaration
template<class WRKCTX> class ThreadPool;





/**
 * \brief Base class for thread pool tasks.
 *
 * These tasks are specialy designed for combined CPU-OpenCL processing.
 * Typical situations in which these tasks help is, when CPU and GPU needs
 * to work closely together (CPU needs to prepare data or calls GPU iteratively).
 */
template<class WRKCTX = ThreadEmptyContext>
class ThreadTask
{
friend class ThreadPool<WRKCTX>;
private:
	volatile bool mEnqueued;	///< Whether the task is currently processed by thread pool.
	volatile bool mCompleted;	///< Whether the task has completed its work.
	volatile bool mError;		///< Whether an error (exception) occured during the execution.
	std::string mErrorMsg;		///< If error occured, this contains the exception message.

protected:
	/**
	 * \brief The task main method. Should be implemented by derived classes.
	 * \param endPoint Reference to existing OpenCL toolkit EndPoint.
	 */
	virtual void execute(WRKCTX &context) = 0;

public:
	ThreadTask() : mEnqueued(false), mCompleted(false), mError(false) {}
	
	virtual ~ThreadTask() {}	// enforce virtual destructor for descendants


	/**
	 * \brief Check whether the task is in thread pool queue or being processed by a thread.
	 */
	bool isEnqueued() const
	{
		return mEnqueued;
	}


	/**
	 * \brief Check whether the task is completed.
	 */
	bool isCompleted() const
	{
		return mCompleted;
	}


	/**
	 * \brief Check whether task has failed the execution.
	 *		Task fails if an exception is thrown.
	 * \note This value is valid only when the task is completed.
	 */
	bool failed() const
	{
		return mError;
	}


	/**
	 * \brief Return an exception error message if the task failed.
	 */
	const std::string& getErrorMessage() const
	{
		return mErrorMsg;
	}


	/**
	 * \brief Reset the task, so it can be recycled for execution.
	 * \note The task may be recycled only when not processed by thread pool.
	 */
	virtual void reset()
	{
		if (isEnqueued())
			throw (ThreadException() << "Cannot perform a task reset while the task is still enqueued in the thread pool.");
		mCompleted = false;
		mError = false;
	}
};





/**
 * \brief Trivial implementation of a task that does nothing.
 */
template<class WRKCTX = ThreadEmptyContext>
class ThreadEmptyTask : public ThreadTask<WRKCTX>
{
protected:
	virtual void execute(WRKCTX &context) {}
};





/**
 * \brief Managed thread pool with a single pair of I/O queues.
 * \para, WRKCTX Type of the worker thread context. The context is available to the
 *		tasks 
 */
template<class WRKCTX = ThreadEmptyContext>
class ThreadPool
{
public:
	typedef ThreadTask<WRKCTX> task_t;	// Type of the thread task class.

private:
	/**
	 * \brief Internal class wrapping one worker of the thread pool.
	 *		The worker is used both as functor for thread body and as TBB thread container.
	 *
	 * A worker is provided with a context that can be accessed by the tasks.
	 */
	class Worker {
	private:
		ThreadPool &mThreadPool;			///< Reference to parent thread pool.
		tbb::tbb_thread *mThread;			///< A TBB thread object (iff the worker is running).

	public:
		WRKCTX mContext;					///< Worker context that is accessible by the tasks.

		/**
		 * \brief Initialize (but not launch) the worker.
		 * \param tp Parent thread pool object (who created and owns the thread).
		 */
		Worker(ThreadPool &tp, WRKCTX context)
			: mThreadPool(tp), mContext(context), mThread(nullptr)
		{}


		/**
		 * \brief Launch the worker (create underlying thread and run the main loop).
		 */
		void start()
		{
			if (mThread) return;
			mThread = new tbb::tbb_thread(*this, &mContext);
		}


		/**
		 * \brief Wait for the thread to terminate.
		 * \note Join SHOULD be called before the worker is destroyed.
		 */
		void join()
		{
			if (mThread == nullptr) return;
			mThread->join();
			delete mThread;
			mThread = nullptr;
		}


		/**
		 * \brief Main method of the functor, that becomes the entry point of the thread code.
		 */
		void operator()(WRKCTX *context)
		{
			while (true) {
				// Get next task for processing.
				task_t *task = nullptr;
				while (task == nullptr)
					mThreadPool.mInQueue.pop(task);

				// If the poison pill was swallowed ...
				if (task == &mThreadPool.mPoisonPill) {
					mThreadPool.mInQueue.push(task);	// Make sure others get it as well,
					return;								// ... and die.
				}

				// Try to execute the task.
				try {
					task->execute(*context);
				}
				catch (ThreadException &e) {
					task->mError = true;
					std::stringstream msg;
					msg << "Internal Thread Error: " << e.what();
					task->mErrorMsg = msg.str();
				}
				catch (std::exception &e) {
					task->mError = true;
					std::stringstream msg;
					msg << "Unhandled Exception: " << e.what();
					task->mErrorMsg = msg.str();
				}
			
				task->mCompleted = true;
				mThreadPool.mOutQueue.push(task);
			}
		}
	};


	/*
	 * Member Variables
	 */

	std::size_t mPendingTasks;	///< Number of pending tasks (enqueued, but not picked from out queue).

	tbb::concurrent_bounded_queue<task_t*> mInQueue;	///< Queue of input tasks waiting to be processed.
	tbb::concurrent_bounded_queue<task_t*> mOutQueue;	///< Queue of completed output tasks.
	std::vector<Worker*> mWorkers;						///< Pool of worker threads (allocated by constructor).

	ThreadEmptyTask<WRKCTX> mPoisonPill;				///< Special tasks that forces the worker to terminate.


	// Disable copy constructor and assignment operator.
	ThreadPool(const ThreadPool&) {}
	void operator=(const ThreadPool&) {}


public:
	/**
	 * \brief Initialize the thread pool and spawn the worker threads.
	 *
	 * \param threads Max number of tasks enlisted in input queue.
	 * \param inQueueCapacity Max number of tasks enlisted in input queue.
	 * \param outQueueCapacity Max number of tasks waiting in output queue.
	 * \param contexts Pointer to an array containing contexts for initialization of each worker.
	 *		The array must contain at least as many items as there are threads.
	 *		If nullptr, the contexts are intialized by default constructors.
	 *
	 * TODO: a feature of on-demand worker spawning may be added in future
	 */
	ThreadPool(std::size_t threads, std::size_t inQueueCapacity = 1024, std::size_t outQueueCapacity = 1024, WRKCTX *contexts = nullptr)
		: mPendingTasks(0)
	{
		if (threads == 0)
			throw (ThreadException() << "Thread pool must contain at least one thread.");

		if (threads > 4096)
			throw (ThreadException() << "Unable to create pool of " << threads << " threads. Current maximum is 4096.");

		mInQueue.set_capacity(inQueueCapacity);
		mOutQueue.set_capacity(outQueueCapacity);

		for (std::size_t i = 0; i < threads; ++i) {
			if (contexts == nullptr)
				mWorkers.push_back(new Worker(*this, WRKCTX()));
			else
				mWorkers.push_back(new Worker(*this, contexts[i]));
			mWorkers.back()->start();
		}
	}
	

	/**
	 * \brief The destructor poisons the threads and waits for them to die.
	 */
	~ThreadPool()
	{
		mInQueue.clear();
		mOutQueue.clear();

		// Poison the input.
		enqueue(mPoisonPill);

		// And wait for the workers to die.
		for (std::size_t i = 0; i < mWorkers.size(); ++i) {
			mWorkers[i]->join();
			delete mWorkers[i];
		}
	}


	/**
	 * \brief Return the number of allocated thread workers.
	 */
	std::size_t workerCount() const
	{
		return mWorkers.size();
	}


	/**
	 * \brief Enqueue another task to be processed.
	 * \param task Task object to be processed. The task should not
	 *		be modified until it is released by the thread pool.
	 * \note The execution of enqueue may block if the input queue is full.
	 *		If maximal number of pending tasks is exceeded,
	 *		an exception is thrown so that deadlock is avoided.
	 */
	void enqueue(task_t &task)
	{
		if (task.isCompleted())
			throw (ThreadException() << "Cannot enqueue already completed task.");

		// Deadlock Prevention
		if (full())
			throw (ThreadException() << "Number of pending tasks would exceed the queue capacities."
				<< " This action would cause a deadlock since the enqueue() is a blocking operation.");

		// Proceed with enlisting.
		task.mEnqueued = true;
		mInQueue.push(&task);
		++mPendingTasks;
	}


	/**
	 * \brief Return the number of pending tasks (enlisted, being processed,
	 *		or waiting to be reclaimed from output queue).
	 */
	std::size_t getPendingTasks() const
	{
		return mPendingTasks;
	}


	/**
	 * \brief Check whether all queues are full and all threads are occupied.
	 * \return True if the pool is full and another enqueue would fail, false otherwise.
	 */
	bool full() const
	{
		return mPendingTasks >= mInQueue.capacity() + mOutQueue.capacity() + mWorkers.size();
	}


	/**
	 * \brief Check whether there are no pending tasks.
	 */
	bool empty() const
	{
		return mPendingTasks == 0;
	}


	/**
	 * \brief Check whether there are any completed tasks waiting in the output queue.
	 */
	bool completedTasksWaiting() const
	{
		return !mOutQueue.empty();
	}


	/**
	 * \brief Reclaim next completed task from the output queue.
	 *		If there are completed tasks yet, the call blocks until
	 *		any task complets.
	 * \throws CLException if there are no pending tasks in the pool
	 *		and the dealock would occur.
	 */
	task_t& getCompletedTask()
	{
		if (mPendingTasks == 0)
			throw (ThreadException() << "There are no pending tasks in the thread pool.");

		task_t *task = nullptr;
		while (task == nullptr)
			mOutQueue.pop(task);
		--mPendingTasks;
		task->mEnqueued = false;
		return *task;
	}


	/**
	 * \brief Retrieve workers context.
	 * \param workerId The (zero-based) index of the worker.
	 * \warning This operation is NOT SAFE if there are any pending tasks in the pool.
	 */
	WRKCTX getContext(std::size_t workerId) const
	{
		if (workerId >= mWorkers.size())
			throw (ThreadException() << "Worker ID " << workerId << " is out of range since only "
				<< mWorkers.size() << " workers are present in the pool.");

		return mWorkers[workerId]->mContext;
	}


	/**
	 * \brief Set workers context.
	 * \param workerId The (zero-based) index of the worker.
	 * \warning This operation is NOT SAFE if there are any pending tasks in the pool.
	 */
	void setContext(std::size_t workerId, WRKCTX context)
	{
		if (workerId >= mWorkers.size())
			throw (ThreadException() << "Worker ID " << workerId << " is out of range since only "
				<< mWorkers.size() << " workers are present in the pool.");

		mWorkers[workerId]->mContext = context;
	}
};


}

#endif
#endif

/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 5.7.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_SYSTEM_FILE_ASYNC_HPP
#define BPPLIB_SYSTEM_FILE_ASYNC_HPP
#ifdef USE_TBB

#include <system/memory_pool.hpp>
#include <system/file.hpp>
#include <para/thread_pool_old.hpp>

#include <set>
#include <map>
#include <cassert>


namespace bpp
{

/**
 * \brief Manager for asynchronous file operations (read/write).
 */
class FileAsyncOps
{
public:
	/**
	 * \brief A special token that is yielded by every async operation.
	 *		The user may test the token or wait for the completion of the
	 *		corresponding operation.
	 */
	class Promise
	{
	friend FileAsyncOps;
	private:
		FileAsyncOps *mFileAsyncOps;	///< The async operation manager that issued this promise.
		size_t mId;						///< Unique promise ID.

		Promise(FileAsyncOps &fileAsyncOps, size_t id)
			: mFileAsyncOps(&fileAsyncOps), mId(id) {}

	public:
		/**
		 * \brief Create an empty promise.
		 */
		Promise() : mFileAsyncOps(nullptr), mId(0) {}

		bool isValid() const
		{
			return mFileAsyncOps != nullptr;
		}

		/**
		 * \brief Check, whether the operation has terminated.
		 */
		bool isDone()
		{
			if (!isValid())
				throw (LogicError() << "Selected promise object is not valid.");
			return mFileAsyncOps->isDone(mId);
		}


		/**
		 * \brief Wait for the operation to terminate (blocking call).
		 */
		void waitFor()
		{
			if (!isValid())
				throw (LogicError() << "Selected promise object is not valid.");
			mFileAsyncOps->waitFor(mId);
		}


		/**
		 * \brief Check whether the operation succeeded.
		 *		This method can be invoked only AFTER the operation has terminated (i.e., isDone() == true).
		 * \throws LogicError if the operation has not terminated yet.
		 */
		bool isOK()
		{
			if (!isValid())
				throw (LogicError() << "Selected promise object is not valid.");
			return mFileAsyncOps->isOK(mId);
		}


		/**
		 * \brief Retrieve an error message in case the operation failed.
		 * \throws LogicError if the operation has not terminated or terminated successfully.
		 */
		const std::string& getErrorMessage()
		{
			if (!isValid())
				throw (LogicError() << "Selected promise object is not valid.");
			return mFileAsyncOps->getErrorMessage(mId);
		}


		/**
		 * \brief Remove the error message and override the operation as successful.
		 * \throws LogicError if the operation has not terminated yet.
		 */
		void clearError()
		{
			if (!isValid())
				throw (LogicError() << "Selected promise object is not valid.");
			mFileAsyncOps->clearError(mId);
		}
	};


private:
	/**
	 * \brief Base class for internal async operation tasks.
	 */
	class Task : public ThreadTask<>
	{
	protected:
		size_t mId;		///< Internal ID of the corresponding Promise object.
		File &mFile;	///< File being used for the async operation.

	public:
		Task(size_t id, File &file) : mId(id), mFile(file) {}
		size_t getId() const	{ return mId; }
	};


	/**
	 * \brief Internal class for reading tasks.
	 * \tparam T Type of the data items being read (passed to the File::read<T>() method template).
	 */
	template<typename T>
	class ReadTask : public Task
	{
	private:
		T* mBuffer;		///< Buffer where the data are to be stored.
		size_t mCount;	///< Number of data items being read.

	protected:
		virtual void execute(ThreadEmptyContext &context)
		{
			mFile.read<T>(mBuffer, mCount);
		}

	public:
		ReadTask(size_t id, File &file, T* buffer, size_t count)
			: Task(id, file), mBuffer(buffer), mCount(count) {}
	};


	/**
	 * \brief Internal class for writing tasks.
	 * \tparam T Type of the data items being written (passed to the File::write<T>() method template).
	 */
	template<typename T>
	class WriteTask : public Task
	{
	private:
		const T* mBuffer;	///< Buffer from which the data are written to file.
		size_t mCount;		///< Number of data items being written.
		bool mFlush;		///< Whether a flush has to be performed after the write.

	protected:
		virtual void execute(ThreadEmptyContext &context)
		{
			mFile.write<T>(mBuffer, mCount);
			if (mFlush)
				mFile.flush();
		}

	public:
		WriteTask(size_t id, File &file, const T* buffer, size_t count, bool flush)
			: Task(id, file), mBuffer(buffer), mCount(count), mFlush(flush) {}
	};



	/*
	 * Member Variables
	 */
	ThreadPool<> mThreadPool;				///< Thread pool that executes the tasks.
	std::set<size_t> mPendingTasks;			///< Set of pending task (promise) IDs.
	std::map<size_t, Task*> mFailedTasks;	///< A map (promiseID - task object) of completed failed tasks.
	size_t mNextId;							///< First promiseID which has not been assigned yet.



	/*
	 * Private Methods
	 */

	/**
	 * \brief Final processing of a task that was acquired from the output queue.
	 *		The task is deleted if successful or moved to failed tasks list.
	 * \param task The task object being finalized.
	 */
	void processCompletedTask(Task* task)
	{
		assert(task != nullptr);

		// Check and update pending tasks list.
		std::set<size_t>::iterator it = mPendingTasks.find(task->getId());
		if (it == mPendingTasks.end())
			throw (RuntimeError() << "Task ID " << task->getId() << " is not on the list of pending tasks.");
		mPendingTasks.erase(it);

		if (task->failed()) {
			// Save the failure message if necessray.
			if (mFailedTasks.find(task->getId()) != mFailedTasks.end())
				throw (RuntimeError() << "Task ID " << task->getId() << " is already on the list of failed tasks.");
			mFailedTasks[task->getId()] = task;
		}
		else
			// Remove the task if it succeded.
			delete task;
	}


	/**
	 * \brief Finalize all tasks that are waiting in the output queue.
	 */
	void processAllCompletedTasks()
	{
		while (mThreadPool.completedTasksWaiting()) {
			Task *task = dynamic_cast<Task*>(&mThreadPool.getCompletedTask());
			if (task == nullptr)
				throw (RuntimeError() << "Invalid task returned from the thread pool.");

			processCompletedTask(task);
		}
	}


	/**
	 * \brief Check whether selected task has already finished. The method MAY empty the output
	 *		queue of the pool, but it is not waiting for any tasks to finish.
	 * \param promiseId The ID of the promise object that corresponds with the task.
	 */
	bool isDone(size_t promiseId)
	{
		// Check the ID is valid.
		if (promiseId >= mNextId)
			throw (RuntimeError() << "Invalid promise ID (" << promiseId << "), only " << mNextId << " IDs was issued.");

		// If the ID has been removed, the task has already terminated.
		if (mPendingTasks.find(promiseId) == mPendingTasks.end())
			return true;

		// Check whether there are completed tasks pending in the output queue.
		processAllCompletedTasks();

		return (mPendingTasks.find(promiseId) == mPendingTasks.end());
	}


	/**
	 * \brief Wait for selected task to finish. The method MAY empty output queue of the pool
	 *		and it blocks if the task has not finished yet. In such case, all tasks that finished
	 *		before selected task (from the output queue perspective) are finalized as well.
	 * \param promiseId The ID of the promise object that corresponds with the task.
	 */
	void waitFor(size_t promiseId)
	{
		if (isDone(promiseId))
			return;

		while (!mThreadPool.empty()) {
			Task *task = dynamic_cast<Task*>(&mThreadPool.getCompletedTask());
			if (task == nullptr)
				throw (RuntimeError() << "Invalid task returned from the thread pool.");

			bool wantedTask = (task->getId() == promiseId);
			processCompletedTask(task);
			if (wantedTask)
				return;
		}

		throw (RuntimeError() << "Unable to wait for task " << promiseId << ", since all tasks have already terminated.");
	}


	/**
	 * \brief Verify that selected task succeeded (i.e., is not on the failed tasks list).
	 * \param promiseId The ID of the promise object that corresponds with the task.
	 * \throws LogicError if the task has not finished yet.
	 */
	bool isOK(size_t promiseId)
	{
		if (!isDone(promiseId))
			throw (LogicError() << "Task " << promiseId << " has not finished yet. It is not possible to test it for failure at present time.");
		return (mFailedTasks.find(promiseId) == mFailedTasks.end());
	}


	/**
	 * \brief Retrieve an error message of selected task.
	 * \param promiseId The ID of the promise object that corresponds with the task.
	 * \throws LogicError if the task has not finished yet or if it succeeded.
	 */
	const std::string& getErrorMessage(size_t promiseId)
	{
		if (!isDone(promiseId))
			throw (LogicError() << "Task " << promiseId << " has not finished yet. It is not possible to test it for failure at present time.");

		std::map<size_t, Task*>::iterator it = mFailedTasks.find(promiseId);
		if (it == mFailedTasks.end())
			throw (LogicError() << "Task " << promiseId << " has not failed. Error message is not available.");

		return it->second->getErrorMessage();
	}


	/**
	 * \brief Mark selected promise as OK by removing the record in failed tasks list.
	 * \param promiseId The ID of the promise object that corresponds with the task.
	 * \throws LogicError if the task has not finished yet.
	 */
	void clearError(size_t promiseId)
	{
		if (!isDone(promiseId))
			throw (LogicError() << "Task " << promiseId << " has not finished yet. It is not possible to test it for failure at present time.");
		
		std::map<size_t, Task*>::iterator it = mFailedTasks.find(promiseId);
		if (it != mFailedTasks.end()) {
			delete it->second;
			mFailedTasks.erase(it);
		}
	}


	/**
	 * \brief Register a new task, enqueue it for processing in the thread pool,
	 *		and create a corresponding Promise object.
	 * \param task Newly created task object to be registered and executed.
	 */
	Promise startNewTask(Task *task)
	{
		assert(task != nullptr);

		// Make sure we can add another task to the pool.
		processAllCompletedTasks();
		while (mThreadPool.full()) {
			Task *task = dynamic_cast<Task*>(&mThreadPool.getCompletedTask());
			if (task == nullptr)
				throw (RuntimeError() << "Invalid task returned from the thread pool.");
			processCompletedTask(task);
		}

		// Add the task to the pending list and to the pool...
		mPendingTasks.insert(task->getId());
		mThreadPool.enqueue(*task);

		return Promise(*this, task->getId());
	}


public:
	/**
	 * \brief Initializes the manager for file asynchronous operations.
	 * \param threads The number of threads of the internal pool (i.e.,
	 *		how many async operations can run concurrently).
	 */
	FileAsyncOps(size_t threads = 1) : mThreadPool(threads), mNextId(0) {}


	/**
	 * \brief Remove all recorded async operation errors.
	 */
	void clearErrors()
	{
		for (std::map<size_t, Task*>::iterator it = mFailedTasks.begin(); it != mFailedTasks.end(); ++it)
			delete it->second;
		mFailedTasks.clear();
	}


	/**
	 * \brief Asynchronous version of the File::read<T>() method.
	 * \note The method is designed as non-blocking, however, it MAY block
	 *		if the internal thread pool is full (i.e., too many async operations are pending).
	 * \return Promise object that corresponds with the asynchronous operation.
	 */
	template<typename T>
	Promise read(File &file, T* buffer, size_t count)
	{
		return startNewTask(new ReadTask<T>(mNextId++, file, buffer, count));
	}


	/**
	 * \brief Asynchronous version of the File::read<T>() method.
	 * \note The method is designed as non-blocking, however, it MAY block
	 *		if the internal thread pool is full (i.e., too many async operations are pending).
	 * \return Promise object that corresponds with the asynchronous operation.
	 */
	template<typename T>
	Promise read(File &file, Buffer<T> &buffer)
	{
		return startNewTask(new ReadTask<T>(mNextId++, file, buffer.get(), buffer.count()));
	}


	/**
	 * \brief Asynchronous version of the File::read<T>() method.
	 * \note The method is designed as non-blocking, however, it MAY block
	 *		if the internal thread pool is full (i.e., too many async operations are pending).
	 * \return Promise object that corresponds with the asynchronous operation.
	 */
	template<typename T>
	Promise write(File &file, const T* buffer, size_t count, bool flush = true)
	{
		return startNewTask(new WriteTask<T>(mNextId++, file, buffer, count, flush));
	}


	/**
	 * \brief Asynchronous version of the File::read<T>() method.
	 * \note The method is designed as non-blocking, however, it MAY block
	 *		if the internal thread pool is full (i.e., too many async operations are pending).
	 * \return Promise object that corresponds with the asynchronous operation.
	 */
	template<typename T>
	Promise write(File &file, const Buffer<T> &buffer, bool flush = true)
	{
		return startNewTask(new WriteTask<T>(mNextId++, file, buffer.get(), buffer.count(), flush));
	}
};


}

#endif
#endif

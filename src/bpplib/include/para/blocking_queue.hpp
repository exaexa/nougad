/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 14.9.2015
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_PARA_BLOCKING_QUEUE_HPP
#define BPPLIB_PARA_BLOCKING_QUEUE_HPP

#include <misc/exception.hpp>

#include <vector>
#include <mutex>
#include <condition_variable>
#include <atomic>



namespace bpp
{


/*
 * \brief A thread-safe queue used for concurrent tasks.
 *		The queue has fixed size and it can be used to syncrhonize
 *		producent-consumer like problems.
 * \tparam T type of items in the queue. The items should be easily copyable and lightweight.
 */
template<typename T>
class BlockingQueue
{
private:
	T* mItems;	///< Items stored in queue.
	std::size_t mCapacity;	///< Size of the allocated space in the queue.
	std::size_t mCount;		///< Number of items in the queue.
	std::size_t mFront;		///< Index to position in the mItems, where items are being inserted.
	std::size_t mBack;		///< Index to position in the mItems, from where the items are being poped.

	std::size_t mPushWaiting;			///< How many threads are waiting on blocking push.
	std::size_t mPopWaiting;			///< How many threads are waiting on blocking pop.
	std::condition_variable mCVPush;	///< Conditional variable where the blocking push operations wait.
	std::condition_variable mCVPop;		///< Conditional variable where the blocking pop operation wait.
	std::mutex mMutex;					///< C++11 synchronization primitive for the whole structure.

	/*
	 * \brief Internal (not locked) method that attemts push operation.
	 * \return True if the operation succeeded, false if the queue is full.
	 */
	bool tryPushUnsafe(T item)
	{
		if (mCount >= mCapacity)
			return false;
		mItems[mFront] = item;
		++mCount;
		mFront = (mFront + 1) % mCapacity;
		return true;
	}


	/*
	* \brief Internal (not locked) method that attemts pop operation.
	* \return True if the operation succeeded, false if the queue is empty.
	*/
	bool tryPopUnsafe(T &item)
	{
		if (mCount == 0)
			return false;
		item = mItems[mBack];
		mItems[mBack] = T();
		--mCount;
		mBack = (mBack + 1) % mCapacity;
		return true;
	}



public:
	/*
	 * \brief Create blocking queue of given capacity.
	 *		The capacity cannot be changed once the queue is created.
	 * \param capacity Maximal number of items in the queue. The capacity should be greater than 0.
	 */
	BlockingQueue(std::size_t capacity = 0)
		: mItems(nullptr), mCapacity(capacity), mCount(0), mFront(0), mBack(0), mPushWaiting(0), mPopWaiting(0)
	{
		if (capacity > 0)
			mItems = new T[capacity];
	}

	BlockingQueue(const BlockingQueue<T> &q) = delete;
	BlockingQueue& operator=(const BlockingQueue<T> &q) = delete;

	BlockingQueue(BlockingQueue<T> &&q)
		: mItems(q.mItems), mCapacity(q.mCapacity), mCount(q.mCount), mFront(q.mFront), mBack(q.mBack), mPushWaiting(q.mPushWaiting), mPopWaiting(q.mPopWaiting)
	{
		q.mItems = nullptr;
		q.mCapacity = 0;
		q.mCount = 0;
		q.mFront = 0;
		q.mBack = 0;
		q.mPushWaiting = 0;
		q.mPopWaiting = 0;
	}


	~BlockingQueue()
	{
		delete[] mItems;
	}


	/*
	 * \brief Return the capacity of the queue.
	 */
	std::size_t capacity() const
	{
		return mCapacity;
	}


	/**
	 * \brief Change capacity of the queue. The queue must be empty while doing so.
	 * \param newCapacity New number of items for which the queue is allocated.
	 */
	void setCapacity(std::size_t newCapacity)
	{
		std::lock_guard<std::mutex> lock(mMutex);

		// Constraint checks ...
		if (mCount != 0)
			throw (bpp::RuntimeError() << "The blocking queue is not empty. There are " << mCount << " items enqueued.");
		if (mPopWaiting != 0 || mPushWaiting != 0)
			throw (bpp::RuntimeError() << "The blocking queue has some clients waiting for push or pop operation.");

		// Remove old buffer.
		if (mCapacity != 0) {
			delete[] mItems;
			mItems = nullptr;
		}

		// Allocate new buffer.
		mCapacity = newCapacity;
		if (mCapacity > 0)
			mItems = new T[mCapacity];

		// Make sure front and back indices are valid no matter the previous usage of the queue.
		mFront = mBack = 0;
	}

	
	/*
	 * \brief Return the number of items in the queue. Note that this value may change immediately
	 *		after the caller thread reads it.
	 */
	std::size_t count()
	{
		std::lock_guard<std::mutex> lock(mMutex);
		return mCount;
	}

	/*
	* \brief Check whether the queue is empty. Note that this value may change immediately
	*		after the caller thread reads it.
	*/
	bool isEmpty()
	{
		return count() == 0;
	}


	/*
	* \brief Check whether the queue is full. Note that this value may change immediately
	*		after the caller thread reads it.
	*/
	bool isFull()
	{
		return count() == capacity();
	}


	/*
	 * \brief Non-blocking version of push operation.
	 * \param item The item being pushed.
	 * \return True if the operation succeeded, false if the queue is full.
	 */
	bool tryPush(T item)
	{
		bool res, signal = false;
		{
			std::lock_guard<std::mutex> lock(mMutex);
			signal = (mPopWaiting > 0);
			res = tryPushUnsafe(item);
		}
		if (signal)
			mCVPop.notify_one();
		return res;
	}


	/*
	 * \brief Non-blocking version of pop operation.
	 * \param item A place where the item is poped if available.
	 * \return True if the operation succeeded, false if the queue is empty.
	 */
	bool tryPop(T &item)
	{
		bool res, signal = false;
		{
			std::lock_guard<std::mutex> lock(mMutex);
			signal = (mPushWaiting > 0);
			res = tryPopUnsafe(item);
		}
		if (signal)
			mCVPush.notify_one();
		return res;
	}


	/*
	 * \brief Blocking version of push operation. If the queue is full, it waits until some items are poped. 
	 * \param item The item being pushed.
	 */
	void push(T item)
	{
		bool signal = false;	// whether the pop-waiters should be notified at the end of push
		{
			std::unique_lock<std::mutex> lock(mMutex);
			signal = (mPopWaiting > 0);

			// Repeatedly try non-blocking push
			while (!tryPushUnsafe(item)) {
				++mPushWaiting;
				mCVPush.wait(lock);		// block on CV until some items are poped
				--mPushWaiting;
				signal = (mPopWaiting > 0);
			}
		}
		if (signal)
			mCVPop.notify_one();
	}


	/*
	 * \brief Blocking version of pop operation. If the queue is empty, it waits until some items are pushed.
	 * \return The item that was poped from the queue.
	 */
	T pop()
	{
		bool signal = false;	// whether the push-waiters should be notified at the end of push
		T result;
		{
			std::unique_lock<std::mutex> lock(mMutex);
			signal = (mPushWaiting > 0);

			// Repeatedly try non-blocking push
			while (!tryPopUnsafe(result)) {
				++mPopWaiting;
				mCVPop.wait(lock);		// block on CV until some items are pushed
				--mPopWaiting;
				signal = (mPushWaiting > 0);
			}
		}
		if (signal)
			mCVPush.notify_one();
		return result;
	}


	/*
	 * \brief Empty all items currently stored in the queue.
	 *		The push-blocked threads are all signaled.
	 */
	void clear()
	{
		bool signal;
		{
			T item;
			std::lock_guard<std::mutex> lock(mMutex);
			signal = (mPushWaiting > 0);
			while (tryPopUnsafe(item)) { /* pop until we cannot pop any more */ }
		}
		if (signal)
			mCVPush.notify_all();
	}
};

}

#endif
/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 1.7.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_DATASTRUCT_HEAP_LOOSER_HPP
#define BPPLIB_DATASTRUCT_HEAP_LOOSER_HPP


#include <datastruct/heap/regular.hpp>

namespace bpp {

/**
 * \brief Another type of heap based on loosers tree (see Knuth vol. 3, sec. 5.4.1).
 *
 *	The only difference is that the top of the tree is kept separatelly at the zero-th
 *	position and d-regular heap is buitl beneath from the position [1].
 *
 * \tparam T Type of the items in the heap.
 * \tparam D Degree of the branching in the heap (minimum is 2).
 * \tparam CMP Comparator class that contains "bool inOrder(T,T)" static method.
 */
template<typename T, int D = 2, class CMP = Comparator<T> >
class LooserHeap
{
public:
	typedef T key_t;
	typedef CMP comparator_t;

private:
	std::vector<T> mData;		///< Vector containing heap data.
	RawHeap<T, D, CMP> mHeap;	///< Raw heap that is used to maintain the data order.

	void checkTop()
	{
		if (mData.size() > 1 && !CMP::inOrder(mData[0], mData[1]))
			std::swap(mData[0], mData[1]);
	}


public:
	/**
	 * \brief Creates empty heap.
	 */
	LooserHeap() {}
	

	/**
	 * \brief Creates a heap by copying data from a constant vector.
	 * \param data Vector with data to be copied.
	 */
	LooserHeap(const std::vector<T> &data)
	{
		assign(data);
	}


	/**
	 * \brief Creates a heap by copying/stealing data from a vector.
	 * \param data Vector with data to be copied/stolen.
	 * \param acquire True if the data are to be swapped from the vector,
	 *		false if they are to be copied.
	 */
	LooserHeap(std::vector<T> &data, bool acquire = false) : mData(0)
	{
		if (acquire) {
			if (mData.size() > 1) {
				mHeap.mData = &mData[1];
				mHeap.mSize = mData.size();
				mHeap.build();
				updatePosition(0);
			}
			else {
				mHeap.mData = nullptr;
				mHeap.mSize = 0;
			}
		}
		else
			assign(data);
	}


	/**
	 * \brief Return the constant value of the top of the heap (i.e., minimum or maximum,
	 *		depending on the comparator).
	 */
	const T& getTop() const
	{
		return (*this)[0];
	}


	/**
	 * \brief Return the value of the top of the heap (i.e., minimum or maximum,
	 *		depending on the comparator). If the value is modified, updatePosition(0)
	 *		MUST be called.
	 */
	T& getTop()
	{
		return (*this)[0];
	}


	/**
	 * \brief Return selected item of the heap.
	 * \param idx Index of the selected item (zero based).
	 */
	const T& operator[](size_t idx) const
	{
		return mData[idx];
	}


	/**
	 * \brief Return selected item of the heap. If the item is modified,
	 *		updatePosition(idx) MUST be called.
	 * \param idx Index of the selected item (zero based).
	 */
	T& operator[](size_t idx)
	{
		return mData[idx];
	}


	/**
	 * \brief Return the number of items in the heap.
	 */
	size_t size() const
	{
		return mData.size();
	}
	

	/**
	 * \brief Check whether the heap is empty.
	 */
	bool empty() const
	{
		return size() == 0;
	}


	/**
	 * \brief Remove all items from the heap.
	 */
	void clear()
	{
		mData.clear();
	}


	/**
	 * \brief Reserve given amount of space in the underlying vector.
	 * \param count Number of items to be reserved.
	 */
	void reserve(size_t count)
	{
		mData.reserve(count);
		if (mData.size() > 1) {
			mHeap.mData = &mData[1];
			mHeap.mSize = mData.size()-1;
		}
		else {
			mHeap.mData = nullptr;
			mHeap.mSize = 0;
		}
	}


	/**
	 * \brief Replace the contents of the heap with given array of items.
	 * \param data Pointer to the beginning of the input array.
	 * \param count Number of items to be copied from the array.
	 */
	void assign(const T *data, size_t count)
	{
		mData.assign(data, data+count);
		if (mData.size() > 1) {
			mHeap.mData = &mData[1];
			mHeap.mSize = mData.size()-1;
			mHeap.build();
			updatePosition(0);
		}
		else {
			mHeap.mData = nullptr;
			mHeap.mSize = 0;
		}
	}


	/**
	 * \brief Replace the contents of the heap with given vector.
	 * \param data A vector with input data.
	 */
	void assign(const std::vector<T> &data)
	{
		assign(&data[0], data.size());
	}


	/**
	 * \brief Add another item to the heap and fix the heap (O(log N)).
	 * \param item Item that is being added.
	 */
	void add(const T &item)
	{
		mData.push_back(item);
		if (mData.size() > 1) {
			mHeap.mData = &mData[1];
			mHeap.mSize = mData.size()-1;
			mHeap.bubbleUp(mHeap.mSize-1);
			checkTop();
		}
	}


	/**
	 * \brief Remove the top of the heap and rearrange it (O(log N)).
	 */
	void removeTop()
	{
		if (mData.empty())
			throw RuntimeError("The heap is empty. Unable to remove top item.");

		if (mData.size() > 1) {
			mData[0] = mData[1];
			mData[1] = mData.back();
		}
		mData.pop_back();

		if (mData.size() > 1) {
			mHeap.mData = &mData[1];
			mHeap.mSize = mData.size()-1;
			mHeap.bubbleDown(0);
		}
	}


	/**
	 * \brief Notify the heap that item at given index may have been modified,
	 *		thus it may need moving within the heap.
	 * \param idx Index of the item to be fixed.
	 * \note Each item MUST be updated immediately after its modification.
	 *		Otherwise the heap DO NOT guarantee correctness of its basic operations.
	 */
	void updatePosition(size_t idx)
	{
		if (idx > 1)
			mHeap.bubbleUp(idx-1);
		
		checkTop();
		
		if (idx > 0) --idx;
		mHeap.bubbleDown(idx);
	}
};


}

#endif

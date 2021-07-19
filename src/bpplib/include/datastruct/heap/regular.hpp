/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 1.7.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_DATASTRUCT_HEAP_REGULAR_HPP
#define BPPLIB_DATASTRUCT_HEAP_REGULAR_HPP

#include <datastruct/comparator.hpp>
#include <misc/exception.hpp>

#include <vector>
#include <algorithm>
#include <utility>

#include <cassert>
#include <cstdint>

namespace bpp
{

/**
 * \brief Low level D-regular heap algorithm wrapper. It build/operate heap data structure
 *		over an user-managed array or vector. The minimum is kept at position [0].
 * \tparam T Type of the items in the heap.
 * \tparam D Degree of the branching in the heap (minimum is 2).
 * \tparam CMP Comparator class that contains "bool inOrder(T,T)" static method.
 */
template<typename T, int D = 2, class CMP = Comparator<T> >
class RawHeap
{
public:
	/**
	 * \brief Pointer at the beginning of the heap.
	 * \note This value is public and the programmer is responsible for its maintanance.
	 */
	T* mData;

	/**
	 * \brief Number of items in the heap.
	 * \note This value is public and the programmer is responsible for its maintanance.
	 */
	size_t mSize;


	RawHeap(T* data = nullptr, size_t size = 0)
		: mData(data), mSize(size) {}

	RawHeap(std::vector<T> &data)
		: mData(&data[0]), mSize(data.size()) {}


	/**
	 * \brief Perform regular heap correction algorithms from item idx towards its ancestors.
	 * \param idx Index of an item that may require moving.
	 */
	void bubbleUp(std::size_t idx)
	{
		while (idx > 0) {
			std::size_t parentIdx = (idx-1) / D;
			if (!CMP::inOrder(mData[parentIdx], mData[idx])) {
				std::swap(mData[idx], mData[parentIdx]);
				idx = parentIdx;
			}
			else break;
		}
	}


	/**
	 * \brief Perform regular heap correction algorithms from item idx towards its descendants.
	 * \param idx Index of an item that may require moving.
	 */
	void bubbleDown(std::size_t idx)
	{
		while (idx*D + 1 < mSize) {
			std::size_t firstChildIdx = idx*D + 1;
			std::size_t childIdx = firstChildIdx;
			for (std::size_t i = 1; i < D && firstChildIdx+i < mSize; ++i) {
				if (!CMP::inOrder(mData[childIdx], mData[firstChildIdx+i]))
					childIdx = firstChildIdx+i;
			}

			if (!CMP::inOrder(mData[idx], mData[childIdx])) {
				std::swap(mData[idx], mData[childIdx]);
				idx = childIdx;
			}
			else break;
		}
	}


	/**
	 * \brief Perform bottom-up build of the heap (in time O(N)).
	 */
	void build()
	{
		std::size_t idx = ((mSize-1) / D) + 1;
		while (idx > 0)
			bubbleDown(--idx);
	}
};





/**
 * \brief A wrapper for RawHeap that is applied on a vector.
 * \tparam T Type of the items in the heap.
 * \tparam D Degree of the branching in the heap (minimum is 2).
 * \tparam CMP Comparator class that contains "bool inOrder(T,T)" static method.
 */
template<typename T, int D = 2, class CMP = Comparator<T> >
class RegularHeap
{
public:
	typedef T key_t;
	typedef CMP comparator_t;

private:
	std::vector<T> mData;		///< Vector containing heap data.
	RawHeap<T, D, CMP> mHeap;	///< Raw heap that is used to maintain the data order.

public:
	/**
	 * \brief Creates empty heap.
	 */
	RegularHeap() {}
	

	/**
	 * \brief Creates a heap by copying data from a constant vector.
	 * \param data Vector with data to be copied.
	 */
	RegularHeap(const std::vector<T> &data)
	{
		assign(data);
	}


	/**
	 * \brief Creates a heap by copying/stealing data from a vector.
	 * \param data Vector with data to be copied/stolen.
	 * \param acquire True if the data are to be swapped from the vector,
	 *		false if they are to be copied.
	 */
	RegularHeap(std::vector<T> &data, bool acquire = false) : mData(0)
	{
		if (acquire) {
			mData.swap(data);
			mHeap.mData = &mData[0];
			mHeap.mSize = mData.size();
			mHeap.build();
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
		mHeap.mData = &mData[0];
		mHeap.mSize = mData.size();
	}


	/**
	 * \brief Replace the contents of the heap with given array of items.
	 * \param data Pointer to the beginning of the input array.
	 * \param count Number of items to be copied from the array.
	 */
	void assign(const T *data, size_t count)
	{
		mData.assign(data, data+count);
		mHeap.mData = &mData[0];
		mHeap.mSize = mData.size();
		mHeap.build();
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
		mHeap.mData = &mData[0];
		mHeap.mSize = mData.size();
		mHeap.bubbleUp(mHeap.mSize-1);
	}


	/**
	 * \brief Remove the top of the heap and rearrange it (O(log N)).
	 */
	void removeTop()
	{
		if (mData.empty())
			throw RuntimeError("The heap is empty. Unable to remove top item.");

		mData[0] = mData.back();
		mData.pop_back();

		if (mData.size() > 0) {
			mHeap.mData = &mData[0];
			mHeap.mSize = mData.size();
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
		mHeap.bubbleDown(idx);
		mHeap.bubbleUp(idx);
	}
};





/**
 *
 */
//template<typename T, class HEAP = RegularHeap<T,2,Comparator<T> > >
//class TopK
//{
//private:
//	size_t mK;
//	HEAP mHeap;
//
//public:
//	kNNResult(size_t k) : mK(k)
//	{
//		mHeap.reserve(k);
//	}
//
//
//	size_t getK() const
//	{
//		return mK;
//	}
//};


}
#endif

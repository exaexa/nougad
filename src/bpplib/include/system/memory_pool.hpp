/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 17.7.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_SYSTEM_MEMORY_POOL_HPP
#define BPPLIB_SYSTEM_MEMORY_POOL_HPP

#include <misc/exception.hpp>

#include <vector>

#include <cassert>
#include <cstdint>


namespace bpp
{

class MemoryPool;	// forward declaration



/**
 * \brief Typed pointer allocated through memory pool which remembers size of the allocated area.
 */
template<typename T = char>
class Buffer
{
friend class MemoryPool;
private:
	T* mStart;				///< Pointer to the buffer start.
	std::size_t mCount;		///< Size of the buffer in multiples of T.

	Buffer(T* start, std::size_t count)
		: mStart(start), mCount(count) {}

public:
	Buffer() : mStart(nullptr), mCount(0) {}

	T* get() const						{ return mStart; }
	std::size_t count() const			{ return mCount; }
	T& operator[](size_t i)				{ return mStart[i]; }
	const T& operator[](size_t i) const	{ return mStart[i]; }
};



/**
 * \brief A memory buffer that allows special aligned allocations.
 *		This buffer is usually used to simulate limited memory resources.
 *		E.g., in specific applications like database management systems.
 * \note All allocations and releases of the memory buffers must be
 *		well-paired, since the buffer acts like a stack.
 */
class MemoryPool
{
private:
	static const std::size_t PAGE_SIZE = 4096;

	std::size_t mSize;		///< Total size of allocated memory (bytes).
	std::size_t mRemaining;	///< Remaining amount of free memory (bytes).
	char *mMemory;			///< All the allocated memory.
	char *mAligned;			///< Start of aligned allocation area.
	char *mFree;			///< Pointer to start of still free area.

	std::vector< Buffer<char> > mAllocations;

	/**
	 * \brief Get nearest aligned and typed pointer within given buffer.
	 */
	template<typename R>
	static R* alignPointer(void *p, std::size_t alignment)
	{
		// Verify the alignment is power of 2
		assert((alignment != 0) && (((alignment-1) | alignment)) == (2*alignment - 1));

		std::uintptr_t aligned = reinterpret_cast<std::uintptr_t>(p);
		aligned = (aligned + static_cast<std::uintptr_t>(alignment)-1) & ~(static_cast<std::uintptr_t>(alignment)-1);
		return reinterpret_cast<R*>(aligned);
	}

public:
	MemoryPool(std::size_t size) : mSize(size), mRemaining(size)
	{
		// Allocate memory pool buffer.
		mMemory = (char*)malloc(size + PAGE_SIZE);
		if (mMemory == nullptr)
			throw (bpp::RuntimeError() << "Unable to allocate memory pool of " << size + PAGE_SIZE << " bytes.");
		mFree = mAligned = alignPointer<char>(mMemory, PAGE_SIZE);
	}
	

	size_t size() const			{ return mSize; }
	size_t remaining() const	{ return mRemaining; }


	template<typename T> Buffer<T> allocateBuffer(std::size_t count)
	{
		T *start = alignPointer<T>(mFree, sizeof(T));
		if (reinterpret_cast<char*>(start + count) > mAligned + mSize)
			throw (bpp::RuntimeError() << "Not enough memory to allocate buffer #" << mAllocations.size() << " of size" << count * sizeof(T) << " B.");

		mAllocations.push_back(Buffer<char>((char*)start, count * sizeof(T)));
		mFree = reinterpret_cast<char*>(start + count);
		mRemaining = mSize - (mFree - mAligned);
		return Buffer<T>(start, count);
	}


	template<typename T> void disposeOf(const Buffer<T> &buffer)
	{
		if (mAllocations.empty())
			throw LogicError("Invalid dispose operation. There are no allocated buffers in the memory pool.");

		if (mAllocations.back().get() != reinterpret_cast<char*>(buffer.get()) || mAllocations.back().count() != (buffer.count() * sizeof(T)))
			throw (LogicError() << "Invalid dispose operation of buffer #" << mAllocations.size());

		mAllocations.pop_back();
		if (mAllocations.empty())
			mFree = mAligned;
		else
			mFree = mAllocations.back().get() + mAllocations.back().count();
		mRemaining = mSize - (mFree - mAligned);
	}


	void disposeAll()
	{
		mAllocations.clear();
		mFree = mAligned;
		mRemaining = mSize;
	}
};


}

#endif

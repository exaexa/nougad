#pragma once

#include <limits>
#include <random>
#include <string>
#include <vector>

#include <misc/exception.hpp>

template <typename F = float>
double compare_floats_relative(F a, F b)
{
	double sum = std::abs((double)a) + std::abs((double)b);
	if (sum < 0.00001)
		sum = 0.00001;
	double diff = std::abs((double)a - (double)b);
	return diff / sum;
}

/**
 * Container holding data points in DIM-dimensional space.
 * This container has fixed AoS layout (each point is a compact structure).
 * \tparam F base type for coordinates (float or double)
 */
template <typename F>
class DataPoints
{
public:
	using real_t = F;

private:
	std::size_t mDim, mSize;
	std::vector<real_t> mPoints; ///< Container storing the features (row-wise, padded to alignment)
	real_t* mDataPtr;

	/**
	 * Swap two points.
	 */
	void swap(std::size_t idx1, std::size_t idx2)
	{
		for (std::size_t d = 0; d < mDim; ++d)
		{
			std::swap((*this)[idx1][d], (*this)[idx2][d]);
		}
	}


public:
	DataPoints(std::size_t dim = 0, std::size_t size = 0, F* data = nullptr) : mDim(dim), mSize(size), mDataPtr(data) 
	{
		if (!data) resize(size);
	}

	std::size_t getDim() const { return mDim; }

	void setDim(std::size_t dim)
	{
		if (dim != mDim)
		{
			clear();
			mDim = dim;
		}
	}

	/**
	 * Change size of the container/how much of mmaped file is accessible.
	 */
	void resize(std::size_t size)
	{
		mSize = size;
		mPoints.resize(size * mDim);
		mDataPtr = mSize > 0 && mDim > 0 ? &mPoints[0] : nullptr;
	}

	// accessors
	std::size_t size() const { return mSize; }

	const F* operator[](std::size_t idx) const { return mDataPtr + (idx * mDim); }

	F* operator[](std::size_t idx) { return mDataPtr + (idx * mDim); }

	const F* data() const { return mDataPtr; }

	F* data() { return mDataPtr; }


	/**
	 * Randomize the order of the features in the container.
	 * Works only on allocated memory, not on mmaped file.
	 */
	void shuffle(std::mt19937_64::result_type seed = std::mt19937_64::default_seed)
	{
		if (mPoints.empty())
		{
			return;
		}

		std::mt19937_64 rng(seed);
		std::size_t count = size();
		while (count > 1)
		{
			std::size_t idx = rng() % count;
			--count;
			if (idx != count)
			{
				swap(idx, count);
			}
		}
	}


	void clear()
	{
		mSize = 0;
		mDataPtr = nullptr;
	}
};

#pragma once

#include <limits>
#include <random>
#include <string>
#include <vector>

#include <algo/strings.hpp>
#include <misc/exception.hpp>
#include <system/file.hpp>
#include <system/mmap_file.hpp>


/**
 * Helper function that peeks in text TSV file and reads the dimension of the data.
 */
std::size_t peekTSVGetDimension(const std::string& fileName)
{
	bpp::File file(fileName);
	file.open("r");

	std::string line;
	if (!file.readLine(line))
	{
		throw(bpp::RuntimeError() << "TSV file " << fileName << " is empty.");
	}

	file.close();

	bpp::SimpleTokenizer tokenizer("\t");
	std::vector<bpp::SimpleTokenizer::ref_t> tokens;
	tokenizer.doTokenize(line, tokens, true);
	if (tokens.size() == 0)
	{
		throw(bpp::RuntimeError() << "TSV file " << fileName << " has no numbers on the first line.");
	}

	return tokens.size();
}

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
	bpp::MMapFile mFile;		 ///< Holds the mmaped file handle (alternative to mPoints)
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
	DataPoints(std::size_t dim = 0, std::size_t size = 0) : mDim(dim), mSize(size), mDataPtr(nullptr) { resize(size); }

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
		if (mFile.opened())
		{
			std::size_t maxSize = mFile.length() / (mDim * sizeof(real_t));
			if (size > maxSize)
			{
				throw(bpp::RuntimeError() << "Unable to change size to" << size << ". MMaped file holds only " << maxSize << " records.");
			}
			mSize = size;
		}
		else
		{
			mSize = size;
			mPoints.resize(size * mDim);
			mDataPtr = mSize > 0 && mDim > 0 ? &mPoints[0] : nullptr;
		}
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
		mPoints.clear();
		if (mFile.opened())
		{
			mFile.close();
		}

		mSize = 0;
		mDataPtr = nullptr;
	}

	/**
	 * Load data from TSV file and allocate memory internally.
	 */
	void loadTSV(const std::string& fileName, std::size_t limit = std::numeric_limits<std::size_t>::max())
	{
		clear();
		mDim = peekTSVGetDimension(fileName);

		bpp::File file(fileName);
		file.open("r");

		std::string line;
		bpp::SimpleTokenizer tokenizer("\t");
		std::vector<bpp::SimpleTokenizer::ref_t> tokens;
		while (mSize < limit && file.readLine(line))
		{
			++mSize;
			tokenizer.doTokenize(line, tokens, true);
			if (tokens.size() < mDim)
			{
				throw(bpp::RuntimeError() << "Insufficient values on row #" << mSize << ". Dimension " << mDim << " expected, but only "
										  << tokens.size() << "tokens found.");
			}

			for (std::size_t i = 0; i < mDim; ++i)
			{
				mPoints.push_back(tokens[i].as<F>());
			}
		}

		file.close();
		mDataPtr = &mPoints[0];
	}

	/**
	 * Mmaps a binary file.
	 */
	void loadBinary(const std::string& fileName, std::size_t limit = std::numeric_limits<std::size_t>::max())
	{
		clear();
		mFile.open(fileName);

		std::size_t fileSize = mFile.length();
		if (fileSize % sizeof(F) != 0)
		{
			mFile.close();
			throw(bpp::RuntimeError() << "Size of file " << fileName << " is not divisible by float size (" << sizeof(F)
									  << "). Invalid data expected.");
		}

		fileSize /= sizeof(F);
		if (mDim == 0 || fileSize % mDim != 0)
		{
			mFile.close();
			throw(bpp::RuntimeError() << "Size of file " << fileName << " is not divisible by feature dimensions (" << mDim
									  << "). Invalid data expected.");
		}

		mSize = std::min(limit, fileSize / mDim);
		mDataPtr = (F*)mFile.getData(); // yes, this is potentially dangerous since mmaped file is read-only, but what the heck
	}


	/**
	 * Write data of this container into text TSV file.
	 */
	void saveTSV(const std::string& fileName) const
	{
		bpp::File file(fileName);
		file.open("wb");
		bpp::TextWriter writer(file);

		for (std::size_t i = 0; i < size(); ++i)
		{
			for (std::size_t d = 0; d < mDim; ++d)
			{
				writer.writeToken((*this)[i][d]);
			}
			writer.writeLine();
		}

		file.close();
	}


	/**
	 * Write data of this container into binary blob file.
	 */
	void saveBinary(const std::string& fileName) const
	{
		bpp::File file(fileName);
		file.open("wb");

		for (std::size_t i = 0; i < size(); ++i)
		{
			for (std::size_t d = 0; d < mDim; ++d)
			{
				file.write<F>((*this)[i] + d);
			}
		}

		file.close();
	}
};

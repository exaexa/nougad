/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 7.12.2014
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_MATH_STATS_HPP
#define BPPLIB_MATH_STATS_HPP

#include <misc/exception.hpp>

#include <vector>
#include <algorithm>
#include <iostream>


namespace bpp {

/**
 * \brief Collection of basic statistic values from set of numerical data.
 */
template<typename F = double>
class BasicStats
{
private:
	F mMinimum;
	F mMaximum;
	F mAverage;		///< Collects sum of all included values
	F mVariance;	///< Collects sum of all included values^2
	size_t mCount;

public:
	BasicStats()
	{
		reset();
	}


	/**
	 * \brief Get smallest value from the data.
	 */
	F getMinimum() const
	{
		if (mCount == 0)
			throw bpp::RuntimeError("Unable to get minimum of an empty dataset.");
		return mMinimum;
	}

	/**
	 * \brief Get largest value from the data.
	 */
	F getMaximum() const
	{
		if (mCount == 0)
			throw bpp::RuntimeError("Unable to get maximum of an empty dataset.");
		return mMaximum;
	}

	/**
	 * \brief Get data mean value (arithmetic average).
	 */
	F getAverage() const
	{
		if (mCount == 0)
			throw bpp::RuntimeError("Unable to get average of an empty dataset.");
		return mAverage / (F)mCount;
	}

	/**
	 * \brief Get variance (std deviation ^2).
	 */
	F getVariance() const
	{
		if (mCount == 0)
			throw bpp::RuntimeError("Unable to get variance of an empty dataset.");
		F avg = getAverage();
		return (mVariance / (F)mCount) - (avg*avg);		// E(X^2) - (EX)^2;
	}

	/**
	 * \brief Get standard deviation (second central momentum).
	 */
	F getStdDeviation() const
	{
		return std::sqrt(getVariance());
	}

	/**
	 * \brief Get the number of values collected.
	 */
	size_t getCount() const		{ return mCount; }

	/**
	 * \brief Check whether the data structure is empty.
	 */
	bool isEmpty() const		{ return mCount == 0; }


	/**
	 * \brief Clear the structure and reset it for another dataset.
	 */
	void reset()
	{
		mCount = 0;
		mMinimum = mMaximum = mAverage = mVariance = (F)0.0;
	}


	void collect(F x)
	{
		if (isEmpty()) {
			mMinimum = mMaximum = x;
		}
		else {
			mMinimum = std::min<F>(mMinimum, x);
			mMaximum = std::max<F>(mMaximum, x);
		}
		mAverage += x;
		mVariance += x*x;
		++mCount;
	}


	/**
	 * \brief Collect the statistics from given dataset.
	 * \tparam T Numerical type that is convertible to F type.
	 * \param data The dataset being analyzed.
	 */
	template<typename T>
	void collect(const T* data, size_t count)
	{
		for (size_t i = 0; i < count; ++i)
			collect((F)data[i]);
	}

	/**
	 * \brief Collect the statistics from given dataset.
	 * \tparam T Numerical type that is convertible to F type.
	 * \param data The dataset being analyzed.
	 */
	template<typename T>
	void collect(const std::vector<T> &data)
	{
		if (data.empty()) return;
		collect(&data[0], data.size());
	}


	/**
	 * \brief Dumps the humanly-readable representation of the statistics into
	 *		given stream (e.g., std. output).
	 * \param out The output stream.
	 * \param endline Flag that indicate that the endline will be added at the end.
	 */
	void print(std::ostream &out = std::cout, bool endline = true)
	{
		out << getCount() << " values from [" << getMinimum() << ", " << getMaximum() << "] range of average "
			<< getAverage() << " and variance " << getVariance() << " (std dev. " << getStdDeviation() << ")";
		if (endline) out << std::endl;
	}
};


/**
 * \brief Keeps statistics about repretitively measured times of the same experiment.
 *		This statistics are particularly usefuly for performance benchmarks,
 *		where each test is repeated multiple times to rule out tainted results.
 */
class TimeStats
{
private:
	double mTolerance;				///< Relative tolerance of the values.
	size_t mMinValues;				///< Minimal number of values required for significant stats.
	std::vector<double> mTimes;		///< Collected measured values.
	double mAverage;				///< Computed average of the times.
	double mVariance;				///< Computed variance (if < 0, the updateStats needs to be called).


	/**
	 * \brief Internal method that sorts and filters times and update mVariance.
	 */
	void updateStats()
	{
		if (mVariance >= 0.0 || mTimes.empty())
			return;
		
		// Make sure that the times are sorted and within tolerance (w.r.t. smallest time).
		std::sort(mTimes.begin(), mTimes.end());
		while ((mTimes.size() > 1) && (mTimes.back() > mTimes.front() * (1.0 + mTolerance)))
			mTimes.pop_back();

		// Compute average and variance.
		BasicStats<double> stats;
		stats.collect(mTimes);
		mAverage = stats.getAverage();
		mVariance = stats.getVariance();
	}


public:
	/**
	 * \brief Initialize the time statistics.
	 * \param tolerance Relative tolerance of variance.
	 * \param minValues How many measured values must be present,
	 *		so that the results are considered accurate.
	 */
	TimeStats(double tolerance = 0.2, size_t minValues = 3)
		: mTolerance(tolerance), mMinValues(minValues), mVariance(-1.0)
	{
		if (tolerance <= 0.0)
			throw (bpp::RuntimeError() << "Invalid tolerance value (" << tolerance << "). Tolerance must be positive.");
	}

	/**
	 * \brief Add another measured time.
	 */
	void add(double time)
	{
		if (time < 0.0)
			throw (bpp::RuntimeError() << "Invalid time value (" << time << "). Measured time must be positive.");
		
		mTimes.push_back(time);
		mVariance = -1.0;			// mark the structure dirty
	}


	/**
	 * \brief Return number of relevant measured times in the TimeStats.
	 */
	size_t validValues()
	{
		updateStats();
		return mTimes.size();
	}


	/**
	 * \brief Reset all measurements.
	 */
	void clear()
	{
		mTimes.clear();
		mVariance = -1.0;			// mark the structure dirty
	}


	/**
	 * \brief Return average time after the outliers are filtered out.
	 */
	double getTime()
	{
		if (mTimes.empty())
			throw bpp::RuntimeError("There are no measured values in the TimeStats.");
		updateStats();
		return mAverage;
	}


	/**
	 * \brief Return time variance after the outliers are filtered out.
	 */
	double getVariance()
	{
		if (mTimes.empty())
			throw bpp::RuntimeError("There are no measured values in the TimeStats.");
		updateStats();
		return mVariance;
	}


	/**
	 * \brief Return true if more values should be collected in order to produce
	 *		statistically sound results.
	 */
	bool moreMeasurementsRecommended()
	{
		updateStats();
		return (mTimes.size() < mMinValues);
	}
};

}

#endif

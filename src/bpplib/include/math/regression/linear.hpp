/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 17.2.2017
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_MATH_REGRESSION_LINEAR_HPP
#define BPPLIB_MATH_REGRESSION_LINEAR_HPP


#include <misc/exception.hpp>

#include <vector>
#include <assert.h>


namespace bpp {

	// Private namespace instead of nested classes (nested classes does not allow template specializations).
	namespace _priv_slr {
		/**
		 * Container getter for SLR.
		 */
		template<typename FLOAT, int DIM, class CONTAINER>
		class DefaultGetter
		{
		public:
			static std::size_t size(const CONTAINER &container) { return container.size(); }
			static FLOAT get(const CONTAINER &container, std::size_t i, std::size_t d) { return container[i][d]; }
		};


		/**
		 * Specialized getter for vector of pairs to be used for simple 2D data.
		 */
		template<typename FLOAT, int DIM>
		class DefaultGetter<FLOAT, DIM, std::vector<std::pair<FLOAT, FLOAT> > >
		{
		public:
			static std::size_t size(const std::vector<std::pair<FLOAT, FLOAT> > &container) { return container.size(); }
			static FLOAT get(const std::vector<std::pair<FLOAT, FLOAT> > &container, std::size_t i, std::size_t d) {
				static_assert(DIM == 1, "Selected DefaultGetter specialization is available only for 2D data (DIM == 1).");
				return (d == 0) ? container[i].first : container[i].second;
			}
		};
	}



/**
 * Linear regression for 2D problems that uses ordinary least-squares estimation.
 * \tparam FLOAT Type representing real numbers (float or double).
 * \tparam DIM Dimension of x-vectors (i.e. DIM==1 is for trivial regression y ~ beta x + epsilon).
 */
template<typename FLOAT = double, int DIM = 1>
class SimpleLinearRegression
{
public:
	/**
	 * Internal structure used to accumulate points.
	 */
	struct Point
	{
		FLOAT values[DIM+1];
		FLOAT& x(std::size_t idx = 0) { return values[idx]; }
		FLOAT& y() { return values[DIM]; }
		
		Point(const FLOAT *p = nullptr)
		{
			if (p == nullptr)
				for (std::size_t d = 0; d <= DIM; ++d) values[d] = (FLOAT)0.0;
			else
				for (std::size_t d = 0; d <= DIM; ++d) values[d] = p[d];
		}

		Point(FLOAT _x, FLOAT _y)
		{
			static_assert(DIM == 1, "Selected function overload is available only for 2D data (DIM == 1).");
			x() = _x;
			y() = _y;
		}

		FLOAT& operator[](std::size_t idx)				{ return values[idx]; }
		const FLOAT& operator[](std::size_t idx) const	{ return values[idx]; }
	};

private:
	std::vector<Point> mPoints;		///< Internal container holding the points.


public:
	// Basic accessors to internal container.
	Point& operator[](std::size_t idx)				{ return mPoints[idx]; }
	const Point& operator[](std::size_t idx) const	{ return mPoints[idx]; }
	std::size_t size() const						{ return mPoints.size(); }

	/**
	 * Add point into internal container.
	 * \param p Point represented as an array of FLOATs (x1, x2, ... y).
	 */
	void addPoint(FLOAT p[])
	{
		mPoints.push_back(Point(p));
	}

	/**
	 * Add point into internal container. This is a special function that works only iff DIM == 1.
	 * \param x The regressor variable.
	 * \param y The dependent variable.
	 */
	void addPoint(FLOAT x, FLOAT y)
	{
		static_assert(DIM == 1, "Selected function overload is available only for 2D data (DIM == 1).");
		mPoints.push_back(Point(x, y));
	}

	/**
	 * Add point into internal container. This is a special function that works only iff DIM == 1.
	 * \param p Pair of variables. First is the regressor (x), second is the dependent variable (y).
	 */
	void addPoint(const std::pair<FLOAT, FLOAT> &p)
	{
		static_assert(DIM == 1, "Selected function overload is available only for 2D data (DIM == 1).");
		mPoints.push_back(Point(p.first, p.second));
	}


	/**
	 * Remove all points from internal container.
	 */
	void clear()
	{
		mPoints.clear();
	}


	/**
	 * Compute leas-square estimate of the slope if the data are centered (regression line passes through [0,0]).
	 * This is a special version for 2D problems.
	 * \tparam CONTAINER Type of the container holding the points.
	 * \tparam GETTER Class with static methods size(c) and get(c, i, d) that handles data fetching from the container.
	 * \param container The container from which the points are taken.
	 */
	template<class CONTAINER, class GETTER = _priv_slr::DefaultGetter<FLOAT, DIM, CONTAINER> >
	static FLOAT estimateCentered(const CONTAINER &container)
	{
		static_assert(DIM == 1, "Selected function overload is available only for 2D data (DIM == 1).");
		std::size_t count = GETTER::size(container);
		if (count == 0)
			throw bpp::RuntimeError("Unable to compute estimate. Given container is empty.");

		FLOAT sumXY = (FLOAT)0.0, sumX2 = (FLOAT)0.0;
		for (std::size_t i = 0; i < count; ++i)
		{
			FLOAT x = GETTER::get(container, i, 0);
			FLOAT y = GETTER::get(container, i, 1);
			sumXY += x * y;
			sumX2 += x * x;
		}
		return sumXY / sumX2;
	}

	/*
	 * Compute leas-square estimate of the slope if the data are centered (regression line passes through [0,0]).
	 * The points from internal container are used. This is a special version for 2D problems.
	 */
	FLOAT estimateCentered() const
	{
		return estimateCentered(mPoints);
	}


	/*
	 * Compute the mean of square errors (residuals).
	 * \tparam CONTAINER Type of the container holding the points.
	 * \tparam GETTER Class with static methods size(c) and get(c, i, d) that handles data fetching from the container.
	 * \param container The container from which the points are taken.
	 * \param coefficients Array of estimated coefficients (beta_1 ... beta_DIM).
	 * \param beta0 Zero coefficient beta_0 which determines the shift of the fitted line from [0,0].
	 */
	template<class CONTAINER, class GETTER = _priv_slr::DefaultGetter<FLOAT, DIM, CONTAINER> >
	static FLOAT meanResiduals(const CONTAINER &container, const FLOAT coefficients[DIM], FLOAT beta0 = (FLOAT)0.0)
	{
		std::size_t count = GETTER::size(container);
		if (count == 0)
			throw bpp::RuntimeError("Unable to compute mean residuals. Given container is empty.");

		FLOAT sum = (FLOAT)0.0;
		for (std::size_t i = 0; i < count; ++i) {
			FLOAT diff = GETTER::get(container, i, DIM) - beta0;	// shifted y
			for (std::size_t d = 0; d < DIM; ++d)
				diff -= GETTER::get(container, i, d) * coefficients[d];
			sum += diff * diff;
		}
		return sum / (FLOAT)count;
	}


	/*
	 * Compute the mean of square errors (residuals). This is a special version for 2D problems.
	 * \tparam CONTAINER Type of the container holding the points.
	 * \tparam GETTER Class with static methods size(c) and get(c, i, d) that handles data fetching from the container.
	 * \param container The container from which the points are taken.
	 * \param beta1 Beta coefficient multiplied with x (slope of the line).
	 * \param beta0 Zero coefficient beta_0 which determines the shift of the fitted line from [0,0].
	 */
	template<class CONTAINER, class GETTER = _priv_slr::DefaultGetter<FLOAT, DIM, CONTAINER> >
	static FLOAT meanResiduals(const CONTAINER &container, FLOAT beta1, FLOAT beta0 = (FLOAT)0.0)
	{
		static_assert(DIM == 1, "Selected function overload is available only for 2D data (DIM == 1).");
		FLOAT coef[] = { beta1 };
		return meanResiduals<CONTAINER, GETTER>(container, coef, beta0);
	}


	/*
	 * Compute the mean of square errors (residuals) from the internal points.
	 * \param coefficients Array of estimated coefficients (beta_1 ... beta_DIM).
	 * \param beta0 Zero coefficient beta_0 which determines the shift of the fitted line from [0,0].
	 */
	FLOAT meanResiduals(const FLOAT coefficients[DIM], FLOAT beta0 = (FLOAT)0.0) const
	{
		return meanResiduals(mPoints, coefficients, beta0);
	}


	/*
	 * Compute the mean of square errors (residuals) from the internal points. This is a special version for 2D problems.
	 * \param beta1 Beta coefficient multiplied with x (slope of the line).
	 * \param beta0 Zero coefficient beta_0 which determines the shift of the fitted line from [0,0].
	 */
	FLOAT meanResiduals(FLOAT beta1, FLOAT beta0 = (FLOAT)0.0) const
	{
		return meanResiduals(mPoints, beta1, beta0);
	}
};




}


#endif

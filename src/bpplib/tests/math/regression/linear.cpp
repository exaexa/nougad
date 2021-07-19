#include "../../test.hpp"
#include <math/regression/linear.hpp>

#include <random>
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdint>



/**
* \brief Tests the math/random.hpp features on 32-bit unsigned ints.
*/
class BPPMathRegressionLinearSimple2DTest : public BPPLibTest
{
private:
	/*
	 * Internal structure that represents point in 2D.
	 */
	struct Point
	{
		double values[2];
		double& operator[](std::size_t i) { return values[i]; }
		const double& operator[](std::size_t i) const { return values[i]; }
	};

public:
	BPPMathRegressionLinearSimple2DTest() : BPPLibTest("math/regression/linear/simple2d") {}

	virtual bool run() const
	{
		// Parameters for the test
		const std::size_t count = 1000;
		const double beta = 0.42;
		const double errorLimit = 0.01;

		// Generate data ...
		std::random_device rd;
		std::mt19937 generator(rd());
		std::uniform_real_distribution<double> distribution(-1.0, 1.0);

		std::cout << "Generating " << count << " 2D points ..." << std::endl;
		bpp::SimpleLinearRegression<> slr;
		std::vector<Point> points(count);
		std::vector< std::pair<double, double> > points2(count);
		for (std::size_t i = 0; i < count; ++i) {
			double x = distribution(generator);
			double y = beta * x + errorLimit * distribution(generator);
			slr.addPoint(x, y);
			points[i][0] = x;  points[i][1] = y;
			points2[i].first = x;  points2[i].second = y;
		}

		// Compute slope estimates for all three datasets and their residuals ...
		std::cout << "Computing linear regression estimates ..." << std::endl;
		double est0 = slr.estimateCentered();
		double est1 = bpp::SimpleLinearRegression<>::estimateCentered(points);
		double est2 = bpp::SimpleLinearRegression<>::estimateCentered(points2);
		double res0 = slr.meanResiduals(est0);
		double res1 = bpp::SimpleLinearRegression<>::meanResiduals(points, est1);
		double res2 = bpp::SimpleLinearRegression<>::meanResiduals(points2, est2);

		// Verify
		if (est0 != est1 || est1 != est2) {
			std::cout << "Three estimates on the same data differs (" << est0 << ", " << est1 << ", " << est2 << ") whilst they should be the same." << std::endl;
			return false;
		}
		if (res0 != res1 || res1 != res2) {
			std::cout << "Three mean square residuals on the same data differs (" << res0 << ", " << res1 << ", " << res2 << ") whilst they should be the same." << std::endl;
			return false;
		}

		std::cout << "Computed beta estimate: " << est0 << " (original: " << beta << "), mean residuals: " << res0 << std::endl;
		if (std::abs(est0 - beta) > 0.01) {
			std::cout << "The estimate is too far from initial beta value." << std::endl;
			return false;
		}

		return true;
	}
};




BPPMathRegressionLinearSimple2DTest _mathRegressionLinearSimple2DTest;

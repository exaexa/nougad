//#define BPP_DEBUG

#include "../../test.hpp"
#include <algo/clustering/k-means.hpp>

#include <random>
#include <vector>
#include <iostream>
#include <cmath>


/**
 * \brief Test of the k-means clustering algorithm on simple 2D points.
 */
template<typename FLOAT = float>
class BPPAlgoClusteringKMeansTest : public BPPLibTest
{
private:
	/**
	 * \brief Point in R^2 space.
	 */
	struct Point
	{
		FLOAT x, y;
		Point(FLOAT _x, FLOAT _y) : x(_x), y(_y) {}
	};

	/**
	 * Just a holder of two static methods used for accessing the point data in our containers.
	 */
	class Getter
	{
	public:
		static FLOAT get(const std::vector<Point> &container, std::size_t idx, std::size_t d)
		{
			return (d == 0) ? container[idx].x : container[idx].y;	// quee, quee
		}

		static std::size_t size(const std::vector<Point> &container)
		{
			return container.size();
		}
	};


	/**
	 * \brief Generate clustered points using set of base points as cluster centers and
	 *		given variance for uniform distribution (in each dimension).
	 * \param basePoints An array of base points (cluster centers)
	 * \param clusters Number of clusters (i.e., and base points).
	 * \param count Number of points being generated.
	 * \param points Vector where the generated points are stored.
	 * \param groundTruth Vector wehere cluster assignments are stored.
	 * \param variance The points are randomly generated in the interval (-variance, variance)
	 *		around corresponding base points (in each dimension respectively).
	 */
	void generatePoints(const std::vector<Point> &basePoints, size_t count,
		std::vector<Point> &points, std::vector<size_t> &groundTruth, FLOAT variance) const
	{
		points.clear();
		groundTruth.clear();
		for (std::size_t i = 0; i < basePoints.size(); ++i) {
			groundTruth.push_back(i);
			points.push_back(basePoints[i]);
		}

		// Generate data ...
		std::random_device rd;
		std::mt19937 generator(rd());
		std::uniform_real_distribution<FLOAT> varDist(-variance, variance);
		std::uniform_int_distribution<std::size_t> sizeDist(0, basePoints.size() - 1);

		while (count > 0) {
			--count;
			size_t cluster = sizeDist(generator);
			groundTruth.push_back(cluster);
			points.push_back(Point(
				basePoints[cluster].x + varDist(generator),
				basePoints[cluster].y + varDist(generator)
			));
		}
	}


public:
	BPPAlgoClusteringKMeansTest() : BPPLibTest(std::string("algo/clustering/k-means/") + std::string(sizeof(FLOAT) < 8 ? "float" : "double")) {}

	virtual bool run() const
	{
		std::vector<Point> basePoints{
			Point((FLOAT)1.0,	(FLOAT)3.0),
			Point((FLOAT)3.0,	(FLOAT)1.0),
			Point((FLOAT)0.0,	(FLOAT)0.0),
			Point((FLOAT)-4.0,	(FLOAT)0.0),
			Point((FLOAT)1.0,	(FLOAT)-3.0),
		};
		const std::size_t clusters = basePoints.size();

		// Generate clusters
		std::vector<Point> points;
		std::vector<size_t> groundTruth;
		generatePoints(basePoints, clusters * 8, points, groundTruth, (FLOAT)0.1);

		// Clustering...
		std::cout << "Running k-means (default) ... ";
		typedef bpp::KMeans<2, std::vector<Point>, Getter, FLOAT> KMeans_t;
		KMeans_t kMeans(clusters, 10);
		kMeans.setSeeds(basePoints);
		std::cout << kMeans.run(points) << " iterations passed." << std::endl;

		// Check the means ...
		const std::vector<typename KMeans_t::Point> &means = kMeans.getMeans();
		if (means.size() != clusters) {
			std::cout << "Invalid number (" << means.size() << ") of clusters, " << clusters << " expected." << std::endl;
			return false;
		}

		bool ok = true;
		for (std::size_t i = 0; i < means.size(); ++i) {
			FLOAT diff1 = (FLOAT)std::abs(means[i][0] - basePoints[i].x);
			FLOAT diff2 = (FLOAT)std::abs(means[i][1] - basePoints[i].y);
			if (diff1 > (FLOAT)0.1 || diff2 > (FLOAT)0.1) {
				std::cout << "Mean #" << i << " is off by abs. values (" << diff1 << ", " << diff2 << ")." << std::endl;
				ok = false;
			}
		}
		if (!ok) return false;

		// Check the assignment ...
		const std::vector<std::size_t>& assignments = kMeans.getAssignments(points);
		if (assignments.size() != points.size()) {
			std::cout << "Total " << points.size() << " points yielded " << assignments.size() << " assignment values." << std::endl;
			return false;
		}

		for (std::size_t i = 0; i < points.size(); ++i) {
			if (assignments[i] != groundTruth[i]) {
				std::cout << "Point #" << i << " is assigned to cluster " << assignments[i]
					<< ", while " << groundTruth[i] << " was expected." << std::endl;
				ok = false;
			}
		}
		if (!ok) return false;


		// Second test that joins all clusters together...
		std::cout << "Running k-means (testing joinDist) ... ";
		KMeans_t kMeans2(clusters, 10, 0, (FLOAT)10.0);
		kMeans2.setSeeds(basePoints);
		std::cout << kMeans2.run(points) << " iterations passed." << std::endl;

		if (kMeans2.getMeans().size() > 1) {
			std::cout << "Only one cluster expected, but " << kMeans2.getMeans().size() << " clusters yielded." << std::endl;
			return false;
		}

		std::vector<std::size_t> assignments2;
		kMeans2.getAssignments(points, assignments2, points.size(), true);
		for (std::size_t i = 0; i < assignments2.size(); ++i)
			if (assignments2[i] != 0) {
				std::cout << "Point #" << i << " is assigned to cluster " << assignments2[i]
					<< ", while 0 was expected." << std::endl;
				ok = false;
			}

		return ok;
	}
};


BPPAlgoClusteringKMeansTest<float> _algoClusteringKMeansFloatTest;
BPPAlgoClusteringKMeansTest<double> _algoClusteringKMeansDoubleTest;

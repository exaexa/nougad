//#define BPP_DEBUG

#include "../../test.hpp"
#include <algo/clustering/k-medoids.hpp>
#include <math/random.hpp>

#include <vector>
#include <iostream>
#include <cmath>


/**
 * \brief Test of the k-medoids clustering algorithm on simple Euclidean 2D space.
 */
template<typename FLOAT = float>
class BPPAlgoClusteringKMedoidsEuclidTest : public BPPLibTest
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
	 * \brief Euclidean distance for 2d points.
	 */
	class PointDist
	{
	public:
		FLOAT operator()(const Point &point1, const Point &point2)
		{
			FLOAT dx = point1.x - point2.x;
			FLOAT dy = point1.y - point2.y;
			return std::sqrt(dx*dx + dy*dy);
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
	void generatePoints(const Point basePoints[], size_t clusters, size_t count,
		std::vector<Point> &points, std::vector<size_t> &groundTruth, FLOAT variance) const
	{
		points.clear();
		groundTruth.clear();

		while (count > 0) {
			--count;
			size_t cluster = bpp::Random<size_t>::next(clusters);
			groundTruth.push_back(cluster);
			points.push_back(Point(
				basePoints[cluster].x + bpp::Random<FLOAT>::next(-variance, variance),
				basePoints[cluster].y + bpp::Random<FLOAT>::next(-variance, variance)
			));
		}
	}


	/**
	 * \brief Verification routine that calculates nearest medoid for given point.
	 * \param point The point for which the medoid is being looked up.
	 * \param points Vector of all points.
	 * \param medoids Vector of medoid indices.
	 * \return Index of the nearest medoid (within the medoids vector).
	 */
	size_t getNearestMedoid(Point point, const std::vector<Point> &points, const std::vector<size_t> &medoids) const
	{
		static PointDist distFnc;
		size_t medoid = 0;
		FLOAT minDist = distFnc(point, points[medoids[0]]);
		for (size_t m = 1; m < medoids.size(); ++m) {
			FLOAT dist = distFnc(point, points[medoids[m]]);
			if (dist < minDist) {
				minDist = dist;
				medoid = m;
			}
		}
		return medoid;
	}

public:
	BPPAlgoClusteringKMedoidsEuclidTest() : BPPLibTest("algo/clustering/k-medoids/euclidean") {}

	virtual bool run() const
	{
		const Point basePoints[] = {
			Point((FLOAT)1.0,	(FLOAT)3.0),
			Point((FLOAT)3.0,	(FLOAT)1.0),
			Point((FLOAT)0.0,	(FLOAT)0.0),
			Point((FLOAT)-4.0,	(FLOAT)0.0),
			Point((FLOAT)1.0,	(FLOAT)-3.0),
		};
		const size_t clusters = sizeof(basePoints) / sizeof(Point);

		// Generate clusters
		std::vector<Point> points;
		std::vector<size_t> groundTruth;
		generatePoints(basePoints, clusters, clusters*8, points, groundTruth, 0.2f);

		// Clustering...
		PointDist pointDist;
		bpp::KMedoids<Point, PointDist, FLOAT> kmedoids(pointDist, clusters, 10);

		std::vector<size_t> medoids;
		std::vector<size_t> assignments;
		std::cout << "Running k-medoids ... ";
		std::cout << kmedoids.run(points, medoids, assignments) << " iterations passed." << std::endl;

		// Print
		#ifdef BPP_DEBUG
			for (size_t m = 0; m < medoids.size(); ++m) {
				for (size_t i = 0; i < assignments.size(); ++i) {
					if (assignments[i] == m) {
						std::cout << points[i].x << ", " << points[i].y << "\t(" << groundTruth[i] << ")";
						if (i == medoids[m])
							std::cout << " *";
						std::cout << std::endl;
					}
				}
				std::cout << std::endl;
			}
		#endif

		if (medoids.size() != clusters) {
			std::cout << "Invalid number (" << medoids.size() << ") of clusters, " << clusters << " expected." << std::endl;
			return false;
		}

		if (assignments.size() != points.size()) {
			std::cout << "Total " << points.size() << " yielded " << assignments.size() << " assignment values." << std::endl;
			return false;
		}

		// Verify that each point has its nearest medoid.
		bool ok = true;
		for (size_t i = 0; i < assignments.size(); ++i) {
			size_t medoid = getNearestMedoid(points[i], points, medoids);
			if (assignments[i] != medoid) {
				std::cout << "Point " << i << " is assigned to " << assignments[i] << ", but nearest medoid is " << medoid << std::endl;
				ok = false;
			}
		}

		std::cout << "Recycle k-medoids for the same run again ... ";
		std::cout << kmedoids.run(points, medoids, assignments) << " iterations passed." << std::endl;

		return ok;
	}
};


BPPAlgoClusteringKMedoidsEuclidTest<float> _algoClusteringKMedoidsEuclidTest;

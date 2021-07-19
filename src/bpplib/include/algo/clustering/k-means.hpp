/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 12.3.2017
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_ALGO_CLUSTERING_K_MEANS_HPP
#define BPPLIB_ALGO_CLUSTERING_K_MEANS_HPP


#include <random>
#include <vector>
#include <algorithm>
#include <limits>
#include <cassert>

#ifdef BPP_DEBUG
#include <iostream>
#endif

namespace bpp
{


/**
 * Configuration parameters used for k-means clustering.
 * \tparam FLOAT Type of real numbers (typically float or double).
 */
template<typename FLOAT = float>
struct KMeansConfig {
public:
	std::size_t k;					///< Initial number of clusters.
	std::size_t maxIters;			///< Maximal number of iterations.
	std::size_t minSize;			///< Minimal size of a cluster (smaller clusters are dismissed).
	FLOAT joinDist;					///< Cluster joining squared distance (if two clusters are closer than this value, smaller one is dismissed).
	FLOAT outlierDist;				///< Maximal squared distance between point and centroid allowed (points which are further may not belong to the cluster).

	KMeansConfig(std::size_t _k = 0, std::size_t _maxIters = 100, std::size_t _minSize = 0, FLOAT _joinDist = 0.0, FLOAT _outlierDist = std::numeric_limits<FLOAT>::infinity())
		: k(_k), maxIters(_maxIters), minSize(_minSize), joinDist(_joinDist), outlierDist(_outlierDist)
	{}
};



/**
 * \brief A K-means algorithm. The object holds algorithm parameters as well as the result of the last run.
 * \tparam DIM Dimension of the points being clustered (i.e., we are working in R^DIM space).
 * \tparam CONTAINER Container type that holds all the points.
 * \tparam GETTER Class with two static methods: get(container, idx, dim) and size(container)
 *		The get() method is used to fetch coordinates from container, size() gets the number of points in it.
 * \tparam FLOAT Float data type with selected precision (used for distances and medoid scores).
 *		It should be set to float or double.
 */
template<int DIM, class CONTAINER, class GETTER, typename FLOAT = float>
class KMeans
{
public:
	/**
	 * Wrapper for points of given dimensionality (basically, they are fixed-sized arrays of FLOATs).
	 */
	class Point
	{
	private:
		FLOAT mCoordinates[DIM];	///< Container for all the point coordinates.
	public:
		Point()
		{
			clear();
		}

		FLOAT& operator[](std::size_t idx) { return mCoordinates[idx]; }
		const FLOAT& operator[](std::size_t idx) const { return mCoordinates[idx]; }
		
		void clear()
		{
			for (std::size_t d = 0; d < DIM; ++d) mCoordinates[d] = (FLOAT)0.0;
		}

		void fetch(const CONTAINER &container, std::size_t idx)
		{
			assert(idx < GETTER::size(container));
			for (std::size_t d = 0; d < DIM; ++d)
				mCoordinates[d] = GETTER::get(container, idx, d);
		}

		/**
		 * Computes L2 squared distance (i.e., without the final sqrt) between this point and given point in container.
		 */
		FLOAT dist(const CONTAINER &container, std::size_t idx) const
		{
			assert(idx < GETTER::size(container));
			FLOAT sum = (FLOAT)0.0;
			for (std::size_t d = 0; d < DIM; ++d) {
				FLOAT diff = mCoordinates[d] - GETTER::get(container, idx, d);
				sum += diff * diff;
			}
			return sum;
		}

		/**
		 * Computes L2 squared distance (i.e., without the final sqrt) between this point and given point.
		 */
		FLOAT dist(const Point &p) const
		{
			FLOAT sum = (FLOAT)0.0;
			for (std::size_t d = 0; d < DIM; ++d) {
				FLOAT diff = mCoordinates[d] - p.mCoordinates[d];
				sum += diff * diff;
			}
			return sum;
		}
	};



protected:
	KMeansConfig<FLOAT> mConfig;			///< Coonfiguration parameters of the clustering algorithm.
	std::vector<Point> mMeans;				///< The centers of existing clusters.
	std::vector<std::size_t> mAssignments;	///< Last computed assignments.

	/**
	 * Internal procedure that computes assignment.
	 * The result is directly stored into mAssignemnts.
	 */
	void computeAssignments(const CONTAINER &points, std::size_t count, std::vector<std::size_t> &assignments)
	{
		if (assignments.size() != count) assignments.resize(count);
		std::size_t k = mMeans.size();
		std::int64_t icount = (std::int64_t)count;	// conversion to signed int for purposes of OMP

		#pragma omp parallel for
		for (std::int64_t ompP = 0; ompP < icount; ++ompP) {
			std::size_t p = (std::size_t)ompP;
			std::size_t nearest = 0;
			FLOAT nearestDist = mMeans[nearest].dist(points, p);
			for (std::size_t i = 1; i < k; ++i) {
				FLOAT dist = mMeans[i].dist(points, p);
				if (dist < nearestDist) {
					nearestDist = dist;
					nearest = i;
				}
			}

			// Save cluster ID (outliers are reported as max values -- i.e., not belonging into any cluster)
			assignments[p] = (nearestDist <= mConfig.outlierDist) ? nearest : std::numeric_limits<std::size_t>::max();
		}
	}


	/**
	 * Postprocess newly computed means. The method is templated by two bools which indicate, what exactly should be performed.
	 *		The expected usage is calling the method using <T,F> and subsequently <F,T>, or calling it just once with <T,T>,
	 *		based on whether the averaging and compacting needs to be separated or not. Calling <F,F> configuration has no effect.
	 * \tparam DIV True if the means should be divided by their respective counts (centroid is computed as an average).
	 * \tparam COMPACT True if the means and counts should be compacted (small clusters are dismissed).
	 * \param means Vector with centroid point (or their sums).
	 * \param counts Vector with numbers of elements assigned to each cluster.
	 * \return True if the means have changed, false if they are the same as in the previous iteration.
	 */
	template<bool DIV, bool COMPACT>
	bool postprocessMeans(std::vector<Point> &means, std::vector<std::size_t> &counts) const
	{
		if (!DIV && !COMPACT) return false;
		bool changed = false;
		std::size_t newPos = 0;								// writing position in the mMeans (when compacting)
		for (std::size_t i = 0; i < mMeans.size(); ++i) {	// i is reading position from newMeans
			// Empty clusters are skipped completely.
			if (counts[i] == 0) {	
				changed |= COMPACT;		// when compacting, empty cluster is dismissed -> change occurs
				continue;
			}

			if (counts[i] > mConfig.minSize || !COMPACT) {	// when compacting, the the cluster must be large enough
				// Value for each dimension has to be processed ...
				for (std::size_t d = 0; d < DIM; ++d) {
					FLOAT val;
					if (DIV) {		// performing averaging (divide by size) ...
						val = means[i][d] / (FLOAT)counts[i];
						changed = changed || (val != mMeans[i][d]);
					}
					else			// only copying ...
						val = means[i][d];
					
					if (COMPACT) {
						// move for compacting
						means[newPos][d] = val;
						counts[newPos] = counts[i];
					}
					else
						// just save the value where it was
						means[i][d] = val;
				}

				if (COMPACT) ++newPos;	// one mean was written (keep track of filled positions)
			}
			else
				changed = true;			// a cluster was skipped (=> change occurs)
		}

		// Finally, resize the arrays if necessary
		if (COMPACT && newPos != means.size()) {
			means.resize(newPos);
			counts.resize(newPos);
		}

		return changed;
	}


	/**
	 * Internal structure that represents cluster metadata.
	 */
	struct Cluster
	{
		std::size_t idx;	///< Original index (reference to related arrays)
		std::size_t count;	///< Number of associated 
		Point mean;

		Cluster(std::size_t _idx, std::size_t _count, Point _mean)
			: idx(_idx), count(_count), mean(_mean) {}
	};


	/**
	 * Join clusters which are closer than join distance.
	 */
	void joinMeans(std::vector<Point> &means, std::vector<std::size_t> &counts) const
	{
		if (mConfig.joinDist <= (FLOAT)0.0) return;

		// Assemble the means and counts into cluster records...
		std::vector<Cluster> clusters;
		clusters.reserve(means.size());
		for (std::size_t i = 0; i < means.size(); ++i) {
			if (counts[i] == 0) continue;
			clusters.push_back(Cluster(i, counts[i], means[i]));
		}

		// ... and sort them by weights.
		std::sort(clusters.begin(), clusters.end(),
			[](const Cluster &a, const Cluster &b) -> bool
			{
				return a.count > b.count || (a.count == b.count && a.idx < b.idx);
			}
		);

		// Perform the joining ...
		std::vector<std::size_t> toJoin;
		toJoin.reserve(clusters.size());
		for (std::size_t i = 0; i < clusters.size(); ++i) {
			if (clusters[i].count == 0) continue;

			// Find all clusters for joining and accumulate their coordinates and counts into mean and count.
			Point mean;
			std::size_t count = 0;
			for (std::size_t j = i + 1; j < clusters.size(); ++j) {
				if (clusters[j].count > 0 && clusters[i].mean.dist(clusters[j].mean) < mConfig.joinDist) {
					// Add the cluster coordinates to newly prepared mean of current cluster.
					for (std::size_t d = 0; d < DIM; ++d)
						mean[d] += clusters[j].mean[d] * (FLOAT)clusters[j].count;
					count += clusters[j].count;

					// Mark the joined cluster as empty (so it will be dismissed).
					counts[clusters[j].idx] = 0;
					clusters[j].count = 0;
				}
			}
			if (count == 0) continue;

			// Finally, compute the new mean of the joined cluster and save it back to means.
			count += clusters[i].count;
			for (std::size_t d = 0; d < DIM; ++d)
				means[clusters[i].idx][d] = (mean[d] + clusters[i].mean[d] * (FLOAT)clusters[i].count) / (FLOAT)count;
			counts[clusters[i].idx] = count;
		}
	}


	/**
	 * The actual algorithm of k-means.
	 * \param points The container holding the points.
	 * \param count Size/limit of the points being processed.
	 */
	virtual std::size_t runIntern(const CONTAINER &points, std::size_t count, bool(*iterCallback)(const KMeans<DIM, CONTAINER, GETTER, FLOAT>&))
	{
		// Get and check points count ...
		count = std::min(count, GETTER::size(points));
		if (mConfig.k > count)
			throw (bpp::LogicError() << "Unable to perform clustering since " << mConfig.k
				<< " clusters are expected but only " << count << " points were given.");

		// Make sure that seeds (initial means) are ready, otherwise take random subset of points.
		if (mMeans.size() < mConfig.k) setSeeds(points, true);	// true = randomize

		// Iteretively refine the means.
		std::size_t iters = 0;
		//NewMeans newMeans;
		std::vector<Point> newMeans;
		std::vector<std::size_t> newCounts;
		while (iters < mConfig.maxIters) {
			++iters;

			// Prepare new means ...
			newMeans.resize(mMeans.size());
			newCounts.resize(mMeans.size());
			for (auto && p : newMeans) p.clear();
			for (auto && c : newCounts) c = 0;

			// Compute assignment and accumulate new means sums...
			computeAssignments(points, count, mAssignments);
			for (std::size_t p = 0; p < count; ++p) {
				std::size_t nearest = mAssignments[p];
				if (nearest >= newMeans.size()) continue;	// outliers does not affect centroids
				for (std::size_t d = 0; d < DIM; ++d)
					newMeans[nearest][d] += GETTER::get(points, p, d);
				newCounts[nearest] += 1;
			}

			// Perform means postprocessing (averaging, joining, compacting...).
			bool changed = false;
			if (mConfig.joinDist > (FLOAT)0.0) {
				bool ch = postprocessMeans<true, false>(newMeans, newCounts);	// Compute centroid coordinates, but do not compact yet...				
				joinMeans(newMeans, newCounts);									// Join clusters which are too close together ...
				changed = postprocessMeans<false, true>(newMeans, newCounts);	// Dismiss small clusters and compact the means
				changed = changed || ch;
			}
			else
				changed = postprocessMeans<true, true>(newMeans, newCounts);		// Both compute coordinates and compact the arrays

			// Stable clustering found -> terminate
			if (!changed) break;

			// Swap new means with current means.
			mMeans.swap(newMeans);

			if (iterCallback && iters < mConfig.maxIters && !iterCallback(*this))
				return iters;
		}
		return iters;
	}


public:
	KMeans(std::size_t k, std::size_t maxIters, std::size_t minSize = 0, FLOAT joinDist = (FLOAT)0.0, FLOAT outlierDist = std::numeric_limits<FLOAT>::infinity())
		: mConfig(k, maxIters, minSize, joinDist*joinDist, outlierDist*outlierDist), mMeans(0)
	{}

	KMeans(const KMeansConfig<FLOAT> &config) : mConfig(config), mMeans(0)
	{
		// To simplify operations, distances are kept as their squares (so we can use L2^2 distance).
		mConfig.joinDist = mConfig.joinDist * mConfig.joinDist;
		mConfig.outlierDist = mConfig.outlierDist * mConfig.outlierDist;
	}


	virtual ~KMeans() {}	// enforce virtual destructor

	/**
	 * Set seeds manually, before the k-means algorithm is initiated.
	 * If the seeds are not set, they are selected randomly from the clustered dataset.
	 * \param seeds Container holding the seeds in the same format as data points. Only mConfig.k seeds are taken.
	 * \param randomize Select random subset of size mConfig.k from the seeds (in a random permutation)
	 *		rather than taking the first mConfig.k seeds.
	 */
	void setSeeds(const CONTAINER &seeds, bool randomize = false)
	{
		std::size_t count = GETTER::size(seeds);
		if (count < mConfig.k)
			throw (bpp::LogicError() << "The initial number of clusters is set to " << mConfig.k
				<< " but only " << count << " seeds were given.");
		
		mMeans.resize(mConfig.k);
		if (randomize) {
			// Select random subset of seeds ...
			std::vector<std::size_t> indices(count);
			for (std::size_t i = 0; i < count; ++i) indices[i] = i;
			
			std::random_device rd;
			std::mt19937 generator(rd());
			std::shuffle(indices.begin(), indices.end(), generator);

			for (std::size_t i = 0; i < mConfig.k; ++i)
				mMeans[i].fetch(seeds, indices[i]);
		}
		else {
			// Select the first mConfig.k seeds ...
			for (std::size_t i = 0; i < mConfig.k; ++i)
				mMeans[i].fetch(seeds, i);
		}
	}


	/**
	 * Perform the k-means clustering. The result (means) is kept in internal structures.
	 * The assignment can be subsequently received by calling getAssignments().
	 * If no seeds were selected, k seeds are taken randomly from the points.
	 * \param points The container that holds the input points.
	 * \param count Limit the number of points. If ommitted, all points are processed.
	 * \param iterCallback Callback invoked at the end of each iteration.
	 *		It can be used both for progress reporting or to stop the algorithm (if it returns false).
	 *		The callback gets reference to *this object.
	 * \return The actual number of iterations performed
	 */
	std::size_t run(const CONTAINER &points, std::size_t count = ~(std::size_t)0,
		bool(*iterCallback)(const KMeans<DIM, CONTAINER, GETTER, FLOAT>&) = nullptr)
	{
		// Just recall private virtual method...
		return this->runIntern(points, count, iterCallback);
	}


	/**
	 * Return the result means. This method should be called after the run() method.
	 */
	const std::vector<Point>& getMeans() const
	{
		return mMeans;
	}


	const std::vector<std::size_t> getAssignments(const CONTAINER &points, std::size_t count = ~(std::size_t)0, bool forceRecount = false)
	{
		count = std::min(count, GETTER::size(points));
		if (forceRecount || mAssignments.size() != count)
			computeAssignments(points, count, mAssignments);
		return mAssignments;
	}


	/**
	 * Return points assignment into clusters. This method should be called after the run() method.
	 * \param points The container that holds the input points.
	 * \param result The vector which will hold the points assignment -- i.e., index of the cluster, to which a point belongs to.
	 *		Outliers are marked with size_t max value.
	 * \param count Limit the number of points. If ommitted, all points are processed.
	 */
	void getAssignments(const CONTAINER &points, std::vector<std::size_t> &result , std::size_t count = ~(std::size_t)0, bool forceRecount = false)
	{
		count = std::min(count, GETTER::size(points));
		if (forceRecount || mAssignments.size() != count)
			computeAssignments(points, count, result);
		else {
			result.resize(count);
			std::int64_t icount = (std::int64_t)count;
			
			#pragma omp parallel for
			for (std::int64_t i = 0; i < icount; ++i)
				result[i] = mAssignments[i];
		}
	}
};


}

#endif

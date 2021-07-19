/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 29.8.2017
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_ALGO_CLUSTERING_K_MEDOIDS_HPP
#define BPPLIB_ALGO_CLUSTERING_K_MEDOIDS_HPP

#include <math/random.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

#include <vector>
#include <algorithm>
#ifdef BPP_DEBUG
#endif
#include <iostream>

namespace bpp
{

/**
 * \brief A K-medoids algorithm.
 * \tparam OBJ Type of objects that are being clustered.
 * \tparam DIST Type of distance functor that computes/provides distances between objects.
 *		The functor must have operator() that takes two OBJs and yields a FLOAT.
 * \tparam FLOAT Float data type with selected precision (used for distances and medoid scores).
 *		It should be set to float or double.
 */
template<class OBJ, class DIST, typename FLOAT = float>
class KMedoids
{
protected:
	DIST &mDistFnc;			///< Functor used to compute object distances (OBJ, OBJ) to FLOAT
	std::size_t mK;			///< Number of desired clusters
	std::size_t mMaxIters;	///< Maximal number of algorithm iterations.

	// Distance statistics
	FLOAT mLastAvgDistance;				///< Average distance between an object and its respective medoid (computed with last update).
	FLOAT mLastAvgClusterDistance;		///< Average distance of average distances within each cluster (computed with last update).


	/**
	 * \brief Verify algorithm parameters. If they do conflict, LogicError is thrown.
	 */
	virtual void checkParams(std::size_t objectsCount)
	{
		if (mK < 2)
			throw (LogicError() << "Too few clusters (" << mK << ") selected. The k value must be at least 2.");
		if (mMaxIters == 0)
			throw (LogicError() << "At least one iteration must be allowed in the algorithm.");
		if (mK > objectsCount)
			throw (LogicError() << "The algorithm is requested to create " << mK << " clusters of " << objectsCount << " objects.");
	}


	/**
	 * \brief Create the initial selection of medoids (uniform random).
	 */
	void initRandMedoids(std::vector<std::size_t> &medoids, std::size_t objectCount) const
	{
		medoids.resize(objectCount);
		for (std::size_t i = 0; i < objectCount; ++i)
			medoids[i] = i;
		Random<std::size_t>::shuffle(medoids);
	}


	/**
	 * \brief Compute current assignment of objects to medoids according to distance functor.
	 * \param objects Input set of objects to be clustered.
	 * \param medoids The medoid objects (as indices to the objects vector).
	 * \param assignments The result assignment value for each object. The assignment vector
	 *		has the same size as objects vector and each value is an index to medoids vector.
	 * \param distances Pointer to an array, where distances to assigned medoids are kept.
	 *		If nullptr, the distances are not saved.
	 */
	void computeAssignments(const OBJ objects[], std::size_t objectCount, std::vector<std::size_t> &medoids,
		std::vector<std::size_t> &assignments, FLOAT *distances = nullptr) const
	{
		assignments.resize(objectCount);

		// Compute assignment for each object individually.
		#pragma omp parallel for
		for (std::size_t i = 0; i < objectCount; ++i) {
			#ifdef BPP_DEBUG
				std::cout << "Computing assignment of " << i << std::endl;
			#endif
			
			std::size_t asgn = 0;
			FLOAT minDist = mDistFnc(objects[i], objects[medoids[asgn]]);
	
			#ifdef BPP_DEBUG
				std::cout << "\tmedoid " << medoids[asgn] << " have distance " << minDist << std::endl;
			#endif

			// Find nearest medoid...
			for (std::size_t m = 1; m < medoids.size(); ++m) {
				FLOAT dist = mDistFnc(objects[i], objects[medoids[m]]);
				#ifdef BPP_DEBUG
					std::cout << "\tmedoid " << medoids[m] << " have distance " << dist << std::endl;
				#endif

				if (dist < minDist) {
					minDist = dist;
					asgn = m;
					if (i == medoids[asgn]) break;	// Break if the object is the medoid.
				}
			}

			// Save the assignment.
			assignments[i] = asgn;
			if (distances != nullptr)
				distances[i] = minDist;

			#ifdef BPP_DEBUG
				std::cout << "\tselected medoid " << medoids[asgn] << std::endl;
			#endif
		}
	}


	/**
	 * \brief Compute the score of selected medoid within given cluster as a sum of distance squares.
	 * \param objects Input set of objects to be clustered.
	 * \param cluster List of object indices within the examined cluster.
	 * \param medoid Index to the cluster list of selected medoid.
	 * \return The score of the selected medoid.
	 */
	FLOAT computeMedoidScore(const OBJ objects[], const std::vector<std::size_t> &cluster, std::size_t medoid) const
	{
		FLOAT score = 0.0;
		for (std::size_t i = 0; i < cluster.size(); ++i) {
			if (i == medoid) continue;
			FLOAT dist = mDistFnc(objects[cluster[i]], objects[cluster[medoid]]);
			score += dist;// * dist;
		}
		#ifdef BPP_DEBUG
			std::cout << "Score of medoid " << medoid << " is " << score << std::endl;
		#endif
		return score;
	}


	/**
	 * \brief Find the best medoid for selected interval of objects in a cluster and return its object index.
	 * \param objects Input set of objects to be clustered.
	 * \param cluster List of object indices within the examined cluster.
	 * \param bestScore Output value where best distance sum of returned medoid is stored.
	 * \return Index to the objects vector of the best medoid found.
	 */
	std::size_t getBestMedoidPartial(const OBJ objects[], const std::vector<std::size_t> &cluster, FLOAT &bestScore,
		std::size_t fromIdx, std::size_t toIdx) const
	{
		// Find a medoid with smallest score.
		std::size_t medoid = fromIdx;
		bestScore = computeMedoidScore(objects, cluster, medoid);

		for (std::size_t i = fromIdx + 1; i < toIdx; ++i) {
			FLOAT score = computeMedoidScore(objects, cluster, i);
			if (score < bestScore) {
				bestScore = score;
				medoid = i;
			}
		}

		return medoid;
	}


	/**
	 * \brief Find the best medoid for selected cluster and return its object index.
	 * \param objects Input set of objects to be clustered.
	 * \param cluster List of object indices within the examined cluster.
	 * \param bestScore Output value where best distance sum of returned medoid is stored.
	 * \return Index to the objects vector of the best medoid found.
	 */
	std::size_t getBestMedoid(const OBJ objects[], const std::vector<std::size_t> &cluster, FLOAT &bestScore) const
	{
		// The cluster is never empty!
		if (cluster.empty())
			throw (RuntimeError() << "Unable to select the best medoid of an empty cluster.");

		// One-medoid show and zwei-medoid buntes are easy to solve.
		if (cluster.size() < 3)
			return cluster[0];

		// Find a medoid with smallest score.
		const std::size_t blockSize = 64;
		std::size_t clusterSize = cluster.size();
		if (clusterSize < blockSize) {
			std::size_t medoid = getBestMedoidPartial(objects, cluster, bestScore, 0, clusterSize);
			return cluster[medoid];
		}
		else {
			// Cluster is larger -> try OMP.
			std::size_t medoid = 0;
			FLOAT bestScoreCopy = computeMedoidScore(objects, cluster, medoid);

			#pragma omp parallel for shared(medoid, bestScoreCopy, objects, cluster) schedule(dynamic)
			for (std::size_t i = 1; i < clusterSize; i += blockSize) {
				FLOAT localBestScore;
				std::size_t localMedoid = getBestMedoidPartial(objects, cluster, localBestScore, i, std::min(i + blockSize, clusterSize));

				#pragma omp critical
				{
					// Since this is run by multiple tasks, the condition is more complex to ensure determinism.
					if (localBestScore < bestScoreCopy || (localBestScore == bestScoreCopy && localMedoid < medoid)) {
						bestScoreCopy = localBestScore;
						medoid = localMedoid;
					}
				}
			}
			bestScore = bestScoreCopy;
			return cluster[medoid];
		}
	}


	/**
	 * \brief Update the medoids according to the new assignments.
	 * \param objects Input set of objects to be clustered.
	 * \param medoids The medoid objects (as indices to the objects vector).
	 * \param assignments The assignment value for each object. The assignment vector
	 *		has the same size as objects vector and each value is an index to medoids vector.
	 * \return True if the medoids vector has been modified, false otherwise.
	 *		If the medoids have not been modified, the algorithm has reached a stable state.
	 */
	bool updateMedoids(const OBJ objects[], std::vector<std::size_t> &medoids, const std::vector<std::size_t> &assignments)
	{
		// Construct a cluster index (vector of clusters, each cluster is a vector of object indices).
		std::vector< std::vector<std::size_t> > clusters;
		clusters.resize(medoids.size());
		for (std::size_t i = 0; i < assignments.size(); ++i)
			clusters[assignments[i]].push_back(i);

		mLastAvgDistance = mLastAvgClusterDistance = (FLOAT)0.0;

		std::size_t clusterCount = clusters.size();
		std::vector<FLOAT> bestScores(clusterCount);
		std::vector<std::size_t> newMedoids(clusterCount);

		// Find the best medoid for each cluster in parallel.
		#pragma omp parallel for schedule(dynamic)
		for (std::size_t m = 0; m < clusterCount; ++m) {
			bestScores[m] = (FLOAT)0.0;
			newMedoids[m] = getBestMedoid(objects, clusters[m], bestScores[m]);
		}

		// Consolidate parallel results.
		bool changed = false;	// Whether the medoids vector was modified.
		for (std::size_t m = 0; m < clusters.size(); ++m) {
			mLastAvgDistance += bestScores[m];
			if (clusters[m].size() > 1)
				mLastAvgClusterDistance += bestScores[m] / (FLOAT)(clusters[m].size()-1);
			
			changed = changed || (newMedoids[m] != medoids[m]);
			medoids[m] = newMedoids[m];
		}

		mLastAvgDistance /= (FLOAT)(assignments.size() - clusters.size());	// all objects except medoids
		mLastAvgClusterDistance /= (FLOAT)clusters.size();					// all clusters

		// Report whether the medoids vector has been modified (if not -> early termination).
		return changed;
	}


	/**
	 * \brief Internal run method called by public run() interface.
	 * \param objects Input set of objects to be clustered.
	 * \param medoids The result medoid objects (as indices to the objects vector).
	 * \param assignments The result assignment value for each object. The assignment vector
	 *		has the same size as objects vector and each value is an index to medoids vector.
	 * \return Number of iterations performed.
	 */
	virtual std::size_t runPrivate(const OBJ objects[], std::size_t objectCount,
		std::vector<std::size_t> &medoids, std::vector<std::size_t> &assignments)
	{
		checkParams(objectCount);
		initRandMedoids(medoids, objectCount);
		medoids.resize(mK);

		std::size_t iter = 0;
		while (iter++ < mMaxIters) {
		#ifdef BPP_DEBUG
			std::cout << "Starting " << iter << " iteration..." << std::endl;
		#endif
			// Compute new assignments of objects to medoids.
			computeAssignments(objects, objectCount, medoids, assignments);

			// Update medoids (terminate if no update occured).
			if (!updateMedoids(objects, medoids, assignments)) break;
		}

		return iter;
	}


public:
	KMedoids(DIST &distFnc, std::size_t k, std::size_t maxIters)
		: mDistFnc(distFnc), mK(k), mMaxIters(maxIters) {}

	std::size_t getK() const					{ return mK; }
	void setK(std::size_t k)					{ mK = k; }

	std::size_t getMaxIters() const				{ return mMaxIters; }
	void setMaxIters(std::size_t maxIters)		{ mMaxIters = maxIters; }

	FLOAT getLastAvgDistance() const			{ return mLastAvgDistance; }
	FLOAT getLastAvgClusterDistance() const		{ return mLastAvgClusterDistance; }


	/**
	 * \brief Run the k-medoids clustering on given data.
	 * \param objects Input set of objects to be clustered.
	 * \param medoids The result medoid objects (as indices to the objects vector).
	 * \param assignments The result assignment value for each object. The assignment vector
	 *		has the same size as objects vector and each value is an index to medoids vector.
	 * \return Number of iterations performed.
	 */
	std::size_t run(const std::vector<OBJ> &objects, std::vector<std::size_t> &medoids, std::vector<std::size_t> &assignments)
	{
		// Just recall private virtual method...
		return run(&objects[0], objects.size(), medoids, assignments);
	}


	/**
	* \brief Run the k-medoids clustering on given data.
	* \param objects Pointer to a C array of objects.
	* \param objectCount Number of objects in the input array.
	* \param medoids The result medoid objects (as indices to the objects vector).
	* \param assignments The result assignment value for each object. The assignment vector
	*		has the same size as objects vector and each value is an index to medoids vector.
	* \return Number of iterations performed.
	*/
	std::size_t run(const OBJ objects[], std::size_t objectCount, std::vector<std::size_t> &medoids, std::vector<std::size_t> &assignments)
	{
#ifdef _OPENMP
		int ompNestedSave = omp_get_nested();
		if (!ompNestedSave) omp_set_nested(true);
#endif

		// Just recall private virtual method...
		auto res = this->runPrivate(objects, objectCount, medoids, assignments);

#ifdef _OPENMP
		if (!ompNestedSave) omp_set_nested(ompNestedSave);
#endif
		return res;
	}
};





/**
 * \brief A K-medoids algorithm with extended parameters.
 * \tparam OBJ Type of objects that are being clustered.
 * \tparam DIST Type of distance functor that computes/provides distances between objects.
 *		The functor must have operator() that takes two OBJs and yields a FLOAT.
 * \tparam FLOAT Float data type with selected precision (used for distances and medoid scores).
 *		It should be set to float or double.
 */
template<class OBJ, class DIST, typename FLOAT = float>
class KMedoidsExtended : public KMedoids<OBJ, DIST, FLOAT>
{
protected:
	std::size_t mInitK;		///< Initial k (must be greater or equal to k, since the number of clusters may only decrease).
	FLOAT mJoinDist;		///< A joining distance threshold.
	std::size_t mMinSize;	///< Minimal size of the cluster
	FLOAT mCutOfDist;		///< A cut off threshold. Object is cut off from its nearest medoid, if its further than this threshold.


	//virtual void checkParams(std::size_t objectsCount)
	//{
	//	KMedoids<OBJ, DIST, FLOAT>::checkParams(objectsCount);
	//}


	void joinClusters(const std::vector<OBJ> &objects, std::vector<std::size_t> &medoids, std::vector<std::size_t> &assignments)
	{
		for (std::size_t i = 0; i < medoids.size(); ++i) {

		}
	}
	
	
	void cutOffOutliers(const std::vector<OBJ> &objects, std::vector<std::size_t> &medoids,
		std::vector<std::size_t> &assignments, std::vector<FLOAT> &distances)
	{

	}


	void reduceClusters(const std::vector<OBJ> &objects, std::vector<std::size_t> &medoids, std::vector<std::size_t> &assignments)
	{
		// Remove small clusters.
		if (mMinSize > 1) {
			// Compute sizes.
			std::vector<std::size_t> sizes(medoids.size());
			for (std::size_t i = 0; i < assignments.size(); ++i)
				++medoids[assignments[i]];

			// Mark medoids of small clusters for deletion.
			for (std::size_t m = 0; m < sizes.size(); ++m)
				if (sizes[m] < mMinSize)
					medoids[m] = ~(std::size_t)0;
		}
	}


	virtual std::size_t runPrivate(const std::vector<OBJ> &objects, std::vector<std::size_t> &medoids, std::vector<std::size_t> &assignments)
	{
		throw (bpp::RuntimeError() << "Extended version of k-medoids is not implemented yet.");

		checkParams(objects.size());
		initRandMedoids(medoids, objects.size());
		medoids.resize(mInitK);

		std::vector<FLOAT> distances(objects.size());
		std::size_t iter = 0;
		while (iter++ < this->mMaxIters) {
		#ifdef BPP_DEBUG
			std::cout << "Starting " << iter << " iteration..." << std::endl;
		#endif

			// Compute new assignments of objects to medoids.
			computeAssignments(objects, medoids, assignments, &distances[0]);

			cutOffOutliers(objects, medoids, assignments, distances);

			// Update medoids (terminate if no update occured).
			if (!updateMedoids(objects, medoids, assignments) && medoids.size() == this->mK) break;

			reduceClusters(objects, medoids, assignments);
		}

		return iter;
	}

public:
	KMedoidsExtended(DIST &distFnc, std::size_t k, std::size_t initK, std::size_t maxIters, FLOAT joinDist = (FLOAT)0.0, std::size_t minSize = 0)
		: KMedoids<OBJ, DIST, FLOAT>(distFnc, k, maxIters), mInitK(initK), mJoinDist(joinDist), mMinSize(minSize) {}
};


}

#endif

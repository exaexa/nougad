/*
* Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
* Last Modification: 27.6.2019
* License: CC 3.0 BY-NC (http://creativecommons.org/)
*/
#ifndef BPPLIB_ALGO_LCS_HPP
#define BPPLIB_ALGO_LCS_HPP


#include <vector>
#include <algorithm>


namespace bpp
{
	namespace _priv
	{
		std::pair<std::size_t, std::size_t> computeWindow(std::size_t r, std::size_t rowSize, std::size_t maxWindowSize)
		{
			std::size_t fromI = 0, toI = rowSize;
			if (maxWindowSize > 0 && maxWindowSize <= toI) {
				fromI = r < (maxWindowSize / 2) ? 0 : r - (maxWindowSize / 2);
				toI = std::min(std::max(r + (maxWindowSize / 2) + 1, fromI + maxWindowSize), toI);
				if (toI == rowSize) {
					fromI = toI - maxWindowSize;
				}
			}
			return std::make_pair(fromI, toI);
		}


		/**
		 * Internal implementation of LCS algorithm, which founds only the length of the LCS itself.
		 * The algorithm has a tuning parameter maxWindowSize, which allows to reduce amount of computation
		 * in exchange for loosing precision (using approximative LCS only).
		 * The window size limits the width of each row being explored in LCS matrix.
		 * \tparam RES The result type (must be an integral type).
		 * \tparam CONTAINER Class holding a sequence. The class must have size() method
		 *         and the comparator must be able to get values from the container based on their indices.
		 * \tparam COMPARATOR Comparator class holds a static method compare(seq1, i1, seq2, i2) -> bool.
		 *         I.e., the comparator is also responsible for fetching values from the seq. containers.
		 */
		template<typename RES = std::size_t, class CONTAINER, typename COMPARATOR>
		RES longest_common_subsequence_length(const CONTAINER& sequence1, const CONTAINER& sequence2,
			COMPARATOR comparator, std::size_t maxWindowSize = 0)
		{
			if (sequence1.size() == 0 || sequence2.size() == 0) return (RES)0;

			// Make sure in seq1 is the longer sequence ...
			const CONTAINER& seq1 = sequence1.size() >= sequence2.size() ? sequence1 : sequence2;
			const CONTAINER& seq2 = sequence1.size() < sequence2.size() ? sequence1 : sequence2;

			std::vector<RES> row((std::size_t)seq2.size());
			std::size_t rows = (std::size_t)seq1.size();

			// Dynamic programming - matrix traversal that keeps only the last row.
			for (std::size_t r = 0; r < rows; ++r) {
				RES lastUpperLeft = 0, lastLeft = 0;
				auto window = computeWindow(r, row.size(), maxWindowSize);
				for (std::size_t i = window.first; i < window.second; ++i) {
					RES upper = row[i];
					row[i] =
						(comparator(seq1, r, seq2, i))
						? lastUpperLeft + 1
						: std::max(lastLeft, upper);
					lastLeft = row[i];
					lastUpperLeft = upper;
				}
			}

			return row.back();
		}


		/**
		 * Internal implementation of longest common subsequence algorithm which founds exactly one common subsequence.
		 * \tparam RES The result type (must be an integral type).
		 * \tparam CONTAINER Class holding the sequence. The class must have size() method
		 *         and the comparator must be able to get values from the container based on their indices.
		 * \tparam COMPARATOR Comparator class holds a static method compare(seq1, i1, seq2, i2) -> bool.
		 *         I.e., the comparator is also responsible for fetching values from the seq. containers.
		 */
		template<typename IDX = std::size_t, class CONTAINER, typename COMPARATOR>
		void longest_common_subsequence(const CONTAINER& sequence1, const CONTAINER& sequence2,
			std::vector<std::pair<IDX, IDX>>& common, COMPARATOR comparator, std::size_t maxWindowSize = 0)
		{
			struct Node {
				std::pair<IDX, IDX> previous;
				IDX length;
				bool match;
			public:
				Node() : length(0), match(false) {}
			};

			
			class NodeMatrix
			{
			private:
				std::size_t mSize1;
				std::vector<Node> mNodes;
			public:
				NodeMatrix(std::size_t size1, std::size_t size2) : mSize1(size1 + 1)
				{
					mNodes.resize((size1 + 1) * (size2 + 1));
					for (std::size_t i = 1; i <= size1; ++i) {
						at(i, 0).previous.first = 1;
					}
					for (std::size_t i = 1; i <= size2; ++i) {
						at(0, i).previous.second = 1;
					}
				}

				Node& at(std::size_t c, std::size_t r)
				{
					return mNodes[r * mSize1 + c];
				}
			};


			common.clear();
			if (sequence1.size() == 0 || sequence2.size() == 0) return;

			const std::size_t size1 = sequence1.size();
			const std::size_t size2 = sequence2.size();
			NodeMatrix matrix(size1, size2);

			// Fill in the LCS matrix by dynamic programming
			for (std::size_t r = 0; r < size2; ++r) { // iterate over rows
				auto window = computeWindow(r, size1, maxWindowSize);
				for (std::size_t c = window.first; c < window.second; ++c) { // iterate over cols
					bool match = matrix.at(c+1, r+1).match = comparator(sequence1, c, sequence2, r);

					if (match) {
						// Matching tokens should prolong the sequence...
						matrix.at(c + 1, r + 1).length = matrix.at(c, r).length + 1;
						matrix.at(c + 1, r + 1).previous.first = 1;
						matrix.at(c + 1, r + 1).previous.second = 1;
					}
					else {
						IDX leftLength = matrix.at(c, r + 1).length;
						IDX upperLength = matrix.at(c + 1, r).length;
						if (leftLength >= upperLength) {
							matrix.at(c + 1, r + 1).previous.first = 1;
							matrix.at(c + 1, r + 1).length = leftLength;
						}
						else {
							matrix.at(c + 1, r + 1).previous.second = 1;
							matrix.at(c + 1, r + 1).length = upperLength;
						}
					}
				}
			}

			// Collect the result path from the matrix...
			std::size_t c = size1;
			std::size_t r = size2;
			while (c > 0 && r > 0) {
				const Node& node = matrix.at(c, r);
				if (node.match) {
					common.push_back(std::make_pair<IDX, IDX>(c - 1, r - 1));
				}

				if (node.previous.first + node.previous.second > 0) {
					c -= node.previous.first;
					r -= node.previous.second;
				}
				else { // let's make sure we will not get stuck (if approx. version of LCS is running)
					if (c >= r) --c;
					if (c <= r) --r;
				}
			}

			// Fix the result (since it was collected backwards)...
			std::reverse(common.begin(), common.end());
		}
	}

	/**
	 * Implements a longest common subsequence algorithm, which founds only the length of the LCS itself.
	 * \tparam RES The result type (must be an integral type).
	 * \tparam CONTAINER Class holding a sequence. The class must have size() method
	 *         and the comparator must be able to get values from the container based on their indices.
	 * \tparam COMPARATOR Comparator class holds a static method compare(seq1, i1, seq2, i2) -> bool.
	 *         I.e., the comparator is also responsible for fetching values from the seq. containers.
	 */
	template<typename RES = std::size_t, class CONTAINER, typename COMPARATOR>
	RES longest_common_subsequence_length(const CONTAINER &sequence1, const CONTAINER &sequence2, COMPARATOR comparator)
	{
		return _priv::longest_common_subsequence_length<RES>(sequence1, sequence2, comparator);
	}


	// Only an overload that uses default comparator.
	template<typename RES = std::size_t, class CONTAINER>
	RES longest_common_subsequence_length(const CONTAINER &sequence1, const CONTAINER &sequence2)
	{
		return longest_common_subsequence_length<RES>(sequence1, sequence2,
			[](const CONTAINER &seq1, std::size_t i1, const CONTAINER &seq2, std::size_t i2) -> bool {
				return seq1[i1] == seq2[i2];
			}
		);
	}



	/**
	 * Implements a longest common subsequence algorithm which founds exactly one common subsequence.
	 * \tparam RES The result type (must be an integral type).
	 * \tparam CONTAINER Class holding the sequence. The class must have size() method
	 *         and the comparator must be able to get values from the container based on their indices.
	 * \tparam COMPARATOR Comparator class holds a static method compare(seq1, i1, seq2, i2) -> bool.
	 *         I.e., the comparator is also responsible for fetching values from the seq. containers.
	 */
	template<typename IDX = std::size_t, class CONTAINER, typename COMPARATOR>
	void longest_common_subsequence(const CONTAINER &sequence1, const CONTAINER &sequence2,
		std::vector<std::pair<IDX, IDX>> &common, COMPARATOR comparator)
	{
		return _priv::longest_common_subsequence<IDX>(sequence1, sequence2, common, comparator);
	}


	// Only an overload that uses default comparator.
	template<typename IDX = std::size_t, class CONTAINER>
	void longest_common_subsequence(const CONTAINER &sequence1, const CONTAINER &sequence2, std::vector<std::pair<IDX, IDX>> &common)
	{
		longest_common_subsequence<IDX>(sequence1, sequence2, common,
			[](const CONTAINER &seq1, std::size_t i1, const CONTAINER &seq2, std::size_t i2) -> bool {
			return seq1[i1] == seq2[i2];
		});
	}


	/*
	 * Approximate LCS
	 */

	/**
	 * Implements approximative version of LCS algorithm, which founds only the length of the LCS itself.
	 * The algorihm may not find the longest subsequence, so the result may be lower or equal to actual LCS,
	 * but it requires much less time as it does not explore the all possible pairings.
	 * \tparam RES The result type (must be an integral type).
	 * \tparam CONTAINER Class holding a sequence. The class must have size() method
	 *         and the comparator must be able to get values from the container based on their indices.
	 * \tparam COMPARATOR Comparator class holds a static method compare(seq1, i1, seq2, i2) -> bool.
	 *         I.e., the comparator is also responsible for fetching values from the seq. containers.
	 */
	template<typename RES = std::size_t, class CONTAINER, typename COMPARATOR>
	RES longest_common_subsequence_approx_length(const CONTAINER& sequence1, const CONTAINER& sequence2,
		COMPARATOR comparator, std::size_t maxWindowSize = 31)
	{
		return _priv::longest_common_subsequence_length<RES>(sequence1, sequence2, comparator, maxWindowSize);
	}


	// Only an overload that uses default comparator.
	template<typename RES = std::size_t, class CONTAINER>
	RES longest_common_subsequence_approx_length(const CONTAINER& sequence1, const CONTAINER& sequence2, std::size_t maxWindowSize = 31)
	{
		return longest_common_subsequence_approx_length<RES>(sequence1, sequence2,
			[](const CONTAINER& seq1, std::size_t i1, const CONTAINER& seq2, std::size_t i2) -> bool {
				return seq1[i1] == seq2[i2];
			},
			maxWindowSize
		);
	}

	
	/**
	 * Implements an approximative version of the longest common subsequence algorithm which founds exactly one common subsequence.
	 * The algorihm may not find the longest subsequence, so the result may be shorter to actual LCS,
	 * but it requires much less time as it does not explore the all possible pairings.
	 * \tparam RES The result type (must be an integral type).
	 * \tparam CONTAINER Class holding the sequence. The class must have size() method
	 *         and the comparator must be able to get values from the container based on their indices.
	 * \tparam COMPARATOR Comparator class holds a static method compare(seq1, i1, seq2, i2) -> bool.
	 *         I.e., the comparator is also responsible for fetching values from the seq. containers.
	 */
	template<typename IDX = std::size_t, class CONTAINER, typename COMPARATOR>
	void longest_common_subsequence_approx(const CONTAINER& sequence1, const CONTAINER& sequence2,
		std::vector<std::pair<IDX, IDX>>& common, COMPARATOR comparator, std::size_t maxWindowSize = 31)
	{
		return _priv::longest_common_subsequence<IDX>(sequence1, sequence2, common, comparator, maxWindowSize);
	}


	// Only an overload that uses default comparator.
	template<typename IDX = std::size_t, class CONTAINER>
	void longest_common_subsequence_approx(const CONTAINER& sequence1, const CONTAINER& sequence2,
		std::vector<std::pair<IDX, IDX>>& common, std::size_t maxWindowSize = 31)
	{
		longest_common_subsequence_approx<IDX>(sequence1, sequence2, common,
			[](const CONTAINER& seq1, std::size_t i1, const CONTAINER& seq2, std::size_t i2) -> bool {
				return seq1[i1] == seq2[i2];
			},
			maxWindowSize);
	}

}

#endif

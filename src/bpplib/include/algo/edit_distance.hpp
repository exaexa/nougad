/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 7.12.2015
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_ALGO_EDIT_DISTANCE_HPP
#define BPPLIB_ALGO_EDIT_DISTANCE_HPP


#include <vector>
#include <algorithm>


namespace bpp
{
	/**
	 * \brief Standard Levenshtein edit distance computed by serial
	 *		Wagner-Fisher algorithm. The standard version allows
	 *		insertions, deletions, and char substitution.
	 * \tparam C Type of the characters in the input strings/arrays.
	 * \tparam RES Type of the result (must be numerical type -- size_t, int, ...).
	 * \param s1 Pointer to the first string.
	 * \param len1 Length of the first string.
	 * \param s2 Pointer to the second string.
	 * \param len2 Length of the second string.
	 * \return Computed distance (0 if the strings are the same).
	 * \note The algorithm holds one row of the distance matrix in the memory.
	 */
	template<typename C, typename RES = std::size_t>
	RES edit_distance(const C *s1, std::size_t len1, const C *s2, std::size_t len2)
	{
		std::vector<RES> row(std::min<std::size_t>(len1, len2));
		std::size_t rows = std::max<std::size_t>(len1, len2);
		if (row.size() == 0) return (RES)rows;

		// Make sure s1 is the shorter string and s2 is the longer one.
		if (len1 > len2) {
			std::swap(s1, s2);
			std::swap(len1, len2);
		}

		// Initial fill of the row structure.
		for (std::size_t i = 0; i < row.size(); ++i)
			row[i] = (RES)i+1;

		for (std::size_t r = 0; r < rows; ++r) {
			RES lastUpper = r;
			RES lastLeft = lastUpper + 1;
			for (std::size_t i = 0; i < row.size(); ++i) {
				RES dist1 = std::min<RES>(row[i], lastLeft) + 1;
				RES dist2 = lastUpper + (s1[i] == s2[r] ? 0 : 1);
				lastUpper = row[i];
				lastLeft = row[i] = std::min<RES>(dist1, dist2);
			}
		}

		return row.back();
	}


	/**
	 * \brief Standard Levenshtein edit distance computed by serial
	 *		Wagner-Fisher algorithm. The standard version allows
	 *		insertions, deletions, and char substitution.
	 * \tparam C Type of the characters in the input strings/arrays.
	 * \tparam RES Type of the result (must be numerical type -- size_t, int, ...).
	 * \param str1 Vector holding the first string.
	 * \param str2 Vector holding the second string.
	 * \return Computed distance (0 if the strings are the same).
	 * \note The algorithm holds one row of the distance matrix in the memory.
	 */
	template<typename C, typename RES = std::size_t>
	RES edit_distance(const std::vector<C> &str1, const std::vector<C> &str2)
	{
		return edit_distance<C, RES>(&str1[0], str1.size(), &str2[0], str2.size());
	}



	/**
	 * \brief Extended Levenshtein edit distance computed by serial
	 *		Wagner-Fisher algorithm. This versions additionally allows
	 *		transpositions (along with insertions, deletions, and char substitution).
	 * \tparam C Type of the characters in the input strings/arrays.
	 * \tparam RES Type of the result (must be numerical type -- size_t, int, ...).
	 * \param s1 Pointer to the first string.
	 * \param len1 Length of the first string.
	 * \param s2 Pointer to the second string.
	 * \param len2 Length of the second string.
	 * \return Computed distance (0 if the strings are the same).
	 * \note The algorithm holds three rows of the distance matrix in the memory.
	 *		Could be implemented with two rows, but its tedious.
	 */
	template<typename C, typename RES = std::size_t>
	RES edit_distance_wt(const C *s1, std::size_t len1, const C *s2, std::size_t len2)
	{
		std::vector<RES> row(std::min<std::size_t>(len1, len2)+1);
		std::size_t rows = std::max<std::size_t>(len1, len2);
		if (row.size() == 1) return (RES)rows;
		std::vector<RES> prevRow(row.size());
		std::vector<RES> newRow(row.size());

		// Make sure s1 is the shorter string and s2 is the longer one.
		if (len1 > len2) {
			std::swap(s1, s2);
			std::swap(len1, len2);
		}

		// Initial fill of the row structure.
		for (std::size_t i = 0; i < row.size(); ++i)
			row[i] = (RES)i;

		for (std::size_t r = 0; r < rows; ++r) {
			newRow[0] = r + 1;
			for (std::size_t i = 1; i < row.size(); ++i) {
				if (s1[i-1] != s2[r]) {
					// String do not match
					RES dist = std::min<RES>({
						row[i],			// insertion
						newRow[i-1],	// deletion
						row[i-1],		// substitution
					});
					if (i > 1 && r > 0 && s1[i-2] == s2[r] && s1[i-1] == s2[r-1]) {
						// Transposition candidate
						dist = std::min<RES>(dist, prevRow[i-2]);
					}
					newRow[i] = dist + 1;
				}
				else	// String matches at this place
					newRow[i] = row[i-1];
			}
			prevRow.swap(row);	// row -> prevRow
			row.swap(newRow);	// newRow -> row (newRow will be rewritten)
		}

		return row.back();
	}


	/**
	 * \brief Extended Levenshtein edit distance computed by serial
	 *		Wagner-Fisher algorithm. This versions additionally allows
	 *		transpositions (along with insertions, deletions, and char substitution).
	 * \tparam C Type of the characters in the input strings/arrays.
	 * \tparam RES Type of the result (must be numerical type -- size_t, int, ...).
	 * \param str1 Vector holding the first string.
	 * \param str2 Vector holding the second string.
	 * \return Computed distance (0 if the strings are the same).
	 * \note The algorithm holds three rows of the distance matrix in the memory.
	 *		Could be implemented with two rows, but its tedious.
	 */
	template<typename C, typename RES = std::size_t>
	RES edit_distance_wt(const std::vector<C> &str1, const std::vector<C> &str2)
	{
		return edit_distance_wt<C, RES>(&str1[0], str1.size(), &str2[0], str2.size());
	}
};

#endif

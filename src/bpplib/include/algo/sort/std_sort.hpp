/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 1.7.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_ALGO_SORT_STD_SORT_HPP
#define BPPLIB_ALGO_SORT_STD_SORT_HPP

#include <algorithm>
#include <utility>


namespace bpp {


namespace _priv
{
	/*
	 * Predicate functions for std::sort.
	 */

	template<typename T>
	bool sort_predicate_first_asc(const T &item1, const T &item2)
	{
		return item1.first < item2.first;
	}


	template<typename T>
	bool sort_predicate_first_desc(const T &item1, const T &item2)
	{
		return item1.first > item2.first;
	}


	template<typename T>
	bool sort_predicate_second_asc(const T &item1, const T &item2)
	{
		return item1.second < item2.second;
	}


	template<typename T>
	bool sort_predicate_second_desc(const T &item1, const T &item2)
	{
		return item1.second > item2.second;
	}
}



/**
 * \brief Sorts vector of pairs by first of the pair in ascending order.
 */
template<typename T1, typename T2>
void sort_by_first(std::vector< std::pair<T1, T2> > &data)
{
	std::sort(data.begin(), data.end(), tools_priv::sort_predicate_first_asc< std::pair<T1,T2> >);
}


/**
 * \brief Sorts vector of pairs by first of the pair in descending order.
 */
template<typename T1, typename T2>
void sort_by_first_desc(std::vector< std::pair<T1, T2> > &data)
{
	std::sort(data.begin(), data.end(), tools_priv::sort_predicate_first_desc< std::pair<T1,T2> >);
}


/**
 * \brief Sorts vector of pairs by second of the pair in ascending order.
 */
template<typename T1, typename T2>
void sort_by_second(std::vector< std::pair<T1, T2> > &data)
{
	std::sort(data.begin(), data.end(), tools_priv::sort_predicate_second_asc< std::pair<T1,T2> >);
}


/**
 * \brief Sorts vector of pairs by second of the pair in descending order.
 */
template<typename T1, typename T2>
void sort_by_second_desc(std::vector< std::pair<T1, T2> > &data)
{
	std::sort(data.begin(), data.end(), tools_priv::sort_predicate_second_desc< std::pair<T1,T2> >);
}


}

#endif

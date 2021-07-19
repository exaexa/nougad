/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 1.7.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_DATASTRUCT_COMPARATOR_HPP
#define BPPLIB_DATASTRUCT_COMPARATOR_HPP


namespace bpp {

template <class T> class Comparator
{
public:
	static bool inOrder(const T &item1, const T &item2)
	{
		return item1 <= item2;
	}
};


//template <class T*> class Comparator
//{
//public:
//	static bool inOrder(const T item1, const T item2)
//	{
//		return (*item1) <= (*item2);
//	}
//};


template <class T> class ComparatorDesc
{
public:
	static bool inOrder(const T &item1, const T &item2)
	{
		return item1 >= item2;
	}
};


//template <class T*> class ComparatorDesc
//{
//public:
//	static bool inOrder(const T item1, const T item2)
//	{
//		return (*item1) >= (*item2);
//	}
//};


}

#endif

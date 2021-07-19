#include "../test.hpp"
#include <algo/edit_distance.hpp>

#include <iostream>



/**
 * \brief Test edit_distance and edit_distance_wt algorithms.
 */
class BPPAlgoEditDistanceTest : public BPPLibTest
{
private:
	bool testStrings(const std::string &str1, const std::string &str2, std::size_t expectDist) const
	{
		std::size_t dist = bpp::edit_distance(str1.c_str(), str1.length(), str2.c_str(), str2.length());
		if (dist != expectDist)
			std::cout << "edit_distance('" << str1 << "', '" << str2 << "') = " << dist
			<< ", but " << expectDist << " expected." << std::endl;
		return dist == expectDist;
	}


	bool testStringsTransp(const std::string &str1, const std::string &str2, std::size_t expectDist) const
	{
		std::size_t dist = bpp::edit_distance_wt(str1.c_str(), str1.length(), str2.c_str(), str2.length());
		if (dist != expectDist)
			std::cout << "edit_distance_wt('" << str1 << "', '" << str2 << "') = " << dist
			<< ", but " << expectDist << " expected." << std::endl;
		return dist == expectDist;
	}


public:
	BPPAlgoEditDistanceTest() : BPPLibTest("algo/edit_distance") {}

	virtual bool run() const
	{
		return testStrings("quite a long string", "quite a long string", 0)
			&& testStrings("kitten", "sitten", 1)
			&& testStrings("aaaaa", "bbbbb", 5)
			&& testStrings("onetwo", "one&two", 1)
			&& testStrings("one&twothree", "onetwo&three", 2)
			&& testStrings("abcdefg", "abcedfg", 2)
			&& testStringsTransp("quite a long string", "quite a long string", 0)
			&& testStringsTransp("kitten", "sitten", 1)
			&& testStringsTransp("aaaaa", "bbbbb", 5)
			&& testStringsTransp("onetwo", "one&two", 1)
			&& testStringsTransp("one&twothree", "onetwo&three", 2)
			&& testStringsTransp("abcdefg", "abcedfg", 1);
	}
};


BPPAlgoEditDistanceTest _algoEditDistanceTest;

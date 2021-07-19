#include "../test.hpp"
#include <algo/lcs.hpp>

#include <iostream>


/**
* \brief Test edit_distance and edit_distance_wt algorithms.
*/
class BPPAlgoLCSTest : public BPPLibTest
{
private:
	bool testLengthStrings(const std::string &str1, const std::string &str2, std::size_t expectDist) const
	{
		std::size_t dist = bpp::longest_common_subsequence_length(str1, str2);
		if (dist != expectDist)
			std::cout << "lcs_length('" << str1 << "', '" << str2 << "') = " << dist
			<< ", but " << expectDist << " was expected." << std::endl;
		return dist == expectDist;
	}


	bool testApproxLengthStrings(const std::string& str1, const std::string& str2, std::size_t maxWindowSize, std::size_t expectDist) const
	{
		std::size_t dist = bpp::longest_common_subsequence_approx_length(str1, str2, maxWindowSize);
		if (dist != expectDist)
			std::cout << "lcs_approx_length('" << str1 << "', '" << str2 << "') = " << dist
			<< ", but " << expectDist << " was expected." << std::endl;
		return dist == expectDist;
	}


	bool testStrings(const std::string &str1, const std::string &str2, const std::string &expected, std::size_t maxWindowSize = 0) const
	{
		std::vector<std::pair<std::size_t, std::size_t>> res;
		std::string lcsPreifx(maxWindowSize == 0 ? "lcs('" : "lcs_approx('");

		if (maxWindowSize == 0)
			bpp::longest_common_subsequence(str1, str2, res);
		else
			bpp::longest_common_subsequence_approx(str1, str2, res, maxWindowSize);

		if (res.size() != expected.length()) {
			std::cout << lcsPreifx << str1 << "', '" << str2 << "') yielded common subseq of length " << res.size() << ", but length "
				<< expected.length() << " was expected." << std::endl;
			return false;
		}

		std::string res1, res2;
		for (std::size_t i = 0; i < expected.length(); ++i) {
			res1 += str1[res[i].first];
			res2 += str2[res[i].second];
		}

		if (res1 != expected || res2 != expected) {
			std::cout << lcsPreifx << str1 << "', '" << str2 << "') yielded unexpected subseq ('" << res1 << "' or '" << res2 << "'), but '"
				<< expected << "' was expected." << std::endl;
			return false;
		}

		return true;
	}


public:
	BPPAlgoLCSTest() : BPPLibTest("algo/longest_common_subsequence") {}

	virtual bool run() const
	{
		return testLengthStrings("quite a long string", "quite a long string", 19)
			&& testLengthStrings("kittens", "sitten", 5)
			&& testLengthStrings("aaaaa", "bbbbb", 0)
			&& testLengthStrings("onetwo", "one&two", 6)
			&& testLengthStrings("abcdefgh", "ghijklm", 2)
			&& testLengthStrings("AGCAT", "GAC", 2)
			&& testLengthStrings("a1b2c3d4e", "a5b6c7d8e", 5)
			&& testLengthStrings("1234abcde", "abcde5678", 5)
			&& testApproxLengthStrings("kittens", "sitten", 3, 5)
			&& testApproxLengthStrings("a1b2c3d4e", "a5b6c7d8e", 3, 5)
			&& testApproxLengthStrings("1234abcde", "abcde5678", 3, 0)
			&& testApproxLengthStrings("1234abcde", "abcde5678", 9, 5)
			&& testStrings("quite a long string", "quite a long string", "quite a long string")
			&& testStrings("kittens", "sitten", "itten")
			&& testStrings("aaaaa", "bbbbb", "")
			&& testStrings("onetwo", "one&two", "onetwo")
			&& testStrings("abcdefgh", "ghijklm", "gh")
			&& testStrings("AGCAT", "GAC", "AC")
			&& testStrings("kittens", "sitten", "itten", 3)
			&& testStrings("a1b2c3d4e", "a5b6c7d8e", "abcde", 3)
			&& testStrings("1234abcde", "abcde5678", "", 3)
			&& testStrings("1234abcde", "abcde5678", "abcde", 9);
	}
};


BPPAlgoLCSTest _algoLCSTest;

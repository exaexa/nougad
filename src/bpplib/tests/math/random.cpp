#include "../test.hpp"
#include <math/random.hpp>

#include <map>
#include <algorithm>
#include <iostream>
#include <cstdint>



/**
 * \brief Tests the math/random.hpp features on 32-bit unsigned ints.
 */
class BPPMathRandomUint32Test : public BPPLibTest
{
public:
	BPPMathRandomUint32Test() : BPPLibTest("math/random/uint32") {}

	virtual bool run() const
	{
		std::cout << "Generating 1000 random numbers ..." << std::endl;
		
		std::map<std::uint32_t, size_t> freqs;
		for (size_t i = 0; i < 1000; ++i) {
			uint32_t x = bpp::Random<uint32_t>::next();
			++freqs[x];
		}

		size_t max = 0;
		for (std::map<std::uint32_t, size_t>::const_iterator it = freqs.begin(); it != freqs.end(); ++it) {
			max = std::max<size_t>(max, it->second);
			if (it->second > 1)
				std::cout << "Number " << it->first << " was generated " << it->second << "x times." << std::endl;
		}

		if (max >= 5) return false;

		for (size_t i = 0; i < 1000; ++i) {
			uint32_t x = bpp::Random<uint32_t>::next(42);
			if (x >= 42) {
				std::cout << "Random::next(42) returned " << x << std::endl; 
				return false;
			}

			x = bpp::Random<uint32_t>::next(42, 54);
			if (x < 42 || x >= 54) {
				std::cout << "Random::next(42, 54) returned " << x << std::endl; 
				return false;
			}
		}

		return true;
	}
};





/**
 * \brief Tests the math/random.hpp features on floats and doubles.
 */
class BPPMathRandomFloatTest : public BPPLibTest
{
private:
	template<typename T>
	bool test(T min, T max) const
	{
		T x = bpp::Random<T>::next(min, max);
		if (x < min || x > max) {
			std::cout << "Random::next(" << min << ", " << max << ") returned " << x << std::endl; 
			return false;
		}
		return true;
	}


public:
	BPPMathRandomFloatTest() : BPPLibTest("math/random/float") {}

	virtual bool run() const
	{
		for (size_t i = 0; i < 10000; ++i) {
			if (!test<float>(0.0f, 1.0f) ||
				!test<float>(-1.0f, 1.0f) ||
				!test<float>(42.0f, 1000000.0f) ||
				!test<double>(0.0f, 1.0f) ||
				!test<double>(-1.0f, 1.0f) ||
				!test<double>(42.0f, 1000000.0f))
				return false;
		}

		return true;
	}
};



BPPMathRandomUint32Test _mathRandomUint32Test;
BPPMathRandomFloatTest _mathRandomFloatTest;

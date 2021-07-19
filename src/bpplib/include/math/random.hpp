/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 2.7.2014
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 * This file is now obsolete if you are using C++11 or newer. Use native C++ random generators instead.
 */
#ifndef BPPLIB_MATH_RANDOM_HPP
#define BPPLIB_MATH_RANDOM_HPP

#include <misc/exception.hpp>

#include <vector>
#include <cstdint>
#include <cstdlib>


namespace bpp
{


namespace _priv {
	class RandomBase
	{
	protected:
		/**
		 * \brief Generate random uint32 number (used for internal generator).
		 * \param max Upper bound for generated num (exclusive). If zero, no restrictions are applied.
		 */
		static std::uint32_t random_u32(std::uint32_t max = 0)
		{
			std::uint32_t res = std::rand();
			res = (res << 9) ^ std::rand();
			res = (res << 9) ^ std::rand();
			return (max > 0) ? res % max : res;
		}
	
	public:
		static void seed(unsigned int seed)
		{
			std::srand(seed);
		}
	};
}



/**
 * \brief A class representing random generator. The template is specialized for most of the numeric types.
 *
 * Depending on the type, the class defines up to 3 versions of function next(), which returns another random value.
 * - next() returns random value from the range of its return type
 * - next(max) returns random value from 0 to max-1, or to maximal value of the type if max == 0
 * - next(min, max) returns random value from [min, max-1] interval on integral types and
 *		from <min, max> on float types.
 */
template<typename T>
class Random : public _priv::RandomBase
{
public:
	static T next()
	{
		throw bpp::RuntimeError("Random generator for selected type is not implemented yet.");
	}
};



template<>
class Random<std::uint8_t> : public _priv::RandomBase
{
public:
	static std::uint8_t next(std::uint8_t max = 0)
	{
		return static_cast<std::uint8_t>(random_u32(max) & 0x000000ff);
	}


	static std::uint8_t next(std::uint8_t min, std::uint8_t max)
	{
		if (max <= min)
			throw (RuntimeError() << "Invalid range constraints (" << min << ", " << max << ").");
		return next(max - min) + min;
	}
};



template<>
class Random<std::uint16_t> : public _priv::RandomBase
{
public:
	static std::uint16_t next(std::uint16_t max = 0)
	{
		return static_cast<std::uint16_t>(random_u32(max) & 0x0000ffff);
	}


	static std::uint16_t next(std::uint16_t min, std::uint16_t max)
	{
		if (max <= min)
			throw (RuntimeError() << "Invalid range constraints (" << min << ", " << max << ").");
		return next(max - min) + min;
	}
};



template<>
class Random<std::uint32_t> : public _priv::RandomBase
{
public:
	static std::uint32_t next(std::uint32_t max = 0)
	{
		return random_u32(max);
	}


	static std::uint32_t next(std::uint32_t min, std::uint32_t max)
	{
		if (max <= min)
			throw (RuntimeError() << "Invalid range constraints (" << min << ", " << max << ").");
		return next(max - min) + min;
	}


	/**
	 * \brief Special class that uses the size_t random generator to shuffle a vector.
	 */
	template<typename VT> 
	static void shuffle(std::vector<VT> &vec)
	{
		if (sizeof(std::uint32_t) != sizeof(typename std::vector<VT>::size_type))
			throw (RuntimeError() << "Invalid random generator used for shuffling.");

		if (vec.empty()) return;
		for (std::uint32_t i = 0; i < static_cast<std::uint32_t>(vec.size()-1); ++i) {
			std::uint32_t j = next(i, static_cast<std::uint32_t>(vec.size()));
			if (i != j) {
				VT tmp(vec[i]);
				vec[i] = vec[j];
				vec[j] = tmp;
			}
		}
	}
};



template<>
class Random<std::uint64_t> : public _priv::RandomBase
{
public:
	static std::uint64_t next(std::uint64_t max = 0)
	{
		std::uint64_t res = (static_cast<std::uint64_t>(random_u32()) << 32) ^ static_cast<std::uint64_t>(random_u32());
		return (max == 0) ? res : (res % max);
	}


	static std::uint64_t next(std::uint64_t min, std::uint64_t max)
	{
		if (max <= min)
			throw (RuntimeError() << "Invalid range constraints (" << min << ", " << max << ").");
		return next(max - min) + min;
	}


	/**
	 * \brief Special class that uses the size_t random generator to shuffle a vector.
	 */
	template<typename VT> 
	static void shuffle(std::vector<VT> &vec)
	{
		if (sizeof(std::uint64_t) != sizeof(typename std::vector<VT>::size_type))
			throw (RuntimeError() << "Invalid random generator used for shuffling.");

		if (vec.empty()) return;
		for (std::uint64_t i = 0; i < static_cast<std::uint64_t>(vec.size()-1); ++i) {
			std::uint64_t j = next(i, static_cast<std::uint64_t>(vec.size()));
			if (i != j) {
				VT tmp(vec[i]);
				vec[i] = vec[j];
				vec[j] = tmp;
			}
		}
	}
};



template<>
class Random<double> : public _priv::RandomBase
{
public:
	static double next(double min = 0.0, double max = 1.0)
	{
		if (max <= min)
			throw (RuntimeError() << "Invalid range constraints (" << min << ", " << max << ").");

		double res = static_cast<double>(Random<std::uint64_t>::next());
		res /= (1.0 + 0xffffffffffffffffL);
		return res * (max-min) + min;
	}
};



template<>
class Random<float> : public _priv::RandomBase
{
public:
	static float next(float min = 0.0f, float max = 1.0f)
	{
		return static_cast<float>(Random<double>::next(min, max));
	}
};

}

#endif

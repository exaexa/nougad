/*
* Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
* Last Modification: 23.12.2015
* License: CC 3.0 BY-NC (http://creativecommons.org/)
*/
#ifndef BPPLIB_MISC_PTR_FIX_HPP
#define BPPLIB_MISC_PTR_FIX_HPP

#include <memory>


// Define pointer restrict keyword for various compilers.
#if defined(__ICC) || defined(__ICL)
	// Intel compiler
	#define RESTRICT __restrict__
#elif defined(_MSC_VER)
	// MSVC
	#define RESTRICT __restrict
#elif defined(__GNUG__)
	// GNU Linux (g++)
	#define RESTRICT __restrict__
#else
	// unknown compiler (define as empty)
	#define RESTRICT
#endif



namespace bpp
{
	// Replacement for std::make_unique for C++ < 14
	template<typename T, typename... Args>
	std::unique_ptr<T> make_unique(Args&&... args)
	{
		return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
	}


	// Replacement for std::make_shared for C++ < 14
	template<typename T, typename... Args>
	std::shared_ptr<T> make_shared(Args&&... args)
	{
		return std::shared_ptr<T>(new T(std::forward<Args>(args)...));
	}
}


#endif

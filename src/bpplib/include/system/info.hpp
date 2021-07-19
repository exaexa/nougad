/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 1.7.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_SYSTEM_INFO_HPP
#define BPPLIB_SYSTEM_INFO_HPP


#ifdef _WIN32
#define NOMINMAX
	#include <windows.h>

	// This macro is defined in wingdi.h, I do non want ERROR macro in my projects!
	#ifdef ERROR
	#undef ERROR
	#endif
#else
	#include <unistd.h>
#endif


namespace bpp {

/**
 * \brief Aggregates methods and other means to get basic system information.
 */
class SysInfo
{
public:
	/**
	 * \brief Returns total amount of system memory (in Bytes).
	 */
	static size_t getTotalMemory()
	{
	#ifdef _WIN32
		MEMORYSTATUSEX status;
		status.dwLength = sizeof(status);
		GlobalMemoryStatusEx(&status);
		return static_cast<size_t>(status.ullTotalPhys);
	#else
		long pages = sysconf(_SC_PHYS_PAGES);
		long page_size = sysconf(_SC_PAGE_SIZE);
		return static_cast<size_t>(pages * page_size);
	#endif
	}
};


}

#endif

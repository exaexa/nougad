/*
* Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
* Last Modification: 26.1.2015
* License: CC 3.0 BY-NC (http://creativecommons.org/)
*/
#ifndef BPPLIB_SYSTEM_HACKS_HPP
#define BPPLIB_SYSTEM_HACKS_HPP

#include <thread>
#include <vector>
#include <cstdint>


namespace bpp
{

	namespace _priv
	{
		// Internal routine performed by each trashing thread.
		void trash_cpu_cache_thread(const std::vector<std::uint32_t> &data)
		{
			std::uint64_t sum = 0;
			for (size_t i = 0; i < data.size(); ++i)
				sum += (uint64_t)data[i];
		}
	}


	/**
	 * \brief Makes each thread read a large block of memory, so it is very likely
	 *		that all the data in the CPU caches will be trashed.
	 * \param expectedL3Size Size of the memory read by each thread (should be more than L3 size).
	 * \param threadCount Number of threads performing the trashing (at least as many as there are cores).
	 */
	void trash_cpu_caches(std::size_t expectedL3Size = 32*1024*1024, std::size_t threadCount = 32)
	{
		// Prepare trashing data ...
		std::vector<std::uint32_t> data(expectedL3Size/sizeof(std::uint32_t));
		for (size_t i = 0; i < data.size(); ++i)
			data[i] = (std::uint32_t)i;

		// Start a pool of threads that will be trashing the caches.
		std::vector<std::thread> threads;
		threads.reserve(threadCount);
		for (size_t i = 0; i < threadCount; ++i)
			threads.push_back(std::thread(_priv::trash_cpu_cache_thread, std::ref(data)));
		
		// Wait for all threads to finish.
		for (auto &th : threads)
			th.join();
	}
}

#endif

#include "../test.hpp"
#include <para/blocking_queue.hpp>

#include <thread>
#include <atomic>
#include <algorithm>
#include <iostream>
#include <random>

//#define BPP_BLOCKING_QUEUE_DEBUG
//#define BPP_BLOCKING_QUEUE_DEBUG_VERBOSE



/**
 * \brief Thest that concurrently inserts/removes items in blocking queue using blocking push/pop operations.
 */
class BPPParaBlockingQueueBlockingTest : public BPPLibTest
{
private:
	// Push sequence of ints <from, from+count) into the queue.
	static void threadWriter(bpp::BlockingQueue<int> &queue, int from, int count)
	{
		for (int i = from; i < from+count; ++i) {
#ifdef BPP_BLOCKING_QUEUE_DEBUG_VERBOSE
			std::cout << "Writing " << i << std::endl;
#endif
			queue.push(i);
		}
#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Terminating writer " << from << std::endl;
#endif
	}


	// Push sequence of ints <from, from+count) into the queue.
	static void threadReader(bpp::BlockingQueue<int> &queue, std::vector<int> &data, std::size_t offset, std::size_t count)
	{
		while (count > 0) {
			data[offset] = queue.pop();
#ifdef BPP_BLOCKING_QUEUE_DEBUG_VERBOSE
			std::cout << "Read " << data[offset] << std::endl;
#endif
			++offset; --count;
		}
#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Terminating reader " << offset - count << std::endl;
#endif
	}


public:
	BPPParaBlockingQueueBlockingTest() : BPPLibTest("para/blocking_queue/blocking") {}

	virtual bool run() const
	{
		const std::size_t threadCount = 16;
		const std::size_t intsPerThread = 4096;
		std::vector<int> data(threadCount * intsPerThread);
		bpp::BlockingQueue<int> queue(200);

		std::vector<std::thread> threads;
		threads.reserve(threadCount*2);
		
		// Start writers.
		std::cout << "Starting writers ..." << std::endl;
		for (std::size_t i = 0; i < threadCount; ++i)
			threads.push_back(std::thread(&threadWriter, std::ref(queue), (int)(i*intsPerThread), (int)intsPerThread));

		// Actively wait until the queue is filled.
		std::cout << "Waiting until the queue is full ..." << std::endl;
		while (!queue.isFull())
			std::this_thread::yield();

		// Start readers.
		std::cout << "Starting readers ..." << std::endl;
		for (std::size_t i = 0; i < threadCount; ++i)
			threads.push_back(std::thread(&threadReader, std::ref(queue), std::ref(data), i*intsPerThread, intsPerThread));

		// Wait for all threads to finish.
		std::cout << "Joining all threads ..." << std::endl;
		for (auto &th : threads)
			th.join();

		// Check the data.
		std::size_t errors = 0;
		std::sort(data.begin(), data.end());
		for (std::size_t i = 0; i < data.size(); ++i) {
			if (data[i] != (int)i) ++errors;
		}

		if (errors > 0) {
			std::cout << "Total " << errors << " errors were found in data ordering." << std::endl;
			return false;
		}

		return true;
	}
};


BPPParaBlockingQueueBlockingTest _paraBlockingQueueBlockingTest;





/**
* \brief Thest that concurrently inserts/removes items in blocking queue using non-blocking (try) push/pop operations.
*/
class BPPParaBlockingQueueNonblockingTest : public BPPLibTest
{
private:
	// Push sequence of ints <from, from+count) into the queue.
	static void threadWriter(bpp::BlockingQueue<int> &queue, int from, int count)
	{
		for (int i = from; i < from+count; ++i) {
#ifdef BPP_BLOCKING_QUEUE_DEBUG_VERBOSE
			std::cout << "Writing " << i << std::endl;
#endif
			while (!queue.tryPush(i))
				std::this_thread::yield();
		}
#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Terminating writer " << from << std::endl;
#endif
	}


	// Push sequence of ints <from, from+count) into the queue.
	static void threadReader(bpp::BlockingQueue<int> &queue, std::vector<int> &data, std::size_t offset, std::size_t count)
	{
		while (count > 0) {
			while (!queue.tryPop(data[offset]))
				std::this_thread::yield();
#ifdef BPP_BLOCKING_QUEUE_DEBUG_VERBOSE
			std::cout << "Read " << data[offset] << std::endl;
#endif
			++offset; --count;
		}
#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Terminating reader " << offset - count << std::endl;
#endif
	}


public:
	BPPParaBlockingQueueNonblockingTest() : BPPLibTest("para/blocking_queue/non-blocking") {}

	virtual bool run() const
	{
		const std::size_t threadCount = 16;
		const std::size_t intsPerThread = 4096;
		std::vector<int> data(threadCount * intsPerThread);
		bpp::BlockingQueue<int> queue(200);

		std::vector<std::thread> threads;
		threads.reserve(threadCount*2);

		// Start writers.
		std::cout << "Starting writers ..." << std::endl;
		for (std::size_t i = 0; i < threadCount; ++i)
			threads.push_back(std::thread(&threadWriter, std::ref(queue), (int)(i*intsPerThread), (int)intsPerThread));

		// Actively wait until the queue is filled.
		std::cout << "Waiting until the queue is full ..." << std::endl;
		while (!queue.isFull())
			std::this_thread::yield();

		// Start readers.
		std::cout << "Starting readers ..." << std::endl;
		for (std::size_t i = 0; i < threadCount; ++i)
			threads.push_back(std::thread(&threadReader, std::ref(queue), std::ref(data), i*intsPerThread, intsPerThread));

		// Wait for all threads to finish.
		std::cout << "Joining all threads ..." << std::endl;
		for (auto &th : threads)
			th.join();

		// Check the data.
		std::size_t errors = 0;
		std::sort(data.begin(), data.end());
		for (std::size_t i = 0; i < data.size(); ++i) {
			if (data[i] != (int)i) ++errors;
		}

		if (errors > 0) {
			std::cout << "Total " << errors << " errors were found in data ordering." << std::endl;
			return false;
		}

		return true;
	}
};


BPPParaBlockingQueueNonblockingTest _paraBlockingQueueNonblockingTest;





/**
 * \brief A pipeline test which uses three blocking queues between four sets of threads.
 *		First stage generates random numbers, second and third stage modifies them,
 *		and last stage performs a single-threaded reduction.
 */
class BPPParaBlockingQueuePipelineTest : public BPPLibTest
{
private:
	// Generate random numbers and push them in queue.
	static void generator(bpp::BlockingQueue<std::size_t> &outQueue, std::size_t count, std::size_t id, bool blocking)
	{
		std::default_random_engine eng;
		std::uniform_int_distribution<std::size_t> dist(1, std::numeric_limits<std::size_t>::max());

#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Starting generator #" << id << (blocking ? " blocking" : " nonblocking") << std::endl;
#endif
		for (std::size_t i = 0; i < count; ++i) {
			std::size_t x = dist(eng);
			if (blocking)
				outQueue.push(x);
			else {
				while (!outQueue.tryPush(x))
					std::this_thread::yield();
			}
		}

#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Terminating generator #" << id << " after " << count << " writes" << std::endl;
#endif
	}


	// Read from a queue, modify numbers, and push them in outgoing queue.
	static void modifier(bpp::BlockingQueue<std::size_t> &inQueue, bpp::BlockingQueue<std::size_t> &outQueue, std::size_t id)
	{
		std::default_random_engine eng;
		std::uniform_int_distribution<std::size_t> dist(1, std::numeric_limits<std::size_t>::max());

#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Starting modifier #" << id << std::endl;
#endif
		while (true) {
			std::size_t x = inQueue.pop();
			if (x == 0) {
#ifdef BPP_BLOCKING_QUEUE_DEBUG
				std::cout << "Modifier #" << id << " received a poison pill ..." << std::endl;
#endif
				// Process poison pill.
				inQueue.push(0);
				break;
			}

			// Modify ...
			std::size_t count = (dist(eng) % 128) + 128;
			for (std::size_t i = 0; (i < count) || (x == 0); ++i)
				x ^= dist(eng);

			outQueue.push(x);
		}

#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Terminating modifier #" << id << std::endl;
#endif
	}


	// Final reduction ...
	static void reducer(bpp::BlockingQueue<std::size_t> &inQueue, std::size_t &out, std::size_t id)
	{
#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Starting reducer #" << id << std::endl;
#endif
		std::size_t reduction = 0;
		std::size_t count = 0;
		while (true) {
			std::size_t x = inQueue.pop();
			if (x == 0) {
#ifdef BPP_BLOCKING_QUEUE_DEBUG
				std::cout << "Reducer #" << id << " received a poison pill ..." << std::endl;
#endif
				// Process poison pill.
				inQueue.push(0);
				break;
			}

			reduction ^= x;
			++count;
		}
		out ^= count;

#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Terminating reducer #" << id << std::endl;
#endif
	}


public:
	BPPParaBlockingQueuePipelineTest() : BPPLibTest("para/blocking_queue/pipeline") {}

	virtual bool run() const
	{
		const std::size_t modifiersCount = std::max<std::size_t>(1, std::thread::hardware_concurrency()/2);
		const std::size_t items = 64*1024;
		bpp::BlockingQueue<std::size_t> queue1(modifiersCount*2);
		bpp::BlockingQueue<std::size_t> queue2(modifiersCount*2);
		bpp::BlockingQueue<std::size_t> queue3(modifiersCount*2);
		std::size_t out = 0;

		std::vector<std::thread> generators;
		generators.reserve(2);
		std::vector<std::thread> modifiers1;
		modifiers1.reserve(modifiersCount);
		std::vector<std::thread> modifiers2;
		modifiers2.reserve(modifiersCount);
		std::vector<std::thread> reducers;
		reducers.reserve(1);

		std::cout << "Starting threads ..." << std::endl;
		for (std::size_t i = 0; i < modifiersCount; ++i)
			modifiers1.push_back(std::thread(&modifier, std::ref(queue1), std::ref(queue2), i));
		for (std::size_t i = 0; i < modifiersCount; ++i)
			modifiers2.push_back(std::thread(&modifier, std::ref(queue2), std::ref(queue3), i+modifiersCount));
		reducers.push_back(std::thread(&reducer, std::ref(queue3), std::ref(out), 1));
		generators.push_back(std::thread(&generator, std::ref(queue1), items, 1, true));
		generators.push_back(std::thread(&generator, std::ref(queue1), items, 2, false));

		// Wait for all threads to finish.
		std::cout << "Joining generators ..." << std::endl;
		for (auto &th : generators)
			th.join();

#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Sending poison pill to queue #1 ..." << std::endl;
#endif
		queue1.push(0);

		std::cout << "Joining modifiers 1 ..." << std::endl;
		for (auto &th : modifiers1)
			th.join();

#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Sending poison pill to queue #2 ..." << std::endl;
#endif
		queue2.push(0);

		std::cout << "Joining modifiers 2 ..." << std::endl;
		for (auto &th : modifiers2)
			th.join();

#ifdef BPP_BLOCKING_QUEUE_DEBUG
		std::cout << "Sending poison pill to queue #3 ..." << std::endl;
#endif
		queue3.push(0);

		std::cout << "Joining reducers ..." << std::endl;
		for (auto &th : reducers)
			th.join();

		// Check the data.
		if (out != items * 2) {
			std::cout << "Wrong number of items was reduced (" << items*2 << "expected, but " << out << " counted)." << std::endl;
			return false;
		}

		return true;
	}
};


BPPParaBlockingQueuePipelineTest _paraBlockingQueuePipelineTest;


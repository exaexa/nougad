#ifdef USE_TBB

#include "../test.hpp"
#include <system/file_async.hpp>
#include <system/file.hpp>
#include <math/random.hpp>

#include <vector>
#include <iostream>


/**
 * \brief Tests the system/info.hpp feature acquiring the amount of total system memory.
 */
class BPPSystemFileAsyncDoubleBufferTest : public BPPLibTest
{
private:
	template<typename T>
	bool compareBlocks(const T *buf1, const T *buf2, size_t count) const
	{
		for (size_t i = 0; i < count; ++i) {
			if (buf1[i] != buf2[i]) {
				std::cout << "Data discrepancy found." << std::endl;
				return false;
			}
		}
		return true;
	}


public:
	BPPSystemFileAsyncDoubleBufferTest() : BPPLibTest("system/file_async/doublebuffer") {}

	virtual bool run() const
	{
		// Prepare some data.
		size_t blockCount = 10;
		size_t blockSize = 1024*1024;
		std::vector<size_t> data(blockSize * blockCount);
		
		// Generate first block.
		for (size_t i = 0; i < blockSize; ++i)
			data[i] = bpp::Random<size_t>::next();

		try {
			bpp::FileAsyncOps asyncOps(4);

			// Create file ...
			bpp::File file(".tmp.file_async.test");
			file.open("wb");

			// Simultaneously gerenate and write data.
			for (size_t block = 1; block < blockCount; ++block) {
				// Start the async write ...
				bpp::FileAsyncOps::Promise promise = asyncOps.write<size_t>(file, &data[(block-1)*blockSize], blockSize, true);

				// ... and meanwhile, generate another block.
				for (size_t i = 0; i < blockSize; ++i)
					data[block*blockSize + i] = bpp::Random<size_t>::next();

				promise.waitFor();
			}

			// Write last block and close.
			file.write<size_t>(&data[(blockCount-1)*blockSize], blockSize);
			file.close();

			// Check file size on the disk.
			size_t fileSize = file.size();
			if (fileSize != data.size() * sizeof(size_t)) {
				std::cout << "File size does not correspond to the size of the data written." << std::endl;
				return false;
			}

			// Verify data ...
			file.open("rb");
			std::vector<size_t> tmp1(blockSize), tmp2(blockSize);
			std::vector<size_t> *reading = &tmp1;
			std::vector<size_t> *verifying = &tmp2;

			// Overlap reading and verification ..
			file.read<size_t>(&(*verifying)[0], blockSize);
			for (size_t block = 0; block < blockCount-1; ++block) {
				bpp::FileAsyncOps::Promise promise = asyncOps.read<size_t>(file, &(*reading)[0], blockSize);
				bool res = compareBlocks<size_t>(&(data[block*blockSize]), &(*verifying)[0], blockSize);
				promise.waitFor();
				if (!res)
					return false;
				std::swap(reading, verifying);
			}

			// Verify final block.
			if (!compareBlocks<size_t>(&(data[(blockCount-1)*blockSize]), &(*verifying)[0], blockSize))
				return false;

			// Close and delete.
			file.close();
			file.unlink();
			return true;
		}
		catch (bpp::FileError &e) {
			std::cout << "File Error Occured: " << e.what() << std::endl;
			return false;
		}
	}
};


BPPSystemFileAsyncDoubleBufferTest _systemFileAsyncDoubleBufferTest;

#endif

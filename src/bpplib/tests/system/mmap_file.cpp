#define _CRT_SECURE_NO_WARNINGS

#include "../test.hpp"
#include <system/mmap_file.hpp>
#include <system/file.hpp>
#include <math/random.hpp>

#include <iostream>


/**
 * \brief Tests the system/mmap_file.hpp by creating a regular file and mapping it into memory.
 */
class BPPSystemMMapFileTest : public BPPLibTest
{
public:
	BPPSystemMMapFileTest() : BPPLibTest("system/mmap_file/basic") {}

	virtual bool run() const
	{
		// Generate data.
		std::vector<size_t> data(425419);
		for (size_t i = 0; i < data.size(); ++i)
			data[i] = bpp::Random<size_t>::next();

		// Create file ...
		const char *fileName = ".tmp.mmap_file.test";
		bpp::File file(fileName);
		file.open("wb");

		// Write the data and close.
		file.write<size_t>(data);
		file.flush();
		file.close();

		try {
			// Map the file into memory.
			bpp::MMapFile mapFile;
			mapFile.open(fileName);
			if (mapFile.opened() != true) {
				std::cout << "The open() was called yet the file does not indicate it is opened." << std::endl;
				return false;
			}

			// Populate pages and get the file beginning.
			mapFile.populate();
			size_t *mapped = static_cast<size_t*>(mapFile.getData());

			// Validate data.
			for (size_t i = 0; i < data.size(); ++i) {
				if (data[i] != mapped[i]) {
					std::cout << "Data discrepancy at position " << i << std::endl;
					return false;
				}
			}

			// Terminate.
			mapFile.close();
			file.unlink();
			return true;
		}
		catch (bpp::RuntimeError &e) {
			std::cout << "Runtime Error: " << e.what() << std::endl;
			return false;
		}
	}
};


BPPSystemMMapFileTest _systemMMapFileTest;

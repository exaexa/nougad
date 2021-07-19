#define _CRT_SECURE_NO_WARNINGS

#include "../test.hpp"
#include <system/file.hpp>
#include <math/random.hpp>

#include <vector>
#include <iostream>


/**
 * \brief Tests the system/file.hpp basic features (I/O, determine size, unlink).
 */
class BPPSystemFileBasicTest : public BPPLibTest
{
public:
	BPPSystemFileBasicTest() : BPPLibTest("system/file/basic") {}

	virtual bool run() const
	{
		// Generate data.
		std::vector<size_t> data(4200);
		for (size_t i = 0; i < data.size(); ++i)
			data[i] = bpp::Random<size_t>::next();

		try {
			// Create file ...
			bpp::File file(".tmp.file.test");
			file.open("wb");
			if (!file.opened()) {
				std::cout << "The file is expected to be opened after successful open() call." << std::endl;
				return false;
			}

			// Write the data and close.
			file.write<size_t>(data);
			file.flush();
			file.close();
			if (file.opened()) {
				std::cout << "The file is expected to be closed after successful close() call." << std::endl;
				return false;
			}

			// Check file size on the disk.
			size_t fileSize = file.size();
			if (fileSize != data.size() * sizeof(size_t)) {
				std::cout << "File size does not correspond to the size of the data written." << std::endl;
				return false;
			}

			// Verify data.
			file.open("rb");
			std::vector<size_t> tmp;
			while (file.tell() < fileSize) {
				size_t x;
				file.read<size_t>(&x);
				tmp.push_back(x);
			}

			if (tmp.size() != data.size()) {
				std::cout << "We have read " << tmp.size() << " items from the file but " << data.size() << " items were previously written." << std::endl;
				return false;
			}

			// Test seek and tell.
			file.seek(42);
			size_t pos = file.tell();
			if (pos != 42) {
				std::cout << "File tells position " << pos << " but 42 was previously set." << std::endl;
				return false;
			}

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


BPPSystemFileBasicTest _systemFileBasicTest;

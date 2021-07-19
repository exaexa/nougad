#define _CRT_SECURE_NO_WARNINGS

#include "../test.hpp"
#include <system/filesystem.hpp>

#include <iostream>
#include <cstdio>


/**
 * \brief Tests the system/filesystem.hpp string path operations.
 */
class BPPSystemFilesystemPathTest : public BPPLibTest
{
	bool cropExtensionTest(const std::string &input, const std::string &correct) const
	{
		std::string output = bpp::Path::cropExtension(input);
		if (output != correct) {
			std::cout << "Path::cropExtension(\"" << input << "\") yielded \"" << output << "\", but \"" << correct << "\" was expected." << std::endl;
			return false;
		}
		return true;
	}


	bool getFileNameTest(const std::string &input, const std::string &correct) const
	{
		std::string output = bpp::Path::getFileName(input);
		if (output != correct) {
			std::cout << "Path::getFileName(\"" << input << "\") yielded \"" << output << "\", but \"" << correct << "\" was expected." << std::endl;
			return false;
		}
		return true;
	}


public:
	BPPSystemFilesystemPathTest() : BPPLibTest("system/filesystem/path") {}

	virtual bool run() const
	{
		return cropExtensionTest("filename.ext", "filename")
			&& cropExtensionTest("/dir/subdir/filename.ext", "/dir/subdir/filename")
			&& cropExtensionTest("./filename.ext", "./filename")
			&& cropExtensionTest("file.with.multiple.exts", "file.with.multiple")
			&& cropExtensionTest("../filename", "../filename")
			&& cropExtensionTest(".", ".")
			&& cropExtensionTest("..", "..")
			&& cropExtensionTest("/some/path/.", "/some/path/.")
			&& cropExtensionTest("/some/path/..", "/some/path/..")
			&& cropExtensionTest("C:\\Windows\\system.dll", "C:\\Windows\\system")
			&& cropExtensionTest("D:\\Special.Dir\\", "D:\\Special.Dir\\")
			&& getFileNameTest("filename.ext", "filename.ext")
			&& getFileNameTest("./filename.ext", "filename.ext")
			&& getFileNameTest("/path/filename.ext", "filename.ext")
			&& getFileNameTest("/dir/subdir/filename.ext", "filename.ext")
			&& getFileNameTest("/dir/subdir/", "")
			&& getFileNameTest("C:\\Dir\\File", "File")
			&& getFileNameTest(".\\File", "File")
			&& getFileNameTest("C:\\Combined/Path\\Separators/File", "File");
	}
};





/**
 * \brief Tests the system/filesystem.hpp file operations.
 */
class BPPSystemFilesystemFileTest : public BPPLibTest
{
public:
	BPPSystemFilesystemFileTest() : BPPLibTest("system/filesystem/file") {}

	virtual bool run() const
	{
		// Create a testing file.
		const char *fileName = ".tmp.filesystem.test";
		std::FILE *fp = fopen(fileName, "wb");
		if (fp == nullptr)
			throw (bpp::RuntimeError() << "Testing file '" << fileName << "' cannot be created.");

		fwrite("abcdef", sizeof(char), 6, fp);
		fclose(fp);

		// Verify the file exists.
		if (!bpp::Path::exists(fileName)) {
			std::cout << "File '" << fileName << "' does not exist even though it has been created." << std::endl;
			return false;
		}
		if (!bpp::Path::isRegularFile(fileName)) {
			std::cout << "File '" << fileName << "' is not a regulare file even though it has been created as such." << std::endl;
			return false;
		}

		// Check that current directory is not a regular file.
		if (bpp::Path::isRegularFile(".")) {
			std::cout << "Current directory was detected as regular file." << std::endl;
			return false;
		}

		// Unlink the file and verify it has been removed.
		bpp::Path::unlink(fileName);
		if (bpp::Path::exists(fileName)) {
			std::cout << "File '" << fileName << "' still exist even though it has been removed." << std::endl;
			return false;
		}
		try {
			bpp::Path::isRegularFile(fileName);
			std::cout << "Testing nonexisting path '" << fileName << "' for a regular file should have failed." << std::endl;
			return false;
		}
		catch (bpp::RuntimeError&) {
			// Nothing to be done here, the exception was expected.
		}

		return true;
	}
};


BPPSystemFilesystemPathTest _systemFilesystemPathTest;
BPPSystemFilesystemFileTest _systemFilesystemFileTest;

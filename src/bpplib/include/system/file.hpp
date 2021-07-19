/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 9.11.2017
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_SYSTEM_FILE_HPP
#define BPPLIB_SYSTEM_FILE_HPP

#define _CRT_SECURE_NO_WARNINGS

#include <system/memory_pool.hpp>
#include <system/filesystem.hpp>
#include <misc/exception.hpp>

#ifdef _WIN32
#define NOMINMAX
	#include <Windows.h>

	// This macro is defined in wingdi.h, I do non want ERROR macro in my projects!
	#ifdef ERROR
	#undef ERROR
	#endif
#else
	#include <sys/types.h>
	#include <sys/stat.h>
	#include <unistd.h>
#endif

#include <sstream>
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>


namespace bpp
{


/**
 * \brief Wrapper for standard file I/O operations.
 *
 * TODO -- rewrite the operations to avoid using std-C library ?
 */
class File
{
private:
	std::string mFileName;	///< Path to the file on the disk.
	std::FILE *mHandle;		///< Standard C-like handler for the file.

	static const size_t MAX_BLOCK_SIZE = 0x4fffffff;	///< Maximal size of a block being read/written at once.

public:
	/**
	 * \brief Create a file wrapper, but do not open it.
	 */
	File(const std::string &fileName)
		: mFileName(fileName), mHandle(nullptr) {}

	virtual ~File() {}

	const std::string& name() const { return mFileName; }


	/**
	 * \brief Open file in selected mode. If the file is already opened, it is closed first.
	 * \param cmode An opening mode in C-style syntax (e.g., "rb" for binary read).
	 * \throws FileError if the file cannot be opened.
	 */
	void open(const char *cmode = "w+bc")
	{
		if (mHandle != nullptr)
			close();

		mHandle = std::fopen(mFileName.c_str(), cmode);
		if (mHandle == nullptr)
			throw (FileError() << "Unable to open file '" << mFileName << "' in mode '" << cmode << "'.");
	}


	/**
	 * \brief Close the file and release its handle.
	 * \throws LogicError if the file was not opened.
	 */
	void close()
	{
		if (mHandle == nullptr)
			throw (LogicError() << "File '" << mFileName << "' must be opened before close is attempted.");

		std::fclose(mHandle);
		mHandle = nullptr;
	}


	/**
	 * \brief Check whether the file is already opened.
	 */
	bool opened() const
	{
		return mHandle != nullptr;
	}


	/**
	 * \brief Check whether the file exists on the disk and if it is a regular file.
	 */
	bool exists() const
	{
		return Path::exists(mFileName) && Path::isRegularFile(mFileName);
	}


	/**
	 * \brief Return the size of the file (in bytes). The file does not need to be opened.
	 */
	size_t size()
	{
#ifdef _WIN32
			WIN32_FILE_ATTRIBUTE_DATA fad;
			if (!GetFileAttributesExA(mFileName.c_str(), GetFileExInfoStandard, &fad))
				throw (FileError() << "Unable to read attributes of '" << mFileName << "'.");
		
			LARGE_INTEGER size;
			size.HighPart = fad.nFileSizeHigh;
			size.LowPart = fad.nFileSizeLow;
			return static_cast<size_t>(size.QuadPart);
#else
			struct stat fileStatus;
			if (stat(mFileName.c_str(), &fileStatus) != 0)
				throw (FileError() << "Unable to determine the status of '" << mFileName << "'.");
			return static_cast<size_t>(fileStatus.st_size);
#endif
	}


	/**
	 * \brief Move the internal position pointer to selected position.
	 * \param offset Offset from the beginning of the file (in bytes).
	 */
	void seek(size_t offset)
	{
		if (!opened())
			throw (LogicError() << "File '" << mFileName << "' must be opened before seek() is invoked.");
#ifdef _WIN32
		int res = _fseeki64(mHandle, offset, SEEK_SET);
#else
		int res = std::fseek(mHandle, offset, SEEK_SET);
#endif
		if (res != 0)
			throw (FileError() << "Unable to set position " << offset << " at file '" << mFileName << "'.");
	}


	/**
	 * \brief Return current position of the internal file pointer (as offset from the beginning in bytes).
	 * \throws LogicError if the file is not opened.
	 */
	size_t tell()
	{
		if (!opened())
			throw (LogicError() << "File '" << mFileName << "' must be opened before tell() is invoked.");
#ifdef _WIN32
		return static_cast<size_t>(_ftelli64(mHandle));
#else
		return static_cast<size_t>(std::ftell(mHandle));
#endif
	}


	/**
	 * \brief Templated binary read. Reads given number of items T into provided buffer.
	 * \tparam T Type of the items being read.
	 * \param buffer A buffer where the data will be loaded.
	 * \param count Number of items to read.
	 */
	template<typename T>
	void read(T* buffer, size_t count = 1)
	{
		if (!opened())
			throw (LogicError() << "File '" << mFileName << "' must be opened before read() is invoked.");

		if (std::fread(buffer, sizeof(T), count, mHandle) != count)
			throw (FileError() << "Reading of " << count*sizeof(T) << " B from file '" << mFileName << "' failed.");
	}


	/**
	 * \brief Specialized templated binary read. Reads given number of items T into std vector.
	 * \tparam T Type of the items being read.
	 * \param buffer A vector to which the data are loaded. The vector is resized to comprise all the data read.
	 * \param count Number of items to read.
	 */
	template<typename T>
	void read(std::vector<T> &buffer, size_t count)
	{
		buffer.resize(count);
		read(&buffer[0], count);
	}


	/**
	 * \brief Specialized templated binary read. Reads given number of items T into provided memory pool buffer.
	 * \tparam T Type of the items being read.
	 * \param buffer A memory pool buffer where the data will be loaded.
	 */
	template<typename T>
	void read(Buffer<T> &buffer)
	{
		read(buffer.get(), buffer.count());
	}


	/**
	 * \brief Templated binary write. Writes given number of items T from provided buffer.
	 * \tparam T Type of the items being written.
	 * \param buffer A buffer from which the data are stored.
	 * \param count Number of items being written.
	 */
	template<typename T>
	void write(const T* buffer, size_t count = 1)
	{
		if (!opened())
			throw (LogicError() << "File '" << mFileName << "' must be opened before write() is invoked.");
		
		if (count * sizeof(T) > MAX_BLOCK_SIZE) {
			// Workaround for large blocks writing ...
			size_t maxCount = MAX_BLOCK_SIZE / sizeof(T);
			size_t offset = 0;
			while (count > 0) {
				size_t currentCount = std::min<size_t>(count, maxCount);
				count -= currentCount;

				if (std::fwrite(buffer + offset, sizeof(T), currentCount, mHandle) != currentCount)
					throw (FileError() << "Writting block of " << currentCount*sizeof(T) << " B at offset "
						<< offset*sizeof(T) << " to file '" << mFileName << "' failed.");
				offset += currentCount;
			}
		}
		else if (std::fwrite(buffer, sizeof(T), count, mHandle) != count)
			throw (FileError() << "Writing of " << count*sizeof(T) << " B to file '" << mFileName << "' failed.");
	}


	/**
	 * \brief Templated binary write. Writes entire vector of data into the file.
	 * \tparam T Type of the items being written.
	 * \param buffer The vector of data being written.
	 */
	template<typename T>
	void write(const std::vector<T> &buffer)
	{
		write<T>(&buffer[0], buffer.size());
	}


	/**
	 * \brief Specialized templated binary write. Writes given number of items T from provided memory pool buffer.
	 * \tparam T Type of the items being written.
	 * \param buffer A memory pool buffer from which the data will be stored.
	 */
	template<typename T>
	void write(const Buffer<T> &buffer)
	{
		write(buffer.get(), buffer.count());
	}


	/**
	 * \brief Specialized writer useful for text files.
	 * \param str Common string to be written in a file.
	 */
	void write(const std::string &str)
	{
		write(str.c_str(), str.length());
	}


	/**
	 * \brief Specialized writer useful for text files. Writes a string followed up with newline.
	 * \param str Common string to be written in a file.
	 */
	void writeLn(const std::string &str = "", const std::string &nl = "\n")
	{
		write(str);
		write(nl);
	}


	/**
	 * \brief Flush pending write operations to the disk.
	 */
	void flush()
	{
		if (!opened())
			throw (LogicError() << "File '" << mFileName << "' must be opened before flush() is invoked.");
		
		if (fflush(mHandle) != 0)
			throw (FileError() << "Unable to flush buffers of '" << mFileName << "'.");
	}


	/**
	 * \brief Remove the file from the filesystem. If the file is opened, it is closed first.
	 */
	void unlink()
	{
		if (opened())
			close();
		
		if (exists())
			Path::unlink(mFileName);
	}


	/**
	 * \brief Simple way how to read text files line by line.
	 *		This function works only on ANSI-compatible text files.
	 *		It skips \r chars and \n is used as delimiter.
	 * \param line String object where the line is stored.
	 * \param skipEmptyLines Flag that indicates whether empty lines should be ignored.
	 * \return True if a valid string was read, false if end of file was reached.
	 */
	bool readLine(std::string &line, bool skipEmptyLines = false)
	{
		if (feof(mHandle)) return false;

		line.clear();
		char ch;
		while ((ch = std::fgetc(mHandle)) != EOF) {
			if (ch == '\r') continue;
			if (ch == '\n') {
				if (!line.empty() || !skipEmptyLines) return true;
			}
			else
				line += ch;
		}
		return !line.empty();
	}
};





/**
 * Since we avoid using fstreams, we need additional class that helps us writing text data.
 */
class TextWriter
{
private:
	File &mFile;					///< File into which the data are written.
	std::string mLineEnd;			///< Line ending (\n by default).
	std::string mSeparator;			///< Token separator.
	bool mFirstToken;				///< Flag indicating whether the following token written will be the first one on the line.
	std::stringstream mStrBuf;		///< Internal buffer for constructed strings.

public:
	TextWriter(File &file, const std::string &lineEnd = "\n", const std::string &separator = " ")
		: mFile(file), mLineEnd(lineEnd), mSeparator(separator), mFirstToken(true)
	{}

	/**
	 * Write raw data into the stream (converted to text using string stream).
	 */
	template<typename T>
	TextWriter& write(const T &data)
	{
		mFirstToken = false;
		mStrBuf << data;
		return *this;	// allow chaining
	}


	/**
	 * Write token into the stream (converted to text using string stream).
	 * Tokens are automatically separated by given separator (except the first token on the line).
	 */
	template<typename T>
	TextWriter& writeToken(const T &data)
	{
		if (mFirstToken)
			mStrBuf << data;
		else
			mStrBuf << mSeparator << data;
		mFirstToken = false;
		return *this;	// allow chaining
	}


	/**
	 * Adds endline and flush the data into the file.
	 * \param flushFile True if the underlying file is to be flushed as well.
	 */
	TextWriter& writeLine(bool flushFile = true)
	{
		mFirstToken = true;
		mStrBuf << mLineEnd;
		flush(flushFile);
		return *this;	// allow chaining
	}


	/**
	 * Flush internal buffer into the underlying file and optionally, flush the file itself.
	 * \param flushFile True if the underlying file is to be flushed as well.
	 */
	TextWriter& flush(bool flushFile = true)
	{
		std::string s = mStrBuf.str();
		if (s.empty()) return *this;
		
		mStrBuf.clear();
		mStrBuf.str(std::string());
		
		mFile.write(s.c_str(), s.size());
		if (flushFile) mFile.flush();
		return *this;	// allow chaining
	}
};

}
#endif

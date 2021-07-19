///*
// * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
// * Last Modification: 28.1.2015
// * License: CC 3.0 BY-NC (http://creativecommons.org/)
// */
//#ifndef BPPLIB_SYSTEM_FILE_RAW_HPP
//#define BPPLIB_SYSTEM_FILE_RAW_HPP
//
//#include <system/memory_pool.hpp>
//#include <system/filesystem.hpp>
//#include <misc/exception.hpp>
//
//#ifdef _WIN32
//#define NOMINMAX
//#include <windows.h>
//#else
//#include <sys/types.h>
//#include <sys/stat.h>
//#include <unistd.h>
//#endif
//
//#include <vector>
//#include <string>
//#include <algorithm>
//#include <cstdio>
//
//
//namespace bpp
//{
//
//
//	/**
//	 * \brief Wrapper for standard file I/O operations.
//	 */
//	class FileRaw
//	{
//	private:
//		std::string mFileName;	///< Path to the file on the disk.
//		std::FILE *mHandle;		///< Standard C-like handler for the file.
//
//		static const size_t MAX_BLOCK_SIZE = 0x4fffffff;	///< Maximal size of a block being read/written at once.
//
//	public:
//		/**
//		* \brief Create a file wrapper, but do not open it.
//		*/
//		File(const std::string &fileName)
//			: mFileName(fileName), mHandle(nullptr)
//		{
//		}
//
//		virtual ~File() {}
//
//		const std::string& name() const { return mFileName; }
//
//
//		/**
//		* \brief Open file in selected mode. If the file is already opened, it is closed first.
//		* \param cmode An opening mode in C-style syntax (e.g., "rb" for binary read).
//		* \throws FileError if the file cannot be opened.
//		*/
//		void open(const char *cmode = "w+bc")
//		{
//			if (mHandle != nullptr)
//				close();
//
//			mHandle = std::fopen(mFileName.c_str(), cmode);
//			if (mHandle == nullptr)
//				throw (FileError() << "Unable to open file '" << mFileName << "' in mode '" << cmode << "'.");
//		}
//
//
//		/**
//		* \brief Close the file and release its handle.
//		* \throws LogicError if the file was not opened.
//		*/
//		void close()
//		{
//			if (mHandle == nullptr)
//				throw (LogicError() << "File '" << mFileName << "' must be opened before close is attempted.");
//
//			std::fclose(mHandle);
//			mHandle = nullptr;
//		}
//
//
//		/**
//		* \brief Check whether the file is already opened.
//		*/
//		bool opened() const
//		{
//			return mHandle != nullptr;
//		}
//
//
//		/**
//		* \brief Check whether the file exists on the disk and if it is a regular file.
//		*/
//		bool exists() const
//		{
//			return Path::exists(mFileName) && Path::isRegularFile(mFileName);
//		}
//
//
//		/**
//		* \brief Return the size of the file (in bytes). The file does not need to be opened.
//		*/
//		size_t size()
//		{
//#ifdef _WIN32
//			WIN32_FILE_ATTRIBUTE_DATA fad;
//			if (!GetFileAttributesExA(mFileName.c_str(), GetFileExInfoStandard, &fad))
//				throw (FileError() << "Unable to read attributes of '" << mFileName << "'.");
//
//			LARGE_INTEGER size;
//			size.HighPart = fad.nFileSizeHigh;
//			size.LowPart = fad.nFileSizeLow;
//			return static_cast<size_t>(size.QuadPart);
//#else
//			struct stat fileStatus;
//			if (stat(mFileName.c_str(), &fileStatus) != 0)
//				throw (FileError() << "Unable to determine the status of '" << mFileName << "'.");
//			return static_cast<size_t>(fileStatus.st_size);
//#endif
//		}
//
//
//		/**
//		* \brief Move the internal position pointer to selected position.
//		* \param offset Offset from the beginning of the file (in bytes).
//		*/
//		void seek(size_t offset)
//		{
//			if (!opened())
//				throw (LogicError() << "File '" << mFileName << "' must be opened before seek() is invoked.");
//#ifdef _WIN32
//			int res = _fseeki64(mHandle, offset, SEEK_SET);
//#else
//			int res = std::fseek(mHandle, offset, SEEK_SET);
//#endif
//			if (res != 0)
//				throw (FileError() << "Unable to set position " << offset << " at file '" << mFileName << "'.");
//		}
//
//
//		/**
//		* \brief Return current position of the internal file pointer (as offset from the beginning in bytes).
//		* \throws LogicError if the file is not opened.
//		*/
//		size_t tell()
//		{
//			if (!opened())
//				throw (LogicError() << "File '" << mFileName << "' must be opened before tell() is invoked.");
//#ifdef _WIN32
//			return static_cast<size_t>(_ftelli64(mHandle));
//#else
//			return static_cast<size_t>(std::ftell(mHandle));
//#endif
//		}
//
//
//		/**
//		* \brief Templated binary read. Reads given number of items T into provided buffer.
//		* \tparam T Type of the items being read.
//		* \param buffer A buffer where the data will be loaded.
//		* \param count Number of items to read.
//		*/
//		template<typename T>
//		void read(T* buffer, size_t count = 1)
//		{
//			if (!opened())
//				throw (LogicError() << "File '" << mFileName << "' must be opened before read() is invoked.");
//
//			if (std::fread(buffer, sizeof(T), count, mHandle) != count)
//				throw (FileError() << "Reading of " << count*sizeof(T) << " B from file '" << mFileName << "' failed.");
//		}
//
//
//		/**
//		* \brief Specialized templated binary read. Reads given number of items T into std vector.
//		* \tparam T Type of the items being read.
//		* \param buffer A vector to which the data are loaded. The vector is resized to comprise all the data read.
//		* \param count Number of items to read.
//		*/
//		template<typename T>
//		void read(std::vector<T> &buffer, size_t count)
//		{
//			buffer.resize(count);
//			read(&buffer[0], count);
//		}
//
//
//		/**
//		* \brief Specialized templated binary read. Reads given number of items T into provided memory pool buffer.
//		* \tparam T Type of the items being read.
//		* \param buffer A memory pool buffer where the data will be loaded.
//		*/
//		template<typename T>
//		void read(Buffer<T> &buffer)
//		{
//			read(buffer.get(), buffer.count());
//		}
//
//
//		/**
//		* \brief Templated binary write. Writes given number of items T from provided buffer.
//		* \tparam T Type of the items being written.
//		* \param buffer A buffer from which the data are stored.
//		* \param count Number of items being written.
//		*/
//		template<typename T>
//		void write(const T* buffer, size_t count = 1)
//		{
//			if (!opened())
//				throw (LogicError() << "File '" << mFileName << "' must be opened before write() is invoked.");
//
//			if (count * sizeof(T) > MAX_BLOCK_SIZE) {
//				// Workaround for large blocks writing ...
//				size_t maxCount = MAX_BLOCK_SIZE / sizeof(T);
//				size_t offset = 0;
//				while (count > 0) {
//					size_t currentCount = std::min<size_t>(count, maxCount);
//					count -= currentCount;
//
//					if (std::fwrite(buffer + offset, sizeof(T), currentCount, mHandle) != currentCount)
//						throw (FileError() << "Writting block of " << currentCount*sizeof(T) << " B at offset "
//						<< offset*sizeof(T) << " to file '" << mFileName << "' failed.");
//					offset += currentCount;
//				}
//			}
//			else if (std::fwrite(buffer, sizeof(T), count, mHandle) != count)
//				throw (FileError() << "Writing of " << count*sizeof(T) << " B to file '" << mFileName << "' failed.");
//		}
//
//
//		/**
//		* \brief Templated binary write. Writes entire vector of data into the file.
//		* \tparam T Type of the items being written.
//		* \param buffer The vector of data being written.
//		*/
//		template<typename T>
//		void write(const std::vector<T> &buffer)
//		{
//			write<T>(&buffer[0], buffer.size());
//		}
//
//
//		/**
//		* \brief Specialized templated binary write. Writes given number of items T from provided memory pool buffer.
//		* \tparam T Type of the items being written.
//		* \param buffer A memory pool buffer from which the data will be stored.
//		*/
//		template<typename T>
//		void write(const Buffer<T> &buffer)
//		{
//			write(buffer.get(), buffer.count());
//		}
//
//
//		/**
//		* \brief Flush pending write operations to the disk.
//		*/
//		void flush()
//		{
//			if (!opened())
//				throw (LogicError() << "File '" << mFileName << "' must be opened before flush() is invoked.");
//
//			if (fflush(mHandle) != 0)
//				throw (FileError() << "Unable to flush buffers of '" << mFileName << "'.");
//		}
//
//
//		/**
//		* \brief Remove the file from the filesystem. If the file is opened, it is closed first.
//		*/
//		void unlink()
//		{
//			if (opened())
//				close();
//
//			if (exists())
//				Path::unlink(mFileName);
//		}
//	};
//
//
//}
//#endif

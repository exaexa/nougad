/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 30.1.2015
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_SYSTEM_FILESYSTEM_HPP
#define BPPLIB_SYSTEM_FILESYSTEM_HPP

#include <misc/exception.hpp>

#ifdef _WIN32
	#define NOMINMAX
	#include <windows.h>

	// This macro is defined in wingdi.h, I do non want ERROR macro in my projects!
	#ifdef ERROR
	#undef ERROR
	#endif
#else
	#include <sys/types.h>
	#include <sys/stat.h>
	#include <unistd.h>
	#include <errno.h>
#endif

#include <string>


namespace bpp
{


/**
 * \brief A special type of runtime error representing error with file operations.
 */
class FileError : public RuntimeError
{
public:
	FileError() : RuntimeError() {}
	FileError(const char *msg) : RuntimeError(msg) {}
	FileError(const std::string &msg) : RuntimeError(msg) {}
	virtual ~FileError() noexcept {}


	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	FileError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};




	
/**
 * \brief Encapsulates various filesystem functions implemented for both Windows and Unix (Linux) OSes.
 */
class Path
{
public:
	/**
	 * \brief Return a file name from the path string.
	 * \param path A path to a file.
	 * \return The filename part of the path (if the path ends with '/' or '\', empty string is returned).
	 * \note This function works for both windows and linux paths.
	 */
	static std::string getFileName(const std::string &path)
	{
		size_t pos = path.find_last_of("/\\");
		return (pos != std::string::npos)
			? path.substr(pos+1) : path;
	}


	/**
	 * \brief Return a path to file without its last extension.
	 * \param path A path to a file.
	 * \return Remaining part of the path after the trailing .* was removed.
	 * \note This function works for both windows and linux paths.
	 */
	static std::string cropExtension(const std::string &path)
	{
		if (path.empty())
			throw RuntimeError("Empty path provided to crop_extension function.");

		if (path[path.length()-1] == '/' || path[path.length()-1] == '\\')
			return path;	// path ends with dir separator => no file name to crop

		std::size_t lastSlashPos = path.find_last_of("/\\");
		std::size_t pos = path.find_last_of('.');

		// Avoid cases where last detected '.' is in directory name.
		if (pos == std::string::npos || (lastSlashPos != std::string::npos && pos < lastSlashPos))
			return path;

		// Avoid croping the current/parent directory paths ('.' and '..').
		if (path[path.length()-1] == '.') {
			if (path.length() == 1 || path[path.length()-2] == '/' || path[path.length()-2] == '\\')
				return path;	// The '.' represents current directory, not an extension separator.
			
			if (path[path.length()-2] == '.') {		// There are two consecutive dots '..'
				if (path.length() == 2 || path[path.length()-3] == '/' || path[path.length()-3] == '\\')
					return path;	// The '..' represents parent directory, not an extension separator.
			}
		}

		return path.substr(0, pos);
	}


	/**
	 * \brief Verify that selected path exists.
	 * \param path A string representing a filesystem path.
	 */
	static bool exists(const std::string &path)
	{
#ifdef _WIN32
		return GetFileAttributesA(path.c_str()) != INVALID_FILE_ATTRIBUTES;
#else
		struct stat fileStatus;
		if (stat(path.c_str(), &fileStatus) == 0)
			return true;

		int err = errno;
		if (err != ENOENT && err != ENOTDIR)
			throw (FileError() << "Error occured when reading status of file '" << path << "' (errno = " << err << ").");

		return false;
#endif
	}


	/**
	* \brief Verify that selected existing path points to a regular file.
	* \param path A string representing a filesystem path.
	*/
	static bool isRegularFile(const std::string &path)
	{
#ifdef _WIN32
		DWORD attrs = GetFileAttributesA(path.c_str());
		if (attrs == INVALID_FILE_ATTRIBUTES)
			throw (FileError() << "Unable to retrieve the attributes of '" << path << "'.");

		return ((attrs & FILE_ATTRIBUTE_DIRECTORY)	== 0)
			&& ((attrs & FILE_ATTRIBUTE_DEVICE)		== 0)
			&& ((attrs & FILE_ATTRIBUTE_VIRTUAL)	== 0);
#else
		struct stat fileStatus;
		if (stat(path.c_str(), &fileStatus) != 0)
			throw (FileError() << "Unable to determine the status of '" << path << "'.");

		return S_ISREG(fileStatus.st_mode);
#endif
	}


	/**
	* \brief Verify that selected existing path points to a directory.
	* \param path A string representing a filesystem path.
	*/
	static bool isDirectory(const std::string &path)
	{
#ifdef _WIN32
		DWORD attrs = GetFileAttributesA(path.c_str());
		if (attrs == INVALID_FILE_ATTRIBUTES)
			throw (FileError() << "Unable to retrieve the attributes of '" << path << "'.");

		return ((attrs & FILE_ATTRIBUTE_DIRECTORY) != 0);
#else
		struct stat fileStatus;
		if (stat(path.c_str(), &fileStatus) != 0)
			throw (FileError() << "Unable to determine the status of '" << path << "'.");

		return S_ISDIR(fileStatus.st_mode);
#endif
	}


	/**
	 * \brief Unlink selected (existing) file.
	 * \param path A string representing a filesystem path to a file which is to be removed.
	 */
	static void unlink(const std::string &path)
	{
		if (std::remove(path.c_str()) != 0)
			throw (FileError() << "Unable to remove '" << path << "'.");

	}
};


}
#endif

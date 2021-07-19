/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 4.7.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_SYSTEM_FILE_POOL_HPP
#define BPPLIB_SYSTEM_FILE_POOL_HPP

#include <system/file.hpp>
#include <misc/exception.hpp>

#include <string>
#include <vector>
#include <sstream>


namespace bpp
{


/**
 * \brief A pool of temporary files.
 */
class FilePool
{
private:
	std::string mDirectory;		///< Path to the directory where the files are created.
	std::string mPrefix;		///< A file name prefix common to all generated files.
	size_t mMinDigits;			///< Minimal number of digits used for file numbering.
	std::vector<File*> mFiles;	///< The list of maintained files.


	/**
	 * \brief Geneate a name for another tmp file.
	 */
	std::string nextFileName()
	{
		std::stringstream buf;
		buf << mFiles.size();
		std::string num(buf.str());

		size_t leadingZeros = std::max<size_t>(num.length(), mMinDigits) - num.length();
		return mPrefix + std::string(leadingZeros, '0') + num + ".tmp";
	}

public:
	/**
	 * \brief Initialize the file poo.
	 * \param directory Path to the directory where the files are being created.
	 * \param prefix The tmp file name prefix common to all generated files.
	 * \param minDigits Minimal number of digits used for file name numbering.
	 */
	FilePool(const std::string &directory, const std::string &prefix = "", size_t minDigits = 3)
		: mDirectory(directory), mPrefix(prefix), mMinDigits(minDigits)
	{
		if (mMinDigits > 16) mMinDigits = 16;
	}


	/**
	 * \brief Close all opened files bud do not delete them.
	 */
	virtual ~FilePool()
	{
		for (size_t i = 0; i < mFiles.size(); ++i) {
			if (mFiles[i] != nullptr && mFiles[i]->opened())
				mFiles[i]->close();
		}
	}


	/**
	 * \brief Get the number of allocated files.
	 */
	size_t count() const { return mFiles.size(); }


	/**
	 * \brief Allocate another file.
	 * \param open Allocate and open (i.e., physically create) the file. The file is opended
	 *		in read write binary mode.
	 * \return Reference to the newly created file.
	 */
	File& newFile(bool open = true)
	{
		mFiles.push_back(new File(mDirectory + "/" + nextFileName()));
		if (open)
			mFiles.back()->open();
		return *mFiles.back();
	}


	/**
	 * \brief Access the i-th allocated file.
	 */
	File& operator[](size_t i)
	{
		if (i >= mFiles.size())
			throw (bpp::StreamException() << "Unable to access tmp file #" << i << ", only " << mFiles.size() << " were issued.");
		return *mFiles[i];
	}


	/**
	 * \brief Delete all files and clear the pool.
	 */
	void clearAll()
	{
		for (size_t i = 0; i < mFiles.size(); ++i) {
			mFiles[i]->unlink();
			delete mFiles[i];
			mFiles[i] = nullptr;
		}
		mFiles.clear();
	}
};


}

#endif

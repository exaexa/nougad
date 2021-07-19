/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 29.8.2017
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_DATASTRUCT_STRING_INDEX_HPP
#define BPPLIB_DATASTRUCT_STRING_INDEX_HPP

#include <misc/exception.hpp>
#include <system/file.hpp>
#include <system/mmap_file.hpp>

#include <map>
#include <vector>
#include <string>


namespace bpp
{

/**
 * \brief A string index that lists unique strings and assign them temporal IDs.
 *
 *	This index intention is to translate strings into numerical IDs which are
 *	far better for equality comparisons and storage. The index is decided
 *	for fast translations in both ways (string to ID and ID to string).
 *
 * \tparam STR Type of the strings kept in the index.
 * \tparam IDX Numeric type used for indices.
 */
template<typename STR = std::string, typename IDX = std::size_t>
class StringIndex
{
private:
	typedef std::map<STR, IDX> map_t;

	std::vector<typename map_t::iterator> mStrings;
	map_t mIndex;


public:
	/**
	 * \brief Retrieves string value by its index.
	 */
	const STR& operator[](IDX idx) const	{ return mStrings[idx]->first; }


	/**
	 * \brief Retrieves copy of a string value by its index.
	 */
	STR operator[](IDX idx)				{ return mStrings[idx]->first; }


	/**
	 * \brief Retrieves index of given string.
	 *		Exception is thrown if the string key does not exist.
	 */
	IDX operator[](const STR &str) const
	{
		auto it = mIndex.find(str);
		if (it == mIndex.end())
			throw (bpp::RuntimeError() << "Key '" << str << "' is not present in the string index.");
		return it->second;
	}


	/**
	 * \brief Retrieves index of given string. The string is added
	 *		to the structure if the string key does not exist.
	 */
	IDX operator[](const STR &str)
	{
		auto res = mIndex.emplace(str, (IDX)mStrings.size());
		if (res.second) {	// The emplace really inserted new element.
			mStrings.push_back(res.first);
		}
		return res.first->second;
	}


	/**
	 * \brief Add new string into the structure.
	 * \param str The string being added.
	 * \return Index of the newly added string.
	 * \throw RuntimeError if the string key already exists.
	 */
	IDX add(const STR &str)
	{
		auto res = mIndex.emplace(str, (IDX)mStrings.size());
		if (!res.second)
			throw (bpp::RuntimeError() << "Key '" << str << "' is already present in the string index.");
		mStrings.push_back(res.first);
		return (IDX)(mStrings.size()-1);
	}


	/**
	 * \brief Return the number of strings in the structure.
	 */
	std::size_t size()	{ return mStrings.size(); }


	/**
	 * \brief Check whether the structure is empty.
	 */
	bool empty()	{ return size() == 0; }
	

	/**
	 * \brief Clear the structure (former indices are no longer valid).
	 */
	void clear()
	{
		mStrings.clear();
		mIndex.clear();
	}


	/**
	 * Save the table as a binary file.
	 */
	void save(const std::string &fileName) const
	{
		bpp::File file(fileName);
		file.open("wb");
		std::uint32_t magic = 0xb197ab1e;
		file.write(&magic);
		
		std::uint32_t size = (std::uint32_t)mStrings.size();
		file.write(&size);
		for (auto && it : mStrings) {
			if (it != mIndex.end()) {
				size = (std::uint32_t)it->first.size();
				file.write(&size);
				file.write<char>(&(it->first[0]), ((size+3) & ~(std::uint32_t)3));	// pad the string to nearest 32bit boundary
			}
			else {
				size = 0;
				file.write(&size);
			}
		}
		file.close();
	}


	/**
	 * Load the table from a file created by save() method.
	 */
	void load(const std::string &fileName)
	{
		clear();
		
		// Memory map entire file.
		bpp::MMapFile file;
		file.open(fileName);
		char *data = (char*)file.getData();

		// Check the magic header.
		std::uint32_t magic = *reinterpret_cast<std::uint32_t*>(data);
		if (magic != 0xb197ab1e)
			throw (bpp::RuntimeError() << "File '" << fileName << "' is not a bpp::StringIndex file.");
		data += sizeof(std::uint32_t);
		
		// Read the number of items on table.
		IDX size = (IDX)*reinterpret_cast<std::uint32_t*>(data);
		data += sizeof(std::uint32_t);

		// Load all strings.
		mStrings.resize(size);
		for (IDX i = 0; i < size; ++i) {
			std::uint32_t len = *reinterpret_cast<std::uint32_t*>(data);
			data += sizeof(std::uint32_t);
			if (len > 0) {
				std::string str(data, len);
				auto res = mIndex.emplace(str, (IDX)i);
				mStrings[i] = res.first;
				data += (len+3) & ~(std::uint32_t)3;
			}
			else
				mStrings[i] = mIndex.end();
		}

		file.close();
	}
};

}
#endif

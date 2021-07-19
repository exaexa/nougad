#include "../test.hpp"
#include <datastruct/string_index.hpp>

#include <iostream>

/**
 * \brief Test for the string index data structure.
 */
class BPPDatastructStringIndexTest : public BPPLibTest
{
public:
	BPPDatastructStringIndexTest() : BPPLibTest("datastruct/string_index") {}

	virtual bool run() const
	{
		bpp::StringIndex<std::string, size_t> index;

		if (index.add("aaa") != 0 || index.add("bbb") != 1 || index["aaa"] != 0 || index["ccc"] != 2) {
			std::cout << "Adding data to string index failed." << std::endl;
			return false;
		}

		if (index.size() != 3) {
			std::cout << "Index size is " << index.size() << ", yet 3 items were inserted." << std::endl;
			return false;
		}

		const bpp::StringIndex<std::string, size_t> &pindex = index;
		if (pindex[0] != "aaa" || pindex[1] != "bbb" || pindex[2] != "ccc") {
			std::cout << "Operator [idx] failed on constant index." << std::endl;
			return false;
		}

		if (pindex["aaa"] != 0 || pindex["bbb"] != 1 || pindex["ccc"] != 2) {
			std::cout << "Operator [str] failed on constant index." << std::endl;
			return false;
		}

		pindex.save(".tmp.string-index.test");

		index.clear();
		if (!index.empty()) {
			std::cout << "Index size is " << index.size() << ", yet the index was cleared." << std::endl;
			return false;
		}

		index.load(".tmp.string-index.test");
		if (index.size() != 3 || index[0] != "aaa" || index[1] != "bbb" || index[2] != "ccc"
			|| index["aaa"] != 0 || index["bbb"] != 1 || index["ccc"] != 2) {
			std::cout << "Index was not loaded correctly." << std::endl;
			return false;
		}

		bpp::Path::unlink(".tmp.string-index.test");
		return true;
	}
};




// Static declarations of singletons
BPPDatastructStringIndexTest _datastructStringIndexTest;

#include "../test.hpp"
#include <datastruct/heap/regular.hpp>
#include <datastruct/heap/looser.hpp>
#include <math/random.hpp>

#include <vector>
#include <algorithm>
#include <iostream>


/**
 * \brief Template class for all heap tests. The template gets parameter that specifies
 *		which heap class is being tested.
 */
template<class HEAP>
class BPPDatastructHeapTest : public BPPLibTest
{
public:
	BPPDatastructHeapTest(const std::string &name) : BPPLibTest("datastruct/heap/" + name) {}

	virtual bool run() const
	{
		std::vector<typename HEAP::key_t> data;
		for (size_t i = 1; i <= 1000; ++i)
			data.push_back(typename HEAP::key_t(i));
		bpp::Random<size_t>::shuffle(data);

		std::vector<typename HEAP::key_t> tmp(data);
		HEAP heap(tmp);
		HEAP heap1(tmp);
		HEAP heap2;
		for (size_t i = 0; i < tmp.size(); ++i)
			heap2.add(tmp[i]);
		HEAP heap3(tmp, true);
		if (!tmp.empty()) {
			std::cout << "The tmp data vector was not emptied by heap acquiring constructor." << std::endl;
			return false;
		}

		std::sort(data.begin(), data.end());
		for (size_t i = 0; i < data.size(); ++i) {
			if (heap1.getTop() != data[i]) {
				std::cout << "Heap1 invalid top value found while processing item " << i << std::endl;
				return false;
			}
			if (heap2.getTop() != data[i]) {
				std::cout << "Heap2 invalid top value found while processing item " << i << std::endl;
				return false;
			}
			if (heap3.getTop() != data[i]) {
				std::cout << "Heap3 invalid top value found while processing item " << i << std::endl;
				return false;
			}
			heap1.removeTop();
			heap2.removeTop();
			heap3.removeTop();
		}

		if (!heap1.empty()) {
			std::cout << "Heap1 is not empty after heapsort." << std::endl;
			return false;
		}
		if (!heap2.empty()) {
			std::cout << "Heap2 is not empty after heapsort." << std::endl;
			return false;
		}
		if (!heap3.empty()) {
			std::cout << "Heap3 is not empty after heapsort." << std::endl;
			return false;
		}


		heap.getTop() = typename HEAP::key_t(10000);
		heap.updatePosition(0);
		if (heap.getTop() != typename HEAP::key_t(2)) {
			std::cout << "Update position [0] (bubble down) failed." << std::endl;
			return false;
		}

		heap[500] = typename HEAP::key_t(0);
		heap.updatePosition(500);
		if (heap.getTop() != typename HEAP::key_t(0)) {
			std::cout << "Update position [500] (bubble up) failed." << std::endl;
			return false;
		}

		return true;
	}
};



class BPPDatastructHeap2RegularUintTest : public BPPDatastructHeapTest< bpp::RegularHeap<std::uint32_t, 2> >
{
public:
	BPPDatastructHeap2RegularUintTest() : BPPDatastructHeapTest< bpp::RegularHeap<std::uint32_t, 2> >("regular/d2-uin32") {}
};


class BPPDatastructHeap5RegularDoubleTest : public BPPDatastructHeapTest< bpp::RegularHeap<double, 5> >
{
public:
	BPPDatastructHeap5RegularDoubleTest() : BPPDatastructHeapTest< bpp::RegularHeap<double, 5> >("regular/d5-double") {}
};


class BPPDatastructHeap2LooserUint64Test : public BPPDatastructHeapTest< bpp::RegularHeap<uint64_t, 2> >
{
public:
	BPPDatastructHeap2LooserUint64Test() : BPPDatastructHeapTest< bpp::RegularHeap<uint64_t, 2> >("looser/d2-uin64") {}
};


class BPPDatastructHeap19LooserIntTest : public BPPDatastructHeapTest< bpp::RegularHeap<int, 19> >
{
public:
	BPPDatastructHeap19LooserIntTest() : BPPDatastructHeapTest< bpp::RegularHeap<int, 19> >("looser/d19-int") {}
};



// Static declarations of singletons
BPPDatastructHeap2RegularUintTest		_datastructHeap2ReguarUintTest;
BPPDatastructHeap5RegularDoubleTest		_datastructHeap5RegularDoubleTest;
BPPDatastructHeap2LooserUint64Test		_datastructHeap2LooserUint64Test;
BPPDatastructHeap19LooserIntTest		_datastructHeap19LooserIntTest;

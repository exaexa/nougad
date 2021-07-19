/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 1.7.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_DATASTRUCT_KEY_VALUE_HPP
#define BPPLIB_DATASTRUCT_KEY_VALUE_HPP

#include <ostream>

namespace bpp
{


/**
 * \brief A structure that holds key-value pair. Comparison operations
 *		are adjusted to compare the key only.
 * \tparam K type of the key.
 * \tparam V type of the value.
 */
template<typename K, typename V>
struct KeyValue
{
public:
	K key;
	V value;
	
	KeyValue() {}
	KeyValue(const KeyValue<K,V> &kv)
		: key(kv.key), value(kv.value) {}
	KeyValue(K _key, V _value) : key(_key), value(_value) {}


	KeyValue<K,V>& operator=(const KeyValue<K,V> &kv)
	{
		key = kv.key;
		value = kv.value;
		return *this;
	}

	bool operator<(const KeyValue<K,V> &kv) const
	{
		return key < kv.key;
	}

	bool operator<=(const KeyValue<K,V> &kv) const
	{
		return key <= kv.key;
	}

	bool operator==(const KeyValue<K,V> &kv) const
	{
		return key == kv.key;
	}

	bool operator!=(const KeyValue<K,V> &kv) const
	{
		return key != kv.key;
	}

	bool operator>=(const KeyValue<K,V> &kv) const
	{
		return key >= kv.key;
	}

	bool operator>(const KeyValue<K,V> &kv) const
	{
		return key > kv.key;
	}

	
};


template<typename K, typename V>
std::ostream& operator<<(std::ostream &stream, const KeyValue<K,V> &kv)
{
	stream << "(" << kv.key << "," << kv.value << ")";
	return stream;
}


}
#endif

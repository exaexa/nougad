/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 30.5.2017
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 *
 * Note: at this point, it is not certain if this class will be useful and maintained.
 * It was supposed to be a part of larger data loading framework, which is currently not implemented.
 */
#ifndef BPPLIB_MISC_SETTER_HPP
#define BPPLIB_MISC_SETTER_HPP

#include "algo/strings.hpp"
#include <memory>


namespace bpp
{


/**
 * Base class for all setters. Setters are helper classes that simplifies member/variable loading and
 * initialization in dynamic scenarios (e.g., when members are addresed by strings).
 *
 * \note At the moment, the setter uses only string tokens. In the future, more elaborate implementation is advisable.
 */
class ISetter
{
protected:
	virtual void setTokenVirt(const bpp::ITokenizer::ref_t &tok) const = 0;

public:
	virtual ~ISetter() {}		// enforce virtual destructors
	void setToken(const bpp::ITokenizer::ref_t &tok) const
	{
		setTokenVirt(tok);
	}
};


/**
 * Simple templated setter. It set all types that can be loaded from a token or simply casted from other source type.
 */
class INamedSetter : public ISetter
{
protected:
	std::string mName;

public:
	INamedSetter(const std::string &name) : mName(name) {}
	const std::string& getName() const { return mName; }
};



/**
 * Templated named setter. It set all types that can be loaded from a token or simply casted from other source type.
 */
template<typename T>
class Setter : public INamedSetter
{
protected:
	T &mTarget;

	virtual void setTokenVirt(const bpp::ITokenizer::ref_t &tok) const
	{
		mTarget = tok.as<T>();
	}

public:
	Setter(T &target, const std::string &name) : INamedSetter(name), mTarget(mTarget) {}
};



}

#endif

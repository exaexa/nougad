/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 22.12.2013
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_MISC_EXCEPTION_HPP
#define BPPLIB_MISC_EXCEPTION_HPP

#include <exception>
#include <string>
#include <sstream>


namespace bpp
{

/**
 * \brief Specific exception that behaves like a stream, so it can cummulate
 *		error messages more easily.
 */
class StreamException : public std::exception
{
protected:
	std::string mMessage;	///< Internal buffer where the message is kept.

public:
	StreamException() : std::exception() {}
	StreamException(const char *msg) : std::exception(), mMessage(msg) {}
	StreamException(const std::string &msg) : std::exception(), mMessage(msg) {}
	virtual ~StreamException() noexcept {}

	virtual const char* what() const throw()
	{
		return mMessage.c_str();
	}

	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	StreamException& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};


/**
 * \brief A stream exception that is base for all runtime errors.
 */
class RuntimeError : public StreamException
{
public:
	RuntimeError() : StreamException() {}
	RuntimeError(const char *msg) : StreamException(msg) {}
	RuntimeError(const std::string &msg) : StreamException(msg) {}
	virtual ~RuntimeError() noexcept {}


	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	RuntimeError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};


/**
 * \brief A special type of runtime error representing error in logic of the
 *		application/algorithm/utility... (e.g., invokation of a method when
 *		the object is in indesirable state or an invalid configuration of
 *		an utility.
 */
class LogicError : public RuntimeError
{
public:
	LogicError() : RuntimeError() {}
	LogicError(const char *msg) : RuntimeError(msg) {}
	LogicError(const std::string &msg) : RuntimeError(msg) {}
	virtual ~LogicError() noexcept {}


	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	LogicError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};


/**
* \brief Internal error that indicate missing implementation or virtual function override.
*/
class NotImplementedError : public RuntimeError
{
public:
	NotImplementedError() : RuntimeError() {}
	NotImplementedError(const char *msg) : RuntimeError(msg) {}
	NotImplementedError(const std::string &msg) : RuntimeError(msg) {}
	virtual ~NotImplementedError() noexcept {}


	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	NotImplementedError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};


}
#endif

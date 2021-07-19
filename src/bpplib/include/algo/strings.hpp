/*
* Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
* Last Modification: 27.5.2017
* License: CC 3.0 BY-NC (http://creativecommons.org/)
*/
#ifndef BPPLIB_ALGO_STRINGS_HPP
#define BPPLIB_ALGO_STRINGS_HPP

#include <misc/exception.hpp>

#include <string>
#include <vector>
#include <sstream>
#include <iomanip>
#include <ctime>

namespace bpp
{

/**
 * \brief A stream exception that is base for all runtime errors.
 */
class TokenizerError : public RuntimeError
{
public:
	TokenizerError() : RuntimeError() {}
	TokenizerError(const char *msg) : RuntimeError(msg) {}
	TokenizerError(const std::string &msg) : RuntimeError(msg) {}
	virtual ~TokenizerError() noexcept {}


	// Overloading << operator that uses stringstream to append data to mMessage.
	template<typename T>
	TokenizerError& operator<<(const T &data)
	{
		std::stringstream stream;
		stream << mMessage << data;
		mMessage = stream.str();
		return *this;
	}
};





/**
 * Base interface for tokenizers. Tokenizer parses a string and returns list
 * of references to individual tokens.
 */
class ITokenizer
{
public:
	/**
	 * Reference structure representing one matched token.
	 */
	struct ref_t
	{
		const char *str;		///< Pointer to the beginning of the token (not ended by zero).
		std::size_t length;		///< Length of the token (number of chars._

		/**
		 * Templated convertor that extracts the token as desired type.
		 */
		template<typename T>
		T as() const
		{
			return T(as<std::string>());
		}


		/**
		 * Try to parse the string token as date time in given format.
		 * Exception is thrown if the format does not match.
		 */
		std::tm asDateTime(const std::string format = "%d.%m.%Y %H:%M:%S") const
		{
			std::tm tm = {};
#ifdef _WIN32
			// Windows version works fine with C++ get_time()
			std::istringstream buf(std::string(str, length));
			buf >> std::get_time(&tm, format.c_str());
			if (buf.fail())
				throw (bpp::TokenizerError() << "Token '" << buf.str() << "' cannot be parsed as datetime using format '" << format << "'.");
#else
			// G++ has a bug in get_time (tokens such ass %d, %m, or %H does not work without leading zeroes.
			// We use strptime from time.h instead.
			std::string date(str, length);
			auto res = strptime(date.c_str(), format.c_str(), &tm);
			if (res == nullptr)
				throw (bpp::TokenizerError() << "Token '" << date << "' cannot be parsed as datetime using format '" << format << "'.");
#endif
			return tm;
		}
	};

protected:
	/**
	 * Internal string comparison for equality.
	 */
	static bool _strcmp(const char *str1, const char *str2, std::size_t len)
	{
		while (len > 0) {
			if (*str1 != *str2) return false;
			--len; ++str1; ++str2;
		}
		return true;
	}

	/**
	 * Virtual function overriden in derived classes to perform the tokenization itself.
	 */
	virtual void tokenizeVirtual(const std::string &str, std::vector<ref_t> &tokens, bool skipEmpty) = 0;

public:
	/**
	 * Perform tokenization and collect the references.
	 * \param str String to be parsed.
	 * \param tokens Vector where the token references will be stored.
	 * \param skipEmpty Whether empty tokens should be included into references or not.
	 */
	void tokenize(const std::string &str, std::vector<ref_t> &tokens, bool skipEmpty = false)
	{
		tokenizeVirtual(str, tokens, skipEmpty);
	}
};

/*
 * Explicit specializations of retyping as() member function.
 */
template<> std::string ITokenizer::ref_t::as<std::string>() const { return std::string(str, length); }
template<> int ITokenizer::ref_t::as<int>() const { return std::stoi(as<std::string>()); }
template<> long ITokenizer::ref_t::as<long>() const { return std::stol(as<std::string>()); }
template<> long long ITokenizer::ref_t::as<long long>() const { return std::stoll(as<std::string>()); }
template<> unsigned int ITokenizer::ref_t::as<unsigned int>() const { return (unsigned int)std::stoul(as<std::string>()); }
template<> unsigned long ITokenizer::ref_t::as<unsigned long>() const { return std::stoul(as<std::string>()); }
template<> unsigned long long ITokenizer::ref_t::as<unsigned long long>() const { return std::stoull(as<std::string>()); }
template<> float ITokenizer::ref_t::as<float>() const { return std::stof(as<std::string>()); }
template<> double ITokenizer::ref_t::as<double>() const { return std::stod(as<std::string>()); }





/**
 * Simple engine for string tokenization. It splits strings using fixed delimiter.
 */
class SimpleTokenizer : public ITokenizer
{
public:
	typedef ITokenizer::ref_t ref_t;

private:
	std::string mDelimiter;		///< Substring used as delimiter.

	virtual void tokenizeVirtual(const std::string &str, std::vector<ref_t> &tokens, bool skipEmpty)
	{
		// Make sure the real implementation is recalled correctly.
		doTokenize(str, tokens, skipEmpty);
	}

public:
	SimpleTokenizer() : mDelimiter(" ") {}
	SimpleTokenizer(const std::string &delim) : mDelimiter(delim) {}

	/**
	 * Change the delimiter substring.
	 */
	void setDelimiter(const std::string &delim)
	{
		mDelimiter = delim;
	}

	/**
	 * The real implementation of tokenization. The function is exposed,
	 * so it can be called directly to avoid late-binding costs in HPC situations.
	 */
	void doTokenize(const std::string &str, std::vector<ref_t> &tokens, bool skipEmpty = false)
	{
		tokens.clear();
		if (mDelimiter.empty()) return;

		std::size_t idx = 0, len = str.length(), dlen = mDelimiter.length();
		const char *cstr = str.c_str();
		const char *delim = mDelimiter.c_str();

		// Current reference being constructed.
		ref_t ref;
		ref.str = cstr;
		ref.length = 0;

		// Scan the input string for delimiters
		while (idx + dlen <= len) {
			if (_strcmp(cstr + idx, delim, dlen)) {		// if current location (at idx) holds the delimiter string
				idx += dlen;							// skip the delimiter
				if (ref.length > 0 || !skipEmpty)
					tokens.push_back(ref);				// save current token reference

														// Start a new token reference
				ref.str = cstr + idx;
				ref.length = 0;
			}
			else {
				++idx;					// advance the scanning position
				++ref.length;			// include current characted in the current reference
			}
		}

		ref.length += len - idx;		// fix the final token's length
		if (ref.length > 0 || !skipEmpty)
			tokens.push_back(ref);		// save the last token
	}
};





/**
 * Tokenizer that parses one CSV record.
 * The tokenizer is capable of processing quoted records, but it does not modify the token strings.
 * I.e., if a string "a""bc" is parsed as token, the result will be a""bc (even though the doubled quotes
 * should be recoded into singlequotes).
 */
class CSVTokenizer : public ITokenizer
{
public:
	typedef ITokenizer::ref_t ref_t;

private:
	/**
	 * Check the configuration (delimiter, quote, and escape char) is correct.
	 * An exception is thrown if not.
	 */
	void checkConfig()
	{
		if (mDelimiter == mQuote)
			throw (bpp::RuntimeError() << "The CSV tokenizer configuration is not correct -- the control characters are the same (delimiter = '"
				<< mDelimiter << "', quote = '" << mQuote << "').");
	}

	virtual void tokenizeVirtual(const std::string &str, std::vector<ref_t> &tokens, bool skipEmpty)
	{
		// Make sure the real implementation is recalled correctly.
		doTokenize(str, tokens, skipEmpty);
	}

public:
	char mDelimiter;	///< Character that separates the records.
	char mQuote;		///< Character that quotes strings.

	CSVTokenizer(char delimiter = ',', char quote = '"')
		: mDelimiter(delimiter), mQuote(quote)
	{
		checkConfig();
	}
	

	/**
	 * The real implementation of tokenization. The function is exposed,
	 * so it can be called directly to avoid late-binding costs in HPC situations.
	 */
	void doTokenize(const std::string &str, std::vector<ref_t> &tokens, bool skipEmpty = false)
	{
		checkConfig();
		tokens.clear();

		std::size_t idx = 0, len = str.length();
		const char *cstr = str.c_str();

		// Current reference being constructed.
		ref_t ref;
		ref.str = cstr;
		ref.length = 0;
		bool quoted = false;

		// Scan the input string for delimiters
		while (idx < len) {
			if (quoted) {
				// Parsing quoted token
				if (cstr[idx] == mQuote) {	// a quote inside quoted token
					if (++idx == len) {
						quoted = false;
					}
					else if (cstr[idx] == mDelimiter) {
						// Token ended - start another one
						if (ref.length > 0 || !skipEmpty)
							tokens.push_back(ref);				// save current token reference
						ref.str = cstr + idx + 1;
						ref.length = 0;
						quoted = false;
					}
					else if (cstr[idx] == mQuote) {
						// It is a double quote
						ref.length += 2;
					}
					else
						throw (bpp::TokenizerError() << "Unexpected end of quoted token at index " << idx << ".");
				}
				else
					++ref.length;	// add another character into the token
			}
			else {
				// Parsing unquoted token
				if (cstr[idx] == mDelimiter) {
					// Token ended - start another one
					if (ref.length > 0 || !skipEmpty)
						tokens.push_back(ref);				// save current token reference
					ref.str = cstr + idx + 1;
					ref.length = 0;
				}
				else if (cstr[idx] == mQuote && ref.length == 0) {
					quoted = true;	// first character is quote -> switch to quoted parsing
					++ref.str;		// the quote does not belong to the final token
				}
				else
					++ref.length;	// add another character into the token
			}
			++idx;		// another character happily parsed
		}

		if (quoted)
			throw bpp::TokenizerError("Quoted token ended unexpectedly.");

		if (ref.length > 0 || !skipEmpty)
			tokens.push_back(ref);		// save the last token
	}


	/**
	 * Helper function that replaces doubled-quotes with regular quotes in tokens.
	 */
	std::string removeDoubleQuotes(ref_t &ref)
	{
		std::string q(1, mQuote);
		std::string qq(2, mQuote);

		std::string str = ref.as<std::string>();
		size_t index = 0;
		while (true) {
			index = str.find(qq, index);
			if (index == std::string::npos) break;	// no more occurences
			str.replace(index, 2, q);				// replace current occurence
			++index;								// let us not replace already replaced quotes
		}
		return str;
	}
};


}

#endif
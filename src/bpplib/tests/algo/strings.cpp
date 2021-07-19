#include "../test.hpp"
#include <algo/strings.hpp>

#include <iostream>



/**
 * \brief Test strings algorithms.
 */
class BPPAlgoStringsSimpleTokenizerTest : public BPPLibTest
{
private:
	template<typename T>
	bool checkParsing(const std::string &str, const std::string &delim, bool skipEmpty, std::vector<T> correct) const
	{
		bpp::SimpleTokenizer tok;
		tok.setDelimiter(delim);
		std::vector<bpp::SimpleTokenizer::ref_t> tokens;
		tok.tokenize(str, tokens, skipEmpty);

		if (tokens.size() != correct.size()) {
			std::cout << "Invalid number of tokens (" << tokens.size() << " found, but " << correct.size() << " expected)." << std::endl;
			std::cout << "String: " << str << std::endl;
			std::cout << "Delimiter: '" << delim << "'" << std::endl;
			return false;
		}

		for (std::size_t i = 0; i < tokens.size(); ++i) {
			T token = tokens[i].as<T>();
			if (token != correct[i]) {
				std::cout << "Token[" << i << "] mismatch: " << token << " != " << correct[i] << std::endl;
				std::cout << "String: " << str << std::endl;
				std::cout << "Delimiter: '" << delim << "'" << std::endl;
				return false;
			}
		}
		return true;
	}

public:
	BPPAlgoStringsSimpleTokenizerTest() : BPPLibTest("algo/strings/simple_tokenizer") {}

	
	virtual bool run() const
	{
		// Date Time parsing test
		bpp::SimpleTokenizer tok;
		tok.setDelimiter(",");
		std::string dateStr("2017-05-29-18-44-42,29.5.2017 18:45:19");
		std::vector<bpp::SimpleTokenizer::ref_t> tokens;
		tok.tokenize(dateStr, tokens);
		if (tokens.size() != 2) {
			std::cout << "Datetime tokenization failed. Two tokens were expected." << std::endl;
			return false;
		}

		std::tm tm = tokens[0].asDateTime("%Y-%m-%d-%H-%M-%S");
		if (tm.tm_year+1900 != 2017 || tm.tm_mon+1 != 5 || tm.tm_mday != 29 || tm.tm_hour != 18 || tm.tm_min != 44 || tm.tm_sec != 42) {
			std::cout << "Datetime '20170529184442' was parsed into " << std::put_time(&tm, "%d.%m.%Y %H:%M:%S") << std::endl;
			return false;
		}
		tm = tokens[1].asDateTime();
		if (tm.tm_year+1900 != 2017 || tm.tm_mon+1 != 5 || tm.tm_mday != 29 || tm.tm_hour != 18 || tm.tm_min != 45 || tm.tm_sec != 19) {
			std::cout << "Datetime '20170529184442' was parsed into " << std::put_time(&tm, "%d.%m.%Y %H:%M:%S") << std::endl;
			return false;
		}

		// Normal tests
		return checkParsing<std::string>("foo bar", " ", false, { "foo", "bar" })
			&& checkParsing<std::string>("foo bar ", " ", false, { "foo", "bar", "" })
			&& checkParsing<std::string>(" foo bar", " ", false, { "", "foo", "bar" })
			&& checkParsing<std::string>("foo  bar", " ", false, { "foo", "", "bar" })
			&& checkParsing<std::string>(" foo   bar  ", " ", true, { "foo", "bar" })
			&& checkParsing<std::string>("foo<br>bar<br>egg", "<br>", false, { "foo", "bar", "egg" })
			&& checkParsing<std::string>("foo...bar...egg", "...", false, { "foo", "bar", "egg" })
			&& checkParsing<int>("1  2 3  42", " ", true, { 1, 2, 3, 42 })
			&& checkParsing<double>(" -3  0.1 4.2 ", " ", true, { -3.0, 0.1, 4.2 })
			;
	}
};




/**
* \brief Test strings algorithms.
*/
class BPPAlgoStringsCSVTokenizerTest : public BPPLibTest
{
private:
	bool checkParsing(const std::string &str, char delim, char quote, bool skipEmpty, std::vector<std::string> correct,
		bool exceptionExpected = false, bool removeDoubleQuotes = false) const
	{
		try {
			bpp::CSVTokenizer tok(delim, quote);
			std::vector<bpp::CSVTokenizer::ref_t> tokens;
			tok.tokenize(str, tokens, skipEmpty);
			if (delim == quote || exceptionExpected) {
				std::cout << "Exception was expected but not thrown." << std::endl;
				return false;
			}

			if (tokens.size() != correct.size()) {
				std::cout << "Invalid number of tokens (" << tokens.size() << " found, but " << correct.size() << " expected)." << std::endl;
				std::cout << "String: " << str << std::endl;
				std::cout << "Delimiter = '" << delim << "', Quote = '" << quote << "'" << std::endl;
				return false;
			}

			for (std::size_t i = 0; i < tokens.size(); ++i) {
				std::string token;
				if (removeDoubleQuotes)
					token = tok.removeDoubleQuotes(tokens[i]);
				else
					token = tokens[i].as<std::string>();
				if (token != correct[i]) {
					std::cout << "Token[" << i << "] mismatch: " << token << " != " << correct[i] << std::endl;
					std::cout << "String: " << str << std::endl;
					std::cout << "Delimiter: '" << delim << "'" << std::endl;
					return false;
				}
			}

			return true;
		}
		catch (bpp::TokenizerError &e) {
			if (delim == quote || exceptionExpected) return true;
			std::cout << "Unexpected tokenizer error: " << e.what() << std::endl;
			return false;
		}
		catch (bpp::RuntimeError &e) {
			std::cerr << "Unexpected runtime error: " << e.what() << std::endl;
			return false;
		}
	}

public:
	BPPAlgoStringsCSVTokenizerTest() : BPPLibTest("algo/strings/csv_tokenizer") {}


	virtual bool run() const
	{
		return checkParsing("foo,bar,spam", ',', '"', false, { "foo", "bar", "spam" })
			&& checkParsing("'foo','bar,bar','spam'", ',', '\'', false, { "foo", "bar,bar", "spam" })
			&& checkParsing("'foo',,'','spam'", ',', '\'', true, { "foo", "spam" })
			&& checkParsing("'foo','bar','spa''m'", ',', '\'', false, { "foo", "bar", "spa''m" })
			&& checkParsing("'foo','bar','spa''m'", ',', '\'', false, { "foo", "bar", "spa'm" }, false, true)
			&& checkParsing("'foo','bar','spa''''m'", ',', '\'', false, { "foo", "bar", "spa''m" }, false, true)
			&& checkParsing("foo;'bar','spam", ';', '\'', false, {}, true)
			&& checkParsing("foo;'bar','spa'm'", ';', '\'', false, {}, true)
			&& checkParsing("foo;'bar','spa'''m'", ';', '\'', false, {}, true)
			;
	}
};


BPPAlgoStringsSimpleTokenizerTest _algoStringsSimpleTokenizerTest;
BPPAlgoStringsCSVTokenizerTest _algoStringsCSVTokenizerTest;

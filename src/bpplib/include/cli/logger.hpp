/*
 * BPPLIB - CLI Logger
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 10.5.2018
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_CLI_LOGGER_HPP
#define BPPLIB_CLI_LOGGER_HPP

#include <misc/ptr_fix.hpp>

#include <vector>
#include <map>
#include <string>
#include <memory>
#include <iostream>
#include <sstream>

namespace bpp {


/**
 * Severity (priority) enum constants.
 */
enum class LogSeverity
{
	UNDEFINED = 0,
	FATAL = 1,
	ERROR = 2,
	WARNING = 3,
	INFO = 4,
	NOTICE = 5,
	DEBUG = 6,
	ANY = 7,		// any must be last, so we have the biggest number here
};


/**
 * A very simple implementation of logger with stream-like interface, which respects message severities and
 * allows buffering and smart size-restrictions imposed on the contents.
 */
class Logger
{
protected:
	/**
	 * Internal structure forming one data block of log.
	 */
	struct Block
	{
		LogSeverity severity;
		std::string data;
	};

	std::ostream &mDefaultSink;						///< Default ostream, where the log is being flushed.
	LogSeverity mSeverity;							///< Current severity (of data in accumulator).
	LogSeverity mMaxSeverity;						///< Severity restriction.
	std::size_t mMaxLength;							///< Log length restrictoin.

	std::stringstream mAccumulator;					///< String buffer where the data are accumulated.
	std::size_t mAccumulatorSize;                    ///< Number of chars in accumulator.
	std::vector<Block> mLog;						///< Log as a sequence of Blocks.
	std::map<LogSeverity, std::size_t> mLengths;	///< How many bytes has each severity level of the log.


	/**
	 * Flush current accumulator into another block in the log.
	 */
	void flushAccumulator()
	{
		if (mAccumulatorSize > 0) {
			// Something was accumulated ... save another block.
			std::string acu = mAccumulator.str();
			mLog.push_back(Block());
			mLog.back().severity = mSeverity;
			mLog.back().data = acu;
			mLengths[mSeverity] += acu.length();
			mAccumulator.str(std::string());
			mAccumulatorSize = 0;
		}
	}


	/**
	 * Compute maximal severity level and how many bytes of that last level can be written out.
	 * \param severity Severity limit. Used both as input (maximal severity limit examined) and
	 *		output (if the level needs adjustment, the value is modified).
	 * \return Length limit imposed on the severity level in the severity parameter. Levels with higher priority
	 *		are returned completely.
	 */
	std::size_t applySizeLimit(LogSeverity &severity)
	{
		std::size_t total = 0;
		for (std::size_t i = (std::size_t)LogSeverity::UNDEFINED; i < (std::size_t)severity; ++i) {
			total += mLengths[(LogSeverity)i];
			if (total >= mMaxLength) {		// max log length would be exceeded including current severity into output 
				severity = (LogSeverity)i;
				break;
			}
		}

		return (total <= mMaxLength) ? ~(std::size_t)0 : mMaxLength - (total - mLengths[(LogSeverity)severity]);
	}

public:
	Logger(std::ostream &defaultSink = std::cerr) : mDefaultSink(defaultSink),
		mSeverity(LogSeverity::UNDEFINED), mMaxSeverity(LogSeverity::ANY), mMaxLength(~(std::size_t)0) {}
	virtual ~Logger() {}

	/**
	 * Adjust current severity level. The adjustment also invokes internal accumulator flush.
	 */
	Logger& setSeverity(LogSeverity severity)
	{
		if (severity == mSeverity) return *this;

		flushAccumulator();
		mSeverity = severity;
		return *this;
	}

	Logger& fatal()		{ return setSeverity(LogSeverity::FATAL); }
	Logger& error()		{ return setSeverity(LogSeverity::ERROR); }
	Logger& warning()	{ return setSeverity(LogSeverity::WARNING); }
	Logger& info()		{ return setSeverity(LogSeverity::INFO); }
	Logger& notice()	{ return setSeverity(LogSeverity::NOTICE); }
	Logger& debug()		{ return setSeverity(LogSeverity::DEBUG); }


	/**
	 * Set internal limit for severity level. Only messages with this level and higher are written out.
	 */
	void restrictSeverity(LogSeverity maxSeverity)
	{
		mMaxSeverity = maxSeverity;
	}

	/**
	 * Impose a limit on the log size. The limit is applied in a smart way, so that messages
	 * with higher priorities are also prioritized in the output, but the actual order of all
	 * messages is preserved.
	 */
	void restrictSize(std::size_t maxLength)
	{
		mMaxLength = maxLength;
	}

	/**
	 * Add another message/value into the log under current severity.
	 * This function is also invoked by the << operator.
	 */
	template<typename T>
	Logger& write(const T& data)
	{
		mAccumulator << data;
		mAccumulatorSize = (std::size_t)mAccumulator.tellp();
		return *this;
	}

	/**
	 * Flush the log to the default sink.
	 */
	void flush()
	{
		flushAccumulator();
		LogSeverity limitSeverity = mMaxSeverity;
		std::size_t limitSize = applySizeLimit(limitSeverity);	// limitSize applies to limitSeverity level only, levels above are printed completely

		for (auto && block : mLog) {
			if (block.severity > limitSeverity) continue;
			if (block.severity == limitSeverity) {
				if (limitSize == 0) continue;
				if (limitSize < block.data.length()) {
					// Special case -- print only part of the block.
					bool newline = block.data.find("\n", limitSize) != std::string::npos;
					mDefaultSink.write(block.data.c_str(), limitSize - (newline ? 1 : 0));
					if (newline) mDefaultSink.write("\n", 1);
					limitSize = 0;
					continue;
				}
				else
					limitSize -= block.data.length();
			}
			mDefaultSink.write(block.data.c_str(), block.data.length());
		}
		
		clear();
	}


	/**
	 * Return number of chars logged so far.
	 * \param severity Applied severiry level. Only messages with this level and above are counted.
	 */
	std::size_t size(LogSeverity severity = LogSeverity::ANY) const
	{
		std::size_t size = 0;
		for (std::size_t i = (std::size_t)LogSeverity::UNDEFINED; i <= (std::size_t)severity; ++i) {
			LogSeverity s = (LogSeverity)i;
			auto it = mLengths.find(s);
			if (it != mLengths.end()) {
				size += it->second;
			}

			if (s == mSeverity) {
				size += mAccumulatorSize;
			}
		}
		return size;
	}


	/**
	 * Check whether the size of the log (for selected severity and above) is full (exceeds the size restriction).
	 * \param severity Applied severiry level. Only messages with this level and above are counted.
	 */
	bool isFull(LogSeverity severity = LogSeverity::ANY) const
	{
		return mMaxLength <= size(severity);
	}


	/**
	 * Empty the log, discard all data.
	 */
	void clear()
	{
		mLog.clear();
		mAccumulator.str(std::string());
		mAccumulatorSize = 0;
		mLengths.clear();
	}


	template<typename T>
	Logger& operator<<(const T& data)
	{
		return write<T>(data);
	}
};


template<>
Logger& Logger::operator<< <LogSeverity>(const LogSeverity& data)
{
	return setSeverity(data);
}



/**
 * Injectable singleton holder and wrapper for logger entitiy.
 */
Logger& log(std::unique_ptr<Logger> && logger = std::unique_ptr<Logger>())
{
	static std::unique_ptr<Logger> residentLogger;
	if (logger) {
		// Register new logger ...
		residentLogger = std::move(logger);
	}

	if (!residentLogger) {
		// No logger, construct default
		residentLogger = bpp::make_unique<Logger>();
	}

	return (*residentLogger.get()).setSeverity(LogSeverity::UNDEFINED);
}


}
#endif

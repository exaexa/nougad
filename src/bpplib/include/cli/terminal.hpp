/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 7.12.2015
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_CLI_TERMINAL_HPP
#define BPPLIB_CLI_TERMINAL_HPP

#include <misc/exception.hpp>

#ifdef _WIN32
	#define NOMINMAX
	#include <windows.h>

	// This macro is defined in wingdi.h, I do not want ERROR macro in my projects!
	#ifdef ERROR
	#undef ERROR
	#endif
#else
	#include <sys/ioctl.h>
	#include <stdio.h>
	#include <unistd.h>
	#include <errno.h>
#endif


namespace bpp
{


/**
 * \brief Get basic info about terminal window (i.e., the size).
 */
class TerminalInfo
{
private:
	std::size_t mWidth, mHeight;

	TerminalInfo(std::size_t width, std::size_t height)
		: mWidth(width), mHeight(height)
	{}


public:
	TerminalInfo() : mWidth(0), mHeight(0) {}


	static TerminalInfo get()
	{
#ifdef _WIN32
		CONSOLE_SCREEN_BUFFER_INFO csbi;
		HANDLE handle = GetStdHandle(STD_OUTPUT_HANDLE);
		if (handle == INVALID_HANDLE_VALUE)
			throw (bpp::RuntimeError() << "Unable to retrieve stdout handle. GetStdHandle function ended with error " << GetLastError());
		if (handle == nullptr)
			throw (bpp::RuntimeError() << "GetStdHandle returned nullptr. The application does not have standard handles associated.");
		if (!GetConsoleScreenBufferInfo(handle, &csbi))
			throw (bpp::RuntimeError() << "WinAPI function GetConsoleScreenBufferInfo ended with error code " << GetLastError());
		
		return TerminalInfo(
			(std::size_t)csbi.srWindow.Right - (std::size_t)csbi.srWindow.Left + (std::size_t)1,
			(std::size_t)csbi.srWindow.Bottom - (std::size_t)csbi.srWindow.Top + (std::size_t)1);
#else
		struct winsize ws_s;
		if (ioctl(STDOUT_FILENO, TIOCGWINSZ, &ws_s) == -1) {
			throw (bpp::RuntimeError() << "Unable to get terminal info, ioctl() call failed (errno == " << errno << ").");
		}
		return TerminalInfo((std::size_t)ws_s.ws_col, (std::size_t)ws_s.ws_row);
#endif
	}


	std::size_t width() const	{ return mWidth; }
	std::size_t height() const	{ return mHeight; }
};

}

#endif

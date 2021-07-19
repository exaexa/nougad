#include "../test.hpp"
#include <cli/terminal.hpp>

#include <iostream>



/**
* \brief Test terminal info functionality.
*/
class BPPCLITerminalInfoTest : public BPPLibTest
{
public:
	BPPCLITerminalInfoTest() : BPPLibTest("cli/terminal/info") {}


	virtual bool run() const
	{
		bpp::TerminalInfo info;
		try {
			info = bpp::TerminalInfo::get();
			std::cout << "Terminal: " << info.width() << "x" << info.height() << std::endl;
			return true;
		}
		catch (std::exception &e) {
			std::cout << "Error: " << e.what() << std::endl;
			return false;
		}
	}
};


BPPCLITerminalInfoTest _cliTerminalInfoTest;

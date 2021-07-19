#include "../test.hpp"
#include <cli/logger.hpp>
#include <misc/exception.hpp>

#include <iostream>
#include <sstream>
#include <string>

class BPPCLILoggerTest : public BPPLibTest
{
private:
	void check(std::stringstream &ref, std::stringstream &rec) const
	{
		bpp::log().flush();
		std::string recStr = rec.str();
		std::string refStr = ref.str();
		rec.str(std::string());
		ref.str(std::string());

		if (recStr != refStr)
			throw (bpp::RuntimeError() << "Expected: " << refStr << "\nConstructed: " << recStr);
	}

public:
	BPPCLILoggerTest() : BPPLibTest("cli/logger/basic") {}

	virtual bool run() const
	{
		try {
			std::stringstream rec, ref;

			bpp::log(bpp::make_unique<bpp::Logger>(rec));
			bpp::log() << "Some data: " << 42 << 54.0 << std::string("\n");
			ref << "Some data: " << 42 << 54.0 << std::string("\n");
			check(ref, rec);

			bpp::log().info() << "Info," << bpp::LogSeverity::WARNING << "Warn," << bpp::LogSeverity::ERROR << "Error";
			bpp::log().restrictSeverity(bpp::LogSeverity::WARNING);
			ref << "Warn," << "Error";
			check(ref, rec);

			bpp::log().info() << "Info," << bpp::LogSeverity::WARNING << "Warn," << bpp::LogSeverity::ERROR << "Error";
			bpp::log().restrictSeverity(bpp::LogSeverity::ANY);
			bpp::log().restrictSize(8);
			ref << "War" << "Error";
			check(ref, rec);

			bpp::log().info() << "Info\n" << bpp::LogSeverity::WARNING << "Warn\n" << bpp::LogSeverity::ERROR << "Error";
			bpp::log().restrictSeverity(bpp::LogSeverity::ANY);
			bpp::log().restrictSize(8);
			ref << "Wa\n" << "Error";
			check(ref, rec);

			bpp::log().info() << "Hello" << bpp::LogSeverity::ERROR << "Kitty";
			if (bpp::log().size() != 10)
				throw (bpp::RuntimeError() << "size() function returned " << bpp::log().size() << ", but 10 was expected.");

			if (bpp::log().size(bpp::LogSeverity::ERROR) != 5)
				throw (bpp::RuntimeError() << "size() function returned " << bpp::log().size(bpp::LogSeverity::ERROR) << ", but 5 was expected.");
		
			return true;
		}
		catch (bpp::RuntimeError &e) {
			std::cerr << e.what() << std::endl;
			return false;
		}
	}
};



BPPCLILoggerTest _cliLoggerBasicTest;

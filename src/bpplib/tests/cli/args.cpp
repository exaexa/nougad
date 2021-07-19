#include "../test.hpp"
#include <cli/args.hpp>
#include <misc/exception.hpp>
#include <misc/ptr_fix.hpp>

#include <iostream>


class BPPCLIArgsTest : public BPPLibTest
{
protected:
	static void processArgs(bpp::ProgramArguments &args, std::initializer_list<const char*> ta)
	{
		std::vector<const char*> testArgs = ta;
		int argc = (int)testArgs.size();
		const char **argv = &testArgs[0];
		args.process(argc, argv);
	}


public:
	BPPCLIArgsTest(const std::string &name) : BPPLibTest("cli/args/" + name) {}
};



/**
 * \brief Test basic commad line arguments parsing.
 */
class BPPCLIArgsBasicTest : public BPPCLIArgsTest
{
public:
	BPPCLIArgsBasicTest() : BPPCLIArgsTest("basic") {}


	virtual bool run() const
	{
		try {
			bpp::ProgramArguments args(1);
			args.registerArg<bpp::ProgramArguments::ArgInt>("int1", "", true, 0, 0, 1000);
			args.registerArg<bpp::ProgramArguments::ArgFloat>("float1", "", true);
			args.registerArg<bpp::ProgramArguments::ArgIntList>("intList", "", true, 0);
			args.registerArg<bpp::ProgramArguments::ArgString>("str", "", true);

			processArgs(args, {
				"BPPLib/test.exe",
				"--int1", "42",
				"--float1", "0.42",
				"--intList", "1k", "2M", "3G", "4T",
				"--str", "foo",
				"--", "output.file",
			});

			if (args.getArgInt("int1").getValue() != 42)
				throw (bpp::RuntimeError() << "Invalid value of int1 argument.");

			if (args.getValueInt("int1") != 42)
				throw (bpp::RuntimeError() << "Invalid value of int1 argument.");

			if (args.getArgFloat("float1").getValue() != 0.42)
				throw (bpp::RuntimeError() << "Invalid value of float1 argument.");

			if (args.getValueFloat("float1") != 0.42)
				throw (bpp::RuntimeError() << "Invalid value of float1 argument.");

			if (args.getArgIntList("intList").count() != 4 ||
				args.getArgIntList("intList").getValue(0) != 1024 ||
				args.getArgIntList("intList").getValue(1) != 2*1024*1024 ||
				args.getArgIntList("intList").getValue(2) != 3LL*1024*1024*1024 ||
				args.getArgIntList("intList").getValue(3) != 4LL*1024*1024*1024*1024)
				throw (bpp::RuntimeError() << "Invalid value of intList arguments.");

			if (args.getArgString("str").getValue() != "foo")
				throw (bpp::RuntimeError() << "Invalid value of str argument: " << args.getArgString("str").getValue());

			if (args[0] != "output.file")
				throw (bpp::RuntimeError() << "Invalid value of the unnamed argument.");
		}
		catch (bpp::ArgumentException &e) {
			std::cout << "ArgumentException: " << e.what() << std::endl;
			return false;
		}
		catch (std::exception &e) {
			std::cout << e.what() << std::endl;
			return false;
		}


		try {
			bpp::ProgramArguments args(1);
			args.allowSingleDash();
			args.registerArg<bpp::ProgramArguments::ArgInt>("int1", "", true, 0, 0, 1000);
			args.registerArg<bpp::ProgramArguments::ArgFloat>("float1", "", true);
			args.registerArg<bpp::ProgramArguments::ArgIntList>("intList", "", true, 0);
			args.registerArg<bpp::ProgramArguments::ArgString>("str", "", true);

			processArgs(args, {
				"BPPLib/test.exe",
				"-int1", "42",
				"-float1", "0.42",
				"--intList", "1k", "2M", "3G", "4T",
				"-str", "foo",
				"--", "output.file",
				});

			if (args.getArgInt("int1").getValue() != 42)
				throw (bpp::RuntimeError() << "Invalid value of int1 argument.");

			if (args.getValueInt("int1") != 42)
				throw (bpp::RuntimeError() << "Invalid value of int1 argument.");

			if (args.getArgFloat("float1").getValue() != 0.42)
				throw (bpp::RuntimeError() << "Invalid value of float1 argument.");

			if (args.getValueFloat("float1") != 0.42)
				throw (bpp::RuntimeError() << "Invalid value of float1 argument.");

			if (args.getArgIntList("intList").count() != 4 ||
				args.getArgIntList("intList").getValue(0) != 1024 ||
				args.getArgIntList("intList").getValue(1) != 2 * 1024 * 1024 ||
				args.getArgIntList("intList").getValue(2) != 3LL * 1024 * 1024 * 1024 ||
				args.getArgIntList("intList").getValue(3) != 4LL * 1024 * 1024 * 1024 * 1024)
				throw (bpp::RuntimeError() << "Invalid value of intList arguments.");

			if (args.getArgString("str").getValue() != "foo")
				throw (bpp::RuntimeError() << "Invalid value of str argument: " << args.getArgString("str").getValue());

			if (args[0] != "output.file")
				throw (bpp::RuntimeError() << "Invalid value of the unnamed argument.");
		}
		catch (bpp::ArgumentException &e) {
			std::cout << "ArgumentException: " << e.what() << std::endl;
			return false;
		}
		catch (std::exception &e) {
			std::cout << e.what() << std::endl;
			return false;
		}

		try {
			bpp::ProgramArguments args(1);
			args.registerArg_deprecated(bpp::make_unique<bpp::ProgramArguments::ArgInt>("int1", "", true, 0, 0, 1000));
			processArgs(args, {	"BPPLib/test.exe", "-int1", "42" });	// Exception expected
			return false;
		}
		catch (bpp::ArgumentException&) {
			return true;
		}
		catch (std::exception &e) {
			std::cout << e.what() << std::endl;
			return false;
		}
	}
};


BPPCLIArgsBasicTest _cliArgsBasicTest;

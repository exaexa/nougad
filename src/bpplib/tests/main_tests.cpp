#define _CRT_SECURE_NO_WARNINGS

#include "test.hpp"

#include <map>
#include <string>
#include <iostream>

using namespace std;

typedef const map<string, const BPPLibTest*> tests_t;

/**
 * Check that given test name is on the argument list.
 * More precisely, whether one of the given keywords (in arguments) is a substring of the name.
 */
bool on_list(const string &name, char *argv[])
{
	if (*argv == nullptr) return true;	// list is empty, let's accept everything ...
	while (*argv != nullptr && name.find(*argv) == string::npos) ++argv;
	return (*argv != nullptr);	// not null => match was found
}


int main(int argc, char *argv[])
{
	try {
		cout << "Testing BPP Lib ..." << endl;
	
		tests_t &tests = BPPLibTest::getTests();
		size_t errors = 0, executed = 0;
		for (tests_t::const_iterator it = tests.begin(); it != tests.end(); ++it) {
			if (!on_list(it->first, argv + 1)) continue;	// argv + 1 ... skip the application name

			++executed;
			cout << "TEST: " << it->first << endl;
			if (!it->second->run()) {
				cout << "FAILED!" << endl;
				++errors;
			}
			else
				cout << "OK" << endl;

			cout << "---------------------------------------- " << endl;
		}

		cout << "Total " << executed << " tests were executed." << endl;
		if (errors > 0)
			cout << "Some (" << errors << ") tests experienced errors!" << endl;
		else
			cout << "Tests completed successfully." << endl;

		return 0;
	}
	catch (std::exception &e) {
		cerr << "Uncaught exception: " << e.what() << endl;
		return 1;
	}
}

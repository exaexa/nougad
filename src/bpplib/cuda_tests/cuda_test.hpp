/*
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 19.8.2015
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_CUDA_TEST_HPP
#define BPPLIB_CUDA_TEST_HPP

#include <misc/exception.hpp>

#include <map>
#include <string>


/**
 * \brief Base class for all tests. It defines virtual interface (method run()) and
 *		naming/registration mechanism for the tests.
 */
class BPPCudaTest
{
private:
	typedef std::map<std::string, const BPPCudaTest*> registry_t;

	/**
	 * \brief Global test registry declared as static singleton.
	 */
	static registry_t& _getTests()
	{
		static registry_t tests;
		return tests;
	}


protected:
	/**
	 * \brief Name of the test. The name should be directly assigned by the derived constructor.
	 *
	 * Name should reflect the library header hierarchy and the local name of the test for selected header.
	 * (e.g., system/info/total_memory is a test of system/info.hpp header feature that retrieves total memory of the system)
	 */
	std::string mName;

public:
	/**
	 * \brief Constructor names and registers the object within the global naming registry.
	 */
	BPPCudaTest(const std::string &name) : mName(std::string("cuda/") + name)
	{
		registry_t &tests = _getTests();
		if (tests.find(mName) != tests.end())
			throw (bpp::RuntimeError() << "BPPCuda test named '" << mName << "' is already registered!");

		tests[mName] = this;
	}


	/**
	 * \brief Enforces virtual destructors and removes the object from the registry.
	 */
	virtual ~BPPCudaTest()
	{
		_getTests().erase(mName);
	}


	/**
	 * \brief Testing interface method that performs the test itself.
	 *		It can fail in two ways -- gently return false, or throw
	 *		an exception, which causes termination of the whole testing
	 *		process (immediate stop). Returns true, if the test was OK.
	 */
	virtual bool run() const = 0;


	/**
	 * \brief A way to access (read-only) the list of registered tests.
	 */
	static const std::map<std::string, const BPPCudaTest*>& getTests()
	{
		return _getTests();
	}
};


#endif

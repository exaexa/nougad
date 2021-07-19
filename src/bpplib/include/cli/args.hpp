/*
 * BPPLIB - CLI Args
 * Author: Martin Krulis <krulis@ksi.mff.cuni.cz>
 * Last Modification: 9.5.2018
 * License: CC 3.0 BY-NC (http://creativecommons.org/)
 */
#ifndef BPPLIB_CLI_ARGS_HPP
#define BPPLIB_CLI_ARGS_HPP

#include <system/filesystem.hpp>
#include <misc/exception.hpp>
#include <misc/ptr_fix.hpp>

#include <set>
#include <map>
#include <string>
#include <vector>
#include <algorithm>
#include <functional>
#include <limits>
#include <iostream>
#include <memory>

#include <cstdint>
#include <cctype>


namespace bpp
{


/**
 * \brief Specific exception thrown from argument processor when internal error occurs.
 */
class ArgumentException : public RuntimeError
{
public:
	ArgumentException() : RuntimeError() {}
	ArgumentException(const char *msg) : RuntimeError(msg) {}
	ArgumentException(const std::string &msg) : RuntimeError(msg) {}
	virtual ~ArgumentException() noexcept {}

	/*
	 * Overloading << operator that uses stringstream to append data to mMessage.
	 * Note that this overload is necessary so the operator returns object of exactly this class.
	 */
	template<typename T> ArgumentException& operator<<(const T &data)
	{
		RuntimeError::operator<<(data);
		return *this;
	}
};



/**
 * \brief Wrapper class that keeps specifications, constraints, and after processing
 *		also the values of program arguments.
 *
 * The wrapper can process both named arguments (with standard '-' or '--' prefix notation)
 * and nameless arguments (like a list of files to be processed). This argument processor
 * does not distinguish between short arguments like (-v) or long arguments like (--verbose).
 * The length of the name does not matter and it can be prefixed with one or two dashes.
 * Furthermore, arguments with value (int, float and string) are supported besides boolean arguments.
 */
class ProgramArguments
{
public:
	/**
	 * \brief Base class for internal argument-representing ojects.
	 */
	class ArgBase
	{
		friend class ProgramArguments;

	protected:
		ProgramArguments *mArgs;
		std::string mName;		///< Name of the argument (both for identification and parsing).
		std::string mComment;	///< Comment used for help printing.

		bool mMandatory;		///< Whether the argument must be present on the command line.
		bool mPresent;			///< True if the argumen was present on the command line (after parsing).

		std::set<std::string> mConflictsWith;	///< List of arguments with which this argument conflicts with.
		std::set<std::string> mRequiresAlso;	///< List of arguments that are required if this argument is present.


		/**
		 * \brief Internal method called by friend class to verify that all constraints
		 *		(mandatory, conflictsWith, requiresAlso) hold.
		 * \param arguments A map indexed by names of all arguments defined.
		 * \throws ArgumentException if any of the constraints are broken.
		 */
		void checkConstraints(const std::map<std::string, std::unique_ptr<ArgBase>> &arguments) const
		{
			if (isMandatory() && !isPresent())
				throw (ArgumentException() << "The argument '" << mName << "' is mandatory but missing.");

			// Constraint checks are done only if the argument is present.
			if (isPresent()) {
				// Check for collisions.
				for (auto it = mConflictsWith.begin(); it != mConflictsWith.end(); ++it) {
					if (arguments.find(*it) == arguments.end())
						throw (ArgumentException() << "Internal Error: Argument '" << mName
							<< "' has unspecified argument '" << *it << "' on its collision list.");

					if (arguments.find(*it)->second->isPresent())
						throw (ArgumentException() << "The argument '" << mName << "' conflicts with argument '" << *it << "'.");
				}

				// Check for requirements.
				for (auto it = mRequiresAlso.begin(); it != mRequiresAlso.end(); ++it) {
					if (arguments.find(*it) == arguments.end())
						throw (ArgumentException() << "Internal Error: Argument '" << mName
							<< "' has unspecified argument '" << *it << "' on its requirements list.");

					if (!arguments.find(*it)->second->isPresent())
						throw (ArgumentException() << "The argument '" << *it << "' is also required when '" << mName << "' was specified.");
				}
			}
		}


		/**
		 * \brief Abstract method that is derived by final classes to parse the argument
		 *		value and update internal status.
		 * \param argc Intermediate copy of argc which is updated during parsing process.
		 * \param argv Intermediate copy of argv which is updated during parsing process.
		 * \note If the arguments parses any values from the command line, the argc and
		 *		argv values should be updated accordingly.
		 */
		virtual void process(int &argc, const char **(&argv)) = 0;


	public:
		/**
		 * \brief Initialize the argument object.
		 * \param name Human-readable identification of the argument.
		 *		It is used for both programming and parsing.
		 *		The name must not start with '-'.
		 * \param comment Text commentary used for printing the documentation.
		 * \param mandatory Flag that indicates whether the argument must be present on commandline.
		 */
		ArgBase(const std::string &name, const std::string &comment, bool mandatory)
			: mArgs(nullptr), mName(name), mComment(comment), mMandatory(mandatory), mPresent(false)
		{
			if (mName.empty())
				throw (ArgumentException() << "Argument name must not be empty.");

			if (mName[0] == '-')
				throw (ArgumentException() << "Argument name must not start with dash '-'.");
		}

		// Enforce virtual destructor for descendants.
		virtual ~ArgBase() {}


		/**
		 * \brief Return the name of the argument (used both for access to its object
		 *		and to write it on command line).
		 * \note The name is without '-' prefix, but on the command line is written as
		 *		'-name' or '--name'.
		 */
		const std::string& getName() const { return mName; }


		/**
		 * \brief Get short description of the argument (e.g., for usage and help printing).
		 */
		const std::string& getComment() const { return mComment; }


		/**
		 * \brief Check whether the argument is mandatory (always must be present).
		 */
		bool isMandatory() const { return mMandatory; }


		/**
		 * \brief Check whether the argument was present in processed arguments.
		 */
		bool isPresent() const { return mPresent; }


		/**
		 * \brief Adds conflict constraint.
		 * \param argName Name of an argument that must not be present after processing
		 *		if this argument is present.
		 * \return Reference to self, so the conflictWith() calls may be chained.
		 * \note The user is responsible for the ambiguity of conflict and requirement constraints.
		 */
		ArgBase& conflictsWith(const std::string &argName)
		{
			mConflictsWith.insert(argName);
			return *this;
		}


		/**
		 * \brief Adds dependency constraint.
		 * \param argName Name of an argument that must be also present after processing
		 *		if this argument is present.
		 * \return Reference to self, so the requiresAlso() calls may be chained.
		 * \note The user is responsible for the ambiguity of conflict and requirement constraints.
		 */
		ArgBase& requiresAlso(const std::string &argName)
		{
			mRequiresAlso.insert(argName);
			return *this;
		}
	};



	/**
	 * \brief Argument without value (simple option switch).
	 *		The value is true if the argument is present and false otherwise.
	 * \note Switch options are never mandatory.
	 */
	class ArgBool : public ArgBase
	{
	public:
		typedef bool value_t;

	protected:
		virtual void process(int&, const char**&)
		{
			this->mPresent = true;
		}


	public:
		ArgBool(const std::string &name, const std::string &comment)
			: ArgBase(name, comment, false) {}

		value_t getValue() const
		{
			return isPresent();
		}
	};



	/**
	 * \brief Argument with one numerical value. The value is stored as int64.
	 */
	class ArgIntBase : public ArgBase
	{
	public:
		typedef std::int64_t value_t;

	protected:
		value_t mMin;	///< Range constraint for the value.
		value_t mMax;	///< Range constraint for the value.

		/**
		 * \brief Internal method that tests whether given string is valid
		 *		decimal representation of an int.
		 */
		bool isInt(const char *str) const
		{
			if (*str == '-') ++str;		// optional sign
			if (!*str) return false;
			while (*str && isdigit(*str))	// digits
				++str;
			if (*str == 'k' || *str == 'M' || *str == 'G' || *str == 'T')	// optional multiplier
				++str;
			return !*str;
		}


		value_t processInt(int &argc, const char **(&argv)) const
		{
			// Check whether the argument value is present and valid.
			if (argc == 0)
				throw (ArgumentException() << "Value of argument '" << this->getName() << "' is missing!");
			if (!isInt(argv[0]))
				throw (ArgumentException() << "Value '" << argv[0] << "' of argument '" << this->getName() << "' cannot be converted to int!");

			value_t multiplier = 1;
			const char *last = argv[0];
			while (last[1] != 0) ++last;
			switch (*last) {
			case 'k': multiplier = 1024;                               break;
			case 'M': multiplier = 1024 * 1024;                        break;
			case 'G': multiplier = 1024 * 1024 * 1024;                 break;
			case 'T': multiplier = 1024LL * 1024LL * 1024LL * 1024LL;  break;
			}

			value_t value = std::atol(argv[0]) * multiplier;

			// Verify constraints ...
			if (value < mMin)
				throw (ArgumentException() << "Value of argument '" << this->getName() << "' is below limits (" << mMin << ").");
			if (value > mMax)
				throw (ArgumentException() << "Value of argument '" << this->getName() << "' is above limits (" << mMax << ").");

			// Update parsing status.
			--argc; ++argv;
			return value;
		}

	public:
		ArgIntBase(const std::string &name, const std::string &comment, bool mandatory = false,
			value_t min = 0, value_t max = std::numeric_limits<value_t>::max())
			: ArgBase(name, comment, mandatory), mMin(min), mMax(max)
		{
			if (mMin > mMax)
				throw (ArgumentException() << "Argument '" << this->getName() << "' has invalid limits (min="
					<< mMin << ", max=" << mMax << ").");
		}
	};



	/**
	 * \brief Argument with one numerical value. The value is stored as int64.
	 */
	class ArgInt : public ArgIntBase
	{
	private:
		value_t mValue;	///< Parsed value of the argument.

	protected:
		virtual void process(int &argc, const char **(&argv))
		{
			mValue = this->processInt(argc, argv);
			this->mPresent = true;
		}

	public:
		ArgInt(const std::string &name, const std::string &comment, bool mandatory = false,
			value_t defaultValue = 0, value_t min = 0, value_t max = std::numeric_limits<value_t>::max())
			: ArgIntBase(name, comment, mandatory, min, max), mValue(defaultValue) {}

		value_t getValue() const
		{
			return mValue;
		}

		std::size_t getAsSize() const
		{
			static_assert(sizeof(std::uint64_t) >= sizeof(std::size_t), "std::size_t is larger than 64-bit unsigned int");

			if (getAsUint() > (std::uint64_t)std::numeric_limits<std::size_t>::max())
				throw (bpp::ArgumentException() << "Unable to convert int argument '" << this->getName() << "' to size_t.");
			return (std::size_t)mValue;
		}

		std::uint64_t getAsUint() const
		{
			if (mValue < 0)
				throw (bpp::ArgumentException() << "Unable to convert int argument '" << this->getName() << "' to 64-bit unsigned int.");
			return (std::uint64_t)mValue;
		}

		std::int32_t getAsInt32() const
		{
			if (mValue > (value_t)std::numeric_limits<std::int32_t>::max())
				throw (bpp::ArgumentException() << "Unable to convert int argument '" << this->getName() << "' to 32-bit int.");
			return (std::int32_t)mValue;
		}

		std::uint32_t getAsUint32() const
		{
			if (mValue > (value_t)std::numeric_limits<std::uint32_t>::max())
				throw (bpp::ArgumentException() << "Unable to convert int argument '" << this->getName() << "' to 32-bit unsigned int.");
			return (std::uint32_t)mValue;
		}
	};



	/**
	 * \brief Argument with list numerical value. The values are stored as int64.
	 * \note The values may be sequenced after the argument or the argument may be
	 *		present multiple times.
	 */
	class ArgIntList : public ArgIntBase
	{
	private:
		std::vector<value_t> mValues;	///< Parsed values of the argument.

	protected:
		virtual void process(int &argc, const char **(&argv))
		{
			if (!this->mPresent)
				mValues.clear();

			mValues.push_back(this->processInt(argc, argv));
			while (argc && (argv[0][0] != '-' || (argv[0][0] && argv[0][1] != '-' && !mArgs->isSingleDashAllowed())))
				mValues.push_back(this->processInt(argc, argv));
			this->mPresent = true;
		}

	public:
		ArgIntList(const std::string &name, const std::string &comment, bool mandatory = false,
			value_t min = 0, value_t max = std::numeric_limits<value_t>::max())
			: ArgIntBase(name, comment, mandatory, min, max) {}

		value_t getValue(std::size_t idx) const
		{
			return mValues[idx];
		}


		/**
		 * \brief Get number of retrieved values in the list.
		 */
		std::size_t count() const
		{
			return mValues.size();
		}


		/**
		 * \brief Push another default value at the end of the value list.
		 * \param value Value being added to the list.
		 * \return Reference to this argument object (for chaining method calls).
		 * \note The default value can be modified only before actual values has
		 *		been successfully parsed from the commandline.
		 */
		ArgIntList& addDefault(value_t value)
		{
			if (this->mPresent)
				throw (ArgumentException() << "Unable to modify default values of argument '" << this->mName
					<< "' when the actual values were parsed from the command line.");
			mValues.push_back(value);
			return *this;
		}
	};



	/**
	 * \brief Argument with one real number value. The value is stored as double.
	 */
	class ArgFloatBase : public ArgBase
	{
	public:
		typedef double value_t;

	protected:
		value_t mMin;	///< Range constraint for the value.
		value_t mMax;	///< Range constraint for the value.

		/**
		 * \brief Internal method that tests whether given string is valid
		 *		decimal representation of a float number.
		 */
		bool isFloat(const char *str) const
		{
			if (*str == '-') ++str;
			if (!*str) return false;
			while (*str && (isdigit(*str) || *str == '.'))
				++str;
			return !*str;
		}


		value_t processFloat(int &argc, const char **(&argv))
		{
			// Check whether the argument value is present and valid.
			if (argc == 0)
				throw (ArgumentException() << "Value of argument '" << this->getName() << "' is missing!");
			if (!isFloat(argv[0]))
				throw (ArgumentException() << "Value '" << argv[0] << "' of argument '" << this->getName() << "' cannot be converted to float!");

			// Get the value
			value_t value = atof(argv[0]);

			// Verify constraints ...
			if (value < mMin)
				throw (ArgumentException() << "Value of argument '" << this->getName() << "' is below limits (" << mMin << ").");
			if (value > mMax)
				throw (ArgumentException() << "Value of argument '" << this->getName() << "' is above limits (" << mMax << ").");

			// Update parsing status.
			--argc; ++argv;
			return value;
		}

	public:
		ArgFloatBase(const std::string &name, const std::string &comment, bool mandatory = false,
			value_t min = 0.0, value_t max = 1.0)
			: ArgBase(name, comment, mandatory), mMin(min), mMax(max)
		{
			if (mMin > mMax)
				throw (ArgumentException() << "Argument '" << this->getName() << "' has invalid limits (min="
					<< mMin << ", max=" << mMax << ").");
		}
	};



	/**
	 * \brief Argument with one real number value. The value is stored as double.
	 */
	class ArgFloat : public ArgFloatBase
	{
	private:
		value_t mValue;	///< Parsed value of the argument.

	protected:
		virtual void process(int &argc, const char **(&argv))
		{
			mValue = this->processFloat(argc, argv);
			this->mPresent = true;
		}

	public:
		ArgFloat(const std::string &name, const std::string &comment, bool mandatory = false,
			value_t defaultValue = 0.0, value_t min = 0.0, value_t max = 1.0)
			: ArgFloatBase(name, comment, mandatory, min, max), mValue(defaultValue)
		{}

		value_t getValue() const
		{
			return mValue;
		}
	};



	/**
	 * \brief Argument with list of real number values. The values are stored as double.
	 * \note The values may be sequenced after the argument or the argument may be
	 *		present multiple times.
	 */
	class ArgFloatList : public ArgFloatBase
	{
	private:
		std::vector<value_t> mValues;	///< Parsed values of the argument.

	protected:
		virtual void process(int &argc, const char **(&argv))
		{
			if (!this->mPresent)
				mValues.clear();

			mValues.push_back(this->processFloat(argc, argv));

			while (argc && (argv[0][0] != '-' || (argv[0][0] && argv[0][1] != '-' && !mArgs->isSingleDashAllowed())))
				mValues.push_back(this->processFloat(argc, argv));
			this->mPresent = true;
		}

	public:
		ArgFloatList(const std::string &name, const std::string &comment, bool mandatory = false,
			value_t min = 0.0, value_t max = 1.0)
			: ArgFloatBase(name, comment, mandatory, min, max)
		{}

		value_t getValue(std::size_t idx) const
		{
			return mValues[idx];
		}


		/**
		 * \brief Get number of retrieved values in the list.
		 */
		std::size_t count() const
		{
			return mValues.size();
		}


		/**
		 * \brief Push another default value at the end of the value list.
		 * \param value Value being added to the list.
		 * \return Reference to this argument object (for chaining method calls).
		 * \note The default value can be modified only before actual values has
		 *		been successfully parsed from the commandline.
		 */
		ArgFloatList& addDefault(value_t value)
		{
			if (this->mPresent)
				throw (ArgumentException() << "Unable to modify default values of argument '" << this->mName
					<< "' when the actual values were parsed from the command line.");
			mValues.push_back(value);
			return *this;
		}
	};



	/**
	 * \brief Argument with one string value.
	 */
	class ArgString : public ArgBase
	{
	public:
		typedef std::string value_t;

	protected:
		std::string mValue;	///< The value of the argument stored after parsing.

		virtual void process(int &argc, const char **(&argv))
		{
			if (argc == 0)
				throw (ArgumentException() << "Value of argument '" << this->getName() << "' is missing!");

			mValue = std::string(argv[0]);
			--argc; ++argv;

			this->mPresent = true;
		}

	public:
		ArgString(const std::string &name, const std::string &comment, bool mandatory = false, const char *defaultValue = "")
			: ArgBase(name, comment, mandatory), mValue(defaultValue)
		{
		}

		const std::string& getValue() const
		{
			return mValue;
		}
	};



	/**
	 * \brief Argument with one string value which must be from a predefined enumeration.
	 */
	class ArgEnum : public ArgString
	{
	public:
		typedef std::string value_t;

	private:
		std::string mNormalizedValue;
		std::set<std::string> mOptions;
		bool mCaseSensitive;

	protected:
		static std::string toLower(const std::string &str)
		{
			std::string s = str;
			std::transform(s.begin(), s.end(), s.begin(), (int(*)(int))std::tolower);
			return s;
		}

		void addOption(const std::string &str)
		{
			mOptions.insert(mCaseSensitive ? str : toLower(str));
		}

		virtual void process(int &argc, const char **(&argv))
		{
			ArgString::process(argc, argv);

			mNormalizedValue = (mCaseSensitive ? this->getValue() : toLower(this->getValue()));
			if (mOptions.find(mNormalizedValue) == mOptions.end())
				throw (ArgumentException() << "Invalid value '" << this->getValue() << "' for argument '" << this->getName() << "'.");
		}

	public:
		ArgEnum(const std::string &name, const std::string &comment, bool mandatory = false,
			bool caseSensitive = true, const char *defaultValue = "", std::initializer_list<std::string> options = {})
			: ArgString(name, comment, mandatory, defaultValue), mCaseSensitive(caseSensitive)
		{
			std::for_each(options.begin(), options.end(),
				std::bind(&ArgEnum::addOption, this, std::placeholders::_1));

			if (!this->getValue().empty())
				mOptions.insert(this->getValue());
		}


		/**
		 * \brief Add more enum options.
		 */
		void addOptions(std::initializer_list<std::string> options)
		{
			std::for_each(options.begin(), options.end(),
				std::bind(&ArgEnum::addOption, this, std::placeholders::_1));
		}


		/**
		 * \brief Checks that the loaded value is equal to given value taking
		 *		internal case-sensitive flag into account.
		 * \param value The value which is being tested to equality.
		 * \return True if the enum argument has the given value.
		 */
		bool is(const std::string &value)
		{
			std::string lower;
			if (!mCaseSensitive) {
				lower = toLower(value);
			}

			if (this->getValue() == (mCaseSensitive ? value : lower))
				return true;

			if (mOptions.find(mCaseSensitive ? value : lower) == mOptions.end())
				throw (ArgumentException() << "Value '" << value << "' is not a valid enum option of '"
					<< this->getName() << " argument.");

			return false;
		}
	};



	/**
	 * \brief Argument with list of string values.
	 */
	class ArgStringList : public ArgBase
	{
	private:
		std::vector<std::string> mValues;	///< The list of values of the argument stored after parsing.

	protected:
		virtual void process(int &argc, const char **(&argv))
		{
			if (argc == 0)
				throw (ArgumentException() << "Value of argument '" << this->getName() << "' is missing!");

			if (!this->mPresent)
				mValues.clear();

			mValues.push_back(argv[0]);
			--argc; ++argv;

			while (argc && (argv[0][0] != '-' || (argv[0][0] && argv[0][1] != '-' && !mArgs->isSingleDashAllowed()))) {
				mValues.push_back(argv[0]);
				--argc; ++argv;
			}

			this->mPresent = true;
		}

	public:
		ArgStringList(const std::string &name, const std::string &comment, bool mandatory = false)
			: ArgBase(name, comment, mandatory) {}

		const std::string& getValue(std::size_t idx) const
		{
			return mValues[idx];
		}


		/**
		 * \brief Get number of retrieved values in the list.
		 */
		std::size_t count() const
		{
			return mValues.size();
		}


		/**
		 * \brief Push another default value at the end of the value list.
		 * \param value Value being added to the list.
		 * \return Reference to this argument object (for chaining method calls).
		 * \note The default value can be modified only before actual values has
		 *		been successfully parsed from the commandline.
		 */
		ArgStringList& addDefault(const std::string &value)
		{
			if (this->mPresent)
				throw (ArgumentException() << "Unable to modify default values of argument '" << this->mName
					<< "' when the actual values were parsed from the command line.");
			mValues.push_back(value);
			return *this;
		}
	};


private:
	std::string mProgramName;	///< Name of the program taken as first record on command line.

	/**
	 * \brief Map of named arguments. The arguments are indexed by their names (no duplicates allowed).
	 */
	std::map<std::string, std::unique_ptr<ArgBase>> mArguments;

	std::size_t mNamelessMin;	///< Range constraing on number of nameless arguments.
	std::size_t mNamelessMax;	///< Range constraing on number of nameless arguments.
	std::vector<std::string> mNamelessArguments;	///< Values of nameless arguments.
	std::vector<std::string> mNamelessCaptions;		///< Captions of nameless arguments (for documentation).

	bool mAllowSingleDash;		///< Whether single dash is allowed as attribute prefix (normally, two dashes prefix long arguments)

	/**
	 * \brief Provides access to named argument object.
	 * \tparam Type of the argument object (must be ArgBase or derived class)
	 * \param name Name of the argument.
	 * \return Reference to argument object.
	 * \throws ArgumentException if the argument is not found or has invalid type.
	 */
	template<typename ARG_T>
	ARG_T& getArgTyped(const std::string &name)
	{
		if (mArguments.find(name) == mArguments.end())
			throw (ArgumentException() << "Argument '" << name << "' was not specified.");
		ARG_T *arg = dynamic_cast<ARG_T*>(mArguments.find(name)->second.get());
		if (arg == nullptr)
			throw (ArgumentException() << "Argument '" << name << "' type mismatch.");
		return *arg;
	}


	/**
	 * \brief Provides read-only access to named argument object.
	 * \tparam Type of the argument object (must be ArgBase or derived class)
	 * \param name Name of the argument.
	 * \return Reference to argument object.
	 * \throws ArgumentException if the argument is not found or has invalid type.
	 */
	template<typename ARG_T>
	const ARG_T& getArgTyped(const std::string &name) const
	{
		if (mArguments.find(name) == mArguments.end())
			throw (ArgumentException() << "Argument '" << name << "' was not specified.");
		const ARG_T *arg = dynamic_cast<const ARG_T*>(mArguments.find(name)->second.get());
		if (arg == nullptr)
			throw (ArgumentException() << "Argument '" << name << "' type mismatch.");
		return *arg;
	}


public:
	/**
	 * \brief Initialize the arguments container.
	 * \param namelessMin Minimal number of required nameless arguments.
	 * \param namelessMax Maximal number of allowed nameless arguments.
	 */
	ProgramArguments(std::size_t namelessMin = 0, std::size_t namelessMax = ~(std::size_t)0)
		: mNamelessMin(namelessMin), mNamelessMax(namelessMax), mAllowSingleDash(false)
	{
		if (namelessMin > namelessMax)
			throw (ArgumentException() << "Nameless arguments minimum (" << namelessMin << ") exceeds the maximum (" << namelessMax << ").");
	}


	/**
	 * Allow or disable whether single dash is acceptable for argument name prefix.
	 * Typically, two dashes are required for long argument names.
	 */
	ProgramArguments& allowSingleDash(bool allow = true)
	{
		mAllowSingleDash = allow;
		return *this;
	}


	/**
	 * Whether a single dash is acceptable for argument name prefix.
	 */
	bool isSingleDashAllowed() const { return mAllowSingleDash; }


	/**
	 * \brief Return name of the program as it was passed to command line.
	 */
	const std::string& getProgramName() const
	{
		return mProgramName;
	}


	/**
	 * \brief Typed accessor to argument object.
	 */
	ArgBase& getArg(const std::string &name)
	{
		return getArgTyped<ArgBase>(name);
	}


	/**
	 * \brief Typed accessor to argument object.
	 */
	const ArgBase& getArg(const std::string &name) const
	{
		return getArgTyped<ArgBase>(name);
	}


	// Macro that will help us define all the accessor methods.
#define ARG_ACCESSOR(TYPE) \
	Arg##TYPE& getArg##TYPE(const std::string &name) { \
		return getArgTyped<Arg##TYPE>(name); \
	} \
	const Arg##TYPE& getArg##TYPE(const std::string &name) const { \
		return getArgTyped<Arg##TYPE>(name); \
	}

#define ARG_ACCESSOR_FULL(TYPE) \
	ARG_ACCESSOR(TYPE) \
	Arg##TYPE::value_t getValue##TYPE(const std::string &name) const { \
		return getArgTyped<Arg##TYPE>(name).getValue(); \
	}


	// Definitions of typed accessors ...
	ARG_ACCESSOR_FULL(Bool)
	ARG_ACCESSOR_FULL(Int)
	ARG_ACCESSOR(IntList)
	ARG_ACCESSOR_FULL(Float)
	ARG_ACCESSOR(FloatList)
	ARG_ACCESSOR_FULL(String)
	ARG_ACCESSOR_FULL(Enum)
	ARG_ACCESSOR(StringList)


	/**
	 * \brief Nameless argument accessor.
	 */
	const std::string& operator[](std::size_t idx) const
	{
		if (idx >= mNamelessArguments.size())
			throw (ArgumentException() << "Unable to retieve nameless argument #" << idx << " (only " << mNamelessArguments.size() << " provided).");
		return mNamelessArguments[idx];
	}


	/**
	 * \brief Set a documentation caption of nameless argument.
	 * \param idx A zero-based index of the nameless argument. It must fall
	 *		into constrained range.
	 * \param caption A text content of the caption.
	 */
	void setNamelessCaption(std::size_t idx, const std::string &caption)
	{
		if (mNamelessCaptions.size() <= idx)
			mNamelessCaptions.resize(idx + 1);
		mNamelessCaptions[idx] = caption;
	}


	/**
	 * \brief Return the number of nameless arguments.
	 */
	std::size_t namelessCount() const
	{
		return mNamelessArguments.size();
	}


	/**
	 * \brief Register new named argument to the parser.
	 * \param arg The argument object to be registred.
	 *
	 * After registration the object becomes a property of ProgramArguments object,
	 * thus the called should not care nor attempt to destroy the arg object passed
	 * to this method. Argument name is acquired from the object itself and it must
	 * not collide with any other registered arguments.
	 */
	void registerArg_deprecated(std::unique_ptr<ArgBase> && arg)
	{
		if (!arg) {
			throw (RuntimeError("Argument object must not be nullptr."));
		}

		if (mArguments.find(arg->getName()) != mArguments.end()) {
			throw (ArgumentException() << "Attempting to register duplicate argument '" << arg->getName() << "'.");
		}
		arg->mArgs = this;
		mArguments[arg->getName()] = std::move(arg);
	}


	/**
	 * \brief Register new named argument to the parser.
	 * \param arg The argument object to be registred.
	 *
	 * After registration the object becomes a property of ProgramArguments object,
	 * thus the called should not care nor attempt to destroy the arg object passed
	 * to this method. Argument name is acquired from the object itself and it must
	 * not collide with any other registered arguments.
	 */
	template<typename ARG_T, typename... ARGS>
	void registerArg(const std::string &name, ARGS&&... args)
	{
		if (mArguments.find(name) != mArguments.end()) {
			throw (ArgumentException() << "Attempting to register duplicate argument '" << name << "'.");
		}

		auto arg = std::make_unique<ARG_T>(name, std::forward<ARGS>(args)...);
		arg->mArgs = this;
		mArguments[name] = std::move(arg);
	}


	/**
	 * \brief Parse the command line and fill registred named arguments and nameless argument values.
	 * \param argc A copy of a parameter of main() function.
	 * \param argv A copy of a parameter of main() function.
	 */
	void process(int argc, const char *argv[])
	{
		// Get program name.
		mProgramName.assign(argv[0]);
		--argc; ++argv;	// skip program name

		// Process named arguments.
		while (argc > 0 && argv[0] && argv[0][0] == '-' && (mAllowSingleDash || argv[0][1] == '-')) {
			// Get argument name.
			const char *name = argv[0] + 1;		// +1 ~ skip leading dash
			--argc; ++argv;

			if (name[0] == '-') ++name;			// skip second dash if present
			if (!name[0])
				break;	// empty name ~ end of named argument;

			// Find corresponding argument object.
			auto argIt = mArguments.find(name);
			if (argIt == mArguments.end())
				throw (ArgumentException() << "Unknown argument '" << name << "'.");

			// Process the argument value(s).
			argIt->second->process(argc, argv);
		}

		// Verify argument constraints.
		for (auto it = mArguments.begin(); it != mArguments.end(); ++it)
			it->second->checkConstraints(mArguments);

		// Process nameless arguments.
		while (argc > 0 && argv[0]) {
			mNamelessArguments.push_back(argv[0]);
			--argc; ++argv;
		}

		// Check nameless arguments size.
		if (mNamelessArguments.size() < mNamelessMin)
			throw (ArgumentException() << "At least " << mNamelessMin << " nameless arguments expected, only "
				<< mNamelessArguments.size() << " were found.");

		if (mNamelessArguments.size() > mNamelessMax)
			throw (ArgumentException() << "Too many nameless arguments (" << mNamelessArguments.size() << " found, "
				<< mNamelessMax << " is the limit).");
	}


	/**
	 * \brief Parse the command line and fill registred named arguments and nameless argument values.
	 * \param argc A copy of a parameter of main() function.
	 * \param argv A copy of a parameter of main() function.
	 * \note This overloaded function takes non-const char** as argv. The reason is that main() typically
	 *		declares char **argv without const and C++ does not allow automatic conversion from
	 *		char** to const char** as it may create a loophole for modifying constant chars.
	 *		However, we are only reading argv, so no harm is done.
	 */
	void process(int argc, char *argv[])
	{
		process(argc, (const char**)argv);
	}


	/**
	 * \brief Print simple usage generated from the argument specifications to std. output.
	 */
	void printUsage() const
	{
		printUsage(std::cout);
	}


	/**
	 * \brief Print simple usage generated from the argument specifications.
	 * \param stream Output where the usage is printed (usually either cout or cerr).
	 */
	void printUsage(std::ostream &stream) const
	{
		stream << "Usage: " << Path::getFileName(getProgramName()) << std::endl;

		stream << "Named arguments:" << std::endl;
		for (auto it = mArguments.begin(); it != mArguments.end(); ++it) {
			stream << "  " << it->first << " - " << it->second->getComment() << std::endl;
		}

		stream << "Nameless arguments (" << mNamelessMin << ", " << mNamelessMax << "):";
		for (std::size_t i = 0; i < mNamelessCaptions.size(); ++i) {
			stream << " " << mNamelessCaptions[i];
		}
		stream << std::endl;
	}
};


}
#endif

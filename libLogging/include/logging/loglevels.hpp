/*
 * loglevels.hpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LOGLEVELS_HPP_
#define LOGLEVELS_HPP_

#include <string>

using std::string;

namespace logging {

/**
 * The loglevels used.
 *
 * TODO is this ok here? Is there a better place? If we place it in the LoggerFactory or Logger, we have circular dependencies?
 */
enum class loglevel {
	PANIC,	// Use to log extreme situations where the entire application execution could be affected due to some cause.
	ERROR,	// Use to log all errors and exceptions that affect the functionality of the application.
	WARN,	// Use to log warnings, i.e. situations that are usually unexpected but do not significantly affect the functionality.
	INFO,	// Use to log statements which are expected to be written to the log during normal operations.
	DEBUG,	// Use to log messages helpful in debugging, but too verbose for normal situations.
	TRACE	// Use to log method invocation or parameters.
};


/**
 * Converts a loglevel to a string.
 *
 * @param[in] logLevel The loglevel that will be converted to a string.
 * @return The loglevel as a string.
 */
static string loglevelToString(loglevel logLevel)
{
	switch (logLevel)
	{
	case loglevel::PANIC:
		return string("PANIC");
		break;
	case loglevel::ERROR:
		return string("ERROR");
		break;
	case loglevel::WARN:
		return string("WARN");
		break;
	case loglevel::INFO:
		return string("INFO");
		break;
	case loglevel::DEBUG:
		return string("DEBUG");
		break;
	case loglevel::TRACE:
		return string("TRACE");
		break;
	default:
		return string("UNDEFINEDLOGLEVEL");
		break;
	}
}

} /* namespace logging */
#endif /* LOGLEVELS_HPP_ */

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
	PANIC,
	ERROR,
	WARN,
	INFO,
	DEBUG,
	TRACE
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

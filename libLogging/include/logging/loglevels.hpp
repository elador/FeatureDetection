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
enum loglevel {
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
	case logging::PANIC:
		return string("PANIC");
		break;
	case logging::ERROR:
		return string("ERROR");
		break;
	case logging::WARN:
		return string("WARN");
		break;
	case logging::INFO:
		return string("INFO");
		break;
	case logging::DEBUG:
		return string("DEBUG");
		break;
	case logging::TRACE:
		return string("TRACE");
		break;
	default:
		return string("");
		break;
	}
}

} /* namespace logging */
#endif /* LOGLEVELS_HPP_ */

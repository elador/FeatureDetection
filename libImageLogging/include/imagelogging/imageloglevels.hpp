/*
 * imageloglevels.hpp
 *
 *  Created on: 06.06.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef IMAGELOGLEVELS_HPP_
#define IMAGELOGLEVELS_HPP_

#include <string>

using std::string;

namespace imagelogging {

/**
 * The loglevels used.
 *
 * TODO is this ok here? Is there a better place? If we place it in the LoggerFactory or Logger, we have circular dependencies?
 */
enum class loglevel {
	FINAL,	// Use to log the final output image of some machinery. errors and exceptions that affect the functionality of the application.
	INTERMEDIATE,	// Use to log intermediate result steps of a more complex mechanism, that may be interesting in normal operation.
	INFO,	// Use to log informational images that are produced during operation but are not part of any result.
	DEBUG,	// Use to log images helpful in debugging, but too verbose for normal situations.
	TRACE	// Use to log debug images at the lowest levels.
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
	case loglevel::FINAL:
		return string("FINAL");
		break;
	case loglevel::INTERMEDIATE:
		return string("INTERMEDIATE");
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

} /* namespace imagelogging */
#endif /* IMAGELOGLEVELS_HPP_ */

/*
 * Appender.hpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef IAPPENDER_HPP_	// TODO change/rename
#define IAPPENDER_HPP_

#include "imagelogging/imageloglevels.hpp"
#include "opencv2/core/core.hpp"
#include <string>
#include <functional>

using cv::Mat;
using std::string;
using std::function;

namespace imagelogging {

/**
 * Extend this class to implement own strategies for printing log messages (e.g. to files or the console).
 *
 * Possible TODO: Make it possible to change the logLevel of an appender during program execution with
 *                like a setter or so. But the problem is, how to get the right Appender.
 */
class Appender {
public:

	/**
	 * Construct a new appender that logs at a certain loglevel.
	 * TODO: 1) Should this really go into the interface?
	 *       2) What about the default constructor, is one created? Should we disable it?
	 *
	 * @param[in] logLevel The log-level at which the appender should log. Required for all appenders.
	 */
	explicit Appender(loglevel logLevel) : logLevel(logLevel) {};

	virtual ~Appender() {};

	/**
	 * Logs a message into its target (e.g. the console or a file).
	 *
	 * @param[in] logLevel The log-level of the message.
	 * @param[in] loggerName The name of the logger that is logging the message.
	 * @param[in] logMessage The message to be logged.
	 */
	virtual void log(const loglevel logLevel, const string loggerName, const string filename, Mat image, function<void ()> functionToApply, const string filenameSuffix) = 0;	// const?

	/**
	 * Tests if this appender is actually doing logging at the given log-level.
	 *
	 * TODO This makes sense in the Appender, but what is actually being used in other logging frameworks
	 *      is that the logger itself, and not the appenders, is tested for isLogLevelEnabled(...).
	 *      But the log-level is different for every appender. So how do other loggers work around this?
	 *
	 * @param[in] logLevel The log-level to be tested for.
	 * @return True if the appender is logging at this level, false otherwise.
	 */
	bool isLogLevelEnabled(const loglevel logLevel) const {
		return (this->logLevel >= logLevel);
	};

	/**
	 * Get the log-level at which this appender is currently logging the messages.
	 *
	 * @return The log-level at which this appender is logging the messages.
	 */
	loglevel getLogLevel() const {
		return logLevel;
	};

	/**
	 * Set the log-level at which this appender should log the messages.
	 *
	 * @param[in] logLevel The log-level.
	 */
	void setLogLevel(const loglevel logLevel) {
		this->logLevel = logLevel;
	};

protected:
	loglevel logLevel;

};

} /* namespace imagelogging */
#endif /* IAPPENDER_HPP_ */

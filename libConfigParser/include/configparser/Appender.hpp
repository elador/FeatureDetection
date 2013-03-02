/*
 * Appender.hpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef APPENDER_HPP_
#define APPENDER_HPP_

#include "logging/loglevels.hpp"
#include <string>

using std::string;

namespace logging {

/**
 * Extend this class to implement own strategies for printing log messages (e.g. to files or the console).
 */
class Appender {
public:

	virtual ~Appender() {};

	/**
	 * Logs a message into its target (e.g. the console or a file).
	 *
	 * @param[in] logLevel The log-level of the message.
	 * @param[in] logMessage The message to be logged.
	 */
	virtual void log(const loglevel logLevel, const string logMessage) = 0;	// const?

	/**
	 * Tests if this appender is actually doing logging at the given log-level.
	 *
	 * @param[in] logLevel The log-level to be tested for.
	 * @return True if the appender is logging at this level, false otherwise.
	 */
	bool isLogLevelEnabled(const loglevel logLevel) const {
		return true;	// TODO
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

} /* namespace logging */
#endif /* APPENDER_HPP_ */

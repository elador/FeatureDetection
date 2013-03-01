/*
 * Appender.hpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef APPENDER_HPP_
#define APPENDER_HPP_

#include <string>

using std::string;

namespace logging {

/**
 * Extend this class to implement own strategies for printing log messages (e.g. to files or the console).
 */
class Appender {
public:

	virtual ~Appender() {}

	/**
	 * Logs a message into its target (e.g. the console or a file).
	 *
	 * @param[in] logLevel The log-level of the message.
	 * @param[in] logMessage The message to be logged.
	 */
	virtual void log(const int logLevel, const string logMessage) = 0;	// const?

	/**
	 * Get the log-level at which this appender is currently logging the messages.
	 *
	 * @return The log-level at which this appender is logging the messages.
	 */
	int getLogLevel() const {
		return logLevel;
	};

	/**
	 * Set the log-level at which this appender should log the messages.
	 *
	 * @param[in] logLevel The log-level.
	 */
	void setLogLevel(const int logLevel) {
		this->logLevel = logLevel;
	};

private:
	int logLevel;

};

} /* namespace logging */
#endif /* APPENDER_HPP_ */

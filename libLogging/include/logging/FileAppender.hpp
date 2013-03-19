/*
 * FileAppender.hpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FILEAPPENDER_HPP_
#define FILEAPPENDER_HPP_

#include "logging/Appender.hpp"
#include <fstream>

using std::ofstream;

namespace logging {

/**
 * An appender that logs all messages equal or below its log-level to a text file.
 * The messages are appended to the log-file if the file already exists.
 */
class FileAppender : public Appender {
public:

	/**
	 * Constructs a new appender that logs to a file. Appends to the file if it already exists.
	 *
	 * param[in] loglevel The loglevel at which to log.
	 * param[in] filename The full path to the file to log to.
	 */
	FileAppender(loglevel logLevel, string filename);

	~FileAppender();

	/**
	 * Appends a message to the opened file.
	 *
	 * @param[in] logLevel The log-level of the message.
	 * @param[in] loggerName The name of the logger that is logging the message.
	 * @param[in] logMessage The log-message itself.
	 */
	void log(const loglevel logLevel, const string loggerName, const string logMessage);

private:

	/**
	 * Creates a new string containing the formatted current time.
	 *
	 * @return The formatted time.
	 */
	string getCurrentTime();

	// TODO: We should make the copy constructor (and assignment operator?) private because we have an ofstream as member variable! Read that somewhere on stackoverflow.
	ofstream file;
};

} /* namespace logging */
#endif /* FILEAPPENDER_HPP_ */

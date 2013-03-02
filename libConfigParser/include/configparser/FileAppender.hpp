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
	 * Constructs a new file appender with default loglevel INFO.
	 *
	 * param[in] filename The full path to the file to log to.
	 */
	FileAppender(string filename);

	~FileAppender();

	/**
	 * Appends a message to the opened file.
	 *
	 * @param[in] logLevel The log-level of the message.
	 * @param[in] logMessage The log-message itself.
	 */
	void log(const loglevel logLevel, const string logMessage);

private:
	ofstream file;
};

} /* namespace logging */
#endif /* FILEAPPENDER_HPP_ */

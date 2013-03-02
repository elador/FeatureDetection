/*
 * FileAppender.cpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */

#include "logging/FileAppender.hpp"
#include "logging/loglevels.hpp"
#include <ios>
#include <iostream>

using std::ios_base;

namespace logging {

FileAppender::FileAppender(string filename)
{
	logLevel = loglevel::INFO;

	file.open(filename, std::ios::out | std::ios::app);
	if (!file.is_open()) {
		throw ios_base::failure("Error: Could not open or create log-file: " + filename);
	}
	file << "Timestamp Starting logging to " << filename << std::endl;
}

FileAppender::~FileAppender()
{
	file.close();
}

void FileAppender::log(const loglevel logLevel, const string logMessage)
{
	if(logLevel <= this->logLevel) {
		file << "[" << loglevelToString(logLevel) << "] " << logMessage << std::endl;
	}
}

} /* namespace logging */

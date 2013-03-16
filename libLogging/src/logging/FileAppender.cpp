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
#include <iomanip>
#include <chrono>

using std::ios_base;
using std::chrono::system_clock;

namespace logging {

FileAppender::FileAppender(loglevel logLevel, string filename) : Appender(logLevel)
{
	file.open(filename, std::ios::out | std::ios::app);
	if (!file.is_open()) {
		throw ios_base::failure("Error: Could not open or create log-file: " + filename);
	}
	system_clock::time_point now = system_clock::now();
	std::time_t now_c = system_clock::to_time_t(now);
	file << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") << " Starting logging to " << filename << std::endl;
	// TODO We should really also output the loggerName here! So... maybe make a member variable that holds a reference to the appenders parent logger?
}

FileAppender::~FileAppender()
{
	file.close();
}

void FileAppender::log(const loglevel logLevel, const string loggerName, const string logMessage)
{
	if(logLevel <= this->logLevel) {
		system_clock::time_point now = system_clock::now();
		std::time_t now_c = system_clock::to_time_t(now);
		file << std::put_time(std::localtime(&now_c), "%Y-%m-%d %H:%M:%S") << " [" << loglevelToString(logLevel) << "] " << "[" << loggerName << "] " << logMessage << std::endl;
		// TODO	- maybe use setw()
		//		- Maybe we want to use miliseconds for the file-logging?
		//		- The windows header seems to behave very strange, certain %F... etc don't seem to work (runtime-error).
	}
}

} /* namespace logging */

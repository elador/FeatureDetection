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
#include <sstream>
#include <iomanip>
#ifdef WIN32
	#include "wingettimeofday.h"
#else
	#include <sys/time.h>
#endif

using std::ios_base;
using std::ostringstream;

namespace logging {

FileAppender::FileAppender(loglevel logLevel, string filename) : Appender(logLevel)
{
	file.open(filename, std::ios::out | std::ios::app);
	if (!file.is_open())
		throw ios_base::failure("Error: Could not open or create log-file: " + filename);
	file << getCurrentTime() << " Starting logging at log-level " << loglevelToString(logLevel) << " to file " << filename << std::endl;
	// TODO We should really also output the loggerName here! So... maybe make a member variable that holds a reference to the appenders parent logger?
}

FileAppender::~FileAppender()
{
	file.close();
}

void FileAppender::log(const loglevel logLevel, const string loggerName, const string logMessage)
{
	if(logLevel <= this->logLevel)
		file << getCurrentTime() << ' ' << loglevelToString(logLevel) << ' ' << "[" << loggerName << "] " << logMessage << std::endl;
}

string FileAppender::getCurrentTime()
{
	timeval time;
	gettimeofday(&time, 0);
	struct tm* tm_time = std::localtime(&time.tv_sec);
	ostringstream os;
	os << std::setfill('0') << std::setw(2);
	os << (1900 + tm_time->tm_year) << '-' << (1 + tm_time->tm_mon) << '-' << tm_time->tm_mday << ' ';
	os << tm_time->tm_hour << ':' << tm_time->tm_min << ':' << tm_time->tm_sec << '.';
	os << std::setw(3);
	os << (time.tv_usec / 1000);
	return os.str();
}

} /* namespace logging */

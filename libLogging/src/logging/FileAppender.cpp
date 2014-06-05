/*
 * FileAppender.cpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */

#include "logging/FileAppender.hpp"
#include "logging/LogLevels.hpp"
#include <ios>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstdint>

using std::string;
using std::ios_base;
using std::ostringstream;
using std::chrono::system_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

namespace logging {

FileAppender::FileAppender(LogLevel logLevel, string filename) : Appender(logLevel)
{
	file.open(filename, std::ios::out | std::ios::app);
	if (!file.is_open())
		throw ios_base::failure("Error: Could not open or create log-file: " + filename);
	file << getCurrentTime() << " Starting logging at log-level " << logLevelToString(logLevel) << " to file " << filename << std::endl;
	// TODO We should really also output the loggerName here! So... maybe make a member variable that holds a reference to the appenders parent logger?
}

FileAppender::~FileAppender()
{
	file.close();
}

void FileAppender::log(const LogLevel logLevel, const string loggerName, const string logMessage)
{
	if(logLevel <= this->logLevel)
		file << getCurrentTime() << ' ' << logLevelToString(logLevel) << ' ' << "[" << loggerName << "] " << logMessage << std::endl;
}

string FileAppender::getCurrentTime()
{
	system_clock::time_point now = system_clock::now();
	duration<int64_t, std::ratio<1>> seconds = duration_cast<duration<int64_t, std::ratio<1>>>(now.time_since_epoch());
	duration<int64_t, std::milli> milliseconds = duration_cast<duration<int64_t, std::milli>>(now.time_since_epoch());
	duration<int64_t, std::milli> msec = milliseconds - seconds;

	std::time_t t_now = system_clock::to_time_t(now);
	struct tm* tm_now = std::localtime(&t_now);
	ostringstream os;
	os.fill('0');
	os << (1900 + tm_now->tm_year) << '-' << std::setw(2) << (1 + tm_now->tm_mon) << '-' << std::setw(2) << tm_now->tm_mday;
	os << ' ' << std::setw(2) << tm_now->tm_hour << ':' << std::setw(2) << tm_now->tm_min << ':' << std::setw(2) << tm_now->tm_sec;
	os << '.' << std::setw(3) << msec.count();
	return os.str();
}

} /* namespace logging */

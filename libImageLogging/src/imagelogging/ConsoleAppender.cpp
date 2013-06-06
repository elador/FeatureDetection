/*
 * ConsoleAppender.cpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */

#include "imagelogging/ConsoleAppender.hpp"
#include "imagelogging/imageloglevels.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <cstdint>

using std::cout;
using std::ostringstream;
using std::chrono::system_clock;
using std::chrono::duration;
using std::chrono::duration_cast;

namespace imagelogging {

ConsoleAppender::ConsoleAppender(loglevel logLevel) : Appender(logLevel) {}

ConsoleAppender::~ConsoleAppender() {}

void ConsoleAppender::log(const loglevel logLevel, const string loggerName, const string logMessage)
{
	if(logLevel <= this->logLevel)
		cout << getCurrentTime() << ' ' << loglevelToString(logLevel) << ' ' << "[" << loggerName << "] " << logMessage << std::endl;
}

string ConsoleAppender::getCurrentTime()
{
	system_clock::time_point now = system_clock::now();
	duration<int64_t, std::ratio<1>> seconds = duration_cast<duration<int64_t, std::ratio<1>>>(now.time_since_epoch());
	duration<int64_t, std::milli> milliseconds = duration_cast<duration<int64_t, std::milli>>(now.time_since_epoch());
	duration<int64_t, std::milli> msec = milliseconds - seconds;

	std::time_t t_now = system_clock::to_time_t(now);
	struct tm* tm_now = std::localtime(&t_now);
	ostringstream os;
	os.fill('0');
	os << std::setw(2) << tm_now->tm_hour << ':' << std::setw(2) << tm_now->tm_min << ':' << std::setw(2) << tm_now->tm_sec;
	os << '.' << std::setw(3) << msec.count();
	return os.str();
}

} /* namespace imagelogging */

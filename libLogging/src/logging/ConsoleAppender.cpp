/*
 * ConsoleAppender.cpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */

#include "logging/ConsoleAppender.hpp"
#include "logging/loglevels.hpp"
#include <iostream>
#include <sstream>
#include <iomanip>
#ifdef WIN32
	#include "wingettimeofday.h"
#else
	#include <sys/time.h>
#endif

using std::cout;
using std::ostringstream;

namespace logging {

ConsoleAppender::ConsoleAppender(loglevel logLevel) : Appender(logLevel) {}

ConsoleAppender::~ConsoleAppender() {}

void ConsoleAppender::log(const loglevel logLevel, const string loggerName, const string logMessage)
{
	if(logLevel <= this->logLevel)
		cout << getCurrentTime() << ' ' << loglevelToString(logLevel) << ' ' << "[" << loggerName << "] " << logMessage << std::endl;
}

string ConsoleAppender::getCurrentTime()
{
	timeval time;
	gettimeofday(&time, 0);
	struct tm* tm_time = std::localtime(&time.tv_sec);
	ostringstream os;
	os << std::setfill('0') << std::setw(2);
	os << tm_time->tm_hour << ':' << tm_time->tm_min << ':' << tm_time->tm_sec << '.';
	os << std::setw(3);
	os << (time.tv_usec / 1000);
	return os.str();
}

} /* namespace logging */

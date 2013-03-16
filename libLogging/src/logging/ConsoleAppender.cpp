/*
 * ConsoleAppender.cpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */

#include "logging/ConsoleAppender.hpp"
#include "logging/loglevels.hpp"
#include <iostream>
#include <iomanip>
#include <chrono>

using std::cout;
using std::chrono::system_clock;

namespace logging {

ConsoleAppender::ConsoleAppender(loglevel logLevel) : Appender(logLevel) {}

ConsoleAppender::~ConsoleAppender() {}

void ConsoleAppender::log(const loglevel logLevel, const string loggerName, const string logMessage)
{
	if(logLevel <= this->logLevel) {
		system_clock::time_point now = system_clock::now();
		std::time_t now_c = system_clock::to_time_t(now);
		cout << std::put_time(std::localtime(&now_c), "%H:%M:%S") << " [" << loglevelToString(logLevel) << "] " << "[" << loggerName << "] " << logMessage << std::endl;
		// TODO	- maybe use setw()
		//		- The windows header seems to behave very strange, certain %F... etc don't seem to work (runtime-error).
	}
}

} /* namespace logging */

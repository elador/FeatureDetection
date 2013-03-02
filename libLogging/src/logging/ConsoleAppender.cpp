/*
 * ConsoleAppender.cpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */

#include "logging/ConsoleAppender.hpp"
#include "logging/loglevels.hpp"
#include <iostream>

using std::cout;

namespace logging {

ConsoleAppender::ConsoleAppender()
{
	logLevel = loglevel::INFO;
}

ConsoleAppender::~ConsoleAppender() {}

void ConsoleAppender::log(const loglevel logLevel, const string logMessage)
{
	if(logLevel <= this->logLevel) {
		cout << "[" << loglevelToString(logLevel) << "] " << logMessage << std::endl;
	}
}

} /* namespace logging */

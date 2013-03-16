/*
 * Logger.cpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */

#include "logging/Logger.hpp"
#include "logging/Appender.hpp"

namespace logging {

Logger::Logger(string name) : name(name) {}

Logger::~Logger() {}

void Logger::log(const loglevel logLevel, string logMessage)
{
	for (shared_ptr<Appender> appender : appenders) {
		appender->log(logLevel, name, logMessage);
	}
}

void Logger::addAppender(shared_ptr<Appender> appender)
{
	appenders.push_back(appender);
}

} /* namespace logging */

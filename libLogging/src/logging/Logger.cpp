/*
 * Logger.cpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */

#include "logging/Logger.hpp"
#include "logging/Appender.hpp"

namespace logging {

Logger::Logger(string name) : name(name), appenders() {}

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

void Logger::trace(const string logMessage)
{
	log(loglevel::TRACE, logMessage);
}

void Logger::debug(const string logMessage)
{
	log(loglevel::DEBUG, logMessage);
}

void Logger::info(const string logMessage)
{
	log(loglevel::INFO, logMessage);
}

void Logger::warn(const string logMessage)
{
	log(loglevel::WARN, logMessage);
}

void Logger::error(const string logMessage)
{
	log(loglevel::ERROR, logMessage);
}

void Logger::panic(const string logMessage)
{
	log(loglevel::PANIC, logMessage);
}

} /* namespace logging */

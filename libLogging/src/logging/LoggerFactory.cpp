/*
 * LoggerFactory.cpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */

#include "logging/LoggerFactory.hpp"

namespace logging {

LoggerFactory::LoggerFactory(void)
{
}


LoggerFactory::~LoggerFactory(void)
{
}

LoggerFactory* LoggerFactory::Instance(void)
{
	static LoggerFactory instance;
	return &instance;
}

} /* namespace logging */

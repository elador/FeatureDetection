/*
 * LoggerFactory.cpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */

#include "logging/LoggerFactory.hpp"

namespace logging {

LoggerFactory::LoggerFactory()
{
}


LoggerFactory::~LoggerFactory()
{
}

LoggerFactory* LoggerFactory::Instance()
{
	static LoggerFactory instance;
	return &instance;
}

} /* namespace logging */

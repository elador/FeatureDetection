/*
 * FileAppender.cpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */

#include "logging/FileAppender.hpp"
#include "logging/loglevels.hpp"

namespace logging {

FileAppender::FileAppender()
{
	logLevel = loglevel::INFO;
}

FileAppender::~FileAppender()
{

}

} /* namespace logging */

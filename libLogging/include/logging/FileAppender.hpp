/*
 * FileAppender.hpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FILEAPPENDER_HPP_
#define FILEAPPENDER_HPP_

#include "logging/Appender.hpp"

namespace logging {

/**
 * An appender that logs all messages equal or below its log-level to a text file.
 */
class FileAppender : public Appender {
public:

	/**
	 * Constructs a new file appender with default loglevel INFO.
	 */
	FileAppender();

	~FileAppender();
};

} /* namespace logging */
#endif /* FILEAPPENDER_HPP_ */

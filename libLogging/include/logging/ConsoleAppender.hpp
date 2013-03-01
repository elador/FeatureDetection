/*
 * ConsoleAppender.hpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef CONSOLEAPPENDER_HPP_
#define CONSOLEAPPENDER_HPP_

#include "logging/Appender.hpp"

namespace logging {

/**
 * An appender that logs all messages equal or above its log-level to the text console.
 */
class ConsoleAppender : public Appender {
public:

	/**
	 * Constructs a new console appender.
	 */
	ConsoleAppender();

	~ConsoleAppender();

};

} /* namespace logging */
#endif /* CONSOLEAPPENDER_HPP_ */

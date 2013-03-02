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
#include "logging/loglevels.hpp"

namespace logging {

/**
 * An appender that logs all messages equal or below its log-level to the text console.
 */
class ConsoleAppender : public Appender {
public:

	/**
	 * Constructs a new console appender with default loglevel INFO.
	 */
	ConsoleAppender();

	~ConsoleAppender();

	/**
	 * Logs a message to the text console.
	 *
	 * @param[in] logLevel The log-level of the message.
	 * @param[in] logMessage The log-message itself.
	 */
	void log(const loglevel logLevel, const string logMessage);

};

} /* namespace logging */
#endif /* CONSOLEAPPENDER_HPP_ */

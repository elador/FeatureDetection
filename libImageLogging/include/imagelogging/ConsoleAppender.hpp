/*
 * ConsoleAppender.hpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef CONSOLEAPPENDER_HPP_
#define CONSOLEAPPENDER_HPP_

#include "imagelogging/Appender.hpp"
#include "imagelogging/imageloglevels.hpp"

namespace imagelogging {

/**
 * An appender that logs all messages equal or below its log-level to the text console.
 */
class ConsoleAppender : public Appender {
public:

	/**
	 * Constructs a new appender that logs to the console.
	 * TODO: This is the same as the one in our base class. Why do we
	 *			need to define this "again" here? Probably because if not,
	 *			the compiler generates a ConsoleAppender() default constructor?
	 *
	 * param[in] loglevel The loglevel at which to log.
	 */
	ConsoleAppender(loglevel logLevel);

	~ConsoleAppender();

	/**
	 * Logs a message to the text console.
	 *
	 * @param[in] logLevel The log-level of the message.
	 * @param[in] loggerName The name of the logger that is logging the message.
	 * @param[in] logMessage The log-message itself.
	 */
	void log(const loglevel logLevel, const string loggerName, const string logMessage);

private:

	/**
	 * Creates a new string containing the formatted current time.
	 *
	 * @return The formatted time.
	 */
	string getCurrentTime();

};

} /* namespace imagelogging */
#endif /* CONSOLEAPPENDER_HPP_ */

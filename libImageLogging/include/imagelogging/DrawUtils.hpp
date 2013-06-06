/*
 * Logger.hpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LOGGER_HPP_
#define LOGGER_HPP_

#include "logging/loglevels.hpp"
#include <string>
#include <vector>
#include <memory>

using std::string;
using std::vector;
using std::shared_ptr;

namespace logging {

class Appender;

/**
 * A logger that is e.g. responsible for a certain class or namespace. A logger can log to several outputs (appenders).
 *
 * Possible TODO: Instead of using a string, make the logger usable like an ostream, like: "log[WARN] << "hello" << var << endl;".
 */
class Logger {
public:

	/**
	 * Constructs a new logger that is not logging anything yet.
	 *
	 * @param[in] name The name of the logger. Usually the namespace of the library it is working in.
	 */
	explicit Logger(string name);

	~Logger();

	/**
	 * Adds an appender to this logger.
	 *
	 * @param[in] The new appender.
	 */
	void addAppender(shared_ptr<Appender> appender);

	/**
	 * Logs a message with log-level TRACE to all appenders (e.g. the console or a file).
	 *
	 * @param[in] logMessage The message to be logged.
	 */
	void trace(const string logMessage);

	/**
	 * Logs a message with log-level DEBUG to all appenders (e.g. the console or a file).
	 *
	 * @param[in] logMessage The message to be logged.
	 */
	void debug(const string logMessage);

	/**
	 * Logs a message with log-level INFO to all appenders (e.g. the console or a file).
	 *
	 * @param[in] logMessage The message to be logged.
	 */
	void info(const string logMessage);

	/**
	 * Logs a message with log-level WARN to all appenders (e.g. the console or a file).
	 *
	 * @param[in] logMessage The message to be logged.
	 */
	void warn(const string logMessage);

	/**
	 * Logs a message with log-level ERROR to all appenders (e.g. the console or a file).
	 *
	 * @param[in] logMessage The message to be logged.
	 */
	void error(const string logMessage);

	/**
	 * Logs a message with log-level PANIC to all appenders (e.g. the console or a file).
	 *
	 * @param[in] logMessage The message to be logged.
	 */
	void panic(const string logMessage);

private:
	vector<shared_ptr<Appender>> appenders;
	string name;

	/**
	 * Logs a message to all appenders (e.g. the console or a file) with corresponding log-levels.
	 *
	 * @param[in] logLevel The log-level of the message.
	 * @param[in] logMessage The message to be logged.
	 */
	void log(const loglevel logLevel, const string logMessage);

};

} /* namespace logging */
#endif /* LOGGER_HPP_ */

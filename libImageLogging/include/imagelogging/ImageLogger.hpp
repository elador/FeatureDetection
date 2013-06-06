/*
 * ImageLogger.hpp
 *
 *  Created on: 04.06.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef IMAGELOGGER_HPP_
#define IMAGELOGGER_HPP_

#include "imagelogging/imageloglevels.hpp"
#include "opencv2/core/core.hpp"
#include <string>
#include <vector>
#include <memory>
#include <functional>

using cv::Mat;
using std::string;
using std::vector;
using std::shared_ptr;
using std::function;

namespace imagelogging {

class Appender;

/**
 * A logger that is e.g. responsible for a certain class or namespace. A logger can log to several outputs (appenders).
 *
 * Possible TODO: Instead of using a string, make the logger usable like an ostream, like: "log[WARN] << "hello" << var << endl;".
 */
class ImageLogger {
public:

	/**
	 * Constructs a new logger that is not logging anything yet.
	 *
	 * @param[in] name The name of the logger. Usually the namespace of the library it is working in.
	 */
	explicit ImageLogger(string name);

	~ImageLogger();

	/**
	 * Adds an appender to this logger.
	 *
	 * @param[in] The new appender.
	 */
	void addAppender(shared_ptr<Appender> appender);

	/**
	 * Sets TODO
	 *
	 * @param[in] TODO.
	 */
	void setCurrentImageName(string imageName);

	/**
	 * Logs a message with log-level TRACE to all appenders (e.g. the console or a file).
	 *
	 * @param[in] logMessage The message to be logged.
	 */
	void trace(Mat image, function<void ()> functionToApply, const string filename);

	/**
	 * Logs a message with log-level DEBUG to all appenders (e.g. the console or a file).
	 *
	 * @param[in] logMessage The message to be logged.
	 */
	void debug(Mat image, function<void ()> functionToApply, const string filename);

	/**
	 * Logs a message with log-level INFO to all appenders (e.g. the console or a file).
	 *
	 * @param[in] logMessage The message to be logged.
	 */
	void info(Mat image, function<void ()> functionToApply, const string filename);

	/**
	 * Logs a message with log-level WARN to all appenders (e.g. the console or a file).
	 *
	 * @param[in] logMessage The message to be logged.
	 */
	void intermediate(Mat image, function<void ()> functionToApply, const string filename);

	/**
	 * Logs a message with log-level ERROR to all appenders (e.g. the console or a file).
	 *
	 * @param[in] logMessage The message to be logged.
	 */
	void final(Mat image, function<void ()> functionToApply, const string filename);

private:
	vector<shared_ptr<Appender>> appenders;
	string name;
	string currentImageName;

	/**
	 * Logs a message to all appenders (e.g. the console or a file) with corresponding log-levels.
	 *
	 * @param[in] logLevel The log-level of the message.
	 * @param[in] logMessage The message to be logged.
	 */
	void log(const loglevel logLevel, Mat image, function<void ()> functionToApply, const string filename);

};

} /* namespace imagelogging */
#endif /* IMAGELOGGER_HPP_ */

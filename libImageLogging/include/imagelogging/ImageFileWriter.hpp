/*
 * ImageFileWriter.hpp
 *
 *  Created on: 05.06.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef IMAGEFILEWRITER_HPP_
#define IMAGEFILEWRITER_HPP_

#include "imagelogging/Appender.hpp"
#ifdef WIN32
	#define BOOST_ALL_DYN_LINK	// Link against the dynamic boost lib. Seems to be necessary because we use /MD, i.e. link to the dynamic CRT.
	#define BOOST_ALL_NO_LIB	// Don't use the automatic library linking by boost with VS2010 (#pragma ...). Instead, we specify everything in cmake.
#endif
#include "boost/filesystem/path.hpp"

using boost::filesystem::path;

namespace imagelogging {

/**
 * An appender that logs all messages equal or below its log-level to a text file.
 * The messages are appended to the log-file if the file already exists.
 */
class ImageFileWriter : public Appender {
public:

	/**
	 * Constructs a new appender that logs to a file. Appends to the file if it already exists.
	 *
	 * param[in] loglevel The loglevel at which to log.
	 * param[in] filename The full path to the file to log to.
	 */
	ImageFileWriter(loglevel logLevel, path directory);

	~ImageFileWriter();

	/**
	 * Appends a message to the opened file.
	 *
	 * @param[in] logLevel The log-level of the message.
	 * @param[in] loggerName The name of the logger that is logging the message.
	 * @param[in] logMessage The log-message itself.
	 */
	void log(const loglevel logLevel, const string loggerName, const string filename, Mat image, function<void ()> functionToApply, const string filenameSuffix);

private:

	path outputDirectory;

	/**
	 * Creates a new string containing the formatted current time.
	 *
	 * @return The formatted time.
	 */
	string getCurrentTime();

};

} /* namespace imagelogging */
#endif /* IMAGEFILEWRITER_HPP_ */

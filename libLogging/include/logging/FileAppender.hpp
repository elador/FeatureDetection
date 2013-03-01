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
 * Image filter that equalizes the histogram using OpenCV's equalizeHist function. The images must
 * be of type 8-bit single channel (CV_8U / CV_8UC1).
 */
class FileAppender : public Appender {
public:

	/**
	 * Constructs a new histogram equalization filter.
	 */
	FileAppender();

	~FileAppender();
};

} /* namespace logging */
#endif /* FILEAPPENDER_HPP_ */

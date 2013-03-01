/*
 * Logger.hpp
 *
 *  Created on: 01.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LOGGER_HPP_
#define LOGGER_HPP_

namespace logging {

/**
 * Image filter that equalizes the histogram using OpenCV's equalizeHist function. The images must
 * be of type 8-bit single channel (CV_8U / CV_8UC1).
 */
class Logger {
public:

	/**
	 * Constructs a new histogram equalization filter.
	 */
	Logger();

	~Logger();

};

} /* namespace logging */
#endif /* LOGGER_HPP_ */

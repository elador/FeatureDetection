/*
 * Kernel.hpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef KERNEL_HPP_
#define KERNEL_HPP_

#include "opencv2/core/core.hpp"

using cv::Mat;

namespace classification {

/**
 * A kernel that calculates ...
 */
class Kernel {
public:

	virtual ~Kernel() {}

	/**
	 * Computes the kernel of ...
	 * TODO I will change this to cv::Mat, once it is tested and running!
	 * TODO Also, the kernel parameters have to be removed from compute(...)!
	 *
	 * @param[in] input The ...
	 * @return The result of the kernel computation
	 */
	virtual double compute(unsigned char* data, unsigned char* support, int nDim) const = 0;
};

} /* namespace classification */
#endif /* KERNEL_HPP_ */

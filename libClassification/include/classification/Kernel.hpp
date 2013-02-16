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
	 *
	 * @param[in] input The ...
	 * @return The result of the kernel computation
	 */
	virtual double compute(const Mat& value) const = 0;
};

} /* namespace classification */
#endif /* KERNEL_HPP_ */

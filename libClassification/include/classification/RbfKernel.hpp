/*
 * RbfKernel.hpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef RBFKERNEL_HPP_
#define RBFKERNEL_HPP_

#include "opencv2/core/core.hpp"

using cv::Mat;

namespace classification {

/**
 * A radial basis function kernel.
 */
class RbfKernel {
public:

	explicit RbfKernel();
	~RbfKernel();

	/**
	 * Computes the radial basis function kernel of ...
	 *
	 * @param[in] input The ...
	 * @return The result of the kernel computation
	 */
	double compute(const Mat& value);
};

} /* namespace classification */
#endif /* RBFKERNEL_HPP_ */

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

struct svm_parameter; // forward decleration from svm.h

namespace classification {

/**
 * A kernel function that computes the dot product of two vectors in a potentially high or infinite dimensional space.
 */
class Kernel {
public:

	virtual ~Kernel() {}

	/**
	 * Computes the kernel value (dot product in a potentially high dimensional space) of two given vectors.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The kernel value of the two vectors.
	 */
	virtual double compute(const Mat& lhs, const Mat& rhs) const = 0;

	/**
	 * Sets the kernel parameters of a given parameter set for libSVM training.
	 *
	 * @param[in] param The libSVM parameters.
	 */
	virtual void setLibSvmParams(struct svm_parameter *param) const = 0;
};

} /* namespace classification */
#endif /* KERNEL_HPP_ */

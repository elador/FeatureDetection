/*
 * RbfKernel.hpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef RBFKERNEL_HPP_
#define RBFKERNEL_HPP_

#include "classification/Kernel.hpp"
#include "svm.h"
#include <stdexcept>

using std::invalid_argument;

namespace classification {

/**
 * A radial basis function kernel of the form k(x, y) = exp(-&gamma; * |u - v|²).
 */
class RbfKernel : public Kernel {
public:

	/**
	 * Constructs a new RBF kernel.
	 *
	 * @param[in] gamma The parameter &gamma; of the radial basis function exp(-&gamma; * |u - v|²).
	 */
	explicit RbfKernel(double gamma) : gamma(gamma), difference() {}

	~RbfKernel() {}

	double compute(const Mat& lhs, const Mat& rhs) const {
		return exp(-gamma * computeSumOfSquaredDifferences(lhs, rhs));
	}

	void setLibSvmParams(struct svm_parameter *param) const {
		param->kernel_type = RBF;
		param->gamma = gamma;
	}

private:

	/**
	 * Computes the sum of the squared differences of two vectors.
	 *
	 * @param[in] lhs The first vector.
	 * @param[in] rhs The second vector.
	 * @return The sum of the squared differences.
	 */
	double computeSumOfSquaredDifferences(const Mat& lhs, const Mat& rhs) const {
		difference = (lhs - rhs);
		return difference.dot(difference);
	}

	double gamma; ///< The parameter &gamma; of the radial basis function exp(-&gamma; * |u - v|²).
	mutable Mat difference; ///< The temporal buffer for the result of the vector difference.
};

} /* namespace classification */
#endif /* RBFKERNEL_HPP_ */

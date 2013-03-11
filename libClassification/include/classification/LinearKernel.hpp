/*
 * LinearKernel.hpp
 *
 *  Created on: 05.03.2013
 *      Author: poschmann
 */

#ifndef LINEARKERNEL_HPP_
#define LINEARKERNEL_HPP_

#include "classification/Kernel.hpp"
#include "svm.h"

namespace classification {

/**
 * A linear kernel of the form k(x, y) = x<sup>T</sup>y.
 */
class LinearKernel : public Kernel {
public:

	/**
	 * Constructs a new linear kernel.
	 */
	explicit LinearKernel() {}

	~LinearKernel() {}

	double compute(const Mat& lhs, const Mat& rhs) const {
		return lhs.dot(rhs);
	}

	void setLibSvmParams(struct svm_parameter *param) const {
		param->kernel_type = LINEAR;
	}
};

} /* namespace classification */
#endif /* LINEARKERNEL_HPP_ */

/*
 * LibSvmKernelParamSetter.hpp
 *
 *  Created on: 19.11.2013
 *      Author: poschmann
 */

#ifndef LIBSVMKERNELPARAMSETTER_HPP_
#define LIBSVMKERNELPARAMSETTER_HPP_

#include "classification/KernelVisitor.hpp"
#include "classification/LinearKernel.hpp"
#include "classification/PolynomialKernel.hpp"
#include "classification/RbfKernel.hpp"
#include "classification/HistogramIntersectionKernel.hpp"
#include "svm.h"

namespace libsvm {

class LibSvmKernelParamSetter : public classification::KernelVisitor {
public:

	/**
	 * Constructs a new libSVM kernel parameter setter.
	 *
	 * @param[in] param The parameters for libSVM training. Will not be deleted on destruction.
	 */
	LibSvmKernelParamSetter(struct svm_parameter *param) : param(param) {}

	~LibSvmKernelParamSetter() {}

	void visit(const classification::LinearKernel& kernel) {
		param->kernel_type = LINEAR;
	}

	void visit(const classification::PolynomialKernel& kernel) {
		param->kernel_type = POLY;
		param->degree = kernel.getDegree();
		param->coef0 = kernel.getConstant();
		param->gamma = kernel.getAlpha();
	}

	void visit(const classification::RbfKernel& kernel) {
		param->kernel_type = RBF;
		param->gamma = kernel.getGamma();
	}

	void visit(const classification::HistogramIntersectionKernel& kernel) {
		param->kernel_type = HIK;
	}

private:

	struct svm_parameter *param; ///< The parameters for libSVM training. Will not be deleted on destruction.
};

} /* namespace libsvm */
#endif /* LIBSVMKERNELPARAMSETTER_HPP_ */

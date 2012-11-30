/*
 * RbfLibSvmParameterBuilder.cpp
 *
 *  Created on: 17.10.2012
 *      Author: poschmann
 */

#include "classification/RbfLibSvmParameterBuilder.h"
#include "svm.h"

namespace classification {

RbfLibSvmParameterBuilder::RbfLibSvmParameterBuilder(double gamma, double C) : LibSvmParameterBuilder(C), gamma(gamma) {}

RbfLibSvmParameterBuilder::~RbfLibSvmParameterBuilder() {}

struct svm_parameter *RbfLibSvmParameterBuilder::createBaseParameters() {
	struct svm_parameter *param = new struct svm_parameter;
	param->kernel_type = RBF;
	param->degree = 0;
	param->gamma = gamma;
	return param;
}

} /* namespace tracking */

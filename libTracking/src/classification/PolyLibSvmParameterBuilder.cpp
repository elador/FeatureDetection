/*
 * PolyLibSvmParameterBuilder.cpp
 *
 *  Created on: 17.10.2012
 *      Author: poschmann
 */

#include "classification/PolyLibSvmParameterBuilder.h"
#include "svm.h"

namespace classification {

PolyLibSvmParameterBuilder::PolyLibSvmParameterBuilder(int degree, double coef, double gamma, double C) :
		LibSvmParameterBuilder(C), degree(degree), coef(coef), gamma(gamma) {}

PolyLibSvmParameterBuilder::~PolyLibSvmParameterBuilder() {}

struct svm_parameter *PolyLibSvmParameterBuilder::createBaseParameters() {
	struct svm_parameter *param = new struct svm_parameter;
	param->kernel_type = POLY;
	param->degree = degree;
	param->coef0 = coef;
	param->gamma = gamma;
	return param;
}

} /* namespace tracking */

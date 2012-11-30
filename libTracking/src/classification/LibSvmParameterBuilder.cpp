/*
 * LibSvmParameterBuilder.cpp
 *
 *  Created on: 17.10.2012
 *      Author: poschmann
 */

#include "classification/LibSvmParameterBuilder.h"
#include "svm.h"
#include <cstdlib>

namespace classification {

LibSvmParameterBuilder::LibSvmParameterBuilder(double C) : C(C) {}

struct svm_parameter *LibSvmParameterBuilder::createParameters(unsigned int positiveCount, unsigned int negativeCount) {
	struct svm_parameter *param = createBaseParameters();
	param->svm_type = C_SVC;
	param->C = C;
	param->cache_size = 100;
	param->eps = 1e-3;
	param->nr_weight = 2;
	param->weight_label = (int*)malloc(2 * sizeof(int));
	param->weight_label[0] = +1;
	param->weight_label[1] = -1;
	param->weight = (double*)malloc(2 * sizeof(double));
	param->weight[0] = positiveCount;
	param->weight[1] = negativeCount;
	param->shrinking = 0;
	param->probability = 0;
	return param;
}

} /* namespace tracking */

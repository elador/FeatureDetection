/*
 * VectorMachineClassifier.cpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#include "classification/ProbabilisticVectorMachineClassifier.hpp"

namespace classification {

ProbabilisticVectorMachineClassifier::ProbabilisticVectorMachineClassifier(void)
{
	posterior_svm[0] = 0.0f;
	posterior_svm[1] = 0.0f;
}


ProbabilisticVectorMachineClassifier::~ProbabilisticVectorMachineClassifier(void)
{
}

} /* namespace classification */

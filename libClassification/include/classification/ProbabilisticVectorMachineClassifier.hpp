/*
 * VectorMachineClassifier.hpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef PROBABILISTICVECTORMACHINECLASSIFIER_HPP_
#define PROBABILISTICVECTORMACHINECLASSIFIER_HPP_

#include "classification/ProbabilisticClassifier.hpp"
#include "classification/VectorMachineClassifier.hpp"

namespace classification {

/**
 * A classifier that uses a VectorMachineClassifier together with a sigmoid function to produce pseudo-probabilistic output.
 * The probability is calculated as 1.0f / (1.0f + exp(posterior_svm[0]*hyperplaneDist + posterior_svm[1])).
 */
class ProbabilisticVectorMachineClassifier : ProbabilisticClassifier
{
public:
	ProbabilisticVectorMachineClassifier(void);
	~ProbabilisticVectorMachineClassifier(void);

private:
	shared_ptr<VectorMachineClassifier> classifier;
	// TODO Sigmoid-stuff
	float posterior_svm[2];	// probabilistic svm output: p(ffp|t) = 1 / (1 + exp(p[0]*t +p[1]))

};

} /* namespace classification */
#endif /* PROBABILISTICVECTORMACHINECLASSIFIER_HPP_ */


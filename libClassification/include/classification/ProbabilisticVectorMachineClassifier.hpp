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
 */
	class ProbabilisticVectorMachineClassifier : ProbabilisticClassifier
{
public:
	ProbabilisticVectorMachineClassifier(void);
	~ProbabilisticVectorMachineClassifier(void);

private:
	VectorMachineClassifier classifier;	// TODO pointer?
	// TODO Sigmoid-stuff

};

} /* namespace classification */
#endif /* PROBABILISTICVECTORMACHINECLASSIFIER_HPP_ */


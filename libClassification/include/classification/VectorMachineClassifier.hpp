/*
 * VectorMachineClassifier.hpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef VECTORMACHINECLASSIFIER_HPP_
#define VECTORMACHINECLASSIFIER_HPP_

#include "classification/BinaryClassifier.hpp"
#include "classification/Kernel.hpp"

namespace classification {

/**
 * A classifier that uses some kind of support vectors to classify a feature vector.
 */
class VectorMachineClassifier : BinaryClassifier
{
public:
	VectorMachineClassifier(void);
	~VectorMachineClassifier(void);

private:
	// TODO: Den Kernel und seine Parameter könnte man auch kapseln.
	Kernel kernel;
	float nonlin_threshold;		// b parameter of the SVM
	int nonLinType;				// 2 = rbf (?)
	float basisParam;
	int polyPower;
	float divisor;
};

} /* namespace classification */
#endif /* VECTORMACHINECLASSIFIER_HPP_ */


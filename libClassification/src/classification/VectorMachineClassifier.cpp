/*
 * VectorMachineClassifier.cpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#include "classification/VectorMachineClassifier.hpp"

namespace classification {

VectorMachineClassifier::VectorMachineClassifier(shared_ptr<Kernel> kernel) :
		kernel(kernel), nonlinThreshold(0.0f), limitReliability(0.0f) {}

VectorMachineClassifier::~VectorMachineClassifier() {}

void VectorMachineClassifier::setLimitReliability(float limitReliability)
{
	this->limitReliability = limitReliability;
}

float VectorMachineClassifier::getLimitReliability()
{
	return limitReliability;
}

} /* namespace classification */

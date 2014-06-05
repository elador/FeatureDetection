/*
 * VectorMachineClassifier.cpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#include "classification/VectorMachineClassifier.hpp"

using std::shared_ptr;

namespace classification {

VectorMachineClassifier::VectorMachineClassifier(shared_ptr<Kernel> kernel) :
		kernel(kernel), bias(0.0f), threshold(0.0f) {}

VectorMachineClassifier::~VectorMachineClassifier() {}

} /* namespace classification */

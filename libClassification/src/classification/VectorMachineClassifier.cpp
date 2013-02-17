/*
 * VectorMachineClassifier.cpp
 *
 *  Created on: 16.02.2013
 *      Author: Patrik Huber
 */

#include "classification/VectorMachineClassifier.hpp"

namespace classification {

VectorMachineClassifier::VectorMachineClassifier(void) : nonlinThreshold(0.0f)
{
	kernel = NULL;
}


VectorMachineClassifier::~VectorMachineClassifier(void)
{
	// Who allocates the kernel? Maybe delete it here?
}

} /* namespace classification */

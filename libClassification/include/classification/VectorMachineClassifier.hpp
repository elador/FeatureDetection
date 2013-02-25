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
#include <memory>
#include <string>

using std::shared_ptr;
using std::string;

namespace classification {

/**
 * A classifier that uses some kind of support vectors to classify a feature vector.
 */
class VectorMachineClassifier : public BinaryClassifier
{
public:
	VectorMachineClassifier(void);
	~VectorMachineClassifier(void);

	virtual void load(const string classifierFilename, const string thresholdsFilename) = 0;

	float getLimitReliability();
	void setLimitReliability(float limitReliability);

protected:
	shared_ptr<Kernel> kernel;	///< The kernel the vector machine uses.
	float nonlinThreshold;		///< The b parameter of the vector machine.
	float limitReliability;	///< Additional (dynamic) threshold to adjust the threshold at which featureVectors pass the vector machine. 

};

} /* namespace classification */
#endif /* VECTORMACHINECLASSIFIER_HPP_ */


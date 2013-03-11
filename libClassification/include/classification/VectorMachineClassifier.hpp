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

	/**
	 * Constructs a new vector machine classifier with a default threshold of zero.
	 *
	 * @param[in] kernel The kernel function.
	 */
	explicit VectorMachineClassifier(shared_ptr<Kernel> kernel);

	virtual ~VectorMachineClassifier();

	float getLimitReliability();
	void setLimitReliability(float limitReliability);

	/**
	 * @return The kernel function.
	 */
	shared_ptr<Kernel> getKernel() {
		return kernel;
	}

	/**
	 * @return The kernel function.
	 */
	const shared_ptr<Kernel> getKernel() const {
		return kernel;
	}

protected:

	shared_ptr<Kernel> kernel; ///< The kernel function.
	float nonlinThreshold;     ///< The bias. TODO rename to bias?
	float limitReliability;    ///< The threshold to compare the hyperplane distance against for determining the label. TODO rename to threshold?
};

} /* namespace classification */
#endif /* VECTORMACHINECLASSIFIER_HPP_ */


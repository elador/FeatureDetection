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
	explicit VectorMachineClassifier(std::shared_ptr<Kernel> kernel);

	virtual ~VectorMachineClassifier();

	/**
	 * @return The threshold to compare the hyperplane distance against for determining the label.
	 */
	float getThreshold() const {
		return threshold;
	}

	/**
	 * @param[in] threshold The new threshold to compare the hyperplane distance against for determining the label.
	 */
	void setThreshold(float threshold) {
		this->threshold = threshold;
	}

	/**
	 * @return The kernel function.
	 */
	std::shared_ptr<Kernel> getKernel() {
		return kernel;
	}

	/**
	 * @return The kernel function.
	 */
	const std::shared_ptr<Kernel> getKernel() const {
		return kernel;
	}

	/**
	 * @return The bias that is subtracted from the sum over all scaled kernel values.
	 */
	float getBias() const {
		return bias;
	}

protected:

	std::shared_ptr<Kernel> kernel; ///< The kernel function.
	float bias;      ///< The bias that is subtracted from the sum over all scaled kernel values.
	float threshold; ///< The threshold to compare the hyperplane distance against for determining the label.
					 // Larger => weniger patches drueber(mehr rejected, langsamer), dh. mehr fn(FRR), weniger fp(FAR)
					 // Smaller => mehr patches drueber(mehr nicht rejected, schneller), dh. weniger fn(FRR), mehr fp(FAR)
};

} /* namespace classification */
#endif /* VECTORMACHINECLASSIFIER_HPP_ */


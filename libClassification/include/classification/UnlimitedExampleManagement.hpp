/*
 * UnlimitedExampleManagement.hpp
 *
 *  Created on: 26.11.2013
 *      Author: poschmann
 */

#ifndef UNLIMITEDEXAMPLEMANAGEMENT_HPP_
#define UNLIMITEDEXAMPLEMANAGEMENT_HPP_

#include "classification/VectorBasedExampleManagement.hpp"

namespace classification {

/**
 * Example storage that never replaces existing training examples (unless cleared).
 */
class UnlimitedExampleManagement : public VectorBasedExampleManagement {
public:

	/**
	 * Constructs a new unlimited example management.
	 *
	 * @param[in] requiredSize Minimum amount of training examples required for training.
	 */
	explicit UnlimitedExampleManagement(size_t requiredSize = 1);

	void add(const std::vector<cv::Mat>& newExamples);
};

} /* namespace classification */
#endif /* UNLIMITEDEXAMPLEMANAGEMENT_HPP_ */

/*
 * AgeBasedExampleManagement.hpp
 *
 *  Created on: 26.11.2013
 *      Author: poschmann
 */

#ifndef AGEBASEDEXAMPLEMANAGEMENT_HPP_
#define AGEBASEDEXAMPLEMANAGEMENT_HPP_

#include "classification/VectorBasedExampleManagement.hpp"

namespace classification {

/**
 * Example storage that, when reaching maximum size, replaces the oldest training examples with
 * new ones.
 */
class AgeBasedExampleManagement : public VectorBasedExampleManagement {
public:

	/**
	 * Constructs a new age based example management.
	 *
	 * @param[in] capacity Maximum amount of stored training examples.
	 * @param[in] requiredSize Minimum amount of training examples required for training.
	 */
	explicit AgeBasedExampleManagement(size_t capacity, size_t requiredSize = 1);

	void add(const std::vector<cv::Mat>& newExamples);

private:

	size_t insertPosition; ///< The insertion index of new examples.
};

} /* namespace classification */
#endif /* AGEBASEDEXAMPLEMANAGEMENT_HPP_ */

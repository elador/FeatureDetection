/*
 * ConfidenceBasedExampleManagement.hpp
 *
 *  Created on: 25.11.2013
 *      Author: poschmann
 */

#ifndef CONFIDENCEBASEDEXAMPLEMANAGEMENT_HPP_
#define CONFIDENCEBASEDEXAMPLEMANAGEMENT_HPP_

#include "classification/VectorBasedExampleManagement.hpp"

namespace classification {

class BinaryClassifier;

/**
 * Example storage that, when reaching maximum size, replaces training examples that have the highest
 * confidence when evaluated by the classifier. The first training example will not be replaced.
 */
class ConfidenceBasedExampleManagement : public VectorBasedExampleManagement {
public:

	/**
	 * Constructs a new confidence based example management.
	 *
	 * @param[in] classifier Classifier for computing the confidences of the training examples.
	 * @param[in] positive Flag that indicates whether this set contains positive training examples.
	 * @param[in] capacity Maximum amount of stored training examples.
	 * @param[in] requiredSize Minimum amount of training examples required for training.
	 */
	ConfidenceBasedExampleManagement(const std::shared_ptr<BinaryClassifier>& classifier, bool positive, size_t capacity, size_t requiredSize = 1);

	void add(const std::vector<cv::Mat>& newExamples);

private:

	const std::shared_ptr<BinaryClassifier> classifier; ///< Classifier for computing the confidences of the training examples.
	bool positive; ///< Flag that indicates whether this set contains positive training examples.
};

} /* namespace classification */
#endif /* CONFIDENCEBASEDEXAMPLEMANAGEMENT_HPP_ */

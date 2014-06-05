/*
 * TrainableClassifier.hpp
 *
 *  Created on: 06.12.2012
 *      Author: poschmann
 */

#ifndef TRAINABLECLASSIFIER_HPP_
#define TRAINABLECLASSIFIER_HPP_

#include "opencv2/core/core.hpp"
#include <vector>

namespace classification {

/**
 * Classifier that may be re-trained using new examples. Re-training is an incremental procedure that adds
 * new examples and refines the classifier. Nevertheless, it may be possible that previous training
 * examples are forgotten to ensure the classifier stays relatively small and efficient.
 */
class TrainableClassifier {
public:

	virtual ~TrainableClassifier() {}

	/**
	 * Determines whether this classifier was trained successfully and may be used.
	 *
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	virtual bool isUsable() const = 0;

	/**
	 * Re-trains this classifier incrementally, adding new training examples. May not change the classifier
	 * if there is not enough training data.
	 *
	 * @param[in] newPositiveExamples The new positive training examples.
	 * @param[in] newNegativeExamples The new negative training examples.
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	virtual bool retrain(const std::vector<cv::Mat>& newPositiveExamples, const std::vector<cv::Mat>& newNegativeExamples) = 0;

	/**
	 * Resets this classifier.
	 */
	virtual void reset() = 0;
};

} /* namespace classification */
#endif /* TRAINABLECLASSIFIER_HPP_ */

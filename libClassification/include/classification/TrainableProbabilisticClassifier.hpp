/*
 * TrainableProbabilisticClassifier.hpp
 *
 *  Created on: 05.03.2013
 *      Author: poschmann
 */

#ifndef TRAINABLEPROBABILISTICCLASSIFIER_HPP_
#define TRAINABLEPROBABILISTICCLASSIFIER_HPP_

#include "classification/TrainableClassifier.hpp"
#include "classification/ProbabilisticClassifier.hpp"

namespace classification {

/**
 * Probabilistic classifier that may be re-trained using new examples. Re-training is an incremental procedure
 * that adds new examples and refines the classifier. Nevertheless, it may be possible that previous
 * training examples are forgotten to ensure the classifier stays relatively small and efficient.
 */
class TrainableProbabilisticClassifier : public TrainableClassifier, public ProbabilisticClassifier {
public:

	virtual ~TrainableProbabilisticClassifier() {}

	/**
	 * Re-trains this classifier incrementally, adding new training examples. The training examples will also
	 * be used as test examples to determine the parameters of the logistic function for probabilistic output.
	 * May not change the classifier if there is not enough training data.
	 *
	 * @param[in] newPositiveExamples The new positive training and test examples.
	 * @param[in] newNegativeExamples The new negative training and test examples.
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	virtual bool retrain(const std::vector<cv::Mat>& newPositiveExamples, const std::vector<cv::Mat>& newNegativeExamples) = 0;

	/**
	 * Re-trains this classifier incrementally, adding new training examples. The test examples will be used to
	 * determine the parameters of the logistic function for probabilistic output. May not change the classifier
	 * if there is not enough training data.
	 *
	 * @param[in] newPositiveExamples The new positive training examples.
	 * @param[in] newNegativeExamples The new negative training examples.
	 * @param[in] newPositiveTestExamples The new positive test examples.
	 * @param[in] newNegativeTestExamples The new negative test examples.
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	virtual bool retrain(const std::vector<cv::Mat>& newPositiveExamples, const std::vector<cv::Mat>& newNegativeExamples,
			const std::vector<cv::Mat>& newPositiveTestExamples, const std::vector<cv::Mat>& newNegativeTestExamples) = 0;
};

} /* namespace classification */
#endif /* TRAINABLEPROBABILISTICCLASSIFIER_HPP_ */

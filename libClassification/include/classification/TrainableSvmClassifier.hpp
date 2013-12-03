/*
 * TrainableSvmClassifier.hpp
 *
 *  Created on: 05.03.2013
 *      Author: poschmann
 */

#ifndef TRAINABLESVMCLASSIFIER_HPP_
#define TRAINABLESVMCLASSIFIER_HPP_

#include "classification/TrainableBinaryClassifier.hpp"
#include <memory>
#include <utility>

namespace classification {

class SvmClassifier;
class Kernel;

/**
 * SVM classifier that can be re-trained. Uses libSVM for training.
 */
class TrainableSvmClassifier : public TrainableBinaryClassifier {
public:

	/**
	 * Constructs a new trainable SVM classifier that wraps around the actual SVM classifier.
	 *
	 * @param[in] svm The actual SVM.
	 */
	explicit TrainableSvmClassifier(std::shared_ptr<SvmClassifier> svm);

	/**
	 * Constructs a new trainable SVM classifier.
	 *
	 * @param[in] kernel The kernel function.
	 */
	explicit TrainableSvmClassifier(std::shared_ptr<Kernel> kernel);

	virtual ~TrainableSvmClassifier();

	bool classify(const cv::Mat& featureVector) const;

	std::pair<bool, double> getConfidence(const cv::Mat& featureVector) const;

	bool isUsable() const;

	/**
	 * @return The actual SVM.
	 */
	std::shared_ptr<SvmClassifier> getSvm();

	/**
	 * @return The actual SVM.
	 */
	const std::shared_ptr<SvmClassifier> getSvm() const;

protected:

	std::shared_ptr<SvmClassifier> svm; ///< The actual SVM.
	bool usable; ///< Flag that indicates whether this classifier is usable.
};

} /* namespace classification */
#endif /* TRAINABLESVMCLASSIFIER_HPP_ */

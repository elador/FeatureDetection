/*
 * FixedTrainableProbabilisticSvmClassifier.hpp
 *
 *  Created on: 08.03.2013
 *      Author: poschmann
 */

#ifndef FIXEDTRAINABLEPROBABILISTICSVMCLASSIFIER_HPP_
#define FIXEDTRAINABLEPROBABILISTICSVMCLASSIFIER_HPP_

#include "classification/TrainableProbabilisticSvmClassifier.hpp"

namespace classification {

/**
 * Trainable probabilistic SVM classifier that assumes fixed mean positive and negative SVM outputs and computes
 * the logistic parameters only once on construction.
 */
class FixedTrainableProbabilisticSvmClassifier : public TrainableProbabilisticSvmClassifier {
public:

	/**
	 * Constructs a new fixed trainable probabilistic SVM classifier with the given logistic parameters.
	 *
	 * @param[in] trainableSvm The trainable SVM classifier.
	 * @param[in] logisticA Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 * @param[in] logisticB Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	 */
	FixedTrainableProbabilisticSvmClassifier(std::shared_ptr<TrainableSvmClassifier> trainableSvm, double logisticA, double logisticB) :
			TrainableProbabilisticSvmClassifier(trainableSvm, 0, 0), logisticA(logisticA), logisticB(logisticB) {}

	/**
	 * Constructs a new fixed trainable probabilistic SVM classifier that computes the logistic parameters from the
	 * given probabilities and mean SVM outputs.
	 *
	 * @param[in] trainableSvm The trainable SVM classifier.
	 * @param[in] highProb The probability of the mean output of positive samples.
	 * @param[in] lowProb The probability of the mean output of negative samples.
	 * @param[in] meanPosOutput The estimated mean SVM output of the positive samples.
	 * @param[in] meanNegOutput The estimated mean SVM output of the negative samples.
	 */
	FixedTrainableProbabilisticSvmClassifier(std::shared_ptr<TrainableSvmClassifier> trainableSvm,
			double highProb, double lowProb, double meanPosOutput, double meanNegOutput) :
					TrainableProbabilisticSvmClassifier(trainableSvm, 0, 0, highProb, lowProb) {
		std::pair<double, double> logisticParameters = computeLogisticParameters(meanPosOutput, meanNegOutput);
		logisticA = logisticParameters.first;
		logisticB = logisticParameters.second;
	}

	~FixedTrainableProbabilisticSvmClassifier() {}

protected:

	using TrainableProbabilisticSvmClassifier::computeLogisticParameters;

	std::pair<double, double> computeLogisticParameters(std::shared_ptr<SvmClassifier> trainableSvm) const {
		return std::make_pair(logisticA, logisticB);
	}

private:

	double logisticA; ///< Parameter a of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
	double logisticB; ///< Parameter b of the logistic function for pseudo-probabilistic output p(x) = 1 / (1 + exp(a + b * x)).
};

} /* namespace classification */
#endif /* FIXEDTRAINABLEPROBABILISTICSVMCLASSIFIER_HPP_ */

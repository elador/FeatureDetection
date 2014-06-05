/*
 * ProbabilisticTwoStageClassifier.cpp
 *
 *  Created on: 21.12.2012
 *      Author: poschmann
 */

#include "classification/ProbabilisticTwoStageClassifier.hpp"

using cv::Mat;
using std::pair;
using std::shared_ptr;

namespace classification {

ProbabilisticTwoStageClassifier::ProbabilisticTwoStageClassifier(
		shared_ptr<ProbabilisticClassifier> first, shared_ptr<ProbabilisticClassifier> second) :
				first(first), second(second) {}

ProbabilisticTwoStageClassifier::~ProbabilisticTwoStageClassifier() {}

bool ProbabilisticTwoStageClassifier::classify(const Mat& featureVector) const {
	if (first->classify(featureVector))
		return second->classify(featureVector);
	return false;
}

pair<bool, double> ProbabilisticTwoStageClassifier::getConfidence(const Mat& featureVector) const {
	pair<bool, double> result = first->getConfidence(featureVector);
	if (result.first)
		return second->getConfidence(featureVector);
	return result;
}

pair<bool, double> ProbabilisticTwoStageClassifier::getProbability(const Mat& featureVector) const {
	pair<bool, double> result = first->getProbability(featureVector);
	if (result.first)
		return second->getConfidence(featureVector);
	return result;
}

} /* namespace classification */

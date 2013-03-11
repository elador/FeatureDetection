/*
 * ProbabilisticTwoStageClassifier.cpp
 *
 *  Created on: 21.12.2012
 *      Author: poschmann
 */

#include "classification/ProbabilisticTwoStageClassifier.hpp"

namespace classification {

ProbabilisticTwoStageClassifier::ProbabilisticTwoStageClassifier(
		shared_ptr<ProbabilisticClassifier> first, shared_ptr<ProbabilisticClassifier> second) :
				first(first), second(second) {}

ProbabilisticTwoStageClassifier::~ProbabilisticTwoStageClassifier() {}

pair<bool, double> ProbabilisticTwoStageClassifier::classify(const Mat& featureVector) const {
	pair<bool, double> result = first->classify(featureVector);
	if (result.first) {
		return second->classify(featureVector);
	} else {
		return result;
	}
}

} /* namespace classification */

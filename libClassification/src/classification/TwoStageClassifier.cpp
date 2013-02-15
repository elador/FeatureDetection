/*
 * TwoStageClassifier.cpp
 *
 *  Created on: 21.12.2012
 *      Author: poschmann
 */

#include "classification/TwoStageClassifier.h"

namespace classification {

TwoStageClassifier::TwoStageClassifier(shared_ptr<Classifier> first, shared_ptr<Classifier> second) :
		first(first), second(second) {}

TwoStageClassifier::~TwoStageClassifier() {}

pair<bool, double> TwoStageClassifier::classify(const FeatureVector& featureVector) const {
	pair<bool, double> result = first->classify(featureVector);
	if (result.first) {
		return second->classify(featureVector);
	} else {
		return result;
	}
}

} /* namespace classification */

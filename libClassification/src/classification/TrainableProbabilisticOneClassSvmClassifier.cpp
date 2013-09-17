/*
 * TrainableProbabilisticOneClassSvmClassifier.cpp
 *
 *  Created on: 13.09.2013
 *      Author: poschmann
 */

#include "classification/TrainableProbabilisticOneClassSvmClassifier.hpp"
#include "classification/TrainableOneClassSvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include <cmath>

using std::make_shared;
using std::make_pair;

namespace classification {

TrainableProbabilisticOneClassSvmClassifier::TrainableProbabilisticOneClassSvmClassifier(
		shared_ptr<TrainableOneClassSvmClassifier> trainableSvm, double highProb, double lowProb, double meanPosOutput, double meanNegOutput) :
				probabilisticSvm(make_shared<ProbabilisticSvmClassifier>(trainableSvm->getSvm())), trainableSvm(trainableSvm),
				highProb(highProb), lowProb(lowProb) {
	double logisticB = (log((1 - lowProb) / lowProb) - log((1 - highProb) / highProb)) / (meanNegOutput - meanPosOutput);
	double logisticA = log((1 - highProb) / highProb) - logisticB * meanPosOutput;
	probabilisticSvm->setLogisticParameters(logisticA, logisticB);
}

TrainableProbabilisticOneClassSvmClassifier::~TrainableProbabilisticOneClassSvmClassifier() {}

bool TrainableProbabilisticOneClassSvmClassifier::isUsable() const {
	return trainableSvm->isUsable();
}

pair<bool, double> TrainableProbabilisticOneClassSvmClassifier::classify(const Mat& featureVector) const {
	return probabilisticSvm->classify(featureVector);
}

bool TrainableProbabilisticOneClassSvmClassifier::retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
	return trainableSvm->retrain(newPositiveExamples, newNegativeExamples);
}

void TrainableProbabilisticOneClassSvmClassifier::reset() {
	trainableSvm->reset();
}

} /* namespace classification */

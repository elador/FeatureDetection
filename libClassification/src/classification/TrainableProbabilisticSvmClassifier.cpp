/*
 * TrainableProbabilisticSvmClassifier.cpp
 *
 *  Created on: 08.03.2013
 *      Author: poschmann
 */

#include "classification/TrainableProbabilisticSvmClassifier.hpp"
#include "classification/TrainableSvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include <cmath>

using std::make_shared;
using std::make_pair;

namespace classification {

TrainableProbabilisticSvmClassifier::TrainableProbabilisticSvmClassifier(
		shared_ptr<TrainableSvmClassifier> trainable, double highProb, double lowProb) :
		probabilisticSvm(make_shared<ProbabilisticSvmClassifier>(trainable->getSvm())), trainableSvm(trainable),
		highProb(highProb), lowProb(lowProb) {}

TrainableProbabilisticSvmClassifier::~TrainableProbabilisticSvmClassifier() {}

bool TrainableProbabilisticSvmClassifier::isUsable() const {
	return trainableSvm->isUsable();
}

pair<bool, double> TrainableProbabilisticSvmClassifier::classify(const Mat& featureVector) const {
	return probabilisticSvm->classify(featureVector);
}

bool TrainableProbabilisticSvmClassifier::retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
	if (trainableSvm->retrain(newPositiveExamples, newNegativeExamples)) {
		pair<double, double> logisticParameters = computeLogisticParameters(trainableSvm);
		probabilisticSvm->setLogisticParameters(logisticParameters.first, logisticParameters.second);
		return true;
	}
	return false;
}

void TrainableProbabilisticSvmClassifier::reset() {
	trainableSvm->reset();
}

pair<double, double> TrainableProbabilisticSvmClassifier::computeLogisticParameters(double meanPosOutput, double meanNegOutput) const {
	double logisticB = (log((1 - lowProb) / lowProb) - log((1 - highProb) / highProb)) / (meanNegOutput - meanPosOutput);
	double logisticA = log((1 - highProb) / highProb) - logisticB * meanPosOutput;
	return make_pair(logisticA, logisticB);
}

pair<double, double> TrainableProbabilisticSvmClassifier::computeLogisticParameters(shared_ptr<TrainableSvmClassifier> trainableSvm) const {
	pair<double, double> meanOutputs = trainableSvm->computeMeanSvmOutputs();
	return computeLogisticParameters(meanOutputs.first, meanOutputs.second);
}

} /* namespace classification */

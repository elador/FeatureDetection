/*
 * TrainableProbabilisticSvmClassifier.cpp
 *
 *  Created on: 08.03.2013
 *      Author: poschmann
 */

#include "classification/TrainableProbabilisticSvmClassifier.hpp"
#include "classification/TrainableSvmClassifier.hpp"
#include "classification/ProbabilisticSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include <cmath>

using cv::Mat;
using std::pair;
using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::make_pair;

namespace classification {

TrainableProbabilisticSvmClassifier::TrainableProbabilisticSvmClassifier(
		shared_ptr<TrainableSvmClassifier> trainableSvm, int positiveCount, int negativeCount, double highProb, double lowProb) :
				probabilisticSvm(make_shared<ProbabilisticSvmClassifier>(trainableSvm->getSvm())), trainableSvm(trainableSvm),
				positiveTestExamples(), negativeTestExamples(), positiveInsertPosition(0), negativeInsertPosition(0),
				highProb(highProb), lowProb(lowProb), adjustThreshold(false), targetProbability(0.5) {
	positiveTestExamples.reserve(positiveCount);
	negativeTestExamples.reserve(negativeCount);
}

TrainableProbabilisticSvmClassifier::~TrainableProbabilisticSvmClassifier() {}

bool TrainableProbabilisticSvmClassifier::isUsable() const {
	return trainableSvm->isUsable();
}

bool TrainableProbabilisticSvmClassifier::classify(const Mat& featureVector) const {
	return probabilisticSvm->classify(featureVector);
}

pair<bool, double> TrainableProbabilisticSvmClassifier::getConfidence(const Mat& featureVector) const {
	return probabilisticSvm->getConfidence(featureVector);
}

pair<bool, double> TrainableProbabilisticSvmClassifier::getProbability(const Mat& featureVector) const {
	return probabilisticSvm->getProbability(featureVector);
}

bool TrainableProbabilisticSvmClassifier::retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
	return retrain(newPositiveExamples, newNegativeExamples, newPositiveExamples, newNegativeExamples);
}

bool TrainableProbabilisticSvmClassifier::retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples,
		const vector<Mat>& newPositiveTestExamples, const vector<Mat>& newNegativeTestExamples) {
	if (positiveTestExamples.capacity() > 0 && negativeTestExamples.capacity() > 0) {
		addTestExamples(positiveTestExamples, newPositiveTestExamples, positiveInsertPosition);
		addTestExamples(negativeTestExamples, newNegativeTestExamples, negativeInsertPosition);
	}
	if (trainableSvm->retrain(newPositiveExamples, newNegativeExamples)) {
		pair<double, double> logisticParameters = computeLogisticParameters(probabilisticSvm->getSvm());
		probabilisticSvm->setLogisticParameters(logisticParameters.first, logisticParameters.second);
		if (adjustThreshold) {
			double targetThreshold = (log(1.0 / targetProbability - 1.0) - logisticParameters.first) / logisticParameters.second;
			probabilisticSvm->getSvm()->setThreshold(targetThreshold);
		}
		return true;
	}
	return false;
}

void TrainableProbabilisticSvmClassifier::addTestExamples(vector<Mat>& examples, const vector<Mat>& newExamples, size_t& insertPosition) {
	// add new examples as long as there is space available
	auto example = newExamples.cbegin();
	for (; examples.size() < examples.capacity() && example != newExamples.cend(); ++example)
		examples.push_back(*example);
	// replace the oldest examples by new ones
	for (; example != newExamples.cend(); ++example) {
		examples[insertPosition] = *example;
		++insertPosition;
		if (insertPosition == examples.size())
			insertPosition = 0;
	}
}


void TrainableProbabilisticSvmClassifier::reset() {
	positiveTestExamples.clear();
	negativeTestExamples.clear();
	trainableSvm->reset();
}

pair<double, double> TrainableProbabilisticSvmClassifier::computeLogisticParameters(double meanPosOutput, double meanNegOutput) const {
	double logisticB = (log((1 - lowProb) / lowProb) - log((1 - highProb) / highProb)) / (meanNegOutput - meanPosOutput);
	double logisticA = log((1 - highProb) / highProb) - logisticB * meanPosOutput;
	return make_pair(logisticA, logisticB);
}

pair<double, double> TrainableProbabilisticSvmClassifier::computeLogisticParameters(shared_ptr<SvmClassifier> svm) const {
	double meanPosOutput = computeMeanOutput(svm, positiveTestExamples);
	double meanNegOutput = computeMeanOutput(svm, negativeTestExamples);
	return computeLogisticParameters(meanPosOutput, meanNegOutput);
}

double TrainableProbabilisticSvmClassifier::computeMeanOutput(shared_ptr<SvmClassifier> svm, const vector<Mat>& examples) const {
	double sum = 0;
	for (const Mat& example : examples)
		sum += svm->computeHyperplaneDistance(example);
	return sum / examples.size();
}

} /* namespace classification */

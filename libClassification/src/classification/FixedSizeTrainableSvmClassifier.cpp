/*
 * FixedSizeTrainableSvmClassifier.cpp
 *
 *  Created on: 08.03.2013
 *      Author: poschmann
 */

#include "classification/FixedSizeTrainableSvmClassifier.hpp"
#include "svm.h"
#include <algorithm>

using std::max;
using std::make_pair;

namespace classification {

FixedSizeTrainableSvmClassifier::FixedSizeTrainableSvmClassifier(shared_ptr<Kernel> kernel, double constraintsViolationCosts,
		unsigned int positiveExamples, unsigned int negativeExamples, unsigned int minPositiveExamples) :
				TrainableSvmClassifier(kernel, constraintsViolationCosts), positiveExamples(), negativeExamples(),
				negativeInsertPosition(0), minPositiveExamples(minPositiveExamples) {
	this->positiveExamples.reserve(positiveExamples);
	this->negativeExamples.reserve(negativeExamples);
	this->minPositiveExamples = max(static_cast<unsigned int>(1), this->minPositiveExamples);
}

FixedSizeTrainableSvmClassifier::~FixedSizeTrainableSvmClassifier() {}

void FixedSizeTrainableSvmClassifier::clearExamples() {
	positiveExamples.clear();
	negativeExamples.clear();
}

void FixedSizeTrainableSvmClassifier::addExamples(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
	addPositiveExamples(newPositiveExamples);
	addNegativeExamples(newNegativeExamples);
}

void FixedSizeTrainableSvmClassifier::addPositiveExamples(const vector<Mat>& newPositiveExamples) {
	// compute hyperplane distances of current positive examples and sort in descending order
	vector<pair<unsigned int, double>> distances;
	distances.reserve(positiveExamples.size());
	if (isUsable()) {
		for (unsigned int i = 1; i < positiveExamples.size(); ++i)
			distances.push_back(make_pair(i, computeSvmOutput(positiveExamples[i].get())));
		sort(distances.begin(), distances.end(), [](pair<unsigned int, double> a, pair<unsigned int, double> b) {
			return a.second > b.second;
		});
	} else {
		for (unsigned int i = 1; i < positiveExamples.size(); ++i)
			distances.push_back(make_pair(i, 0.5));
	}
	// add new positive examples as long as no examples have to be removed
	auto example = newPositiveExamples.cbegin();
	for (; positiveExamples.size() < positiveExamples.capacity() && example != newPositiveExamples.cend(); ++example)
		positiveExamples.push_back(move(utils.createNode(*example)));
	// replace existing examples (beginning with the high distance ones) with new examples
	auto distanceIndex = distances.cbegin();
	for (; example != newPositiveExamples.cend() && distanceIndex != distances.cend(); ++example, ++distanceIndex)
		positiveExamples[distanceIndex->first] = move(utils.createNode(*example));
}

void FixedSizeTrainableSvmClassifier::addNegativeExamples(const vector<Mat>& newNegativeExamples) {
	// add new negative examples as long as there is space available
	auto example = newNegativeExamples.cbegin();
	for (; negativeExamples.size() < negativeExamples.capacity() && example != newNegativeExamples.cend(); ++example)
		negativeExamples.push_back(move(utils.createNode(*example)));
	// replace the oldest negative examples by new ones
	for (; example != newNegativeExamples.cend(); ++example) {
		negativeExamples[negativeInsertPosition] = move(utils.createNode(*example));
		++negativeInsertPosition;
		if (negativeInsertPosition == negativeExamples.size())
			negativeInsertPosition = 0;
	}
}

unsigned int FixedSizeTrainableSvmClassifier::getPositiveCount() const {
	return positiveExamples.size();
}

unsigned int FixedSizeTrainableSvmClassifier::getNegativeCount() const {
	return negativeExamples.size();
}

bool FixedSizeTrainableSvmClassifier::isRetrainingReasonable() const {
	return positiveExamples.size() >= minPositiveExamples && !negativeExamples.empty();
}

unsigned int FixedSizeTrainableSvmClassifier::fillProblem(struct svm_problem *problem) const {
	int i = 0;
	for (auto& example : positiveExamples) {
		problem->y[i] = 1;
		problem->x[i] = example.get();
		++i;
	}
	for (auto& example : negativeExamples) {
		problem->y[i] = -1;
		problem->x[i] = example.get();
		++i;
	}
	return i;
}

} /* namespace classification */

/*
 * FrameBasedTrainableSvmClassifier.cpp
 *
 *  Created on: 06.03.2013
 *      Author: poschmann
 */

#include "classification/FrameBasedTrainableSvmClassifier.hpp"
#include "svm.h"

namespace classification {

FrameBasedTrainableSvmClassifier::FrameBasedTrainableSvmClassifier(shared_ptr<Kernel> kernel,
		double constraintsViolationCosts, int frameLength, float minAvgSamples) :
				TrainableSvmClassifier(kernel, constraintsViolationCosts), frameLength(frameLength), minAvgSamples(minAvgSamples),
				positiveExamples(frameLength), negativeExamples(frameLength), oldestEntry(0) {}

FrameBasedTrainableSvmClassifier::~FrameBasedTrainableSvmClassifier() {}

unsigned int FrameBasedTrainableSvmClassifier::getRequiredPositiveCount() const {
	return static_cast<unsigned int>(ceil(minAvgSamples * static_cast<float>(frameLength)));
}

unsigned int FrameBasedTrainableSvmClassifier::getPositiveCount() const {
	unsigned int count = 0;
	for (auto examples = positiveExamples.cbegin(); examples < positiveExamples.cend(); ++examples)
		count += examples->size();
	return count;
}

unsigned int FrameBasedTrainableSvmClassifier::getNegativeCount() const {
	unsigned int count = 0;
	for (auto examples = negativeExamples.cbegin(); examples < negativeExamples.cend(); ++examples)
		count += examples->size();
	return count;
}

void FrameBasedTrainableSvmClassifier::clearExamples() {
	for (auto examples = positiveExamples.begin(); examples != positiveExamples.end(); ++examples)
		examples->clear();
	for (auto examples = negativeExamples.begin(); examples != negativeExamples.end(); ++examples)
		examples->clear();
	oldestEntry = 0;
}

void FrameBasedTrainableSvmClassifier::addExamples(
		const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
	replaceExamples(positiveExamples[oldestEntry], newPositiveExamples);
	replaceExamples(negativeExamples[oldestEntry], newNegativeExamples);
	++oldestEntry;
	if (oldestEntry >= frameLength)
		oldestEntry = 0;
}

void FrameBasedTrainableSvmClassifier::replaceExamples(
		vector<unique_ptr<struct svm_node[], NodeDeleter>>& examples, const vector<Mat>& newExamples) {
	examples.clear();
	for (auto example = newExamples.cbegin(); example != newExamples.cend(); ++example)
		examples.push_back(move(createNode(*example)));
}

bool FrameBasedTrainableSvmClassifier::isRetrainingReasonable() const {
	return getPositiveCount() >= getRequiredPositiveCount();
}

unsigned int FrameBasedTrainableSvmClassifier::fillProblem(struct svm_problem *problem) const {
	int i = 0;
	for (auto examples = positiveExamples.cbegin(); examples != positiveExamples.cend(); ++examples) {
		for (auto example = examples->cbegin(); example != examples->cend(); ++example) {
			problem->y[i] = 1;
			problem->x[i] = example->get();
			++i;
		}
	}
	for (auto examples = negativeExamples.cbegin(); examples != negativeExamples.cend(); ++examples) {
		for (auto example = examples->cbegin(); example != examples->cend(); ++example) {
			problem->y[i] = -1;
			problem->x[i] = example->get();
			++i;
		}
	}
	return i;
}

} /* namespace classification */

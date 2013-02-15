/*
 * FrameBasedLibSvmTraining.cpp
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#include "classification/FrameBasedLibSvmTraining.h"
#include "classification/FeatureVector.h"
#include <cmath>
#include <cstdlib>

namespace classification {

FrameBasedLibSvmTraining::FrameBasedLibSvmTraining(int frameLength, float minAvgSamples,
		shared_ptr<LibSvmParameterBuilder> parameterBuilder,
		shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation) :
				LibSvmTraining(parameterBuilder, sigmoidParameterComputation), frameLength(frameLength),
				minAvgSamples(minAvgSamples), dimensions(0), positiveExamples(frameLength),
				negativeExamples(frameLength), oldestEntry(0) {}

FrameBasedLibSvmTraining::FrameBasedLibSvmTraining(int frameLength, float minAvgSamples, std::string negativesFilename,
		int negatives, shared_ptr<LibSvmParameterBuilder> parameterBuilder,
		shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation) :
				LibSvmTraining(parameterBuilder, sigmoidParameterComputation), frameLength(frameLength),
				minAvgSamples(minAvgSamples), dimensions(0), positiveExamples(frameLength),
				negativeExamples(frameLength), oldestEntry(0) {
	readStaticNegatives(negativesFilename, negatives);
}

FrameBasedLibSvmTraining::~FrameBasedLibSvmTraining() {
	vector<vector<struct svm_node *> >::iterator sit;
	for (sit = positiveExamples.begin(); sit < positiveExamples.end(); ++sit)
		freeExamples(*sit);
	for (sit = negativeExamples.begin(); sit < negativeExamples.end(); ++sit)
		freeExamples(*sit);
}

int FrameBasedLibSvmTraining::getRequiredPositiveSampleCount() const {
	return (int)ceil((double)minAvgSamples * (double)frameLength);
}

int FrameBasedLibSvmTraining::getPositiveSampleCount() const {
	int count = 0;
	for (vector<vector<struct svm_node *> >::const_iterator it = positiveExamples.begin(); it < positiveExamples.end(); ++it)
		count += it->size();
	return count;
}

bool FrameBasedLibSvmTraining::isTrainingReasonable() const {
	return getPositiveSampleCount() >= getRequiredPositiveSampleCount();
}

void FrameBasedLibSvmTraining::reset(LibSvmClassifier& svm) {
	vector<vector<struct svm_node *> >::iterator sit;
	for (sit = positiveExamples.begin(); sit < positiveExamples.end(); ++sit)
		freeExamples(*sit);
	for (sit = negativeExamples.begin(); sit < negativeExamples.end(); ++sit)
		freeExamples(*sit);
}

bool FrameBasedLibSvmTraining::retrain(LibSvmClassifier& svm,
		const vector<shared_ptr<FeatureVector> >& newPositiveExamples,
		const vector<shared_ptr<FeatureVector> >& newNegativeExamples) {
	addExamples(newPositiveExamples, newNegativeExamples);
	if (isTrainingReasonable())
		return train(svm);
	return false;
}

void FrameBasedLibSvmTraining::addExamples(const vector<shared_ptr<FeatureVector> >& newPositiveExamples,
		const vector<shared_ptr<FeatureVector> >& newNegativeExamples) {
	replaceExamples(positiveExamples[oldestEntry], newPositiveExamples);
	replaceExamples(negativeExamples[oldestEntry], newNegativeExamples);
	++oldestEntry;
	if (oldestEntry >= frameLength)
		oldestEntry = 0;
}

void FrameBasedLibSvmTraining::replaceExamples(vector<struct svm_node *>& examples,
		const vector<shared_ptr<FeatureVector> >& newExamples) {
	freeExamples(examples);
	for (vector<shared_ptr<FeatureVector> >::const_iterator fvit = newExamples.begin(); fvit != newExamples.end(); ++fvit) {
		shared_ptr<FeatureVector> featureVector = *fvit;
		dimensions = featureVector->getSize();
		examples.push_back(createNode(*featureVector));
	}
}

bool FrameBasedLibSvmTraining::train(LibSvmClassifier& svm) {
	vector<struct svm_node *> allPositiveExamples = collectNodes(positiveExamples);
	vector<struct svm_node *> allNegativeExamples = collectNodes(negativeExamples);
	return LibSvmTraining::train(svm, dimensions, allPositiveExamples, allNegativeExamples);
}

vector<struct svm_node *> FrameBasedLibSvmTraining::collectNodes(const vector<vector<struct svm_node *> >& nodes) {
	vector<struct svm_node *> allNodes;
	for (vector<vector<struct svm_node *> >::const_iterator fit = nodes.begin(); fit < nodes.end(); ++fit) {
		for (vector<struct svm_node *>::const_iterator nit = fit->begin(); nit < fit->end(); ++nit)
			allNodes.push_back(*nit);
	}
	return allNodes;
}

} /* namespace classification */

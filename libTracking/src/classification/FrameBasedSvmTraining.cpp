/*
 * FrameBasedSvmTraining.cpp
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#include "classification/FrameBasedSvmTraining.h"
#include "classification/FeatureVector.h"
#include <iostream>
#include <cmath>
#include <cstdlib>

namespace classification {

FrameBasedSvmTraining::FrameBasedSvmTraining(int frameLength, float minAvgSamples,
		shared_ptr<LibSvmParameterBuilder> parameterBuilder,
		shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation) :
				LibSvmTraining(parameterBuilder, sigmoidParameterComputation), frameLength(frameLength),
				minAvgSamples(minAvgSamples), dimensions(0), positiveTrainingSamples(frameLength),
				negativeTrainingSamples(frameLength), oldestEntry(0) {}

FrameBasedSvmTraining::FrameBasedSvmTraining(int frameLength, float minAvgSamples, std::string negativesFilename,
		int negatives, shared_ptr<LibSvmParameterBuilder> parameterBuilder,
		shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation) :
				LibSvmTraining(parameterBuilder, sigmoidParameterComputation), frameLength(frameLength),
				minAvgSamples(minAvgSamples), dimensions(0), positiveTrainingSamples(frameLength),
				negativeTrainingSamples(frameLength), oldestEntry(0) {
	readStaticNegatives(negativesFilename, negatives);
}

FrameBasedSvmTraining::~FrameBasedSvmTraining() {
	vector<vector<struct svm_node *> >::iterator sit;
	for (sit = positiveTrainingSamples.begin(); sit < positiveTrainingSamples.end(); ++sit)
		freeSamples(*sit);
	for (sit = negativeTrainingSamples.begin(); sit < negativeTrainingSamples.end(); ++sit)
		freeSamples(*sit);
}

int FrameBasedSvmTraining::getRequiredPositiveSampleCount() const {
	return (int)ceil((double)minAvgSamples * (double)frameLength);
}

int FrameBasedSvmTraining::getPositiveSampleCount() const {
	return getSampleCount(positiveTrainingSamples);
}

int FrameBasedSvmTraining::getNegativeSampleCount() const {
	return getSampleCount(negativeTrainingSamples) + staticNegativeTrainingSamples.size();
}

bool FrameBasedSvmTraining::isTrainingReasonable() const {
	return getPositiveSampleCount() >= getRequiredPositiveSampleCount();
}

void FrameBasedSvmTraining::reset(LibSvmClassifier& svm) {
	vector<vector<struct svm_node *> >::iterator sit;
	for (sit = positiveTrainingSamples.begin(); sit < positiveTrainingSamples.end(); ++sit)
		freeSamples(*sit);
	for (sit = negativeTrainingSamples.begin(); sit < negativeTrainingSamples.end(); ++sit)
		freeSamples(*sit);
}

bool FrameBasedSvmTraining::retrain(LibSvmClassifier& svm, const vector<shared_ptr<FeatureVector> >& positiveSamples,
		const vector<shared_ptr<FeatureVector> >& negativeSamples) {
	addSamples(positiveSamples, negativeSamples);
	if (isTrainingReasonable())
		return train(svm);
	return false;
}

void FrameBasedSvmTraining::addSamples(const vector<shared_ptr<FeatureVector> >& positiveSamples,
		const vector<shared_ptr<FeatureVector> >& negativeSamples) {
	replaceSamples(positiveTrainingSamples[oldestEntry], positiveSamples);
	replaceSamples(negativeTrainingSamples[oldestEntry], negativeSamples);
	++oldestEntry;
	if (oldestEntry >= frameLength)
		oldestEntry = 0;
}

void FrameBasedSvmTraining::replaceSamples(vector<struct svm_node *>& trainingSamples,
		const vector<shared_ptr<FeatureVector> >& samples) {
	freeSamples(trainingSamples);
	for (vector<shared_ptr<FeatureVector> >::const_iterator fvit = samples.begin(); fvit != samples.end(); ++fvit) {
		shared_ptr<FeatureVector> featureVector = *fvit;
		dimensions = featureVector->getSize();
		struct svm_node* data = new struct svm_node[dimensions + 1];
		for (unsigned int i = 0; i < dimensions; ++i) {
			data[i].index = i;
			data[i].value = featureVector->get(i);
		}
		data[featureVector->getSize()].index = -1;
		trainingSamples.push_back(data);
	}
}

int FrameBasedSvmTraining::getSampleCount(const vector<vector<struct svm_node *> >& samples) const {
	int count = 0;
	vector<vector<struct svm_node *> >::const_iterator it;
	for (it = samples.begin(); it < samples.end(); ++it)
		count += it->size();
	return count;
}

bool FrameBasedSvmTraining::train(LibSvmClassifier& svm) {
	unsigned int positiveCount = getPositiveSampleCount();
	unsigned int negativeCount = getNegativeSampleCount();
	unsigned int count = positiveCount + negativeCount;

	struct svm_parameter *param = createParameters(positiveCount, negativeCount);
	struct svm_problem *problem = createProblem(count);
	const char* message = svm_check_parameter(problem, param);
	if (message != 0) {
		std::cerr << "invalid SVM parameters: " << message << std::endl;
		svm_destroy_param(param);
		delete param;
		delete[] problem->x;
		delete[] problem->y;
		delete problem;
		return false;
	}
	struct svm_model *model = svm_train(problem, param);
	changeSvmParameters(svm, dimensions, model, problem, positiveCount, negativeCount);
	svm_destroy_param(param);
	delete param;
	delete[] problem->x;
	delete[] problem->y;
	delete problem;
	return true;
}

struct svm_problem *FrameBasedSvmTraining::createProblem(unsigned int count) {
	struct svm_problem *problem = new struct svm_problem;
	problem->l = count;
	problem->y = new double[count];
	problem->x = new struct svm_node *[count];
	int i = 0;
	vector<vector<struct svm_node *> >::iterator sit;
	for (sit = positiveTrainingSamples.begin(); sit < positiveTrainingSamples.end(); ++sit) {
		std::vector<struct svm_node *>::iterator dit;
		for (dit = sit->begin(); dit < sit->end(); ++dit) {
			problem->y[i] = 1;
			problem->x[i] = *dit;
			i++;
		}
	}
	for (sit = negativeTrainingSamples.begin(); sit < negativeTrainingSamples.end(); ++sit) {
		std::vector<struct svm_node *>::iterator dit;
		for (dit = sit->begin(); dit < sit->end(); ++dit) {
			problem->y[i] = -1;
			problem->x[i] = *dit;
			i++;
		}
	}
	std::vector<struct svm_node *>::iterator dit;
	for (dit = staticNegativeTrainingSamples.begin(); dit < staticNegativeTrainingSamples.end(); ++dit) {
		problem->y[i] = -1;
		problem->x[i] = *dit;
		i++;
	}
	return problem;
}

} /* namespace tracking */

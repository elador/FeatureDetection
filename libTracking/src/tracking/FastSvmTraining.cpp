/*
 * FastSvmTraining.cpp
 *
 *  Created on: 18.09.2012
 *      Author: poschmann
 */

#include "tracking/FastSvmTraining.h"
#include "FdPatch.h"
#include <iostream>
#include <cmath>
#include <limits>

namespace tracking {

FastSvmTraining::FastSvmTraining(unsigned int minPosCount, unsigned int minNegCount, unsigned int maxCount,
		shared_ptr<LibSvmParameterBuilder> parameterBuilder,
		shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation) :
				LibSvmTraining(parameterBuilder, sigmoidParameterComputation), minPosCount(minPosCount),
				minNegCount(minNegCount), maxCount(maxCount), positiveSamples(), negativeSamples() {}

FastSvmTraining::~FastSvmTraining() {
	freeSamples(positiveSamples);
	freeSamples(negativeSamples);
}

void FastSvmTraining::reset(ChangableDetectorSvm& svm) {
	freeSamples(positiveSamples);
	freeSamples(negativeSamples);
}

bool FastSvmTraining::retrain(ChangableDetectorSvm& svm, const std::vector<FdPatch*>& positivePatches,
		const std::vector<FdPatch*>& negativePatches) {
	if (positivePatches.empty() && negativePatches.empty()) {
		reset(svm);
		return false;
	}
	addSamples(positivePatches, negativePatches);
	if (isTrainingReasonable())
		return train(svm);
	return false;
}

void FastSvmTraining::addSamples(const std::vector<FdPatch*>& positivePatches,
		const std::vector<FdPatch*>& negativePatches) {
	addSamples(positiveSamples, positivePatches);
	addSamples(negativeSamples, negativePatches);
}

void FastSvmTraining::addSamples(std::vector<struct svm_node *>& samples, const std::vector<FdPatch*>& patches) {
	for (std::vector<FdPatch*>::const_iterator pit = patches.begin(); pit < patches.end(); ++pit) {
		FdPatch* patch = *pit;
		unsigned int dataLength = patch->w * patch->h;
		struct svm_node* data = new struct svm_node[dataLength + 1];
		for (unsigned int i = 0; i < dataLength; ++i) {
			data[i].index = i;
			data[i].value = patch->data[i] / 255.0;
		}
		data[dataLength].index = -1;
		samples.push_back(data);
	}
}

bool FastSvmTraining::isTrainingReasonable() const {
	unsigned int negativeCount = negativeSamples.size() + staticNegativeSamples.size();
	return positiveSamples.size() > minPosCount && negativeCount > minNegCount;
}

bool FastSvmTraining::train(ChangableDetectorSvm& svm) {
	struct svm_parameter *param = createParameters(
			positiveSamples.size(), negativeSamples.size() + staticNegativeSamples.size());
	struct svm_problem *problem = createProblem();
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
	unsigned int negativeCount = negativeSamples.size() + staticNegativeSamples.size();
	changeSvmParameters(svm, model, problem, positiveSamples.size(), negativeCount);
	std::vector<struct svm_node *> removedVectors = retainSupportVectors(model);
	freeSamples(removedVectors);
	svm_free_and_destroy_model(&model);
	svm_destroy_param(param);
	delete param;
	delete[] problem->x;
	delete[] problem->y;
	delete problem;
	return true;
}

struct svm_problem *FastSvmTraining::createProblem() {
	struct svm_problem *problem = new struct svm_problem;
	unsigned int count = positiveSamples.size() + negativeSamples.size() + staticNegativeSamples.size();
	problem->l = count;
	problem->y = new double[count];
	problem->x = new struct svm_node *[count];
	int i = 0;
	std::vector<struct svm_node *>::iterator sit;
	for (sit = positiveSamples.begin(); sit < positiveSamples.end(); ++sit) {
		problem->y[i] = 1;
		problem->x[i] = *sit;
		i++;
	}
	for (sit = negativeSamples.begin(); sit < negativeSamples.end(); ++sit) {
		problem->y[i] = -1;
		problem->x[i] = *sit;
		i++;
	}
	for (sit = staticNegativeSamples.begin(); sit < staticNegativeSamples.end(); ++sit) {
		problem->y[i] = -1;
		problem->x[i] = *sit;
		i++;
	}
	return problem;
}

std::vector<struct svm_node *> FastSvmTraining::retainSupportVectors(struct svm_model *model) {
	std::vector<struct svm_node *> removedSupportVectors;
	unsigned int count = positiveSamples.size() + negativeSamples.size();
	if (count <= maxCount)
		return removedSupportVectors;
	positiveSamples = extractSupportVectors(positiveSamples, model, model->nSV[0]);
	negativeSamples = extractSupportVectors(negativeSamples, model, model->nSV[1]);
	count = positiveSamples.size() + negativeSamples.size();
	if (count <= maxCount)
		return removedSupportVectors;
	// TODO test for using alphas instead of hyperplane distances for explusion
//	std::vector<double> positiveAlphas = getCoefficients(positiveSamples, model);
//	std::vector<double> negativeAlphas = getCoefficients(negativeSamples, model);
//	std::pair<unsigned int, double> minPositive = getMin(positiveAlphas);
//	std::pair<unsigned int, double> minNegative = getMin(negativeAlphas);
//	do {
//		// TODO parameters for values 3 and 5
//		bool insufficientNegativeSamples = 3 * positiveSamples.size() > negativeSamples.size();
//		bool insufficientPositiveSamples = 5 * positiveSamples.size() < negativeSamples.size();
//		if (insufficientNegativeSamples || (!insufficientPositiveSamples && minPositive.second < minNegative.second)) {
////		if (minPositive.second < minNegative.second) {
//			std::vector<struct svm_node *>::iterator sit = positiveSamples.begin() + minPositive.first;
//			removedSupportVectors.push_back(*sit);
//			positiveSamples.erase(sit);
//			positiveAlphas.erase(positiveAlphas.begin() + minPositive.first);
//			minPositive = getMin(positiveAlphas);
//		} else {
//			std::vector<struct svm_node *>::iterator sit = negativeSamples.begin() + minNegative.first;
//			removedSupportVectors.push_back(*sit);
//			negativeSamples.erase(sit);
//			negativeAlphas.erase(negativeAlphas.begin() + minNegative.first);
//			minNegative = getMin(negativeAlphas);
//		}
//		--count;
//	} while (count > maxCount);
	std::vector<double> positiveDistances = computeHyperplaneDistances(positiveSamples, model);
	std::vector<double> negativeDistances = computeHyperplaneDistances(negativeSamples, model);
	std::pair<unsigned int, double> maxPositive = getMax(positiveDistances);
	std::pair<unsigned int, double> maxNegative = getMax(negativeDistances);
	do {
		if (maxPositive.second > maxNegative.second) {
			std::vector<struct svm_node *>::iterator sit = positiveSamples.begin() + maxPositive.first;
			removedSupportVectors.push_back(*sit);
			positiveSamples.erase(sit);
			positiveDistances.erase(positiveDistances.begin() + maxPositive.first);
			maxPositive = getMax(positiveDistances);
		} else {
			std::vector<struct svm_node *>::iterator sit = negativeSamples.begin() + maxNegative.first;
			removedSupportVectors.push_back(*sit);
			negativeSamples.erase(sit);
			negativeDistances.erase(negativeDistances.begin() + maxNegative.first);
			maxNegative = getMax(negativeDistances);
		}
		--count;
	} while (count > maxCount);
	return removedSupportVectors;
}

std::vector<struct svm_node *> FastSvmTraining::extractSupportVectors(
		std::vector<struct svm_node *>& samples, struct svm_model *model, unsigned int count) {
	std::vector<struct svm_node *> supportVectors;
	supportVectors.reserve(count);
	for (std::vector<struct svm_node *>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		if (isSupportVector(*sit, model))
			supportVectors.push_back(*sit);
		else
			delete[] *sit;
	}
	samples.clear();
	return supportVectors;
}

bool FastSvmTraining::isSupportVector(struct svm_node *vector, struct svm_model *model) {
	for (int i = 0; i < model->l; ++i) {
		if (model->SV[i] == vector)
			return true;
	}
	return false;
}

// TODO only used for alphas
std::vector<double> FastSvmTraining::getCoefficients(std::vector<struct svm_node *>& supportVectors, struct svm_model *model) {
	std::vector<double> coefficients;
	coefficients.reserve(supportVectors.size());
	for (std::vector<struct svm_node *>::iterator sit = supportVectors.begin(); sit < supportVectors.end(); ++sit) {
		for (int i = 0; i < model->l; ++i) {
			if (model->SV[i] == *sit) {
				coefficients.push_back(fabs(model->sv_coef[0][i]));
				break;
			}
		}
	}
	return coefficients;
}

// TODO only used for hyperplane distances
std::vector<double> FastSvmTraining::computeHyperplaneDistances(
		std::vector<struct svm_node *>& samples, struct svm_model *model) {
	std::vector<double> distances;
	distances.reserve(samples.size());
	double* dec_values = new double[1];
	for (std::vector<struct svm_node *>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		svm_predict_values(model, *sit, dec_values);
		distances.push_back(fabs(dec_values[0]));
	}
	delete[] dec_values;
	return distances;
}

// TODO only used for alphas
std::pair<unsigned int, double> FastSvmTraining::getMin(std::vector<double> values) {
	unsigned int minIndex = -1;
	double minValue = std::numeric_limits<double>::max();
	for (unsigned int i = 0; i < values.size(); ++i) {
		if (values[i] < minValue) {
			minIndex = i;
			minValue = values[i];
		}
	}
	return std::make_pair(minIndex, minValue);
}

// TODO only used for hyperplane distances
std::pair<unsigned int, double> FastSvmTraining::getMax(std::vector<double> values) {
	unsigned int maxIndex = -1;
	double maxValue = 0;
	for (unsigned int i = 0; i < values.size(); ++i) {
		if (values[i] > maxValue) {
			maxIndex = i;
			maxValue = values[i];
		}
	}
	return std::make_pair(maxIndex, maxValue);
}

} /* namespace tracking */

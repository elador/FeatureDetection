/*
 * FastSvmTraining.cpp
 *
 *  Created on: 18.09.2012
 *      Author: poschmann
 */

#include "classification/FastSvmTraining.h"
#include "classification/FeatureVector.h"
#include <iostream>
#include <cmath>
#include <limits>

namespace classification {

FastSvmTraining::FastSvmTraining(unsigned int minPosCount, unsigned int minNegCount, unsigned int maxCount,
		shared_ptr<LibSvmParameterBuilder> parameterBuilder,
		shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation) :
				LibSvmTraining(parameterBuilder, sigmoidParameterComputation), minPosCount(minPosCount),
				minNegCount(minNegCount), maxCount(maxCount), dimensions(0),
				positiveTrainingSamples(), negativeTrainingSamples() {}

FastSvmTraining::~FastSvmTraining() {
	freeSamples(positiveTrainingSamples);
	freeSamples(negativeTrainingSamples);
}

void FastSvmTraining::reset(LibSvmClassifier& svm) {
	freeSamples(positiveTrainingSamples);
	freeSamples(negativeTrainingSamples);
}

bool FastSvmTraining::retrain(LibSvmClassifier& svm, const vector<shared_ptr<FeatureVector> >& positiveSamples,
		const vector<shared_ptr<FeatureVector> >& negativeSamples) {
	if (positiveSamples.empty() && negativeSamples.empty()) {
		reset(svm);
		return false;
	}
	addSamples(positiveSamples, negativeSamples);
	if (isTrainingReasonable())
		return train(svm);
	return false;
}

void FastSvmTraining::addSamples(const vector<shared_ptr<FeatureVector> >& positiveSamples,
		const vector<shared_ptr<FeatureVector> >& negativeSamples) {
	addSamples(positiveTrainingSamples, positiveSamples);
	addSamples(negativeTrainingSamples, negativeSamples);
}

void FastSvmTraining::addSamples(vector<struct svm_node *>& trainingSamples,
		const vector<shared_ptr<FeatureVector> >& samples) {
	for (vector<shared_ptr<FeatureVector> >::const_iterator fvit = samples.begin(); fvit != samples.end(); ++fvit) {
		shared_ptr<FeatureVector> featureVector = *fvit;
		dimensions = featureVector->getSize();
		const float* values = featureVector->getValues();
		struct svm_node* data = new struct svm_node[dimensions + 1];
		for (unsigned int i = 0; i < dimensions; ++i) {
			data[i].index = i;
			data[i].value = values[i];
		}
		data[featureVector->getSize()].index = -1;
		trainingSamples.push_back(data);
	}
}

bool FastSvmTraining::isTrainingReasonable() const {
	unsigned int negativeCount = negativeTrainingSamples.size() + staticNegativeTrainingSamples.size();
	return positiveTrainingSamples.size() > minPosCount && negativeCount > minNegCount;
}

bool FastSvmTraining::train(LibSvmClassifier& svm) {
	struct svm_parameter *param = createParameters(
			positiveTrainingSamples.size(), negativeTrainingSamples.size() + staticNegativeTrainingSamples.size());
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
	unsigned int negativeCount = negativeTrainingSamples.size() + staticNegativeTrainingSamples.size();
	changeSvmParameters(svm, dimensions, model, problem, positiveTrainingSamples.size(), negativeCount);
	vector<struct svm_node *> removedVectors = retainSupportVectors(model);
	freeSamples(removedVectors);
	svm_destroy_param(param);
	delete param;
	delete[] problem->x;
	delete[] problem->y;
	delete problem;
	return true;
}

struct svm_problem *FastSvmTraining::createProblem() {
	struct svm_problem *problem = new struct svm_problem;
	unsigned int count = positiveTrainingSamples.size() + negativeTrainingSamples.size() + staticNegativeTrainingSamples.size();
	problem->l = count;
	problem->y = new double[count];
	problem->x = new struct svm_node *[count];
	int i = 0;
	vector<struct svm_node *>::iterator sit;
	for (sit = positiveTrainingSamples.begin(); sit < positiveTrainingSamples.end(); ++sit) {
		problem->y[i] = 1;
		problem->x[i] = *sit;
		i++;
	}
	for (sit = negativeTrainingSamples.begin(); sit < negativeTrainingSamples.end(); ++sit) {
		problem->y[i] = -1;
		problem->x[i] = *sit;
		i++;
	}
	for (sit = staticNegativeTrainingSamples.begin(); sit < staticNegativeTrainingSamples.end(); ++sit) {
		problem->y[i] = -1;
		problem->x[i] = *sit;
		i++;
	}
	return problem;
}

vector<struct svm_node *> FastSvmTraining::retainSupportVectors(struct svm_model *model) {
	vector<struct svm_node *> removedSupportVectors;
	unsigned int count = positiveTrainingSamples.size() + negativeTrainingSamples.size();
	if (count <= maxCount)
		return removedSupportVectors;
	positiveTrainingSamples = extractSupportVectors(positiveTrainingSamples, model, model->nSV[0]);
	negativeTrainingSamples = extractSupportVectors(negativeTrainingSamples, model, model->nSV[1]);
	count = positiveTrainingSamples.size() + negativeTrainingSamples.size();
	if (count <= maxCount)
		return removedSupportVectors;
	// TODO test for using alphas instead of hyperplane distances for explusion
//	vector<double> positiveAlphas = getCoefficients(positiveTrainingSamples, model);
//	vector<double> negativeAlphas = getCoefficients(negativeTrainingSamples, model);
//	std::pair<unsigned int, double> minPositive = getMin(positiveAlphas);
//	std::pair<unsigned int, double> minNegative = getMin(negativeAlphas);
//	do {
//		// TODO parameters for values 3 and 5
//		bool insufficientNegativeSamples = 3 * positiveTrainingSamples.size() > negativeTrainingSamples.size();
//		bool insufficientPositiveSamples = 5 * positiveTrainingSamples.size() < negativeTrainingSamples.size();
//		if (insufficientNegativeSamples || (!insufficientPositiveSamples && minPositive.second < minNegative.second)) {
////		if (minPositive.second < minNegative.second) {
//			vector<struct svm_node *>::iterator sit = positiveTrainingSamples.begin() + minPositive.first;
//			removedSupportVectors.push_back(*sit);
//			positiveSamples.erase(sit);
//			positiveAlphas.erase(positiveAlphas.begin() + minPositive.first);
//			minPositive = getMin(positiveAlphas);
//		} else {
//			vector<struct svm_node *>::iterator sit = negativeTrainingSamples.begin() + minNegative.first;
//			removedSupportVectors.push_back(*sit);
//			negativeSamples.erase(sit);
//			negativeAlphas.erase(negativeAlphas.begin() + minNegative.first);
//			minNegative = getMin(negativeAlphas);
//		}
//		--count;
//	} while (count > maxCount);
	vector<double> positiveDistances = computeHyperplaneDistances(positiveTrainingSamples, model);
	vector<double> negativeDistances = computeHyperplaneDistances(negativeTrainingSamples, model);
	std::pair<unsigned int, double> maxPositive = getMax(positiveDistances);
	std::pair<unsigned int, double> maxNegative = getMax(negativeDistances);
	do {
		if (maxPositive.second > maxNegative.second) {
			vector<struct svm_node *>::iterator sit = positiveTrainingSamples.begin() + maxPositive.first;
			removedSupportVectors.push_back(*sit);
			positiveTrainingSamples.erase(sit);
			positiveDistances.erase(positiveDistances.begin() + maxPositive.first);
			maxPositive = getMax(positiveDistances);
		} else {
			vector<struct svm_node *>::iterator sit = negativeTrainingSamples.begin() + maxNegative.first;
			removedSupportVectors.push_back(*sit);
			negativeTrainingSamples.erase(sit);
			negativeDistances.erase(negativeDistances.begin() + maxNegative.first);
			maxNegative = getMax(negativeDistances);
		}
		--count;
	} while (count > maxCount);
	return removedSupportVectors;
}

vector<struct svm_node *> FastSvmTraining::extractSupportVectors(
		vector<struct svm_node *>& samples, struct svm_model *model, unsigned int count) {
	vector<struct svm_node *> supportVectors;
	supportVectors.reserve(count);
	for (vector<struct svm_node *>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
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
vector<double> FastSvmTraining::getCoefficients(vector<struct svm_node *>& supportVectors, struct svm_model *model) {
	vector<double> coefficients;
	coefficients.reserve(supportVectors.size());
	for (vector<struct svm_node *>::iterator sit = supportVectors.begin(); sit < supportVectors.end(); ++sit) {
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
vector<double> FastSvmTraining::computeHyperplaneDistances(
		vector<struct svm_node *>& samples, struct svm_model *model) {
	vector<double> distances;
	distances.reserve(samples.size());
	double* dec_values = new double[1];
	for (vector<struct svm_node *>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		svm_predict_values(model, *sit, dec_values);
		distances.push_back(fabs(dec_values[0]));
	}
	delete[] dec_values;
	return distances;
}

// TODO only used for alphas
std::pair<unsigned int, double> FastSvmTraining::getMin(vector<double> values) {
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
std::pair<unsigned int, double> FastSvmTraining::getMax(vector<double> values) {
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

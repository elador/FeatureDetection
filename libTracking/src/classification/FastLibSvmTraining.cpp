/*
 * FastLibSvmTraining.cpp
 *
 *  Created on: 18.09.2012
 *      Author: poschmann
 */

#include "classification/FastLibSvmTraining.h"
#include "classification/FeatureVector.h"
#include <cmath>
#include <limits>

namespace classification {

FastLibSvmTraining::FastLibSvmTraining(unsigned int minPosCount, unsigned int minNegCount, unsigned int maxCount,
		shared_ptr<LibSvmParameterBuilder> parameterBuilder,
		shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation) :
				LibSvmTraining(parameterBuilder, sigmoidParameterComputation), minPosCount(minPosCount),
				minNegCount(minNegCount), maxCount(maxCount), dimensions(0), positiveExamples(), negativeExamples() {}

FastLibSvmTraining::~FastLibSvmTraining() {
	freeExamples(positiveExamples);
	freeExamples(negativeExamples);
}

void FastLibSvmTraining::reset(LibSvmClassifier& svm) {
	freeExamples(positiveExamples);
	freeExamples(negativeExamples);
}

bool FastLibSvmTraining::retrain(LibSvmClassifier& svm, const vector<shared_ptr<FeatureVector> >& positiveExamples,
		const vector<shared_ptr<FeatureVector> >& negativeExamples) {
	if (positiveExamples.empty() && negativeExamples.empty()) {
		reset(svm);
		return false;
	}
	addExamples(positiveExamples, negativeExamples);
	if (isTrainingReasonable())
		return train(svm);
	return false;
}

void FastLibSvmTraining::addExamples(const vector<shared_ptr<FeatureVector> >& newPositiveExamples,
		const vector<shared_ptr<FeatureVector> >& newNegativeSamples) {
	addExamples(positiveExamples, newPositiveExamples);
	addExamples(negativeExamples, newNegativeSamples);
}

void FastLibSvmTraining::addExamples(vector<struct svm_node *>& examples,
		const vector<shared_ptr<FeatureVector> >& newExamples) {
	for (vector<shared_ptr<FeatureVector> >::const_iterator fvit = newExamples.begin(); fvit != newExamples.end(); ++fvit) {
		shared_ptr<FeatureVector> featureVector = *fvit;
		dimensions = featureVector->getSize();
		examples.push_back(createNode(*featureVector));
	}
}

bool FastLibSvmTraining::isTrainingReasonable() const {
	return positiveExamples.size() > minPosCount && negativeExamples.size() + getStaticNegativeCount() > minNegCount;
}

bool FastLibSvmTraining::train(LibSvmClassifier& svm) {
	bool successful = LibSvmTraining::train(svm, dimensions, positiveExamples, negativeExamples);
	vector<struct svm_node *> removedVectors = retainSupportVectors(svm.getModel());
	freeExamples(removedVectors);
	return successful;
}

vector<struct svm_node *> FastLibSvmTraining::retainSupportVectors(struct svm_model *model) {
	vector<struct svm_node *> removedSupportVectors;
	unsigned int count = positiveExamples.size() + negativeExamples.size();
	if (count <= maxCount)
		return removedSupportVectors;
	positiveExamples = extractSupportVectors(positiveExamples, model, model->nSV[0]);
	negativeExamples = extractSupportVectors(negativeExamples, model, model->nSV[1]);
	count = positiveExamples.size() + negativeExamples.size();
	if (count <= maxCount)
		return removedSupportVectors;
	// TODO test for using alphas instead of hyperplane distances for explusion
//	vector<double> positiveAlphas = getCoefficients(positiveExamples, model);
//	vector<double> negativeAlphas = getCoefficients(negativeExamples, model);
//	std::pair<unsigned int, double> minPositive = getMin(positiveAlphas);
//	std::pair<unsigned int, double> minNegative = getMin(negativeAlphas);
//	do {
//		// TODO parameters for values 3 and 5
//		bool insufficientNegativeSamples = 3 * positiveExamples.size() > negativeExamples.size();
//		bool insufficientPositiveSamples = 5 * positiveExamples.size() < negativeExamples.size();
//		if (insufficientNegativeSamples || (!insufficientPositiveSamples && minPositive.second < minNegative.second)) {
////		if (minPositive.second < minNegative.second) {
//			vector<struct svm_node *>::iterator sit = positiveExamples.begin() + minPositive.first;
//			removedSupportVectors.push_back(*sit);
//			positiveSamples.erase(sit);
//			positiveAlphas.erase(positiveAlphas.begin() + minPositive.first);
//			minPositive = getMin(positiveAlphas);
//		} else {
//			vector<struct svm_node *>::iterator sit = negativeExamples.begin() + minNegative.first;
//			removedSupportVectors.push_back(*sit);
//			negativeSamples.erase(sit);
//			negativeAlphas.erase(negativeAlphas.begin() + minNegative.first);
//			minNegative = getMin(negativeAlphas);
//		}
//		--count;
//	} while (count > maxCount);
	vector<double> positiveDistances = computeHyperplaneDistances(positiveExamples, model);
	vector<double> negativeDistances = computeHyperplaneDistances(negativeExamples, model);
	std::pair<unsigned int, double> maxPositive = getMax(positiveDistances);
	std::pair<unsigned int, double> maxNegative = getMax(negativeDistances);
	do {
		if (maxPositive.second > maxNegative.second) {
			vector<struct svm_node *>::iterator sit = positiveExamples.begin() + maxPositive.first;
			removedSupportVectors.push_back(*sit);
			positiveExamples.erase(sit);
			positiveDistances.erase(positiveDistances.begin() + maxPositive.first);
			maxPositive = getMax(positiveDistances);
		} else {
			vector<struct svm_node *>::iterator sit = negativeExamples.begin() + maxNegative.first;
			removedSupportVectors.push_back(*sit);
			negativeExamples.erase(sit);
			negativeDistances.erase(negativeDistances.begin() + maxNegative.first);
			maxNegative = getMax(negativeDistances);
		}
		--count;
	} while (count > maxCount);
	return removedSupportVectors;
}

vector<struct svm_node *> FastLibSvmTraining::extractSupportVectors(
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

bool FastLibSvmTraining::isSupportVector(struct svm_node *vector, struct svm_model *model) {
	for (int i = 0; i < model->l; ++i) {
		if (model->SV[i] == vector)
			return true;
	}
	return false;
}

// TODO only used for alphas
vector<double> FastLibSvmTraining::getCoefficients(vector<struct svm_node *>& supportVectors, struct svm_model *model) {
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
vector<double> FastLibSvmTraining::computeHyperplaneDistances(
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
std::pair<unsigned int, double> FastLibSvmTraining::getMin(vector<double> values) {
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
std::pair<unsigned int, double> FastLibSvmTraining::getMax(vector<double> values) {
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

} /* namespace classification */

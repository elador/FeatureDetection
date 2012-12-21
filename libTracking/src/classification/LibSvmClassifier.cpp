/*
 * LibSvmClassifier.cpp
 *
 *  Created on: 20.11.2012
 *      Author: poschmann
 */

#include "classification/LibSvmClassifier.h"
#include "classification/FeatureVector.h"
#include <cmath>
#include <iostream>

namespace classification {

static double dot(const FeatureVector& x, const FeatureVector& y) {
	float sum = 0;
	for (int i = 0, n = (int)x.getSize(); i < n; ++i)
		sum += x[i] * y[i];
	return sum;
}

static inline double powi(double base, int times) {
	double tmp = base, ret = 1.0;
	for (int t = times; t > 0; t /= 2) {
		if (t % 2 == 1)
			ret *= tmp;
		tmp = tmp * tmp;
	}
	return ret;
}

LibSvmClassifier::LibSvmClassifier(shared_ptr<Training<LibSvmClassifier> > training) :
		training(training), model(0), supportVectors(), probParamA(0), probParamB(0) {}

LibSvmClassifier::~LibSvmClassifier() {
	deleteModel();
}

pair<bool, double> LibSvmClassifier::classify(const FeatureVector& featureVector) const {
	if (model == 0)
		return std::make_pair(false, 0);
	double *sv_coef = model->sv_coef[0];
	double svmOutput = -model->rho[0];
	for (int i = 0; i < model->l; i++)
		svmOutput += sv_coef[i] * kernel(featureVector, supportVectors[i], model->param);
	double probability = 1.0 / (1.0 + exp(probParamA * svmOutput + probParamB));
	return std::make_pair(svmOutput > 0, probability);
}

double LibSvmClassifier::kernel(const FeatureVector& x, const FeatureVector& y, const svm_parameter& param) const {
	if (param.kernel_type == LINEAR)
		return dot(x, y);
	if (param.kernel_type == POLY)
		return powi(param.gamma * dot(x, y) + param.coef0, param.degree);
	if (param.kernel_type == RBF) {
		float sum = 0;
		for (int i = 0, n = (int)x.getSize(); i < n; ++i) {
			float d = x[i] - y[i];
			sum += d * d;
		}
		return exp(-param.gamma * sum);
	}
	return 0;
}

bool LibSvmClassifier::retrain(const vector<shared_ptr<FeatureVector> >& positiveExamples,
			const vector<shared_ptr<FeatureVector> >& negativeExamples) {
	return training->retrain(*this, positiveExamples, negativeExamples);
}

void LibSvmClassifier::reset() {
	training->reset(*this);
}

void LibSvmClassifier::setModel(int dimensions, svm_model *model, double probParamA, double probParamB) {
	if (model->param.svm_type != C_SVC)
		std::cerr << "the SVM model must be of type C_SVC" << std::endl;
	if (model->param.kernel_type != LINEAR && model->param.kernel_type != POLY && model->param.kernel_type != RBF)
		std::cerr << "the SVM kernel type must be LINEAR, POLY or RBF" << std::endl;
	if (model->nr_class != 2)
		std::cerr << "the SVM must distinguish between only two classes" << std::endl;
	deleteModel();
	this->model = model;
	supportVectors.clear();
	supportVectors.reserve(model->l);
	for (int i = 0; i < model->l; ++i) {
		FeatureVector supportVector(dimensions);
		const svm_node *sv = model->SV[i];
		for (int j = 0; j < (int)supportVector.getSize(); ++j) {
			if (sv->index == j) {
				supportVector[j] = sv->value;
				++sv;
			} else {
				supportVector[j] = 0;
			}
		}
		supportVectors.push_back(supportVector);
	}
	this->probParamA = probParamA;
	this->probParamB = probParamB;
}

void LibSvmClassifier::deleteModel() {
	if (model != 0) {
		svm_free_and_destroy_model(&model);
		model = 0;
	}
}

} /* namespace classification */

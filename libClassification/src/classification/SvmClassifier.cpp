/*
 * SvmClassifier.cpp
 *
 *  Created on: 17.02.2013
 *      Author: Patrik Huber
 */

#include "classification/SvmClassifier.hpp"

using cv::Mat;
using std::pair;
using std::vector;
using std::shared_ptr;
using std::make_pair;

namespace classification {

SvmClassifier::SvmClassifier(shared_ptr<Kernel> kernel) :
		VectorMachineClassifier(kernel), supportVectors(), coefficients() {}

bool SvmClassifier::classify(const Mat& featureVector) const {
	return classify(computeHyperplaneDistance(featureVector));
}

pair<bool, double> SvmClassifier::getConfidence(const Mat& featureVector) const {
	return getConfidence(computeHyperplaneDistance(featureVector));
}

bool SvmClassifier::classify(double hyperplaneDistance) const {
	return hyperplaneDistance >= threshold;
}

pair<bool, double> SvmClassifier::getConfidence(double hyperplaneDistance) const {
	if (classify(hyperplaneDistance))
		return make_pair(true, hyperplaneDistance);
	else
		return make_pair(false, -hyperplaneDistance);
}

double SvmClassifier::computeHyperplaneDistance(const Mat& featureVector) const {
	double distance = -bias;
	for (size_t i = 0; i < supportVectors.size(); ++i)
		distance += coefficients[i] * kernel->compute(featureVector, supportVectors[i]);
	return distance;
}

void SvmClassifier::setSvmParameters(vector<Mat> supportVectors, vector<float> coefficients, double bias) {
	this->supportVectors = supportVectors;
	this->coefficients = coefficients;
	this->bias = bias;
}

} /* namespace classification */

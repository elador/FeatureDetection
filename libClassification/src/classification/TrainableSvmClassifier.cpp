/*
 * TrainableSvmClassifier.cpp
 *
 *  Created on: 05.03.2013
 *      Author: poschmann
 */

#include "classification/TrainableSvmClassifier.hpp"
#include "classification/SvmClassifier.hpp"
#include "classification/Kernel.hpp"

using cv::Mat;
using std::pair;
using std::shared_ptr;
using std::make_shared;

namespace classification {

TrainableSvmClassifier::TrainableSvmClassifier(shared_ptr<SvmClassifier> svm) :
		svm(svm), usable(false) {}

TrainableSvmClassifier::TrainableSvmClassifier(shared_ptr<Kernel> kernel) :
		svm(make_shared<SvmClassifier>(kernel)), usable(false) {}

TrainableSvmClassifier::~TrainableSvmClassifier() {}

bool TrainableSvmClassifier::classify(const Mat& featureVector) const {
	return svm->classify(featureVector);
}

pair<bool, double> TrainableSvmClassifier::getConfidence(const Mat& featureVector) const {
	return svm->getConfidence(featureVector);
}

bool TrainableSvmClassifier::isUsable() const {
	return usable;
}

shared_ptr<SvmClassifier> TrainableSvmClassifier::getSvm() {
	return svm;
}

const shared_ptr<SvmClassifier> TrainableSvmClassifier::getSvm() const {
	return svm;
}

} /* namespace classification */

/*
 * PositionDependentMeasurementModel.cpp
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#include "tracking/PositionDependentMeasurementModel.h"
#include "tracking/Sample.h"
#include "classification/FeatureVector.h"
#include "classification/FeatureExtractor.h"
#include "classification/LibSvmClassifier.h"
#include "classification/LibSvmTraining.h"
#include "boost/make_shared.hpp"
#include "boost/unordered_map.hpp"
#include <utility>

using boost::make_shared;
using boost::unordered_map;
using std::pair;

namespace tracking {

PositionDependentMeasurementModel::PositionDependentMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<LibSvmClassifier> classifier, shared_ptr<LibSvmTraining> training) :
				featureExtractor(featureExtractor),
				classifier(classifier),
				training(training),
				usable(false) {}

PositionDependentMeasurementModel::~PositionDependentMeasurementModel() {}

void PositionDependentMeasurementModel::evaluate(const Mat& image, vector<Sample>& samples) {
	featureExtractor->init(image);
	unordered_map<shared_ptr<FeatureVector>, pair<bool, double> > results;
	for (vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		sit->setObject(false);
		shared_ptr<FeatureVector> featureVector = featureExtractor->extract(sit->getX(), sit->getY(), sit->getSize());
		if (!featureVector) {
			sit->setWeight(0);
		} else {
			pair<bool, double> result;
			unordered_map<shared_ptr<FeatureVector>, pair<bool, double> >::iterator rit = results.find(featureVector);
			if (rit == results.end()) {
				result = classifier->classify(*featureVector);
				pair<const shared_ptr<FeatureVector>, pair<bool, double> > entry = make_pair(featureVector, result);
				results.insert(entry);
			} else {
				result = rit->second;
			}
			sit->setObject(result.first);
			sit->setWeight(result.second);
		}
	}
}

void PositionDependentMeasurementModel::adapt(const Mat& image, const vector<Sample>& samples, const Sample& target) {
	vector<Sample> positiveSamples;
	int offset = std::max(1, target.getSize() / 20);
	positiveSamples.push_back(target);
	positiveSamples.push_back(Sample(target.getX() - offset, target.getY(), target.getSize()));
	positiveSamples.push_back(Sample(target.getX() + offset, target.getY(), target.getSize()));
	positiveSamples.push_back(Sample(target.getX(), target.getY() - offset, target.getSize()));
	positiveSamples.push_back(Sample(target.getX(), target.getY() + offset, target.getSize()));

	vector<Sample> negativeSamples;
	double deviationFactor = 0.5;
	int boundOffset = (int)(deviationFactor * target.getSize());
	int xLowBound = target.getX() - boundOffset;
	int xHighBound = target.getX() + boundOffset;
	int yLowBound = target.getY() - boundOffset;
	int yHighBound = target.getY() + boundOffset;
	int sizeLowBound = (int)((1 - deviationFactor) * target.getSize());
	int sizeHighBound = (int)((1 + 2 * deviationFactor) * target.getSize());

	negativeSamples.push_back(Sample(xLowBound, target.getY(), target.getSize()));
	negativeSamples.push_back(Sample(xHighBound, target.getY(), target.getSize()));
	negativeSamples.push_back(Sample(target.getX(), yLowBound, target.getSize()));
	negativeSamples.push_back(Sample(target.getX(), yHighBound, target.getSize()));
	negativeSamples.push_back(Sample(xLowBound, yLowBound, target.getSize()));
	negativeSamples.push_back(Sample(xLowBound, yHighBound, target.getSize()));
	negativeSamples.push_back(Sample(xHighBound, yLowBound, target.getSize()));
	negativeSamples.push_back(Sample(xHighBound, yHighBound, target.getSize()));

	negativeSamples.push_back(Sample(target.getX(), target.getY(), sizeLowBound));
	negativeSamples.push_back(Sample(xLowBound, target.getY(), sizeLowBound));
	negativeSamples.push_back(Sample(xHighBound, target.getY(), sizeLowBound));
	negativeSamples.push_back(Sample(target.getX(), yLowBound, sizeLowBound));
	negativeSamples.push_back(Sample(target.getX(), yHighBound, sizeLowBound));
//		negativeSamples.push_back(Sample(xLowBound, yLowBound, sizeLowBound));
//		negativeSamples.push_back(Sample(xLowBound, yHighBound, sizeLowBound));
//		negativeSamples.push_back(Sample(xHighBound, yLowBound, sizeLowBound));
//		negativeSamples.push_back(Sample(xHighBound, yHighBound, sizeLowBound));

	negativeSamples.push_back(Sample(target.getX(), target.getY(), sizeHighBound));
	negativeSamples.push_back(Sample(xLowBound, target.getY(), sizeHighBound));
	negativeSamples.push_back(Sample(xHighBound, target.getY(), sizeHighBound));
	negativeSamples.push_back(Sample(target.getX(), yLowBound, sizeHighBound));
	negativeSamples.push_back(Sample(target.getX(), yHighBound, sizeHighBound));
//		negativeSamples.push_back(Sample(xLowBound, yLowBound, sizeHighBound));
//		negativeSamples.push_back(Sample(xLowBound, yHighBound, sizeHighBound));
//		negativeSamples.push_back(Sample(xHighBound, yLowBound, sizeHighBound));
//		negativeSamples.push_back(Sample(xHighBound, yHighBound, sizeHighBound));

	for (vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		if (sit->isObject()
				&& (sit->getX() <= xLowBound || sit->getX() >= xHighBound
				|| sit->getY() <= yLowBound || sit->getY() >= yHighBound
				|| sit->getSize() <= sizeLowBound || sit->getSize() >= sizeHighBound)) {
			negativeSamples.push_back(*sit);
		}
	}

	if (!isUsable()) // feature extractor has to be initialized on the image when this model was not used for evaluation
		featureExtractor->init(image);
	usable = training->retrain(*classifier,
			getFeatureVectors(positiveSamples), getFeatureVectors(negativeSamples));
}

void PositionDependentMeasurementModel::reset() {
	training->reset(*classifier);
	usable = false;
}

void PositionDependentMeasurementModel::adapt(const Mat& image, const vector<Sample>& samples) {
	const vector<shared_ptr<FeatureVector> > empty;
	usable = training->retrain(*classifier, empty, empty);
}

vector<shared_ptr<FeatureVector> > PositionDependentMeasurementModel::getFeatureVectors(vector<Sample>& samples) {
	vector<shared_ptr<FeatureVector> > trainingSamples;
	trainingSamples.reserve(samples.size());
	for (vector<Sample>::iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		shared_ptr<FeatureVector> featureVector = featureExtractor->extract(sit->getX(), sit->getY(), sit->getSize());
		if (featureVector)
			trainingSamples.push_back(featureVector);
	}
	return trainingSamples;
}

} /* namespace tracking */

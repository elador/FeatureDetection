/*
 * SelfLearningMeasurementModel.cpp
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#include "tracking/SelfLearningMeasurementModel.h"
#include "tracking/Sample.h"
#include "classification/FeatureVector.h"
#include "classification/FeatureExtractor.h"
#include "classification/LibSvmClassifier.h"
#include "classification/LibSvmTraining.h"
#include "boost/unordered_map.hpp"
#include <algorithm>

using boost::unordered_map;
using std::make_pair;
using std::sort;
using std::reverse;

namespace tracking {

static bool compareFeatureProbabilityPairs(pair<shared_ptr<FeatureVector>, double> a,
		pair<shared_ptr<FeatureVector>, double> b) {
	return a.second < b.second;
}

SelfLearningMeasurementModel::SelfLearningMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<LibSvmClassifier> classifier, shared_ptr<LibSvmTraining> training,
		double positiveThreshold, double negativeThreshold) :
				featureExtractor(featureExtractor),
				classifier(classifier),
				training(training),
				usable(false),
				positiveThreshold(positiveThreshold),
				negativeThreshold(negativeThreshold),
				positiveTrainingSamples(),
				negativeTrainingSamples() {}

SelfLearningMeasurementModel::~SelfLearningMeasurementModel() {}

void SelfLearningMeasurementModel::evaluate(const Mat& image, vector<Sample>& samples) {
	positiveTrainingSamples.clear();
	negativeTrainingSamples.clear();
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
			if (result.second > positiveThreshold)
				positiveTrainingSamples.push_back(make_pair(featureVector, result.second));
			else if (result.second < negativeThreshold)
				negativeTrainingSamples.push_back(make_pair(featureVector, result.second));
			sit->setWeight(result.second);
		}
	}

	if (positiveTrainingSamples.size() > 10) {
		sort(positiveTrainingSamples.begin(), positiveTrainingSamples.end(), compareFeatureProbabilityPairs);
		positiveTrainingSamples.resize(10);
	}
	if (negativeTrainingSamples.size() > 10) {
		sort(negativeTrainingSamples.begin(), negativeTrainingSamples.end(), compareFeatureProbabilityPairs);
		reverse(negativeTrainingSamples.begin(), negativeTrainingSamples.end());
		negativeTrainingSamples.resize(10);
	}
}

void SelfLearningMeasurementModel::reset() {
	training->reset(*classifier);
	usable = false;
}

void SelfLearningMeasurementModel::adapt(const Mat& image, const vector<Sample>& samples, const Sample& target) {
	adapt(image, samples);
}

void SelfLearningMeasurementModel::adapt(const Mat& image, const vector<Sample>& samples) {
	if (isUsable()) {
		usable = training->retrain(*classifier,
					getFeatureVectors(positiveTrainingSamples), getFeatureVectors(negativeTrainingSamples));
		positiveTrainingSamples.clear();
		negativeTrainingSamples.clear();
	} else {
		vector<Sample> goodSamples;
		vector<Sample> badSamples;
		for (vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
			if (sit->getWeight() > positiveThreshold)
				goodSamples.push_back(*sit);
			else if (sit->getWeight() < negativeThreshold)
				badSamples.push_back(*sit);
		}
		if (goodSamples.size() > 10) {
			sort(goodSamples.begin(), goodSamples.end(), Sample::WeightComparison());
			goodSamples.resize(10);
		}
		if (badSamples.size() > 10) {
			sort(badSamples.begin(), badSamples.end(), Sample::WeightComparison());
			reverse(badSamples.begin(), badSamples.end());
			badSamples.resize(10);
		}
		featureExtractor->init(image);
		usable = training->retrain(*classifier, getFeatureVectors(goodSamples), getFeatureVectors(badSamples));
	}
}

vector<shared_ptr<FeatureVector> > SelfLearningMeasurementModel::getFeatureVectors(
		vector<pair<shared_ptr<FeatureVector>, double> >& pairs) {
	vector<shared_ptr<FeatureVector> > trainingSamples;
	trainingSamples.reserve(pairs.size());
	for (vector<pair<shared_ptr<FeatureVector>, double> >::iterator fvpit = pairs.begin(); fvpit != pairs.end(); ++fvpit)
		trainingSamples.push_back(fvpit->first);
	return trainingSamples;
}

vector<shared_ptr<FeatureVector> > SelfLearningMeasurementModel::getFeatureVectors(vector<Sample>& samples) {
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

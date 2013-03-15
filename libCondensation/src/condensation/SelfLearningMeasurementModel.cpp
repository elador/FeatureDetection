/*
 * SelfLearningMeasurementModel.cpp
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#include "condensation/SelfLearningMeasurementModel.h"
#include "condensation/Sample.h"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "classification/TrainableProbabilisticClassifier.hpp"
#include <unordered_map>
#include <algorithm>

using imageprocessing::Patch;
using imageprocessing::FeatureExtractor;
using std::unordered_map;
using std::make_pair;
using std::sort;
using std::reverse;

namespace condensation {

static bool comparePatchProbabilityPairs(pair<shared_ptr<Patch>, double> a, pair<shared_ptr<Patch>, double> b) {
	return a.second < b.second;
}

SelfLearningMeasurementModel::SelfLearningMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<TrainableProbabilisticClassifier> classifier, double positiveThreshold, double negativeThreshold) :
				featureExtractor(featureExtractor),
				classifier(classifier),
				usable(false),
				positiveThreshold(positiveThreshold),
				negativeThreshold(negativeThreshold),
				positiveTrainingSamples(),
				negativeTrainingSamples() {}

SelfLearningMeasurementModel::~SelfLearningMeasurementModel() {}

void SelfLearningMeasurementModel::evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples) {
	positiveTrainingSamples.clear();
	negativeTrainingSamples.clear();
	featureExtractor->update(image);
	// TODO das folgende macht nur dann sinn, wenn featureExtractor bereits duplikate erkennt und schonmal extrahiertes rausgibt
	// TODO ClassifiedPatch?
	unordered_map<shared_ptr<Patch>, pair<bool, double>> results;
	for (auto sample = samples.begin(); sample != samples.end(); ++sample) {
		sample->setObject(false);
		shared_ptr<Patch> patch = featureExtractor->extract(sample->getX(), sample->getY(), sample->getSize(), sample->getSize());
		if (!patch) {
			sample->setWeight(0);
		} else {
			pair<bool, double> result;
			auto resIt = results.find(patch);
			if (resIt == results.end()) {
				result = classifier->classify(patch->getData());
				results.insert(make_pair(patch, result));
			} else {
				result = resIt->second;
			}
			sample->setObject(result.first);
			if (result.second > positiveThreshold)
				positiveTrainingSamples.push_back(make_pair(patch, result.second));
			else if (result.second < negativeThreshold)
				negativeTrainingSamples.push_back(make_pair(patch, result.second));
			sample->setWeight(result.second);
		}
	}

	if (positiveTrainingSamples.size() > 10) {
		sort(positiveTrainingSamples.begin(), positiveTrainingSamples.end(), comparePatchProbabilityPairs);
		reverse(positiveTrainingSamples.begin(), positiveTrainingSamples.end());
		positiveTrainingSamples.resize(10);
	}
	if (negativeTrainingSamples.size() > 10) {
		sort(negativeTrainingSamples.begin(), negativeTrainingSamples.end(), comparePatchProbabilityPairs);
		negativeTrainingSamples.resize(10);
	}
}

void SelfLearningMeasurementModel::reset() {
	classifier->reset();
	usable = false;
}

void SelfLearningMeasurementModel::adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples, const Sample& target) {
	adapt(image, samples);
}

void SelfLearningMeasurementModel::adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples) {
	if (isUsable()) {
		usable = classifier->retrain(getFeatureVectors(positiveTrainingSamples), getFeatureVectors(negativeTrainingSamples));
		positiveTrainingSamples.clear();
		negativeTrainingSamples.clear();
	} else {
		vector<Sample> goodSamples;
		vector<Sample> badSamples;
		for (auto sample = samples.cbegin(); sample != samples.cend(); ++sample) {
			if (sample->getWeight() > positiveThreshold)
				goodSamples.push_back(*sample);
			else if (sample->getWeight() < negativeThreshold)
				badSamples.push_back(*sample);
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
		featureExtractor->update(image);
		usable = classifier->retrain(getFeatureVectors(goodSamples), getFeatureVectors(badSamples));
	}
}

vector<Mat> SelfLearningMeasurementModel::getFeatureVectors(vector<pair<shared_ptr<Patch>, double>>& pairs) {
	vector<Mat> trainingExamples;
	trainingExamples.reserve(pairs.size());
	for (auto pair = pairs.begin(); pair != pairs.end(); ++pair)
		trainingExamples.push_back(pair->first->getData());
	return trainingExamples;
}

vector<Mat> SelfLearningMeasurementModel::getFeatureVectors(vector<Sample>& samples) {
	vector<Mat> trainingExamples;
	trainingExamples.reserve(samples.size());
	for (auto sample = samples.cbegin(); sample != samples.cend(); ++sample) {
		shared_ptr<Patch> patch = featureExtractor->extract(sample->getX(), sample->getY(), sample->getSize(), sample->getSize());
		if (patch)
			trainingExamples.push_back(patch->getData());
	}
	return trainingExamples;
}

} /* namespace condensation */

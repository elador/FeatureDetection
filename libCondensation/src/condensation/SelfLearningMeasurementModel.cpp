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
#include "detection/ClassifiedPatch.hpp"
#include "boost/iterator/indirect_iterator.hpp"
#include <unordered_map>
#include <algorithm>
#include <functional>
#include <iostream> // TODO
using imageprocessing::Patch;
using imageprocessing::FeatureExtractor;
using boost::make_indirect_iterator;
using std::unordered_map;
using std::make_pair;
using std::make_shared;
using std::sort;
using std::greater;

namespace condensation {

SelfLearningMeasurementModel::SelfLearningMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<TrainableProbabilisticClassifier> classifier, double positiveThreshold, double negativeThreshold) :
				featureExtractor(featureExtractor),
				classifier(classifier),
				usable(false),
				positiveThreshold(positiveThreshold),
				negativeThreshold(negativeThreshold),
				positiveTrainingExamples(),
				negativeTrainingExamples() {}

SelfLearningMeasurementModel::~SelfLearningMeasurementModel() {}

void SelfLearningMeasurementModel::evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples) {
	positiveTrainingExamples.clear();
	negativeTrainingExamples.clear();
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
				positiveTrainingExamples.push_back(make_shared<ClassifiedPatch>(patch, result));
			else if (result.second < negativeThreshold)
				negativeTrainingExamples.push_back(make_shared<ClassifiedPatch>(patch, result));
			sample->setWeight(result.second);
		}
	}

	if (positiveTrainingExamples.size() > 10) {
		sort(make_indirect_iterator(positiveTrainingExamples.begin()), make_indirect_iterator(positiveTrainingExamples.end()), greater<ClassifiedPatch>());
		positiveTrainingExamples.resize(10);
	}
	if (negativeTrainingExamples.size() > 10) {
		sort(make_indirect_iterator(negativeTrainingExamples.begin()), make_indirect_iterator(negativeTrainingExamples.end()));
		negativeTrainingExamples.resize(10);
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
		usable = classifier->retrain(getFeatureVectors(positiveTrainingExamples), getFeatureVectors(negativeTrainingExamples));
		positiveTrainingExamples.clear();
		negativeTrainingExamples.clear();
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
			sort(goodSamples.begin(), goodSamples.end(), greater<Sample>());
			goodSamples.resize(10);
		}
		if (badSamples.size() > 10) {
			sort(badSamples.begin(), badSamples.end());
			badSamples.resize(10);
		}
		featureExtractor->update(image);
		usable = classifier->retrain(getFeatureVectors(goodSamples), getFeatureVectors(badSamples));
	}
}

vector<Mat> SelfLearningMeasurementModel::getFeatureVectors(const vector<shared_ptr<ClassifiedPatch>>& patches) {
	vector<Mat> trainingExamples;
	trainingExamples.reserve(patches.size());
	for (auto patch = make_indirect_iterator(patches.cbegin()); patch != make_indirect_iterator(patches.cend()); ++patch)
		trainingExamples.push_back(patch->getPatch()->getData());
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

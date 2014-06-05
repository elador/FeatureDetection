/*
 * SelfLearningMeasurementModel.cpp
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#include "condensation/SelfLearningMeasurementModel.hpp"
#include "condensation/Sample.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "classification/TrainableProbabilisticClassifier.hpp"
#include "detection/ClassifiedPatch.hpp"
#include "boost/iterator/indirect_iterator.hpp"
#include <algorithm>
#include <functional>

using imageprocessing::Patch;
using imageprocessing::VersionedImage;
using imageprocessing::FeatureExtractor;
using classification::TrainableProbabilisticClassifier;
using detection::ClassifiedPatch;
using cv::Mat;
using boost::make_indirect_iterator;
using std::pair;
using std::sort;
using std::vector;
using std::greater;
using std::shared_ptr;
using std::make_shared;
using std::unordered_map;

namespace condensation {

SelfLearningMeasurementModel::SelfLearningMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<TrainableProbabilisticClassifier> classifier, double positiveThreshold, double negativeThreshold) :
				featureExtractor(featureExtractor),
				classifier(classifier),
				usable(false),
				cache(),
				positiveThreshold(positiveThreshold),
				negativeThreshold(negativeThreshold),
				positiveTrainingExamples(),
				negativeTrainingExamples() {}

void SelfLearningMeasurementModel::update(shared_ptr<VersionedImage> image) {
	cache.clear();
	positiveTrainingExamples.clear();
	negativeTrainingExamples.clear();
	featureExtractor->update(image);
}

void SelfLearningMeasurementModel::evaluate(Sample& sample) const {
	shared_ptr<Patch> patch = featureExtractor->extract(sample.getX(), sample.getY(), sample.getWidth(), sample.getHeight());
	if (!patch) {
		sample.setTarget(false);
		sample.setWeight(0);
	} else {
		pair<bool, double> result;
		auto resIt = cache.find(patch);
		if (resIt == cache.end()) {
			result = classifier->getProbability(patch->getData());
			cache.emplace(patch, result);
		} else {
			result = resIt->second;
		}
		sample.setTarget(result.first);
		if (result.second > positiveThreshold)
			positiveTrainingExamples.push_back(make_shared<ClassifiedPatch>(patch, result));
		else if (result.second < negativeThreshold)
			negativeTrainingExamples.push_back(make_shared<ClassifiedPatch>(patch, result));
		sample.setWeight(result.second);
	}
}

void SelfLearningMeasurementModel::reset() {
	classifier->reset();
	usable = false;
}

bool SelfLearningMeasurementModel::initialize(shared_ptr<VersionedImage> image, Sample& target) {
	return false;
}

bool SelfLearningMeasurementModel::adapt(shared_ptr<VersionedImage> image, const vector<shared_ptr<Sample>>& samples, const Sample& target) {
	return adapt(image, samples);
}

bool SelfLearningMeasurementModel::adapt(shared_ptr<VersionedImage> image, const vector<shared_ptr<Sample>>& samples) {
	if (isUsable()) {
		if (positiveTrainingExamples.size() > 10) {
			sort(make_indirect_iterator(positiveTrainingExamples.begin()), make_indirect_iterator(positiveTrainingExamples.end()), greater<ClassifiedPatch>());
			positiveTrainingExamples.resize(10);
		}
		if (negativeTrainingExamples.size() > 10) {
			sort(make_indirect_iterator(negativeTrainingExamples.begin()), make_indirect_iterator(negativeTrainingExamples.end()));
			negativeTrainingExamples.resize(10);
		}
		usable = classifier->retrain(getFeatureVectors(positiveTrainingExamples), getFeatureVectors(negativeTrainingExamples));
		positiveTrainingExamples.clear();
		negativeTrainingExamples.clear();
	} else {
		vector<shared_ptr<Sample>> goodSamples;
		vector<shared_ptr<Sample>> badSamples;
		for (shared_ptr<Sample> sample : samples) {
			if (sample->getWeight() > positiveThreshold)
				goodSamples.push_back(sample);
			else if (sample->getWeight() < negativeThreshold)
				badSamples.push_back(sample);
		}
		if (goodSamples.size() > 10) {
			sort(goodSamples.begin(), goodSamples.end(), [](const shared_ptr<Sample>& a, const shared_ptr<Sample>& b) {
				return a->getWeight() > b->getWeight();
			});
			goodSamples.resize(10);
		}
		if (badSamples.size() > 10) {
			sort(badSamples.begin(), badSamples.end(), [](const shared_ptr<Sample>& a, const shared_ptr<Sample>& b) {
				return a->getWeight() < b->getWeight();
			});
			badSamples.resize(10);
		}
		featureExtractor->update(image);
		usable = classifier->retrain(getFeatureVectors(goodSamples), getFeatureVectors(badSamples));
	}
	return true;
}

vector<Mat> SelfLearningMeasurementModel::getFeatureVectors(const vector<shared_ptr<ClassifiedPatch>>& patches) {
	vector<Mat> trainingExamples;
	trainingExamples.reserve(patches.size());
	for (auto patch = make_indirect_iterator(patches.cbegin()); patch != make_indirect_iterator(patches.cend()); ++patch)
		trainingExamples.push_back(patch->getPatch()->getData());
	return trainingExamples;
}

vector<Mat> SelfLearningMeasurementModel::getFeatureVectors(vector<shared_ptr<Sample>>& samples) {
	vector<Mat> trainingExamples;
	trainingExamples.reserve(samples.size());
	for (shared_ptr<Sample> sample : samples) {
		shared_ptr<Patch> patch = featureExtractor->extract(sample->getX(), sample->getY(), sample->getWidth(), sample->getHeight());
		if (patch)
			trainingExamples.push_back(patch->getData());
	}
	return trainingExamples;
}

} /* namespace condensation */

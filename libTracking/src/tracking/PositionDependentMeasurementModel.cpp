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
#include "classification/TrainableClassifier.h"
#include <unordered_map>
#include <utility>

using std::make_shared;
using std::unordered_map;
using std::pair;

namespace tracking {

PositionDependentMeasurementModel::PositionDependentMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<TrainableClassifier> classifier, int startFrameCount, int stopFrameCount,
		float positiveOffsetFactor, float negativeOffsetFactor,
		bool sampleNegativesAroundTarget, bool sampleFalsePositives, unsigned int randomNegatives) :
				featureExtractor(featureExtractor),
				classifier(classifier),
				usable(false),
				frameCount(0),
				startFrameCount(startFrameCount),
				stopFrameCount(stopFrameCount),
				positiveOffsetFactor(positiveOffsetFactor),
				negativeOffsetFactor(negativeOffsetFactor),
				sampleNegativesAroundTarget(sampleNegativesAroundTarget),
				sampleFalsePositives(sampleFalsePositives),
				randomNegatives(randomNegatives) {}

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

void PositionDependentMeasurementModel::reset() {
	classifier->reset();
	usable = false;
}

void PositionDependentMeasurementModel::adapt(const Mat& image, const vector<Sample>& samples, const Sample& target) {
	if (!isUsable()) {
		frameCount++;
		if (frameCount < startFrameCount)
			return;
		// feature extractor has to be initialized on the image when this model was not used for evaluation
		featureExtractor->init(image);
	} else {
		frameCount = 0;
	}

	vector<Sample> positiveSamples;
	if (positiveOffsetFactor == 0) {
		positiveSamples.reserve(1);
		positiveSamples.push_back(target);
	} else {
		int offset = std::max(1, (int)(positiveOffsetFactor * target.getSize()));
		positiveSamples.reserve(5);
		positiveSamples.push_back(target);
		positiveSamples.push_back(Sample(target.getX() - offset, target.getY(), target.getSize()));
		positiveSamples.push_back(Sample(target.getX() + offset, target.getY(), target.getSize()));
		positiveSamples.push_back(Sample(target.getX(), target.getY() - offset, target.getSize()));
		positiveSamples.push_back(Sample(target.getX(), target.getY() + offset, target.getSize()));
	}

	vector<Sample> negativeSamples;
	int boundOffset = (int)(negativeOffsetFactor * target.getSize());
	int xLowBound = target.getX() - boundOffset;
	int xHighBound = target.getX() + boundOffset;
	int yLowBound = target.getY() - boundOffset;
	int yHighBound = target.getY() + boundOffset;
	int sizeLowBound = (int)((1 - negativeOffsetFactor) * target.getSize());
	int sizeHighBound = (int)((1 + 2 * negativeOffsetFactor) * target.getSize());

	if (sampleNegativesAroundTarget) {
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
	}

	if (sampleFalsePositives) {
		for (vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
			if (sit->isObject()
					&& (sit->getX() <= xLowBound || sit->getX() >= xHighBound
					|| sit->getY() <= yLowBound || sit->getY() >= yHighBound
					|| sit->getSize() <= sizeLowBound || sit->getSize() >= sizeHighBound)) {
				negativeSamples.push_back(*sit);
			}
		}
	}

	if (randomNegatives > 0) {
		vector<Sample> additionalNegatives;
		additionalNegatives.reserve(randomNegatives);
		for (vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
			if ((!sampleFalsePositives || !sit->isObject())
					&& (sit->getX() <= xLowBound || sit->getX() >= xHighBound
					|| sit->getY() <= yLowBound || sit->getY() >= yHighBound
					|| sit->getSize() <= sizeLowBound || sit->getSize() >= sizeHighBound)) {
				if (additionalNegatives.size() < randomNegatives) {
					vector<Sample>::iterator low = lower_bound(additionalNegatives.begin(), additionalNegatives.end(), *sit, Sample::WeightComparison());
					additionalNegatives.insert(low, *sit);
				} else if (additionalNegatives.front().getWeight() < sit->getWeight()) {
					additionalNegatives.front() = *sit;
					Sample tmp;
					for (unsigned int i = 1; i < additionalNegatives.size() && additionalNegatives[i - 1].getWeight() > additionalNegatives[i].getWeight(); ++i) {
						tmp = additionalNegatives[i - 1];
						additionalNegatives[i - 1] = additionalNegatives[i];
						additionalNegatives[i] = tmp;
					}
				}
			}
		}
		while (additionalNegatives.size() < randomNegatives) {
			Sample sample = createRandomSample(image);
			if (sample.getX() <= xLowBound || sample.getX() >= xHighBound
					|| sample.getY() <= yLowBound || sample.getY() >= yHighBound
					|| sample.getSize() <= sizeLowBound || sample.getSize() >= sizeHighBound) {
				if (featureExtractor->extract(sample.getX(), sample.getY(), sample.getSize()))
					additionalNegatives.push_back(sample);
			}
		}
		negativeSamples.insert(negativeSamples.end(), additionalNegatives.begin(), additionalNegatives.end());
	}

	usable = classifier->retrain(getFeatureVectors(positiveSamples), getFeatureVectors(negativeSamples));
}

void PositionDependentMeasurementModel::adapt(const Mat& image, const vector<Sample>& samples) {
	if (isUsable())
		frameCount++;
	if (frameCount == stopFrameCount) {
		reset();
		frameCount = 0;
	} else {
		const vector<shared_ptr<FeatureVector> > empty;
		usable = classifier->retrain(empty, empty);
	}
}

Sample PositionDependentMeasurementModel::createRandomSample(const Mat& image) {
	int minSize = (int)(0.1 * std::min(image.cols, image.rows));
	int maxSize = (int)(0.9 * std::min(image.cols, image.rows));
	int size = distribution(generator, maxSize - minSize) + minSize;
	int halfSize = size / 2;
	int x = distribution(generator, image.cols - size) + halfSize;
	int y = distribution(generator, image.rows - size) + halfSize;
	return Sample(x, y, size);
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

/*
 * PositionDependentMeasurementModel.cpp
 *
 *  Created on: 20.09.2012
 *      Author: poschmann
 */

#include "condensation/PositionDependentMeasurementModel.h"
#include "condensation/Sample.h"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "classification/TrainableProbabilisticClassifier.hpp"

using imageprocessing::Patch;
using imageprocessing::FeatureExtractor;

namespace condensation {

PositionDependentMeasurementModel::PositionDependentMeasurementModel(shared_ptr<FeatureExtractor> featureExtractor,
		shared_ptr<TrainableProbabilisticClassifier> classifier, int startFrameCount, int stopFrameCount,
		float positiveOffsetFactor, float negativeOffsetFactor,
		bool sampleNegativesAroundTarget, bool sampleFalsePositives, unsigned int randomNegatives) :
				SingleClassifierModel(featureExtractor, classifier),
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

void PositionDependentMeasurementModel::reset() {
	classifier->reset();
	usable = false;
}

void PositionDependentMeasurementModel::adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples, const Sample& target) {
	if (!isUsable()) {
		frameCount++;
		if (frameCount < startFrameCount)
			return;
		// feature extractor has to be initialized on the image when this model was not used for evaluation
		featureExtractor->update(image);
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
		for (auto sample = samples.cbegin(); sample != samples.cend(); ++sample) {
			if (sample->isObject()
					&& (sample->getX() <= xLowBound || sample->getX() >= xHighBound
					|| sample->getY() <= yLowBound || sample->getY() >= yHighBound
					|| sample->getSize() <= sizeLowBound || sample->getSize() >= sizeHighBound)) {
				negativeSamples.push_back(*sample);
			}
		}
	}

	if (randomNegatives > 0) {
		vector<Sample> additionalNegatives;
		additionalNegatives.reserve(randomNegatives);
		for (auto sample = samples.cbegin(); sample != samples.cend(); ++sample) {
			if ((!sampleFalsePositives || !sample->isObject())
					&& (sample->getX() <= xLowBound || sample->getX() >= xHighBound
					|| sample->getY() <= yLowBound || sample->getY() >= yHighBound
					|| sample->getSize() <= sizeLowBound || sample->getSize() >= sizeHighBound)) {
				if (additionalNegatives.size() < randomNegatives) {
					vector<Sample>::iterator low = lower_bound(additionalNegatives.begin(), additionalNegatives.end(), *sample);
					additionalNegatives.insert(low, *sample);
				} else if (additionalNegatives.front().getWeight() < sample->getWeight()) {
					additionalNegatives.front() = *sample;
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
			Sample sample = createRandomSample(image->getData());
			if (sample.getX() <= xLowBound || sample.getX() >= xHighBound
					|| sample.getY() <= yLowBound || sample.getY() >= yHighBound
					|| sample.getSize() <= sizeLowBound || sample.getSize() >= sizeHighBound) {
				if (featureExtractor->extract(sample.getX(), sample.getY(), sample.getSize(), sample.getSize()))
					additionalNegatives.push_back(sample);
			}
		}
		negativeSamples.insert(negativeSamples.end(), additionalNegatives.begin(), additionalNegatives.end());
	}

	usable = classifier->retrain(getFeatureVectors(positiveSamples), getFeatureVectors(negativeSamples));
}

void PositionDependentMeasurementModel::adapt(shared_ptr<VersionedImage> image, const vector<Sample>& samples) {
	if (isUsable())
		frameCount++;
	if (frameCount == stopFrameCount) {
		reset();
		frameCount = 0;
	} else {
		// TODO das könnte quark sein - retraining nicht nötig (und abfrage im training auch wegmachen)
		// TODO nur FrameBasedTraining hätte das gebraucht, aber über nutzung entscheidet am ende das measurement model
		const vector<Mat> empty;
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

vector<Mat> PositionDependentMeasurementModel::getFeatureVectors(vector<Sample>& samples) {
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

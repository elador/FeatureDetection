/*
 * PositionDependentLearningStrategy.cpp
 *
 *  Created on: 21.09.2012
 *      Author: poschmann
 */

#include "tracking/PositionDependentLearningStrategy.h"
#include "tracking/LearningMeasurementModel.h"

namespace tracking {

PositionDependentLearningStrategy::PositionDependentLearningStrategy() {}

PositionDependentLearningStrategy::~PositionDependentLearningStrategy() {}

void PositionDependentLearningStrategy::update(LearningMeasurementModel& model, cv::Mat& image,
		const std::vector<Sample>& samples) {
	model.update();
}

void PositionDependentLearningStrategy::update(LearningMeasurementModel& model, cv::Mat& image,
		const std::vector<Sample>& samples, const Sample& position) {

	std::vector<Sample> positiveSamples;
	int offset = std::max(1, position.getSize() / 20);
	positiveSamples.push_back(position);
	positiveSamples.push_back(Sample(position.getX() - offset, position.getY(), position.getSize()));
	positiveSamples.push_back(Sample(position.getX() + offset, position.getY(), position.getSize()));
	positiveSamples.push_back(Sample(position.getX(), position.getY() - offset, position.getSize()));
	positiveSamples.push_back(Sample(position.getX(), position.getY() + offset, position.getSize()));

	std::vector<Sample> negativeSamples;
	double deviationFactor = 0.5;
	int boundOffset = (int)(deviationFactor * position.getSize());
	int xLowBound = position.getX() - boundOffset;
	int xHighBound = position.getX() + boundOffset;
	int yLowBound = position.getY() - boundOffset;
	int yHighBound = position.getY() + boundOffset;
	int sizeLowBound = (int)((1 - deviationFactor) * position.getSize());
	int sizeHighBound = (int)((1 + 2 * deviationFactor) * position.getSize());

	negativeSamples.push_back(Sample(xLowBound, position.getY(), position.getSize()));
	negativeSamples.push_back(Sample(xHighBound, position.getY(), position.getSize()));
	negativeSamples.push_back(Sample(position.getX(), yLowBound, position.getSize()));
	negativeSamples.push_back(Sample(position.getX(), yHighBound, position.getSize()));
	negativeSamples.push_back(Sample(xLowBound, yLowBound, position.getSize()));
	negativeSamples.push_back(Sample(xLowBound, yHighBound, position.getSize()));
	negativeSamples.push_back(Sample(xHighBound, yLowBound, position.getSize()));
	negativeSamples.push_back(Sample(xHighBound, yHighBound, position.getSize()));

	negativeSamples.push_back(Sample(position.getX(), position.getY(), sizeLowBound));
	negativeSamples.push_back(Sample(xLowBound, position.getY(), sizeLowBound));
	negativeSamples.push_back(Sample(xHighBound, position.getY(), sizeLowBound));
	negativeSamples.push_back(Sample(position.getX(), yLowBound, sizeLowBound));
	negativeSamples.push_back(Sample(position.getX(), yHighBound, sizeLowBound));
//		negativeSamples.push_back(Sample(xLowBound, yLowBound, sizeLowBound));
//		negativeSamples.push_back(Sample(xLowBound, yHighBound, sizeLowBound));
//		negativeSamples.push_back(Sample(xHighBound, yLowBound, sizeLowBound));
//		negativeSamples.push_back(Sample(xHighBound, yHighBound, sizeLowBound));

	negativeSamples.push_back(Sample(position.getX(), position.getY(), sizeHighBound));
	negativeSamples.push_back(Sample(xLowBound, position.getY(), sizeHighBound));
	negativeSamples.push_back(Sample(xHighBound, position.getY(), sizeHighBound));
	negativeSamples.push_back(Sample(position.getX(), yLowBound, sizeHighBound));
	negativeSamples.push_back(Sample(position.getX(), yHighBound, sizeHighBound));
//		negativeSamples.push_back(Sample(xLowBound, yLowBound, sizeHighBound));
//		negativeSamples.push_back(Sample(xLowBound, yHighBound, sizeHighBound));
//		negativeSamples.push_back(Sample(xHighBound, yLowBound, sizeHighBound));
//		negativeSamples.push_back(Sample(xHighBound, yHighBound, sizeHighBound));

	for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		if (sit->isObject()
				&& (sit->getX() <= xLowBound || sit->getX() >= xHighBound
				|| sit->getY() <= yLowBound || sit->getY() >= yHighBound
				|| sit->getSize() <= sizeLowBound || sit->getSize() >= sizeHighBound)) {
			negativeSamples.push_back(*sit);
		}
	}

	model.update(image, positiveSamples, negativeSamples);
}

} /* namespace tracking */

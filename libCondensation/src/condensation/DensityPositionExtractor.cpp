/*
 * DensityPositionExtractor.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "condensation/DensityPositionExtractor.h"
#include "condensation/Rectangle.h"
#include <iostream>

namespace condensation {

DensityPositionExtractor::DensityPositionExtractor(int bandwidth) : invertedBandwidth(1.0 / bandwidth),
		invertedBandwidthProduct(invertedBandwidth * invertedBandwidth * invertedBandwidth) {}

DensityPositionExtractor::~DensityPositionExtractor() {}

boost::optional<Sample> DensityPositionExtractor::extract(const vector<Sample>& samples) {
	if (samples.size() == 0)
		return boost::optional<Sample>();
	// compute start value (mean)
	double x = 0;
	double y = 0;
	double s = 0;
	for (auto sample = samples.cbegin(); sample != samples.cend(); ++sample) {
		x += sample->getX();
		y += sample->getY();
		s += sample->getSize();
	}
	Sample point((int)(x / samples.size() + 0.5), (int)(y / samples.size() + 0.5), (int)(s / samples.size() + 0.5));
	Sample oldPoint;
	// iteratively compute densest point
	x = 0, y = 0, s = 0;
	int i = 0;
	double weightSum = 0;
	do {
		oldPoint = point;
		for (vector<Sample>::const_iterator sample = samples.cbegin(); sample != samples.cend(); ++sample) {
			double weight = sample->getWeight() * getScaledKernelValue(*sample, point);
			x += weight * sample->getX();
			y += weight * sample->getY();
			s += weight * sample->getSize();
			weightSum += weight;
		}
		point.setX((int)(x / weightSum + 0.5));
		point.setY((int)(y / weightSum + 0.5));
		point.setSize((int)(s / weightSum + 0.5));
		++i;
	} while (i < 100 && point.getX() != oldPoint.getX() && point.getY() != oldPoint.getY() && point.getSize() != oldPoint.getSize());
	if (i >= 100)// TODO logging
		std::cerr << "too many iterations: (" << point.getX() << ", " << point.getY() << ", " << point.getSize() << ") <> ("
				<< oldPoint.getX() << ", " << oldPoint.getY() << ", " << oldPoint.getSize() << ")" << std::endl;
	point.setWeight(computeDensity(samples, point));
	return optional<Sample>(point);
}

double DensityPositionExtractor::computeDensity(const vector<Sample>& samples, const Sample& position) {
	double kernelValueSum = 0;
	double weightSum = 0;
	for (auto sample = samples.cbegin(); sample != samples.cend(); ++sample) {
		kernelValueSum += sample->getWeight() * getScaledKernelValue(*sample, position);
		weightSum += sample->getWeight();
	}
	return kernelValueSum / weightSum;
}

} /* namespace condensation */

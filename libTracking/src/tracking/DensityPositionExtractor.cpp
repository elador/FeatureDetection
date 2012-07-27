/*
 * DensityPositionExtractor.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "tracking/DensityPositionExtractor.h"
#include "tracking/Rectangle.h"
#include <iostream>

namespace tracking {

DensityPositionExtractor::DensityPositionExtractor(int bandwidth) : invertedBandwidth(1.0 / bandwidth),
		invertedBandwidthProduct(invertedBandwidth * invertedBandwidth * invertedBandwidth) {}

DensityPositionExtractor::~DensityPositionExtractor() {}

boost::optional<Sample> DensityPositionExtractor::extract(const std::vector<Sample>& samples) {
	if (samples.size() == 0)
		return boost::optional<Sample>();
	// compute start value (mean)
	double x = 0;
	double y = 0;
	double s = 0;
	for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		x += sit->getX();
		y += sit->getY();
		s += sit->getSize();
	}
	Sample point(x / samples.size() + 0.5, y / samples.size() + 0.5, s / samples.size() + 0.5);
	Sample oldPoint;
	// iteratively compute densest point
	x = 0, y = 0, s = 0;
	int i = 0;
	double weightSum = 0;
	do {
		oldPoint = point;
		for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
			double weight = sit->getWeight() * getScaledKernelValue(*sit, point);
			x += weight * sit->getX();
			y += weight * sit->getY();
			s += weight * sit->getSize();
			weightSum += weight;
		}
		point.setX(x / weightSum + 0.5);
		point.setY(y / weightSum + 0.5);
		point.setSize(s / weightSum + 0.5);
		++i;
	} while (i < 100 && point.getX() != oldPoint.getX() && point.getY() != oldPoint.getY() && point.getSize() != oldPoint.getSize());
	// TODO wenn sample double/float bekommt, dann hier anders...
	if (i >= 100)
		std::cout << "too many iterations: (" << point.getX() << ", " << point.getY() << ", " << point.getSize() << ") <> ("
				<< oldPoint.getX() << ", " << oldPoint.getY() << ", " << oldPoint.getSize() << ")" << std::endl;
	point.setWeight(computeDensity(samples, point));
	return boost::optional<Sample>(point);
}

double DensityPositionExtractor::computeDensity(const std::vector<Sample>& samples, const Sample& position) {
	double kernelValueSum = 0;
	double weightSum = 0;
	for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		kernelValueSum += sit->getWeight() * getScaledKernelValue(*sit, position);
		weightSum += sit->getWeight();
	}
	return kernelValueSum / weightSum;
}

} /* namespace tracking */

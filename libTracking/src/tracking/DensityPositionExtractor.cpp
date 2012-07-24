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

DensityPositionExtractor::DensityPositionExtractor(int bandwidth) : bandwidth(bandwidth), invertedBandwidth(1.0 / bandwidth),
		invertedBandwidthProduct(invertedBandwidth * invertedBandwidth * invertedBandwidth) {}

DensityPositionExtractor::~DensityPositionExtractor() {}

boost::optional<Sample> DensityPositionExtractor::extract(const std::vector<Sample>& samples) {
	// compute start value (mean)
	double x = 0;
	double y = 0;
	double s = 0;
	for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		x += sit->getX();
		y += sit->getY();
		s += sit->getSize();
	}
	Sample mean(x / samples.size() + 0.5, y / samples.size() + 0.5, s / samples.size() + 0.5);
	Sample oldMean;
	x = 0, y = 0, s = 0;
	int i = 0;
	double weightSum = 0;
	do {
		oldMean = mean;
		for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
			double weight = getScaledKernelValue(*sit, mean);
			x += weight * sit->getX();
			y += weight * sit->getY();
			s += weight * sit->getSize();
			weightSum += weight;
		}
		mean.setX(x / weightSum + 0.5);
		mean.setY(y / weightSum + 0.5);
		mean.setSize(s / weightSum + 0.5);
		++i;
	} while (i < 100 && mean.getX() != oldMean.getX() && mean.getY() != oldMean.getY() && mean.getSize() != oldMean.getSize());
	// TODO wenn sample double/float bekommt, dann hier anders...
	if (i >= 100)
		std::cout << "too many iterations: (" << mean.getX() << ", " << mean.getY() << ", " << mean.getSize() << ") <> ("
				<< oldMean.getX() << ", " << oldMean.getY() << ", " << oldMean.getSize() << ")" << std::endl;
	mean.setWeight(computeDensity(samples, mean));
	if (mean.getWeight() > 0) // TODO
		return boost::optional<Sample>(mean);
	return boost::optional<Sample>();
}

double DensityPositionExtractor::computeDensity(const std::vector<Sample>& samples, const Sample& position) {
	double kernelValueSum = 0;
	for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit)
		kernelValueSum += getScaledKernelValue(*sit, position);
	return kernelValueSum / samples.size();
}

} /* namespace tracking */

/*
 * LowVarianceSampling.cpp
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#include "tracking/LowVarianceSampling.h"
#include "tracking/Sample.h"

#include <ctime>

namespace tracking {

LowVarianceSampling::LowVarianceSampling() : generator(boost::mt19937(time(0))),
		distribution(boost::uniform_01<>()) {}

LowVarianceSampling::~LowVarianceSampling() {}

void LowVarianceSampling::resample(const std::vector<Sample>& samples,
		unsigned int count, std::vector<Sample>& newSamples) {
	newSamples.clear();
	if (samples.size() > 0) {
		double avg = computeAverageWeight(samples);
		if (avg > 0) {
			double start = avg * distribution(generator);
			std::vector<Sample>::const_iterator sit = samples.begin();
			double weightSum = sit->getWeight();
			for (unsigned int i = 0; i < count; ++i) {
				double weightPointer = start + i * avg;
				while (weightPointer > weightSum) {
					++sit;
					weightSum += sit->getWeight();
				}
				newSamples.push_back(*sit);
			}
		}
	}
}

double LowVarianceSampling::computeAverageWeight(const std::vector<Sample>& samples) {
	double weightSum = 0;
	std::vector<Sample>::const_iterator it = samples.begin();
	std::vector<Sample>::const_iterator end = samples.end();
	for (; it != end; ++it)
		weightSum += it->getWeight();
	return weightSum / samples.size();
}

} /* namespace tracking */

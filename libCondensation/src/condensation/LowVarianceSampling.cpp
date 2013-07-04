/*
 * LowVarianceSampling.cpp
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#include "condensation/LowVarianceSampling.hpp"
#include "condensation/Sample.hpp"
#include <ctime>

namespace condensation {

LowVarianceSampling::LowVarianceSampling() : generator(boost::mt19937(time(0))),
		distribution(boost::uniform_01<>()) {}

LowVarianceSampling::~LowVarianceSampling() {}

void LowVarianceSampling::resample(const vector<Sample>& samples, unsigned int count, vector<Sample>& newSamples) {
	newSamples.clear();
	if (samples.size() > 0) {
		double step = computeWeightSum(samples) / count;
		if (step > 0) {
			double start = step * distribution(generator);
			vector<Sample>::const_iterator sample = samples.cbegin();
			double weightSum = sample->getWeight();
			for (unsigned int i = 0; i < count; ++i) {
				double weightPointer = start + i * step;
				while (weightPointer > weightSum) {
					++sample;
					weightSum += sample->getWeight();
				}
				newSamples.push_back(*sample);
			}
		}
	}
}

double LowVarianceSampling::computeWeightSum(const vector<Sample>& samples) {
	double weightSum = 0;
	for (auto sample = samples.cbegin(); sample != samples.cend(); ++sample)
		weightSum += sample->getWeight();
	return weightSum;
}

} /* namespace condensation */

/*
 * LowVarianceSampling.cpp
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#include "condensation/LowVarianceSampling.hpp"
#include "condensation/Sample.hpp"
#include <ctime>

using std::vector;
using std::shared_ptr;

namespace condensation {

LowVarianceSampling::LowVarianceSampling() : generator(boost::mt19937(time(0))),
		distribution(boost::uniform_01<>()) {}

void LowVarianceSampling::resample(const vector<shared_ptr<Sample>>& samples, size_t count, vector<shared_ptr<Sample>>& newSamples) {
	newSamples.reserve(count);
	if (samples.size() > 0) {
		double weightSum = computeWeightSum(samples);
		double step = weightSum / count;
		if (step > 0) {
			double start = step * distribution(generator);
			vector<shared_ptr<Sample>>::const_iterator sample = samples.cbegin();
			double weightSum = (*sample)->getWeight();
			for (unsigned int i = 0; i < count; ++i) {
				double weightPointer = start + i * step;
				while (weightPointer > weightSum) {
					++sample;
					weightSum += (*sample)->getWeight();
				}
				newSamples.emplace_back(new Sample(*sample));
			}
		}
	}
}

double LowVarianceSampling::computeWeightSum(const vector<shared_ptr<Sample>>& samples) {
	double weightSum = 0;
	for (const shared_ptr<Sample> sample : samples)
		weightSum += sample->getWeight();
	return weightSum;
}

} /* namespace condensation */

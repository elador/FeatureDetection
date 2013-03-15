/*
 * WeightedMeanPositionExtractor.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "condensation/WeightedMeanPositionExtractor.h"
#include "condensation/Sample.h"

namespace condensation {

WeightedMeanPositionExtractor::WeightedMeanPositionExtractor() {}

WeightedMeanPositionExtractor::~WeightedMeanPositionExtractor() {}

optional<Sample> WeightedMeanPositionExtractor::extract(const vector<Sample>& samples) {
	double weightedSumX = 0;
	double weightedSumY = 0;
	double weightedSumSize = 0;
	double weightSum = 0;
	for (auto sample = samples.cbegin(); sample < samples.cend(); ++sample) {
		weightedSumX += sample->getWeight() * sample->getX();
		weightedSumY += sample->getWeight() * sample->getY();
		weightedSumSize += sample->getWeight() * sample->getSize();
		weightSum += sample->getWeight();
	}
	if (weightSum == 0)
		return optional<Sample>();
	double weightedMeanX = weightedSumX / weightSum;
	double weightedMeanY = weightedSumY / weightSum;
	double weightedMeanSize = weightedSumSize / weightSum;
	return optional<Sample>(
			Sample((int)(weightedMeanX + 0.5), (int)(weightedMeanY + 0.5), (int)(weightedMeanSize + 0.5)));
}

} /* namespace condensation */

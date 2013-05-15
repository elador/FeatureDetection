/*
 * WeightedMeanPositionExtractor.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "condensation/WeightedMeanPositionExtractor.hpp"
#include "condensation/Sample.hpp"

namespace condensation {

WeightedMeanPositionExtractor::WeightedMeanPositionExtractor() {}

WeightedMeanPositionExtractor::~WeightedMeanPositionExtractor() {}

optional<Sample> WeightedMeanPositionExtractor::extract(const vector<Sample>& samples) {
	double weightedSumX = 0;
	double weightedSumY = 0;
	double weightedSumSize = 0;
	double weightedSumVx = 0;
	double weightedSumVy = 0;
	double weightedSumVSize = 0;
	double weightSum = 0;
	for (auto sample = samples.cbegin(); sample < samples.cend(); ++sample) {
		weightedSumX += sample->getWeight() * sample->getX();
		weightedSumY += sample->getWeight() * sample->getY();
		weightedSumSize += sample->getWeight() * sample->getSize();
		weightedSumVx += sample->getWeight() * sample->getVx();
		weightedSumVy += sample->getWeight() * sample->getVy();
		weightedSumVSize += sample->getWeight() * sample->getVSize();
		weightSum += sample->getWeight();
	}
	if (weightSum == 0)
		return optional<Sample>();
	double weightedMeanX = weightedSumX / weightSum;
	double weightedMeanY = weightedSumY / weightSum;
	double weightedMeanSize = weightedSumSize / weightSum;
	double weightedMeanVx = weightedSumVx / weightSum;
	double weightedMeanVy = weightedSumVy / weightSum;
	double weightedMeanVSize = weightedSumVSize / weightSum;
	return optional<Sample>(Sample(
			(int)(weightedMeanX + 0.5), (int)(weightedMeanY + 0.5), (int)(weightedMeanSize + 0.5),
			(int)(weightedMeanVx + 0.5), (int)(weightedMeanVy + 0.5), (int)(weightedMeanVSize + 0.5)));
}

} /* namespace condensation */

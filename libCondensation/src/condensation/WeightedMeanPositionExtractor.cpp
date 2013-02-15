/*
 * WeightedMeanPositionExtractor.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "tracking/WeightedMeanPositionExtractor.h"
#include "tracking/Sample.h"

namespace tracking {

WeightedMeanPositionExtractor::WeightedMeanPositionExtractor() {}

WeightedMeanPositionExtractor::~WeightedMeanPositionExtractor() {}

optional<Sample> WeightedMeanPositionExtractor::extract(const vector<Sample>& samples) {
	double weightedSumX = 0;
	double weightedSumY = 0;
	double weightedSumSize = 0;
	double weightSum = 0;
	for (vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		weightedSumX += sit->getWeight() * sit->getX();
		weightedSumY += sit->getWeight() * sit->getY();
		weightedSumSize += sit->getWeight() * sit->getSize();
		weightSum += sit->getWeight();
	}
	if (weightSum == 0)
		return optional<Sample>();
	double weightedMeanX = weightedSumX / weightSum;
	double weightedMeanY = weightedSumY / weightSum;
	double weightedMeanSize = weightedSumSize / weightSum;
	return optional<Sample>(
			Sample((int)(weightedMeanX + 0.5), (int)(weightedMeanY + 0.5), (int)(weightedMeanSize + 0.5)));
}

} /* namespace tracking */

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

boost::optional<Sample> WeightedMeanPositionExtractor::extract(const std::vector<Sample>& samples) {
	double weightedSumX = 0;
	double weightedSumY = 0;
	double weightedSumSize = 0;
	double weightSum = 0;
	for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		weightedSumX += sit->getWeight() * sit->getX();
		weightedSumY += sit->getWeight() * sit->getY();
		weightedSumSize += sit->getWeight() * sit->getSize();
		weightSum += sit->getWeight();
	}
	if (weightSum == 0)
		return boost::optional<Sample>();
	double weightedMeanX = weightedSumX / weightSum;
	double weightedMeanY = weightedSumY / weightSum;
	double weightedMeanSize = weightedSumSize / weightSum;
	return boost::optional<Sample>(Sample(weightedMeanX + 0.5, weightedMeanY + 0.5, weightedMeanSize + 0.5));
}

} /* namespace tracking */

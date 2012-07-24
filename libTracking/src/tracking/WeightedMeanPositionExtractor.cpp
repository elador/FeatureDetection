/*
 * WeightedMeanPositionExtractor.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "tracking/WeightedMeanPositionExtractor.h"
#include "tracking/Sample.h"
#include <iostream>
namespace tracking {

WeightedMeanPositionExtractor::WeightedMeanPositionExtractor() {}

WeightedMeanPositionExtractor::~WeightedMeanPositionExtractor() {}

boost::optional<Sample> WeightedMeanPositionExtractor::extract(const std::vector<Sample>& samples) {
	if (samples.empty())
		return boost::optional<Sample>();
	double weightedSumX = 0;
	double weightedSumY = 0;
	double weightedSumSize = 0;
//	double weightedSquaredSumX = 0;
//	double weightedSquaredSumY = 0;
//	double weightedSquaredSumSize = 0;
	double weightSum = 0;
	for (std::vector<Sample>::const_iterator sit = samples.begin(); sit < samples.end(); ++sit) {
		weightedSumX += sit->getWeight() * sit->getX();
		weightedSumY += sit->getWeight() * sit->getY();
		weightedSumSize += sit->getWeight() * sit->getSize();
//		weightedSquaredSumX += sit->getWeight() * sit->getX() * sit->getX();
//		weightedSquaredSumY += sit->getWeight() * sit->getY() * sit->getY();
//		weightedSquaredSumSize += sit->getWeight() * sit->getSize() * sit->getSize();
		weightSum += sit->getWeight();
	}
	if (weightSum == 0)
		return boost::optional<Sample>();
	double weightedMeanX = weightedSumX / weightSum;
	double weightedMeanY = weightedSumY / weightSum;
	double weightedMeanSize = weightedSumSize / weightSum;
//	double weightedSquaredMeanX = weightedSquaredSumX / weightSum;
//	double weightedSquaredMeanY = weightedSquaredSumY / weightSum;
//	double weightedSquaredMeanSize = weightedSquaredSumSize / weightSum;
//	double weightedVarianceX = weightedSquaredMeanX - weightedMeanX * weightedMeanX;
//	double weightedVarianceY = weightedSquaredMeanY - weightedMeanY * weightedMeanY;
//	double weightedVarianceSize = weightedSquaredMeanSize - weightedMeanSize * weightedMeanSize;
	// variance thresholds
//	double threshold = 0.5 * weightedMeanSize * 0.5 * weightedMeanSize;
//	std::cout << weightedVarianceX << " < " << threshold << std::endl;
//	std::cout << weightedVarianceY << " < " << threshold << std::endl;
//	std::cout << weightedVarianceSize << " < " << threshold << std::endl;
//	if (weightedVarianceX > threshold || weightedVarianceY > threshold || weightedVarianceSize > threshold)
//		return boost::optional<Sample>();
	return boost::optional<Sample>(Sample(weightedMeanX + 0.5, weightedMeanY + 0.5, weightedMeanSize + 0.5));
}

} /* namespace tracking */

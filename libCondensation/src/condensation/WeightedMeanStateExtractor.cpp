/*
 * WeightedMeanStateExtractor.cpp
 *
 *  Created on: 19.07.2012
 *      Author: poschmann
 */

#include "condensation/WeightedMeanStateExtractor.hpp"
#include "condensation/Sample.hpp"
#include <unordered_map>
#include <algorithm>

using std::vector;
using std::shared_ptr;
using std::make_shared;
using std::unordered_map;
using std::pair;

namespace condensation {

WeightedMeanStateExtractor::WeightedMeanStateExtractor() {}

shared_ptr<Sample> WeightedMeanStateExtractor::extract(const vector<shared_ptr<Sample>>& samples) {
	unordered_map<int, vector<shared_ptr<Sample>>> clusters;
	for (const shared_ptr<Sample>& sample : samples)
		clusters[sample->getClusterId()].push_back(sample);
	auto it = std::max_element(clusters.begin(), clusters.end(),
			[](const pair<int, vector<shared_ptr<Sample>>>& a, const pair<int, vector<shared_ptr<Sample>>>& b) {
					return a.second.size() < b.second.size();
	});
	if (it == clusters.end())
		return shared_ptr<Sample>();
	vector<shared_ptr<Sample>>& cluster = it->second;

	double weightedSumX = 0;
	double weightedSumY = 0;
	double weightedSumSize = 0;
	double weightedSumVx = 0;
	double weightedSumVy = 0;
	double weightedSumVSize = 0;
	double weightSum = 0;
	for (shared_ptr<Sample> sample : cluster) {
		weightedSumX += sample->getWeight() * sample->getX();
		weightedSumY += sample->getWeight() * sample->getY();
		weightedSumSize += sample->getWeight() * sample->getSize();
		weightedSumVx += sample->getWeight() * sample->getVx();
		weightedSumVy += sample->getWeight() * sample->getVy();
		weightedSumVSize += sample->getWeight() * sample->getVSize();
		weightSum += sample->getWeight();
	}
	if (weightSum == 0)
		return shared_ptr<Sample>();
	double weightedMeanX = weightedSumX / weightSum;
	double weightedMeanY = weightedSumY / weightSum;
	double weightedMeanSize = weightedSumSize / weightSum;
	double weightedMeanVx = weightedSumVx / weightSum;
	double weightedMeanVy = weightedSumVy / weightSum;
	double weightedMeanVSize = weightedSumVSize / weightSum;
	return make_shared<Sample>(
			(int)(weightedMeanX + 0.5), (int)(weightedMeanY + 0.5), (int)(weightedMeanSize + 0.5),
			(int)(weightedMeanVx + 0.5), (int)(weightedMeanVy + 0.5), (int)(weightedMeanVSize + 0.5));
}

} /* namespace condensation */

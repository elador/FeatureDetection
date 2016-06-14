/*
 * NonMaximumSuppression.cpp
 *
 *  Created on: 28.10.2015
 *      Author: poschmann
 */

#include "detection/NonMaximumSuppression.hpp"
#include <stdexcept>

using cv::Rect;
using std::vector;

namespace detection {

NonMaximumSuppression::NonMaximumSuppression(double overlapThreshold, MaximumType maximumType) :
		overlapThreshold(overlapThreshold), maximumType(maximumType) {}

double NonMaximumSuppression::getOverlapThreshold() const {
	return overlapThreshold;
}

NonMaximumSuppression::MaximumType NonMaximumSuppression::getMaximumType() const {
	return maximumType;
}

vector<Detection> NonMaximumSuppression::eliminateRedundantDetections(vector<Detection> candidates) const {
	if (overlapThreshold == 1.0) // with this threshold, there would be an endless loop - this check assumes distinct bounding boxes
		return candidates;
	sortByScore(candidates);
	vector<vector<Detection>> clusters = cluster(candidates);
	return getMaxima(clusters);
}

void NonMaximumSuppression::sortByScore(vector<Detection>& candidates) const {
	std::sort(candidates.begin(), candidates.end(), [](const Detection& a, const Detection& b) {
		return a.score < b.score;
	});
}

vector<vector<Detection>> NonMaximumSuppression::cluster(vector<Detection>& candidates) const {
	vector<vector<Detection>> clusters;
	while (!candidates.empty())
		clusters.push_back(extractOverlappingDetections(candidates.back(), candidates));
	return clusters;
}

vector<Detection> NonMaximumSuppression::extractOverlappingDetections(Detection detection, vector<Detection>& candidates) const {
	vector<Detection> overlappingDetections;
	auto firstOverlapping = std::stable_partition(candidates.begin(), candidates.end(), [&](const Detection& candidate) {
		return computeOverlap(detection.bounds, candidate.bounds) <= overlapThreshold;
	});
	std::move(firstOverlapping, candidates.end(), std::back_inserter(overlappingDetections));
	std::reverse(overlappingDetections.begin(), overlappingDetections.end());
	candidates.erase(firstOverlapping, candidates.end());
	return overlappingDetections;
}

double NonMaximumSuppression::computeOverlap(Rect a, Rect b) const {
	double intersectionArea = (a & b).area();
	double unionArea = a.area() + b.area() - intersectionArea;
	return intersectionArea / unionArea;
}

vector<Detection> NonMaximumSuppression::getMaxima(const vector<vector<Detection>>& clusters) const {
	vector<Detection> finalDetections;
	finalDetections.reserve(clusters.size());
	for (const vector<Detection>& cluster : clusters)
		finalDetections.push_back(getMaximum(cluster));
	return finalDetections;
}

Detection NonMaximumSuppression::getMaximum(const vector<Detection>& cluster) const {
	if (maximumType == MaximumType::MAX_SCORE) {
		return cluster.front();
	} else if (maximumType == MaximumType::AVERAGE) {
		double xSum = 0;
		double ySum = 0;
		double wSum = 0;
		double hSum = 0;
		for (const Detection& elem : cluster) {
			xSum += elem.bounds.x;
			ySum += elem.bounds.y;
			wSum += elem.bounds.width;
			hSum += elem.bounds.height;
		}
		int x = static_cast<int>(std::round(xSum / cluster.size()));
		int y = static_cast<int>(std::round(ySum / cluster.size()));
		int w = static_cast<int>(std::round(wSum / cluster.size()));
		int h = static_cast<int>(std::round(hSum / cluster.size()));
		float score = cluster.front().score;
		Rect averageBounds(x, y, w, h);
		return Detection{score, averageBounds};
	} else if (maximumType == MaximumType::WEIGHTED_AVERAGE) {
		double weightSum = 0;
		double xSum = 0;
		double ySum = 0;
		double wSum = 0;
		double hSum = 0;
		for (const Detection& elem : cluster) {
			double weight = elem.score;
			weightSum += weight;
			xSum += weight * elem.bounds.x;
			ySum += weight * elem.bounds.y;
			wSum += weight * elem.bounds.width;
			hSum += weight * elem.bounds.height;
		}
		int x = static_cast<int>(std::round(xSum / weightSum));
		int y = static_cast<int>(std::round(ySum / weightSum));
		int w = static_cast<int>(std::round(wSum / weightSum));
		int h = static_cast<int>(std::round(hSum / weightSum));
		float score = cluster.front().score;
		Rect averageBounds(x, y, w, h);
		return Detection{score, averageBounds};
	} else {
		throw std::runtime_error("NonMaximumSuppression: unsupported maximum type");
	}
}

} /* namespace detection */

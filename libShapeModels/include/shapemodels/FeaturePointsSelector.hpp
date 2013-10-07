/*
 * FeaturePointsSelector.hpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#pragma once

#ifndef FEATUREPOINTSSELECTOR_HPP_
#define FEATUREPOINTSSELECTOR_HPP_

#include "imageprocessing/Patch.hpp"

#include <memory>
#include <map>
#include <string>
#include <vector>
#include <random>
#include <utility>

using std::shared_ptr;
using std::string;
using std::vector;
using std::map;
using std::pair;

namespace shapemodels {

/**
 * A ... . Note: Maybe we can make a (Distinct|Unique)FeaturePointsSelector later that keeps track of used feature points.
 * I have to re-structure this. Rename the functions to getPoint(...) and make a RandomFeaturePointsSelector. But
 * random != ransac. But the ransac algorithm belongs somewhere else I'd say.
 */
class FeaturePointsSelector {
public:
	
	FeaturePointsSelector() {
		engine.seed(1);
	};
	FeaturePointsSelector(map<string, vector<shared_ptr<imageprocessing::Patch>>> landmarks) : landmarks(landmarks) {
		// C++11: Call default c'tor
		engine.seed(1);
	};

	~FeaturePointsSelector() {};

	/**
	 * Does xyz
	 *
	 * @param[in] param The parameter.
	 */
	void setLandmarks(map<string, vector<shared_ptr<imageprocessing::Patch>>> landmarks) {
		this->landmarks = landmarks;
	};

	/**
	 * A getter.
	 *
	 * @return Something.
	 */
	//virtual const int getter() const;

	map<string, shared_ptr<imageprocessing::Patch>> getDistinctRandomPoints(int numPoints) {
		if (numPoints > landmarks.size()) { // We cannot return more distinct points than the number of features we detected
			// Logger warning
			// throw or return empty map
			return map<string, shared_ptr<imageprocessing::Patch>>();
		}
		
		// Convert map to vec. Maybe use vec everywhere instead of map!
		vector<pair<string, vector<shared_ptr<imageprocessing::Patch>>>> landmarksVector;
		for (const auto& l : landmarks) {
			landmarksVector.emplace_back(l);
		}
		
		// Select #numPoints distinct landmarks from all of our landmarks
		map<string, shared_ptr<imageprocessing::Patch>> distinctLandmarks;
		for (unsigned int i = 0; i < numPoints; ++i) {
			// random ffp:
			int featureIdx = -1;
			std::uniform_int_distribution<int> rndFeature(0, landmarksVector.size()-1); // Todo: assumes landmarksVector.size() > 0. Check for that?
			do {
				featureIdx = rndFeature(engine);
			} while (landmarksVector[featureIdx].second.size() == 0); // Todo: This can result in an infinite loop. Better erase the empty one and do rnd with 1 number less.
			// random point:
			std::uniform_int_distribution<int> rndPoint(0, landmarksVector[featureIdx].second.size()-1);
			int pointIdx = rndPoint(engine);
			distinctLandmarks.insert(make_pair(landmarksVector[featureIdx].first, landmarksVector[featureIdx].second[pointIdx]));
			// we selected a landmark and a point, so delete the landmark
			landmarksVector.erase(begin(landmarksVector) + featureIdx);
		}




		return distinctLandmarks;
	};

	/**
	 * First selects a feature uniformly and then returns a random 
	 * point from that set. If the selected set is empty, the process
	 * is repeated until a point is found.
	 * 
	 *
	 * @return A randomly selected patch. Returns a nullptr only if it 
	 *         is not possible to return even one single point.
	 */
	shared_ptr<imageprocessing::Patch> getRandomPoint() {
		return nullptr;
	};
	
	/**
	 * Returns a random point (with uniform probability) of the given feature. May be empty.
	 *
	 * @return A randomly selected patch. May be a nullptr.
	 */
	shared_ptr<imageprocessing::Patch> getRandomPoint(string feature) {
		return nullptr;
	};

	// Needed for now, not sure this whole Selector thing makes sense
	map<string, vector<shared_ptr<imageprocessing::Patch>>> getAllLandmarks() {
		return landmarks;
	}

private:
	map<string, vector<shared_ptr<imageprocessing::Patch>>> landmarks;
	std::mt19937 engine; // Mersenne twister MT19937

};

} /* namespace shapemodels */
#endif // FEATUREPOINTSSELECTOR_HPP_

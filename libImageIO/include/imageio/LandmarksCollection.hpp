/*
 * LandmarksCollection.hpp
 *
 *  Created on: 23.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LANDMARKSCOLLECTION_HPP_
#define LANDMARKSCOLLECTION_HPP_

#include "opencv2/core/core.hpp"
#include <string>

using cv::Vec3f;
using std::string;
using std::vector;

namespace imageio {

class Landmark;

/**
 * A collection of landmarks, i.e. all landmarks given in a
 * landmarks-file for a certain image.
 */
class LandmarksCollection {
public:

	/**
	 * Constructs a new landmark collection.
	 *
	 */
	LandmarksCollection();

	~LandmarksCollection();

	void insert(const Landmark& landmark);

	bool hasLandmark(const string& name) const;

	Landmark getLandmark(const string& name) const;

	const vector<Landmark>& getLandmarksList() const;

	operator vector<Landmark> const& () const {
		return landmarks;
	}
	
private:
	vector<Landmark> landmarks;		///< A vector of all the landmarks.
	std::map<string, size_t> landmarksMap;	///< A map from the landmarks names to the indices in the vector.

};

} /* namespace imageio */
#endif /* LANDMARKSCOLLECTION_HPP_ */

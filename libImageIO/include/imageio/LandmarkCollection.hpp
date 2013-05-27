/*
 * LandmarkCollection.hpp
 *
 *  Created on: 23.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LANDMARKSCOLLECTION_HPP_
#define LANDMARKSCOLLECTION_HPP_

#include "imageio/Landmark.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>

using std::string;
using std::vector;
using std::map;
using std::shared_ptr;

namespace imageio {

/**
 * A collection of landmarks, i.e. all landmarks given in a
 * landmarks-file for a certain image.
 */
class LandmarkCollection {
public:

	/**
	 * Constructs a new empty landmark collection.
	 */
	LandmarkCollection();

	~LandmarkCollection();

	/**
	 * Removes all landmarks.
	 */
	void clear();

	/**
	 * Adds a new landmark to the collection.
	 *
	 * @param[in] landmark The landmark.
	 */
	void insert(shared_ptr<Landmark> landmark);

	/**
	 * Determines whether this collection is empty.
	 *
	 * @return True if this collection is empty, false otherwise.
	 */
	bool isEmpty() const;

	/**
	 * Determines whether a certain landmark exists inside this collection.
	 *
	 * @param[in] name The name of the landmark.
	 * @return True if there exists a landmark with the given name, false otherwise.
	 */
	bool hasLandmark(const string& name) const;

	/**
	 * Retrieves a landmark from this collection. Throws an exception if there is no landmark with
	 * the given name.
	 *
	 * @param[in] name The name of the landmark.
	 * @return The landmark with the given name.
	 */
	const Landmark& getLandmark(const string& name) const;

	/**
	 * Retrieves the first landmark of this collection. Throws an exception if there is not landmark.
	 * Useful if there is only one landmark to be expected, because the name is irrelevant in that case.
	 *
	 * @return The first landmark of this collection.
	 */
	const Landmark& getLandmark() const;

	/**
	 * @return The landmarks.
	 */
	const vector<shared_ptr<Landmark>>& getLandmarks() const;

	operator vector<shared_ptr<Landmark>> const& () const {
		return landmarks;
	}
	
private:

	vector<shared_ptr<Landmark>> landmarks; ///< A vector of all the landmarks.
	map<string, size_t> landmarksMap;	      ///< A map from the landmarks names to the indices in the vector.
};

} /* namespace imageio */
#endif /* LANDMARKSCOLLECTION_HPP_ */

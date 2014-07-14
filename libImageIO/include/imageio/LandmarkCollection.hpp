/*
 * LandmarkCollection.hpp
 *
 *  Created on: 23.03.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef LANDMARKCOLLECTION_HPP_
#define LANDMARKCOLLECTION_HPP_

#include "imageio/Landmark.hpp"
#include <string>
#include <vector>
#include <map>
#include <memory>

namespace imageio {

/**
 * A collection of landmarks, i.e. all landmarks given in a
 * landmarks-file for a certain image.
 * The landmarks have to be unique, i.e. each landmark may only
 * exist once in the collection.
 */
class LandmarkCollection {
public:

	/**
	 * Constructs a new empty landmark collection.
	 */
	LandmarkCollection();

	/**
	 * Removes all landmarks.
	 */
	void clear();

	/**
	 * Adds a new landmark to the collection.
	 *
	 * @param[in] landmark The landmark.
	 */
	void insert(std::shared_ptr<Landmark> landmark);

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
	bool hasLandmark(const std::string& name) const;

	/**
	 * Retrieves a landmark from this collection. Throws an exception if there is no landmark with
	 * the given name.
	 *
	 * @param[in] name The name of the landmark.
	 * @return The landmark with the given name.
	 */
	const std::shared_ptr<Landmark> getLandmark(const std::string& name) const;

	/**
	 * Retrieves the first landmark of this collection. Throws an exception if there is not landmark.
	 * Useful if there is only one landmark to be expected, because the name is irrelevant in that case.
	 *
	 * @return The first landmark of this collection.
	 */
	const std::shared_ptr<Landmark> getLandmark() const;

	/**
	 * @return The landmarks.
	 */
	const std::vector<std::shared_ptr<Landmark>>& getLandmarks() const;

	operator std::vector<std::shared_ptr<Landmark>> const& () const {
		return landmarks;
	}
	
private:

	std::vector<std::shared_ptr<Landmark>> landmarks; ///< A vector of all the landmarks.
	std::map<std::string, size_t> landmarksMap;	      ///< A map from the landmarks names to the indices in the vector.
};

/**
 * Calculate the rectangular bounding box of the landmarks in the given
 * collection.
 *
 * @param[in] landmarks A collection of landmarks.
 * @return The bounding box encompassing the landmarks.
 */
cv::Rect getBoundingBox(LandmarkCollection landmarks);

} /* namespace imageio */
#endif /* LANDMARKCOLLECTION_HPP_ */

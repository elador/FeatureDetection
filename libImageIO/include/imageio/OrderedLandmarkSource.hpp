/*
 * OrderedLandmarkSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef ORDEREDLANDMARKSOURCE_HPP_
#define ORDEREDLANDMARKSOURCE_HPP_

namespace imageio {

class LandmarkCollection;

/**
 * Source of subsequent landmark collections. The landmark collections have the same order as the images
 * they belong to.
 */
class OrderedLandmarkSource {
public:

	virtual ~OrderedLandmarkSource() {}

	/**
	 * Moves the landmark source forward to the next collection of landmarks.
	 *
	 * @return True if the landmark source contains a next collection of landmarks, false otherwise.
	 */
	virtual const bool next() = 0;

	/**
	 * Retrieves the current collection of landmarks and moves the landmark source forward to
	 * the next collection.
	 *
	 * @return The collection of landmarks (that may be empty if no data could be retrieved).
	 */
	virtual const LandmarkCollection get() = 0;

	/**
	 * Retrieves the current collection of landmarks.
	 *
	 * @return The collection of landmarks (that may be empty if no data could be retrieved).
	 */
	virtual const LandmarkCollection getLandmarks() const = 0;
};

} /* namespace imageio */
#endif /* ORDEREDLANDMARKSOURCE_HPP_ */

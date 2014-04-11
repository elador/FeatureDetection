/*
 * LandmarkSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef LANDMARKSOURCE_HPP_
#define LANDMARKSOURCE_HPP_

namespace imageio {

class LandmarkCollection;

/**
 * Source of subsequent landmark collections. The landmark collections have the same order as the images
 * they belong to.
 */
class LandmarkSource {
public:

	virtual ~LandmarkSource() {}

	/**
	 * Resets the landmark source to its initial state.
	 */
	virtual void reset() = 0;

	/**
	 * Moves the landmark source forward to the next collection of landmarks.
	 *
	 * @return True if the landmark source contains a next collection of landmarks, false otherwise.
	 */
	virtual bool next() = 0;

	/**
	 * Retrieves the current collection of landmarks.
	 *
	 * @return The collection of landmarks (that may be empty if no data could be retrieved).
	 */
	virtual LandmarkCollection getLandmarks() const = 0;
};

} /* namespace imageio */
#endif /* LANDMARKSOURCE_HPP_ */

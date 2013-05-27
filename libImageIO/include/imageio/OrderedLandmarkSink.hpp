/*
 * OrderedLandmarkSink.hpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#ifndef ORDEREDLANDMARKSINK_HPP_
#define ORDEREDLANDMARKSINK_HPP_

namespace imageio {

class LandmarkCollection;

/**
 * Sink for subsequent landmark collections.
 */
class OrderedLandmarkSink {
public:

	virtual ~OrderedLandmarkSink() {}

	/**
	 * Adds a landmark collection.
	 *
	 * @param[in] collection The landmark collection.
	 */
	virtual void add(const LandmarkCollection& collection) = 0;
};

} /* namespace imageio */
#endif /* ORDEREDLANDMARKSINK_HPP_ */

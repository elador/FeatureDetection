/*
 * OrderedLandmarkSink.hpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#ifndef ORDEREDLANDMARKSINK_HPP_
#define ORDEREDLANDMARKSINK_HPP_

#include <string>

namespace imageio {

class LandmarkCollection;

/**
 * Sink for subsequent landmark collections.
 */
class OrderedLandmarkSink {
public:

	virtual ~OrderedLandmarkSink() {}

	/**
	 * Determines whether this landmark sink is open.
	 *
	 * @return True if this landmark sink was opened (and not closed since), false otherwise.
	 */
	virtual bool isOpen() = 0;

	/**
	 * Opens the file writer. Needs to be called before adding the first landmark collection.
	 *
	 * @param[in] filename The name of the file to write the landmark data into.
	 */
	virtual void open(const std::string& filename) = 0;

	/**
	 * Closes the file writer.
	 */
	virtual void close() = 0;

	/**
	 * Adds a landmark collection.
	 *
	 * @param[in] collection The landmark collection.
	 */
	virtual void add(const LandmarkCollection& collection) = 0;
};

} /* namespace imageio */
#endif /* ORDEREDLANDMARKSINK_HPP_ */

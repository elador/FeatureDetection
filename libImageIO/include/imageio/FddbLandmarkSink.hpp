/*
 * FddbLandmarkSink.hpp
 *
 *  Created on: 09.12.2013
 *      Author: Patrik Huber
 */
#pragma once

#ifndef FDDBLANDMARKSINK_HPP_
#define FDDBLANDMARKSINK_HPP_

#include "imageio/RectLandmark.hpp"

#include <string>
#include <fstream>
#include <vector>

namespace imageio {

class Landmark;

/**
 * Landmark sink that writes landmarks to a file using the format of 
 * the FDDB face-detection benchmark. http://vis-www.cs.umass.edu/fddb/
 */
class FddbLandmarkSink { // TODO: Shouldn't we derive from Ordered/NamedLandmarkSink? Probably we can't, since add() is different (needs the detection score)
public:

	/**
	 * Constructs a new FDDB landmark sink.
	 *
	 * @param[in] annotatedList TODO.
	 */
	explicit FddbLandmarkSink(const std::string& annotatedList);

	~FddbLandmarkSink();

	/**
	 * Determines whether this landmark sink is open.
	 *
	 * @return True if this landmark sink was opened (and not closed since), false otherwise.
	 */
	bool isOpen();

	/**
	 * Opens the file writer. Needs to be called before adding the first landmark collection.
	 *
	 * @param[in] filename TODO.
	 */
	void open(const std::string& filename);

	/**
	 * Closes the file writer.
	 */
	void close(); // or move to d'tor?

	void add(const std::string imageFilename, const std::vector<RectLandmark> faceLandmarks, const std::vector<float> detectionScores); // or write(...)

private:
	FddbLandmarkSink(FddbLandmarkSink& other);        // Copy c'tor and assignment operator private
	FddbLandmarkSink operator=(FddbLandmarkSink rhs); // since we own an ofstream object.

	std::ofstream output; ///< The file output stream.
	std::string annotatedList; ///< The images have to be sorted according to the entries in this file.

};

} /* namespace imageio */
#endif /* FDDBLANDMARKSINK_HPP_ */

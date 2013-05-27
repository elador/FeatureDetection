/*
 * BobotLandmarkSink.hpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#ifndef BOBOTLANDMARKSINK_HPP_
#define BOBOTLANDMARKSINK_HPP_

#include "imageio/OrderedLandmarkSink.hpp"
#include <string>
#include <fstream>

using std::string;
using std::ofstream;

namespace imageio {

class Landmark;

/**
 * Landmark source that writes rectangular landmarks to a file using the Bonn Benchmark on Tracking (BoBoT) format.
 */
class BobotLandmarkSink : public OrderedLandmarkSink {
public:

	/**
	 * Constructs a new BoBoT landmark sink.
	 *
	 * @param[in] landmarkName The name of the landmark. Can be ommitted if there is only one landmark per image.
	 */
	explicit BobotLandmarkSink(const string& landmarkName = "");

	~BobotLandmarkSink();

	/**
	 * Determines whether this landmark since is open.
	 *
	 * @return True if this landmark sink was opened (and not closed since), false otherwise.
	 */
	bool isOpen();

	/**
	 * Opens the file writer. Needs to be called before adding the first landmark collection.
	 *
	 * @param[in] filename The name of the file containing the landmark data in BoBoT format.
	 * @param[in] videoFilename The name of the video file the landmarks belong to (will be written in the first line of the file).
	 * @param[in] imageWidth The width of the images.
	 * @param[in] imageHeight The height of the images.
	 */
	void open(const string& filename, const string& videoFilename, float imageWidth, float imageHeight);

	/**
	 * Closes the file writer.
	 */
	void close();

	void add(const LandmarkCollection& collection);

private:

	/**
	 * Extracts the landmark that should be written to the file. If the landmark name is an empty string, the first
	 * landmark will be taken (there will be assumed to be only one landmark within the collection). Otherwise, the
	 * landmark with the specified name will be taken.
	 *
	 * @param[in] collection The landmark collection.
	 * @return The landmark.
	 */
	const Landmark& getLandmark(const LandmarkCollection& collection);

	const string landmarkName; ///< The name of the landmark.
	ofstream output; ///< The file output stream.
	unsigned int index; ///< The index of the next landmark that is written to the file.
	float imageWidth; ///< The width of the images.
	float imageHeight; ///< The height of the images.
};

} /* namespace imageio */
#endif /* BOBOTLANDMARKSINK_HPP_ */

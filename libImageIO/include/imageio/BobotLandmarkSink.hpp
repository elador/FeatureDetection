/*
 * BobotLandmarkSink.hpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#ifndef BOBOTLANDMARKSINK_HPP_
#define BOBOTLANDMARKSINK_HPP_

#include "imageio/OrderedLandmarkSink.hpp"
#include <fstream>
#include <memory>

namespace imageio {

class Landmark;
class ImageSource;

/**
 * Landmark sink that writes rectangular landmarks to a file using the Bonn Benchmark on Tracking (BoBoT) format.
 */
class BobotLandmarkSink : public OrderedLandmarkSink {
public:

	/**
	 * Constructs a new BoBoT landmark sink with explicit image dimensions.
	 *
	 * @param[in] videoFilename The name of the video file the landmarks belong to (will be written in the first line of the file).
	 * @param[in] imageWidth The width of the images.
	 * @param[in] imageHeight The height of the images.
	 * @param[in] landmarkName The name of the landmark. Can be ommitted if there is only one landmark per image.
	 */
	explicit BobotLandmarkSink(const std::string& videoFilename, int imageWidth, int imageHeight, const std::string& landmarkName = "");

	/**
	 * Constructs a new BoBoT landmark sink that takes the image dimensions from an image source.
	 *
	 * @param[in] videoFilename The name of the video file the landmarks belong to (will be written in the first line of the file).
	 * @param[in] imageSource The source of the images. Is assumed to be at the position the added landmarks belong to.
	 * @param[in] landmarkName The name of the landmark. Can be ommitted if there is only one landmark per image.
	 */
	BobotLandmarkSink(const std::string& videoFilename, std::shared_ptr<ImageSource> imageSource, const std::string& landmarkName = "");

	bool isOpen();

	void open(const std::string& filename);

	void close();

	void add(const LandmarkCollection& collection);

private:
	BobotLandmarkSink(BobotLandmarkSink& other);        // Copy c'tor and assignment operator private
	BobotLandmarkSink operator=(BobotLandmarkSink rhs); // since we own an ofstream object.

	/**
	 * Extracts the landmark that should be written to the file. If the landmark name is an empty string, the first
	 * landmark will be taken (there will be assumed to be only one landmark within the collection). Otherwise, the
	 * landmark with the specified name will be taken.
	 *
	 * @param[in] collection The landmark collection.
	 * @return The landmark.
	 */
	const std::shared_ptr<Landmark> getLandmark(const LandmarkCollection& collection);

	const std::string videoFilename; ///< The name of the video file the landmarks belong to (will be written in the first line of the file).
	int imageWidth;  ///< The width of the images.
	int imageHeight; ///< The height of the images.
	std::shared_ptr<ImageSource> imageSource; ///< The source of the images. Is assumed to be at the position the added landmarks belong to.
	const std::string landmarkName; ///< The name of the landmark.
	std::ofstream output; ///< The file output stream.
	unsigned int index; ///< The index of the next landmark that is written to the file.
};

} /* namespace imageio */
#endif /* BOBOTLANDMARKSINK_HPP_ */

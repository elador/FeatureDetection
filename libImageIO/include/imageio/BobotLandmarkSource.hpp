/*
 * BobotLandmarkSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */
#pragma once

#ifndef BOBOTLANDMARKSOURCE_HPP_
#define BOBOTLANDMARKSOURCE_HPP_

#include "imageio/NamedLandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "opencv2/core/core.hpp"
#include "boost/filesystem/path.hpp"
#include <string>
#include <vector>
#include <unordered_map>
#include <memory>

namespace imageio {

class ImageSource;

/**
 * Landmark source that reads rectangular landmarks from a Bonn Benchmark on Tracking (BoBoT) file. Uses the
 * associated image source to determine the size of the images. Each image will have one associated landmark
 * whose name is "target".
 */
class BobotLandmarkSource : public NamedLandmarkSource {
public:

	/**
	 * Constructs a new BoBoT landmark source with explicit image dimensions.
	 *
	 * @param[in] filename The name of the file containing the landmark data in BoBoT format.
	 * @param[in] imageWidth The width of the images.
	 * @param[in] imageHeight The height of the images.
	 */
	BobotLandmarkSource(const std::string& filename, int imageWidth, int imageHeight);

	/**
	 * Constructs a new BoBoT landmark source that takes the image dimensions from an image source.
	 *
	 * @param[in] filename The name of the file containing the landmark data in BoBoT format.
	 * @param(in] imageSource The source of the images. Is assumed to be at the same position as this landmark source.
	 */
	BobotLandmarkSource(const std::string& filename, std::shared_ptr<ImageSource> imageSource);

	/**
	 * @return The name/path of the video file associated with this landmark source (first line of file).
	 */
	const std::string& getVideoFilename() const;

	void reset();

	bool next();

	LandmarkCollection get(const boost::filesystem::path& imagePath); 	// Note: Modifies the state of this LandmarkSource

	LandmarkCollection getLandmarks() const;

	boost::filesystem::path getName() const;

private:

	/**
	 * Reads the target positions for each frame from the given file.
	 *
	 * @param[in] filename The name of the file containing the landmark data in BoBoT format.
	 */
	void readPositions(const std::string& filename);

	static const std::string landmarkName; ///< The name of the landmarks.
	mutable int imageWidth;  ///< The width of the images.
	mutable int imageHeight; ///< The height of the images.
	std::shared_ptr<ImageSource> imageSource; ///< The source of the images. Is assumed to be at the same position as this landmark source.
	std::string videoFilename; ///< The name/path of the video file associated with this landmark source (first line of file).
	std::vector<cv::Rect_<float>> positions; ///< The target positions inside each image.
	std::unordered_map<std::string, size_t> name2index; ///< Mapping between image name and position index.
	std::unordered_map<size_t, std::string> index2name; ///< Mapping between position index and image name.
	int index; ///< The index of the current target position.
};

} /* namespace imageio */
#endif /* BOBOTLANDMARKSOURCE_HPP_ */

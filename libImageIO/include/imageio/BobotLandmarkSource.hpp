/*
 * BobotLandmarkSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef BOBOTLANDMARKSOURCE_HPP_
#define BOBOTLANDMARKSOURCE_HPP_

#include "imageio/NamedLandmarkSource.hpp"
#include "imageio/OrderedLandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"
#include "opencv2/core/core.hpp"
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
class BobotLandmarkSource : public NamedLandmarkSource, public OrderedLandmarkSource {
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

	void reset();

	bool next();

	LandmarkCollection get();

	LandmarkCollection get(const path& imagePath); 	// Note: Modifies the LandmarkSource

	LandmarkCollection getLandmarks() const;

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
	std::vector<cv::Rect_<float>> positions; ///< The target positions inside each image.
	std::unordered_map<std::string, size_t> name2index; ///< Mapping between image name and position index.
	int index; ///< The index of the current target position.
};

} /* namespace imageio */
#endif /* BOBOTLANDMARKSOURCE_HPP_ */

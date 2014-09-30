/*
 * BobotAnnotationSink.hpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#ifndef BOBOTANNOTATIONSINK_HPP_
#define BOBOTANNOTATIONSINK_HPP_

#include "imageio/AnnotationSink.hpp"
#include <fstream>
#include <memory>

namespace imageio {

class ImageSource;

/**
 * Annotation sink that writes rectangular annotations to a file using the Bonn Benchmark on Tracking (BoBoT) format.
 */
class BobotAnnotationSink : public AnnotationSink {
public:

	/**
	 * Constructs a new BoBoT annotation sink with explicit image dimensions.
	 *
	 * @param[in] videoFilename The name of the video file the annotations belong to (will be written in the first line of the file).
	 * @param[in] imageWidth The width of the images.
	 * @param[in] imageHeight The height of the images.
	 */
	explicit BobotAnnotationSink(const std::string& videoFilename, int imageWidth, int imageHeight);

	/**
	 * Constructs a new BoBoT annotation sink that takes the image dimensions from an image source.
	 *
	 * @param[in] videoFilename The name of the video file the annotations belong to (will be written in the first line of the file).
	 * @param[in] imageSource The source of the images. Is assumed to be at the position the added annotations belong to.
	 */
	BobotAnnotationSink(const std::string& videoFilename, std::shared_ptr<ImageSource> imageSource);

	bool isOpen();

	void open(const std::string& filename);

	void close();

	void add(const cv::Rect& annotation);

private:

	// Copy c'tor and assignment operator private since we own an ofstream object.
	BobotAnnotationSink(BobotAnnotationSink& other);
	BobotAnnotationSink operator=(BobotAnnotationSink rhs);

	const std::string videoFilename; ///< The name of the video file the landmarks belong to (will be written in the first line of the file).
	int imageWidth;  ///< The width of the images.
	int imageHeight; ///< The height of the images.
	std::shared_ptr<ImageSource> imageSource; ///< The source of the images. Is assumed to be at the position the added landmarks belong to.
	std::ofstream output; ///< The file output stream.
	size_t index; ///< The index of the next landmark that is written to the file.
};

} /* namespace imageio */
#endif /* BOBOTANNOTATIONSINK_HPP_ */

/*
 * VersionedImage.hpp
 *
 *  Created on: 14.03.2013
 *      Author: poschmann
 */

#ifndef VERSIONEDIMAGE_HPP_
#define VERSIONEDIMAGE_HPP_

#include "imageprocessing/Version.hpp"
#include "opencv2/core/core.hpp"

namespace imageprocessing {

/**
 * Image with a version number.
 */
class VersionedImage {
public:

	/**
	 * Constructs a new empty versioned image.
	 */
	VersionedImage() : data(), version() {}

	/**
	 * Constructs a new versioned image.
	 *
	 * @param[in] data The image data.
	 */
	explicit VersionedImage(const cv::Mat& data) : data(data), version() {}

	/**
	 * @return The image data.
	 */
	cv::Mat& getData() {
		return data;
	}

	/**
	 * @return The image data.
	 */
	const cv::Mat& getData() const {
		return data;
	}

	/**
	 * @param[in] data The new image data.
	 */
	void setData(const cv::Mat& data) {
		this->data = data;
		++version;
	}

	/**
	 * @return The version.
	 */
	Version getVersion() const {
		return version;
	}

private:

	cv::Mat data; ///< The image data.
	Version version; ///< The version.
};

} /* namespace imageprocessing */
#endif /* VERSIONEDIMAGE_HPP_ */

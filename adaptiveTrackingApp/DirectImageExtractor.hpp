/*
 * DirectImageExtractor.hpp
 *
 *  Created on: 30.06.2013
 *      Author: ex-ratt
 */

#ifndef DIRECTIMAGEEXTRACTOR_HPP_
#define DIRECTIMAGEEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include <memory>

using std::make_shared;

namespace imageprocessing {

class DirectImageExtractor : public FeatureExtractor {
public:

	DirectImageExtractor();

	virtual ~DirectImageExtractor();

	void update(const Mat& image) {
		this->image = filter.applyTo(image);
		version = -1;
	}

	void update(shared_ptr<VersionedImage> image) {
		if (version != image->getVersion()) {
			this->image = filter.applyTo(image->getData());
			version = image->getVersion();
		}
	}

	shared_ptr<Patch> extract(int x, int y, int width, int height) const;

	void addImageFilter(shared_ptr<ImageFilter> filter) {}
	void addLayerFilter(shared_ptr<ImageFilter> filter) {}
	void addPatchFilter(shared_ptr<ImageFilter> filter) {}

	/**
	 * @return The width of the image data of the extracted patches.
	 */
	int getPatchWidth() const {
		return patchWidth;
	}

	/**
	 * @param[in] width The new width of the image data of the extracted patches.
	 */
	void setPatchWidth(int width) {
		patchWidth = width;
	}

	/**
	 * @return The height of the image data of the extracted patches.
	 */
	int getPatchHeight() const {
		return patchHeight;
	}

	/**
	 * @param[in] height The new height of the image data of the extracted patches.
	 */
	void setPatchHeight(int height) {
		patchHeight = height;
	}

	/**
	 * Changes the size of the image data of the extracted patches.
	 *
	 * @param[in] width The new width of the image data of the extracted patches.
	 * @param[in] height The new height of the image data of the extracted patches.
	 */
	void setPatchSize(int width, int height) {
		patchWidth = width;
		patchHeight = height;
	}

private:

	int version;                            ///< The version number.
	Mat image;
	int patchWidth;  ///< The width of the image data of the extracted patches.
	int patchHeight; ///< The height of the image data of the extracted patches.
	GrayscaleFilter filter;
};

} /* namespace imageprocessing */
#endif /* DIRECTIMAGEEXTRACTOR_HPP_ */

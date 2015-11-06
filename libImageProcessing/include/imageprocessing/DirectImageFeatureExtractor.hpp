/*
 * DirectImageFeatureExtractor.hpp
 *
 *  Created on: 30.06.2013
 *      Author: poschmann
 */

#ifndef DIRECTIMAGEFEATUREEXTRACTOR_HPP_
#define DIRECTIMAGEFEATUREEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/Version.hpp"
#include <memory>

namespace imageprocessing {

class ImageFilter;
class ChainedFilter;

/**
 * Feature extractor that extracts patches from images.
 */
class DirectImageFeatureExtractor : public FeatureExtractor {
public:

	/**
	 * Constructs a new direct image feature extractor.
	 */
	DirectImageFeatureExtractor();

	/**
	 * Adds an image filter that is applied to the image.
	 *
	 * @param[in] filter The new image filter.
	 */
	void addImageFilter(std::shared_ptr<ImageFilter> filter);

	/**
	 * Adds a new filter that is applied to the patches.
	 *
	 * @param[in] filter The new patch filter.
	 */
	void addPatchFilter(std::shared_ptr<ImageFilter> filter);

	using FeatureExtractor::update;

	void update(std::shared_ptr<VersionedImage> image);

	std::shared_ptr<Patch> extract(int x, int y, int width, int height) const;

private:

	Version version; ///< The version.
	cv::Mat image; ///< The filtered image.
	std::shared_ptr<ChainedFilter> imageFilter; ///< Filter that is applied to the image.
	std::shared_ptr<ChainedFilter> patchFilter; ///< Filter that is applied to the patches.
};

} /* namespace imageprocessing */
#endif /* DIRECTIMAGEFEATUREEXTRACTOR_HPP_ */

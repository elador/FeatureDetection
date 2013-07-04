/*
 * DirectImageExtractor.hpp
 *
 *  Created on: 30.06.2013
 *      Author: ex-ratt
 */

#ifndef HAARIMAGEEXTRACTOR_HPP_
#define HAARIMAGEEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/IntegralImageFilter.hpp"
#include <memory>
#include <stdexcept>

using cv::Rect_;
using std::make_shared;
using std::vector;

namespace imageprocessing {

class HaarImageExtractor : public FeatureExtractor {
public:

	HaarImageExtractor();

	~HaarImageExtractor();

	void update(const Mat& image) {
		this->image = integralFilter.applyTo(filter.applyTo(image));
		version = -1;
		if (this->image.type() != CV_32S)
			throw std::invalid_argument("oh noes! " + this->image.type());
	}

	void update(shared_ptr<VersionedImage> image) {
		if (version != image->getVersion()) {
			this->image = integralFilter.applyTo(filter.applyTo(image->getData()));
			version = image->getVersion();
			if (this->image.type() != CV_32S)
				throw std::invalid_argument("oh noes! " + this->image.type());
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
		return 20;
	}

	/**
	 * @param[in] width The new width of the image data of the extracted patches.
	 */
	void setPatchWidth(int width) {}

	/**
	 * @return The height of the image data of the extracted patches.
	 */
	int getPatchHeight() const {
		return 20;
	}

	/**
	 * @param[in] height The new height of the image data of the extracted patches.
	 */
	void setPatchHeight(int height) {}

	/**
	 * Changes the size of the image data of the extracted patches.
	 *
	 * @param[in] width The new width of the image data of the extracted patches.
	 * @param[in] height The new height of the image data of the extracted patches.
	 */
	void setPatchSize(int width, int height) {}

private:

	struct HaarFeature {
		vector<Rect_<float>> rects;
		vector<float> weights;
		float factor;
		float area;
	};

	int version;                            ///< The version number.
	Mat image;
	GrayscaleFilter filter;
	IntegralImageFilter integralFilter;
	vector<HaarFeature> features;
};

} /* namespace imageprocessing */
#endif /* HAARIMAGEEXTRACTOR_HPP_ */

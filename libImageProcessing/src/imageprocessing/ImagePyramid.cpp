/*
 * ImagePyramid.cpp
 *
 *  Created on: 15.02.2013
 *      Author: huber & poschmann
 */

#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/MultipleImageFilter.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <stdexcept>

using cv::Size;
using cv::resize;
using std::make_shared;
using std::invalid_argument;

namespace imageprocessing {

// cheap comparison of two images - they are equal if they share the same size, flags and data (no element-wise comparison)
static bool imagesAreEqual(const Mat& lhs, const Mat& rhs) {
	return lhs.cols == rhs.cols && lhs.rows == rhs.rows && lhs.flags == rhs.flags && lhs.data == rhs.data;
}

ImagePyramid::ImagePyramid(double minScaleFactor, double maxScaleFactor, double incrementalScaleFactor) :
		minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor), incrementalScaleFactor(incrementalScaleFactor),
		firstLayer(0), layers(), sourceImage(Mat()), sourcePyramid(),
		imageFilter(make_shared<MultipleImageFilter>()), layerFilter(make_shared<MultipleImageFilter>()) {}

ImagePyramid::ImagePyramid(shared_ptr<ImagePyramid> pyramid, double minScaleFactor, double maxScaleFactor) :
		minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor), incrementalScaleFactor(pyramid->incrementalScaleFactor),
		firstLayer(0), layers(), sourceImage(Mat()), sourcePyramid(pyramid),
		imageFilter(make_shared<MultipleImageFilter>()), layerFilter(make_shared<MultipleImageFilter>()) {}

ImagePyramid::~ImagePyramid() {}

void ImagePyramid::setSource(const Mat& image) {
	if (incrementalScaleFactor <= 0 || incrementalScaleFactor >= 1)
		throw invalid_argument("ImagePyramid: the incremental scale factor must be greater than zero and smaller than one");
	if (minScaleFactor <= 0)
		throw invalid_argument("ImagePyramid: the minimum scale factor must be greater than zero");
	if (maxScaleFactor > 1)
		throw invalid_argument("ImagePyramid: the maximum scale factor must not exceed one");
	sourcePyramid.reset();
	sourceImage = image;
}

void ImagePyramid::setSource(shared_ptr<ImagePyramid> pyramid) {
	sourceImage = Mat();
	sourcePyramid = pyramid;
}

void ImagePyramid::update() {
	if (sourcePyramid) {
		incrementalScaleFactor = sourcePyramid->incrementalScaleFactor;
		layers.clear();
		const vector<shared_ptr<ImagePyramidLayer>>& sourceLayers = sourcePyramid->layers;
		for (vector<shared_ptr<ImagePyramidLayer>>::const_iterator layIt = sourceLayers.begin(); layIt != sourceLayers.end(); ++layIt) {
			shared_ptr<ImagePyramidLayer> layer = *layIt;
			if (layer->getScaleFactor() > maxScaleFactor)
				continue;
			if (layer->getScaleFactor() < minScaleFactor)
				break;
			if (layers.empty())
				firstLayer = layer->getIndex();
			layers.push_back(make_shared<ImagePyramidLayer>(
					layer->getIndex(), layer->getScaleFactor(), layerFilter->applyTo(layer->getScaledImage())));
		}
	} else if (!sourceImage.empty()) {
		layers.clear();
		Mat filteredImage = imageFilter->applyTo(sourceImage);
		double scaleFactor = 1;
		Mat previousScaledImage;
		for (int i = 0; ; ++i, scaleFactor *= incrementalScaleFactor) {
			if (scaleFactor > maxScaleFactor)	// This goes into an endless loop if the user specifies a scale-factor with which it is impossible to produce the desired pyramids. We could add a check for this here, but I (Patrik) think we could also leave it this way because every 'if' costs runtime performance.
				continue;
			if (scaleFactor < minScaleFactor)
				break;

			// All but the first scaled image use the previous scaled image as the base,
			// therefore the scaling itself adds more and more blur to the image
			// and because of that no additional Gaussian blur is applied.
			// When scaling the image down a lot, the bilinear interpolation would lead to some artifacts (higher frequencies),
			// therefore the first down-scaling is done using an area interpolation which produces much better results.
			// The bilinear interpolation is used for the following down-scalings because of speed and similar results as area.
			// TODO resizing-strategy
			Mat scaledImage;
			Size scaledImageSize(cvRound(sourceImage.cols * scaleFactor), cvRound(sourceImage.rows * scaleFactor));
			if (layers.empty()) {
				firstLayer = i;
				resize(filteredImage, scaledImage, scaledImageSize, 0, 0, cv::INTER_AREA);
			} else {
				resize(previousScaledImage, scaledImage, scaledImageSize, 0, 0, cv::INTER_LINEAR);
			}
			layers.push_back(make_shared<ImagePyramidLayer>(i, scaleFactor, layerFilter->applyTo(scaledImage)));
			previousScaledImage = scaledImage;
		}
	} else { // neither source pyramid nor source image are set, therefore the other parameters are missing, too
		// TODO exception? oder ignore? oder noch besser: logging!
		std::cerr << "ImagePyramid: could not update because there is no source (image or pyramid)" << std::endl;
	}
}

void ImagePyramid::update(const Mat& image) {
	if (sourcePyramid) {
		sourcePyramid->update(image);
		update();
	} else if (!imagesAreEqual(sourceImage, image)) {
		setSource(image);
		update();
	}
}

void ImagePyramid::addImageFilter(shared_ptr<ImageFilter> filter) {
	imageFilter->add(filter);
}

void ImagePyramid::addLayerFilter(shared_ptr<ImageFilter> filter) {
	layerFilter->add(filter);
}

const shared_ptr<ImagePyramidLayer> ImagePyramid::getLayer(double scaleFactor) const {
	double power = log(scaleFactor) / log(incrementalScaleFactor);
	int index = cvRound(power) - firstLayer;
	if (index < 0 || index >= (int)layers.size())
		return NULL;
	return layers[index];
}

} /* namespace imageprocessing */

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

using cv::Size;
using cv::resize;
using std::make_shared;

namespace imageprocessing {

// cheap comparison of two images - they are equal if they share the same data (no element-wise comparison)
static bool imagesAreEqual(const Mat& lhs, const Mat& rhs) {
	return lhs.cols == rhs.cols && lhs.rows == rhs.rows && lhs.flags == rhs.flags && lhs.data == rhs.data;
}

ImagePyramid::ImagePyramid() :
		incrementalScaleFactor(0), maxScaleFactor(0), minScaleFactor(0),
		firstLayer(0), layers(), sourceImage(Mat()), sourcePyramid(),
		imageFilter(make_shared<MultipleImageFilter>()), layerFilter(make_shared<MultipleImageFilter>()) {}

ImagePyramid::~ImagePyramid() {}

void ImagePyramid::build(const Mat& image, double incrementalScaleFactor, double maxScaleFactor, double minScaleFactor) {
	sourcePyramid.reset();
	sourceImage = image;
	this->incrementalScaleFactor = incrementalScaleFactor;
	this->maxScaleFactor = maxScaleFactor;
	this->minScaleFactor = minScaleFactor;
	layers.clear();
	Mat filteredImage = imageFilter->applyTo(image);
	double scaleFactor = 1;
	Mat previousScaledImage;
	for (int i = 0; ; ++i, scaleFactor *= incrementalScaleFactor) {
		if (scaleFactor > maxScaleFactor)
			continue;
		if (scaleFactor < minScaleFactor)
			break;

		// All but the first scaled image use the previous scaled image as the base,
		// therefore the scaling itself adds more and more blur to the image
		// and because of that no additional Gaussian blur is applied.
		// When scaling the image down a lot, the bilinear interpolation would lead to some artifacts (higher frequencies),
		// therefore the first down-scaling is done using an area interpolation which produces much better results.
		// The bilinear interpolation is used for the following down-scalings because of speed and similar results as area.
		Mat scaledImage;
		Size scaledImageSize(cvRound(image.cols / scaleFactor), cvRound(image.rows / scaleFactor));
		if (layers.empty()) {
			firstLayer = i;
			resize(filteredImage, scaledImage, scaledImageSize, 0, 0, cv::INTER_AREA);
		} else {
			resize(previousScaledImage, scaledImage, scaledImageSize, 0, 0, cv::INTER_LINEAR);
		}
		previousScaledImage = scaledImage;
		layers.push_back(make_shared<ImagePyramidLayer>(i, scaleFactor, layerFilter->applyTo(scaledImage)));
	}
}

void ImagePyramid::build(shared_ptr<ImagePyramid> pyramid) {
	build(pyramid, pyramid->maxScaleFactor, pyramid->minScaleFactor);
}

void ImagePyramid::build(shared_ptr<ImagePyramid> pyramid, double maxScaleFactor, double minScaleFactor) {
	sourceImage = Mat();
	sourcePyramid = pyramid;
	incrementalScaleFactor = sourcePyramid->incrementalScaleFactor;
	this->maxScaleFactor = maxScaleFactor;
	this->minScaleFactor = minScaleFactor;
	firstLayer = sourcePyramid->firstLayer;
	layers.clear();
	const vector<shared_ptr<ImagePyramidLayer>>& sourceLayers = sourcePyramid->layers;
	for (vector<shared_ptr<ImagePyramidLayer>>::const_iterator layIt = sourceLayers.begin(); layIt != sourceLayers.end(); ++layIt) {
		shared_ptr<ImagePyramidLayer> layer = *layIt;
		if (layer->getScaleFactor() > maxScaleFactor)
			continue;
		if (layer->getScaleFactor() < minScaleFactor)
			break;
		layers.push_back(make_shared<ImagePyramidLayer>(layer->getIndex(), layer->getScaleFactor(), layerFilter->applyTo(layer->getScaledImage())));
	}
}

void ImagePyramid::update(const Mat& image) {
	if (sourcePyramid) {
		sourcePyramid->update(image);
		const vector<shared_ptr<ImagePyramidLayer>>& sourceLayers = sourcePyramid->layers;
		vector<shared_ptr<ImagePyramidLayer>>::iterator layIt = layers.begin();
		vector<shared_ptr<ImagePyramidLayer>>::const_iterator slayIt = sourceLayers.begin();
		for (; layIt != layers.end() && slayIt != sourceLayers.end(); ++layIt, ++slayIt) {
			shared_ptr<ImagePyramidLayer> layer = *layIt;
			shared_ptr<ImagePyramidLayer> sourceLayer = *slayIt;
			layerFilter->applyTo(sourceLayer->getScaledImage(), layer->getScaledImage());
		}
	} else if (!sourceImage.empty()) {
		if (!imagesAreEqual(sourceImage, image)) {
			if (!layers.empty()) {
				Mat filteredImage = imageFilter->applyTo(image);
				vector<shared_ptr<ImagePyramidLayer>>::iterator layIt = layers.begin();
				shared_ptr<ImagePyramidLayer> layer = *layIt;
				Mat scaledImage;
				Size scaledImageSize(cvRound(image.cols / layer->getScaleFactor()), cvRound(image.rows / layer->getScaleFactor()));
				resize(filteredImage, scaledImage, scaledImageSize, 0, 0, cv::INTER_AREA);
				layerFilter->applyTo(scaledImage, layer->getScaledImage());
				Mat previousScaledImage = scaledImage;
				for (; layIt != layers.end(); ++layIt) {
					layer = *layIt;
					scaledImageSize.width = cvRound(image.cols / layer->getScaleFactor());
					scaledImageSize.height = cvRound(image.rows / layer->getScaleFactor());
					resize(previousScaledImage, scaledImage, scaledImageSize, 0, 0, cv::INTER_LINEAR);
					layerFilter->applyTo(scaledImage, layer->getScaledImage());
					previousScaledImage = scaledImage;
				}
			}
		}
	} else { // neither source pyramid nor source image are set, therefore the other parameters are missing, too
		// TODO exception? oder ignore?
		std::cerr << "image pyramid could not be updated because it was never built" << std::endl;
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

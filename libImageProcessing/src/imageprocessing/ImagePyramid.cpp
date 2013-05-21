/*
 * ImagePyramid.cpp
 *
 *  Created on: 15.02.2013
 *      Author: huber & poschmann
 */

#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/MultipleImageFilter.hpp"
#include "logging/LoggerFactory.hpp"
#include "logging/Logger.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "boost/iterator/indirect_iterator.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

using logging::LoggerFactory;
using cv::Size;
using cv::resize;
using boost::make_indirect_iterator;
using std::string; // TODO nur für createException
using std::ostringstream; // TODO nur für createException
using std::make_shared;
using std::invalid_argument;

namespace imageprocessing {

template<class T>// TODO in header-datei verschieben, die in jedem projekt eingebunden werden kann
static T createException(const string& file, int line, const string& message) {
	ostringstream text;
	text << file.substr(file.find_last_of("/\\") + 1) << ':' << line << ':' << ' ' << message;
	return T(text.str());
}

ImagePyramid::ImagePyramid(double minScaleFactor, double maxScaleFactor, double incrementalScaleFactor) :
		minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor), incrementalScaleFactor(incrementalScaleFactor),
		firstLayer(0), layers(), sourceImage(), sourcePyramid(), version(-1),
		imageFilter(make_shared<MultipleImageFilter>()), layerFilter(make_shared<MultipleImageFilter>()) {
	if (incrementalScaleFactor <= 0 || incrementalScaleFactor >= 1)
		throw createException<invalid_argument>(__FILE__, __LINE__, "the incremental scale factor must be greater than zero and smaller than one");
	if (maxScaleFactor > 1)
		throw createException<invalid_argument>(__FILE__, __LINE__, "the maximum scale factor must not exceed one");
}

ImagePyramid::ImagePyramid(shared_ptr<ImagePyramid> pyramid, double minScaleFactor, double maxScaleFactor) :
		minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor), incrementalScaleFactor(pyramid->incrementalScaleFactor),
		firstLayer(0), layers(), sourceImage(), sourcePyramid(pyramid), version(-1),
		imageFilter(make_shared<MultipleImageFilter>()), layerFilter(make_shared<MultipleImageFilter>()) {}

ImagePyramid::~ImagePyramid() {}

void ImagePyramid::setSource(const Mat& image) {
	setSource(make_shared<VersionedImage>(image));
	version = -1;
}

void ImagePyramid::setSource(shared_ptr<VersionedImage> image) {
	if (minScaleFactor <= 0) // this is checked here because it is allowed to be zero if this pyramid has another pyramid as its source
		throw createException<invalid_argument>(__FILE__, __LINE__, "the minimum scale factor must be greater than zero");
	sourcePyramid.reset();
	sourceImage = image;
}

void ImagePyramid::setSource(shared_ptr<ImagePyramid> pyramid) {
	sourceImage.reset();
	sourcePyramid = pyramid;
}

void ImagePyramid::update() {
	if (sourceImage) {
		if (version != sourceImage->getVersion()) {
			layers.clear();
			Mat filteredImage = imageFilter->applyTo(sourceImage->getData());
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
				// TODO resizing-strategy
				Mat scaledImage;
				Size scaledImageSize(cvRound(sourceImage->getData().cols * scaleFactor), cvRound(sourceImage->getData().rows * scaleFactor));
				if (layers.empty()) {
					firstLayer = i;
					resize(filteredImage, scaledImage, scaledImageSize, 0, 0, cv::INTER_AREA);
				} else {
					resize(previousScaledImage, scaledImage, scaledImageSize, 0, 0, cv::INTER_LINEAR);
				}
				layers.push_back(make_shared<ImagePyramidLayer>(i, scaleFactor, layerFilter->applyTo(scaledImage)));
				previousScaledImage = scaledImage;
			}
			version = sourceImage->getVersion();
		}
	} else if (sourcePyramid) {
		if (version != sourcePyramid->getVersion()) {
			incrementalScaleFactor = sourcePyramid->incrementalScaleFactor;
			layers.clear();
			const vector<shared_ptr<ImagePyramidLayer>>& sourceLayers = sourcePyramid->layers;
			for (auto layer = make_indirect_iterator(layers.begin()); layer != make_indirect_iterator(layers.end()); ++layer) {
				if (layer->getScaleFactor() > maxScaleFactor)
					continue;
				if (layer->getScaleFactor() < minScaleFactor)
					break;
				if (layers.empty())
					firstLayer = layer->getIndex();
				layers.push_back(make_shared<ImagePyramidLayer>(
						layer->getIndex(), layer->getScaleFactor(), layerFilter->applyTo(layer->getScaledImage())));
			}
			version = sourcePyramid->getVersion();
		}
	} else { // neither source pyramid nor source image are set, therefore the other parameters are missing, too
		Loggers->getLogger("ImageProcessing").warn("ImagePyramid: could not update because there is no source (image or pyramid)");
	}
}

void ImagePyramid::update(const Mat& image) {
	if (sourcePyramid) {
		sourcePyramid->update(image);
		update();
	} else {
		setSource(image);
		update();
	}
}

void ImagePyramid::update(shared_ptr<VersionedImage> image) {
	if (sourcePyramid) {
		sourcePyramid->update(image);
		update();
	} else {
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

Size ImagePyramid::getImageSize() const {
	if (sourceImage)
		return Size(sourceImage->getData().cols, sourceImage->getData().rows);
	else if (sourcePyramid)
		return sourcePyramid->getImageSize();
	else
		return Size();
}

const shared_ptr<ImagePyramidLayer> ImagePyramid::getLayer(int index) const {
	index -= firstLayer;
	if (index < 0 || index >= (int)layers.size())
		return nullptr;
	return layers[index];
}

const shared_ptr<ImagePyramidLayer> ImagePyramid::getLayer(double scaleFactor) const {
	double power = log(scaleFactor) / log(incrementalScaleFactor);
	return getLayer(cvRound(power));
}

vector<Size> ImagePyramid::getLayerSizes() const {
	vector<Size> sizes;
	sizes.resize(layers.size());
	for (auto layer = make_indirect_iterator(layers.begin()); layer != make_indirect_iterator(layers.end()); ++layer)
		sizes.push_back(layer->getSize());
	return sizes;
}

} /* namespace imageprocessing */

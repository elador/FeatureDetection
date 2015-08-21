/*
 * ImagePyramid.cpp
 *
 *  Created on: 15.02.2013
 *      Author: huber & poschmann
 */

#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/ChainedFilter.hpp"
#include "logging/LoggerFactory.hpp"
#include "logging/Logger.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <sstream>
#include <stdexcept>

using logging::LoggerFactory;
using cv::Mat;
using cv::Size;
using cv::resize;
using std::vector;
using std::shared_ptr;
using std::pair;
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

ImagePyramid::ImagePyramid(size_t octaveLayerCount, double minScaleFactor, double maxScaleFactor) :
		octaveLayerCount(octaveLayerCount), incrementalScaleFactor(0),
		minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor),
		firstLayer(0), layers(), sourceImage(), sourcePyramid(), version(-1),
		imageFilter(make_shared<ChainedFilter>()), layerFilter(make_shared<ChainedFilter>()) {
	if (octaveLayerCount == 0)
		throw createException<invalid_argument>(__FILE__, __LINE__, "the number of layers per octave must be greater than zero");
	if (minScaleFactor <= 0)
		throw createException<invalid_argument>(__FILE__, __LINE__, "the minimum scale factor must be greater than zero");
	if (maxScaleFactor > 1)
		throw createException<invalid_argument>(__FILE__, __LINE__, "the maximum scale factor must not exceed one");
	incrementalScaleFactor = pow(0.5, 1. / octaveLayerCount);
}

ImagePyramid::ImagePyramid(double incrementalScaleFactor, double minScaleFactor, double maxScaleFactor) :
		octaveLayerCount(0), incrementalScaleFactor(0),
		minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor),
		firstLayer(0), layers(), sourceImage(), sourcePyramid(), version(-1),
		imageFilter(make_shared<ChainedFilter>()), layerFilter(make_shared<ChainedFilter>()) {
	if (incrementalScaleFactor <= 0 || incrementalScaleFactor >= 1)
		throw createException<invalid_argument>(__FILE__, __LINE__, "the incremental scale factor must be greater than zero and smaller than one");
	if (minScaleFactor <= 0)
		throw createException<invalid_argument>(__FILE__, __LINE__, "the minimum scale factor must be greater than zero");
	if (maxScaleFactor > 1)
		throw createException<invalid_argument>(__FILE__, __LINE__, "the maximum scale factor must not exceed one");
	octaveLayerCount = static_cast<size_t>(std::round(log(0.5) / log(incrementalScaleFactor)));
	this->incrementalScaleFactor = pow(0.5, 1. / octaveLayerCount);
}

ImagePyramid::ImagePyramid(double minScaleFactor, double maxScaleFactor) :
		octaveLayerCount(0), incrementalScaleFactor(0),
		minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor),
		firstLayer(0), layers(), sourceImage(), sourcePyramid(), version(-1),
		imageFilter(make_shared<ChainedFilter>()), layerFilter(make_shared<ChainedFilter>()) {}

ImagePyramid::ImagePyramid(shared_ptr<ImagePyramid> pyramid, double minScaleFactor, double maxScaleFactor) :
		octaveLayerCount(pyramid->octaveLayerCount), incrementalScaleFactor(pyramid->incrementalScaleFactor),
		minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor),
		firstLayer(0), layers(), sourceImage(), sourcePyramid(pyramid), version(-1),
		imageFilter(make_shared<ChainedFilter>()), layerFilter(make_shared<ChainedFilter>()) {}

void ImagePyramid::setSource(const Mat& image) {
	setSource(make_shared<VersionedImage>(image));
	version = -1;
}

void ImagePyramid::setSource(const shared_ptr<VersionedImage>& image) {
	if (octaveLayerCount == 0)
		throw createException<invalid_argument>(__FILE__, __LINE__, "the number of layers per octave must be greater than zero");
	if (minScaleFactor <= 0)
		throw createException<invalid_argument>(__FILE__, __LINE__, "the minimum scale factor must be greater than zero");
	sourcePyramid.reset();
	sourceImage = image;
}

void ImagePyramid::setSource(const shared_ptr<ImagePyramid>& pyramid) {
	sourceImage.reset();
	sourcePyramid = pyramid;
}

void ImagePyramid::update() {
	if (sourceImage) {
		if (version != sourceImage->getVersion()) {
			layers.clear();
			Mat filteredImage = imageFilter->applyTo(sourceImage->getData());
			// TODO wenn maxscale <= 0.5 -> erstmal pyrdown auf bild (etc pp)
			for (size_t i = 0; i < octaveLayerCount; ++i) {
				double scaleFactor = pow(incrementalScaleFactor, i);
				Mat scaledImage;
				Size scaledImageSize(cvRound(filteredImage.cols * scaleFactor), cvRound(filteredImage.rows * scaleFactor));
				resize(filteredImage, scaledImage, scaledImageSize, 0, 0, cv::INTER_LINEAR);
				if (scaleFactor <= maxScaleFactor && scaleFactor >= minScaleFactor)
					layers.push_back(make_shared<ImagePyramidLayer>(i, scaleFactor, layerFilter->applyTo(scaledImage)));
				Mat previousScaledImage = scaledImage;
				scaleFactor *= 0.5;
				for (size_t j = 1; scaleFactor >= minScaleFactor && previousScaledImage.cols > 1; ++j, scaleFactor *= 0.5) {
					Mat downSampledImage;
					pyrDown(previousScaledImage, scaledImage);
					if (scaleFactor <= maxScaleFactor)
						layers.push_back(make_shared<ImagePyramidLayer>(i + j * octaveLayerCount, scaleFactor, layerFilter->applyTo(scaledImage)));
					previousScaledImage = scaledImage;
				}
			}
			std::sort(layers.begin(), layers.end(), [](const shared_ptr<ImagePyramidLayer>& a, const shared_ptr<ImagePyramidLayer>& b) {
				return a->getIndex() < b->getIndex();
			});
			if (!layers.empty())
				firstLayer = layers.front()->getIndex();
			version = sourceImage->getVersion();
		}
	} else if (sourcePyramid) {
		if (version != sourcePyramid->getVersion()) {
			incrementalScaleFactor = sourcePyramid->incrementalScaleFactor;
			layers.clear();
			for (const shared_ptr<ImagePyramidLayer>& layer : sourcePyramid->layers) {
				if (layer->getScaleFactor() > maxScaleFactor)
					continue;
				if (layer->getScaleFactor() < minScaleFactor)
					break;
				layers.push_back(make_shared<ImagePyramidLayer>(
						layer->getIndex(), layer->getScaleFactor(), layerFilter->applyTo(layer->getScaledImage())));
			}
			if (!layers.empty())
				firstLayer = layers.front()->getIndex();
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

void ImagePyramid::update(const shared_ptr<VersionedImage>& image) {
	if (sourcePyramid) {
		sourcePyramid->update(image);
		update();
	} else {
		setSource(image);
		update();
	}
}

void ImagePyramid::addImageFilter(const shared_ptr<ImageFilter>& filter) {
	imageFilter->add(filter);
}

void ImagePyramid::addLayerFilter(const shared_ptr<ImageFilter>& filter) {
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
	int realIndex = index - firstLayer;
	if (realIndex < 0 || realIndex >= (int)layers.size())
		return shared_ptr<ImagePyramidLayer>();
	return layers[realIndex];
}

const shared_ptr<ImagePyramidLayer> ImagePyramid::getLayer(double scaleFactor) const {
	double power = log(scaleFactor) / log(incrementalScaleFactor);
	return getLayer(static_cast<int>(std::round(power)));
}

vector<pair<int, double>> ImagePyramid::getLayerScales() const {
	vector<pair<int, double>> scales;
	scales.reserve(layers.size());
	for (size_t i = 0; i < layers.size(); ++i)
		scales.emplace_back(i + firstLayer, layers[i]->getScaleFactor());
	return scales;
}

vector<Size> ImagePyramid::getLayerSizes() const {
	vector<Size> sizes;
	sizes.reserve(layers.size());
	for (const auto& layer : layers)
		sizes.push_back(layer->getSize());
	return sizes;
}

} /* namespace imageprocessing */

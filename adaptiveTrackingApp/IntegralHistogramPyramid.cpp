/*
 * ImagePyramid.cpp
 *
 *  Created on: 15.02.2013
 *      Author: huber & poschmann
 */

#include "IntegralHistogramPyramid.hpp"
#include "IntegralHistogramPyramidLayer.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/MultipleImageFilter.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/GradientFilter.hpp"
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
using std::make_shared;
using std::invalid_argument;

namespace imageprocessing {

IntegralHistogramPyramid::IntegralHistogramPyramid(unsigned int bins, bool signedGradients, double offset,
		double minScaleFactor, double maxScaleFactor, double incrementalScaleFactor) :
				minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor), incrementalScaleFactor(incrementalScaleFactor),
				firstLayer(0), layers(), bins(bins), offset(offset), sourceImage(), sourcePyramid(), version(-1),
				imageFilter(make_shared<MultipleImageFilter>()), layerFilter(make_shared<MultipleImageFilter>()) {
	if (incrementalScaleFactor <= 0 || incrementalScaleFactor >= 1)
		throw invalid_argument("the incremental scale factor must be greater than zero and smaller than one");
	if (maxScaleFactor > 1)
		throw invalid_argument("the maximum scale factor must not exceed one");
	imageFilter->add(make_shared<GrayscaleFilter>());
	layerFilter->add(make_shared<GradientFilter>(1, 0));
	union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} gradientCode;
	Vec2b binCode;
	// build the look-up table
	// index of the look-up table is the binary concatanation of the gradients of x and y
	// value of the look-up table is the binary concatanation of the bin index and weight (scaled to 255)
	gradientCode.gradient.x = 0;
	for (int x = 0; x < 256; ++x) {
		double gradientX = (static_cast<double>(x) - 127) / 127;
		gradientCode.gradient.y = 0;
		for (int y = 0; y < 256; ++y) {
			double gradientY = (static_cast<double>(y) - 127) / 127;
			double direction = atan2(gradientY, gradientX);
			double magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);
			if (signedGradients) {
				direction += CV_PI;
				binCode[0] = static_cast<uchar>(floor((direction + offset) * bins / (2 * CV_PI))) % bins;
			} else { // unsigned gradients
				if (direction < 0)
					direction += CV_PI;
				binCode[0] = static_cast<uchar>(floor((direction + offset) * bins / CV_PI)) % bins;
			}
			binCode[1] = cv::saturate_cast<uchar>(255 * magnitude);
			binCodes[gradientCode.index] = binCode;
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

IntegralHistogramPyramid::IntegralHistogramPyramid(unsigned int bins, bool signedGradients, double offset,
		shared_ptr<ImagePyramid> pyramid, double minScaleFactor, double maxScaleFactor) :
				minScaleFactor(minScaleFactor), maxScaleFactor(maxScaleFactor), incrementalScaleFactor(pyramid->getIncrementalScaleFactor()),
				firstLayer(0), layers(), bins(bins), offset(offset), sourceImage(), sourcePyramid(pyramid), version(-1),
				imageFilter(make_shared<MultipleImageFilter>()), layerFilter(make_shared<MultipleImageFilter>()) {
	imageFilter->add(make_shared<GrayscaleFilter>());
	layerFilter->add(make_shared<GradientFilter>(1, 0));
	union {
		ushort index;
		struct {
			uchar x, y;
		} gradient;
	} gradientCode;
	Vec2b binCode;
	// build the look-up table
	// index of the look-up table is the binary concatanation of the gradients of x and y
	// value of the look-up table is the binary concatanation of the bin index and weight (scaled to 255)
	gradientCode.gradient.x = 0;
	for (int x = 0; x < 256; ++x) {
		double gradientX = (static_cast<double>(x) - 127) / 127;
		gradientCode.gradient.y = 0;
		for (int y = 0; y < 256; ++y) {
			double gradientY = (static_cast<double>(y) - 127) / 127;
			double direction = atan2(gradientY, gradientX);
			double magnitude = sqrt(gradientX * gradientX + gradientY * gradientY);
			if (signedGradients) {
				direction += CV_PI;
				binCode[0] = static_cast<uchar>(floor((direction + offset) * bins / (2 * CV_PI))) % bins;
			} else { // unsigned gradients
				if (direction < 0)
					direction += CV_PI;
				binCode[0] = static_cast<uchar>(floor((direction + offset) * bins / CV_PI)) % bins;
			}
			binCode[1] = cv::saturate_cast<uchar>(255 * magnitude);
			binCodes[gradientCode.index] = binCode;
			++gradientCode.gradient.y;
		}
		++gradientCode.gradient.x;
	}
}

IntegralHistogramPyramid::~IntegralHistogramPyramid() {}

void IntegralHistogramPyramid::setSource(const Mat& image) {
	setSource(make_shared<VersionedImage>(image));
	version = -1;
}

void IntegralHistogramPyramid::setSource(shared_ptr<VersionedImage> image) {
	if (minScaleFactor <= 0) // this is checked here because it is allowed to be zero if this pyramid has another pyramid as its source
		throw invalid_argument("the minimum scale factor must be greater than zero");
	sourcePyramid.reset();
	sourceImage = image;
}

void IntegralHistogramPyramid::setSource(shared_ptr<ImagePyramid> pyramid) {
	sourceImage.reset();
	sourcePyramid = pyramid;
}

void IntegralHistogramPyramid::update() {
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
				layers.push_back(make_shared<IntegralHistogramPyramidLayer>(i, scaleFactor, createIntegralHistogram(scaledImage)));
				previousScaledImage = scaledImage;
			}
			version = sourceImage->getVersion();
		}
	} else if (sourcePyramid) {
		if (version != sourcePyramid->getVersion()) {
			incrementalScaleFactor = sourcePyramid->getIncrementalScaleFactor();
			layers.clear();
			const vector<shared_ptr<ImagePyramidLayer>>& sourceLayers = sourcePyramid->getLayers();
			for (auto layer = make_indirect_iterator(sourceLayers.begin()); layer != make_indirect_iterator(sourceLayers.end()); ++layer) {
				if (layer->getScaleFactor() > maxScaleFactor)
					continue;
				if (layer->getScaleFactor() < minScaleFactor)
					break;
				if (layers.empty())
					firstLayer = layer->getIndex();
				layers.push_back(make_shared<IntegralHistogramPyramidLayer>(
						layer->getIndex(), layer->getScaleFactor(), createIntegralHistogram(layer->getScaledImage())));
			}
			version = sourcePyramid->getVersion();
		}
	} else { // neither source pyramid nor source image are set, therefore the other parameters are missing, too
		Loggers->getLogger("ImageProcessing").warn("ImagePyramid: could not update because there is no source (image or pyramid)");
	}
}

vector<Mat> IntegralHistogramPyramid::createIntegralHistogram(const Mat& image) {
	Mat gradientImage = layerFilter->applyTo(image);
	if (gradientImage.type() != CV_8UC2)
		throw invalid_argument("GradientHistogramFilter: the image must by of type CV_8UC2");

	int rows = gradientImage.rows;
	int cols = gradientImage.cols;
	Mat histogramBin(rows, cols, CV_32F);
	if (gradientImage.isContinuous() && histogramBin.isContinuous()) {
		cols *= rows;
		rows = 1;
	}
	vector<Mat> integralHistogram;
	integralHistogram.reserve(bins);
	for (unsigned int bin = 0; bin < bins; ++bin) {
		histogramBin.setTo(0);
		for (int row = 0; row < rows; ++row) {
			const ushort* gradientCode = gradientImage.ptr<ushort>(row); // concatenation of x gradient and y gradient (both uchar)
			float* histogramBinRow = histogramBin.ptr<float>(row);
			for (int col = 0; col < cols; ++col) {
				Vec2b& binCode = binCodes[gradientCode[col]];
				if (binCode[0] == bin)
					histogramBinRow[col] = static_cast<float>(binCode[1]) / 255.0f;
			}
		}
		Mat integralHistogramBin(gradientImage.rows + 1, gradientImage.cols + 1, CV_32F);
		integral(histogramBin, integralHistogramBin, CV_32F);
		integralHistogram.push_back(integralHistogramBin);
	}
	return integralHistogram;
}

void IntegralHistogramPyramid::update(const Mat& image) {
	if (sourcePyramid) {
		sourcePyramid->update(image);
		update();
	} else {
		setSource(image);
		update();
	}
}

void IntegralHistogramPyramid::update(shared_ptr<VersionedImage> image) {
	if (sourcePyramid) {
		sourcePyramid->update(image);
		update();
	} else {
		setSource(image);
		update();
	}
}

Size IntegralHistogramPyramid::getImageSize() const {
	if (sourceImage)
		return Size(sourceImage->getData().cols, sourceImage->getData().rows);
	else if (sourcePyramid)
		return sourcePyramid->getImageSize();
	else
		return Size();
}

const shared_ptr<IntegralHistogramPyramidLayer> IntegralHistogramPyramid::getLayer(int index) const {
	index -= firstLayer;
	if (index < 0 || index >= (int)layers.size())
		return nullptr;
	return layers[index];
}

const shared_ptr<IntegralHistogramPyramidLayer> IntegralHistogramPyramid::getLayer(double scaleFactor) const {
	double power = log(scaleFactor) / log(incrementalScaleFactor);
	return getLayer(cvRound(power));
}

vector<Size> IntegralHistogramPyramid::getLayerSizes() const {
	vector<Size> sizes;
	sizes.resize(layers.size());
	for (auto layer = make_indirect_iterator(layers.begin()); layer != make_indirect_iterator(layers.end()); ++layer)
		sizes.push_back(layer->getSize());
	return sizes;
}

} /* namespace imageprocessing */

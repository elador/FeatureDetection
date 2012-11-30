/*
 * ImagePyramidFeatureExtractor.cpp
 *
 *  Created on: 19.11.2012
 *      Author: poschmann
 */

#include "classification/ImagePyramidFeatureExtractor.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <utility>

using cv::Size;
using cv::Mat;
using cv::Rect;
using std::vector;
using std::min;
using std::max;
using std::pair;

namespace classification {

Patch::Patch(int x, int y) : x(x), y(y), featureVector() {}

Patch::~Patch() {}

bool Patch::operator==(const Patch& other) const {
	return this->x == other.x && this->y == other.y;
}

PyramidLevel::PyramidLevel(double scaleFactor, const Mat& scaledImage) :
		scaleFactor(scaleFactor), scaledImage(scaledImage), patches() {}

PyramidLevel::~PyramidLevel() {}

int PyramidLevel::getScaled(int value) const {
	return cvRound(value / scaleFactor);
}

int PyramidLevel::getOriginal(int value) const {
	return cvRound(value * scaleFactor);
}

double PyramidLevel::getScaleFactor() const {
	return scaleFactor;
}

const Mat& PyramidLevel::getScaledImage() const {
	return scaledImage;
}

ImagePyramidFeatureExtractor::ImagePyramidFeatureExtractor(
		Size featureSize, double scaleFactor, double minHeight, double maxHeight) :
				featureSize(featureSize),
				minHeight(minHeight),
				maxHeight(maxHeight),
				scaleFactor(scaleFactor),
				firstLevel(0),
				levels() {
	if (this->scaleFactor < 1)
		this->scaleFactor = 1 / this->scaleFactor;
	this->minHeight = max(0.0, min(1.0, this->minHeight));
	this->maxHeight = max(0.0, min(1.0, this->maxHeight));
}

ImagePyramidFeatureExtractor::~ImagePyramidFeatureExtractor() {
	clearLevels();
}

void ImagePyramidFeatureExtractor::init(const Mat& image) {
	clearLevels();
	Mat grayImage = image;
	if (grayImage.channels() > 1) {
		Mat temp;
		cvtColor(grayImage, temp, CV_BGR2GRAY);
		grayImage = temp;
	}
	Size minSize;
	minSize.height = max(featureSize.height, cvRound(minHeight * image.rows));
	minSize.width = max(featureSize.width, cvRound(minSize.height * featureSize.width / featureSize.height));
	Size maxSize;
	maxSize.height = min(image.rows, cvRound(maxHeight * image.rows));
	maxSize.width = min(image.cols, cvRound(maxSize.height * featureSize.width / featureSize.height));
	double factor = 1;
	for (int i = 0; ; ++i, factor *= scaleFactor) {
		Size scaledFeatureSize(cvRound(factor * featureSize.width), cvRound(factor * featureSize.height));
		if (scaledFeatureSize.width < minSize.width || scaledFeatureSize.height < minSize.height)
			continue;
		if (scaledFeatureSize.width > maxSize.width || scaledFeatureSize.height > maxSize.height)
			break;

		// All but the first scaled image use the previous scaled image as the base,
		// therefore the scaling itself adds more and more blur to the image
		// and because of that no additional guassian blur is applied.
		// When scaling the image down a lot, the bilinear interpolation would lead to some artefacts (higher frequencies),
		// therefore the first down-scaling is done using an area interpolation which produces much better results.
		// The bilinear interpolation is used for the following down-scalings because of speed and similar results as area.
		Mat scaledImage;
		Size scaledImageSize(cvRound(image.cols / factor), cvRound(image.rows / factor));
		if (levels.empty()) {
			firstLevel = i;
			resize(grayImage, scaledImage, scaledImageSize, 0, 0, cv::INTER_AREA);
		} else {
			resize(levels[levels.size() - 1]->getScaledImage(), scaledImage, scaledImageSize, 0, 0, cv::INTER_LINEAR);
		}
		initScale(scaledImage);
		levels.push_back(new PyramidLevel(factor, scaledImage));
	}
}

shared_ptr<FeatureVector> ImagePyramidFeatureExtractor::extract(int x, int y, int size) {
	PyramidLevel* level = getLevel(size);
	if (level == NULL)
		return shared_ptr<FeatureVector>();
	int scaledX = level->getScaled(x);
	int scaledY = level->getScaled(y);
	Patch patch(scaledX, scaledY);
	unordered_set<Patch, Patch::hash>::iterator pit = level->getPatches().find(patch);
	if (pit == level->getPatches().end()) { // patch does not exist, feature vector has to be created
		const Mat& image = level->getScaledImage();
		int patchBeginX = scaledX - featureSize.width / 2; // inclusive
		int patchBeginY = scaledY - featureSize.height / 2; // inclusive
		int patchEndX = patchBeginX + featureSize.width; // exclusive
		int patchEndY = patchBeginY + featureSize.height; // exclusive
		if (patchBeginX < 0 || patchEndX > image.cols
				|| patchBeginY < 0 || patchEndY > image.rows)
			return shared_ptr<FeatureVector>();
		shared_ptr<FeatureVector> featureVector = extract(
				Mat(image, Rect(patchBeginX, patchBeginY, featureSize.width, featureSize.height)));
		patch.setFeatureVector(featureVector);
		level->getPatches().insert(patch);
		return featureVector;
	} else { // patch and feature vector exist already
		return pit->getFeatureVector();
	}
	// TODO the following code might work if std::unordered_set does not return a const_iterator
	/*pair<unordered_set<Patch, Patch::hash>::iterator, bool> insertion = level->getPatches().insert(patch);
	if (insertion.second) { // patch was inserted (was not existing before)
		const Mat& image = level->getScaledImage();
		int patchBeginX = scaledX - featureSize.width / 2; // inclusive
		int patchBeginY = scaledY - featureSize.height / 2; // inclusive
		int patchEndX = patchBeginX + featureSize.width; // exclusive
		int patchEndY = patchBeginY + featureSize.height; // exclusive
		if (patchBeginX < 0 || patchEndX > image.cols
				|| patchBeginY < 0 || patchEndY > image.rows)
			return shared_ptr<FeatureVector>();
		Mat patch(image, Rect(patchBeginX, patchBeginY, patchEndX, patchEndY));
		shared_ptr<FeatureVector> featureVector = extract(patch);
		insertion.first->setFeatureVector(featureVector);
		return featureVector;
	} else {
		insertion.first.
		return insertion.first->getFeatureVector();
	}*/
}

void ImagePyramidFeatureExtractor::clearLevels() {
	for (vector<PyramidLevel*>::iterator it = levels.begin(); it != levels.end(); ++it)
		delete (*it);
	levels.clear();
}

PyramidLevel* ImagePyramidFeatureExtractor::getLevel(int size) {
	double factor = (double)size / (double)featureSize.width;
	double power = log(factor) / log(scaleFactor);
	int index = cvRound(power) - firstLevel;
	if (index < 0 || index >= (int)levels.size())
		return NULL;
	return levels[index];
}

} /* namespace tracking */

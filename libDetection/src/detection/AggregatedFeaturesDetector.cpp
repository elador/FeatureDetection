/*
 * AggregatedFeaturesDetector.cpp
 *
 *  Created on: 22.10.2015
 *      Author: poschmann
 */

#include "detection/AggregatedFeaturesDetector.hpp"
#include "classification/LinearKernel.hpp"
#include "imageprocessing/ConvolutionFilter.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/Patch.hpp"
#include <stdexcept>

using classification::LinearKernel;
using classification::SvmClassifier;
using cv::Point;
using cv::Rect;
using cv::Size;
using cv::Mat;
using imageprocessing::ConvolutionFilter;
using imageprocessing::GrayscaleFilter;
using imageprocessing::ImageFilter;
using imageprocessing::ImagePyramid;
using imageprocessing::ImagePyramidLayer;
using imageprocessing::Patch;
using imageprocessing::VersionedImage;
using imageprocessing::extraction::AggregatedFeaturesExtractor;
using std::make_shared;
using std::pair;
using std::shared_ptr;
using std::vector;

namespace detection {

AggregatedFeaturesDetector::AggregatedFeaturesDetector(shared_ptr<ImageFilter> imageFilter, shared_ptr<ImageFilter> layerFilter,
		int cellSize, Size windowSize, int octaveLayerCount, shared_ptr<SvmClassifier> svm, shared_ptr<NonMaximumSuppression> nms) :
				AggregatedFeaturesDetector(make_shared<AggregatedFeaturesExtractor>(
						imageFilter, layerFilter, windowSize, cellSize, octaveLayerCount), svm, nms) {}

AggregatedFeaturesDetector::AggregatedFeaturesDetector(shared_ptr<ImageFilter> filter, int cellSize, Size windowSize,
		int octaveLayerCount, shared_ptr<SvmClassifier> svm, shared_ptr<NonMaximumSuppression> nms) :
				AggregatedFeaturesDetector(make_shared<AggregatedFeaturesExtractor>(
						filter, windowSize, cellSize, octaveLayerCount), svm, nms) {}

AggregatedFeaturesDetector::AggregatedFeaturesDetector(shared_ptr<AggregatedFeaturesExtractor> featureExtractor,
		shared_ptr<SvmClassifier> svm, shared_ptr<NonMaximumSuppression> nms) :
				featureExtractor(featureExtractor),
				nonMaximumSuppression(nms),
				kernelSize(svm->getSupportVectors()[0].size()),
				scoreThreshold(0) {
	if (!dynamic_cast<LinearKernel*>(svm->getKernel().get()))
		throw std::invalid_argument("AggregatedFeaturesDetector: the SVM must use a LinearKernel");
	shared_ptr<ConvolutionFilter> convolutionFilter = make_shared<ConvolutionFilter>(CV_32F);
	convolutionFilter->setKernel(svm->getSupportVectors()[0]);
	convolutionFilter->setAnchor(Point(0, 0));
	convolutionFilter->setDelta(-svm->getBias() - svm->getThreshold());
	scorePyramid = make_shared<ImagePyramid>(featureExtractor->getFeaturePyramid());
	scorePyramid->addLayerFilter(convolutionFilter);
}

vector<Rect> AggregatedFeaturesDetector::detect(shared_ptr<VersionedImage> image) {
	update(image);
	return detect();
}

void AggregatedFeaturesDetector::update(shared_ptr<VersionedImage> image) {
	featureExtractor->update(image);
	scorePyramid->update(image);
}

vector<Rect> AggregatedFeaturesDetector::detect() {
	vector<Detection> candidates = getPositiveWindows();
	vector<Detection> detections = nonMaximumSuppression->eliminateRedundantDetections(candidates);
	return extractBoundingBoxes(detections);
}

std::vector<Detection> AggregatedFeaturesDetector::getPositiveWindows() {
	vector<Detection> positiveBounds;
	for (const shared_ptr<ImagePyramidLayer>& layer : scorePyramid->getLayers()) {
		const Mat& scoreMap = layer->getScaledImage();
		int validHeight = scoreMap.rows - kernelSize.height + 1;
		int validWidth = scoreMap.cols - kernelSize.width + 1;
		for (int y = 0; y < validHeight; ++y) {
			for (int x = 0; x < validWidth; ++x) {
				float score = scoreMap.at<float>(y, x);
				if (score > scoreThreshold) {
					Rect boundsInLayer = Rect(Point(x, y), kernelSize);
					Rect boundsInImage = featureExtractor->computeBoundsInImagePixels(boundsInLayer, *layer);
					positiveBounds.push_back({score, boundsInImage});
				}
			}
		}
	}
	return positiveBounds;
}

vector<Rect> AggregatedFeaturesDetector::extractBoundingBoxes(vector<Detection> detections) {
	vector<Rect> boundingBoxes;
	boundingBoxes.reserve(detections.size());
	for (const Detection& detection : detections)
		boundingBoxes.push_back(detection.bounds);
	return boundingBoxes;
}

float AggregatedFeaturesDetector::getScoreThreshold() const {
	return scoreThreshold;
}

void AggregatedFeaturesDetector::setScoreThreshold(float threshold) {
	scoreThreshold = threshold;
}

} /* namespace detection */

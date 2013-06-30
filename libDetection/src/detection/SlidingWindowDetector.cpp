/*
 * SlidingWindowDetector.cpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#include "detection/SlidingWindowDetector.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/PyramidFeatureExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "classification/ProbabilisticClassifier.hpp"
#include "detection/ClassifiedPatch.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using imageprocessing::PyramidFeatureExtractor;
using std::make_shared;

namespace detection {

SlidingWindowDetector::SlidingWindowDetector(shared_ptr<ProbabilisticClassifier> classifier, shared_ptr<PyramidFeatureExtractor> featureExtractor, int stepSizeX, int stepSizeY) :
		classifier(classifier), featureExtractor(featureExtractor), stepSizeX(stepSizeX), stepSizeY(stepSizeY)
{

}

vector<shared_ptr<ClassifiedPatch>> SlidingWindowDetector::detect(const Mat& image)
{
	featureExtractor->update(image);
	// hacked-in to show the patch-sizes:
	vector<cv::Size> patchSizes = featureExtractor->getPatchSizes();
	Mat tmp = image.clone();
	for (const auto& p : patchSizes) {
		cv::rectangle(tmp, cv::Rect(0, 0, p.width, p.height), cv::Scalar(0.0f, 0.0f, 0.0f));
	}
	return detect();
}

vector<shared_ptr<ClassifiedPatch>> SlidingWindowDetector::detect(shared_ptr<VersionedImage> image)
{
	featureExtractor->update(image);
	return detect();
}

vector<shared_ptr<ClassifiedPatch>> SlidingWindowDetector::detect() const
{
	vector<shared_ptr<ClassifiedPatch>> classifiedPatches;
	vector<shared_ptr<Patch>> pyramidPatches = featureExtractor->extract(stepSizeX, stepSizeY);

	for (unsigned int i = 0; i < pyramidPatches.size(); ++i) {
		pair<bool, double> res = classifier->classify(pyramidPatches[i]->getData());
		if(res.first==true)
			classifiedPatches.push_back(make_shared<ClassifiedPatch>(pyramidPatches[i], res));
	}

	return classifiedPatches;
}

vector<Mat> SlidingWindowDetector::calculateProbabilityMaps(const Mat& image)
{
	//for(auto currentPyramidLayer : featureExtractor->getPyramid()->getLayers()) {
		//Mat tmp = currentPyramidLayer->getScaledImage();
	//}
	vector<Mat> tmp;
	return tmp;
}

} /* namespace detection */

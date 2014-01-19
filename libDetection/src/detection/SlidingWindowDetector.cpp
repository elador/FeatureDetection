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
#include "imagelogging/ImageLoggerFactory.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using imageprocessing::PyramidFeatureExtractor;
using imagelogging::ImageLogger;
using imagelogging::ImageLoggerFactory;
using std::make_shared;

namespace detection {

SlidingWindowDetector::SlidingWindowDetector(shared_ptr<ProbabilisticClassifier> classifier, shared_ptr<PyramidFeatureExtractor> featureExtractor, int stepSizeX, int stepSizeY) :
		classifier(classifier), featureExtractor(featureExtractor), stepSizeX(stepSizeX), stepSizeY(stepSizeY)
{

}

void drawRects(Mat image, vector<cv::Size> patchSizes)
{
	for (const auto& p : patchSizes) {
		cv::rectangle(image, cv::Rect(0, 0, p.width, p.height), cv::Scalar(0.0, 0.0, 0.0));
	}
}

vector<shared_ptr<ClassifiedPatch>> SlidingWindowDetector::detect(const Mat& image)
{
	featureExtractor->update(image);
	// Log the scales on which we are detecting:
	vector<cv::Size> patchSizes = featureExtractor->getPatchSizes();
	ImageLogger imageLogger = ImageLoggers->getLogger("detection");
	Mat scalesImage = image.clone();
	imageLogger.intermediate(scalesImage, bind(drawRects, scalesImage, patchSizes), "00scales"); // Note: Another option: We could "send" the logger the scale-info here. It could then draw it into the output image, depending on a config-flag if it should draw it. Optimally: Only get & send the scale-info if loglevel>xyz... i.e. the info is actually outputted. But that kind of is another concept than the current loglevels, e.g. it is a separate switch...

	return detect();
}


vector<shared_ptr<ClassifiedPatch>> SlidingWindowDetector::detect(const Mat& image, const Rect& roi)
{
	/* We have a few different use-cases:
		- operate on the whole image
		- operate on a Rect, one ROI. Can call featureExtractor->extract(stepSizeX, stepSizeY, Rect);
		- operate on a ROI that is not a Rect but a Mat mask
		- instead of a static 0/1 mask, a dynamic mask that's connected with the FD probability. That goes a little bit into the direction of condensation.
	*/
	featureExtractor->update(image);

	// Log the scales on which we are detecting: (Note: 1) code-duplication, see above. 2) This could even go into the extactor?)
	vector<cv::Size> patchSizes = featureExtractor->getPatchSizes();
	ImageLogger imageLogger = ImageLoggers->getLogger("detection");
	Mat scalesImage = image.clone();
	imageLogger.intermediate(scalesImage, bind(drawRects, scalesImage, patchSizes), "00scales"); // Note: Another option: We could "send" the logger the scale-info here. It could then draw it into the output image, depending on a config-flag if it should draw it. Optimally: Only get & send the scale-info if loglevel>xyz... i.e. the info is actually outputted. But that kind of is another concept than the current loglevels, e.g. it is a separate switch...

	vector<shared_ptr<ClassifiedPatch>> classifiedPatches;
	vector<shared_ptr<Patch>> pyramidPatches = featureExtractor->extract(stepSizeX, stepSizeY, roi);

	for (unsigned int i = 0; i < pyramidPatches.size(); ++i) {
		pair<bool, double> res = classifier->getProbability(pyramidPatches[i]->getData());
		if(res.first==true)
			classifiedPatches.push_back(make_shared<ClassifiedPatch>(pyramidPatches[i], res));
	}
	return classifiedPatches;
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
		pair<bool, double> res = classifier->getProbability(pyramidPatches[i]->getData());
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

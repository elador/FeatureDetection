/*
 * FiveStageSlidingWindowDetector.cpp
 *
 *  Created on: 10.05.2013
 *      Author: Patrik Huber
 */

#include "detection/FiveStageSlidingWindowDetector.hpp"
#include "detection/ClassifiedPatch.hpp"
#include "logging/LoggerFactory.hpp"
#include "imagelogging/ImageLoggerFactory.hpp"

#include "opencv2/highgui/highgui.hpp"
//#include "boost/iterator/indirect_iterator.hpp"
#include <algorithm>
#include <functional>

using logging::Logger;
using logging::LoggerFactory;
using imagelogging::ImageLogger;
using imagelogging::ImageLoggerFactory;
//using boost::make_indirect_iterator;
using std::sort;
using std::greater;

namespace detection {

FiveStageSlidingWindowDetector::FiveStageSlidingWindowDetector(shared_ptr<SlidingWindowDetector> slidingWindowDetector, shared_ptr<OverlapElimination> overlapElimination, shared_ptr<ProbabilisticClassifier> strongClassifier) :
		slidingWindowDetector(slidingWindowDetector), overlapElimination(overlapElimination), strongClassifier(strongClassifier)
{

}

/*
 * TODO: move the image showing stuff to an image logger. Can't really all go into an app, some debug images we just want in the lib.
 * Or can it? Make a detectFirstStage(...), ...second..., ... and draw in the app.
 * But an image logger would make things so much more natural. Prefer this.
 */

#include <functional>
using std::function;
using std::bind;

void drawBoxes(Mat image, vector<shared_ptr<ClassifiedPatch>> patches)
{
	for(const auto& cpatch : patches) {
		shared_ptr<Patch> patch = cpatch->getPatch();
		cv::rectangle(image, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((cpatch->getProbability())/1.0)   ));
	}
}


vector<shared_ptr<ClassifiedPatch>> FiveStageSlidingWindowDetector::detect(const Mat& image)
{
	vector<shared_ptr<ClassifiedPatch>> classifiedPatches;

	Logger logger = Loggers->getLogger("detection");
	ImageLogger imageLogger = ImageLoggers->getLogger("detection");

	// Log the original image?
	
	// WVM stage
	classifiedPatches = slidingWindowDetector->detect(image);
	Mat imgWvm = image.clone();
	imageLogger.intermediate(imgWvm, bind(drawBoxes, imgWvm, classifiedPatches), "01wvm"); // The detector could send a MESSAGE to the logger here, and in the image-logger config we could configure them (which one to output, what filename). E.g. here the message could be something like FIVESTAGE...STAGE1... (is it always a wvm?) and also the name of the detector or feature, but maybe that's not available here. (=>we could set it in the imagelogger externally before the call)

	// WVM OE stage
	classifiedPatches = overlapElimination->eliminate(classifiedPatches);
	Mat imgWvmOe = image.clone();
	imageLogger.intermediate(imgWvmOe, bind(drawBoxes, imgWvmOe, classifiedPatches), "02oe");

	// SVM stage
	vector<shared_ptr<ClassifiedPatch>> svmPatches;
	for(const auto &patch : classifiedPatches) {
		svmPatches.push_back(make_shared<ClassifiedPatch>(patch->getPatch(), strongClassifier->classify(patch->getPatch()->getData())));
	}
	Mat imgSvmAll = image.clone();
	imageLogger.intermediate(imgSvmAll, bind(drawBoxes, imgSvmAll, svmPatches), "03svmall");

	// Only the positive SVM patches
	vector<shared_ptr<ClassifiedPatch>> svmPatchesPositive;
	for(const auto& classifiedPatch : svmPatches) {
		if(classifiedPatch->isPositive()) {
			svmPatchesPositive.push_back(classifiedPatch);
		}
	}
	Mat imgSvmPos = image.clone();
	imageLogger.intermediate(imgSvmPos, bind(drawBoxes, imgSvmPos, svmPatchesPositive), "03svmpos");

	// The highest one of all the positively classified SVM patches
	//sort(make_indirect_iterator(svmPatches.begin()), make_indirect_iterator(svmPatches.end()), greater<ClassifiedPatch>());
	// TODO: Move to a function NMS or similar...? Similar than OE? Is there a family of functions that work on a vector of patches or classifiedPatches?
	sort(svmPatchesPositive.begin(), svmPatchesPositive.end(), [](shared_ptr<ClassifiedPatch> a, shared_ptr<ClassifiedPatch> b) { return *a > *b; });
	vector<shared_ptr<ClassifiedPatch>> svmPatchesMaxPositive;
	if(svmPatchesPositive.size()>0) {	
		svmPatchesMaxPositive.push_back(svmPatchesPositive[0]);
	}
	Mat imgSvmMaxPos = image.clone();
	imageLogger.final(imgSvmMaxPos, bind(drawBoxes, imgSvmMaxPos, svmPatchesMaxPositive), "04svmmaxpos");
	
	return svmPatchesPositive;
	//return svmPatches;
}

vector<shared_ptr<ClassifiedPatch>> FiveStageSlidingWindowDetector::detect(shared_ptr<VersionedImage> image)
{
	vector<shared_ptr<ClassifiedPatch>> classifiedPatches;
	Loggers->getLogger("detection").error("FiveStageSlidingWindowDetector::detect: This function is not yet implemented for a VersionedImage.");
	return classifiedPatches;
}

vector<shared_ptr<ClassifiedPatch>> FiveStageSlidingWindowDetector::detect(const Mat& image, const Rect& roi)
{
	vector<shared_ptr<ClassifiedPatch>> classifiedPatches;

	Logger logger = Loggers->getLogger("detection");
	ImageLogger imageLogger = ImageLoggers->getLogger("detection");

	// Log the original image?

	// WVM stage
	classifiedPatches = slidingWindowDetector->detect(image, roi); // TODO: All the code in this function, except the 'mask' here, is an exact copy of the function above. Improve that!
	Mat imgWvm = image.clone();
	imageLogger.intermediate(imgWvm, bind(drawBoxes, imgWvm, classifiedPatches), "01wvm"); // The detector could send a MESSAGE to the logger here, and in the image-logger config we could configure them (which one to output, what filename). E.g. here the message could be something like FIVESTAGE...STAGE1... (is it always a wvm?) and also the name of the detector or feature, but maybe that's not available here. (=>we could set it in the imagelogger externally before the call)

	// WVM OE stage
	classifiedPatches = overlapElimination->eliminate(classifiedPatches);
	Mat imgWvmOe = image.clone();
	imageLogger.intermediate(imgWvmOe, bind(drawBoxes, imgWvmOe, classifiedPatches), "02oe");

	// SVM stage
	vector<shared_ptr<ClassifiedPatch>> svmPatches;
	for(const auto &patch : classifiedPatches) {
		svmPatches.push_back(make_shared<ClassifiedPatch>(patch->getPatch(), strongClassifier->classify(patch->getPatch()->getData())));
	}
	Mat imgSvmAll = image.clone();
	imageLogger.intermediate(imgSvmAll, bind(drawBoxes, imgSvmAll, svmPatches), "03svmall");

	// Only the positive SVM patches
	vector<shared_ptr<ClassifiedPatch>> svmPatchesPositive;
	for(const auto& classifiedPatch : svmPatches) {
		if(classifiedPatch->isPositive()) {
			svmPatchesPositive.push_back(classifiedPatch);
		}
	}
	Mat imgSvmPos = image.clone();
	imageLogger.intermediate(imgSvmPos, bind(drawBoxes, imgSvmPos, svmPatchesPositive), "03svmpos");

	// The highest one of all the positively classified SVM patches
	//sort(make_indirect_iterator(svmPatches.begin()), make_indirect_iterator(svmPatches.end()), greater<ClassifiedPatch>());
	// TODO: Move to a function NMS or similar...? Similar than OE? Is there a family of functions that work on a vector of patches or classifiedPatches?
	sort(svmPatchesPositive.begin(), svmPatchesPositive.end(), [](shared_ptr<ClassifiedPatch> a, shared_ptr<ClassifiedPatch> b) { return *a > *b; });
	vector<shared_ptr<ClassifiedPatch>> svmPatchesMaxPositive;
	if(svmPatchesPositive.size()>0) {	
		svmPatchesMaxPositive.push_back(svmPatchesPositive[0]);
	}
	Mat imgSvmMaxPos = image.clone();
	imageLogger.final(imgSvmMaxPos, bind(drawBoxes, imgSvmMaxPos, svmPatchesMaxPositive), "04svmmaxpos");

	return svmPatchesPositive;
	//return svmPatches;
}



vector<Mat> FiveStageSlidingWindowDetector::calculateProbabilityMaps(const Mat& image)
{
	//for(auto currentPyramidLayer : featureExtractor->getPyramid()->getLayers()) {
		//Mat tmp = currentPyramidLayer->getScaledImage();
	//}
	vector<Mat> tmp;
	return tmp;
}

const shared_ptr<PyramidFeatureExtractor> FiveStageSlidingWindowDetector::getPyramidFeatureExtractor() const
{
	return slidingWindowDetector->getPyramidFeatureExtractor();
}

} /* namespace detection */

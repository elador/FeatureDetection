/*
 * FiveStageSlidingWindowDetector.cpp
 *
 *  Created on: 10.05.2013
 *      Author: Patrik Huber
 */

#include "detection/FiveStageSlidingWindowDetector.hpp"
#include "detection/ClassifiedPatch.hpp"
#include "logging/LoggerFactory.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "boost/iterator/indirect_iterator.hpp"
#include <algorithm>
#include <functional>

using logging::Logger;
using logging::LoggerFactory;
using boost::make_indirect_iterator;
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

vector<shared_ptr<ClassifiedPatch>> FiveStageSlidingWindowDetector::detect(const Mat& image)
{
	vector<shared_ptr<ClassifiedPatch>> classifiedPatches;

	Logger logger = Loggers->getLogger("detection");

	classifiedPatches = slidingWindowDetector->detect(image);

	Mat imgWvm = image.clone();
	for(auto pit = classifiedPatches.begin(); pit != classifiedPatches.end(); pit++) {
		shared_ptr<ClassifiedPatch> classifiedPatch = *pit;
		shared_ptr<Patch> patch = classifiedPatch->getPatch();
		cv::rectangle(imgWvm, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
	}
	cv::namedWindow("wvm", CV_WINDOW_AUTOSIZE); cv::imshow("wvm", imgWvm);
	cvMoveWindow("wvm", 0, 0);

	classifiedPatches = overlapElimination->eliminate(classifiedPatches);
	Mat imgWvmOe = image.clone();
	for(auto pit = classifiedPatches.begin(); pit != classifiedPatches.end(); pit++) {
		shared_ptr<ClassifiedPatch> classifiedPatch = *pit;
		shared_ptr<Patch> patch = classifiedPatch->getPatch();
		cv::rectangle(imgWvmOe, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
	}
	cv::namedWindow("wvmoe", CV_WINDOW_AUTOSIZE); cv::imshow("wvmoe", imgWvmOe);
	cvMoveWindow("wvmoe", 550, 0);

	vector<shared_ptr<ClassifiedPatch>> svmPatches;
	for(auto &patch : classifiedPatches) {
		svmPatches.push_back(make_shared<ClassifiedPatch>(patch->getPatch(), strongClassifier->classify(patch->getPatch()->getData())));
	}

	vector<shared_ptr<ClassifiedPatch>> svmPatchesPositive;
	Mat imgSvm = image.clone();
	for(auto classifiedPatch : svmPatches) {
		shared_ptr<Patch> patch = classifiedPatch->getPatch();
		if(classifiedPatch->isPositive()) {
			cv::rectangle(imgSvm, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
			svmPatchesPositive.push_back(classifiedPatch);
		}
	}
	cv::namedWindow("svm", CV_WINDOW_AUTOSIZE); cv::imshow("svm", imgSvm);
	cvMoveWindow("svm", 0, 500);

	//sort(make_indirect_iterator(svmPatches.begin()), make_indirect_iterator(svmPatches.end()), greater<ClassifiedPatch>());
	typedef shared_ptr<ClassifiedPatch> iter;
	sort(svmPatches.begin(), svmPatches.end(), [](iter a, iter b) { return *a > *b; });

	Mat imgSvmEnd = image.clone();
	cv::rectangle(imgSvmEnd, cv::Point(svmPatches[0]->getPatch()->getX() - svmPatches[0]->getPatch()->getWidth()/2, svmPatches[0]->getPatch()->getY() - svmPatches[0]->getPatch()->getHeight()/2), cv::Point(svmPatches[0]->getPatch()->getX() + svmPatches[0]->getPatch()->getWidth()/2, svmPatches[0]->getPatch()->getY() + svmPatches[0]->getPatch()->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((svmPatches[0]->getProbability())/1.0)   ));
	cv::namedWindow("svmMostProbable", CV_WINDOW_AUTOSIZE); cv::imshow("svmMostProbable", imgSvmEnd);
	cvMoveWindow("svmMostProbable", 550, 500);

	return svmPatchesPositive;
	//return svmPatches;
}

vector<shared_ptr<ClassifiedPatch>> FiveStageSlidingWindowDetector::detect(shared_ptr<VersionedImage> image)
{
	vector<shared_ptr<ClassifiedPatch>> classifiedPatches;
	Loggers->getLogger("detection").error("FiveStageSlidingWindowDetector::detect: This function is not yet implemented for a VersionedImage.");
	return classifiedPatches;
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

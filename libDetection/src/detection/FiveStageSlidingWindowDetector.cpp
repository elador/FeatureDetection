/*
 * FiveStageSlidingWindowDetector.cpp
 *
 *  Created on: 10.05.2013
 *      Author: Patrik Huber
 */

#include "detection/FiveStageSlidingWindowDetector.hpp"
#include "logging/LoggerFactory.hpp"

using logging::Logger;
using logging::LoggerFactory;

namespace detection {

FiveStageSlidingWindowDetector::FiveStageSlidingWindowDetector(shared_ptr<SlidingWindowDetector> slidingWindowDetector, shared_ptr<OverlapElimination> overlapElimination, shared_ptr<ProbabilisticClassifier> probabilisticClassifier) :
		slidingWindowDetector(slidingWindowDetector), overlapElimination(overlapElimination), probabilisticClassifier(probabilisticClassifier)
{

}

/*
 * TODO: move the image showing stuff to an image logger. Can't really all go into an app, some debug images we just want in the lib.
 * Or can it? Make a detectFirstStage(...), ...second..., ... and draw in the app.
 */

vector<shared_ptr<ClassifiedPatch>> FiveStageSlidingWindowDetector::detect(const Mat& image)
{
	vector<shared_ptr<ClassifiedPatch>> classifiedPatches;

	Logger logger = Loggers->getLogger("detection");

	classifiedPatches = slidingWindowDetector->detect(image);
	/*
	Mat imgWvm = image.clone();
	for(auto pit = resultingPatches.begin(); pit != resultingPatches.end(); pit++) {
		shared_ptr<ClassifiedPatch> classifiedPatch = *pit;
		shared_ptr<Patch> patch = classifiedPatch->getPatch();
		cv::rectangle(imgWvm, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
	}
	cv::namedWindow("wvm", CV_WINDOW_AUTOSIZE); cv::imshow("wvm", imgWvm);
	cvMoveWindow("wvm", 0, 0);

	resultingPatches = oe->eliminate(resultingPatches);
	Mat imgWvmOe = image.clone();
	for(auto pit = resultingPatches.begin(); pit != resultingPatches.end(); pit++) {
		shared_ptr<ClassifiedPatch> classifiedPatch = *pit;
		shared_ptr<Patch> patch = classifiedPatch->getPatch();
		cv::rectangle(imgWvmOe, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
	}
	cv::namedWindow("wvmoe", CV_WINDOW_AUTOSIZE); cv::imshow("wvmoe", imgWvmOe);
	cvMoveWindow("wvmoe", 550, 0);

	vector<shared_ptr<ClassifiedPatch>> svmPatches;
	for(auto &patch : resultingPatches) {
		svmPatches.push_back(make_shared<ClassifiedPatch>(patch->getPatch(), psvm->classify(patch->getPatch()->getData())));
	}

	vector<shared_ptr<ClassifiedPatch>> svmPatchesPositive;
	Mat imgSvm = image.clone();
	for(auto pit = svmPatches.begin(); pit != svmPatches.end(); pit++) {
		shared_ptr<ClassifiedPatch> classifiedPatch = *pit;
		shared_ptr<Patch> patch = classifiedPatch->getPatch();
		if(classifiedPatch->isPositive()) {
			cv::rectangle(imgSvm, cv::Point(patch->getX() - patch->getWidth()/2, patch->getY() - patch->getHeight()/2), cv::Point(patch->getX() + patch->getWidth()/2, patch->getY() + patch->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((classifiedPatch->getProbability())/1.0)   ));
			svmPatchesPositive.push_back(classifiedPatch);
		}
	}
	cv::namedWindow("svm", CV_WINDOW_AUTOSIZE); cv::imshow("svm", imgSvm);
	cvMoveWindow("svm", 0, 500);

	sort(make_indirect_iterator(svmPatches.begin()), make_indirect_iterator(svmPatches.end()), greater<ClassifiedPatch>());

	Mat imgSvmEnd = image.clone();
	cv::rectangle(imgSvmEnd, cv::Point(svmPatches[0]->getPatch()->getX() - svmPatches[0]->getPatch()->getWidth()/2, svmPatches[0]->getPatch()->getY() - svmPatches[0]->getPatch()->getHeight()/2), cv::Point(svmPatches[0]->getPatch()->getX() + svmPatches[0]->getPatch()->getWidth()/2, svmPatches[0]->getPatch()->getY() + svmPatches[0]->getPatch()->getHeight()/2), cv::Scalar(0, 0, (float)255 * ((svmPatches[0]->getProbability())/1.0)   ));
	cv::namedWindow("svmEnd", CV_WINDOW_AUTOSIZE); cv::imshow("svmEnd", imgSvmEnd);
	cvMoveWindow("svmEnd", 550, 500);
	*/
	return classifiedPatches;
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

} /* namespace detection */

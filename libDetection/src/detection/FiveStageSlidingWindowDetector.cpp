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

#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <algorithm>
#include <functional>

using logging::Logger;
using logging::LoggerFactory;
using imagelogging::ImageLogger;
using imagelogging::ImageLoggerFactory;
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


/* 
 *  Software License Agreement (BSD License)
 *
 *  Copyright (c) 2012, Willow Garage, Inc.
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of Willow Garage, Inc. nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 *  File:    nms.cpp
 *  Author:  Hilton Bristow
 *  Created: Jul 19, 2012
 */

//#include <stdio.h>
//#include <iostream>
//#include <limits>
/*! @brief suppress non-maximal values
 *
 * nonMaximaSuppression produces a mask (dst) such that every non-zero
 * value of the mask corresponds to a local maxima of src. The criteria
 * for local maxima is as follows:
 *
 * 	For every possible (sz x sz) region within src, an element is a
 * 	local maxima of src iff it is strictly greater than all other elements
 * 	of windows which intersect the given element
 *
 * Intuitively, this means that all maxima must be at least sz+1 pixels
 * apart, though the spacing may be greater
 *
 * A gradient image or a constant image has no local maxima by the definition
 * given above
 *
 * The method is derived from the following paper:
 * A. Neubeck and L. Van Gool. "Efficient Non-Maximum Suppression," ICPR 2006
 *
 * Example:
 * \code
 * 	// create a random test image
 * 	Mat random(Size(2000,2000), DataType<float>::type);
 * 	randn(random, 1, 1);
 *
 * 	// only look for local maxima above the value of 1
 * 	Mat mask = (random > 1);
 *
 * 	// find the local maxima with a window of 50
 * 	Mat maxima;
 * 	nonMaximaSuppression(random, 50, maxima, mask);
 *
 * 	// optionally set all non-maxima to zero
 * 	random.setTo(0, maxima == 0);
 * \endcode
 *
 * @param src the input image/matrix, of any valid cv type
 * @param sz the size of the window
 * @param dst the mask of type CV_8U, where non-zero elements correspond to
 * local maxima of the src
 * @param mask an input mask to skip particular elements
 */
using cv::Mat_;
using cv::Range;
using cv::Point;
using cv::Size;
using std::min;
using std::max;
//#include <utility>
void nonMaximaSuppression(const Mat& src, const int sz, Mat& dst, const Mat mask) {

	// initialise the block mask and destination
	const int M = src.rows;
	const int N = src.cols;
	const bool masked = !mask.empty();
	Mat block = 255*Mat_<uint8_t>::ones(Size(2*sz+1,2*sz+1));
	dst = Mat_<uint8_t>::zeros(src.size());

	// iterate over image blocks
	for (int m = 0; m < M; m+=sz+1) {
		for (int n = 0; n < N; n+=sz+1) {
			Point  ijmax;
			double vcmax, vnmax;

			// get the maximal candidate within the block
			Range ic(m, min(m+sz+1,M));
			Range jc(n, min(n+sz+1,N));
			minMaxLoc(src(ic,jc), NULL, &vcmax, NULL, &ijmax, masked ? mask(ic,jc) : cv::noArray());
			Point cc = ijmax + Point(jc.start,ic.start);

			// search the neighbours centered around the candidate for the true maxima
			Range in(max(cc.y-sz,0), min(cc.y+sz+1,M));
			Range jn(max(cc.x-sz,0), min(cc.x+sz+1,N));

			// mask out the block whose maxima we already know
			Mat_<uint8_t> blockmask;
			block(Range(0,in.size()), Range(0,jn.size())).copyTo(blockmask);
			Range iis(ic.start-in.start, min(ic.start-in.start+sz+1, in.size()));
			Range jis(jc.start-jn.start, min(jc.start-jn.start+sz+1, jn.size()));
			blockmask(iis, jis) = Mat_<uint8_t>::zeros(Size(jis.size(),iis.size()));

			minMaxLoc(src(in,jn), NULL, &vnmax, NULL, &ijmax, masked ? mask(in,jn).mul(blockmask) : blockmask);
			Point cn = ijmax + Point(jn.start, in.start);

			// if the block centre is also the neighbour centre, then it's a local maxima
			if (vcmax > vnmax) {
				dst.at<uint8_t>(cc.y, cc.x) = 255;
			}
		}
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

	// NEW NMS
/*	Mat probabilityMap = Mat::zeros(image.rows, image.cols, CV_32FC1);
	for (const auto& p : classifiedPatches) {
		int px = p->getPatch()->getX();
		int py = p->getPatch()->getY();
		if (probabilityMap.at<float>(py, px) < p->getProbability()) {
			probabilityMap.at<float>(py, px) = p->getProbability();
		}
		
	}
	Mat mask = (probabilityMap > 0.3f);
	Mat maxima;
	nonMaximaSuppression(probabilityMap, 35, maxima, mask); // Mat() for empty mask
	vector<cv::Point2i> maximaCoords;
	cv::findNonZero(maxima, maximaCoords);
	vector<shared_ptr<ClassifiedPatch>> classifiedPatchesNewNMS;
	sort(begin(classifiedPatches), end(classifiedPatches), [](shared_ptr<ClassifiedPatch> a, shared_ptr<ClassifiedPatch> b) { return *a > *b; });
	for (const auto& p : maximaCoords) {
		const auto& foundPatchIt = std::find_if(begin(classifiedPatches), end(classifiedPatches), [p](shared_ptr<ClassifiedPatch> a) { 
			return (a->getPatch()->getX() == p.x && a->getPatch()->getY() == p.y); }); // should never fail but we should handle an error anyway
		classifiedPatchesNewNMS.push_back(*foundPatchIt);
	}
	classifiedPatches = classifiedPatchesNewNMS; */

	

/*	Mat probabilityMapSmoothed2;
	Mat probabilityMapSmoothed5;
	cv::GaussianBlur(probabilityMap, probabilityMapSmoothed2, Size(5, 5), 0, 0);
	cv::bilateralFilter(probabilityMap, probabilityMapSmoothed5, 9, 10.0, 10.0);
	//cv::adaptiveBilateralFilter(probabilityMap, probabilityMapSmoothed5, Size(5, 5), 10.0);
	Mat maxima2;
	nonMaximaSuppression(probabilityMapSmoothed2, 35, maxima2, Mat());
	Mat maxima5;
	nonMaximaSuppression(probabilityMapSmoothed5, 35, maxima5, Mat()); */


	// create a random test image
	//Mat random(Size(512, 512), cv::DataType<float>::type);
	//Mat random(Size(512, 512), CV_32FC1);
	//randn(random, 1, 1);
	// only look for local maxima above the value of 1
	//Mat mask = (random > 1);
	// find the local maxima with a window of 50
	//Mat maxima;
	//nonMaximaSuppression(random, 50, maxima, mask);
	// optionally set all non-maxima to zero
	//random.setTo(0, maxima == 0);
	// END NEW NMS
	

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

	// new NMS
	Mat probabilityMap = Mat::zeros(image.rows, image.cols, CV_32FC1);
	for (const auto& p : svmPatchesPositive) {
		int px = p->getPatch()->getX();
		int py = p->getPatch()->getY();
		if (probabilityMap.at<float>(py, px) < p->getProbability()) {
			probabilityMap.at<float>(py, px) = p->getProbability();
		}
		
	}
	Mat mask = (probabilityMap > 0.3f);
	Mat maxima;
	nonMaximaSuppression(probabilityMap, 35, maxima, mask); // Mat() for empty mask
	vector<cv::Point2i> maximaCoords;
	int numNonZero = cv::countNonZero(maxima); // findNonZero fails when countNonZero is 0...
	if (numNonZero == 0) {
		nonMaximaSuppression(probabilityMap, 35, maxima, Mat()); // Mat() for empty mask
		numNonZero = cv::countNonZero(maxima);
		if (numNonZero == 0) {
			return svmPatchesPositive; // Should be empty.
		}
	}
	cv::findNonZero(maxima, maximaCoords);
	vector<shared_ptr<ClassifiedPatch>> classifiedPatchesNewNMS;
	sort(begin(svmPatchesPositive), end(svmPatchesPositive), [](shared_ptr<ClassifiedPatch> a, shared_ptr<ClassifiedPatch> b) { return *a > *b; });
	for (const auto& p : maximaCoords) {
		const auto& foundPatchIt = std::find_if(begin(svmPatchesPositive), end(svmPatchesPositive), [p](shared_ptr<ClassifiedPatch> a) { 
			return (a->getPatch()->getX() == p.x && a->getPatch()->getY() == p.y); }); // should never fail but we should handle an error anyway
		classifiedPatchesNewNMS.push_back(*foundPatchIt);
	}
	svmPatchesPositive = classifiedPatchesNewNMS; 
	// end new nms

	// The highest one of all the positively classified SVM patches
	//sort(make_indirect_iterator(svmPatches.begin()), make_indirect_iterator(svmPatches.end()), greater<ClassifiedPatch>());
	// TODO: Move to a function NMS or similar...? Similar than OE? Is there a family of functions that work on a vector of patches or classifiedPatches?
	sort(svmPatchesPositive.begin(), svmPatchesPositive.end(), [](shared_ptr<ClassifiedPatch> a, shared_ptr<ClassifiedPatch> b) { return *a > *b; });
	vector<shared_ptr<ClassifiedPatch>> svmPatchesMaxPositive;
	if(svmPatchesPositive.size()>0) {	
		svmPatchesMaxPositive.push_back(svmPatchesPositive[0]);
	}
	Mat imgSvmMaxPos = image.clone();
	//imageLogger.final(imgSvmMaxPos, bind(drawBoxes, imgSvmMaxPos, svmPatchesMaxPositive), "04svmmaxpos");
	imageLogger.final(imgSvmMaxPos, bind(drawBoxes, imgSvmMaxPos, svmPatchesPositive), "04svmmaxpos"); // all patches from new NMS
	
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

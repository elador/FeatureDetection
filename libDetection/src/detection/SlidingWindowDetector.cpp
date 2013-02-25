/*
 * SlidingWindowDetector.cpp
 *
 *  Created on: 22.02.2013
 *      Author: Patrik Huber
 */

#include "detection/SlidingWindowDetector.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/ImagePyramid.hpp"
//#include "imageprocessing/ImagePyramidLayer.hpp"
#include "imageprocessing/PyramidPatchExtractor.hpp"
#include "imageprocessing/FilteringFeatureTransformer.hpp"
#include "imageprocessing/IdentityFeatureTransformer.hpp"
#include "imageprocessing/MultipleImageFilter.hpp"
#include "imageprocessing/HistogramEqualizationFilter.hpp"
//#include "imageprocessing/FeatureExtractor.hpp"
#include "classification/BinaryClassifier.hpp"

#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

#include <iostream>

using namespace imageprocessing;

namespace detection {

SlidingWindowDetector::SlidingWindowDetector( shared_ptr<BinaryClassifier> classifier, int stepSizeX, int stepSizeY ) :
		classifier(classifier), stepSizeX(stepSizeX), stepSizeY(stepSizeY)
{

}

vector<pair<shared_ptr<Patch>, pair<bool, double>>> SlidingWindowDetector::detect( shared_ptr<ImagePyramid> imagePyramid ) const
{
	vector<pair<shared_ptr<Patch>, pair<bool, double>>> classifiedPatches;
	// For now, this just loops over all pyramid layers given. In the long run, we have to think about pyramid sharing.
	PyramidPatchExtractor ppe(imagePyramid, 20, 20);
	vector<shared_ptr<Patch>> pyramidPatches = ppe.extract(stepSizeX, stepSizeY);

	shared_ptr<IdentityFeatureTransformer> ift = make_shared<IdentityFeatureTransformer>();
	shared_ptr<MultipleImageFilter> mif = make_shared<MultipleImageFilter>();
	shared_ptr<HistogramEqualizationFilter> hef = make_shared<HistogramEqualizationFilter>();
	mif->add(hef);
	shared_ptr<FilteringFeatureTransformer> fft = make_shared<FilteringFeatureTransformer>(ift, mif);

	for(unsigned int i=0; i<pyramidPatches.size(); ++i) {
		cv::namedWindow("p", CV_WINDOW_AUTOSIZE); cv::imshow("p", pyramidPatches[i]->getData());
		pair<bool, double> res = classifier->classify(pyramidPatches[i]->getData());
		std::cout << res.second << ", ";
		fft->transform(pyramidPatches[i]->getData());
		cv::namedWindow("p2", CV_WINDOW_AUTOSIZE); cv::imshow("p2", pyramidPatches[i]->getData());
		pair<bool, double> res2 = classifier->classify(pyramidPatches[i]->getData());
		std::cout << res2.second << std::endl;
		cv::waitKey();
	}

	for(unsigned int i=0; i<pyramidPatches.size(); ++i) {
		pair<bool, double> res = classifier->classify(pyramidPatches[i]->getData());
	}
	
	// TODO: Try with a FeatureExtractor! (combines PatchExtractor & FeatureTransformer)
				//classifier->classify()



	return classifiedPatches;
}

} /* namespace detection */

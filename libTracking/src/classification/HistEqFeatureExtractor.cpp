/*
 * HistEqFeatureExtractor.cpp
 *
 *  Created on: 20.11.2012
 *      Author: poschmann
 */

#include "classification/HistEqFeatureExtractor.h"
#include "classification/FeatureVector.h"
#include "opencv2/imgproc/imgproc.hpp"
#include <memory>

using std::shared_ptr;
using std::make_shared;

namespace classification {

HistEqFeatureExtractor::HistEqFeatureExtractor(
		Size featureSize, double scaleFactor, double minHeight, double maxHeight) :
				ImagePyramidFeatureExtractor(featureSize, scaleFactor, minHeight, maxHeight), histEqPatch() {}

HistEqFeatureExtractor::~HistEqFeatureExtractor() {}

void HistEqFeatureExtractor::initScale(Mat image) {}

shared_ptr<FeatureVector> HistEqFeatureExtractor::extract(const Mat& patch) {
	cv::equalizeHist(patch, histEqPatch);
	int cols = histEqPatch.cols;
	int rows = histEqPatch.rows;
	if (histEqPatch.isContinuous()) {
		cols *= rows;
		rows = 1;
	}
	shared_ptr<FeatureVector> featureVector = make_shared<FeatureVector>(rows * cols);
	int index = 0;
	for (int r = 0; r < rows; ++r) {
		const unsigned char* Mrow = histEqPatch.ptr<unsigned char>(r);
		for (int c = 0; c < cols; ++c)
			featureVector->set(index++, (float)Mrow[c] / 255.0f);
	}
	return featureVector;
}

} /* namespace classification */

/*
 * TemplateMatchingClassifier.cpp
 *
 *  Created on: 19.06.2013
 *      Author: poschmann
 */

#include "classification/TemplateMatchingClassifier.hpp"
#include "opencv2/imgproc/imgproc.hpp"

namespace classification {

TemplateMatchingClassifier::TemplateMatchingClassifier() :
		templateVector(), positiveTemplates(), negativeTemplates(), result(1, 1, CV_32F) {}

TemplateMatchingClassifier::~TemplateMatchingClassifier() {}

pair<bool, double> TemplateMatchingClassifier::classify(const Mat& featureVector) const {
	// CV_TM_CCOEFF_NORMED -> [-1; 1]
	// CV_TM_CCORR_NORMED -> [0; 1]
	// CV_TM_SQDIFF_NORMED -> [0; 1]

//	cv::matchTemplate(featureVector, templateVector, result, CV_TM_CCOEFF_NORMED);
////	cv::matchTemplate(featureVector, templateVector, result, CV_TM_CCORR_NORMED);
//	double probability = 0.5 * (result.ptr<float>(0)[0] + 1);
////	double probability = result.ptr<float>(0)[0];
//	if (probability < 0 || probability > 1)
//		std::cout << result.ptr<float>(0)[0] << std::endl;
//	return std::make_pair(probability > 0.8, probability);

	int method = CV_TM_CCOEFF_NORMED;
//	int method = CV_TM_CCORR_NORMED;
//	int method = CV_TM_SQDIFF_NORMED;
	float bestPositive = 0;
	for (auto templateVector = positiveTemplates.begin(); templateVector != positiveTemplates.end(); ++templateVector) {
		cv::matchTemplate(featureVector, *templateVector, result, method);
		bestPositive = std::max(bestPositive, result.ptr<float>(0)[0]);
	}
	float bestNegative = 0;
	for (auto templateVector = negativeTemplates.begin(); templateVector != negativeTemplates.end(); ++templateVector) {
		cv::matchTemplate(featureVector, *templateVector, result, method);
		bestNegative = std::max(bestNegative, result.ptr<float>(0)[0]);
	}
	if (method == CV_TM_CCOEFF_NORMED) {
		bestPositive = 0.5 * (bestPositive + 1);
		bestNegative = 0.5 * (bestNegative + 1);
	} else if (method == CV_TM_SQDIFF_NORMED) {
		bestPositive = 1 - bestPositive;
		bestNegative = 1 - bestNegative;
	}
	double probability = bestPositive / (bestPositive + bestNegative);
	return std::make_pair(probability > 0.55, probability);
}

bool TemplateMatchingClassifier::isUsable() const {
//	return templateVector.rows * templateVector.cols > 0;
	return !positiveTemplates.empty() && !negativeTemplates.empty();
}

bool TemplateMatchingClassifier::retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples) {
//	if (templateVector.rows * templateVector.cols == 0 && !newPositiveExamples.empty())
//		templateVector = newPositiveExamples[0];
	positiveTemplates.insert(positiveTemplates.end(), newPositiveExamples.begin(), newPositiveExamples.end());
	negativeTemplates.insert(negativeTemplates.end(), newNegativeExamples.begin(), newNegativeExamples.end());
	return isUsable();
}

void TemplateMatchingClassifier::reset() {
//	templateVector = Mat();
	positiveTemplates.clear();
	negativeTemplates.clear();
}

} /* namespace classification */

/*
 * TemplateMatchingClassifier.hpp
 *
 *  Created on: 19.06.2013
 *      Author: poschmann
 */

#ifndef TEMPLATEMATCHINGCLASSIFIER_HPP_
#define TEMPLATEMATCHINGCLASSIFIER_HPP_

#include "classification/TrainableProbabilisticClassifier.hpp"

namespace classification {

class TemplateMatchingClassifier : public TrainableProbabilisticClassifier {
public:

	TemplateMatchingClassifier();

	~TemplateMatchingClassifier();

	pair<bool, double> classify(const Mat& featureVector) const;

	bool isUsable() const;

	bool retrain(const vector<Mat>& newPositiveExamples, const vector<Mat>& newNegativeExamples);

	void reset();

private:

	Mat templateVector; ///< The template.
	vector<Mat> positiveTemplates;
	vector<Mat> negativeTemplates;
	Mat result; ///< The buffer of the result of the matching.
};

} /* namespace classification */
#endif /* TEMPLATEMATCHINGCLASSIFIER_HPP_ */

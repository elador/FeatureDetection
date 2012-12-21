/*
 * TrainableClassifier.h
 *
 *  Created on: 06.12.2012
 *      Author: poschmann
 */

#ifndef TRAINABLECLASSIFIER_H_
#define TRAINABLECLASSIFIER_H_

#include "classification/Classifier.h"
#include "boost/shared_ptr.hpp"
#include <vector>

using boost::shared_ptr;
using std::vector;

namespace classification {

/**
 * Classifier that may be re-trained using new examples. Re-training is an iterative procedure that adds
 * new examples and refines the classifier. Nevertheless, it may be possible that previous training
 * examples are forgotten to ensure the classifier stays relatively small and efficient.
 */
class TrainableClassifier : public Classifier {
public:

	virtual ~TrainableClassifier() {}

	virtual pair<bool, double> classify(const FeatureVector& featureVector) const = 0;

	/**
	 * Re-trains this classifier. May not change the classifier if there are not enough samples.
	 *
	 * @param[in] positiveExamples The new positive training examples.
	 * @param[in] negativeExamples The new negative training examples.
	 * @return True if this classifier was trained successfully and may be used, false otherwise.
	 */
	virtual bool retrain(const vector<shared_ptr<FeatureVector> >& positiveExamples,
			const vector<shared_ptr<FeatureVector> >& negativeExamples) = 0;

	/**
	 * Resets this classifier. May not change the classifier at all, but it should not be used
	 * afterwards until it is re-trained.
	 *
	 * @param[in] classifier The classifier to reset.
	 */
	virtual void reset() = 0;
};

} /* namespace classification */
#endif /* TRAINABLECLASSIFIER_H_ */

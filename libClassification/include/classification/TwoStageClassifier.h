/*
 * TwoStageClassifier.h
 *
 *  Created on: 21.12.2012
 *      Author: poschmann
 */

#ifndef TWOSTAGECLASSIFIER_H_
#define TWOSTAGECLASSIFIER_H_

#include "classification/Classifier.h"
#include <memory>

using std::shared_ptr;

namespace classification {

/**
 * Classifier that consists of two stages. The first stage acts as a guard for the second one, so
 * feature vectors have to get past the first classifier to get classified by the second one. If
 * they do not get past the first one, they will be regarded as negative.
 */
class TwoStageClassifier : public Classifier {
public:

	/**
	 * Constructs a new two-stage classifier.
	 *
	 * @param[in] first The first classifier.
	 * @param[in] second The second classifier.
	 */
	explicit TwoStageClassifier(shared_ptr<Classifier> first, shared_ptr<Classifier> second);

	virtual ~TwoStageClassifier();

	pair<bool, double> classify(const FeatureVector& featureVector) const;

private:

	shared_ptr<Classifier> first;  ///< The first classifier.
	shared_ptr<Classifier> second; ///< The second classifier.
};

} /* namespace classification */
#endif /* TWOSTAGECLASSIFIER_H_ */

/*
 * TrainableTwoStageClassifier.h
 *
 *  Created on: 21.12.2012
 *      Author: poschmann
 */

#ifndef TRAINABLETWOSTAGECLASSIFIER_H_
#define TRAINABLETWOSTAGECLASSIFIER_H_

#include "classification/TrainableClassifier.h"
#include "classification/TwoStageClassifier.h"

namespace classification {

/**
 * Two-stage classifier that may be re-trained using new examples.
 */
class TrainableTwoStageClassifier : public TwoStageClassifier, public TrainableClassifier {
public:

	/**
	 * Constructs a new trainable two-stage classifier.
	 *
	 * @param[in] first The first classifier.
	 * @param[in] second The second classifier (will be re-trained).
	 */
	explicit TrainableTwoStageClassifier(shared_ptr<Classifier> first, shared_ptr<TrainableClassifier> second);

	~TrainableTwoStageClassifier();

	pair<bool, double> classify(const FeatureVector& featureVector) const {
		return TwoStageClassifier::classify(featureVector);
	}

	bool retrain(const vector<shared_ptr<FeatureVector> >& positiveExamples,
			const vector<shared_ptr<FeatureVector> >& negativeExamples) {
		return trainable->retrain(positiveExamples, negativeExamples);
	}

	void reset() {
		return trainable->reset();
	}

private:

	shared_ptr<TrainableClassifier> trainable; ///< The classifier that will be trained.
};

} /* namespace classification */
#endif /* TRAINABLETWOSTAGECLASSIFIER_H_ */

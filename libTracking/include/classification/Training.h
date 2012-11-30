/*
 * Training.h
 *
 *  Created on: 20.11.2012
 *      Author: poschmann
 */

#ifndef TRAINING_H_
#define TRAINING_H_

#include "boost/shared_ptr.hpp"
#include <vector>

using boost::shared_ptr;
using std::vector;

namespace classification {

class FeatureVector;

/**
 * Algorithm for training classifiers.
 */
template <class T>
class Training {
public:

	virtual ~Training() {}

	/**
	 * Re-trains a classifier. May not change the classifier if there are not enough samples.
	 *
	 * @param[in] classifier The classifier to re-train.
	 * @param[in] positiveSamples The new positive training samples.
	 * @param[in] negativeSamples The new negative training samples.
	 * @return True if the classifier was trained successfully, false otherwise.
	 */
	virtual bool retrain(T& classifier, const vector<shared_ptr<FeatureVector> >& positiveSamples,
			const vector<shared_ptr<FeatureVector> >& negativeSamples) = 0;

	/**
	 * Resets the training and the classifier. May not change the classifier at all, but it should not be used
	 * afterwards until it is re-trained.
	 *
	 * @param[in] classifier The classifier to reset.
	 */
	virtual void reset(T& classifier) = 0;
};

} /* namespace tracking */
#endif /* TRAINING_H_ */

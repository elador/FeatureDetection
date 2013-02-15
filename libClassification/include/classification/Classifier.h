/*
 * Classifier.h
 *
 *  Created on: 20.11.2012
 *      Author: poschmann
 */

#ifndef CLASSIFIER_H_
#define CLASSIFIER_H_

#include <utility>

using std::pair;

namespace classification {

class FeatureVector;

/**
 * Classifier that determines whether a feature vector is of a certain class or not.
 */
class Classifier {
public:

	virtual ~Classifier() {}

	/**
	 * Classifies a feature vector.
	 *
	 * @param[in] featureVector The feature vector.
	 * @return A pair containing a binary flag (true if positively classified) and a probability for being positive.
	 */
	virtual pair<bool, double> classify(const FeatureVector& featureVector) const = 0;
};

} /* namespace classification */
#endif /* CLASSIFIER_H_ */

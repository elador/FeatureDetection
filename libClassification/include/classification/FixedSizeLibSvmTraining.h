/*
 * FixedSizeLibSvmTraining.h
 *
 *  Created on: 07.12.2012
 *      Author: poschmann
 */

#ifndef FIXEDSIZELIBSVMTRAINING_H_
#define FIXEDSIZELIBSVMTRAINING_H_

#include "classification/LibSvmTraining.h"

namespace classification {

/**
 * LibSVM training that stores a certain amount of positive and negative training examples. The very
 * first positive training example will be stored forever, while the others may be replaced by new
 * positive examples. The examples the current classifier is most certain about will be replaced.
 * New negative examples will replace the oldest known negative examples.
 */
class FixedSizeLibSvmTraining : public LibSvmTraining {
public:

	/**
	 * Constructs a new fixed size libSVM training.
	 *
	 * @param[in] positiveExamples The amount of positive examples that is stored for training.
	 * @param[in] negativeExamples The amount of negative examples that is stored for training.
	 * @param[in] minPositiveExamples The minimum amount of positive training examples needed for training.
	 */
	explicit FixedSizeLibSvmTraining(unsigned int positiveExamples = 10, unsigned int negativeExamples = 100,
			unsigned int minPositiveExamples = 1);

	virtual ~FixedSizeLibSvmTraining();

	bool retrain(LibSvmClassifier& classifier, const vector<shared_ptr<FeatureVector> >& newPositiveExamples,
			const vector<shared_ptr<FeatureVector> >& newNegativeExamples);

	void reset(LibSvmClassifier& svm);

private:

	/**
	 * Adds new positive examples. May replace existing examples.
	 *
	 * @param[in] newPositiveExamples The new positive examples.
	 * @param[in] classifier The current classifier.
	 */
	void addPositiveExamples(const vector<shared_ptr<FeatureVector> >& newPositiveExamples, LibSvmClassifier& classifier);

	/**
	 * Adds new negative examples. May replace existing examples.
	 *
	 * @param[in] newNegativeExamples The new negative examples.
	 */
	void addNegativeExamples(const vector<shared_ptr<FeatureVector> >& newNegativeExamples);

	unsigned int dimensions;  ///< The amount of dimensions of the feature vectors.
	vector<struct svm_node *> positiveExamples; ///< The positive training examples.
	vector<struct svm_node *> negativeExamples; ///< The negative training examples.
	unsigned int negativeInsertPosition; ///< The insertion index of new negative examples.
	unsigned int minPositiveExamples; ///< The minimum amount of positive training examples needed for training.
};

} /* namespace classification */
#endif /* FIXEDSIZELIBSVMTRAINING_H_ */

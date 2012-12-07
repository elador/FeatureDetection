/*
 * FrameBasedLibSvmTraining.h
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#ifndef FRAMEBASEDLIBSVMTRAINING_H_
#define FRAMEBASEDLIBSVMTRAINING_H_

#include "classification/LibSvmTraining.h"
#include "classification/LibSvmParameterBuilder.h"
#include "classification/RbfLibSvmParameterBuilder.h"
#include "classification/SigmoidParameterComputation.h"
#include "classification/FixedApproximateSigmoidParameterComputation.h"
#include "svm.h"
#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include <string>
#include <vector>
#include <utility>

using boost::shared_ptr;
using boost::make_shared;
using std::vector;

namespace classification {

/**
 * LibSVM training that uses the samples of the last frames for training.
 */
class FrameBasedLibSvmTraining : public LibSvmTraining {
public:

	/**
	 * Constructs a new frame based libSVM training.
	 *
	 * @param[in] frameLength The length of the memory in frames.
	 * @param[in] minAvgSamples The minimum average positive training examples per frame for the training to be reasonable.
	 * @param[in] parameterBuilder The libSVM parameter builder.
	 * @param[in] sigmoidParameterComputation The computation of the sigmoid parameters.
	 */
	explicit FrameBasedLibSvmTraining(int frameLength, float minAvgSamples,
			shared_ptr<LibSvmParameterBuilder> parameterBuilder = make_shared<RbfLibSvmParameterBuilder>(),
			shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation
					= make_shared<FixedApproximateSigmoidParameterComputation>());

	/**
	 * Constructs a new frame based libSVM training with some static negative training examples.
	 *
	 * @param[in] frameLength The length of the memory in frames.
	 * @param[in] minAvgSamples The minimum average positive training examples per frame for the training to be reasonable.
	 * @param[in] negativesFilename The name of the file containing the static negative training examples.
	 * @param[in] negatives The amount of static negative training examples to use.
	 * @param[in] parameterBuilder The libSVM parameter builder.
	 * @param[in] sigmoidParameterComputation The computation of the sigmoid parameters.
	 */
	explicit FrameBasedLibSvmTraining(int frameLength, float minAvgSamples, std::string negativesFilename, int negatives,
			shared_ptr<LibSvmParameterBuilder> parameterBuilder = make_shared<RbfLibSvmParameterBuilder>(),
			shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation
					= make_shared<FixedApproximateSigmoidParameterComputation>());

	~FrameBasedLibSvmTraining();

	bool retrain(LibSvmClassifier& svm, const vector<shared_ptr<FeatureVector> >& newPositiveExamples,
			const vector<shared_ptr<FeatureVector> >& newNegativeExamples);

	void reset(LibSvmClassifier& svm);

	/**
	 * @return The required amount of positive training examples for the training to be reasonable.
	 */
	int getRequiredPositiveSampleCount() const;

	/**
	 * @return The number of positive training examples.
	 */
	int getPositiveSampleCount() const;

	/**
	 * @return The number of negative training examples (static and dynamic).
	 */
	int getNegativeSampleCount() const;

private:

	/**
	 * @return True if the training is reasonable, false otherwise.
	 */
	bool isTrainingReasonable() const;

	/**
	 * Adds new training examples.
	 *
	 * @param[in] newPositiveExamples The new positive training examples.
	 * @param[in] newNegativeExamples The new negative training examples.
	 */
	void addExamples(const vector<shared_ptr<FeatureVector> >& newPositiveExamples,
			const vector<shared_ptr<FeatureVector> >& newNegativeExamples);

	/**
	 * Replaces training examples by new ones.
	 *
	 * @param[in] examples The vector of training examples whose content should be replaced with new ones.
	 * @param[in] newExamples The new training examples.
	 */
	void replaceExamples(vector<struct svm_node *>& examples, const vector<shared_ptr<FeatureVector> >& newExamples);

	/**
	 * Trains a libSVM classifier with the positive and negative samples.
	 *
	 * @param[in] svm The libSVM classifier that should be trained.
	 * @return True if the training was successful, false otherwise.
	 */
	bool train(LibSvmClassifier& svm);

	/**
	 * Collects the nodes of the vectors and puts them into a single vector.
	 *
	 * @param[in] nodes The vectors of nodes.
	 * @return A single vector containing all nodes.
	 */
	vector<struct svm_node *> collectNodes(const vector<vector<struct svm_node *> >& nodes);

	int frameLength;     ///< The length of the memory in frames.
	float minAvgSamples; ///< The minimum average positive training examples per frame for the training to be reasonable.
	unsigned int dimensions; ///< The amount of dimensions of the feature vectors.
	vector<vector<struct svm_node *> > positiveExamples; ///< The positive training examples of the last frames.
	vector<vector<struct svm_node *> > negativeExamples; ///< The negative training examples of the last frames.
	int oldestEntry;                                     ///< The index of the oldest example entry.
};

} /* namespace classification */
#endif /* FRAMEBASEDLIBSVMTRAINING_H_ */

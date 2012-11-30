/*
 * FrameBasedSvmTraining.h
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#ifndef FRAMEBASEDSVMTRAINING_H_
#define FRAMEBASEDSVMTRAINING_H_

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
 * SVM training that uses the samples of the last frames for training.
 */
class FrameBasedSvmTraining : public LibSvmTraining {
public:

	/**
	 * Constructs a new frame based SVM training.
	 *
	 * @param[in] frameLength The length of the memory in frames.
	 * @param[in] minAvgSamples The minimum average positive samples per frame for the training to be reasonable.
	 * @param[in] parameterBuilder The libSVM parameter builder.
	 * @param[in] sigmoidParameterComputation The computation of the sigmoid parameters.
	 */
	explicit FrameBasedSvmTraining(int frameLength, float minAvgSamples,
			shared_ptr<LibSvmParameterBuilder> parameterBuilder = make_shared<RbfLibSvmParameterBuilder>(),
			shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation
					= make_shared<FixedApproximateSigmoidParameterComputation>());

	/**
	 * Constructs a new frame based SVM training with some static negative samples.
	 *
	 * @param[in] frameLength The length of the memory in frames.
	 * @param[in] minAvgSamples The minimum average positive samples per frame for the training to be reasonable.
	 * @param[in] negativesFilename The name of the file containing the static negative samples.
	 * @param[in] negatives The amount of static negative samples to use.
	 * @param[in] parameterBuilder The libSVM parameter builder.
	 * @param[in] sigmoidParameterComputation The computation of the sigmoid parameters.
	 */
	explicit FrameBasedSvmTraining(int frameLength, float minAvgSamples, std::string negativesFilename, int negatives,
			shared_ptr<LibSvmParameterBuilder> parameterBuilder = make_shared<RbfLibSvmParameterBuilder>(),
			shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation
					= make_shared<FixedApproximateSigmoidParameterComputation>());

	~FrameBasedSvmTraining();

	bool retrain(LibSvmClassifier& svm, const vector<shared_ptr<FeatureVector> >& positivePatches,
			const vector<shared_ptr<FeatureVector> >& negativePatches);

	void reset(LibSvmClassifier& svm);

	/**
	 * @return The required amount of positive samples for the training to be reasonable.
	 */
	int getRequiredPositiveSampleCount() const;

	/**
	 * @return The number of positive samples.
	 */
	int getPositiveSampleCount() const;

	/**
	 * @return The number of negative samples (static and dynamic).
	 */
	int getNegativeSampleCount() const;

private:

	/**
	 * @return True if the training is reasonable, false otherwise.
	 */
	bool isTrainingReasonable() const;

	/**
	 * Adds new training samples.
	 *
	 * @param[in] positiveSamples The new positive samples.
	 * @param[in] negativeSamples The new negative samples.
	 */
	void addSamples(const vector<shared_ptr<FeatureVector> >& positiveSamples,
			const vector<shared_ptr<FeatureVector> >& negativeSamples);

	/**
	 * Replaces training samples by new ones.
	 *
	 * @param[in] trainingSamples The vector of training samples whose content should be replaced with new ones.
	 * @param[in] samples The samples that should be transformed into the new training samples.
	 */
	void replaceSamples(vector<struct svm_node *>& trainingSamples, const vector<shared_ptr<FeatureVector> >& samples);

	/**
	 * Counts the samples.
	 *
	 * @param[in] samples The samples.
	 * @return The number of samples.
	 */
	int getSampleCount(const vector<vector<struct svm_node *> >& samples) const;

	/**
	 * Trains a libSVM classifier with the positive and negative samples.
	 *
	 * @param[in] svm The libSVM classifier that should be trained.
	 * @return True if the training was successful, false otherwise.
	 */
	bool train(LibSvmClassifier& svm);

	/**
	 * Creates the libSVM problem.
	 *
	 * @param[in] count The amount of samples.
	 * @return The libSVM problem.
	 */
	struct svm_problem *createProblem(unsigned int count);

	int frameLength;     ///< The length of the memory in frames.
	float minAvgSamples; ///< The minimum average positive samples per frame for the training to be reasonable.
	unsigned int dimensions; ///< The amount of dimensions of the feature vectors.
	vector<vector<struct svm_node *> > positiveTrainingSamples; ///< The positive training samples of the last frames.
	vector<vector<struct svm_node *> > negativeTrainingSamples; ///< The negative training samples of the last frames.
	int oldestEntry;                                            ///< The index of the oldest sample entry.
};

} /* namespace tracking */
#endif /* FRAMEBASEDSVMTRAINING_H_ */

/*
 * FrameBasedSvmTraining.h
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#ifndef FRAMEBASEDSVMTRAINING_H_
#define FRAMEBASEDSVMTRAINING_H_

#include "tracking/LibSvmTraining.h"
#include "tracking/SigmoidParameterComputation.h"
#include "tracking/FastApproximateSigmoidParameterComputation.h"
#include "svm.h"
#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include <string>
#include <vector>
#include <utility>

class FdPatch;

using boost::shared_ptr;
using boost::make_shared;

namespace tracking {

/**
 * SVM training that uses the patches of the last frames for training.
 */
class FrameBasedSvmTraining : public LibSvmTraining {
public:

	/**
	 * Constructs a new frame based SVM training.
	 *
	 * @param[in] frameLength The length of the memory in frames.
	 * @param[in] minAvgSamples The minimum average positive samples per frame for the training to be reasonable.
	 * @param[in] sigmoidParameterComputation The computation of the sigmoid parameters.
	 */
	explicit FrameBasedSvmTraining(int frameLength, float minAvgSamples,
			shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation
					= make_shared<FastApproximateSigmoidParameterComputation>());

	/**
	 * Constructs a new frame based SVM training with some static negative samples.
	 *
	 * @param[in] frameLength The length of the memory in frames.
	 * @param[in] minAvgSamples The minimum average positive samples per frame for the training to be reasonable.
	 * @param[in] negativesFilename The name of the file containing the static negative samples.
	 * @param[in] negatives The amount of static negative samples to use.
	 * @param[in] sigmoidParameterComputation The computation of the sigmoid parameters.
	 */
	explicit FrameBasedSvmTraining(int frameLength, float minAvgSamples, std::string negativesFilename, int negatives,
			shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation
					= make_shared<FastApproximateSigmoidParameterComputation>());

	~FrameBasedSvmTraining();

	bool retrain(ChangableDetectorSvm& svm, const std::vector<FdPatch*>& positivePatches,
			const std::vector<FdPatch*>& negativePatches);

	void reset(ChangableDetectorSvm& svm);

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
	 * Adds new samples based on image patches with positive or negative label.
	 *
	 * @param[in] positivePatches The new positive patches.
	 * @param[in] negativePatches The new negative patches.
	 */
	void addSamples(const std::vector<FdPatch*>& positivePatches, const std::vector<FdPatch*>& negativePatches);

	/**
	 * Replaces samples by new ones obtained from patches.
	 *
	 * @param[in] samples The vector of samples whose content should be replaced with new samples.
	 * @param[in] patches The patches that should be transformed into the new samples.
	 */
	void replaceSamples(std::vector<struct svm_node *>& samples, const std::vector<FdPatch*>& patches);

	/**
	 * Counts the samples.
	 *
	 * @param[in] samples The samples.
	 * @return The number of samples.
	 */
	int getSampleCount(const std::vector<std::vector<struct svm_node *> >& samples) const;

	/**
	 * Trains a support vector machine with the positive and negative samples.
	 *
	 * @param[in] svm The SVM that should be trained.
	 * @return True if the training was successful, false otherwise.
	 */
	bool train(ChangableDetectorSvm& svm);

	/**
	 * Creates the libSVM parameters.
	 *
	 * @param[in] positiveCount The amount of positive samples.
	 * @param[in] negativeCount The amount of negative samples.
	 * @return The libSVM parameters.
	 */
	struct svm_parameter *createParameters(unsigned int positiveCount, unsigned int negativeCount);

	/**
	 * Creates the libSVM problem.
	 *
	 * @param[in] count The amount of samples.
	 * @return The libSVM problem.
	 */
	struct svm_problem *createProblem(unsigned int count);

	int frameLength;     ///< The length of the memory in frames.
	float minAvgSamples; ///< The minimum average positive samples per frame for the training to be reasonable.
	std::vector<std::vector<struct svm_node *> > positiveSamples; ///< The positive samples of the last frames.
	std::vector<std::vector<struct svm_node *> > negativeSamples; ///< The negative samples of the last frames.
	int oldestEntry;                                              ///< The index of the oldest sample entry.
};

} /* namespace tracking */
#endif /* FRAMEBASEDSVMTRAINING_H_ */

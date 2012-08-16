/*
 * FrameBasedSvmTraining.h
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#ifndef FRAMEBASEDSVMTRAINING_H_
#define FRAMEBASEDSVMTRAINING_H_

#include "tracking/SvmTraining.h"
#include "tracking/SigmoidParameterComputation.h"
#include "tracking/ApproximateSigmoidParameterComputation.h"
#include "svm.h"
#include <string>
#include <vector>
#include <utility>

class FdPatch;
struct svm_node;
struct svm_model;

namespace tracking {

/**
 * SVM training that uses the patches of the last frames for training.
 */
class FrameBasedSvmTraining : public SvmTraining {
public:

	/**
	 * Constructs a new frame based SVM training.
	 *
	 * @param[in] frameLength The length of the memory in frames.
	 * @param[in] minAvgSamples The minimum average positive samples per frame for the training to be reasonable.
	 * @param[in] negativesFilename The name of the file containing the static negative samples.
	 * @param[in] negatives The amount of static negative samples to use.
	 * @param[in] sigmoidParameterComputation The computation of the sigmoid parameters. Will be deleted at destruction.
	 */
	explicit FrameBasedSvmTraining(int frameLength, float minAvgSamples, std::string negativesFilename, int negatives,
			SigmoidParameterComputation *sigmoidParameterComputation = new ApproximateSigmoidParameterComputation());
	virtual ~FrameBasedSvmTraining();

	bool retrain(ChangableDetectorSvm* svm, const std::vector<FdPatch*>& positivePatches,
			const std::vector<FdPatch*>& negativePatches);

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
	 * Reads the static negative samples from a file.
	 *
	 * @param[in] negativesFilename The name of the file containing the static negative samples.
	 * @param[in] maxNegatives The amount of static negative samples to use.
	 */
	void readStaticNegatives(const std::string negativesFilename, int maxNegatives);

	/**
	 * Deletes the sample data and clears the vector.
	 *
	 * @param[in] samples The samples that should be deleted.
	 */
	void freeSamples(std::vector<struct svm_node *>& samples);

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
	bool train(ChangableDetectorSvm* svm);

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

	/**
	 * Changes the parameters of an SVM given a libSVM model.
	 *
	 * @param[in] The SVM whose parameters should be changed.
	 * @param[in] model The libSVM model.
	 * @param[in] problem The libSVM problem.
	 * @param[in] positiveCount The amount of positive samples used for the training of the SVM.
	 * @param[in] negativeCount The amount of negative samples used for the training of the SVM.
	 */
	void changeSvmParameters(ChangableDetectorSvm* svm, struct svm_model *model,
			struct svm_problem *problem, unsigned int positiveCount, unsigned int negativeCount);

	int frameLength;     ///< The length of the memory in frames.
	float minAvgSamples; ///< The minimum average positive samples per frame for the training to be reasonable.
	std::vector<struct svm_node *> staticNegativeSamples;         ///< The static negative samples.
	std::vector<std::vector<struct svm_node *> > positiveSamples; ///< The positive samples of the last frames.
	std::vector<std::vector<struct svm_node *> > negativeSamples; ///< The negative samples of the last frames.
	int oldestEntry;                                              ///< The index of the oldest sample entry.
	SigmoidParameterComputation *sigmoidParameterComputation; ///< The computation of the sigmoid parameters.
};

} /* namespace tracking */
#endif /* FRAMEBASEDSVMTRAINING_H_ */

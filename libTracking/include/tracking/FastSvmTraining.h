/*
 * FastSvmTraining.h
 *
 *  Created on: 18.09.2012
 *      Author: poschmann
 */

#ifndef FASTSVMTRAINING_H_
#define FASTSVMTRAINING_H_

#include "tracking/LibSvmTraining.h"
#include "tracking/LibSvmParameterBuilder.h"
#include "tracking/RbfLibSvmParameterBuilder.h"
#include "tracking/SigmoidParameterComputation.h"
#include "tracking/FastApproximateSigmoidParameterComputation.h"
#include "svm.h"
#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include <vector>
#include <utility>

class FdPatch;

using boost::shared_ptr;
using boost::make_shared;

namespace tracking {

/**
 * SVM training that limits the amount of training data to ensure a fast training. It will use the support
 * vectors of the previously trained SVM and the new samples for training. If the amount of support vectors
 * is bigger than a pre-defined threshold, some of the support vectors will be removed from the training
 * data of the next learning cycle, so at most a certain amount of support vectors will be used for the next
 * training. The support vectors that will be removed are chosen to be the ones with the biggest distance to
 * the separating hyperplane.
 */
class FastSvmTraining : public LibSvmTraining {
public:

	/**
	 * Constructs a new fast SVM training.
	 *
	 * @param[in] minPosCount The minimum amount of positive training samples necessary for training (default ist 10).
	 * @param[in] minNegCount The minimum amount of negative training samples necessary for training (default ist 10).
	 * @param[in] maxCount The maximum amount of support vectors used for training (default is 50).
	 * @param[in] parameterBuilder The libSVM parameter builder.
	 * @param[in] sigmoidParameterComputation The computation of the sigmoid parameters.
	 */
	explicit FastSvmTraining(unsigned int minPosCount = 10, unsigned int minNegCount = 10, unsigned int maxCount = 50,
			shared_ptr<LibSvmParameterBuilder> parameterBuilder = make_shared<RbfLibSvmParameterBuilder>(),
			shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation
					= make_shared<FastApproximateSigmoidParameterComputation>());
	~FastSvmTraining();

	bool retrain(ChangableDetectorSvm& svm, const std::vector<FdPatch*>& positivePatches,
			const std::vector<FdPatch*>& negativePatches);

	void reset(ChangableDetectorSvm& svm);

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
	 * Adds new samples based on image patches.
	 *
	 * @param[in] samples The samples.
	 * @param[in] patches The new patches.
	 */
	void addSamples(std::vector<struct svm_node *>& samples, const std::vector<FdPatch*>& patches);

	/**
	 * Trains a support vector machine with the positive and negative samples.
	 *
	 * @param[in] svm The SVM that should be trained.
	 * @return True if the training was successful, false otherwise.
	 */
	bool train(ChangableDetectorSvm& svm);

	/**
	 * Creates the libSVM problem.
	 *
	 * @return The libSVM problem.
	 */
	struct svm_problem *createProblem();

	/**
	 * Removes all samples that are not contained within the SVM model as support vectors.
	 *
	 * @param[in] model The libSVM model.
	 * @return The support vectors that should be deleted when the model is destroyed.
	 */
	std::vector<struct svm_node *> retainSupportVectors(struct svm_model *model);

	/**
	 * Creates a list of samples that are contained within the SVM model as support vectors and destroys the other ones.
	 *
	 * @param[in] samples The samples.
	 * @param[in] model The libSVM model.
	 * @param[in] count The estimated amount of support vectors (used to initialize the list).
	 * @return The samples that are support vectors.
	 */
	std::vector<struct svm_node *> extractSupportVectors(
			std::vector<struct svm_node *>& samples, struct svm_model *model, unsigned int count = 1);

	/**
	 * Determines whether a vector is a support vector.
	 *
	 * @param[in] vector The vector.
	 * @param[in] model The libSVM model.
	 * @return True if the vector is a support vector, false otherwise.
	 */
	bool isSupportVector(struct svm_node *vector, struct svm_model *model);

	/**
	 * Determines whether two vectors are equal.
	 *
	 * @param[in] v1 The first vector.
	 * @param[in] v2 The second vector.
	 * @return True if the vectors are equal, false otherwise.
	 */
	bool areVectorsEqual(struct svm_node *v1, struct svm_node *v2);

	/**
	 * Determines the coefficients (alphas) to the given support vectors.
	 *
	 * @param[in] supportVectors The support vectors.
	 * @param[in] model The libSVM model.
	 * @return The coefficients of the support vectors.
	 */
	std::vector<double> getCoefficients(std::vector<struct svm_node *>& supportVectors, struct svm_model *model);

	/**
	 * Computes the distance to the separating hyperplane of each sample.
	 *
	 * @param[in] samples The samples.
	 * @param[in] model The libSVM model.
	 * @return The distances.
	 */
	std::vector<double> computeHyperplaneDistances(std::vector<struct svm_node *>& samples, struct svm_model *model);

	/**
	 * Determines the smallest element.
	 *
	 * @param[in] values The values.
	 * @return A pair containing the index and value of the min element.
	 */
	std::pair<unsigned int, double> getMin(std::vector<double> values);

	/**
	 * Determines the biggest element.
	 *
	 * @param[in] values The values.
	 * @return A pair containing the index and value of the max element.
	 */
	std::pair<unsigned int, double> getMax(std::vector<double> values);

	unsigned int minPosCount; ///< The minimum amount of positive training samples necessary for training.
	unsigned int minNegCount; ///< The minimum amount of negative training samples necessary for training.
	unsigned int maxCount;    ///< The maximum amount of support vectors used for training.
	std::vector<struct svm_node *> positiveSamples; ///< The positive samples.
	std::vector<struct svm_node *> negativeSamples; ///< The negative samples.
	shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation; ///< The computation of the sigmoid parameters.
};

} /* namespace tracking */
#endif /* FASTSVMTRAINING_H_ */

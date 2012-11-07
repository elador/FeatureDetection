/*
 * LibSvmTraining.h
 *
 *  Created on: 24.09.2012
 *      Author: poschmann
 */

#ifndef LIBSVMTRAINING_H_
#define LIBSVMTRAINING_H_

#include "tracking/SvmTraining.h"
#include "tracking/LibSvmParameterBuilder.h"
#include "tracking/RbfLibSvmParameterBuilder.h"
#include "tracking/SigmoidParameterComputation.h"
#include "tracking/FastApproximateSigmoidParameterComputation.h"
#include "svm.h"
#include "boost/shared_ptr.hpp"
#include "boost/make_shared.hpp"
#include <vector>

struct svm_parameter;

using boost::shared_ptr;
using boost::make_shared;

namespace tracking {

/**
 * SVM training that is based on libSVM.
 */
class LibSvmTraining : public SvmTraining {
public:

	/**
	 * Constructs a new libSVM based SVM training.
	 *
	 * @param[in] parameterBuilder The libSVM parameter builder.
	 * @param[in] sigmoidParameterComputation The computation of the sigmoid parameters.
	 */
	explicit LibSvmTraining(shared_ptr<LibSvmParameterBuilder> parameterBuilder = make_shared<RbfLibSvmParameterBuilder>(),
			shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation = make_shared<FastApproximateSigmoidParameterComputation>());

	virtual ~LibSvmTraining();

	virtual bool retrain(ChangableDetectorSvm& svm,
			const std::vector<FdPatch*>& positivePatches, const std::vector<FdPatch*>& negativePatches) = 0;

	virtual void reset(ChangableDetectorSvm& svm) = 0;

	/**
	 * Reads the static negative samples from a file.
	 *
	 * @param[in] negativesFilename The name of the file containing the static negative samples.
	 * @param[in] maxNegatives The amount of static negative samples to use.
	 */
	void readStaticNegatives(const std::string negativesFilename, int maxNegatives);

protected:

	/**
	 * Deletes the sample data and clears the vector.
	 *
	 * @param[in] samples The samples that should be deleted.
	 */
	void freeSamples(std::vector<struct svm_node *>& samples);

	/**
	 * Creates the libSVM parameters. In order to free their memory, svm_destroy_param has to be called.
	 *
	 * @param[in] positiveCount The amount of positive samples.
	 * @param[in] negativeCount The amount of negative samples.
	 * @return The libSVM parameters.
	 */
	struct svm_parameter *createParameters(unsigned int positiveCount, unsigned int negativeCount);

	/**
	 * Changes the parameters of an SVM given a libSVM model.
	 *
	 * @param[in] The SVM whose parameters should be changed.
	 * @param[in] model The libSVM model.
	 * @param[in] problem The libSVM problem.
	 * @param[in] positiveCount The amount of positive samples used for the training of the SVM.
	 * @param[in] negativeCount The amount of negative samples used for the training of the SVM.
	 */
	void changeSvmParameters(ChangableDetectorSvm& svm, struct svm_model *model,
			struct svm_problem *problem, unsigned int positiveCount, unsigned int negativeCount);

	std::vector<struct svm_node *> staticNegativeSamples; ///< The static negative samples.

private:

	shared_ptr<LibSvmParameterBuilder> parameterBuilder;                 ///< The libSVM parameter builder.
	shared_ptr<SigmoidParameterComputation> sigmoidParameterComputation; ///< The computation of the sigmoid parameters.
};

} /* namespace tracking */
#endif /* LIBSVMTRAINING_H_ */

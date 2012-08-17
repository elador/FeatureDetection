/*
 * SvmTraining.h
 *
 *  Created on: 31.07.2012
 *      Author: poschmann
 */

#ifndef SVMTRAINING_H_
#define SVMTRAINING_H_

#include <vector>

class FdPatch;
struct svm_node;

namespace tracking {

class ChangableDetectorSvm;

/**
 * Algorithm for re-training SVMs.
 */
class SvmTraining {
public:
	virtual ~SvmTraining() {}

	/**
	 * Re-trains a support vector machine. May not change the SVM if there are not enough samples.
	 *
	 * @param[in] svm The SVM to re-train.
	 * @param[in] positivePatches The new positive patches.
	 * @param[in] negativePatches The new negative patches.
	 * @return True if the SVM was trained successfully, false otherwise.
	 */
	virtual bool retrain(ChangableDetectorSvm& svm,
			const std::vector<FdPatch*>& positivePatches, const std::vector<FdPatch*>& negativePatches) = 0;
};

} /* namespace tracking */
#endif /* SVMTRAINING_H_ */

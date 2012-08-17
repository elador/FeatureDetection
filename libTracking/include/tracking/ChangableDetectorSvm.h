/*
 * ChangableDetectorSvm.h
 *
 *  Created on: 01.08.2012
 *      Author: poschmann
 */

#ifndef CHANGABLEDETECTORSVM_H_
#define CHANGABLEDETECTORSVM_H_

#include "DetectorSVM.h"

namespace tracking {

/**
 * SVM based detector whose parameters (support vectors etc.) can be changed.
 */
class ChangableDetectorSvm : public DetectorSVM {
public:

	/**
	 * Constructs a new changable detector SVM.
	 */
	explicit ChangableDetectorSvm();

	~ChangableDetectorSvm();

	/**
	 * @return The number of dimensions.
	 */
	inline unsigned int getDimensions() const {
		return filter_size_x * filter_size_y;
	}

	/**
	 * @return Parameter A of the probabilistic output equation p(ffp|t) = 1 / (1 + exp(A * t + B)).
	 */
	inline float getProbParamA() const {
		return posterior_svm[0];
	}

	/**
	 * @return Parameter B of the probabilistic output equation p(ffp|t) = 1 / (1 + exp(A * t + B)).
	 */
	inline float getProbParamB() const {
		return posterior_svm[1];
	}

	/**
	 * Changes the parameters of this SVM for classification with the RBF kernel.
	 *
	 * @param[in] numSv The number of support vectors.
	 * @param[in] supportVectors The support vectors.
	 * @param[in] alphas The weights of the support vectors.
	 * @param[in] rho The bias of the separating hyperplane.
	 * @param[in] gamma Parameter of the RBF kernel.
	 * @param[in] probParamA Parameter A of the probabilistic output equation (sigmoid function).
	 * @param[in] probParamB Parameter B of the probabilistic output equation (sigmoid function).
	 */
	void changeRbfParameters(int numSv, unsigned char** supportVectors, float* alphas, float rho, float gamma,
			float probParamA, float probParamB);
};

} /* namespace tracking */
#endif /* CHANGABLEDETECTORSVM_H_ */

/*
 * TransitionModel.hpp
 *
 *  Created on: 10.07.2012
 *      Author: poschmann
 */

#ifndef TRANSITIONMODEL_HPP_
#define TRANSITIONMODEL_HPP_

#include "opencv2/core/core.hpp"
#include <vector>
#include <memory>

using cv::Mat;
using std::vector;
using std::shared_ptr;

namespace condensation {

class Sample;

/**
 * Transition model that predicts the new state of samples.
 */
class TransitionModel {
public:

	virtual ~TransitionModel() {}

	/**
	 * Initializes this transition model.
	 *
	 * @param[in] image The current image.
	 */
	virtual void init(const Mat& image) = 0;

	/**
	 * Predicts the new state of the samples and adds some noise.
	 *
	 * @param[in,out] samples The samples.
	 * @param[in] image The new image.
	 * @param[in] target The previous target state.
	 */
	virtual void predict(vector<shared_ptr<Sample>>& samples, const Mat& image, const shared_ptr<Sample> target) = 0;
};

} /* namespace condensation */
#endif /* TRANSITIONMODEL_HPP_ */

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
	virtual void init(const cv::Mat& image) = 0;

	/**
	 * Predicts the new state of the samples and adds some noise.
	 *
	 * @param[in,out] samples The samples.
	 * @param[in] image The new image.
	 * @param[in] target The previous target state.
	 */
	virtual void predict(std::vector<std::shared_ptr<Sample>>& samples, const cv::Mat& image, const std::shared_ptr<Sample> target) = 0;
};

} /* namespace condensation */
#endif /* TRANSITIONMODEL_HPP_ */

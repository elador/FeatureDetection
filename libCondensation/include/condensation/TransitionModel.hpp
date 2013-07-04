/*
 * TransitionModel.hpp
 *
 *  Created on: 10.07.2012
 *      Author: poschmann
 */

#ifndef TRANSITIONMODEL_HPP_
#define TRANSITIONMODEL_HPP_

#include "opencv2/core/core.hpp"
#include "boost/optional.hpp"
#include <vector>

using cv::Mat;
using boost::optional;
using std::vector;

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
	virtual void predict(vector<Sample>& samples, const Mat& image, const optional<Sample>& target) = 0;
};

} /* namespace condensation */
#endif /* TRANSITIONMODEL_HPP_ */

/*
 * MeasurementModel.h
 *
 *  Created on: 12.07.2012
 *      Author: poschmann
 */

#ifndef MEASUREMENTMODEL_H_
#define MEASUREMENTMODEL_H_

#include "opencv2/highgui/highgui.hpp"
#include <vector>

using cv::Mat;
using std::vector;

namespace tracking {

class Sample;

/**
 * Measurement model for samples.
 */
class MeasurementModel {
public:
	virtual ~MeasurementModel() {}

	/**
	 * Changes the weights of samples according to the likelihood of an object existing at that positions an image.
	 *
	 * @param[in] image The image.
	 * @param[in] samples The samples whose weight will be changed according to the likelihoods.
	 */
	virtual void evaluate(const Mat& image, vector<Sample>& samples) = 0;
};

} /* namespace tracking */
#endif /* MEASUREMENTMODEL_H_ */

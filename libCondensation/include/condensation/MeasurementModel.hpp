/*
 * MeasurementModel.hpp
 *
 *  Created on: 12.07.2012
 *      Author: poschmann
 */

#ifndef MEASUREMENTMODEL_HPP_
#define MEASUREMENTMODEL_HPP_

#include <vector>
#include <memory>

using std::vector;
using std::shared_ptr;

namespace imageprocessing {
	class VersionedImage;
}
using imageprocessing::VersionedImage;

namespace condensation {

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
	virtual void evaluate(shared_ptr<VersionedImage> image, vector<Sample>& samples) = 0;
};

} /* namespace condensation */
#endif /* MEASUREMENTMODEL_HPP_ */

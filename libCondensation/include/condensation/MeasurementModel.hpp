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

namespace imageprocessing {
	class VersionedImage;
}

namespace condensation {

class Sample;

/**
 * Measurement model for samples.
 */
class MeasurementModel {
public:

	virtual ~MeasurementModel() {}

	/**
	 * Updates this model so all subsequent calls to evaluate use the data of the new image.
	 *
	 * @param[in] image The new image data.
	 */
	virtual void update(std::shared_ptr<imageprocessing::VersionedImage> image) = 0;

	/**
	 * Changes the weight of the sample according to the likelihood of an object existing at that positions an image.
	 *
	 * @param[in] sample The sample whose weight will be changed according to the likelihood.
	 */
	virtual void evaluate(Sample& sample) const = 0;

	/**
	 * Changes the weights of samples according to the likelihood of an object existing at that positions an image. Can
	 * be used instead of update and calls to evaluate for each sample individually.
	 *
	 * @param[in] image The image.
	 * @param[in] samples The samples whose weight will be changed according to the likelihoods.
	 */
	virtual void evaluate(std::shared_ptr<imageprocessing::VersionedImage> image, std::vector<std::shared_ptr<Sample>>& samples) {
		update(image);
		for (std::shared_ptr<Sample> sample : samples)
			evaluate(*sample);
	}
};

} /* namespace condensation */
#endif /* MEASUREMENTMODEL_HPP_ */

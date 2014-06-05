/*
 * CondensationTracker.cpp
 *
 *  Created on: 29.06.2012
 *      Author: poschmann
 */

#include "condensation/Sample.hpp"
#include "condensation/CondensationTracker.hpp"
#include "condensation/Sampler.hpp"
#include "condensation/MeasurementModel.hpp"
#include "condensation/StateExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"

using imageprocessing::VersionedImage;
using cv::Mat;
using cv::Rect;
using boost::optional;
using std::vector;
using std::shared_ptr;
using std::make_shared;

namespace condensation {

CondensationTracker::CondensationTracker(shared_ptr<Sampler> sampler,
		shared_ptr<MeasurementModel> measurementModel, shared_ptr<StateExtractor> extractor) :
				samples(),
				oldSamples(),
				state(),
				image(make_shared<VersionedImage>()),
				sampler(sampler),
				measurementModel(measurementModel),
				extractor(extractor) {}

optional<Rect> CondensationTracker::process(const Mat& imageData) {
	image->setData(imageData);
	samples.swap(oldSamples);
	samples.clear();
	sampler->sample(oldSamples, samples, image->getData(), state);
	// evaluate samples and extract position
	measurementModel->evaluate(image, samples);
	state = extractor->extract(samples);
	// return position
	if (state)
		return optional<Rect>(state->getBounds());
	return optional<Rect>();
}

} /* namespace condensation */

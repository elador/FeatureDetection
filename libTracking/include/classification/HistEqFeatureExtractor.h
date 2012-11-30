/*
 * HistEqFeatureExtractor.h
 *
 *  Created on: 20.11.2012
 *      Author: poschmann
 */

#ifndef HISTEQFEATUREEXTRACTOR_H_
#define HISTEQFEATUREEXTRACTOR_H_

#include "classification/ImagePyramidFeatureExtractor.h"

using cv::Size;
using cv::Mat;

namespace classification {

/**
 * Image pyramid based feature extractor that applies histogram equalization to image patches.
 */
class HistEqFeatureExtractor : public ImagePyramidFeatureExtractor {
public:

	/**
	 * Constructs a new histogram equalization feature extractor.
	 *
	 * @param[in] featureSize The size of the image patch used for feature extraction.
	 * @param[in] scaleFactor The scale factor between two levels of the pyramid.
	 * @param[in] minHeight The minimum height of feature patches relative to the image height.
	 * @param[in] maxHeight The maximum height of feature patches relative to the image height.
	 */
	HistEqFeatureExtractor(Size featureSize = Size(20, 20), double scaleFactor = 1.2,
			double minHeight = 0, double maxHeight = 1);

	virtual ~HistEqFeatureExtractor();

protected:

	void initScale(Mat image);

	shared_ptr<FeatureVector> extract(const Mat& patch);

private:

	Mat histEqPatch; ///< Temporary memory for the histogram equalized patch data.
};

} /* namespace tracking */
#endif /* HISTEQFEATUREEXTRACTOR_H_ */

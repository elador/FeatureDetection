/*
 * HistogramFilter.hpp
 *
 *  Created on: 13.10.2015
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_HISTOGRAMFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_HISTOGRAMFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {
namespace filtering {

/**
 * Filter that computes a sparse histogram for each pixel.
 *
 * The input image must have a depth of CV_32F and one or two channels. The first channel contains the value that
 * should be discretized into histogram bins, while the optional second channel contains the weight. The resulting
 * image has as many channels as there are bins in the histogram, with each pixel containing a full histogram
 * (with all but one or two of its values being zero).
 */
class HistogramFilter : public ImageFilter {
public:

	/**
	 * Constructs a new histogram filter.
	 *
	 * @param[in] binCount Number of bins.
	 * @param[in] upperBound Upper bound (exclusive) of the values.
	 * @param[in] circular Flag that indicates whether the values are circular (upper bound and zero are equivalent).
	 * @param[in] interpolate Flag that indicates whether to linearly interpolate between the neighboring bins.
	 */
	explicit HistogramFilter(int binCount, float upperBound = 1.0f, bool circular = false, bool interpolate = false);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	struct Bins {
		int bin1;
		int bin2;
		float weight1;
		float weight2;
	};

	/**
	 * Computes the histogram of a pixel location.
	 *
	 * @param[in,out] histogram Pointer to the (first bin of the) histogram with all bins having zero weight.
	 * @param[in] value Value at the pixel location inside the original image.
	 * @param[in] weight Corresponding weight of the value.
	 */
	void computeHistogram(float* histogram, float value, float weight) const;

	int computeBin(float value) const;

	int computeCircularBin(float value) const;

	int computeNoncircularBin(float value) const;

	// bin1 and bin2 are guaranteed to be different
	Bins computeInterpolatedBins(float value, float weight) const;

	// bin1 and bin2 are guaranteed to be different
	Bins computeCircularInterpolatedBins(float value, float weight) const;

	// bin1 and bin2 are guaranteed to be different
	Bins computeNoncircularInterpolatedBins(float value, float weight) const;

	int binCount; ///< Number of bins.
	float value2bin; ///< Factor that computes the corresponding (floating point) bin when multiplied by a value.
	bool circular; ///< Flag that indicates whether the values are circular (upper bound and zero are equivalent).
	bool interpolate; ///< Flag that indicates whether to linearly interpolate between the neighboring bins.
};

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_HISTOGRAMFILTER_HPP_ */

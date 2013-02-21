/*
 * WhiteningFilter.hpp
 *
 *  Created on: 21.02.2013
 *      Author: poschmann
 */

#ifndef WHITENINGFILTER_HPP_
#define WHITENINGFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter that approximately whitens the power spectrum of natrual images. The algorithm was taken
 * from http://sun360.csail.mit.edu/jxiao/SFMedu/SFMedu/lib/vlfeat/toolbox/imop/vl_imwhiten.m.
 */
class WhiteningFilter : public ImageFilter {
public:

	/**
	 * Constructs a new whitening filter.
	 *
	 * @param[in] alpha Decay of modulus of spectrum is assumed as 1/frequancy^alpha.
	 * @param[in] cutoffFrequency The cut-off frequency of the additional low-pass filter (only applied when greater than zero).
	 */
	WhiteningFilter(float alpha = 1, float cutoffFrequency = 0.390625);

	~WhiteningFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered);

	void applyInPlace(Mat& image);

private:

	/**
	 * Provides the whitening filter for the image in the frequency domain.
	 *
	 * @param[in] width The width of the image (and filter).
	 * @param[in] height The height of the image (and filter).
	 * @return The filter of the given size.
	 */
	const Mat& getFilter(int width, int height);

	float alpha;           ///< Decay of modulus of spectrum is assumed as 1/frequancy^alpha.
	float cutoffFrequency; ///< The cut-off frequency of the additional low-pass filter (only applied when greater than zero).
	Mat filter;       ///< The current filter.
	Mat floatImage;   ///< Temporal buffer for the float conversion of the image.
	Mat fourierImage; ///< Temporal buffer for the Fourier transformation of the image.
};

} /* namespace imageprocessing */
#endif /* WHITENINGFILTER_HPP_ */

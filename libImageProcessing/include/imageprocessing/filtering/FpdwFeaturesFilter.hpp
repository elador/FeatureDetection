/*
 * FpdwFeaturesFilter.hpp
 *
 *  Created on: 09.02.2016
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_FPDWFEATURESFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_FPDWFEATURESFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/filtering/BgrToLuvConverter.hpp"
#include "imageprocessing/filtering/GradientFilter.hpp"
#include "imageprocessing/filtering/GradientMagnitudeFilter.hpp"
#include "imageprocessing/filtering/GradientOrientationFilter.hpp"

namespace imageprocessing {
namespace filtering {

/**
 * Image filter that computes per pixel features similar to the ones used by the "Fastest Pedestrian Detector in the West" [1].
 *
 * Expects a BGR image of type CV_8UC3 or CV_32FC3 and creates an image of depth CV_32F with ten channels. The first six
 * channels are bins of an unsigned gradient histogram, followed by the normalized gradient magnitude and CIE L*u*v*
 * color channels. The gradiens are computed on the original BGR image or on a grayscale version of that image. This is
 * a deviation from [1], where the gradients are computed on the smoothed L*u*v* image (according to their code [2]).
 * Additionally, the computation of the L*u*v* color space of OpenCV differs from the one used in [2].
 *
 * [1] Dollár et al., The Fastest Pedestrian Detector in the West, BMVC, 2010.
 * [2] https://github.com/pdollar/toolbox/tree/master/
 */
class FpdwFeaturesFilter : public ImageFilter {
public:

	/**
	 * Constructs a new FPDW features filter.
	 *
	 * @param[in] fastGradient Flag that indicates whether to compute the gradient on the grayscale image instead of all color channels.
	 * @param[in] interpolate Flag that indicates whether to linearly interpolate between the neighboring bins.
	 * @param[in] normalizationRadius Radius of the magnitude normalization.
	 * @param[in] normalizationConstant Small normalization constant to prevent division by zero.
	 */
	FpdwFeaturesFilter(bool fastGradient, bool interpolate, int normalizationRadius = 5, double normalizationConstant = 0.01);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	struct Bins {
		int bin1;
		int bin2;
		float weight1;
		float weight2;
	};

	struct LutEntry {
		Bins fullBins;
		float magnitude;
	};

	void createGradientLut();

	cv::Mat computeGradientImage(const cv::Mat& bgrImage) const;

	cv::Mat reduceToStrongestGradient(const cv::Mat& gradientImage) const;

	void computeDescriptorImage(const cv::Mat& bgrImage, const cv::Mat& singleGradientImage,
			const cv::Mat& magnitudeImage, cv::Mat& descriptorImage) const;

	float computeOrientation(float gradientX, float gradientY) const;

	float computeMagnitude(float gradientX, float gradientY) const;

	float computeSquaredMagnitude(float gradientX, float gradientY) const;

	int computeBin(float value) const;

	// bin1 and bin2 are guaranteed to be different
	Bins computeInterpolatedBins(float value, float weight) const;

	int computeHalfBin(int bin) const;

	GrayscaleFilter grayConverter; ///< Image filter that converts a BGR image to a grayscale image.
	BgrToLuvConverter luvConverter; ///< Image filter that converts a BGR image to a LUV image.
	GradientFilter gradientFilter; ///< Image filter that computes image gradients.
	GradientMagnitudeFilter magnitudeFilter; ///< Image filter that computes the gradient magnitudes given image gradients.
	GradientOrientationFilter orientationFilter; ///< Image filter that is used to compute gradient orientations.
	bool fastGradient; ///< Flag that indicates whether to compute the gradient on the grayscale image instead of all color channels.
	int binCount; ///< Number of bins of the gradient histogram.
	float value2bin; ///< Factor that computes the corresponding (floating point) bin when multiplied by a value.
	bool interpolate; ///< Flag that indicates whether to linearly interpolate between the neighboring bins.
	std::array<LutEntry, 256 * 256> binLut; ///< Look-up table for bin indices and weights of CV_8U gradient images.
};

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_FPDWFEATURESFILTER_HPP_ */

/*
 * GradientMagnitudeFilter.hpp
 *
 *  Created on: 07.01.2016
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_FILTERING_GRADIENTMAGNITUDEFILTER_HPP_
#define IMAGEPROCESSING_FILTERING_GRADIENTMAGNITUDEFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/filtering/TriangularConvolutionFilter.hpp"

namespace imageprocessing {
namespace filtering {

/**
 * Filter that computes the gradient magnitude given an image containing gradients for x and y.
 *
 * The input image is expected to have an even number of channels, where the odd channels are gradients of x and
 * the even channels are gradients of y. If the input image has more than two channels, meaning that the gradients
 * were computed over several channels, then each of the gradient pairs are used to create one magnitude image
 * channel. The depth of the input image can be CV_8U (with a value of 127 indicating no gradient at all) or CV_32F.
 *
 * The resulting image is of depth CV_32F with half as many channels as the input image, where each channel is the
 * magnitude of a gradient channel. The magnitude may be normalized. In that case, it is divided by the smoothed
 * magnitude plus a small constant. The smoothed magnitude is obtained by convolving with a triangular filter
 * kernel. The small constant prevents division by zero and additionally reduces the impact of small magnitudes. The
 * bigger the constant gets, the more smaller gradients are suppressed.
 *
 * After normalization, magnitudes that are equal to their vicinity will have a value around one, whereas magnitudes
 * stronger than their vicinity will be bigger than one and magnitudes weaker than their vicinity will be less than
 * one.
 */
class GradientMagnitudeFilter : public ImageFilter {
public:

	/**
	 * Constructs a new gradient magnitude filter.
	 *
	 * @param[in] normalizationRadius Radius of the magnitude's smoothing filter (size is 2 * radius + 1), 0 to prevent normalization.
	 * @param[in] normalizationConstant Small constant that is added to the smoothed magnitude to prevent division by zero.
	 */
	explicit GradientMagnitudeFilter(int normalizationRadius = 0, double normalizationConstant = 0.005);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	/**
	 * Determines whether the gradient magnitudes are normalized. This depends on the normalization radius given at
	 * construction - if it is zero, then there is no normalization, otherwise there is.
	 *
	 * @return True if the gradient magnitudes are normalized, false otherwise.
	 */
	bool isNormalizing() const;

	/**
	 * Computes a smoothed version of the given magnitudes, which then are used for normalization. The smoothed magnitudes
	 * are offset by the constant given at construction time to ensure they are bigger than zero. To normalize the gradient
	 * magnitudes, they have to be divided element-wise by the computed normalizers.
	 *
	 * @param[in] magnitude Image containing gradient magnitudes per pixel.
	 * @return Image containing the normalizers (smoothed gradient magnitudes) per pixel.
	 */
	cv::Mat computeNormalizer(const cv::Mat& magnitude) const;

private:

	void createMagnitudeLut();

	void computeMagnitudeImage(const cv::Mat& gradients, cv::Mat& magnitude) const;

	template<typename T>
	void computeMagnitudeImage_(const cv::Mat& gradients, cv::Mat& magnitude) const;

	template<typename T>
	void computeMagnitudesImage_(const cv::Mat& gradients, cv::Mat& magnitudes) const;

	template<typename T>
	float getMagnitude(T gradient) const;

public:

	float computeMagnitude(float gradientX, float gradientY) const;

	float computeSquaredMagnitude(float gradientX, float gradientY) const;

private:

	void normalizeMagnitude(cv::Mat& magnitude) const;

	void normalizeMagnitude(cv::Mat& magnitude, const cv::Mat& smoothedMagnitude) const;

	void normalizeMagnitude1(cv::Mat& magnitude, const cv::Mat& smoothedMagnitude) const;

	void normalizeMagnitudeN(cv::Mat& magnitude, const cv::Mat& smoothedMagnitude) const;

	std::array<float, 256 * 256> magnitudeLut; ///< Look-up table for gradient magnitude of CV_8U gradient images.
	TriangularConvolutionFilter smoothingFilter; ///< 2D triangular convolution filter for smoothing the magnitude.
};

template<typename T>
void GradientMagnitudeFilter::computeMagnitudeImage_(const cv::Mat& gradients, cv::Mat& magnitude) const {
	for (int row = 0; row < gradients.rows; ++row) {
		for (int col = 0; col < gradients.cols; ++col) {
			magnitude.at<float>(row, col) = getMagnitude(gradients.at<T>(row, col));
		}
	}
}

template<typename T>
void GradientMagnitudeFilter::computeMagnitudesImage_(const cv::Mat& gradients, cv::Mat& magnitudes) const {
	int channels = gradients.channels() / 2;
	for (int row = 0; row < gradients.rows; ++row) {
		for (int col = 0; col < gradients.cols; ++col) {
			const T* gradientValues = gradients.ptr<T>(row, col);
			float* magnitudeValues = magnitudes.ptr<float>(row, col);
			for (int ch = 0; ch < channels; ++ch)
				magnitudeValues[ch] = getMagnitude(gradientValues[ch]);
		}
	}
}

template<>
inline float GradientMagnitudeFilter::getMagnitude(ushort gradient) const {
	return magnitudeLut[gradient];
}

template<>
inline float GradientMagnitudeFilter::getMagnitude(cv::Vec2f gradient) const {
	return computeMagnitude(gradient[0], gradient[1]);
}

} /* namespace filtering */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_FILTERING_GRADIENTMAGNITUDEFILTER_HPP_ */

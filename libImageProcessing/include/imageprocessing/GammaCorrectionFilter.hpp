/*
 * GammaCorrectionFilter.hpp
 *
 *  Created on: 30.07.2013
 *      Author: poschmann
 */

#ifndef GAMMACORRECTIONFILTER_HPP_
#define GAMMACORRECTIONFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter that does a gamma correction. The image must have one channel and be of type CV_8U, CV_32F or CV_64F.
 * The values are supposed to be between zero and one (except for CV_8U). The output image type will be the same as
 * the input image type.
 *
 * Takes the intensity value of each pixel to the power of gamma. In case of the image type being CV_8U, the values
 * will be divided by 255 before and multiplied by 255 afterwards.
 */
class GammaCorrectionFilter : public ImageFilter {
public:

	/**
	 * Constructs a new gamma correction filter.
	 *
	 * @param[in] gamma The power of the intensity. Must be greater than zero.
	 */
	explicit GammaCorrectionFilter(double gamma);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	void applyInPlace(cv::Mat& image) const;

private:

	template<class T>
	cv::Mat applyGammaCorrection(const cv::Mat& image, cv::Mat& filtered) const {
		int rows = image.rows;
		int cols = image.cols;
		filtered.create(image.rows, image.cols, image.type());
		if (image.isContinuous() && filtered.isContinuous()) {
			cols *= rows;
			rows = 1;
		}
		for (int row = 0; row < rows; ++row) {
			const T* originalRow = image.ptr<T>(0);
			T* filteredRow = filtered.ptr<T>(0);
			for (int col = 0; col < cols; ++col)
				filteredRow[col] = pow(originalRow[col], gamma);
		}
		return filtered;
	}

	double gamma; ///< The power of the intensity. Must be greater than zero.
	cv::Mat lut; ///< Look-up table of gamma corrected uchar values.
};

} /* namespace imageprocessing */
#endif /* GAMMACORRECTIONFILTER_HPP_ */

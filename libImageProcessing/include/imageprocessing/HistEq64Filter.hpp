/*
 * HistEq64Filter.hpp
 *
 *  Created on: 26.02.2013
 *      Author: Patrik Huber
 */

#ifndef HISTEQ64FILTER_HPP_
#define HISTEQ64FILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter that equalizes the histogram using our old implementation. It uses 64 bins and corresponds to the matlab
 * implementation used in the training of WVM/SVM's. The images must be of type 8-bit single channel (CV_8U / CV_8UC1).
 * The output image will be CV_8U as well.
 */
class HistEq64Filter : public ImageFilter {
public:

	/**
	 * Constructs a new histogram equalization filter for a certain detector window size.
	 *
	 * @param[in] detectorSizeX The width of the detector
	 * @param[in] detectorSizeY Image filter that is applied to each patch before the transformation.
	 */
	HistEq64Filter();

	~HistEq64Filter();

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

	void applyInPlace(cv::Mat& image) const;

private:

	unsigned char* LUTbin; ///< lookup table for the histogram equalization
};

} /* namespace imageprocessing */
#endif /* HISTEQ64FILTER_HPP_ */

/*
 * GradientFeatureExtractor.hpp
 *
 *  Created on: 23.05.2013
 *      Author: poschmann
 */

#ifndef GRADIENTFEATUREEXTRACTOR_HPP_
#define GRADIENTFEATUREEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include <vector>

using cv::Rect_;
using std::vector;

namespace imageprocessing {

class GradientFeatureExtractor : public FeatureExtractor {
public:

	GradientFeatureExtractor();

	~GradientFeatureExtractor();

	/**
	 * Adds a new filter that is applied to the image.
	 *
	 * @param[in] filter The new image filter.
	 */
	void addImageFilter(shared_ptr<ImageFilter> filter);

	void update(const Mat& image);

	void update(shared_ptr<VersionedImage> image);

	shared_ptr<Patch> extract(int x, int y, int width, int height) const;

private:

	int getSum(int x, int y) const;

	double getAreaSum(float x1, float y1, float x2, float y2) const;

	/**
	 * Computes the value at (xa, ya) using bilinear interpolation between the points (x1, y1), (x1, y2), (x2, y1) and (x2, y2).
	 *
	 * @param[in] x1 Location x1.
	 * @param[in] x2 Location x2.
	 * @param[in] y1 Location y1.
	 * @param[in] y2 Location y2.
	 * @param[in] f11 Value at (x1, y1).
	 * @param[in] f12 Value at (x1, y2).
	 * @param[in] f21 Value at (x2, y1).
	 * @param[in] f22 Value at (x2, y2).
	 * @param[in] xa Location xa.
	 * @param[in] ya Location ya.
	 * @return The interpolated value at (xa, ya).
	 */
	double interpolate(float x1, float x2, float y1, float y2, double f11, double f12, double f21, double f22, float xa, float ya) const;

	/**
	 * Computes the value at xa using linear interpolation between the points (x1, fx1) and (x2, fx2).
	 *
	 * @param[in] x1 Location x1.
	 * @param[in] x2 Location x2.
	 * @param[in] fx1 Value at x1.
	 * @param[in] fx2 Value at x2.
	 * @param[in] xa Location xa.
	 * @return The interpolated value xa.
	 */
	double interpolate(float x1, float x2, double fx1, double fx2, float xa) const;

	GrayscaleFilter imageFilter; ///< Filter that is applied to the image.
	int version; ///< The version number.
	Mat grayscaleImage; /// The grayscale image.
	Mat integralImage; ///< The integral image.
	vector<Rect_<float>> areas; ///< The areas to compute the gradients of.
};

} /* namespace imageprocessing */
#endif /* GRADIENTFEATUREEXTRACTOR_HPP_ */

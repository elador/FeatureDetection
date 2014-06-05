/*
 * HaarFeatureFilter.hpp
 *
 *  Created on: 16.07.2013
 *      Author: poschmann
 */

#ifndef HAARFEATUREFILTER_HPP_
#define HAARFEATUREFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Image filter that expects an integral image and computes Haar features at certain positions. The output
 * is a vector containing the feature values. Each feature value is computed at a square area that is
 * partitioned into two, three or four different rectangular areas. Each of those rectangular areas has a
 * weight assigned to it that is used for summing up the values.
 */
class HaarFeatureFilter : public ImageFilter {
public:

	static const int TYPE_2RECTANGLE = 1; ///< Haar feature type with two equal-sized rectangles next to each other (2x1 / 1x2).

	static const int TYPE_3RECTANGLE = 2; ///< Haar feature type with three equal-sized rectangles next to each other (3x1 / 1x3).

	static const int TYPE_4RECTANGLE = 4; ///< Haar feature type with four equal-sized rectangles next to each other (2x2).

	static const int TYPE_CENTER_SURROUND = 8; ///< Haar feature type with two rectangles (one big, one small), both having the same center.

	static const int TYPES_ALL = TYPE_2RECTANGLE | TYPE_3RECTANGLE | TYPE_4RECTANGLE | TYPE_CENTER_SURROUND; ///< All kinds of Haar features.

	/**
	 * Constructs a new Haar feature filter computing all feature types with sizes 0.2 and 0.4 on a 5x5 grid.
	 */
	HaarFeatureFilter();

	/**
	 * Constructs a new Haar feature filter with the features being on an equal distant square grid.
	 *
	 * @param[in] sizes The sizes of the feature relative to the image size.
	 * @param[in] count The length of the square grid.
	 * @param[in] types The feature types.
	 */
	HaarFeatureFilter(std::vector<float> sizes, unsigned int count, int types = TYPES_ALL);

	/**
	 * Constructs a new Haar feature filter with the features being on an equal-distant grid.
	 *
	 * @param[in] sizes The sizes of the feature relative to the image size.
	 * @param[in] xCount The horizontal length of the grid.
	 * @param[in] yCount The vertical length of the grid.
	 * @param[in] types The feature types.
	 */
	HaarFeatureFilter(std::vector<float> sizes, unsigned int xCount, unsigned int yCount, int types = TYPES_ALL);

	/**
	 * Constructs a new Haar feature filter with the features being on a square grid.
	 *
	 * @param[in] sizes The sizes of the feature relative to the image size.
	 * @param[in] coords The x and y coordinates of the grid points relative to the image size.
	 * @param[in] types The feature types.
	 */
	HaarFeatureFilter(std::vector<float> sizes, std::vector<float> coords, int types = TYPES_ALL);

	/**
	 * Constructs a new Haar feature filter with the features being on a grid.
	 *
	 * @param[in] sizes The sizes of the feature relative to the image size.
	 * @param[in] xs The x coordinates of the grid points relative to the image size.
	 * @param[in] ys The y coordinates of the grid points relative to the image size.
	 * @param[in] types The feature types.
	 */
	HaarFeatureFilter(std::vector<float> sizes, std::vector<float> xs, std::vector<float> ys, int types = TYPES_ALL);

	using ImageFilter::applyTo;

	cv::Mat applyTo(const cv::Mat& image, cv::Mat& filtered) const;

private:

	/**
	 * Builds the features being on an equal-distant grid.
	 *
	 * @param[in] sizes The sizes of the feature relative to the image size.
	 * @param[in] xCount The horizontal length of the grid.
	 * @param[in] yCount The vertical length of the grid.
	 * @param[in] types The feature types.
	 */
	void buildFeatures(std::vector<float> sizes, unsigned int xCount, unsigned int yCount, int types);

	/**
	 * Builds the features being on a grid.
	 *
	 * @param[in] sizes The sizes of the feature relative to the image size.
	 * @param[in] xs The x coordinates of the grid points relative to the image size.
	 * @param[in] ys The y coordinates of the grid points relative to the image size.
	 * @param[in] types The feature types.
	 */
	void buildFeatures(std::vector<float> s, std::vector<float> x, std::vector<float> y, int types);

	/**
	 * Single haar feature that describes how to compute the feature value.
	 */
	struct HaarFeature {
		std::vector<cv::Rect_<float>> rects; ///< The rectangular areas to sum the intensity values over.
		std::vector<float> weights;          ///< The weights of the areas.
		float factor; ///< The factor of the feature.
		float area;   ///< The area of the feature (sum of area of all rects).
	};

	std::vector<HaarFeature> features; ///< The features.
};

} /* namespace imageprocessing */
#endif /* HAARFEATUREFILTER_HPP_ */

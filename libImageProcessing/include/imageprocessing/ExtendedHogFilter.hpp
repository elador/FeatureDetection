/*
 * ExtendedHogFilter.hpp
 *
 *  Created on: 01.08.2013
 *      Author: poschmann
 */

#ifndef EXTENDEDHOGFILTER_HPP_
#define EXTENDEDHOGFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"
#include <vector>

using std::vector;

namespace imageprocessing {

/**
 * Filter that builds exteded HOG feature vectors. The extended HOG features were described in [1]. HOG descriptors are
 * computed for the inner cells only, so the cells at the border are only used for computing the gradient energies.
 *
 * The input is an image with depth CV_8U containing bin information (bin indices and optionally weights). It must have
 * one, two or four channels. In case of one channel, it just contains the bin index. The second channel adds a weight
 * that will be divided by 255 to be between zero and one. The third channel is another bin index and the fourth channel
 * is the corresponding weight. The output will be a row vector of type CV_32F.
 *
 * [1] Felzenszwalb et al., Object Detection with Discriminatively Trained Part-Based Models, PAMI, 2010.
 */
class ExtendedHogFilter : public ImageFilter {
public:

	/**
	 * Constructs a new extended HOG filter with square cells and blocks.
	 *
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] cellSize The preferred width and height of the cells in pixels (actual size might deviate).
	 * @param[in] interpolation Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 * @param[in] alpha Truncation threshold of the orientation bin values (applied after normalization).
	 */
	ExtendedHogFilter(unsigned int bins, int cellSize, bool interpolation, bool signedAndUnsigned, float alpha = 0.2);

	/**
	 * Constructs a new extended HOG filter.
	 *
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] cellWidth The preferred width of the cells in pixels (actual width might deviate).
	 * @param[in] cellHeight The preferred height of the cells in pixels (actual height might deviate).
	 * @param[in] interpolation Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 * @param[in] alpha Truncation threshold of the orientation bin values (applied after normalization).
	 */
	ExtendedHogFilter(unsigned int bins, int cellWidth, int cellHeight, bool interpolation, bool signedAndUnsigned, float alpha = 0.2);

	~ExtendedHogFilter();

	using ImageFilter::applyTo;

	Mat applyTo(const Mat& image, Mat& filtered) const;

	void applyInPlace(Mat& image) const;

private:

	/**
	 * Entry of the cache for the linear interpolation.
	 */
	struct CacheEntry {
		int index;    ///< First index (second one is index + 1).
		float weight; ///< Weight of the second index (weight of first index is 1 - weight).
	};

	/**
	 * Creates the cache contents if the size of the cache does not match the requested size.
	 *
	 * @param[in] cache The cache.
	 * @param[in] size The necessary size of the cache.
	 * @param[in] count The number of cells.
	 */
	void createCache(vector<CacheEntry>& cache, unsigned int size, int count) const;

	unsigned int bins; ///< The amount of bins inside the histogram.
	int cellWidth;     ///< The preferred width of the cells in pixels (actual width might deviate).
	int cellHeight;    ///< The preferred height of the cells in pixels (actual height might deviate).
	bool interpolation;     ///< Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	bool signedAndUnsigned; ///< Flag that indicates whether signed and unsigned gradients should be used.
	float alpha; ///< Truncation threshold of the orientation bin values (applied after normalization).
	mutable vector<CacheEntry> rowCache; ///< Cache for the linear interpolation of the row indices.
	mutable vector<CacheEntry> colCache; ///< Cache for the linear interpolation of the column indices.
	static const float eps; ///< The small value being added to the norms to prevent divisions by zero.
};

} /* namespace imageprocessing */
#endif /* EXTENDEDHOGFILTER_HPP_ */

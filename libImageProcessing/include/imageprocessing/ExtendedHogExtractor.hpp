/*
 * ExtendedHogExtractor.hpp
 *
 *  Created on: 01.08.2013
 *      Author: poschmann
 */

#ifndef EXTENDEDHOGEXTRACTOR_HPP_
#define EXTENDEDHOGEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include <vector>

using std::vector;

namespace imageprocessing {

/**
 * Feature extractor that builds exteded HOG feature vectors based on patches containing orientation histogram data (bin
 * indices and optionally weights) per pixel. The extended HOG features were described in [1]. HOG descriptors are computed
 * for the inner cells only, so the cells at the border are only used for computing the gradient energies.
 *
 * The patch data must be of type CV_8U and have one, two or four channels. In case of one channel, it just contains the
 * bin index. The second channel adds a weight that will be divided by 255 to be between zero and one. The third channel
 * is another bin index and the fourth channel is the corresponding weight.
 *
 * [1] Felzenszwalb et al., Object Detection with Discriminatively Trained Part-Based Models, PAMI, 2010.
 */
class ExtendedHogExtractor : public FeatureExtractor {
public:

	/**
	 * Constructs a new extended HOG feature extractor with square cells and blocks.
	 *
	 * @param[in] extractor The underlying feature extractor (has to deliver histogram bin indices per pixel with optional weight).
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] cellSize The preferred width and height of the cells in pixels (actual size might deviate).
	 * @param[in] interpolation Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 * @param[in] alpha Truncation threshold of the orientation bin values (applied after normalization).
	 */
	ExtendedHogExtractor(shared_ptr<FeatureExtractor> extractor, unsigned int bins,
			int cellSize, bool interpolation, bool signedAndUnsigned, float alpha = 0.2);

	/**
	 * Constructs a new extended HOG feature extractor.
	 *
	 * @param[in] extractor The underlying feature extractor (has to deliver histogram bin indices per pixel with optional weight).
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] cellWidth The preferred width of the cells in pixels (actual width might deviate).
	 * @param[in] cellHeight The preferred height of the cells in pixels (actual height might deviate).
	 * @param[in] interpolation Flag that indicates whether each pixel should contribute to the four cells around it using bilinear interpolation.
	 * @param[in] signedAndUnsigned Flag that indicates whether signed and unsigned gradients should be used.
	 * @param[in] alpha Truncation threshold of the orientation bin values (applied after normalization).
	 */
	ExtendedHogExtractor(shared_ptr<FeatureExtractor> extractor, unsigned int bins,
			int cellWidth, int cellHeight, bool interpolation, bool signedAndUnsigned, float alpha = 0.2);

	~ExtendedHogExtractor();

	void update(const Mat& image);

	void update(shared_ptr<VersionedImage> image);

	shared_ptr<Patch> extract(int x, int y, int width, int height) const;

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

	shared_ptr<FeatureExtractor> extractor; ///< The underlying feature extractor.
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
#endif /* EXTENDEDHOGEXTRACTOR_HPP_ */

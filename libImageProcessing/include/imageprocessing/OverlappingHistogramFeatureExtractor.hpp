/*
 * OverlappingHistogramFeatureExtractor.hpp
 *
 *  Created on: 30.05.2013
 *      Author: poschmann
 */

#ifndef OVERLAPPINGHISTOGRAMFEATUREEXTRACTOR_HPP_
#define OVERLAPPINGHISTOGRAMFEATUREEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"

namespace imageprocessing {

/**
 * Feature extractor that creates overlapping histograms. Builds upon a feature extractor that delivers patches
 * containing histogram information (bin indices and optionally weights).
 */
class OverlappingHistogramFeatureExtractor : public FeatureExtractor {
public:

	/**
	 * Normalization method.
	 */
	enum Normalization { NONE, L2NORM, L2HYS, L1NORM, L1SQRT };

	/**
	 * Constructs a new overlapping histogram feature extractor with square cells and blocks.
	 *
	 * @param[in] extractor The underlying feature extractor (has to deliver histogram bin indices per pixel with optional weight).
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] cellSize The preferred width and height of the cells in pixels (actual size might deviate).
	 * @param[in] blockSize The width and height of the blocks in cells.
	 * @param[in] normalization The normalization method of the block histograms.
	 */
	OverlappingHistogramFeatureExtractor(shared_ptr<FeatureExtractor> extractor, unsigned int bins,
			int cellSize, int blockSize, Normalization normalization = L2NORM);

	/**
	 * Constructs a new overlapping histogram feature extractor.
	 *
	 * @param[in] extractor The underlying feature extractor (has to deliver histogram bin indices per pixel with optional weight).
	 * @param[in] bins The amount of bins inside the histogram.
	 * @param[in] cellWidth The preferred width of the cells in pixels (actual width might deviate).
	 * @param[in] cellHeight The preferred height of the cells in pixels (actual height might deviate).
	 * @param[in] blockWidth The width of the blocks in cells.
	 * @param[in] blockHeight The height of the blocks in cells.
	 * @param[in] normalization The normalization method of the block histograms.
	 */
	OverlappingHistogramFeatureExtractor(shared_ptr<FeatureExtractor> extractor, unsigned int bins,
			int cellWidth, int cellHeight, int blockWidth, int blockHeight, Normalization normalization);

	~OverlappingHistogramFeatureExtractor();

	void update(const Mat& image);

	void update(shared_ptr<VersionedImage> image);

	shared_ptr<Patch> extract(int x, int y, int width, int height) const;

private:

	/**
	 * Normalizes the given histogram according to L2-norm.
	 *
	 * @param[in,out] histogram The histogram that should be normalized.
	 */
	void normalizeL2(Mat& histogram) const;

	/**
	 * Normalizes the given histogram according to L2-hys (L2-norm followed by clipping (limit maximum values to 0.2)
	 * and renormalizing).
	 *
	 * @param[in,out] histogram The histogram that should be normalized.
	 */
	void normalizeL2Hys(Mat& histogram) const;

	/**
	 * Normalizes the given histogram according to L1-norm.
	 *
	 * @param[in,out] histogram The histogram that should be normalized.
	 */
	void normalizeL1(Mat& histogram) const;

	/**
	 * Normalizes the given histogram according to L1-sqrt (L1-norm followed by computing the square root over all elements).
	 *
	 * @param[in,out] histogram The histogram that should be normalized.
	 */
	void normalizeL1Sqrt(Mat& histogram) const;

	static const float eps; ///< The small value being added to the norms to prevent divisions by zero.

	shared_ptr<FeatureExtractor> extractor; ///< The underlying feature extractor.
	unsigned int bins; ///< The amount of bins inside the histogram.
	int cellWidth;     ///< The preferred width of the cells in pixels (actual width might deviate).
	int cellHeight;    ///< The preferred height of the cells in pixels (actual height might deviate).
	int blockWidth;    ///< The width of the blocks in cells.
	int blockHeight;   ///< The height of the blocks in cells.
	Normalization normalization; ///< The normalization method of the block histograms.
};

} /* namespace imageprocessing */
#endif /* OVERLAPPINGHISTOGRAMFEATUREEXTRACTOR_HPP_ */

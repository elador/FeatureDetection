/*
 * DirectImageExtractor.hpp
 *
 *  Created on: 30.06.2013
 *      Author: ex-ratt
 */

#ifndef HOGIMAGEEXTRACTOR_HPP_
#define HOGIMAGEEXTRACTOR_HPP_

#include "imageprocessing/HistogramFeatureExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/GrayscaleFilter.hpp"
#include "imageprocessing/GradientFilter.hpp"
#include <memory>
#include <array>

using std::make_shared;
using std::array;
using std::vector;

namespace imageprocessing {

class HogImageExtractor : public HistogramFeatureExtractor {
public:

	HogImageExtractor(unsigned int bins, double offset, bool signedGradients, int cellCount, int blockSize, bool combineHistograms = false, Normalization normalization = NONE);

	~HogImageExtractor();

	void update(const Mat& image);

	void update(shared_ptr<VersionedImage> image);

	shared_ptr<Patch> extract(int x, int y, int width, int height) const;

	void addImageFilter(shared_ptr<ImageFilter> filter) {}
	void addLayerFilter(shared_ptr<ImageFilter> filter) {}
	void addPatchFilter(shared_ptr<ImageFilter> filter) {}

	/**
	 * @return The width of the image data of the extracted patches.
	 */
	int getPatchWidth() const {
		return 20;
	}

	/**
	 * @param[in] width The new width of the image data of the extracted patches.
	 */
	void setPatchWidth(int width) {}

	/**
	 * @return The height of the image data of the extracted patches.
	 */
	int getPatchHeight() const {
		return 20;
	}

	/**
	 * @param[in] height The new height of the image data of the extracted patches.
	 */
	void setPatchHeight(int height) {}

	/**
	 * Changes the size of the image data of the extracted patches.
	 *
	 * @param[in] width The new width of the image data of the extracted patches.
	 * @param[in] height The new height of the image data of the extracted patches.
	 */
	void setPatchSize(int width, int height) {}

	mutable Mat hogImage;

private:

	struct BinEntry {
		int bin1, bin2;
		float weight1, weight2;
	};

	void buildHistogram(const Mat& image);

	int version;                            ///< The version number.
	Mat image;
	GrayscaleFilter filter;
	GradientFilter gradientFilter;
	unsigned int bins; ///< The amount of bins.
	double offset; ///< Lower boundary of the first bin.
	array<BinEntry, 256 * 256> binData; ///< The look-up tables of the bin codes, the gradient codes are used as the index.

	vector<Mat> integralHistogram;

	int cellCols;     ///< The preferred width of the cells in pixels (actual width might deviate).
	int cellRows;    ///< The preferred height of the cells in pixels (actual height might deviate).
	int blockWidth;    ///< The width of the blocks in cells.
	int blockHeight;   ///< The height of the blocks in cells.
	bool combineHistograms; ///< Flag that indicates whether the histograms of the cells should be added up to form the block histogram.
};

} /* namespace imageprocessing */
#endif /* HOGIMAGEEXTRACTOR_HPP_ */

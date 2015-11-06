/*
 * FilteringPyramidFeatureExtractor.hpp
 *
 *  Created on: 22.03.2013
 *      Author: poschmann
 */

#ifndef FILTERINGPYRAMIDFEATUREEXTRACTOR_HPP_
#define FILTERINGPYRAMIDFEATUREEXTRACTOR_HPP_

#include "imageprocessing/PyramidFeatureExtractor.hpp"
#include "imageprocessing/ChainedFilter.hpp"
#include "imageprocessing/Patch.hpp"

namespace imageprocessing {

/**
 * Pyramid feature extractor that uses another pyramid feature extractor and applies additional filters to the extracted patches.
 */
class FilteringPyramidFeatureExtractor : public PyramidFeatureExtractor {
public:

	/**
	 * Constructs a new filtering pyramid feature extractor.
	 *
	 * @param[in] extractor The underlying pyramid feature extractor.
	 */
	FilteringPyramidFeatureExtractor(std::shared_ptr<PyramidFeatureExtractor> extractor) :
		extractor(extractor), patchFilter(std::make_shared<ChainedFilter>()) {}

	/**
	 * Adds a new filter that is applied to the patches.
	 *
	 * @param[in] filter The new patch filter.
	 */
	void addPatchFilter(std::shared_ptr<ImageFilter> filter) {
		patchFilter->add(filter);
	}

	using FeatureExtractor::update;

	void update(std::shared_ptr<VersionedImage> image) {
		extractor->update(image);
	}

	std::shared_ptr<Patch> extract(int x, int y, int width, int height) const {
		std::shared_ptr<Patch> patch = extractor->extract(x, y, width, height);
		if (patch)
			patchFilter->applyInPlace(patch->getData());
		return patch;
	}

	std::vector<std::shared_ptr<Patch>> extract(int stepX, int stepY, cv::Rect roi = cv::Rect(),
			int firstLayer = -1, int lastLayer = -1, int stepLayer = 1) const {
		std::vector<std::shared_ptr<Patch>> patches = extractor->extract(stepX, stepY, roi, firstLayer, lastLayer, stepLayer);
		for (std::shared_ptr<Patch>& patch : patches)
			patchFilter->applyInPlace(patch->getData());
		return patches;
	}

	std::shared_ptr<Patch> extract(int layer, int x, int y) const {
		std::shared_ptr<Patch> patch = extractor->extract(layer, x, y);
		if (patch)
			patchFilter->applyInPlace(patch->getData());
		return patch;
	}

	int getLayerIndex(int width, int height) const {
		return extractor->getLayerIndex(width, height);
	}

	double getMinScaleFactor() const {
		return extractor->getMinScaleFactor();
	}

	double getMaxScaleFactor() const {
		return extractor->getMaxScaleFactor();
	}

	double getIncrementalScaleFactor() const {
		return extractor->getIncrementalScaleFactor();
	}

	cv::Size getPatchSize() const {
		return extractor->getPatchSize();
	}

	cv::Size getImageSize() const {
		return extractor->getImageSize();
	}

	std::vector<std::pair<int, double>> getLayerScales() const {
		return extractor->getLayerScales();
	}

	std::vector<cv::Size> getLayerSizes() const {
		return extractor->getLayerSizes();
	}

	std::vector<cv::Size> getPatchSizes() const {
		return extractor->getPatchSizes();
	}

private:

	std::shared_ptr<PyramidFeatureExtractor> extractor; ///< The underlying feature extractor.
	std::shared_ptr<ChainedFilter> patchFilter;         ///< Filter that is applied to the patches.
};

} /* namespace imageprocessing */
#endif /* FILTERINGPYRAMIDFEATUREEXTRACTOR_HPP_ */

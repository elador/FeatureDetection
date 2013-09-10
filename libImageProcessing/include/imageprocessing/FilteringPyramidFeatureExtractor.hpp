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
#include "boost/iterator/indirect_iterator.hpp"

using boost::indirect_iterator;
using std::make_shared;

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
	FilteringPyramidFeatureExtractor(shared_ptr<PyramidFeatureExtractor> extractor) :
		extractor(extractor), patchFilter(make_shared<ChainedFilter>()) {}

	~FilteringPyramidFeatureExtractor() {}

	/**
	 * Adds a new filter that is applied to the patches.
	 *
	 * @param[in] filter The new patch filter.
	 */
	void addPatchFilter(shared_ptr<ImageFilter> filter) {
		patchFilter->add(filter);
	}

	void update(const Mat& image) {
		extractor->update(image);
	}

	void update(shared_ptr<VersionedImage> image) {
		extractor->update(image);
	}

	shared_ptr<Patch> extract(int x, int y, int width, int height) const {
		shared_ptr<Patch> patch = extractor->extract(x, y, width, height);
		if (patch)
			patchFilter->applyInPlace(patch->getData());
		return patch;
	}

	vector<shared_ptr<Patch>> extract(int stepX, int stepY, Rect roi = Rect(),
			int firstLayer = -1, int lastLayer = -1, int stepLayer = 1) const {
		vector<shared_ptr<Patch>> patches = extractor->extract(stepX, stepY, roi, firstLayer, lastLayer, stepLayer);
		for (shared_ptr<Patch>& patch : patches)
			patchFilter->applyInPlace(patch->getData());
		return patches;
	}

	shared_ptr<Patch> extract(int layer, int x, int y) const {
		shared_ptr<Patch> patch = extractor->extract(layer, x, y);
		if (patch)
			patchFilter->applyInPlace(patch->getData());
		return patch;
	}

	Rect getCenterRoi(const Rect& roi) const {
		return extractor->getCenterRoi(roi);
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

	Size getPatchSize() const {
		return extractor->getPatchSize();
	}

	Size getImageSize() const {
		return extractor->getImageSize();
	}

	vector<Size> getLayerSizes() const {
		return extractor->getLayerSizes();
	}

	vector<Size> getPatchSizes() const {
		return extractor->getPatchSizes();
	}

private:

	shared_ptr<PyramidFeatureExtractor> extractor; ///< The underlying feature extractor.
	shared_ptr<ChainedFilter> patchFilter;         ///< Filter that is applied to the patches.
};

} /* namespace imageprocessing */
#endif /* FILTERINGPYRAMIDFEATUREEXTRACTOR_HPP_ */

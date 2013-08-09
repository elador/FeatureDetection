/*
 * FilteringFeatureExtractor.hpp
 *
 *  Created on: 20.03.2013
 *      Author: poschmann
 */

#ifndef FILTERINGFEATUREEXTRACTOR_HPP_
#define FILTERINGFEATUREEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/ChainedFilter.hpp"
#include "imageprocessing/Patch.hpp"

using std::make_shared;

namespace imageprocessing {

/**
 * Feature extractor that uses another feature extractor and applies additional filters to the extracted patches.
 */
class FilteringFeatureExtractor : public FeatureExtractor {
public:

	/**
	 * Constructs a new filtering feature extractor.
	 *
	 * @param[in] extractor The underlying feature extractor.
	 */
	explicit FilteringFeatureExtractor(shared_ptr<FeatureExtractor> extractor) :
			extractor(extractor), patchFilter(make_shared<ChainedFilter>()) {}

	~FilteringFeatureExtractor() {}

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

private:

	shared_ptr<FeatureExtractor> extractor; ///< The underlying feature extractor.
	shared_ptr<ChainedFilter> patchFilter;  ///< Filter that is applied to the patches.
};

} /* namespace imageprocessing */
#endif /* FILTERINGFEATUREEXTRACTOR_HPP_ */

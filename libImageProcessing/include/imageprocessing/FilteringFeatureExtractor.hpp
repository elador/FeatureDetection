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
	explicit FilteringFeatureExtractor(std::shared_ptr<FeatureExtractor> extractor) :
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

private:

	std::shared_ptr<FeatureExtractor> extractor; ///< The underlying feature extractor.
	std::shared_ptr<ChainedFilter> patchFilter;  ///< Filter that is applied to the patches.
};

} /* namespace imageprocessing */
#endif /* FILTERINGFEATUREEXTRACTOR_HPP_ */

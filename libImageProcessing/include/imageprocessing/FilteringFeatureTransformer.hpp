/*
 * FilteringFeatureTransformer.hpp
 *
 *  Created on: 19.02.2013
 *      Author: poschmann
 */

#ifndef FILTERINGFEATURETRANSFORMER_HPP_
#define FILTERINGFEATURETRANSFORMER_HPP_

#include "imageprocessing/FeatureTransformer.hpp"
#include "imageprocessing/MultipleImageFilter.hpp"
#include <memory>

using std::shared_ptr;
using std::make_shared;

namespace imageprocessing {

/**
 * Feature transformer that applies image filters to an image patch before the transformation.
 */
class FilteringFeatureTransformer : public FeatureTransformer {
public:

	/**
	 * Constructs a new feature transformer with an empty multiple image filter that is applied to the patches.
	 *
	 * @param[in] transformer Feature transformer that is applied to the filtered patch.
	 */
	explicit FilteringFeatureTransformer(shared_ptr<FeatureTransformer> transformer) :
			transformer(transformer), filter(make_shared<MultipleImageFilter>()) {}

	/**
	 * Constructs a new feature transformer with the given multiple image filter that is applied to the patches.
	 *
	 * @param[in] transformer Feature transformer that is applied to the filtered patch.
	 * @param[in] filter Image filter that is applied to each patch before the transformation.
	 */
	explicit FilteringFeatureTransformer(shared_ptr<FeatureTransformer> transformer, shared_ptr<MultipleImageFilter> filter) :
					transformer(transformer), filter(filter) {}

	virtual ~FilteringFeatureTransformer() {}

	/**
	 * Extracts the feature vector based on an image patch.
	 *
	 * @param[in] patch The image patch.
	 * @return A row vector containing the feature values.
	 */
	void transform(Mat& patch) const {
		filter->applyTo(patch, patch);
		transformer->transform(patch);
	}

	/**
	 * Adds a new image filter that is applied to patches after the currently existing filters.
	 *
	 * @param[in] filter The new image filter.
	 */
	void add(shared_ptr<ImageFilter> filter) {
		this->filter->add(filter);
	}

private:

	shared_ptr<FeatureTransformer> transformer;  ///< Feature transformer that is applied to the filtered patch.
	shared_ptr<MultipleImageFilter> filter;      ///< Image filter that is applied to each patch before the transformation.
};

} /* namespace imageprocessing */
#endif /* FILTERINGFEATURETRANSFORMER_HPP_ */

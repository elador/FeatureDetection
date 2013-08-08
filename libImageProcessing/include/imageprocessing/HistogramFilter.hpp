/*
 * HistogramFilter.hpp
 *
 *  Created on: 06.06.2013
 *      Author: poschmann
 */

#ifndef HISTOGRAMFILTER_HPP_
#define HISTOGRAMFILTER_HPP_

#include "imageprocessing/ImageFilter.hpp"

namespace imageprocessing {

/**
 * Filter whose results are histograms. Does only provide a method for histogram normalization, can not be instantiated
 * directly.
 */
class HistogramFilter : public ImageFilter {
public:

	/**
	 * Normalization method.
	 */
	enum class Normalization { NONE, L2NORM, L2HYS, L1NORM, L1SQRT };

	/**
	 * Constructs a new histogram feature extractor.
	 *
	 * @param[in] normalization The normalization method of the histograms.
	 */
	explicit HistogramFilter(Normalization normalization);

	virtual ~HistogramFilter();

	void applyInPlace(Mat& image) const;

protected:

	/**
	 * Normalizes the given histogram.
	 *
	 * @param[in,out] histogram The histogram that should be normalized.
	 */
	void normalize(Mat& histogram) const;

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

	Normalization normalization; ///< The normalization method of the histograms.
};

} /* namespace imageprocessing */
#endif /* HISTOGRAMFILTER_HPP_ */

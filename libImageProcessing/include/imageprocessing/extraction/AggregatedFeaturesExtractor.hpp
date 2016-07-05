/*
 * AggregatedFeaturesExtractor.hpp
 *
 *  Created on: 30.10.2015
 *      Author: poschmann
 */

#ifndef IMAGEPROCESSING_EXTRACTION_AGGREGATEDFEATURESEXTRACTOR_HPP_
#define IMAGEPROCESSING_EXTRACTION_AGGREGATEDFEATURESEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/ImageFilter.hpp"
#include "imageprocessing/ImagePyramid.hpp"
#include "imageprocessing/Patch.hpp"
#include "imageprocessing/VersionedImage.hpp"

namespace imageprocessing {
namespace extraction {

/**
 * Extractor of aggregated features.
 */
class AggregatedFeaturesExtractor : public FeatureExtractor {
public:

	/**
	 * Constructs a new aggregated features extractor based on the given feature pyramid.
	 *
	 * @param[in] featurePyramid Image pyramid whose images contain the aggregated feature cells.
	 * @param[in] patchSizeInCells
	 * @param[in] cellSizeInPixels
	 * @param[in] adjustMinScaleFactor Flag that indicates whether to adjust the pyramid's min scale factor to find the largest targets.
	 * @param[in] minPatchWidthInPixels Width of the smallest patches that should be extracted in pixels (cannot be smaller than patch width).
	 */
	AggregatedFeaturesExtractor(std::shared_ptr<ImagePyramid> featurePyramid,
			cv::Size patchSizeInCells, int cellSizeInPixels, bool adjustMinScaleFactor, int minPatchWidthInPixels = 0);

	/**
	 * Constructs a new aggregated features extractor that builds a feature pyramid where the given filter is
	 * applied to the downscaled images.
	 *
	 * @param[in] layerFilter Image filter that aggregates features over square cells.
	 * @param[in] patchSizeInCells
	 * @param[in] cellSizeInPixels
	 * @param[in] octaveLayerCount The number of image pyramid layers per octave.
	 * @param[in] minPatchWidthInPixels Width of the smallest patches that should be extracted in pixels (cannot be smaller than patch width).
	 */
	AggregatedFeaturesExtractor(std::shared_ptr<ImageFilter> layerFilter,
			cv::Size patchSizeInCells, int cellSizeInPixels, int octaveLayerCount, int minPatchWidthInPixels = 0);

	/**
	 * Constructs a new aggregated features extractor that builds a feature pyramid where the image filter is
	 * applied to the image before downscaling and the layer filter is applied to the downscaled images.
	 *
	 * @param[in] imageFilter Image filter that is used on the whole image before downscaling.
	 * @param[in] layerFilter Image filter that aggregates features over square cells and is applied after downscaling.
	 * @param[in] patchSizeInCells
	 * @param[in] cellSizeInPixels
	 * @param[in] octaveLayerCount The number of image pyramid layers per octave.
	 * @param[in] minPatchWidthInPixels Width of the smallest patches that should be extracted in pixels (cannot be smaller than patch width).
	 */
	AggregatedFeaturesExtractor(std::shared_ptr<ImageFilter> imageFilter, std::shared_ptr<ImageFilter> layerFilter,
			cv::Size patchSizeInCells, int cellSizeInPixels, int octaveLayerCount, int minPatchWidthInPixels = 0);

	using FeatureExtractor::update;

	void update(std::shared_ptr<VersionedImage> image) override;

	std::shared_ptr<Patch> extract(int centerX, int centerY, int width, int height) const override;

	std::shared_ptr<Patch> extract(cv::Rect bounds) const;

	std::shared_ptr<ImagePyramid> getFeaturePyramid();

	/**
	 * Converts bounds given as cell indices of an image pyramid layer to pixel indices of the original image.
	 *
	 * @param[in] boundsInLayerCells Bounds in cell indices of a pyramid layer.
	 * @param[in] layer Pyramid layer.
	 * @return Bounds in original image pixels.
	 */
	cv::Rect computeBoundsInImagePixels(cv::Rect boundsInLayerCells, const ImagePyramidLayer& layer) const;

private:

	/**
	 * Determines the maximum scale factor necessary to not detect targets smaller than the given width.
	 *
	 * @param[in] minPatchWidthInPixels Width of the smallest patches that should be extracted in pixels (cannot be smaller than patch width).
	 * @return Maximum necessary scale factor.
	 */
	double getMaxScaleFactor(int minPatchWidthInPixels) const;

	/**
	 * Determines the minimum scale factor necessary to detect targets at the largest scale.
	 *
	 * @param[in] image Current image.
	 * @return Minimum necessary scale factor.
	 */
	double getMinScaleFactor(const cv::Mat& image) const;

	/**
	 * Determines the maximum possible width of a target that still fits into an image.
	 *
	 * @param[in] image Current image.
	 * @return Maximum possible target width.
	 */
	int getMaxWidth(const cv::Mat& image) const;

	/**
	 * Retrieves the image pyramid layer whose patches rescaled to the original image are close to the given width.
	 *
	 * @param[in] width Requested width of a patch.
	 * @return Image pyramid layer whose patches come closest to the given width.
	 */
	const std::shared_ptr<ImagePyramidLayer> getLayer(int width) const;

	/**
	 * Computes the coordinates of the cell the given point falls into.
	 *
	 * @param[in] pointInImagePixels Image point in pixels to compute the layer cell of.
	 * @param[in] layer Image pyramid layer to compute the cell coordinates in.
	 * @return Cell coordinates in the pyramid layer.
	 */
	cv::Point computePointInLayerCells(cv::Point_<double> pointInImagePixels, const ImagePyramidLayer& layer) const;

	std::shared_ptr<Patch> extract(const ImagePyramidLayer& layer, cv::Rect boundsInLayerCells) const;

	bool isPatchWithinImage(cv::Rect bounds, const cv::Mat& image) const;

	std::shared_ptr<ImagePyramid> featurePyramid;
	cv::Size patchSizeInCells;
	cv::Size patchSizeInPixels;
	int cellSizeInPixels;
	bool adjustMinScaleFactor; ///< Flag that indicates whether to adjust the pyramid's min scale factor to find the largest targets.
};

} /* namespace extraction */
} /* namespace imageprocessing */

#endif /* IMAGEPROCESSING_EXTRACTION_AGGREGATEDFEATURESEXTRACTOR_HPP_ */

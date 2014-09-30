/*
 * PyramidFeatureExtractor.hpp
 *
 *  Created on: 22.03.2013
 *      Author: poschmann
 */

#ifndef PYRAMIDFEATUREEXTRACTOR_HPP_
#define PYRAMIDFEATUREEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"

namespace imageprocessing {

class ImagePyramid;
class ImagePyramidLayer;
class ImageFilter;
class ChainedFilter;

/**
 * Feature extractor whose features are patches of a constant size extracted from an image pyramid.
 *
 * Does only consider the given width when extracting single patches, as this extractor assumes the given aspect ratio
 * to be the same as the one given at construction, so the extracted patches will not be scaled to fit.
 */
class PyramidFeatureExtractor : public FeatureExtractor {
public:

	/**
	 * Constructs a new direct pyramid feature extractor that is based on an image pyramid.
	 *
	 * @param[in] pyramid The image pyramid.
	 * @param[in] width The width of the image data of the extracted patches.
	 * @param[in] height The height of the image data of the extracted patches.
	 */
	PyramidFeatureExtractor(std::shared_ptr<ImagePyramid> pyramid, int width, int height);

	/**
	 * Constructs a new direct pyramid feature extractor that internally builds its own image pyramid.
	 *
	 * @param[in] width The width of the image data of the extracted patches.
	 * @param[in] height The height of the image data of the extracted patches.
	 * @param[in] minWidth The width of the smallest patches that will be extracted.
	 * @param[in] maxWidth The width of the biggest patches that will be extracted.
	 * @param[in] octaveLayerCount The number of layers per octave.
	 */
	PyramidFeatureExtractor(int width, int height, int minWidth, int maxWidth, int octaveLayerCount = 5);

	/**
	 * Adds an image filter to the image pyramid that is applied to the original image.
	 *
	 * @param[in] filter The new image filter.
	 */
	void addImageFilter(std::shared_ptr<ImageFilter> filter);

	/**
	 * Adds an image filter to the image pyramid that is applied to the down-scaled images.
	 *
	 * @param[in] filter The new layer filter.
	 */
	void addLayerFilter(std::shared_ptr<ImageFilter> filter);

	/**
	 * Adds a new filter that is applied to the patches.
	 *
	 * @param[in] filter The new patch filter.
	 */
	void addPatchFilter(std::shared_ptr<ImageFilter> filter);

	void update(const cv::Mat& image);

	void update(std::shared_ptr<VersionedImage> image);

	/**
	 * Extracts a patch from the corresponding image pyramid. The given width will be used to determine the appropriate
	 * pyramid layer and the patch will be extracted according to the width and height given at construction time. The
	 * patch will not be extracted according to the given height and therefore no rescaling of the height is necessary.
	 * Therefore, the given height is just ignored.
	 *
	 * @param[in] x The x-coordinate of the patch center in the original image.
	 * @param[in] y The y-coordinate of the patch center in the original image.
	 * @param[in] width The width of the patch in the original image.
	 * @param[in] height The height of the patch in the original image.
	 * @return The extracted patch or an empty pointer in case the patch could not be extracted.
	 */
	virtual std::shared_ptr<Patch> extract(int x, int y, int width, int height) const;

	/**
	 * @return The image pyramid.
	 */
	const std::shared_ptr<ImagePyramid> getPyramid() const;

	/**
	 * @return The width of the image data of the extracted patches.
	 */
	int getPatchWidth() const;

	/**
	 * @return The height of the image data of the extracted patches.
	 */
	int getPatchHeight() const;

protected:

	/**
	 * Computes the scaled representation of an original value (coordinate, size, ...) and rounds accordingly.
	 *
	 * @param[in] layer The pyramid layer.
	 * @param[in] value The value in the original image.
	 * @return The corresponding value in the layer.
	 */
	virtual int getScaled(const ImagePyramidLayer& layer, int value) const;

	/**
	 * Computes the original representation of a scaled value (coordinate, size, ...) and rounds accordingly.
	 *
	 * @param[in] layer The pyramid layer.
	 * @param[in] value The value in this layer.
	 * @return corresponding The value in the original image.
	 */
	virtual int getOriginal(const ImagePyramidLayer& layer, int value) const;

	/**
	 * Determines the pyramid layer that approximately contains patches of the given width.
	 *
	 * @param[in] width The width of the patches in the original image.
	 * @return The pyramid layer or an empty pointer if there is no layer with an appropriate patch width.
	 */
	virtual const std::shared_ptr<ImagePyramidLayer> getLayer(int width) const;

	/**
	 * Creates a new image pyramid whose min and max scale factors are chosen to enable the extraction of patches
	 * of certain widths.
	 *
	 * @param[in] width The width of the extracted patch data.
	 * @param[in] minWidth The width of the smallest patches that will be extracted.
	 * @param[in] maxWidth The width of the biggest patches that will be extracted.
	 * @param[in] octaveLayerCount The number of layers per octave.
	 * @return A newly created image pyramid.
	 */
	static std::shared_ptr<ImagePyramid> createPyramid(int width, int minWidth, int maxWidth, int octaveLayerCount);

private:

	std::shared_ptr<ImagePyramid> pyramid; ///< The image pyramid.
	std::shared_ptr<ChainedFilter> patchFilter; ///< Filter that is applied to the patches.
	int patchWidth; ///< The width of the image data of the extracted patches.
	int patchHeight; ///< The height of the image data of the extracted patches.
};

} /* namespace imageprocessing */
#endif /* PYRAMIDFEATUREEXTRACTOR_HPP_ */

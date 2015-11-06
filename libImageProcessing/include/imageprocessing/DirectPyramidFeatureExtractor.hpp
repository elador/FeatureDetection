/*
 * DirectPyramidFeatureExtractor.hpp
 *
 *  Created on: 22.03.2013
 *      Author: poschmann
 */

#ifndef DIRECTPYRAMIDFEATUREEXTRACTOR_HPP_
#define DIRECTPYRAMIDFEATUREEXTRACTOR_HPP_

#include "imageprocessing/PyramidFeatureExtractor.hpp"

namespace imageprocessing {

class ImagePyramid;
class ImagePyramidLayer;
class ImageFilter;
class ChainedFilter;

/**
 * Pyramid based feature extractor that directly operates on an image pyramid to extract patches of a constant size.
 * Does only consider the given width when extracting single patches, as this extractor assumes the given aspect ratio
 * to be the same as the one given at construction, so the extracted patches will not be scaled to fit.
 */
class DirectPyramidFeatureExtractor : public PyramidFeatureExtractor {
public:

	/**
	 * Constructs a new direct pyramid feature extractor that is based on an image pyramid.
	 *
	 * @param[in] pyramid The image pyramid.
	 * @param[in] width The width of the image data of the extracted patches.
	 * @param[in] height The height of the image data of the extracted patches.
	 */
	DirectPyramidFeatureExtractor(std::shared_ptr<ImagePyramid> pyramid, int width, int height);

	/**
	 * Constructs a new direct pyramid feature extractor that internally builds its own image pyramid.
	 *
	 * @param[in] width The width of the image data of the extracted patches.
	 * @param[in] height The height of the image data of the extracted patches.
	 * @param[in] minWidth The width of the smallest patches that will be extracted.
	 * @param[in] maxWidth The width of the biggest patches that will be extracted.
	 * @param[in] octaveLayerCount The number of layers per octave.
	 */
	DirectPyramidFeatureExtractor(int width, int height, int minWidth, int maxWidth, int octaveLayerCount = 5);

	/**
	 * Constructs a new direct pyramid feature extractor that internally builds its own image pyramid.
	 *
	 * @param[in] width The width of the image data of the extracted patches.
	 * @param[in] height The height of the image data of the extracted patches.
	 * @param[in] minWidth The width of the smallest patches that will be extracted.
	 * @param[in] maxWidth The width of the biggest patches that will be extracted.
	 * @param[in] incrementalScaleFactor The incremental scale factor between two layers of the pyramid.
	 */
	DirectPyramidFeatureExtractor(int width, int height, int minWidth, int maxWidth, double incrementalScaleFactor = 0.85);

	virtual ~DirectPyramidFeatureExtractor();

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

	using FeatureExtractor::update;

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
	 * Extracts several patches from the layers of the corresponding image pyramid.
	 *
	 * @param[in] stepX The step size in x-direction in pixels (will be the same absolute value in all pyramid layers).
	 * @param[in] stepY The step size in y-direction in pixels (will be the same absolute value in all pyramid layers).
	 * @param[in] roi The region of interest inside the original image (region will be scaled accordingly to the layers).
	 * @param[in] firstLayer The index of the first layer to extract patches from.
	 * @param[in] lastLayer The index of the last layer to extract patches from.
	 * @param[in] stepLayer The step size for proceeding to the next layer (values greater than one will skip layers).
	 * @return The extracted patches.
	 */
	virtual std::vector<std::shared_ptr<Patch>> extract(int stepX, int stepY, cv::Rect roi = cv::Rect(),
			int firstLayer = -1, int lastLayer = -1, int stepLayer = 1) const;

	/**
	 * Extracts a single patch from a layer of the corresponding image pyramid.
	 *
	 * @param[in] layer The index of the layer.
	 * @param[in] x The x-coordinate of the patch center inside the layer.
	 * @param[in] y The y-coordinate of the patch center inside the layer.
	 * @return The extracted patch or an empty pointer in case the patch could not be extracted.
	 */
	virtual std::shared_ptr<Patch> extract(int layer, int x, int y) const;

	/**
	 * Determines the index of the pyramid layer that approximately contains patches of the given width. The height will
	 * be ignored.
	 *
	 * @param[in] width The width of the patches in the original image.
	 * @param[in] height The height of the patches in the original image.
	 * @return The index of the pyramid layer or -1 if there is no layer with an appropriate patch size.
	 */
	int getLayerIndex(int width, int height) const;

	double getMinScaleFactor() const;

	double getMaxScaleFactor() const;

	double getIncrementalScaleFactor() const;

	cv::Size getPatchSize() const;

	cv::Size getImageSize() const;

	virtual std::vector<std::pair<int, double>> getLayerScales() const;

	std::vector<cv::Size> getLayerSizes() const;

	std::vector<cv::Size> getPatchSizes() const;

	/**
	 * @return The image pyramid.
	 */
	std::shared_ptr<ImagePyramid> getPyramid();

	/**
	 * @return The image pyramid.
	 */
	const std::shared_ptr<ImagePyramid> getPyramid() const;

	/**
	 * @return The width of the image data of the extracted patches.
	 */
	int getPatchWidth() const;

	/**
	 * @param[in] width The new width of the image data of the extracted patches.
	 */
	void setPatchWidth(int width);

	/**
	 * @return The height of the image data of the extracted patches.
	 */
	int getPatchHeight() const;

	/**
	 * @param[in] height The new height of the image data of the extracted patches.
	 */
	void setPatchHeight(int height);

	/**
	 * Changes the size of the image data of the extracted patches.
	 *
	 * @param[in] width The new width of the image data of the extracted patches.
	 * @param[in] height The new height of the image data of the extracted patches.
	 */
	void setPatchSize(int width, int height);

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

	/**
	 * Extracts a feature vector from a certain location (patch) of a pyramid layer.
	 *
	 * @param[in] layer The pyramid layer.
	 * @param[in] bounds The patch bounds.
	 * @return A pointer to the patch (with its feature vector) that might be empty if the patch could not be created.
	 */
	std::shared_ptr<Patch> extract(const ImagePyramidLayer& layer, const cv::Rect bounds) const;

	std::shared_ptr<ImagePyramid> pyramid; ///< The image pyramid.
	int patchWidth;  ///< The width of the image data of the extracted patches.
	int patchHeight; ///< The height of the image data of the extracted patches.
	std::shared_ptr<ChainedFilter> patchFilter; ///< Filter that is applied to the patches.
};

} /* namespace imageprocessing */
#endif /* DIRECTPYRAMIDFEATUREEXTRACTOR_HPP_ */

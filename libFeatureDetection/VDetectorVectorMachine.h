#pragma once
#include "VDetector.h"

#include <vector>
#include <map>

class FdPatch;
class FdImage;
class Pyramid;

class VDetectorVectorMachine : public VDetector
{
public:
	VDetectorVectorMachine(void);
	virtual ~VDetectorVectorMachine(void);

	virtual int load(const std::string filename) = 0;
	virtual bool classify(FdPatch*) = 0;

	int initPyramids(FdImage*);	// img -> pyrs (save in img)
	std::vector<FdPatch*> detectOnImage(FdImage*);
	std::vector<FdPatch*> detectOnPatchvec(std::vector<FdPatch*>&);

	/**
	 * Classifies a single patch.
	 *
	 * @param[in] patch The patch that should be classified.
	 * @return True if the patch is considered to contain the object in question, false otherwise.
	 */
	bool detectOnPatch(FdPatch* patch);

	int extractToPyramids(FdImage*);	// all pyrs -> patches (save in img)
	std::vector<FdPatch*> getPatchesROI(FdImage*, int, int, int, int, int, int, std::string);

	/**
	 * Extracts a single patch and saves it inside the pyramid.
	 *
	 * @param[in] The original image containing the image pyramid.
	 * @param[in] x The x value of the patch's center point inside the original (unscaled) image.
	 * @param[in] y The y value of the patch's center point inside the original (unscaled) image.
	 * @param[in] width The width of the patch inside the original (unscaled) image.
	 * @return A pointer to the extracted patch (should not be deleted manually, the image pyramid will take care of it)
	 *         or NULL if the given coordinates are invalid (e.g. outside the image or pyramid).
	 */
	FdPatch* extractPatchToPyramid(FdImage *image, int x, int y, int width);

protected:
	int extractAndHistEq64(const Pyramid*, FdPatch*);	// private (one patch out of pyr)

	float nonlin_threshold;		// b parameter of the SVM
	int nonLinType;				// 2 = rbf (?)
	float basisParam;
	int polyPower;
	float divisor;

	int filter_size_x, filter_size_y;	// width and height of the detector patch

	int subsamplingMinHeight;
	int numSubsamplingLevels;
	float subsamplingFactor;

	int subsamplingLevelStart;
	int subsamplingLevelEnd;
	//float *subsampfac;
	std::map<int, float> subsampfac;

	int *pyramid_widths;

	unsigned char* LUT_bin; // lookup table for the histogram equalization
	float stretch_fac; // stretch factor for histogram equalizaiton

private:

	/**
	 * Determines the depth index of the pyramid whose patches have approximately the given width.
	 *
	 * @param[in] patchWidth The width of the patch in the original (unscaled) image.
	 * @return The index of the pyramid whose image patches have approximately the given width inside the original image.
	 *         Might be an invalid index (e.g. below 0 or greater than number of subsampling levels).
	 */
	int getDepthIndex(int patchWidth);

	/**
	 * Retrieves the pyramid of a given depth index from an image.
	 *
	 * @param[in] img The image.
	 * @param[in] depthIndex The depth index.
	 * @return A pointer to the pyramid or NULL if there is none at the given index.
	 */
	Pyramid* getPyramid(const FdImage *image, int depthIndex);

	/**
	 * Inserts a new patch into the given pyramid. If there already exists a patch at the given position,
	 * the existing patch will be returned without creating a new one.
	 *
	 * @param[in] pyramid The pyramid.
	 * @param[in] origX The x value of the patch's center point inside the original (unscaled) image.
	 * @param[in] origY The y value of the patch's center point inside the original (unscaled) image.
	 * @param[in] scaledX The x value of the patch's center point inside the given pyramid (scaled image).
	 * @param[in] scaledY The y value of the patch's center point inside the given pyramid (scaled image).
	 * @param[in] scale The scale factor of the pyramid (scaled image).
	 * @return A pointer to the patch (should not be deleted manually, the image pyramid will take care of it).
	 */
	FdPatch* insertPatchIntoPyramid(Pyramid* pyramid, int origX, int origY, int scaledX, int scaledY, float scale);

};

/*
 * ImagePyramidFeatureExtractor.h
 *
 *  Created on: 19.11.2012
 *      Author: poschmann
 */

#ifndef IMAGEPYRAMIDFEATUREEXTRACTOR_H_
#define IMAGEPYRAMIDFEATUREEXTRACTOR_H_

#include "classification/FeatureExtractor.h"
#include "boost/unordered_set.hpp"
#include <vector>

using cv::Size;
using cv::Mat;
using boost::shared_ptr;
using boost::unordered_set;
using std::vector;

namespace classification {

class FeatureVector;

/**
 * Image patch of a certain pyramid level.
 */
class Patch {
public:

	/**
	 * Constructs a new patch.
	 *
	 * @param[in] x The x-coordinate of the patch center.
	 * @param[in] y The y-coordinate of the patch center.
	 */
	explicit Patch(int x, int y);

	~Patch();

	/**
	 * @return The associated feature vector.
	 */
	inline shared_ptr<FeatureVector> getFeatureVector() const {
		return featureVector;
	}

	/**
	 * @param[in] featureVector The new feature vector.
	 */
	inline void setFeatureVector(shared_ptr<FeatureVector> featureVector) {
		this->featureVector = featureVector;
	}

	/**
	 * Determines whether another patch is equal to this one (has the same coordinates).
	 *
	 * @param[in] other The other patch.
	 * @return True if the patches have the same coordinates, false otherwise.
	 */
	bool operator==(const Patch& other) const;

	/**
	 * Hash function for patches.
	 */
	struct hash : std::unary_function<Patch, std::size_t> {
	    std::size_t operator()(const Patch& patch) const {
	        return patch.x + 31 * patch.y;
	    }
	};

private:

	int x; ///< The x-coordinate of the patch center.
	int y; ///< The y-coordinate of the patch center.
	shared_ptr<FeatureVector> featureVector; ///< The associated feature vector.
};

/**
 * Level of an image pyramid.
 */
class PyramidLevel {
public:

	/**
	 * Constructs a new pyramid level.
	 *
	 * @param[in] scaleFactor The scale factor of this level compared to the first (original) level.
	 * @param[in] scaledImage The scaled image.
	 */
	explicit PyramidLevel(double scaleFactor, const Mat& scaledImage);

	~PyramidLevel();

	/**
	 * Computes the scaled representation of an original value (coordinate, size, ...) and rounds accordingly.
	 *
	 * @param[in] value The value in the original (first level) space.
	 * @return The value in the scaled space of this level.
	 */
	int getScaled(int value) const;

	/**
	 * Computes the original representation of a scaled value (coordinate, size, ...) and rounds accordingly.
	 *
	 * @param[in] value The value in the scaled space of this level.
	 * @return The value in the original (first level) space.
	 */
	int getOriginal(int value) const;

	/**
	 * @return The scale factor of this level compared to the first (original) level.
	 */
	double getScaleFactor() const;

	/**
	 * @return The scaled image.
	 */
	const Mat& getScaledImage() const;

	/**
	 * @return A reference of the patches of the extracted feature vectors.
	 */
	inline unordered_set<Patch, Patch::hash>& getPatches() {
		return patches;
	}

private:

	double scaleFactor;    ///< The scale factor of this level compared to the first (original) level.
	const Mat scaledImage; ///< The scaled image.
	unordered_set<Patch, Patch::hash> patches; ///< The patches of the extracted feature vectors.
};

/**
 * Feature extractor that builds a grayscale image pyramid.
 */
class ImagePyramidFeatureExtractor : public FeatureExtractor {
public:

	/**
	 * Constructs a new image pyramid feature extractor.
	 *
	 * @param[in] featureSize The size of the image patch used for feature extraction.
	 * @param[in] scaleFactor The scale factor between two levels of the pyramid.
	 * @param[in] minHeight The minimum height of feature patches relative to the image height.
	 * @param[in] maxHeight The maximum height of feature patches relative to the image height.
	 */
	explicit ImagePyramidFeatureExtractor(Size featureSize, double scaleFactor,
			double minHeight = 0, double maxHeight = 1);

	virtual ~ImagePyramidFeatureExtractor();

	virtual void init(const Mat& image);

	virtual shared_ptr<FeatureVector> extract(int x, int y, int size);

protected:

	/**
	 * Initializes the scaled image of a pyramid layer.
	 *
	 * @param[in] image The scaled image.
	 */
	virtual void initScale(Mat image) = 0;

	/**
	 * Creates the feature vector of an image patch.
	 *
	 * @param[in] patch The image patch.
	 * @return The feature vector.
	 */
	virtual shared_ptr<FeatureVector> extract(const Mat& patch) = 0;

private:

	/**
	 * Clears the pyramid levels.
	 */
	void clearLevels();

	/**
	 * Determines the pyramid level that represents the given patch size.
	 *
	 * @param[in] size The patch size.
	 * @return The pyramid level that represents the patch size or null if there is none.
	 */
	PyramidLevel* getLevel(int size);

	Size featureSize;             ///< The size of the image patch used for feature extraction.
	double minHeight;             ///< The minimum height of feature patches relative to the image height.
	double maxHeight;             ///< The maximum height of feature patches relative to the image height.
	double scaleFactor;           ///< The scale factor between two levels of the pyramid.
	int firstLevel;               ///< The index of the first stored pyramid level.
	vector<PyramidLevel*> levels; ///< The levels of the image pyramid.
};

} /* namespace tracking */
#endif /* IMAGEPYRAMIDFEATUREEXTRACTOR_H_ */

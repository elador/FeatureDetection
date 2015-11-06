/*
 * CachingPyramidFeatureExtractor.hpp
 *
 *  Created on: 22.03.2013
 *      Author: poschmann
 */

#ifndef CACHINGPYRAMIDFEATUREEXTRACTOR_HPP_
#define CACHINGPYRAMIDFEATUREEXTRACTOR_HPP_

#include "imageprocessing/PyramidFeatureExtractor.hpp"
#include "imageprocessing/Version.hpp"
#include <unordered_map>

namespace imageprocessing {

/**
 * Pyramid feature extractor that builds upon another pyramid feature extractor and stores the extracted patches
 * for later extractions.
 */
class CachingPyramidFeatureExtractor : public PyramidFeatureExtractor {
private:

	/**
	 * Key of a patch inside a cache layer.
	 */
	class CacheKey {
	public:

		/**
		 * Constructs a new cache key.
		 *
		 * @param[in] x The x-coordinate of the patch inside the layer.
		 * @param[in] y The y-coordinate of the patch inside the layer.
		 */
		CacheKey(int x, int y) : x(x), y(y) {}

		bool operator==(const CacheKey& other) const {
			return x == other.x && y == other.y;
		}

		/**
		 * Hash function for keys.
		 */
		struct hash : std::unary_function<CacheKey, size_t> {
		    size_t operator()(const CacheKey& key) const {
		        return key.x + 31 * key.y;
		    }
		};

	private:

		int x; ///< The x-coordinate of the patch inside the layer.
		int y; ///< The y-coordinate of the patch inside the layer.
	};

	/**
	 * Layer of the cache representing a layer of the image pyramid.
	 */
	class CacheLayer {
	public:

		/**
		 * Default constructor.
		 */
		CacheLayer() : index(-1), scaleFactor(0), cache() {}

		/**
		 * Constructs a new cache layer.
		 *
		 * @param[in] index The index of this layer (0 is the original sized layer).
		 * @param[in] scaleFactor The scale factor of this layer compared to the original image.
		 */
		CacheLayer(int index, double scaleFactor) : index(index), scaleFactor(scaleFactor) {}

		/**
		 * @return The index of this layer (0 is the original sized layer).
		 */
		int getIndex() {
			return index;
		}

		/**
		 * Computes the scaled representation of an original value (coordinate, size, ...) and rounds accordingly.
		 *
		 * @param[in] value The value in the original image.
		 * @return The corresponding value in this layer.
		 */
		int getScaled(int value) {
			return cvRound(value * scaleFactor);
		}

		/**
		 * @return The cache.
		 */
		std::unordered_map<CacheKey, std::shared_ptr<Patch>, CacheKey::hash>& getCache() {
			return cache;
		}

	private:

		int index;          ///< The index of this layer (0 is the original sized layer).
		double scaleFactor; ///< The scale factor of this layer compared to the original image.
		std::unordered_map<CacheKey, std::shared_ptr<Patch>, CacheKey::hash> cache; ///< The cache.
	};

public:

	/**
	 * Caching strategy.
	 * SHARING - all callers share the patches with each other (see each others changes)
	 * COPYING - each caller gets its own copy of the patches
	 * INPUT_COPYING - first caller gets its own patches, subsequent callers share patches
	 * OUTPUT_COPYING - cache shares patches of first caller, subsequent callers get copies of that patch
	 */
	enum class Strategy { SHARING, COPYING, INPUT_COPYING, OUTPUT_COPYING };

	explicit CachingPyramidFeatureExtractor(std::shared_ptr<PyramidFeatureExtractor> extractor, Strategy strategy = Strategy::COPYING);

	using FeatureExtractor::update;

	void update(std::shared_ptr<VersionedImage> image);

	std::shared_ptr<Patch> extract(int x, int y, int width, int height) const;

	std::vector<std::shared_ptr<Patch>> extract(int stepX, int stepY, cv::Rect roi = cv::Rect(),
			int firstLayer = -1, int lastLayer = -1, int stepLayer = 1) const;

	std::shared_ptr<Patch> extract(int layer, int x, int y) const;

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

	cv::Size getPatchSize() const {
		return extractor->getPatchSize();
	}

	cv::Size getImageSize() const {
		return extractor->getImageSize();
	}

	std::vector<std::pair<int, double>> getLayerScales() const {
		return extractor->getLayerScales();
	}

	std::vector<cv::Size> getLayerSizes() const {
		return extractor->getLayerSizes();
	}

	std::vector<cv::Size> getPatchSizes() const {
		return extractor->getPatchSizes();
	}

private:

	/**
	 * Creates new empty cache layers based on the scale factors of the underlying feature extractor.
	 */
	void buildCache();

	std::shared_ptr<Patch> extractSharing(CacheLayer& layer, int x, int y) const;

	std::shared_ptr<Patch> extractCopying(CacheLayer& layer, int x, int y) const;

	std::shared_ptr<Patch> extractInputCopying(CacheLayer& layer, int x, int y) const;

	std::shared_ptr<Patch> extractOutputCopying(CacheLayer& layer, int x, int y) const;

	std::shared_ptr<PyramidFeatureExtractor> extractor; ///< The underlying feature extractor.
	mutable std::vector<CacheLayer> cache; ///< The current cache of stored patches.
	mutable int firstCacheIndex;           ///< The index of the first stored cache layer.
	Strategy strategy; ///< The caching strategy (copies of patches will be stored vs. patches will be shared).
	Version version; ///< The version.
};

} /* namespace imageprocessing */
#endif /* CACHINGPYRAMIDFEATUREEXTRACTOR_HPP_ */

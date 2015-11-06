/*
 * CachingFeatureExtractor.hpp
 *
 *  Created on: 21.03.2013
 *      Author: poschmann
 */

#ifndef CACHINGFEATUREEXTRACTOR_HPP_
#define CACHINGFEATUREEXTRACTOR_HPP_

#include "imageprocessing/FeatureExtractor.hpp"
#include "imageprocessing/Version.hpp"
#include <unordered_map>

namespace imageprocessing {

/**
 * Feature extractor that builds upon another feature extractor and stores the extracted patches for later extractions.
 */
class CachingFeatureExtractor : public FeatureExtractor {
private:

	/**
	 * Key for the cache consisting of the patch parameters (center position and size).
	 */
	class CacheKey {
	public:

		/**
		 * Constructs a new cache key.
		 *
		 * @param[in] x The x-coordinate of the center of the patch.
		 * @param[in] y The y-coordinate of the center of the patch.
		 * @param[in] width The width of the patch.
		 * @param[in] height The height of the patch.
		 */
		CacheKey(int x, int y, int width, int height) : x(x), y(y), width(width), height(height) {}

		bool operator==(const CacheKey& other) const {
			return x == other.x && y == other.y && width == other.width && height == other.height;
		}

		int x;      ///< The x-coordinate of the center of the patch.
		int y;      ///< The y-coordinate of the center of the patch.
		int width;  ///< The width of the patch.
		int height; ///< The height of the patch.
	};

	/**
	 * The hash operation for cache keys.
	 */
	struct KeyHash {
		size_t operator()(const CacheKey& key) const {
			size_t prime = 31;
			size_t hash = 1;
			hash = prime * hash + std::hash<int>()(key.x);
			hash = prime * hash + std::hash<int>()(key.y);
			hash = prime * hash + std::hash<int>()(key.width);
			hash = prime * hash + std::hash<int>()(key.height);
			return hash;
		}
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

	/**
	 * Constructs a new caching feature extractor.
	 *
	 * @param[in] extractor The underlying feature extractor.
	 * @param[in] strategy The caching strategy (copies of patches will be stored vs. patches will be shared).
	 */
	explicit CachingFeatureExtractor(std::shared_ptr<FeatureExtractor> extractor, Strategy strategy = Strategy::COPYING);

	using FeatureExtractor::update;

	void update(std::shared_ptr<VersionedImage> image);

	std::shared_ptr<Patch> extract(int x, int y, int width, int height) const;

private:

	std::shared_ptr<Patch> extractSharing(int x, int y, int width, int height) const;

	std::shared_ptr<Patch> extractCopying(int x, int y, int width, int height) const;

	std::shared_ptr<Patch> extractInputCopying(int x, int y, int width, int height) const;

	std::shared_ptr<Patch> extractOutputCopying(int x, int y, int width, int height) const;

	std::shared_ptr<FeatureExtractor> extractor; ///< The underlying feature extractor.
	mutable std::unordered_map<CacheKey, std::shared_ptr<Patch>, KeyHash> cache; ///< The current cache of stored patches.
	Strategy strategy; ///< The caching strategy (copies of patches will be stored vs. patches will be shared).
	Version version; ///< The version.
};

} /* namespace imageprocessing */
#endif /* CACHINGFEATUREEXTRACTOR_HPP_ */

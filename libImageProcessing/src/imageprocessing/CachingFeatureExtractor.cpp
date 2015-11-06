/*
 * CachingFeatureExtractor.cpp
 *
 *  Created on: 21.03.2013
 *      Author: poschmann
 */

#include "imageprocessing/CachingFeatureExtractor.hpp"
#include "imageprocessing/VersionedImage.hpp"
#include "imageprocessing/Patch.hpp"
#include <stdexcept>

using cv::Mat;
using std::shared_ptr;
using std::make_shared;
using std::invalid_argument;

namespace imageprocessing {

CachingFeatureExtractor::CachingFeatureExtractor(shared_ptr<FeatureExtractor> extractor, Strategy strategy) :
		extractor(extractor), cache(), strategy(strategy), version() {}

void CachingFeatureExtractor::update(shared_ptr<VersionedImage> image) {
	extractor->update(image);
	if (version != image->getVersion()) {
		cache.clear();
		version = image->getVersion();
	}
}

shared_ptr<Patch> CachingFeatureExtractor::extract(int x, int y, int width, int height) const {
	switch (strategy) {
	case Strategy::SHARING:
		return extractSharing(x, y, width, height);
	case Strategy::COPYING:
		return extractCopying(x, y, width, height);
	case Strategy::INPUT_COPYING:
		return extractInputCopying(x, y, width, height);
	case Strategy::OUTPUT_COPYING:
		return extractOutputCopying(x, y, width, height);
	default: // should never be reached
		return extractor->extract(x, y, width, height);
	}
}

shared_ptr<Patch> CachingFeatureExtractor::extractSharing(int x, int y, int width, int height) const {
	CacheKey key(x, y, width, height);
	auto iterator = cache.find(key);
	if (iterator == cache.end()) {
		shared_ptr<Patch> patch = extractor->extract(x, y, width, height);
		cache.emplace(key, patch);
		return patch;
	}
	return iterator->second;
}

shared_ptr<Patch> CachingFeatureExtractor::extractCopying(int x, int y, int width, int height) const {
	CacheKey key(x, y, width, height);
	auto iterator = cache.find(key);
	if (iterator == cache.end()) {
		shared_ptr<Patch> patch = extractor->extract(x, y, width, height);
		if (patch) // store a copy of the patch only if it exists
			cache.emplace(key, make_shared<Patch>(*patch));
		return patch;
	}
	return make_shared<Patch>(*(iterator->second));
}

shared_ptr<Patch> CachingFeatureExtractor::extractInputCopying(int x, int y, int width, int height) const {
	CacheKey key(x, y, width, height);
	auto iterator = cache.find(key);
	if (iterator == cache.end()) {
		shared_ptr<Patch> patch = extractor->extract(x, y, width, height);
		if (patch) // store a copy of the patch only if it exists
			cache.emplace(key, make_shared<Patch>(*patch));
		return patch;
	}
	return iterator->second;
}

shared_ptr<Patch> CachingFeatureExtractor::extractOutputCopying(int x, int y, int width, int height) const {
	CacheKey key(x, y, width, height);
	auto iterator = cache.find(key);
	if (iterator == cache.end()) {
		shared_ptr<Patch> patch = extractor->extract(x, y, width, height);
		cache.emplace(key, patch);
		return patch;
	}
	return make_shared<Patch>(*(iterator->second));
}

} /* namespace imageprocessing */

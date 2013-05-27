/*
 * EmptyLandmarkSource.hpp
 *
 *  Created on: 23.05.2013
 *      Author: poschmann
 */

#ifndef EMPTYLANDMARKSOURCE_HPP_
#define EMPTYLANDMARKSOURCE_HPP_

#include "imageio/NamedLandmarkSource.hpp"
#include "imageio/OrderedLandmarkSource.hpp"
#include "imageio/LandmarkCollection.hpp"

namespace imageio {

/**
 * Landmark source without any landmarks.
 */
class EmptyLandmarkSource : public NamedLandmarkSource, public OrderedLandmarkSource {
public:

	/**
	 * Constructs a new empty landmark source.
	 */
	EmptyLandmarkSource() : empty() {}

	~EmptyLandmarkSource() {}

	const bool next() {
		return true;
	}

	const LandmarkCollection& get() {
		return empty;
	}

	const LandmarkCollection& get(const path& imagePath) {
		return empty;
	}

	const LandmarkCollection& getLandmarks() const {
		return empty;
	}

private:

	const LandmarkCollection empty; ///< Empty landmark collection.
};

} /* namespace imageio */
#endif /* EMPTYLANDMARKSOURCE_HPP_ */

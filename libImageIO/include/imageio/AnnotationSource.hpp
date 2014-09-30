/*
 * AnnotationSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef ANNOTATIONSOURCE_HPP_
#define ANNOTATIONSOURCE_HPP_

#include "opencv2/core/core.hpp"

namespace imageio {

/**
 * Source of subsequent rectangular annotations.
 */
class AnnotationSource {
public:

	virtual ~AnnotationSource() {}

	/**
	 * Resets this annotation source to its initial state.
	 */
	virtual void reset() = 0;

	/**
	 * Moves this source forward to the next annotation.
	 *
	 * @return True if the source contains a next annotation, false otherwise.
	 */
	virtual bool next() = 0;

	/**
	 * Retrieves the current annotation.
	 *
	 * @return The annotation (has a size of zero if there is no annotation).
	 */
	virtual cv::Rect getAnnotation() const = 0;
};

} /* namespace imageio */
#endif /* ANNOTATIONSOURCE_HPP_ */

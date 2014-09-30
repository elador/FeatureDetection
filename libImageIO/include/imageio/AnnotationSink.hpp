/*
 * AnnotationSink.hpp
 *
 *  Created on: 27.05.2013
 *      Author: poschmann
 */

#ifndef ANNOTATIONSINK_HPP_
#define ANNOTATIONSINK_HPP_

#include "opencv2/core/core.hpp"
#include <string>

namespace imageio {

/**
 * Sink for subsequent rectangular annotations.
 */
class AnnotationSink {
public:

	virtual ~AnnotationSink() {}

	/**
	 * Determines whether this annotation sink is open.
	 *
	 * @return True if this annotation sink was opened (and not closed since), false otherwise.
	 */
	virtual bool isOpen() = 0;

	/**
	 * Opens the file writer. Needs to be called before adding the first annotation.
	 *
	 * @param[in] filename The name of the file to write the annotation data into.
	 */
	virtual void open(const std::string& filename) = 0;

	/**
	 * Closes the file writer.
	 */
	virtual void close() = 0;

	/**
	 * Adds an annotation.
	 *
	 * @param[in] annotation The annotation.
	 */
	virtual void add(const cv::Rect& annotation) = 0;
};

} /* namespace imageio */
#endif /* ANNOTATIONSINK_HPP_ */

/*
 * SimpleAnnotationSink.hpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#ifndef SIMPLEANNOTATIONSINK_HPP_
#define SIMPLEANNOTATIONSINK_HPP_

#include "imageio/AnnotationSink.hpp"
#include <fstream>

namespace imageio {

/**
 * Annotation sink that writes rectangular annotations to a file, one per line. There will be four values:
 * the x and y coordinate of the upper left corner, followed by the width and height. If the annotation is
 * invisible or invalid (zero size), all values will be zero. There will be whitespaces between the values.
 */
class SimpleAnnotationSink : public AnnotationSink {
public:

	/**
	 * Constructs a new simple annotation sink.
	 */
	SimpleAnnotationSink();

	bool isOpen();

	void open(const std::string& filename);

	void close();

	void add(const cv::Rect& annotation);

private:

	// Copy c'tor and assignment operator private since we own an ofstream object.
	SimpleAnnotationSink(SimpleAnnotationSink& other);
	SimpleAnnotationSink operator=(SimpleAnnotationSink rhs);

	std::ofstream output; ///< The file output stream.
};

} /* namespace imageio */
#endif /* SIMPLEANNOTATIONSINK_HPP_ */

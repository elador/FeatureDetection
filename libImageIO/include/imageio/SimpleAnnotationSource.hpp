/*
 * SimpleAnnotationSource.hpp
 *
 *  Created on: 21.01.2014
 *      Author: poschmann
 */

#ifndef SIMPLEANNOTATIONSOURCE_HPP_
#define SIMPLEANNOTATIONSOURCE_HPP_

#include "imageio/AnnotationSource.hpp"
#include <string>
#include <vector>

namespace imageio {

class ImageSource;

/**
 * Annotation source that reads rectangular annotations from a file with one line per annotation. Each line has
 * four values: the x and y coordinate of the upper left corner, followed by the width and height of the bounding
 * box. The values should be separated by white space or a delimiter character.
 */
class SimpleAnnotationSource : public AnnotationSource {
public:

	/**
	 * Constructs a new simple annotation source.
	 *
	 * @param[in] filename The name of the file containing the annotation data.
	 */
	SimpleAnnotationSource(const std::string& filename);

	void reset();

	bool next();

	cv::Rect getAnnotation() const;

private:

	std::vector<cv::Rect> annotations; ///< The target annotations.
	int index; ///< The index of the current target annotation.
};

} /* namespace imageio */
#endif /* SIMPLEANNOTATIONSOURCE_HPP_ */

/*
 * AnnotatedImageSource.hpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#ifndef ANNOTATEDIMAGESOURCE_HPP_
#define ANNOTATEDIMAGESOURCE_HPP_

#include "imageio/ImageSource.hpp"
#include <memory>

namespace imageio {

class AnnotationSource;

/**
 * Image source with associated annotations.
 */
class AnnotatedImageSource : public ImageSource {
public:

	/**
	 * Constructs a new annotated image source.
	 *
	 * @param[in] imageSource The source of the images.
	 * @param[in] annotationSource The source of the annotations.
	 */
	AnnotatedImageSource(std::shared_ptr<ImageSource> imageSource, std::shared_ptr<AnnotationSource> annotationSource);

	void reset();

	bool next();

	const cv::Mat getImage() const;

	/**
	 * Retrieves the current annotation.
	 *
	 * @return The annotation (has a size of zero if there is no annotation).
	 */
	const cv::Rect getAnnotation() const;

private:

	std::shared_ptr<ImageSource> imageSource; ///< The source of the images.
	std::shared_ptr<AnnotationSource> annotationSource; ///< The source of the annotations.
};

} /* namespace imageio */
#endif /* ANNOTATEDIMAGESOURCE_HPP_ */

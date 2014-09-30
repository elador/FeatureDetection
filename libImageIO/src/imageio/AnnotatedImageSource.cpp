/*
 * AnnotatedImageSource.cpp
 *
 *  Created on: 22.05.2013
 *      Author: poschmann
 */

#include "imageio/AnnotatedImageSource.hpp"
#include "imageio/AnnotationSource.hpp"

using cv::Mat;
using cv::Rect;
using std::shared_ptr;

namespace imageio {

AnnotatedImageSource::AnnotatedImageSource(shared_ptr<ImageSource> imageSource, shared_ptr<AnnotationSource> annotationSource) :
		imageSource(imageSource), annotationSource(annotationSource) {}

void AnnotatedImageSource::reset() {
	imageSource->reset();
	annotationSource->reset();
}

bool AnnotatedImageSource::next() {
	annotationSource->next();
	return imageSource->next();
}

const Mat AnnotatedImageSource::getImage() const {
	return imageSource->getImage();
}

const Rect AnnotatedImageSource::getAnnotation() const {
	return annotationSource->getAnnotation();
}

} /* namespace imageio */
